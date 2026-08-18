from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase3 import load_pinned_tokenizer
from .phase3_training import QLoRARuntime, _require_local_cuda
from .phase4_training import (
    _write_json,
    run_phase4_adapter_reload,
    run_phase4_preflight,
    run_phase4_training,
)
from .sft2 import (
    SFT2_ARTIFACT_ID,
    SFT2_ENDPOINT_STEP,
    SFT2_TRAINING_SCHEMA_VERSION,
    SFT2Config,
    load_sft2_endpoint_binding,
    load_sft2_workloads,
    validate_sft2_parent,
    validate_step0_reference,
)

SFT2_PREFLIGHT_SCHEMA_VERSION = "sft2-preflight-v1"
SFT2_ADAPTER_RELOAD_SCHEMA_VERSION = "sft2-adapter-reload-v1"
SFT2_SCHEDULER_MARKER = "scheduler_restarted_for_complete_sft2_stage"


def load_sft2_qlora_runtime(
    config: SFT2Config, parent_adapter_dir: Path
) -> QLoRARuntime:
    torch, device_index, _ = _require_local_cuda()
    try:
        import bitsandbytes as bnb
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "SFT-2 requires the training optional dependencies"
        ) from error

    tokenizer = load_pinned_tokenizer(config)  # type: ignore[arg-type]
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = PeftConfig.from_pretrained(parent_adapter_dir)
    parent = config.parent_adapter
    peft_type = getattr(adapter_config.peft_type, "value", adapter_config.peft_type)
    task_type = getattr(adapter_config.task_type, "value", adapter_config.task_type)
    if (
        adapter_config.base_model_name_or_path != config.model["model_id"]
        or adapter_config.revision != config.model["model_revision"]
        or peft_type != "LORA"
        or task_type != "CAUSAL_LM"
        or int(adapter_config.r) != int(parent["rank"])
        or int(adapter_config.lora_alpha) != int(parent["alpha"])
        or float(adapter_config.lora_dropout) != float(parent["dropout"])
    ):
        raise RuntimeError("SFT-2 parent PEFT configuration differs from D015")

    quantization = config.quantization
    base = AutoModelForCausalLM.from_pretrained(
        str(config.model["model_id"]),
        revision=str(config.model["model_revision"]),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=bool(quantization["load_in_4bit"]),
            bnb_4bit_quant_type=str(quantization["quantization_type"]),
            bnb_4bit_use_double_quant=bool(quantization["double_quantization"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if not bool(getattr(base, "is_loaded_in_4bit", False)):
        raise RuntimeError("SFT-2 base model was not loaded in 4-bit")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, parent_adapter_dir, is_trainable=True, adapter_name="default"
    )
    active = getattr(model, "active_adapters", None)
    active_names = [active] if isinstance(active, str) else list(active or [])
    if active_names != ["default"] or set(model.peft_config) != {"default"}:
        raise RuntimeError("SFT-2 must continue exactly one existing PEFT adapter")
    quantized_modules = sum(
        1 for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)
    )
    if quantized_modules == 0:
        raise RuntimeError("SFT-2 found no bitsandbytes Linear4bit modules")
    return QLoRARuntime(
        model=model,
        tokenizer=tokenizer,
        # The adapter is already attached and trainable. Passing another config
        # to SFTTrainer would create a second, stacked LoRA.
        lora_config=None,
        quantized_linear_modules=quantized_modules,
    )


def _resolved_inputs(config: SFT2Config, workload_path: Path) -> tuple[SFT2Config, Any]:
    workloads = load_sft2_workloads(workload_path, config)
    resolved = config.resolve_for_training_examples(len(workloads.train))
    return resolved, workloads


def run_sft2_preflight(
    config: SFT2Config,
    workload_path: Path,
    parent_adapter_dir: Path,
    candidate_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    resolved, workloads = _resolved_inputs(config, workload_path)
    parent = validate_sft2_parent(resolved, parent_adapter_dir, candidate_manifest_path)
    value = run_phase4_preflight(
        resolved,  # type: ignore[arg-type]
        workload_path,
        output,
        workload_loader=load_sft2_workloads,
        schema_version=SFT2_PREFLIGHT_SCHEMA_VERSION,
        phase_name="SFT-2",
        workloads_override=workloads,
        runtime_loader=lambda item: load_sft2_qlora_runtime(item, parent_adapter_dir),
        runtime_evidence=parent,
    )
    after = validate_sft2_parent(resolved, parent_adapter_dir, candidate_manifest_path)
    unchanged = after == parent
    value["continuation_parent"]["unchanged_after_preflight"] = unchanged
    value["existing_adapter_continued_without_stacking"] = True
    if not unchanged:
        raise RuntimeError("SFT-2 preflight mutated the immutable reference parent")
    _write_json(output, value)
    return value


def run_sft2_training(
    config: SFT2Config,
    workload_path: Path,
    parent_adapter_dir: Path,
    candidate_manifest_path: Path,
    output_dir: Path,
    *,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    resolved, workloads = _resolved_inputs(config, workload_path)
    parent = validate_sft2_parent(resolved, parent_adapter_dir, candidate_manifest_path)
    if resume_from_checkpoint is not None:
        prior_path = output_dir / "run.json"
        if not prior_path.is_file():
            raise ValueError("SFT-2 resume requires the prior run evidence")
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_parent = dict(prior.get("continuation_parent") or {})
        prior_parent.pop("unchanged_after_process_leg", None)
        if prior_parent != parent:
            raise ValueError("SFT-2 resume parent differs from the first process leg")

    value = run_phase4_training(
        resolved,  # type: ignore[arg-type]
        workload_path,
        output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        workload_loader=load_sft2_workloads,
        schema_version=SFT2_TRAINING_SCHEMA_VERSION,
        phase_name="SFT-2",
        scheduler_marker=SFT2_SCHEDULER_MARKER,
        workloads_override=workloads,
        runtime_loader=lambda item: load_sft2_qlora_runtime(item, parent_adapter_dir),
        fixed_endpoint_step=SFT2_ENDPOINT_STEP,
        pre_training_validator=lambda metrics: validate_step0_reference(
            resolved, metrics
        ),
        runtime_evidence=parent,
    )
    after = validate_sft2_parent(resolved, parent_adapter_dir, candidate_manifest_path)
    if after != parent:
        raise RuntimeError("SFT-2 training mutated the immutable reference parent")
    value["continuation_parent"]["unchanged_after_process_leg"] = True
    value["trajectory"].update(
        {
            "stage_initialization": "immutable reference-sft-v1 adapter",
            "optimizer_restart_at_staged_step_0": True,
            "scheduler_restart_at_staged_step_0": True,
            "fresh_warmup_steps": 312,
            "uninterrupted_two_epoch_equivalence_claimed": False,
            "fixed_primary_endpoint_step": SFT2_ENDPOINT_STEP,
            "intermediate_checkpoints_diagnostic_only": [2491, 4981, 7472],
        }
    )
    accounting = value["trajectory"]["one_pass_data_accounting"]
    if (
        int(accounting["eligible_training_examples"]) != 79696
        or int(accounting["planned_optimizer_steps"]) != SFT2_ENDPOINT_STEP
        or int(accounting["final_optimizer_update_examples"]) != 8
        or bool(accounting["duplicate_final_batch_fill"])
    ):
        raise RuntimeError("SFT-2 one-pass accounting differs from the contract")
    _write_json(output_dir / "run.json", value)
    return value


def run_sft2_adapter_reload(
    config: SFT2Config,
    workload_path: Path,
    training_path: Path,
    adapter_dir: Path,
    output: Path,
) -> dict[str, Any]:
    resolved, workloads = _resolved_inputs(config, workload_path)
    value = run_phase4_adapter_reload(
        resolved,  # type: ignore[arg-type]
        workload_path,
        training_path,
        adapter_dir,
        output,
        workload_loader=load_sft2_workloads,
        binding_loader=load_sft2_endpoint_binding,
        schema_version=SFT2_ADAPTER_RELOAD_SCHEMA_VERSION,
        phase_name="SFT-2",
        workloads_override=workloads,
    )
    if (
        value["adapter_artifact_id"] != SFT2_ARTIFACT_ID
        or int(value["selected_optimizer_step"]) != SFT2_ENDPOINT_STEP
    ):
        raise RuntimeError("SFT-2 reload did not use the fixed Q4 endpoint")
    value["fixed_complete_q4_endpoint"] = True
    _write_json(output, value)
    return value
