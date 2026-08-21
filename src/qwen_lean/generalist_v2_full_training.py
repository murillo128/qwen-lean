from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from .dataset_v2 import sha256_file
from .generalist_v2 import (
    LORA_TARGET_REGEX,
    MODEL_ID,
    MODEL_REVISION,
    GeneralistV2Config,
    deterministic_training_order,
    one_pass_membership_trajectory,
)
from .generalist_v2_dataset import (
    DATASET_BINDING_SCHEMA_VERSION,
    load_bound_training_variants,
)
from .generalist_v2_training import (
    ACTIVATION_CPU_OFFLOAD,
    ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS,
    GRADIENT_CHECKPOINTING_USE_REENTRANT,
    LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
    LM_HEAD_LOSS_CHUNK_TOKENS,
    MLP_SEQUENCE_CHUNK_TOKENS,
    build_weighted_sft_trainer,
    load_training_runtime,
    should_offload_activations,
    summarize_finite_optimizer_logs,
    tokenize_weighted_training_selection,
    validate_bounded_training_evidence,
    validate_production_preflight_gate,
    validate_q0_training_gate,
)

FULL_TRAINING_SCHEMA_VERSION = "generalist-v2-full-training-v1"
EXPECTED_FULL_PROOF_VARIANTS = 182812
EXPECTED_FULL_OPTIMIZER_STEPS = 22852
EXPECTED_QUARTER_STEPS = {"Q1": 5713, "Q2": 11426, "Q3": 17139, "Q4": 22852}


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def validate_quarter_checkpoint_inventory(
    trainer_root: Path, trajectory: Mapping[str, Any]
) -> dict[str, Any]:
    boundaries = trajectory.get("checkpoint_optimizer_steps")
    if boundaries != EXPECTED_QUARTER_STEPS:
        raise ValueError("generalist-v2 quarter boundaries differ from Dataset v2")
    checkpoints: dict[str, Any] = {}
    for checkpoint_id, step in EXPECTED_QUARTER_STEPS.items():
        root = trainer_root / f"checkpoint-{step}"
        required = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "trainer_state.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "training_args.bin",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(
                f"generalist-v2 {checkpoint_id} is incomplete: missing={missing}"
            )
        state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
        if int(state.get("global_step", -1)) != step:
            raise ValueError(f"generalist-v2 {checkpoint_id} trainer step differs")
        adapter_config = json.loads(
            (root / "adapter_config.json").read_text(encoding="utf-8")
        )
        if (
            adapter_config.get("base_model_name_or_path") != MODEL_ID
            or adapter_config.get("revision") != MODEL_REVISION
            or int(adapter_config.get("r", -1)) != 16
            or int(adapter_config.get("lora_alpha", -1)) != 32
            or adapter_config.get("target_modules") != LORA_TARGET_REGEX
            or adapter_config.get("bias") != "none"
            or adapter_config.get("task_type") != "CAUSAL_LM"
            or adapter_config.get("inference_mode") is not True
            or any(root.glob("model-*.safetensors"))
        ):
            raise ValueError(f"generalist-v2 {checkpoint_id} adapter identity differs")
        checkpoints[checkpoint_id] = {
            "optimizer_step": step,
            "relative_path": f"trainer-state/checkpoint-{step}",
            "adapter_model_sha256": sha256_file(root / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256_file(root / "adapter_config.json"),
            "trainer_state_sha256": sha256_file(root / "trainer_state.json"),
            "adapter_only": True,
            "reloadable_files_complete": True,
            "optimizer_scheduler_rng_state_complete": True,
        }
    return checkpoints


def run_full_generalist_training(
    config: GeneralistV2Config,
    package_root: Path,
    binding_path: Path,
    q0_evidence_path: Path,
    production_preflight_path: Path,
    overfit_run_path: Path,
    smoke_run_path: Path,
    output_dir: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Train one complete deterministic Dataset-v2 pass and retain Q1-Q4."""
    config.validate()
    if output_dir.exists():
        raise ValueError("full generalist-v2 training requires a fresh output path")
    q0_gate = validate_q0_training_gate(q0_evidence_path)
    production_gate = validate_production_preflight_gate(
        production_preflight_path, q0_gate
    )
    overfit_gate = validate_bounded_training_evidence(
        overfit_run_path,
        "generalist-v2-overfit64-v1",
        q0_gate,
        production_gate,
    )
    smoke_gate = validate_bounded_training_evidence(
        smoke_run_path,
        "generalist-v2-smoke4096-v1",
        q0_gate,
        production_gate,
    )
    if {
        production_gate["selected_lane"],
        overfit_gate["selected_lane"],
        smoke_gate["selected_lane"],
    } != {production_gate["selected_lane"]}:
        raise ValueError("generalist-v2 preflight/smoke precision lanes differ")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("schema_version") != DATASET_BINDING_SCHEMA_VERSION:
        raise ValueError("full training requires the accepted Dataset-v2 binding")
    if (
        production_gate["binding_manifest_sha256"]
        != binding["dataset"]["manifest_sha256"]
    ):
        raise ValueError("full training binding differs from production preflight")
    context_tokens = int(binding["serialization"]["lengths"]["selected_context_tokens"])
    if context_tokens != int(config.training["resolved_context_tokens"]):
        raise ValueError("full training context differs from Dataset-v2 binding")

    records = load_bound_training_variants(package_root)
    ordered = deterministic_training_order(records)
    membership = [(item.statement_id, item.proof_variant_id) for item in ordered]
    trajectory = one_pass_membership_trajectory(
        membership,
        effective_batch_size=int(config.training["gradient_accumulation_steps"]),
        membership_is_ordered=True,
    )
    if (
        len(ordered) != EXPECTED_FULL_PROOF_VARIANTS
        or trajectory["optimizer_visible_variants"] != EXPECTED_FULL_PROOF_VARIANTS
        or trajectory["optimizer_steps"] != EXPECTED_FULL_OPTIMIZER_STEPS
        or trajectory["checkpoint_optimizer_steps"] != EXPECTED_QUARTER_STEPS
    ):
        raise RuntimeError("full generalist-v2 trajectory differs from the binding")

    runtime = load_training_runtime(config, model_snapshot=model_snapshot)
    if runtime.lane != production_gate["selected_lane"]:
        raise RuntimeError("full training precision lane differs from passed gates")
    examples, weight_normalizer, tokenization = tokenize_weighted_training_selection(
        records,
        ordered,
        runtime.tokenizer,
        maximum_sequence_tokens=context_tokens,
    )
    if (
        len(examples) != EXPECTED_FULL_PROOF_VARIANTS
        or tokenization["truncated_or_dropped_variants"] != 0
        or tokenization["serialization"]["selected_context_tokens"] != context_tokens
    ):
        raise RuntimeError("full generalist-v2 tokenization changed the trajectory")
    observed_weight_mean = fmean(item.example_weight for item in examples)
    activation_cpu_offload_example_count = sum(
        should_offload_activations(len(item.input_ids)) for item in examples
    )
    if not math.isclose(
        observed_weight_mean, weight_normalizer, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError("full generalist-v2 weight normalization changed")
    del records, ordered
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=False)
    trainer_root = output_dir / "trainer-state"
    trainer = build_weighted_sft_trainer(
        runtime,
        examples,
        config,
        trainer_root,
        maximum_sequence_tokens=context_tokens,
        save_quarter_checkpoints=True,
        weight_normalizer=weight_normalizer,
    )
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("full generalist-v2 training requires PyTorch") from error
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - started
    completed_steps = int(trainer.state.global_step)
    completed_epoch = float(trainer.state.epoch or 0.0)
    if completed_steps != EXPECTED_FULL_OPTIMIZER_STEPS or not math.isclose(
        completed_epoch, 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            "full generalist-v2 training did not complete exactly one pass: "
            f"steps={completed_steps}, epoch={completed_epoch}"
        )
    logs = summarize_finite_optimizer_logs(
        trainer.state.log_history, EXPECTED_FULL_OPTIMIZER_STEPS
    )
    checkpoints = validate_quarter_checkpoint_inventory(trainer_root, trajectory)
    torch.cuda.synchronize(device_index)
    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_index))
    selected_lane = runtime.lane
    trainable_parameter_count = runtime.trainable_parameter_count
    sequence_chunked_mlp_module_count = runtime.sequence_chunked_mlp_module_count
    gated_delta_rule_backend = runtime.gated_delta_rule_backend
    del trainer, runtime, examples
    gc.collect()
    torch.cuda.empty_cache()

    value = {
        "schema_version": FULL_TRAINING_SCHEMA_VERSION,
        "status": "passed",
        "model": config.model,
        "dataset": {
            "package_id": config.dataset["package_id"],
            "binding_manifest_sha256": binding["dataset"]["manifest_sha256"],
            "training_statements": binding["dataset"]["general_train"]["statements"],
            "training_proof_variants": binding["dataset"]["general_train"][
                "proof_variants"
            ],
        },
        "gates": {
            "q0": q0_gate,
            "production_preflight": production_gate,
            "overfit64": overfit_gate,
            "smoke4096": smoke_gate,
        },
        "selected_lane": selected_lane,
        "training": {
            **config.training,
            "train_sampling_strategy": "sequential",
            "full_membership_weight_normalizer": weight_normalizer,
            "full_membership_weight_mean": observed_weight_mean,
            "trajectory": trajectory,
            "completed_optimizer_steps": completed_steps,
            "completed_epochs": completed_epoch,
            "exactly_one_complete_pass": True,
            "every_optimizer_visible_variant_consumed_once": True,
            "logs": logs,
        },
        "tokenization": tokenization,
        "checkpoints": checkpoints,
        "adapter": {
            "format": "peft-lora",
            "merged": False,
            "trainable_parameter_count": trainable_parameter_count,
            "base_model_shards_saved": False,
        },
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "python": platform.python_version(),
            "cuda_device_index": device_index,
            "cuda_device": properties.name,
            "cuda_device_total_memory_bytes": int(properties.total_memory),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "reserved_memory_headroom_bytes": int(properties.total_memory)
            - peak_reserved,
            "training_wall_time_seconds": wall_time,
            "torch_cuda_version": torch.version.cuda,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "gradient_checkpointing_use_reentrant": (
                GRADIENT_CHECKPOINTING_USE_REENTRANT
            ),
            "linear_attention_chunk_size": LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
            "activation_cpu_offload": ACTIVATION_CPU_OFFLOAD,
            "activation_cpu_offload_min_sequence_tokens": (
                ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS
            ),
            "activation_cpu_offload_example_count": (
                activation_cpu_offload_example_count
            ),
            "lm_head_loss_chunk_tokens": LM_HEAD_LOSS_CHUNK_TOKENS,
            "target_only_checkpointed_causal_loss": True,
            "mlp_sequence_chunk_tokens": MLP_SEQUENCE_CHUNK_TOKENS,
            "sequence_chunked_mlp_module_count": (sequence_chunked_mlp_module_count),
            "gated_delta_rule_backend": gated_delta_rule_backend,
            "packages": _package_versions(
                (
                    "torch",
                    "transformers",
                    "peft",
                    "trl",
                    "bitsandbytes",
                    "accelerate",
                    "datasets",
                    "huggingface-hub",
                    "safetensors",
                    "flash-linear-attention",
                )
            ),
        },
    }
    (output_dir / "run.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value
