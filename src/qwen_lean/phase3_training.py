from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .phase3 import (
    BASE_REVISION,
    IGNORE_INDEX,
    Phase3Config,
    TargetOnlyDataCollator,
    TokenizedSFTExample,
    load_phase3_workload,
    load_pinned_tokenizer,
    trainer_dataset_rows,
)


PREFLIGHT_SCHEMA_VERSION = "phase3-preflight-v1"
TRAINING_RUN_SCHEMA_VERSION = "phase3-training-run-v1"
ADAPTER_RELOAD_SCHEMA_VERSION = "phase3-adapter-reload-v1"


@dataclass
class QLoRARuntime:
    model: Any
    tokenizer: Any
    lora_config: Any
    quantized_linear_modules: int
    trainable_parameter_count: int = 0
    total_parameter_count: int = 0
    trainable_parameter_names: tuple[str, ...] = ()


def _require_local_cuda() -> tuple[Any, int, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Phase 3 requires the training optional dependencies"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 requires a local NVIDIA CUDA GPU")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return torch, device_index, properties


def _package_versions() -> dict[str, str]:
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "trl": "trl",
        "peft": "peft",
        "bitsandbytes": "bitsandbytes",
        "datasets": "datasets",
        "accelerate": "accelerate",
    }
    return {
        name: importlib.metadata.version(distribution)
        for name, distribution in packages.items()
    }


def _cuda_metadata(torch: Any, device_index: int, properties: Any) -> dict[str, Any]:
    return {
        "inference_execution": "local_cuda",
        "cuda_device_index": device_index,
        "cuda_device": properties.name,
        "cuda_device_capability": [properties.major, properties.minor],
        "cuda_device_total_memory_bytes": properties.total_memory,
        "torch_cuda_version": torch.version.cuda,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
    }


def load_qlora_runtime(config: Phase3Config) -> QLoRARuntime:
    torch, device_index, _ = _require_local_cuda()
    try:
        import bitsandbytes as bnb
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Phase 3 requires the training optional dependencies"
        ) from error

    tokenizer = load_pinned_tokenizer(config)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = config.quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["quantization_type"]),
        bnb_4bit_use_double_quant=bool(quantization["double_quantization"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(config.model["model_id"]),
        revision=str(config.model["model_revision"]),
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise RuntimeError("Phase 3 base model was not loaded in 4-bit")
    if getattr(model.config, "_name_or_path", None) != str(config.model["model_id"]):
        raise RuntimeError("loaded base model identity differs from Phase 3 config")
    model.config.use_cache = False
    lora = config.lora
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        bias=str(lora["bias"]),
        target_modules=[str(item) for item in lora["target_modules"]],
        modules_to_save=None,
        revision=BASE_REVISION,
    )
    quantized_modules = sum(
        1 for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)
    )
    if quantized_modules == 0:
        raise RuntimeError("Phase 3 found no bitsandbytes Linear4bit modules")
    return QLoRARuntime(
        model=model,
        tokenizer=tokenizer,
        lora_config=lora_config,
        quantized_linear_modules=quantized_modules,
    )


def _capture_and_validate_trainables(
    runtime: QLoRARuntime, config: Phase3Config
) -> None:
    trainable_names = tuple(
        name
        for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad
    )
    if not trainable_names or any(
        ".lora_A." not in name and ".lora_B." not in name for name in trainable_names
    ):
        raise RuntimeError("Phase 3 found a trainable parameter outside LoRA A/B")
    if any("lm_head" in name for name in trainable_names):
        raise RuntimeError("Phase 3 must not train or save the base lm_head")
    target_modules = tuple(str(item) for item in config.lora["target_modules"])
    if any(
        not any(f".{target}." in name for target in target_modules)
        for name in trainable_names
    ):
        raise RuntimeError("Phase 3 found a LoRA parameter outside target modules")
    runtime.trainable_parameter_names = trainable_names
    runtime.trainable_parameter_count = sum(
        parameter.numel()
        for parameter in runtime.model.parameters()
        if parameter.requires_grad
    )
    runtime.total_parameter_count = sum(
        parameter.numel() for parameter in runtime.model.parameters()
    )


def _build_trainer(
    runtime: QLoRARuntime,
    examples: Sequence[TokenizedSFTExample],
    config: Phase3Config,
    output_dir: Path,
    *,
    callbacks: list[Any] | None = None,
) -> Any:
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError("Phase 3 requires TRL and Datasets") from error
    training = config.training
    arguments = SFTConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        eval_strategy="no",
        per_device_train_batch_size=int(training["per_device_micro_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["maximum_gradient_norm"]),
        max_steps=int(training["maximum_optimizer_steps"]),
        lr_scheduler_type=str(training["lr_schedule"]),
        warmup_steps=int(training["warmup_steps"]),
        optim=str(training["optimizer"]),
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        bf16=True,
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(training["maximum_sequence_tokens"]),
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False,
    )
    dataset = Dataset.from_list(trainer_dataset_rows(examples))
    trainer = SFTTrainer(
        model=runtime.model,
        args=arguments,
        train_dataset=dataset,
        data_collator=TargetOnlyDataCollator(int(runtime.tokenizer.pad_token_id)),
        processing_class=runtime.tokenizer,
        callbacks=callbacks,
        peft_config=runtime.lora_config,
    )
    runtime.model = trainer.model
    _capture_and_validate_trainables(runtime, config)
    return trainer


def teacher_forced_metrics(
    model: Any,
    examples: Sequence[TokenizedSFTExample],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    collator = TargetOnlyDataCollator(pad_token_id)
    device = next(
        parameter for parameter in model.parameters() if parameter.device.type == "cuda"
    ).device
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    target_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for example in examples:
            batch = collator(trainer_dataset_rows([example]))
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, use_cache=False)
            shift_logits = outputs.logits[:, :-1, :].float()
            shift_labels = batch["labels"][:, 1:]
            active = shift_labels.ne(IGNORE_INDEX)
            count = int(active.sum().item())
            if count != example.completion_tokens + 1:
                raise RuntimeError(
                    f"teacher-forced token count differs for {example.record_id}"
                )
            loss = functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            loss_sum += float(loss.item())
            predictions = shift_logits.argmax(dim=-1)
            correct += int((predictions[active] == shift_labels[active]).sum().item())
            target_tokens += count
    if was_training:
        model.train()
    cross_entropy = loss_sum / target_tokens
    accuracy = correct / target_tokens
    if not math.isfinite(cross_entropy):
        raise RuntimeError("teacher-forced cross-entropy is non-finite")
    return {
        "examples": len(examples),
        "target_tokens": target_tokens,
        "mean_target_token_cross_entropy": cross_entropy,
        "target_token_next_token_accuracy": accuracy,
        "correct_target_tokens": correct,
        "wall_time_seconds": time.perf_counter() - started,
    }


def run_training_preflight(
    config: Phase3Config, workload_path: Path, output: Path
) -> dict[str, Any]:
    torch, device_index, properties = _require_local_cuda()
    torch.cuda.reset_peak_memory_stats(device_index)
    examples, eos_token_id = load_phase3_workload(workload_path, config)
    runtime = load_qlora_runtime(config)
    trainer = _build_trainer(
        runtime, examples, config, output.parent / "trainer-preflight"
    )
    trainer.create_optimizer()
    feature = trainer_dataset_rows([examples[0]])
    batch = trainer.data_collator(feature)
    if batch["labels"][0, : examples[0].prompt_tokens].ne(IGNORE_INDEX).any():
        raise RuntimeError("production GPU preflight batch supervises prompt tokens")
    if int(batch["labels"][0, -1].item()) != eos_token_id:
        raise RuntimeError("production GPU preflight batch does not supervise EOS")
    device = next(
        parameter
        for parameter in runtime.model.parameters()
        if parameter.device.type == "cuda"
    ).device
    batch = {key: value.to(device) for key, value in batch.items()}

    adapter_name, adapter_parameter = next(
        (name, parameter)
        for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad and ".lora_B." in name
    )
    base_name, base_parameter = next(
        (name, parameter)
        for name, parameter in runtime.model.named_parameters()
        if not parameter.requires_grad and "base_layer.weight" in name
    )
    adapter_before = adapter_parameter.detach().clone()
    base_before = base_parameter.detach().clone()
    runtime.model.train()
    trainer.optimizer.zero_grad()
    with (
        trainer.compute_loss_context_manager(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        loss = trainer.compute_loss(runtime.model, batch)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Phase 3 preflight loss is non-finite")
    trainer.accelerator.backward(loss)
    gradient_tensors = [
        parameter.grad
        for parameter in runtime.model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_tensors or any(
        not bool(torch.isfinite(gradient).all().item()) for gradient in gradient_tensors
    ):
        raise RuntimeError("Phase 3 preflight gradients are missing or non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [
            parameter
            for parameter in runtime.model.parameters()
            if parameter.requires_grad
        ],
        float(config.training["maximum_gradient_norm"]),
    )
    trainer.optimizer.step()
    adapter_changed = not torch.equal(adapter_before, adapter_parameter.detach())
    base_unchanged = torch.equal(base_before, base_parameter.detach())
    if not adapter_changed or not base_unchanged:
        raise RuntimeError(
            f"Phase 3 preflight update mismatch: adapter_changed={adapter_changed}, "
            f"base_unchanged={base_unchanged}"
        )

    value = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "passed": True,
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "workload_id": config.workload["id"],
        "record_id": examples[0].record_id,
        "batch": {
            "sequence_tokens": len(examples[0].input_ids),
            "prompt_tokens": examples[0].prompt_tokens,
            "completion_tokens_excluding_eos": examples[0].completion_tokens,
            "supervised_tokens_including_eos": int(
                batch["labels"].ne(IGNORE_INDEX).sum()
            ),
            "micro_batch_size": config.training["per_device_micro_batch_size"],
            "gradient_accumulation_steps": config.training[
                "gradient_accumulation_steps"
            ],
        },
        "quantization": config.quantization,
        "lora": config.lora,
        "quantized_linear_modules": runtime.quantized_linear_modules,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "total_parameter_count": runtime.total_parameter_count,
        "trainable_parameter_fraction": (
            runtime.trainable_parameter_count / runtime.total_parameter_count
        ),
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clipping": float(gradient_norm),
        "all_gradients_finite": True,
        "adapter_parameter_checked": adapter_name,
        "adapter_parameter_changed": adapter_changed,
        "frozen_parameter_checked": base_name,
        "frozen_parameter_unchanged": base_unchanged,
        "packages": _package_versions(),
        "runtime": {
            "python": platform.python_version(),
            **_cuda_metadata(torch, device_index, properties),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def run_overfit_training(
    config: Phase3Config, workload_path: Path, output_dir: Path
) -> dict[str, Any]:
    torch, device_index, properties = _require_local_cuda()
    torch.manual_seed(int(config.training["seed"]))
    torch.cuda.manual_seed_all(int(config.training["seed"]))
    torch.cuda.reset_peak_memory_stats(device_index)
    examples, _ = load_phase3_workload(workload_path, config)
    runtime = load_qlora_runtime(config)

    try:
        from transformers import TrainerCallback
    except ImportError as error:
        raise RuntimeError("Phase 3 requires Transformers") from error

    probes: list[dict[str, Any]] = []
    interval = int(config.training["memorization_probe_interval_steps"])
    loss_threshold = float(config.training["target_cross_entropy_threshold"])
    accuracy_threshold = float(config.training["target_accuracy_threshold"])
    pad_token_id = int(runtime.tokenizer.pad_token_id)

    class Phase3SafetyAndProbeCallback(TrainerCallback):
        def on_pre_optimizer_step(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            model = kwargs["model"]
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or any(
                not bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            ):
                raise RuntimeError(
                    f"missing or non-finite gradients before optimizer step {state.global_step + 1}"
                )
            return control

        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> Any:
            if logs is not None:
                for key in ("loss", "grad_norm"):
                    if key in logs and not math.isfinite(float(logs[key])):
                        raise RuntimeError(
                            f"non-finite {key} logged at optimizer step {state.global_step}"
                        )
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            step = int(state.global_step)
            if (
                step
                and step % interval == 0
                and not any(probe["optimizer_step"] == step for probe in probes)
            ):
                metrics = teacher_forced_metrics(
                    kwargs["model"], examples, pad_token_id=pad_token_id
                )
                metrics["optimizer_step"] = step
                probes.append(metrics)
                if (
                    metrics["mean_target_token_cross_entropy"] <= loss_threshold
                    and metrics["target_token_next_token_accuracy"]
                    >= accuracy_threshold
                ):
                    control.should_training_stop = True
            return control

    trainer = _build_trainer(
        runtime,
        examples,
        config,
        output_dir / "trainer-state",
        callbacks=[Phase3SafetyAndProbeCallback()],
    )
    pre_training = teacher_forced_metrics(
        runtime.model, examples, pad_token_id=pad_token_id
    )
    started = time.perf_counter()
    train_result = trainer.train()
    training_wall_time = time.perf_counter() - started
    final_metrics = teacher_forced_metrics(
        runtime.model, examples, pad_token_id=pad_token_id
    )
    completed_steps = int(trainer.state.global_step)
    if completed_steps % interval != 0:
        raise RuntimeError("Phase 3 training stopped outside a 100-step probe boundary")
    if not (
        final_metrics["mean_target_token_cross_entropy"] <= loss_threshold
        and final_metrics["target_token_next_token_accuracy"] >= accuracy_threshold
    ):
        raise RuntimeError(
            "Phase 3 failed the teacher-forced memorization gate by "
            f"{completed_steps} optimizer steps"
        )
    if not (
        final_metrics["mean_target_token_cross_entropy"]
        < pre_training["mean_target_token_cross_entropy"]
    ):
        raise RuntimeError("Phase 3 target-token loss did not improve")

    adapter_dir = output_dir / "adapter"
    runtime.model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    expected_adapter_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    missing = sorted(
        name for name in expected_adapter_files if not (adapter_dir / name).is_file()
    )
    if missing:
        raise RuntimeError(f"saved PEFT adapter is incomplete: {missing}")
    if any(path.name.startswith("model-") for path in adapter_dir.iterdir()):
        raise RuntimeError("Phase 3 unexpectedly saved merged base-model shards")

    value = {
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "status": "teacher_forced_gate_passed",
        "model": config.model,
        "dataset": config.value["dataset"],
        "serialization": config.value["serialization"],
        "workload": {
            "id": config.workload["id"],
            "selected_record_ids": list(config.selected_record_ids),
            "examples": len(examples),
        },
        "quantization": config.quantization,
        "lora": config.lora,
        "training": config.training,
        "optimizer_steps_completed": completed_steps,
        "pre_training_teacher_forced": pre_training,
        "memorization_probes": probes,
        "final_teacher_forced": final_metrics,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "total_parameter_count": runtime.total_parameter_count,
        "trainable_parameter_fraction": (
            runtime.trainable_parameter_count / runtime.total_parameter_count
        ),
        "quantized_linear_modules": runtime.quantized_linear_modules,
        "adapter": {
            "artifact_id": config.lora["artifact_id"],
            "relative_path": "adapter",
            "format": "peft-lora",
            "merged": False,
            "files": sorted(
                path.name for path in adapter_dir.iterdir() if path.is_file()
            ),
        },
        "packages": _package_versions(),
        "runtime": {
            "python": platform.python_version(),
            "training_wall_time_seconds": training_wall_time,
            "trainer_metrics": {
                key: value
                for key, value in train_result.metrics.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            },
            **_cuda_metadata(torch, device_index, properties),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def run_adapter_reload_check(
    config: Phase3Config,
    workload_path: Path,
    adapter_dir: Path,
    output: Path,
) -> dict[str, Any]:
    torch, device_index, properties = _require_local_cuda()
    torch.cuda.reset_peak_memory_stats(device_index)
    examples, _ = load_phase3_workload(workload_path, config)
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("Phase 3 requires PEFT and Transformers") from error
    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    if adapter_config.base_model_name_or_path != str(config.model["model_id"]):
        raise RuntimeError("saved adapter identifies a different base model")
    if adapter_config.revision != str(config.model["model_revision"]):
        raise RuntimeError("saved adapter identifies a different base revision")
    quantization = config.quantization
    base = AutoModelForCausalLM.from_pretrained(
        str(config.model["model_id"]),
        revision=str(config.model["model_revision"]),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization["quantization_type"]),
            bnb_4bit_use_double_quant=bool(quantization["double_quantization"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    tokenizer = load_pinned_tokenizer(config)
    inputs = tokenizer(
        examples[0].prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    device = next(
        parameter for parameter in model.parameters() if parameter.device.type == "cuda"
    ).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs, use_cache=False).logits
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("reloaded adapter produced non-finite logits")
    value = {
        "schema_version": ADAPTER_RELOAD_SCHEMA_VERSION,
        "passed": True,
        "base_model_id": config.model["model_id"],
        "base_model_revision": config.model["model_revision"],
        "adapter_artifact_id": config.lora["artifact_id"],
        "adapter_base_model_name_or_path": adapter_config.base_model_name_or_path,
        "adapter_base_revision": adapter_config.revision,
        "adapter_rank": adapter_config.r,
        "adapter_merged": False,
        "forward_record_id": examples[0].record_id,
        "finite_logits": True,
        "packages": _package_versions(),
        "runtime": {
            "python": platform.python_version(),
            **_cuda_metadata(torch, device_index, properties),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return value
