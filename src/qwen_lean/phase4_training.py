from __future__ import annotations

import gc
import json
import math
import platform
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from .phase3 import (
    IGNORE_INDEX,
    TokenizedSFTExample,
    load_pinned_tokenizer,
    trainer_dataset_rows,
)
from .phase3_training import (
    _RESUMABLE_CHECKPOINT_FILES,
    _build_trainer,
    _checkpoint_inventory,
    _cuda_metadata,
    _package_versions,
    _require_local_cuda,
    load_qlora_runtime,
    teacher_forced_metrics,
)
from .phase4 import (
    Phase4Config,
    load_phase4_workloads,
    load_selected_adapter_binding,
)


PHASE4_PREFLIGHT_SCHEMA_VERSION = "phase4-preflight-v1"
PHASE4_TRAINING_SCHEMA_VERSION = "phase4-training-run-v1"
PHASE4_ADAPTER_RELOAD_SCHEMA_VERSION = "phase4-adapter-reload-v1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _peak_within_ceiling(runtime: dict[str, Any], config: Phase4Config) -> bool:
    return int(runtime["peak_cuda_reserved_bytes"]) < int(
        config.training["memory_ceiling_bytes"]
    )


def _release_cuda(torch: Any, *values: Any) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def run_phase4_preflight(
    config: Phase4Config,
    workload_path: Path,
    output: Path,
    *,
    workload_loader: Any = load_phase4_workloads,
    schema_version: str = PHASE4_PREFLIGHT_SCHEMA_VERSION,
    phase_name: str = "Phase 4",
    workloads_override: Any | None = None,
) -> dict[str, Any]:
    torch, device_index, properties = _require_local_cuda()
    torch.manual_seed(int(config.training["seed"]))
    torch.cuda.manual_seed_all(int(config.training["seed"]))
    torch.cuda.reset_peak_memory_stats(device_index)
    workloads = (
        workload_loader(workload_path, config)
        if workloads_override is None
        else workloads_override
    )
    example = max(workloads.train, key=lambda item: len(item.input_ids))
    runtime = load_qlora_runtime(config)
    trainer = _build_trainer(
        runtime,
        [example],
        config,
        output.parent / "trainer-preflight",
        save_checkpoints=False,
    )
    trainer.create_optimizer()
    batch = trainer.data_collator(trainer_dataset_rows([example]))
    prompt_masked = not bool(
        batch["labels"][0, : example.prompt_tokens].ne(IGNORE_INDEX).any()
    )
    padding = batch["attention_mask"].eq(0)
    padding_masked = not bool(batch["labels"][padding].ne(IGNORE_INDEX).any())
    eos_supervised = int(batch["labels"][0, -1].item()) == workloads.eos_token_id
    if not prompt_masked or not padding_masked or not eos_supervised:
        raise RuntimeError(
            f"{phase_name} production preflight label masking is invalid"
        )

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
        raise RuntimeError(f"{phase_name} preflight loss is non-finite")
    trainer.accelerator.backward(loss)
    gradients = [
        parameter.grad
        for parameter in runtime.model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(
        not bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    ):
        raise RuntimeError(
            f"{phase_name} preflight gradients are missing or non-finite"
        )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [
            parameter
            for parameter in runtime.model.parameters()
            if parameter.requires_grad
        ],
        float(config.training["maximum_gradient_norm"]),
    )
    trainer.optimizer.step()
    torch.cuda.synchronize(device_index)
    adapter_changed = not torch.equal(adapter_before, adapter_parameter.detach())
    base_unchanged = torch.equal(base_before, base_parameter.detach())
    if not adapter_changed or not base_unchanged:
        raise RuntimeError(
            f"{phase_name} preflight parameter update mismatch: "
            f"adapter_changed={adapter_changed}, base_unchanged={base_unchanged}"
        )
    runtime_metadata = _cuda_metadata(torch, device_index, properties)
    memory_passed = _peak_within_ceiling(runtime_metadata, config)
    if not memory_passed:
        raise RuntimeError(f"{phase_name} preflight exceeds the 24 GiB memory ceiling")

    value = {
        "schema_version": schema_version,
        "passed": True,
        "fresh_model_state_discarded_after_process": True,
        "model": config.model,
        "workload_id": config.workloads["train"]["id"],
        "near_maximum_selection": {
            "record_id": example.record_id,
            "sequence_tokens": len(example.input_ids),
            "selected_workload_maximum_sequence_tokens": max(
                len(item.input_ids) for item in workloads.train
            ),
            "configured_maximum_sequence_tokens": config.training[
                "maximum_sequence_tokens"
            ],
        },
        "batch": {
            "micro_batch_size": config.training["per_device_micro_batch_size"],
            "gradient_accumulation_steps": config.training[
                "gradient_accumulation_steps"
            ],
            "prompt_tokens": example.prompt_tokens,
            "completion_tokens_excluding_eos": example.completion_tokens,
            "supervised_tokens_including_eos": int(
                batch["labels"].ne(IGNORE_INDEX).sum().item()
            ),
            "padding_tokens": int(padding.sum().item()),
            "prompt_labels_masked": prompt_masked,
            "padding_labels_masked": padding_masked,
            "eos_supervised": eos_supervised,
        },
        "quantization": config.quantization,
        "lora": config.lora,
        "quantized_linear_modules": runtime.quantized_linear_modules,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "total_parameter_count": runtime.total_parameter_count,
        "trainable_parameter_fraction": (
            runtime.trainable_parameter_count / runtime.total_parameter_count
        ),
        "only_intended_lora_parameters_trainable": True,
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clipping": float(gradient_norm),
        "all_gradients_finite": True,
        "adapter_parameter_checked": adapter_name,
        "adapter_parameter_changed": adapter_changed,
        "frozen_parameter_checked": base_name,
        "frozen_parameter_unchanged": base_unchanged,
        "memory_ceiling_bytes": config.training["memory_ceiling_bytes"],
        "memory_ceiling_passed": memory_passed,
        "packages": _package_versions(),
        "runtime": {
            "python": platform.python_version(),
            **runtime_metadata,
        },
    }
    _write_json(output, value)
    _release_cuda(torch, trainer, runtime)
    return value


def validate_phase4_resume_checkpoint(
    config: Phase4Config, checkpoint: Path, *, phase_name: str = "Phase 4"
) -> dict[str, Any]:
    expected_step = int(config.training["mandatory_process_stop_step"])
    if checkpoint.name != f"checkpoint-{expected_step}":
        raise ValueError(
            f"{phase_name} must resume from checkpoint-{expected_step}, "
            f"got {checkpoint.name}"
        )
    missing = [
        name
        for name in _RESUMABLE_CHECKPOINT_FILES
        if not (checkpoint / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint} is not full-state resumable; missing: {missing}"
        )
    try:
        trainer_state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        observed_step = int(trainer_state["global_step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{phase_name} checkpoint has invalid trainer state"
        ) from error
    if observed_step != expected_step:
        raise ValueError(
            f"{phase_name} checkpoint global step is {observed_step}, "
            f"expected {expected_step}"
        )
    return {
        "optimizer_step": observed_step,
        "optimizer_state_preserved": True,
        "scheduler_state_preserved": True,
        "rng_state_preserved": True,
        "data_position_preserved": True,
        "data_position_mechanism": "trainer global_step with ignore_data_skip=false",
        "trainer_epoch": trainer_state.get("epoch"),
    }


def select_validation_checkpoint(
    validation_probes: Sequence[dict[str, Any]],
    candidate_steps: Sequence[int] = (128, 256, 384, 512),
    *,
    phase_name: str = "Phase 4",
) -> dict[str, Any]:
    expected = tuple(int(step) for step in candidate_steps)
    by_step: dict[int, dict[str, Any]] = {}
    for probe in validation_probes:
        step = int(probe["optimizer_step"])
        if step in by_step:
            raise ValueError(f"duplicate {phase_name} validation probe at step {step}")
        cross_entropy = float(probe["mean_target_token_cross_entropy"])
        if not math.isfinite(cross_entropy):
            raise ValueError(f"non-finite {phase_name} validation loss at step {step}")
        by_step[step] = probe
    if tuple(sorted(by_step)) != tuple(sorted(expected)):
        raise ValueError(
            f"{phase_name} checkpoint selection requires exactly the configured boundaries"
        )
    selected_step = min(
        expected,
        key=lambda step: (
            float(by_step[step]["mean_target_token_cross_entropy"]),
            step,
        ),
    )
    return {
        "rule": "minimum validation mean target-token cross-entropy; ties earlier",
        "metric": "validation_mean_target_token_cross_entropy",
        "candidate_steps": list(expected),
        "selected_optimizer_step": selected_step,
        "selected_mean_target_token_cross_entropy": float(
            by_step[selected_step]["mean_target_token_cross_entropy"]
        ),
        "heldout_or_minif2f_consulted": False,
    }


def summarize_finite_training_logs(
    log_history: Sequence[dict[str, Any]], expected_optimizer_steps: int
) -> dict[str, Any]:
    entries = [item for item in log_history if "loss" in item]
    observed_steps = [int(item.get("step", -1)) for item in entries]
    expected_steps = list(range(1, expected_optimizer_steps + 1))
    if observed_steps != expected_steps:
        raise RuntimeError(
            "training logs do not cover every optimizer step exactly once: "
            f"observed {len(observed_steps)}, expected {expected_optimizer_steps}"
        )
    losses = [float(item["loss"]) for item in entries]
    try:
        gradient_norms = [float(item["grad_norm"]) for item in entries]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("training logs are missing gradient norms") from error
    if not all(math.isfinite(value) for value in (*losses, *gradient_norms)):
        raise RuntimeError("training logs contain a non-finite loss or gradient norm")
    return {
        "logged_optimizer_steps": len(entries),
        "covers_every_optimizer_step_exactly_once": True,
        "all_losses_finite": True,
        "all_gradient_norms_finite": True,
        "loss": {
            "minimum": min(losses),
            "maximum": max(losses),
            "mean": fmean(losses),
        },
        "gradient_norm_before_clipping": {
            "minimum": min(gradient_norms),
            "maximum": max(gradient_norms),
            "mean": fmean(gradient_norms),
        },
    }


def _load_prior_run(
    output_dir: Path,
    config: Phase4Config,
    train_examples: Sequence[TokenizedSFTExample],
    validation_examples: Sequence[TokenizedSFTExample],
    *,
    phase_name: str = "Phase 4",
    scheduler_marker: str = "scheduler_configured_for_512_steps",
) -> dict[str, Any]:
    path = output_dir / "run.json"
    if not path.is_file():
        raise ValueError(
            f"{phase_name} resume requires prior trajectory metadata at {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_step = int(config.training["mandatory_process_stop_step"])
    if int(value.get("optimizer_steps_completed", -1)) != expected_step:
        raise ValueError(f"{phase_name} prior trajectory stopped at the wrong boundary")
    if value.get("model") != config.model or value.get("training") != config.training:
        raise ValueError(
            f"{phase_name} prior trajectory changed model or training config"
        )
    expected_train_ids = [example.record_id for example in train_examples]
    expected_validation_ids = [example.record_id for example in validation_examples]
    workload = value.get("workloads", {})
    if workload.get("train", {}).get("selected_record_ids") != expected_train_ids:
        raise ValueError(f"{phase_name} prior trajectory changed training data/order")
    if (
        workload.get("validation", {}).get("selected_record_ids")
        != expected_validation_ids
    ):
        raise ValueError(f"{phase_name} prior trajectory changed validation data/order")
    if not value.get("trajectory", {}).get(scheduler_marker):
        raise ValueError(f"{phase_name} prior trajectory used a shortened LR schedule")
    return value


def run_phase4_training(
    config: Phase4Config,
    workload_path: Path,
    output_dir: Path,
    *,
    resume_from_checkpoint: Path | None = None,
    workload_loader: Any = load_phase4_workloads,
    schema_version: str = PHASE4_TRAINING_SCHEMA_VERSION,
    phase_name: str = "Phase 4",
    scheduler_marker: str = "scheduler_configured_for_512_steps",
    workloads_override: Any | None = None,
) -> dict[str, Any]:
    torch, device_index, properties = _require_local_cuda()
    torch.manual_seed(int(config.training["seed"]))
    torch.cuda.manual_seed_all(int(config.training["seed"]))
    torch.cuda.reset_peak_memory_stats(device_index)
    workloads = (
        workload_loader(workload_path, config)
        if workloads_override is None
        else workloads_override
    )
    train_examples = workloads.train
    validation_examples = workloads.validation
    stop_step = int(config.training["mandatory_process_stop_step"])
    maximum_steps = int(config.training["maximum_optimizer_steps"])

    if resume_from_checkpoint is None:
        if (output_dir / "run.json").exists() or any(
            (output_dir / "trainer-state").glob("checkpoint-*")
        ):
            raise ValueError(
                f"fresh {phase_name} training requires a new output trajectory directory"
            )
        process_leg = 1
        prior_run: dict[str, Any] | None = None
        resume_metadata: dict[str, Any] | None = None
        process_stop_after = stop_step
    else:
        resolved_checkpoint = resume_from_checkpoint.resolve()
        expected_parent = (output_dir / "trainer-state").resolve()
        if resolved_checkpoint.parent != expected_parent:
            raise ValueError(
                f"{phase_name} resume checkpoint must belong to this trajectory"
            )
        resume_metadata = validate_phase4_resume_checkpoint(
            config, resolved_checkpoint, phase_name=phase_name
        )
        prior_run = _load_prior_run(
            output_dir,
            config,
            train_examples,
            validation_examples,
            phase_name=phase_name,
            scheduler_marker=scheduler_marker,
        )
        resume_from_checkpoint = resolved_checkpoint
        process_leg = 2
        process_stop_after = maximum_steps

    runtime = load_qlora_runtime(config)
    try:
        from transformers import TrainerCallback
    except ImportError as error:
        raise RuntimeError(f"{phase_name} requires Transformers") from error

    validation_probes: list[dict[str, Any]] = (
        [] if prior_run is None else list(prior_run["validation_probes"])
    )
    pad_token_id = int(runtime.tokenizer.pad_token_id)

    boundary_steps = tuple(
        int(step) for step in config.training["checkpoint_candidates"]
    )

    class Phase4SafetyValidationAndStopCallback(TrainerCallback):
        def on_pre_optimizer_step(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            gradients = [
                parameter.grad
                for parameter in kwargs["model"].parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or any(
                not bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            ):
                raise RuntimeError(
                    f"missing or non-finite gradients before {phase_name} optimizer step "
                    f"{state.global_step + 1}"
                )
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Any = None,
            **kwargs: Any,
        ) -> Any:
            if logs is not None:
                for key in ("loss", "grad_norm"):
                    if key in logs and not math.isfinite(float(logs[key])):
                        raise RuntimeError(
                            f"non-finite {phase_name} {key} at step {state.global_step}"
                        )
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            step = int(state.global_step)
            if step in boundary_steps and not any(
                int(probe["optimizer_step"]) == step for probe in validation_probes
            ):
                metrics = teacher_forced_metrics(
                    kwargs["model"],
                    validation_examples,
                    pad_token_id=pad_token_id,
                )
                metrics["optimizer_step"] = step
                validation_probes.append(metrics)
                if bool(config.training.get("manual_checkpoint_boundaries", False)):
                    control.should_save = True
            if step >= process_stop_after:
                control.should_training_stop = True
            return control

    trainer = _build_trainer(
        runtime,
        train_examples,
        config,
        output_dir / "trainer-state",
        callbacks=[Phase4SafetyValidationAndStopCallback()],
        save_checkpoints=True,
    )
    pre_training = (
        teacher_forced_metrics(
            runtime.model, validation_examples, pad_token_id=pad_token_id
        )
        if prior_run is None
        else prior_run["pre_training_validation"]
    )
    started = time.perf_counter()
    train_result = trainer.train(
        resume_from_checkpoint=(
            None if resume_from_checkpoint is None else str(resume_from_checkpoint)
        )
    )
    stage_wall_time = time.perf_counter() - started
    completed_steps = int(trainer.state.global_step)
    if completed_steps != process_stop_after:
        raise RuntimeError(
            f"{phase_name} process stopped at {completed_steps}, "
            f"expected {process_stop_after}"
        )
    expected_probe_steps = [step for step in boundary_steps if step <= completed_steps]
    if [
        int(item["optimizer_step"]) for item in validation_probes
    ] != expected_probe_steps:
        raise RuntimeError(
            f"{phase_name} validation boundaries are incomplete or reordered"
        )
    training_log_summary = summarize_finite_training_logs(
        trainer.state.log_history, completed_steps
    )

    checkpoint_candidates: list[dict[str, Any]] = []
    probes_by_step = {
        int(probe["optimizer_step"]): probe for probe in validation_probes
    }
    for step in expected_probe_steps:
        checkpoint = output_dir / "trainer-state" / f"checkpoint-{step}"
        metadata = _checkpoint_inventory(checkpoint)
        if any(path.name.startswith("model-") for path in checkpoint.iterdir()):
            raise RuntimeError(
                f"{phase_name} unexpectedly saved merged base-model shards"
            )
        checkpoint_candidates.append(
            {
                "optimizer_step": step,
                "validation": probes_by_step[step],
                **metadata,
            }
        )

    torch.cuda.synchronize(device_index)
    runtime_metadata = _cuda_metadata(torch, device_index, properties)
    stage_memory_passed = _peak_within_ceiling(runtime_metadata, config)
    previous_peak_allocated = (
        0
        if prior_run is None
        else int(prior_run["runtime"]["peak_cuda_allocated_bytes"])
    )
    previous_peak_reserved = (
        0
        if prior_run is None
        else int(prior_run["runtime"]["peak_cuda_reserved_bytes"])
    )
    cumulative_wall_time = stage_wall_time + (
        0.0
        if prior_run is None
        else float(prior_run["runtime"]["cumulative_training_wall_time_seconds"])
    )
    process_legs = (
        [] if prior_run is None else list(prior_run["trajectory"]["process_legs"])
    )
    process_legs.append(
        {
            "process_leg": process_leg,
            "started_from_optimizer_step": 0 if process_leg == 1 else stop_step,
            "stopped_at_optimizer_step": completed_steps,
            "resume_checkpoint": (
                None
                if resume_from_checkpoint is None
                else f"trainer-state/{resume_from_checkpoint.name}"
            ),
            "training_wall_time_seconds": stage_wall_time,
        }
    )

    selection: dict[str, Any] | None = None
    validation_improved: bool | None = None
    if completed_steps == maximum_steps:
        selection = select_validation_checkpoint(
            validation_probes,
            candidate_steps=config.training["checkpoint_candidates"],
            phase_name=phase_name,
        )
        validation_improved = bool(
            selection["selected_mean_target_token_cross_entropy"]
            < float(pre_training["mean_target_token_cross_entropy"])
        )

    value = {
        "schema_version": schema_version,
        "status": (
            "stopped_at_mandatory_resume_boundary"
            if completed_steps == stop_step
            else "passed"
            if validation_improved and stage_memory_passed
            else "failed"
        ),
        "model": config.model,
        "dataset": config.value["dataset"],
        "serialization": config.value["serialization"],
        "workloads": {
            "train": {
                "id": config.workloads["train"]["id"],
                "examples": len(train_examples),
                "selected_record_ids": [item.record_id for item in train_examples],
            },
            "validation": {
                "id": config.workloads["validation"]["id"],
                "examples": len(validation_examples),
                "selected_record_ids": [item.record_id for item in validation_examples],
                "optimizer_batches": False,
            },
            "heldout": {
                "id": config.workloads["heldout"]["id"],
                "selected_record_ids": [item.record_id for item in workloads.heldout],
                "optimizer_batches": False,
            },
            "cross_split_record_ids_disjoint": True,
        },
        "quantization": config.quantization,
        "lora": config.lora,
        "training": config.training,
        "optimizer_steps_completed": completed_steps,
        "trajectory": {
            "process_legs": process_legs,
            "mandatory_process_stop_observed": completed_steps >= stop_step,
            "same_trajectory_resume": process_leg == 2,
            "resume_state": resume_metadata,
            scheduler_marker: True,
            "scheduler_configured_for_complete_trajectory": True,
            "ignore_data_skip": False,
            "data_order_seed": int(config.training["seed"]),
            "effective_batch_size": int(config.training["per_device_micro_batch_size"])
            * int(config.training["gradient_accumulation_steps"]),
            "examples_consumed_at_step": min(
                len(train_examples),
                completed_steps
                * int(config.training["per_device_micro_batch_size"])
                * int(config.training["gradient_accumulation_steps"]),
            ),
            "one_pass_data_accounting": {
                "eligible_training_examples": len(train_examples),
                "planned_optimizer_steps": maximum_steps,
                "full_effective_batches_before_final": max(maximum_steps - 1, 0),
                "final_optimizer_update_examples": (
                    len(train_examples)
                    - max(maximum_steps - 1, 0)
                    * int(config.training["per_device_micro_batch_size"])
                    * int(config.training["gradient_accumulation_steps"])
                ),
                "duplicate_final_batch_fill": bool(
                    config.training.get("duplicate_final_batch_fill", False)
                ),
                "all_eligible_examples_consumed_exactly_once": (
                    completed_steps == maximum_steps
                ),
            },
        },
        "pre_training_validation": pre_training,
        "validation_probes": validation_probes,
        "checkpoint_candidates": checkpoint_candidates,
        "checkpoint_selection": selection,
        "selected_beats_pre_training_validation": validation_improved,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "total_parameter_count": runtime.total_parameter_count,
        "trainable_parameter_fraction": (
            runtime.trainable_parameter_count / runtime.total_parameter_count
        ),
        "quantized_linear_modules": runtime.quantized_linear_modules,
        "adapter": (
            None
            if selection is None
            else {
                "artifact_id": config.lora["artifact_id"],
                "relative_path": (
                    f"trainer-state/checkpoint-{selection['selected_optimizer_step']}"
                ),
                "format": "peft-lora",
                "merged": False,
            }
        ),
        "memory_ceiling_bytes": config.training["memory_ceiling_bytes"],
        "memory_ceiling_passed": stage_memory_passed,
        "packages": _package_versions(),
        "runtime": {
            "python": platform.python_version(),
            "stage_training_wall_time_seconds": stage_wall_time,
            "cumulative_training_wall_time_seconds": cumulative_wall_time,
            "cumulative_examples_per_second": min(
                len(train_examples),
                completed_steps
                * int(config.training["per_device_micro_batch_size"])
                * int(config.training["gradient_accumulation_steps"]),
            )
            / cumulative_wall_time,
            "cumulative_optimizer_steps_per_second": (
                completed_steps / cumulative_wall_time
            ),
            "trainer_metrics": {
                key: item
                for key, item in train_result.metrics.items()
                if isinstance(item, (str, int, float, bool)) or item is None
            },
            "training_log_summary": training_log_summary,
            **runtime_metadata,
            "peak_cuda_allocated_bytes": max(
                previous_peak_allocated,
                int(runtime_metadata["peak_cuda_allocated_bytes"]),
            ),
            "peak_cuda_reserved_bytes": max(
                previous_peak_reserved,
                int(runtime_metadata["peak_cuda_reserved_bytes"]),
            ),
        },
    }
    _write_json(output_dir / "run.json", value)
    _release_cuda(torch, trainer, runtime)
    if not stage_memory_passed:
        raise RuntimeError(f"{phase_name} training exceeds the 24 GiB memory ceiling")
    if completed_steps == maximum_steps and not validation_improved:
        raise RuntimeError(f"{phase_name} validation loss did not improve over step 0")
    return value


def run_phase4_adapter_reload(
    config: Phase4Config,
    workload_path: Path,
    training_path: Path,
    adapter_dir: Path,
    output: Path,
    *,
    workload_loader: Any = load_phase4_workloads,
    binding_loader: Any = load_selected_adapter_binding,
    schema_version: str = PHASE4_ADAPTER_RELOAD_SCHEMA_VERSION,
    phase_name: str = "Phase 4",
    workloads_override: Any | None = None,
) -> dict[str, Any]:
    training, binding = binding_loader(
        training_path,
        expected_artifact_id=str(config.lora["artifact_id"]),
        adapter_dir=adapter_dir,
    )
    torch, device_index, properties = _require_local_cuda()
    torch.cuda.reset_peak_memory_stats(device_index)
    workloads = (
        workload_loader(workload_path, config)
        if workloads_override is None
        else workloads_override
    )
    selection = training.get("checkpoint_selection") or {}
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(f"{phase_name} requires PEFT and Transformers") from error
    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    if adapter_config.base_model_name_or_path != str(config.model["model_id"]):
        raise RuntimeError(f"{phase_name} adapter identifies a different base model")
    if adapter_config.revision != str(config.model["model_revision"]):
        raise RuntimeError(f"{phase_name} adapter identifies a different base revision")
    base = AutoModelForCausalLM.from_pretrained(
        str(config.model["model_id"]),
        revision=str(config.model["model_revision"]),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.quantization["quantization_type"]),
            bnb_4bit_use_double_quant=bool(config.quantization["double_quantization"]),
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
        workloads.train[0].prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    device = next(
        parameter for parameter in model.parameters() if parameter.device.type == "cuda"
    ).device
    inputs = {key: item.to(device) for key, item in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs, use_cache=False).logits
    finite_logits = bool(torch.isfinite(logits).all().item())
    if not finite_logits:
        raise RuntimeError(f"{phase_name} reloaded adapter produced non-finite logits")
    runtime_metadata = _cuda_metadata(torch, device_index, properties)
    value = {
        "schema_version": schema_version,
        "passed": True,
        "base_model_id": config.model["model_id"],
        "base_model_revision": config.model["model_revision"],
        "adapter_artifact_id": config.lora["artifact_id"],
        "selected_optimizer_step": binding.selected_optimizer_step,
        "selected_adapter_binding": binding.to_dict(),
        "adapter_training_relative_path": binding.training_relative_path,
        "training_artifact_sha256": binding.training_artifact_sha256,
        "selection_metric": selection["metric"],
        "adapter_base_model_name_or_path": adapter_config.base_model_name_or_path,
        "adapter_base_revision": adapter_config.revision,
        "adapter_rank": adapter_config.r,
        "adapter_merged": False,
        "adapter_format": "peft-lora",
        "forward_record_id": workloads.train[0].record_id,
        "finite_logits": finite_logits,
        "packages": _package_versions(),
        "runtime": {
            "python": platform.python_version(),
            **runtime_metadata,
        },
    }
    _write_json(output, value)
    _release_cuda(torch, model, base)
    return value
