from __future__ import annotations

from pathlib import Path
from typing import Any

from .phase4_training import (
    run_phase4_adapter_reload,
    run_phase4_preflight,
    run_phase4_training,
    select_validation_checkpoint,
    validate_phase4_resume_checkpoint,
)
from .phase5 import (
    Phase5Config,
    load_phase5_selected_adapter_binding,
    load_phase5_workloads,
)


PHASE5_PREFLIGHT_SCHEMA_VERSION = "phase5-preflight-v1"
PHASE5_TRAINING_SCHEMA_VERSION = "phase5-training-run-v1"
PHASE5_ADAPTER_RELOAD_SCHEMA_VERSION = "phase5-adapter-reload-v1"
PHASE5_SCHEDULER_MARKER = "scheduler_configured_for_complete_trajectory"


def _load_and_resolve(
    config: Phase5Config, workload_path: Path
) -> tuple[Phase5Config, Any]:
    workloads = load_phase5_workloads(workload_path, config)
    resolved = config.resolve_for_training_examples(len(workloads.train))
    return resolved, workloads


def run_phase5_preflight(
    config: Phase5Config, workload_path: Path, output: Path
) -> dict[str, Any]:
    resolved, workloads = _load_and_resolve(config, workload_path)
    return run_phase4_preflight(
        resolved,
        workload_path,
        output,
        workload_loader=load_phase5_workloads,
        schema_version=PHASE5_PREFLIGHT_SCHEMA_VERSION,
        phase_name="Phase 5",
        workloads_override=workloads,
    )


def validate_phase5_resume_checkpoint(
    config: Phase5Config, checkpoint: Path
) -> dict[str, Any]:
    return validate_phase4_resume_checkpoint(config, checkpoint, phase_name="Phase 5")


def select_phase5_validation_checkpoint(
    validation_probes: list[dict[str, Any]], candidate_steps: list[int]
) -> dict[str, Any]:
    return select_validation_checkpoint(
        validation_probes,
        candidate_steps=candidate_steps,
        phase_name="Phase 5",
    )


def run_phase5_training(
    config: Phase5Config,
    workload_path: Path,
    output_dir: Path,
    *,
    resume_from_checkpoint: Path | None = None,
) -> dict[str, Any]:
    resolved, workloads = _load_and_resolve(config, workload_path)
    value = run_phase4_training(
        resolved,
        workload_path,
        output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        workload_loader=load_phase5_workloads,
        schema_version=PHASE5_TRAINING_SCHEMA_VERSION,
        phase_name="Phase 5",
        scheduler_marker=PHASE5_SCHEDULER_MARKER,
        workloads_override=workloads,
    )
    accounting = value["trajectory"]["one_pass_data_accounting"]
    expected = workloads.trajectory
    if (
        int(accounting["planned_optimizer_steps"]) != expected.maximum_optimizer_steps
        or int(accounting["final_optimizer_update_examples"])
        != expected.final_optimizer_update_examples
        or bool(accounting["duplicate_final_batch_fill"])
    ):
        raise RuntimeError("Phase 5 one-pass data accounting differs from the workload")
    return value


def run_phase5_adapter_reload(
    config: Phase5Config,
    workload_path: Path,
    training_path: Path,
    adapter_dir: Path,
    output: Path,
) -> dict[str, Any]:
    resolved, workloads = _load_and_resolve(config, workload_path)
    return run_phase4_adapter_reload(
        resolved,
        workload_path,
        training_path,
        adapter_dir,
        output,
        workload_loader=load_phase5_workloads,
        binding_loader=load_phase5_selected_adapter_binding,
        schema_version=PHASE5_ADAPTER_RELOAD_SCHEMA_VERSION,
        phase_name="Phase 5",
        workloads_override=workloads,
    )
