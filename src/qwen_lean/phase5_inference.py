from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .baseline import run_phase1_baseline
from .phase2_extraction import Phase2Config
from .phase4_inference import (
    _phase1_config,
    _write_json,
    compare_phase4_heldout_runs,
    heldout_generation_request,
    phase4_adapter_spec,
    run_phase4_heldout,
)
from .phase5 import (
    Phase5Config,
    load_phase5_selected_adapter_binding,
    load_phase5_workloads,
)


PHASE5_HELDOUT_RUN_SCHEMA_VERSION = "phase5-heldout-run-v1"
PHASE5_HELDOUT_COMPARISON_SCHEMA_VERSION = "phase5-heldout-comparison-v1"
PHASE5_HELDOUT_INTEGRITY_KEY = "phase5_heldout_integrity_passed"


def phase5_heldout_generation_request(
    config: Phase5Config, adapter_dir: Path | None
) -> dict[str, Any]:
    return heldout_generation_request(config, adapter_dir)


def run_phase5_heldout(
    config: Phase5Config,
    phase2_config: Phase2Config,
    dataset_dir: Path,
    mathlib_root: Path,
    workload_path: Path,
    training_path: Path,
    output_dir: Path,
    *,
    adapter_dir: Path | None,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    return run_phase4_heldout(
        config,
        phase2_config,
        dataset_dir,
        mathlib_root,
        workload_path,
        training_path,
        output_dir,
        adapter_dir=adapter_dir,
        verification_workers=verification_workers,
        timeout_seconds=timeout_seconds,
        workload_loader=load_phase5_workloads,
        binding_loader=load_phase5_selected_adapter_binding,
        schema_version=PHASE5_HELDOUT_RUN_SCHEMA_VERSION,
        phase_name="Phase 5",
        integrity_summary_key=PHASE5_HELDOUT_INTEGRITY_KEY,
    )


def compare_phase5_heldout_runs(
    training_path: Path, base_dir: Path, adapter_dir: Path, output: Path
) -> dict[str, Any]:
    return compare_phase4_heldout_runs(
        training_path,
        base_dir,
        adapter_dir,
        output,
        binding_loader=load_phase5_selected_adapter_binding,
        schema_version=PHASE5_HELDOUT_COMPARISON_SCHEMA_VERSION,
        phase_name="Phase 5",
        artifact_prefix="artifacts/phase5/training",
        integrity_summary_key=PHASE5_HELDOUT_INTEGRITY_KEY,
    )


def _validate_accepted_phase1_base(
    config: Phase5Config,
    phase1: Any,
    accepted_run: dict[str, Any],
    accepted_summary: dict[str, Any],
) -> None:
    expected_identity = {
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "workload_id": config.value["minif2f"]["workload_id"],
        "benchmark_revision": phase1.benchmark["revision"],
        "candidates_per_task": phase1.sampling["candidates_per_task"],
    }
    for key, expected in expected_identity.items():
        if accepted_run.get(key) != expected:
            raise ValueError(f"accepted Phase 1 base {key} differs from Phase 5")
    expected_candidates = int(
        phase1.value["workloads"][config.value["minif2f"]["workload_id"]][
            "expected_task_count"
        ]
    ) * int(phase1.sampling["candidates_per_task"])
    if not bool(
        accepted_summary.get("complete")
        and int(accepted_summary.get("candidate_count", -1)) == expected_candidates
        and int(accepted_summary.get("infrastructure_error_count", -1)) == 0
        and int(accepted_summary.get("verifier_timeout_count", -1)) == 0
    ):
        raise ValueError("accepted Phase 1 full-validation base evidence is incomplete")


def run_phase5_minif2f(
    config: Phase5Config,
    benchmark_root: Path,
    training_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[Any, Sequence[Any], dict[str, Any]]:
    _, binding = load_phase5_selected_adapter_binding(
        training_path,
        expected_artifact_id=str(config.lora["artifact_id"]),
        adapter_dir=adapter_dir,
    )
    phase1 = _phase1_config(config)
    adapter = phase4_adapter_spec(config, adapter_dir)
    adapter.validate(phase1)
    worker_count = int(
        config.value["verification"]["workers"]
        if verification_workers is None
        else verification_workers
    )
    timeout = float(
        config.value["verification"]["minif2f_timeout_seconds"]
        if timeout_seconds is None
        else timeout_seconds
    )
    if worker_count < 1 or timeout <= 0:
        raise ValueError("Phase 5 miniF2F verification settings must be positive")
    metadata, results, summary = run_phase1_baseline(
        phase1,
        benchmark_root,
        str(config.value["minif2f"]["workload_id"]),
        output_dir,
        timeout_seconds=timeout,
        verification_workers=worker_count,
        adapter=adapter,
    )
    metadata = replace(metadata, selected_adapter_binding=binding.to_dict())
    _write_json(output_dir / "run.json", metadata.to_dict())

    expected_tasks = int(
        phase1.value["workloads"][config.value["minif2f"]["workload_id"]][
            "expected_task_count"
        ]
    )
    expected_candidates = expected_tasks * int(phase1.sampling["candidates_per_task"])
    passed = bool(
        summary["complete"]
        and len(results) == expected_candidates
        and int(summary["infrastructure_error_count"]) == 0
        and int(summary["verifier_timeout_count"]) == 0
    )
    project_root = config.path.parents[1]
    accepted_run_path = project_root / str(config.value["minif2f"]["accepted_base_run"])
    accepted_summary_path = project_root / str(
        config.value["minif2f"]["accepted_base_summary"]
    )
    accepted_run = json.loads(accepted_run_path.read_text(encoding="utf-8"))
    accepted_summary = json.loads(accepted_summary_path.read_text(encoding="utf-8"))
    _validate_accepted_phase1_base(config, phase1, accepted_run, accepted_summary)
    summary.update(
        {
            "phase5_minif2f_comparison": True,
            "phase1_quality_comparable": True,
            "selected_optimizer_step": binding.selected_optimizer_step,
            "selected_adapter_binding": binding.to_dict(),
            "adapter_training_relative_path": binding.training_relative_path,
            "training_artifact_sha256": binding.training_artifact_sha256,
            "checkpoint_selection_influenced_by_minif2f": False,
            "adapter_enabled": True,
            "adapter_id": config.lora["artifact_id"],
            "expected_tasks": expected_tasks,
            "observed_tasks": len(summary["per_task"]),
            "expected_candidates": expected_candidates,
            "observed_candidates": len(results),
            "miniF2F_test_evaluated": False,
            "accepted_phase1_base_reference": {
                "run_path": str(config.value["minif2f"]["accepted_base_run"]),
                "summary_path": str(config.value["minif2f"]["accepted_base_summary"]),
                "pass_at_k": accepted_summary["pass_at_k"],
                "regenerated": False,
            },
            "phase5_minif2f_passed": passed,
        }
    )
    _write_json(output_dir / "summary.json", summary)
    if not passed:
        raise RuntimeError(
            "Phase 5 miniF2F adapter evaluation failed integrity gates: "
            f"candidates={len(results)}, errors={summary['infrastructure_error_count']}, "
            f"timeouts={summary['verifier_timeout_count']}"
        )
    return metadata, results, summary
