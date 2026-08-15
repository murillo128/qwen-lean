from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _compact_memorization(
    value: dict[str, Any], *, optimizer_step: int, ignored_local_path: str
) -> dict[str, Any]:
    return {
        key: item for key, item in value.items() if key not in {"adapter", "results"}
    } | {
        "optimizer_step": optimizer_step,
        "adapter": {
            "artifact_id": value["adapter"]["adapter_id"],
            "ignored_local_path": ignored_local_path,
            "rank": value["adapter"]["adapter_rank"],
            "base_model_id": value["adapter"]["base_model_id"],
            "base_model_revision": value["adapter"]["base_model_revision"],
            "merged": value["adapter"]["merged"],
        },
        "candidate_results_retained_outside_git": True,
    }


def _amended_memorization_results(artifact_dir: Path) -> list[dict[str, Any]]:
    directory = artifact_dir / "memorization-amended"
    values: list[dict[str, Any]] = []
    for path in directory.glob("step-*.json"):
        match = re.fullmatch(r"step-(\d+)\.json", path.name)
        if match is None:
            continue
        step = int(match.group(1))
        value = _read(path)
        recorded_step = value.get("optimizer_step")
        if recorded_step is not None and int(recorded_step) != step:
            raise ValueError(f"memorization step mismatch in {path}")
        value["optimizer_step"] = step
        values.append(value)
    values.sort(key=lambda item: int(item["optimizer_step"]))
    if not values:
        raise ValueError("no amended Phase 3 memorization results found")
    steps = [int(value["optimizer_step"]) for value in values]
    if steps != list(range(100, steps[-1] + 1, 100)):
        raise ValueError(f"amended memorization boundaries are incomplete: {steps}")
    return values


def _write_amended_evidence(artifact_dir: Path, evidence_dir: Path) -> None:
    preflight = _read(artifact_dir / "preflight.json")
    training = _read(artifact_dir / "training-amended" / "run.json")
    reload = _read(artifact_dir / "adapter-reload-amended.json")
    memorization = _amended_memorization_results(artifact_dir)
    probes = {
        int(probe["optimizer_step"]): probe for probe in training["memorization_probes"]
    }
    checkpoint_rows: list[dict[str, Any]] = []
    for result in memorization:
        step = int(result["optimizer_step"])
        if step not in probes:
            raise ValueError(f"missing teacher-forced probe for step {step}")
        checkpoint_rows.append(
            {
                "optimizer_step": step,
                "teacher_forced_eligible": (
                    probes[step]["mean_target_token_cross_entropy"]
                    <= training["training"]["target_cross_entropy_threshold"]
                    and probes[step]["target_token_next_token_accuracy"]
                    >= training["training"]["target_accuracy_threshold"]
                ),
                "teacher_forced": {
                    key: value
                    for key, value in probes[step].items()
                    if key != "optimizer_step"
                },
                "vllm_exact_matches": result["exact_matches"],
                "vllm_minimum_exact_matches": result["minimum_exact_matches"],
                "vllm_generation_infrastructure_errors": result[
                    "generation_infrastructure_errors"
                ],
                "vllm_status": result["status"],
                "adapter_ignored_local_path": (
                    f"artifacts/phase3/training-amended/trainer-state/checkpoint-{step}"
                ),
            }
        )
    accepted = next(
        (row for row in checkpoint_rows if row["vllm_status"] == "passed"), None
    )
    status = "passed" if accepted is not None else "design_required"
    final = memorization[-1]

    _write(evidence_dir / "preflight.json", preflight)
    _write(evidence_dir / "training.json", training)
    _write(evidence_dir / "adapter-reload.json", reload)
    _write(
        evidence_dir / "memorization.json",
        _compact_memorization(
            final,
            optimizer_step=int(final["optimizer_step"]),
            ignored_local_path=(
                "artifacts/phase3/training-amended/trainer-state/"
                f"checkpoint-{final['optimizer_step']}"
            ),
        )
        | {
            "checkpoint_series_file": "memorization-checkpoints.json",
            "accepted_checkpoint_step": (
                None if accepted is None else accepted["optimizer_step"]
            ),
        },
    )
    _write(
        evidence_dir / "memorization-checkpoints.json",
        {
            "schema_version": "phase3-checkpoint-memorization-series-v1",
            "status": status,
            "stopping_contract": (
                "first 100-step checkpoint passing teacher-forced eligibility and "
                "at least 56/64 exact matches in fresh BF16 vLLM inference"
            ),
            "maximum_optimizer_steps": training["training"]["maximum_optimizer_steps"],
            "accepted_checkpoint_step": (
                None if accepted is None else accepted["optimizer_step"]
            ),
            "checkpoints": checkpoint_rows,
            "detailed_candidate_outputs_retained_outside_git": True,
        },
    )
    _write(
        evidence_dir / "diagnosis.json",
        {
            "schema_version": "phase3-free-generation-diagnosis-v2",
            "status": status,
            "teacher_forced_thresholds_passed_at_every_boundary": all(
                row["teacher_forced_eligible"] for row in checkpoint_rows
            ),
            "vllm_exact_matches_by_optimizer_step": {
                str(row["optimizer_step"]): row["vllm_exact_matches"]
                for row in checkpoint_rows
            },
            "required_exact_matches": final["minimum_exact_matches"],
            "vllm_generation_infrastructure_errors_by_optimizer_step": {
                str(row["optimizer_step"]): row["vllm_generation_infrastructure_errors"]
                for row in checkpoint_rows
            },
            "maximum_optimizer_steps_exhausted": (
                int(training["optimizer_steps_completed"])
                == int(training["training"]["maximum_optimizer_steps"])
            ),
            "interpretation": (
                "The amended same-trajectory run preserved complete resumable "
                "checkpoints and satisfied teacher-forced eligibility at every "
                "boundary, but no permitted BF16 vLLM checkpoint achieved the "
                "required sequence-level exact-match gate."
            ),
            "minif2f_adapter_smoke_run": False,
            "minif2f_adapter_smoke_not_run_reason": (
                "No checkpoint passed the required 56/64 adapter memorization "
                "prerequisite."
            ),
        },
    )
    exact_series = ", ".join(
        f"{row['optimizer_step']}→{row['vllm_exact_matches']}/64"
        for row in checkpoint_rows
    )
    (evidence_dir / "README.md").write_text(
        "# Phase 3 evidence\n\n"
        "`preflight.json`, `training.json`, and `adapter-reload.json` record the "
        "successful real-GPU QLoRA plumbing and resumable-training checks. "
        "`memorization-checkpoints.json` records every amended stopping boundary; "
        "`memorization.json` retains the compact final-boundary result.\n\n"
        "**OBSERVED:** all six 100-step checkpoints passed teacher-forced "
        "eligibility. Fresh BF16 vLLM exact matches were "
        f"{exact_series}, with zero generation infrastructure errors.\n\n"
        "**BLOCKED:** no checkpoint through the fixed 600-step maximum reached the "
        "required 56/64 vLLM gate. The downstream miniF2F adapter smoke was not "
        "run because memorization is its prerequisite. Adapter weights, full "
        "trainer checkpoints, and detailed candidate outputs remain under ignored "
        "`artifacts/`.\n",
        encoding="utf-8",
    )


def write_phase3_evidence(artifact_dir: Path, evidence_dir: Path) -> None:
    """Copy compact Phase 3 technical evidence while excluding local adapters and outputs."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if (artifact_dir / "training-amended" / "run.json").is_file():
        _write_amended_evidence(artifact_dir, evidence_dir)
        return
    preflight = _read(artifact_dir / "preflight.json")
    training = _read(artifact_dir / "training/run.json")
    reload = _read(artifact_dir / "adapter-reload.json")
    memorization = _read(artifact_dir / "memorization.json")
    diagnostic_4bit = _read(artifact_dir / "transformers-4bit-diagnostic.json")
    diagnostic_bf16 = _read(artifact_dir / "transformers-bf16-diagnostic.json")

    _write(evidence_dir / "preflight.json", preflight)
    _write(evidence_dir / "training.json", training)
    _write(evidence_dir / "adapter-reload.json", reload)
    _write(
        evidence_dir / "memorization.json",
        {
            key: value
            for key, value in memorization.items()
            if key not in {"adapter", "results"}
        }
        | {
            "adapter": {
                "artifact_id": memorization["adapter"]["adapter_id"],
                "ignored_local_path": "artifacts/phase3/training/adapter",
                "rank": memorization["adapter"]["adapter_rank"],
                "base_model_id": memorization["adapter"]["base_model_id"],
                "base_model_revision": memorization["adapter"]["base_model_revision"],
                "merged": memorization["adapter"]["merged"],
            },
            "candidate_results_retained_outside_git": True,
        },
    )
    _write(
        evidence_dir / "diagnosis.json",
        {
            "schema_version": "phase3-free-generation-diagnosis-v1",
            "status": "design_required",
            "accepted_vllm_bf16_exact_matches": memorization["exact_matches"],
            "transformers_4bit_exact_matches": diagnostic_4bit["exact_matches"],
            "transformers_bf16_exact_matches": diagnostic_bf16["exact_matches"],
            "required_exact_matches": memorization["minimum_exact_matches"],
            "vllm_generation_infrastructure_errors": memorization[
                "generation_infrastructure_errors"
            ],
            "vllm_and_transformers_bf16_target_exact_counts_equal": (
                memorization["exact_matches"] == diagnostic_bf16["exact_matches"]
            ),
            "interpretation": (
                "The accepted step-100 teacher-forced stop threshold did not imply "
                "the required sequence-level autoregressive memorization. The adapter "
                "also lost exact matches when moved from the NF4 training base to the "
                "unchanged BF16 Phase 1 inference base; BF16 Transformers reproduced "
                "the vLLM target-exact count, so this is not a vLLM-only loading fault."
            ),
            "detailed_candidate_diagnostics_retained_outside_git": True,
            "minif2f_adapter_smoke_run": False,
            "minif2f_adapter_smoke_not_run_reason": (
                "The required 56/64 adapter memorization prerequisite failed."
            ),
        },
    )
    (evidence_dir / "README.md").write_text(
        "# Phase 3 evidence\n\n"
        "`preflight.json`, `training.json`, and `adapter-reload.json` record the "
        "successful real-GPU QLoRA plumbing checks. `memorization.json` records the "
        "failed accepted vLLM free-generation gate without copying bulky per-example "
        "outputs. `diagnosis.json` separates the 4-bit training-runtime result from "
        "the unchanged BF16 Phase 1 inference-base result.\n\n"
        "**OBSERVED:** the first qualifying teacher-forced checkpoint occurred at "
        "optimizer step 100, but exact free generation reached only 49/64 on the NF4 "
        "training runtime and 27/64 on both BF16 Transformers and vLLM.\n\n"
        "**BLOCKED:** the required 56/64 vLLM gate is unmet. The downstream miniF2F "
        "adapter smoke was not run because the controlling issue makes memorization "
        "its prerequisite. Adapter weights, the materialized workload, trainer state, "
        "and detailed candidate outputs remain under ignored `artifacts/`.\n",
        encoding="utf-8",
    )
