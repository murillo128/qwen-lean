from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_phase3_evidence(artifact_dir: Path, evidence_dir: Path) -> None:
    """Copy compact Phase 3 technical evidence while excluding local adapters and outputs."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
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
