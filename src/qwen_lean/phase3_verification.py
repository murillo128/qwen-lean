from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .phase2_corpus import read_jsonl_records, substitute_span
from .phase2_extraction import Phase2Config, validate_checkout_revision
from .phase2_schema import MathlibProofRecord
from .phase2_verification import run_lean_source, validate_record_source_identity
from .phase3 import OVERFIT_WORKLOAD_ID, Phase3Config
from .phase3_inference import MEMORIZATION_SCHEMA_VERSION
from .prompt import normalize_transport


PHASE3_SEMANTIC_VERIFICATION_SCHEMA_VERSION = "phase3-training-set-lean-verification-v1"
SEMANTIC_PROOF_PREFIX = "by\n  "


def reconstruct_generated_proof(
    source: str, record: MathlibProofRecord, candidate: str
) -> str:
    """Replace the original proof with the raw transported Phase 3 continuation."""
    replacement = SEMANTIC_PROOF_PREFIX + normalize_transport(candidate)
    return substitute_span(source, record.proof_span, replacement)


def load_selected_train_records(
    dataset_dir: Path, selected_record_ids: tuple[str, ...]
) -> list[MathlibProofRecord]:
    records = read_jsonl_records(dataset_dir / "train.jsonl")
    selected_ids = set(selected_record_ids)
    selected: dict[str, MathlibProofRecord] = {}
    for record in records:
        if record.id not in selected_ids:
            continue
        if record.split != "train":
            raise ValueError(f"Phase 3 semantic record {record.id} is not train data")
        if record.id in selected:
            raise ValueError(f"duplicate Phase 3 semantic record ID: {record.id}")
        selected[record.id] = record
    missing = selected_ids - selected.keys()
    if missing:
        raise ValueError(
            "Phase 3 semantic records are missing from the Phase 2 train split: "
            + ", ".join(sorted(missing))
        )
    return [selected[record_id] for record_id in selected_record_ids]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_phase2_environment(
    config: Phase2Config, dataset_dir: Path, mathlib_root: Path
) -> dict[str, Any]:
    config.validate_project_pins()
    validate_checkout_revision(
        mathlib_root, str(config.source["revision"]), "mathlib verification checkout"
    )
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "Mathlib"],
        cwd=mathlib_root,
        check=False,
    )
    if source_diff.returncode != 0:
        raise ValueError("mathlib verification source differs from the pinned revision")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    expected = config.source
    for key in ("repository", "revision", "lean_toolchain"):
        if source.get(key) != expected[key]:
            raise ValueError(f"Phase 2 manifest source {key} differs from its config")
    if (
        manifest.get("dataset_schema_version")
        != config.value["dataset"]["schema_version"]
    ):
        raise ValueError("Phase 2 manifest dataset schema differs from its config")
    return manifest


def _validate_gate_inputs(
    config: Phase3Config,
    records: list[MathlibProofRecord],
    memorization: dict[str, Any],
    training: dict[str, Any],
    *,
    optimizer_step: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    selected_ids = list(config.selected_record_ids)
    if memorization.get("schema_version") != MEMORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 3 memorization schema")
    if memorization.get("workload_id") != OVERFIT_WORKLOAD_ID:
        raise ValueError("memorization workload differs from Phase 3")
    if memorization.get("selected_record_ids") != selected_ids:
        raise ValueError("memorization selected record IDs differ from Phase 3")
    results = list(memorization.get("results", []))
    if len(results) != len(records) or int(memorization.get("examples", -1)) != len(
        records
    ):
        raise ValueError("memorization result count differs from the selected workload")
    result_ids = [str(result.get("record_id")) for result in results]
    if result_ids != selected_ids:
        raise ValueError("memorization result order differs from the selected workload")
    if any(result.get("generation_error") is not None for result in results):
        raise ValueError(
            "memorization results contain generation infrastructure errors"
        )
    if int(memorization.get("generation_infrastructure_errors", -1)) != 0:
        raise ValueError(
            "memorization summary reports generation infrastructure errors"
        )
    adapter = memorization.get("adapter", {})
    expected_adapter = {
        "adapter_id": config.lora["artifact_id"],
        "adapter_rank": config.lora["r"],
        "base_model_id": config.model["model_id"],
        "base_model_revision": config.model["model_revision"],
        "enabled": True,
        "merged": False,
    }
    for key, expected in expected_adapter.items():
        if adapter.get(key) != expected:
            raise ValueError(f"memorization adapter {key} differs from Phase 3")

    record_by_id = {record.id: record for record in records}
    recomputed_exact = 0
    for result in results:
        record = record_by_id[str(result["record_id"])]
        if str(result.get("target_completion")) != record.completion:
            raise ValueError(f"memorization target differs for {record.id}")
        exact = normalize_transport(
            str(result["candidate_text"])
        ) == normalize_transport(record.completion)
        if bool(result.get("normalized_exact_match")) != exact:
            raise ValueError(f"memorization exact-match flag differs for {record.id}")
        recomputed_exact += int(exact)
    if int(memorization.get("exact_matches", -1)) != recomputed_exact:
        raise ValueError(
            "memorization exact-match summary differs from its raw results"
        )

    adapter_path = Path(str(adapter["adapter_path"]))
    recorded_step = memorization.get("optimizer_step")
    if recorded_step is not None and int(recorded_step) != optimizer_step:
        raise ValueError(
            "memorization optimizer step differs from the requested checkpoint"
        )
    if adapter_path.name != f"checkpoint-{optimizer_step}":
        raise ValueError(
            "memorization adapter does not identify the requested checkpoint"
        )
    if not (adapter_path / "adapter_config.json").is_file():
        raise ValueError("memorization adapter checkpoint is unavailable")
    if int(training.get("optimizer_steps_completed", -1)) != optimizer_step:
        raise ValueError("training run does not end at the requested checkpoint")
    for key in ("model", "quantization", "lora", "serialization", "dataset"):
        if training.get(key) != config.value[key]:
            raise ValueError(f"training {key} differs from Phase 3")
    expected_workload = {
        "id": config.workload["id"],
        "examples": config.workload["expected_examples"],
        "selected_record_ids": selected_ids,
    }
    if training.get("workload") != expected_workload:
        raise ValueError("training workload differs from Phase 3")
    probes = {
        int(probe["optimizer_step"]): probe
        for probe in training.get("memorization_probes", [])
    }
    if optimizer_step not in probes:
        raise ValueError("training run has no teacher-forced probe for the checkpoint")
    probe = probes[optimizer_step]
    semantic = config.value["semantic_verification"]
    maximum_ce = float(semantic["maximum_target_cross_entropy"])
    minimum_accuracy = float(semantic["minimum_target_accuracy"])
    observed_ce = float(probe["mean_target_token_cross_entropy"])
    observed_accuracy = float(probe["target_token_next_token_accuracy"])
    if observed_ce > maximum_ce or observed_accuracy < minimum_accuracy:
        raise ValueError("checkpoint fails the amended teacher-forced fit gate")
    minimum_exact = int(
        config.value["memorization_generation"]["minimum_exact_matches"]
    )
    if recomputed_exact < minimum_exact:
        raise ValueError("checkpoint fails the amended BF16 vLLM exact-match gate")
    return results, {
        "optimizer_step": optimizer_step,
        "mean_target_token_cross_entropy": observed_ce,
        "maximum_target_cross_entropy": maximum_ce,
        "target_token_next_token_accuracy": observed_accuracy,
        "minimum_target_accuracy": minimum_accuracy,
        "vllm_exact_matches": recomputed_exact,
        "minimum_vllm_exact_matches": minimum_exact,
    }


def _semantic_summary(
    results: list[dict[str, Any]], *, minimum_accepted: int
) -> dict[str, Any]:
    accepted = sum(result["lean_check"]["status"] == "accepted" for result in results)
    exact_and_accepted = sum(
        result["normalized_exact_match"]
        and result["lean_check"]["status"] == "accepted"
        for result in results
    )
    non_exact_and_accepted = accepted - exact_and_accepted
    rejected = sum(result["lean_check"]["status"] == "rejected" for result in results)
    timeouts = sum(result["lean_check"]["status"] == "timeout" for result in results)
    infrastructure_errors = sum(
        result["lean_check"]["status"] == "infrastructure_error"
        or result["source_identity"] != "matched"
        for result in results
    )
    passed = (
        len(results) == 64
        and timeouts == 0
        and infrastructure_errors == 0
        and accepted >= minimum_accepted
    )
    return {
        "attempted": len(results),
        "normalized_exact_matches": sum(
            bool(result["normalized_exact_match"]) for result in results
        ),
        "lean_accepted": accepted,
        "exact_and_lean_accepted": exact_and_accepted,
        "non_exact_and_lean_accepted": non_exact_and_accepted,
        "lean_rejected": rejected,
        "timeouts": timeouts,
        "infrastructure_errors": infrastructure_errors,
        "minimum_lean_accepted": minimum_accepted,
        "passed": passed,
    }


def run_phase3_semantic_verification(
    config: Phase3Config,
    phase2_config: Phase2Config,
    dataset_dir: Path,
    mathlib_root: Path,
    memorization_path: Path,
    training_path: Path,
    output_path: Path,
    *,
    optimizer_step: int = 600,
    workers: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    manifest = _validate_phase2_environment(phase2_config, dataset_dir, mathlib_root)
    records = load_selected_train_records(dataset_dir, config.selected_record_ids)
    memorization = json.loads(memorization_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    candidates, eligibility = _validate_gate_inputs(
        config, records, memorization, training, optimizer_step=optimizer_step
    )
    semantic = config.value["semantic_verification"]
    worker_count = int(semantic["workers"] if workers is None else workers)
    timeout = float(
        semantic["timeout_seconds"] if timeout_seconds is None else timeout_seconds
    )
    if worker_count < 1 or timeout <= 0:
        raise ValueError("semantic verification workers and timeout must be positive")

    def check(pair: tuple[MathlibProofRecord, dict[str, Any]]) -> dict[str, Any]:
        record, candidate = pair
        candidate_text = str(candidate["candidate_text"])
        exact = bool(candidate["normalized_exact_match"])
        try:
            source = validate_record_source_identity(record, mathlib_root)
            reconstructed = reconstruct_generated_proof(source, record, candidate_text)
        except (OSError, ValueError) as error:
            return {
                "record_id": record.id,
                "file_path": record.file_path,
                "declaration_name": record.declaration_name,
                "source_identity": "failed",
                "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
                "normalized_exact_match": exact,
                "lean_check": {
                    "status": "infrastructure_error",
                    "exit_code": None,
                    "latency_seconds": 0.0,
                    "diagnostic": str(error),
                },
            }
        lean_check = run_lean_source(
            reconstructed,
            mathlib_root,
            timeout_seconds=timeout,
            lean_environment_root=config.path.parents[1],
        )
        return {
            "record_id": record.id,
            "file_path": record.file_path,
            "declaration_name": record.declaration_name,
            "source_identity": "matched",
            "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
            "normalized_exact_match": exact,
            "lean_check": lean_check.__dict__,
        }

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(check, zip(records, candidates, strict=True)))
    minimum_accepted = int(semantic["minimum_lean_accepted"])
    summary = _semantic_summary(results, minimum_accepted=minimum_accepted)
    if summary["normalized_exact_matches"] != eligibility["vllm_exact_matches"]:
        raise ValueError("semantic result exact count differs from its validated input")
    evidence = {
        "schema_version": PHASE3_SEMANTIC_VERIFICATION_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed" if summary["passed"] else "failed",
        "pipeline_interpretation": (
            "training-workload semantic verification, not held-out generalization"
        ),
        "reconstruction_contract": {
            "source_context": "pinned original Phase 2 mathlib file",
            "replacement": "by\\n  + normalize_transport(raw_candidate)",
            "candidate_extraction_or_repair": False,
            "lean_command": "lake env lean -E hasSorry Reconstructed.lean",
        },
        "source": {
            "repository": phase2_config.source["repository"],
            "revision": phase2_config.source["revision"],
            "lean_toolchain": phase2_config.source["lean_toolchain"],
            "dataset_schema_version": manifest["dataset_schema_version"],
        },
        "inputs": {
            "memorization_sha256": _sha256(memorization_path),
            "training_sha256": _sha256(training_path),
            "optimizer_step": optimizer_step,
            "selected_record_ids": list(config.selected_record_ids),
        },
        "eligibility": eligibility,
        "summary": summary,
        "records": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not summary["passed"]:
        if (
            summary["infrastructure_errors"] == 0
            and summary["timeouts"] == 0
            and summary["lean_accepted"] < minimum_accepted
        ):
            raise RuntimeError(
                "Phase 3 semantic verification requires investigation: "
                f"accepted={summary['lean_accepted']}/64"
            )
        raise RuntimeError(
            "Phase 3 semantic verification failed: "
            f"accepted={summary['lean_accepted']}/64, "
            f"timeouts={summary['timeouts']}, "
            f"infrastructure_errors={summary['infrastructure_errors']}"
        )
    return evidence
