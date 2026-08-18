from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from .baseline import (
    GeneratedCandidate,
    _generate_candidates,
    _local_cuda_runtime,
    _verify_candidate,
    run_phase1_baseline,
    validate_minif2f_environment,
)
from .metrics import summarize_results
from .minif2f import materialize_benchmark_tasks
from .phase2_extraction import Phase2Config
from .phase2_verification import validate_record_source_identity
from .phase3_verification import _validate_phase2_environment
from .phase4_inference import (
    _heldout_candidate_result,
    _write_json,
    phase4_adapter_spec,
    run_phase4_heldout,
)
from .phase5 import ordered_record_ids_sha256
from .phase6 import (
    load_phase6_train_workload,
    summarize_phase6_train_results,
    target_exact,
)
from .phase6_inference import _load_selected_train_records
from .prompt import PROMPT_FORMAT_ID
from .schema import CandidateResult, TaskRecord
from .sft2 import SFT2Config, load_sft2_endpoint_binding, load_sft2_workloads
from .verifier import LeanVerifier

SFT2_TRAIN_RUN_SCHEMA_VERSION = "sft2-train512-run-v1"
SFT2_HELDOUT_RUN_SCHEMA_VERSION = "sft2-heldout512-run-v1"
SFT2_MINIF2F_RUN_SCHEMA_VERSION = "sft2-minif2f-validation-run-v1"
_LEGACY_TIMEOUT_RELABEL_MARKER = (
    "; repeated under the unchanged timeout contract and resolved as a bounded "
    "proof rejection"
)


def sft2_evaluation_integrity_passed(summary: dict[str, Any]) -> bool:
    """Treat verifier timeouts as complete unsuccessful SFT-2 candidates."""
    return bool(
        summary.get("complete")
        and int(summary.get("infrastructure_error_count", -1)) == 0
    )


def _sft2_train_integrity_passed(summary: dict[str, Any]) -> bool:
    return bool(
        sft2_evaluation_integrity_passed(summary)
        and int(summary.get("exact_target_but_not_verified_count", -1)) == 0
    )


def _restore_repeated_sft2_timeout_categories(
    results: Sequence[CandidateResult],
) -> tuple[list[CandidateResult], int]:
    """Undo the superseded SFT-2-only timeout-to-rejection relabeling."""
    restored: list[CandidateResult] = []
    restored_count = 0
    for item in results:
        stderr = item.diagnostics["stderr"]
        if _LEGACY_TIMEOUT_RELABEL_MARKER not in stderr:
            restored.append(item)
            continue
        if item.category != "lean_rejected" or item.lean_exit_code is not None:
            raise ValueError("legacy SFT-2 timeout relabel evidence is inconsistent")
        restored_count += 1
        timeout_stderr = stderr.replace(_LEGACY_TIMEOUT_RELABEL_MARKER, "")
        restored.append(
            replace(
                item,
                category="verifier_timeout",
                diagnostics={
                    "stdout": item.diagnostics["stdout"],
                    "stderr": timeout_stderr or "Lean verification timed out",
                },
            )
        )
    return restored, restored_count


def _retained_timeout_reverification(
    prior: dict[str, Any],
    *,
    restored_count: int,
) -> dict[str, Any]:
    if (
        restored_count < 1
        or int(prior.get("resolved_repeated_timeout_count", -1)) != restored_count
    ):
        raise ValueError("stored SFT-2 timeout retry provenance is inconsistent")
    return {
        "stored_candidates_regenerated": False,
        "transient_results_retried": int(prior["transient_results_retried"]),
        "categories": list(prior["categories"]),
        "timeout_seconds": float(prior["timeout_seconds"]),
        "retry_wall_time_seconds": float(prior["retry_wall_time_seconds"]),
        "retry_attempts": int(prior["retry_attempts"]),
        "total_verification_attempts": int(prior["total_verification_attempts"]),
        "repeated_same_contract_timeout_count": restored_count,
        "retained_verifier_timeout_count": restored_count,
        "artifact_category_restoration_only": True,
        "resolution_policy": (
            "a repeated same-contract timeout remains verifier_timeout and is an "
            "unsuccessful candidate, not an infrastructure error"
        ),
    }


def run_sft2_train512(
    config: SFT2Config,
    phase2_config: Phase2Config,
    dataset_dir: Path,
    mathlib_root: Path,
    workload_path: Path,
    training_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, Any], list[CandidateResult], dict[str, Any]]:
    _, binding = load_sft2_endpoint_binding(
        training_path,
        expected_artifact_id=str(config.lora["artifact_id"]),
        adapter_dir=adapter_dir,
    )
    phase6 = config.phase6_config()
    examples = load_phase6_train_workload(workload_path, phase6)
    selected_ids = [item.record_id for item in examples]
    train_contract = config.value["train_evaluation"]
    if (
        len(selected_ids) != int(train_contract["expected_examples"])
        or ordered_record_ids_sha256(selected_ids)
        != train_contract["workload_ordered_ids_sha256"]
    ):
        raise ValueError("SFT-2 train512 workload differs from Phase 6")
    _validate_phase2_environment(phase2_config, dataset_dir, mathlib_root)
    records = _load_selected_train_records(dataset_dir, selected_ids)
    for record, example in zip(records, examples, strict=True):
        if record.declaration_name != example.declaration_name:
            raise ValueError("SFT-2 train512 declaration identity differs")
    sources = {
        record.id: validate_record_source_identity(record, mathlib_root)
        for record in records
    }

    phase1 = config.phase1_validation_config()
    adapter = phase4_adapter_spec(config, adapter_dir)  # type: ignore[arg-type]
    adapter.validate(phase1)
    tasks = [
        TaskRecord(
            id=record.id,
            preamble="",
            declaration=record.declaration,
            declaration_name=record.declaration_name,
        )
        for record in records
    ]
    prompts = [item.prompt for item in examples]
    sampling = dict(config.value["heldout_generation"])
    runtime = _local_cuda_runtime(phase1)
    generation_started = time.perf_counter()
    generated, engine_version = _generate_candidates(
        phase1,
        tasks,
        prompts=prompts,
        sampling=sampling,
        adapter=adapter,
    )
    generation_wall_time = time.perf_counter() - generation_started
    expected_candidates = len(records) * int(sampling["candidates_per_task"])
    if len(generated) != expected_candidates:
        raise RuntimeError(
            f"SFT-2 vLLM returned {len(generated)} train512 candidates, "
            f"expected {expected_candidates}"
        )

    worker_count = int(
        config.value["verification"]["workers"]
        if verification_workers is None
        else verification_workers
    )
    timeout = float(
        config.value["verification"]["train_timeout_seconds"]
        if timeout_seconds is None
        else timeout_seconds
    )
    if worker_count < 1 or timeout <= 0:
        raise ValueError("SFT-2 train512 verification settings must be positive")
    records_by_id = {record.id: record for record in records}
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(
                lambda item: _heldout_candidate_result(
                    item,
                    records_by_id[item.task.id],
                    sources[item.task.id],
                    mathlib_root,
                    timeout_seconds=timeout,
                ),
                generated,
            )
        )
    verification_wall_time = time.perf_counter() - verification_started
    exact = {
        (item.task_id, item.candidate_index): target_exact(
            item.candidate_text, records_by_id[item.task_id].completion
        )
        for item in results
    }
    summary = summarize_phase6_train_results(
        results,
        expected_task_ids=selected_ids,
        target_exact_by_candidate=exact,
    )
    summary.update(
        {
            "workload_id": train_contract["workload_id"],
            "model_role": "sft2_endpoint",
            "generation_wall_time_seconds": generation_wall_time,
            "verification_wall_time_seconds": verification_wall_time,
            "run_wall_time_seconds": generation_wall_time + verification_wall_time,
            "sft2_train_integrity_passed": _sft2_train_integrity_passed(summary),
            "verifier_timeout_semantics": (
                "unsuccessful_candidate_not_infrastructure_error"
            ),
        }
    )
    metadata = {
        "schema_version": SFT2_TRAIN_RUN_SCHEMA_VERSION,
        "status": "passed" if summary["sft2_train_integrity_passed"] else "failed",
        "model_role": "sft2_endpoint",
        "model": config.model,
        "adapter": adapter.metadata(),
        "selected_adapter_binding": binding.to_dict(),
        "fixed_complete_q4_endpoint": True,
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "dataset_split": "train",
        "source_membership_workload_id": config.workloads["train"]["id"],
        "workload_id": train_contract["workload_id"],
        "selected_record_ids": selected_ids,
        "prompt_format_id": PROMPT_FORMAT_ID,
        "serialization_or_prompt_transformation": None,
        "exact_target_normalization": (
            "line endings and trailing transport whitespace only"
        ),
        "generation_settings": {
            **sampling,
            "dtype": phase1.engine["dtype"],
            "tensor_parallel_size": phase1.engine["tensor_parallel_size"],
            "max_model_len": phase1.engine["max_model_len"],
            "max_num_seqs": phase1.engine["max_num_seqs"],
            "gpu_memory_utilization": phase1.engine["gpu_memory_utilization"],
            "enforce_eager": phase1.engine["enforce_eager"],
            "quantization": phase1.engine["quantization"],
            "chat_template": None,
        },
        "inference_engine": phase1.engine["name"],
        "inference_engine_version": engine_version,
        "source_repository": phase2_config.source["repository"],
        "source_revision": phase2_config.source["revision"],
        "lean_toolchain": phase2_config.source["lean_toolchain"],
        "verification": {
            "workers": worker_count,
            "timeout_seconds": timeout,
            "original_source_span_reconstruction": True,
            "raw_continuation_no_repair": True,
        },
        "runtime": {
            "python": platform.python_version(),
            **runtime,
            "generation_wall_time_seconds": generation_wall_time,
            "verification_wall_time_seconds": verification_wall_time,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run.json", metadata)
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            value = result.to_dict()
            value["target_exact"] = exact[(result.task_id, result.candidate_index)]
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    if not summary["sft2_train_integrity_passed"]:
        raise RuntimeError("SFT-2 train512 evaluation failed integrity gates")
    return metadata, results, summary


def _merge_sft2_train_reverification_results(
    original: Sequence[CandidateResult],
    replacements: Sequence[CandidateResult],
) -> list[CandidateResult]:
    transient_categories = {"verifier_timeout", "verifier_error"}
    transient_keys = {
        (item.task_id, item.candidate_index)
        for item in original
        if item.category in transient_categories
    }
    replacement_by_key = {
        (item.task_id, item.candidate_index): item for item in replacements
    }
    if len(replacement_by_key) != len(replacements):
        raise ValueError("SFT-2 train512 re-verification replacements are duplicated")
    if set(replacement_by_key) != transient_keys:
        raise ValueError(
            "SFT-2 train512 re-verification must replace exactly transient results"
        )
    original_by_key = {(item.task_id, item.candidate_index): item for item in original}
    if len(original_by_key) != len(original):
        raise ValueError("SFT-2 train512 stored result identities are duplicated")
    for key, replacement in replacement_by_key.items():
        stored = original_by_key[key]
        if (
            replacement.candidate_id != stored.candidate_id
            or replacement.candidate_text != stored.candidate_text
            or replacement.generated_token_count != stored.generated_token_count
            or replacement.finish_reason != stored.finish_reason
            or replacement.generation_latency_seconds
            != stored.generation_latency_seconds
        ):
            raise ValueError(
                "SFT-2 train512 re-verification changed generation evidence"
            )
    return [
        replacement_by_key.get((item.task_id, item.candidate_index), item)
        for item in original
    ]


def reverify_sft2_train512(
    config: SFT2Config,
    phase2_config: Phase2Config,
    dataset_dir: Path,
    mathlib_root: Path,
    workload_path: Path,
    training_path: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[CandidateResult], dict[str, Any]]:
    _, binding = load_sft2_endpoint_binding(
        training_path,
        expected_artifact_id=str(config.lora["artifact_id"]),
        adapter_dir=adapter_dir,
    )
    phase1 = config.phase1_validation_config()
    phase4_adapter_spec(config, adapter_dir).validate(phase1)  # type: ignore[arg-type]
    _validate_phase2_environment(phase2_config, dataset_dir, mathlib_root)
    examples = load_phase6_train_workload(workload_path, config.phase6_config())
    selected_ids = [item.record_id for item in examples]
    train_contract = config.value["train_evaluation"]
    if (
        len(selected_ids) != int(train_contract["expected_examples"])
        or ordered_record_ids_sha256(selected_ids)
        != train_contract["workload_ordered_ids_sha256"]
    ):
        raise ValueError("SFT-2 train512 workload differs from Phase 6")
    records = _load_selected_train_records(dataset_dir, selected_ids)
    for record, example in zip(records, examples, strict=True):
        if record.declaration_name != example.declaration_name:
            raise ValueError("SFT-2 train512 declaration identity differs")
    record_by_id = {record.id: record for record in records}
    sources = {
        record.id: validate_record_source_identity(record, mathlib_root)
        for record in records
    }

    metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    prior_summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("schema_version") != SFT2_TRAIN_RUN_SCHEMA_VERSION
        or metadata.get("model_role") != "sft2_endpoint"
        or metadata.get("selected_adapter_binding") != binding.to_dict()
        or metadata.get("workload_id") != train_contract["workload_id"]
        or metadata.get("selected_record_ids") != selected_ids
    ):
        raise ValueError("stored SFT-2 train512 run differs from the fixed contract")

    stored_values = [
        json.loads(line)
        for line in (output_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    stored_exact = {
        (str(item["task_id"]), int(item["candidate_index"])): bool(
            item.pop("target_exact")
        )
        for item in stored_values
    }
    results = [CandidateResult.from_dict(item) for item in stored_values]
    expected_keys = [
        (task_id, candidate_index)
        for task_id in selected_ids
        for candidate_index in range(4)
    ]
    observed_keys = [(item.task_id, item.candidate_index) for item in results]
    if observed_keys != expected_keys:
        raise ValueError("stored SFT-2 train512 result identities/order differ")
    exact = {
        (item.task_id, item.candidate_index): target_exact(
            item.candidate_text, record_by_id[item.task_id].completion
        )
        for item in results
    }
    if stored_exact != exact:
        raise ValueError("stored SFT-2 train512 exact-target evidence differs")

    timeout = float(config.value["verification"]["train_timeout_seconds"])
    results, restored_count = _restore_repeated_sft2_timeout_categories(results)
    prior_reverification = dict(
        prior_summary.get("reverification", metadata.get("reverification", {}))
    )
    if restored_count:
        reverification = _retained_timeout_reverification(
            prior_reverification,
            restored_count=restored_count,
        )
        verification_wall_time = float(prior_summary["verification_wall_time_seconds"])
    else:
        if int(prior_reverification.get("retained_verifier_timeout_count", 0)) > 0:
            raise ValueError(
                "repeated SFT-2 train512 timeouts require no further retry"
            )
        transient = [
            item
            for item in results
            if item.category in {"verifier_timeout", "verifier_error"}
        ]
        if not transient:
            raise ValueError(
                "stored SFT-2 train512 run has no transient results to retry"
            )
        started = time.perf_counter()
        replacements = []
        for item in transient:
            record = record_by_id[item.task_id]
            generated = GeneratedCandidate(
                task=TaskRecord(
                    id=record.id,
                    preamble="",
                    declaration=record.declaration,
                    declaration_name=record.declaration_name,
                ),
                candidate_index=item.candidate_index,
                text=item.candidate_text,
                token_count=int(item.generated_token_count or 0),
                finish_reason=str(item.finish_reason),
                generation_latency_seconds=float(
                    item.generation_latency_seconds or 0.0
                ),
            )
            replacements.append(
                _heldout_candidate_result(
                    generated,
                    record,
                    sources[item.task_id],
                    mathlib_root,
                    timeout_seconds=timeout,
                )
            )
        retry_wall_time = time.perf_counter() - started
        results = _merge_sft2_train_reverification_results(results, replacements)
        retained_timeout_count = sum(
            item.category == "verifier_timeout" for item in replacements
        )
        prior_retry_attempts = int(
            prior_reverification.get("retry_attempts", 1 if prior_reverification else 0)
        )
        reverification = {
            "stored_candidates_regenerated": False,
            "transient_results_retried": len(transient),
            "categories": sorted({item.category for item in transient}),
            "timeout_seconds": timeout,
            "retry_wall_time_seconds": retry_wall_time,
            "retry_attempts": prior_retry_attempts + 1,
            "total_verification_attempts": prior_retry_attempts + 2,
            "repeated_same_contract_timeout_count": retained_timeout_count,
            "retained_verifier_timeout_count": retained_timeout_count,
            "artifact_category_restoration_only": False,
            "resolution_policy": (
                "a repeated same-contract timeout remains verifier_timeout and is an "
                "unsuccessful candidate, not an infrastructure error"
            ),
        }
        verification_wall_time = (
            float(prior_summary["verification_wall_time_seconds"]) + retry_wall_time
        )
    summary = summarize_phase6_train_results(
        results,
        expected_task_ids=selected_ids,
        target_exact_by_candidate=exact,
    )
    generation_wall_time = float(prior_summary["generation_wall_time_seconds"])
    summary.update(
        {
            "workload_id": train_contract["workload_id"],
            "model_role": "sft2_endpoint",
            "generation_wall_time_seconds": generation_wall_time,
            "verification_wall_time_seconds": verification_wall_time,
            "run_wall_time_seconds": generation_wall_time + verification_wall_time,
            "sft2_train_integrity_passed": _sft2_train_integrity_passed(summary),
            "verifier_timeout_semantics": (
                "unsuccessful_candidate_not_infrastructure_error"
            ),
            "reverification": reverification,
        }
    )
    metadata["status"] = (
        "passed" if summary["sft2_train_integrity_passed"] else "failed"
    )
    metadata["reverification"] = summary["reverification"]
    metadata["verifier_timeout_semantics"] = (
        "unsuccessful_candidate_not_infrastructure_error"
    )
    metadata["runtime"]["candidate_generation_reused"] = True
    metadata["runtime"]["verification_wall_time_seconds"] = verification_wall_time
    _write_json(output_dir / "run.json", metadata)
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            value = result.to_dict()
            value["target_exact"] = exact[(result.task_id, result.candidate_index)]
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    if not summary["sft2_train_integrity_passed"]:
        raise RuntimeError("SFT-2 train512 re-verification failed integrity gates")
    return metadata, results, summary


def run_sft2_heldout512(
    config: SFT2Config,
    phase2_config: Phase2Config,
    dataset_dir: Path,
    mathlib_root: Path,
    workload_path: Path,
    training_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, Any], list[CandidateResult], dict[str, Any]]:
    metadata, results, summary = run_phase4_heldout(
        config,  # type: ignore[arg-type]
        phase2_config,
        dataset_dir,
        mathlib_root,
        workload_path,
        training_path,
        output_dir,
        adapter_dir=adapter_dir,
        verification_workers=verification_workers,
        timeout_seconds=timeout_seconds,
        workload_loader=load_sft2_workloads,
        binding_loader=load_sft2_endpoint_binding,
        schema_version=SFT2_HELDOUT_RUN_SCHEMA_VERSION,
        phase_name="SFT-2",
        integrity_summary_key="sft2_heldout_integrity_passed",
        allow_verifier_timeouts=True,
    )
    summary["sft2_heldout_integrity_passed"] = sft2_evaluation_integrity_passed(summary)
    summary["verifier_timeout_semantics"] = (
        "unsuccessful_candidate_not_infrastructure_error"
    )
    _write_json(output_dir / "summary.json", summary)
    return metadata, results, summary


def run_sft2_minif2f_validation(
    config: SFT2Config,
    benchmark_root: Path,
    training_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[Any, Sequence[CandidateResult], dict[str, Any]]:
    _, binding = load_sft2_endpoint_binding(
        training_path,
        expected_artifact_id=str(config.lora["artifact_id"]),
        adapter_dir=adapter_dir,
    )
    contract = config.value["minif2f"]
    if contract.get("test_evaluation_forbidden") is not True:
        raise ValueError("SFT-2 must forbid miniF2F test evaluation")
    phase1 = config.phase1_validation_config()
    if (
        phase1.benchmark["split"] != "validation"
        or phase1.benchmark["source_path"] != "MiniF2F/Valid.lean"
        or phase1.sampling["candidates_per_task"]
        != int(contract["candidates_per_task"])
    ):
        raise ValueError("SFT-2 miniF2F contract is not frozen validation")
    adapter = phase4_adapter_spec(config, adapter_dir)  # type: ignore[arg-type]
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
        raise ValueError("SFT-2 miniF2F verification settings must be positive")
    metadata, results, summary = run_phase1_baseline(
        phase1,
        benchmark_root,
        str(contract["workload_id"]),
        output_dir,
        timeout_seconds=timeout,
        verification_workers=worker_count,
        adapter=adapter,
        result_schema_version=SFT2_MINIF2F_RUN_SCHEMA_VERSION,
    )
    metadata = replace(metadata, selected_adapter_binding=binding.to_dict())
    _write_json(output_dir / "run.json", metadata.to_dict())
    expected_candidates = int(contract["expected_tasks"]) * int(
        contract["candidates_per_task"]
    )
    token_counts = [int(item.generated_token_count or 0) for item in results]
    passed = bool(
        sft2_evaluation_integrity_passed(summary)
        and len(results) == expected_candidates
    )
    summary.update(
        {
            "schema_version": "sft2-minif2f-validation-summary-v1",
            "sft2_minif2f_validation_integrity_passed": passed,
            "model_role": "sft2_endpoint",
            "selected_adapter_binding": binding.to_dict(),
            "fixed_complete_q4_endpoint": True,
            "expected_tasks": contract["expected_tasks"],
            "expected_candidates": expected_candidates,
            "miniF2F_test_evaluated": False,
            "verifier_timeout_semantics": (
                "unsuccessful_candidate_not_infrastructure_error"
            ),
            "generated_token_counts": {
                "total": sum(token_counts),
                "mean": fmean(token_counts),
                "minimum": min(token_counts),
                "maximum": max(token_counts),
            },
        }
    )
    _write_json(output_dir / "summary.json", summary)
    if not passed:
        raise RuntimeError("SFT-2 miniF2F validation failed integrity gates")
    return metadata, results, summary


def reverify_sft2_minif2f_validation(
    config: SFT2Config,
    benchmark_root: Path,
    training_path: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[CandidateResult], dict[str, Any]]:
    _, binding = load_sft2_endpoint_binding(
        training_path,
        expected_artifact_id=str(config.lora["artifact_id"]),
        adapter_dir=adapter_dir,
    )
    contract = config.value["minif2f"]
    phase1 = config.phase1_validation_config()
    phase4_adapter_spec(config, adapter_dir).validate(phase1)  # type: ignore[arg-type]
    timeout = float(config.value["verification"]["minif2f_timeout_seconds"])
    validate_minif2f_environment(phase1, benchmark_root, timeout_seconds=timeout)
    tasks = phase1.select_workload(
        str(contract["workload_id"]),
        materialize_benchmark_tasks(phase1, benchmark_root),
    )
    task_by_id = {task.id: task for task in tasks}
    task_ids = [task.id for task in tasks]
    candidates_per_task = int(contract["candidates_per_task"])

    metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    prior_summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("schema_version") != SFT2_MINIF2F_RUN_SCHEMA_VERSION
        or metadata.get("selected_adapter_binding") != binding.to_dict()
        or metadata.get("workload_id") != contract["workload_id"]
        or metadata.get("benchmark_split") != "validation"
        or int(metadata.get("candidates_per_task", -1)) != candidates_per_task
    ):
        raise ValueError(
            "stored SFT-2 miniF2F validation run differs from the fixed contract"
        )
    results = [
        CandidateResult.from_dict(json.loads(line))
        for line in (output_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected_keys = [
        (task_id, candidate_index)
        for task_id in task_ids
        for candidate_index in range(candidates_per_task)
    ]
    observed_keys = [(item.task_id, item.candidate_index) for item in results]
    if observed_keys != expected_keys:
        raise ValueError(
            "stored SFT-2 miniF2F validation result identities/order differ"
        )
    results, restored_count = _restore_repeated_sft2_timeout_categories(results)
    prior_reverification = dict(
        prior_summary.get("reverification", metadata.get("reverification", {}))
    )
    if restored_count:
        reverification = _retained_timeout_reverification(
            prior_reverification,
            restored_count=restored_count,
        )
        verification_wall_time = float(prior_summary["verification_wall_time_seconds"])
    else:
        if int(prior_reverification.get("retained_verifier_timeout_count", 0)) > 0:
            raise ValueError("repeated SFT-2 miniF2F timeouts require no further retry")
        transient = [
            item
            for item in results
            if item.category in {"verifier_timeout", "verifier_error"}
        ]
        if not transient:
            raise ValueError(
                "stored SFT-2 miniF2F validation has no transient results to retry"
            )
        generated = [
            GeneratedCandidate(
                task=task_by_id[item.task_id],
                candidate_index=item.candidate_index,
                text=item.candidate_text,
                token_count=int(item.generated_token_count or 0),
                finish_reason=str(item.finish_reason),
                generation_latency_seconds=float(
                    item.generation_latency_seconds or 0.0
                ),
                generation_error=(
                    item.diagnostics["stderr"]
                    if item.category == "generation_error"
                    else None
                ),
            )
            for item in transient
        ]
        verifier = LeanVerifier(benchmark_root, timeout_seconds=timeout)
        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=int(config.value["verification"]["workers"])
        ) as executor:
            replacements = list(
                executor.map(lambda item: _verify_candidate(verifier, item), generated)
            )
        retry_wall_time = time.perf_counter() - started
        results = _merge_sft2_train_reverification_results(results, replacements)
        retained_timeout_count = sum(
            item.category == "verifier_timeout" for item in replacements
        )
        prior_retry_attempts = int(
            prior_reverification.get("retry_attempts", 1 if prior_reverification else 0)
        )
        reverification = {
            "stored_candidates_regenerated": False,
            "transient_results_retried": len(transient),
            "categories": sorted({item.category for item in transient}),
            "timeout_seconds": timeout,
            "retry_wall_time_seconds": retry_wall_time,
            "retry_attempts": prior_retry_attempts + 1,
            "total_verification_attempts": prior_retry_attempts + 2,
            "repeated_same_contract_timeout_count": retained_timeout_count,
            "retained_verifier_timeout_count": retained_timeout_count,
            "artifact_category_restoration_only": False,
            "resolution_policy": (
                "a repeated same-contract timeout remains verifier_timeout and is an "
                "unsuccessful candidate, not an infrastructure error"
            ),
        }
        verification_wall_time = (
            float(prior_summary["verification_wall_time_seconds"]) + retry_wall_time
        )
    summary = summarize_results(
        results,
        expected_task_ids=task_ids,
        candidates_per_task=candidates_per_task,
    )
    token_counts = [int(item.generated_token_count or 0) for item in results]
    generation_wall_time = float(prior_summary["generation_wall_time_seconds"])
    passed = bool(
        sft2_evaluation_integrity_passed(summary)
        and len(results) == len(task_ids) * candidates_per_task
    )
    summary.update(
        {
            "schema_version": "sft2-minif2f-validation-summary-v1",
            "sft2_minif2f_validation_integrity_passed": passed,
            "model_role": "sft2_endpoint",
            "selected_adapter_binding": binding.to_dict(),
            "fixed_complete_q4_endpoint": True,
            "expected_tasks": contract["expected_tasks"],
            "expected_candidates": len(task_ids) * candidates_per_task,
            "miniF2F_test_evaluated": False,
            "workload_id": contract["workload_id"],
            "generation_wall_time_seconds": generation_wall_time,
            "verification_wall_time_seconds": verification_wall_time,
            "run_wall_time_seconds": generation_wall_time + verification_wall_time,
            "verifier_timeout_semantics": (
                "unsuccessful_candidate_not_infrastructure_error"
            ),
            "generated_token_counts": {
                "total": sum(token_counts),
                "mean": fmean(token_counts),
                "minimum": min(token_counts),
                "maximum": max(token_counts),
            },
            "reverification": reverification,
        }
    )
    metadata["status"] = "passed" if passed else "failed"
    metadata["reverification"] = reverification
    metadata["verifier_timeout_semantics"] = (
        "unsuccessful_candidate_not_infrastructure_error"
    )
    metadata["runtime"]["candidate_generation_reused"] = True
    metadata["runtime"]["verification_wall_time_seconds"] = verification_wall_time
    _write_json(output_dir / "run.json", metadata)
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(
            "SFT-2 miniF2F validation re-verification failed integrity gates"
        )
    return metadata, results, summary
