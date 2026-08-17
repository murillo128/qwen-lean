from __future__ import annotations

import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from .baseline import (
    LoRAAdapterSpec,
    _generate_candidates,
    _local_cuda_runtime,
    run_phase1_baseline,
)
from .phase2_corpus import read_jsonl_records
from .phase2_extraction import Phase2Config
from .phase2_schema import MathlibProofRecord
from .phase2_verification import validate_record_source_identity
from .phase3_verification import _validate_phase2_environment
from .phase4_inference import _heldout_candidate_result
from .phase6 import (
    REFERENCE_SFT_ID,
    TRAIN_WORKLOAD_ID,
    Phase6Config,
    _write_json,
    load_phase6_train_workload,
    load_reference_candidate,
    summarize_phase6_train_results,
    target_exact,
)
from .prompt import PROMPT_FORMAT_ID
from .schema import CandidateResult, TaskRecord

PHASE6_TRAIN_RUN_SCHEMA_VERSION = "phase6-train-run-v1"
PHASE6_TEST_RUN_SCHEMA_VERSION = "phase6-minif2f-test-run-v1"


def phase6_adapter_spec(config: Phase6Config, adapter_dir: Path) -> LoRAAdapterSpec:
    return LoRAAdapterSpec(
        adapter_id=str(config.adapter["artifact_id"]),
        path=adapter_dir.resolve(),
        rank=int(config.adapter["rank"]),
        base_model_id=str(config.model["model_id"]),
        base_model_revision=str(config.model["model_revision"]),
    )


def _load_selected_train_records(
    dataset_dir: Path, selected_record_ids: list[str]
) -> list[MathlibProofRecord]:
    wanted = set(selected_record_ids)
    selected: dict[str, MathlibProofRecord] = {}
    for record in read_jsonl_records(dataset_dir / "train.jsonl"):
        if record.id not in wanted:
            continue
        if record.split != "train":
            raise ValueError(f"Phase 6 train record {record.id} has wrong split")
        if record.id in selected:
            raise ValueError(f"duplicate Phase 6 train record {record.id}")
        selected[record.id] = record
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(
            "Phase 6 train records are missing from Phase 2: "
            + ", ".join(sorted(missing)[:10])
        )
    return [selected[record_id] for record_id in selected_record_ids]


def run_phase6_train(
    config: Phase6Config,
    phase2_config: Phase2Config,
    dataset_dir: Path,
    mathlib_root: Path,
    workload_path: Path,
    candidate_manifest_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    mode: str,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, Any], list[CandidateResult], dict[str, Any]]:
    if mode not in {"base", "adapter"}:
        raise ValueError(f"unknown Phase 6 train model role: {mode}")
    candidate = load_reference_candidate(config, candidate_manifest_path, adapter_dir)
    _validate_phase2_environment(phase2_config, dataset_dir, mathlib_root)
    examples = load_phase6_train_workload(workload_path, config)
    selected_ids = [item.record_id for item in examples]
    records = _load_selected_train_records(dataset_dir, selected_ids)
    for record, example in zip(records, examples, strict=True):
        if record.declaration_name != example.declaration_name:
            raise ValueError("Phase 6 train declaration identity differs")
        if record.id not in selected_ids:
            raise ValueError("Phase 6 train record is outside the frozen workload")
    sources = {
        record.id: validate_record_source_identity(record, mathlib_root)
        for record in records
    }

    phase1 = config.phase1_test_config()
    adapter = phase6_adapter_spec(config, adapter_dir) if mode == "adapter" else None
    if adapter is not None:
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
    sampling = dict(config.train_generation)
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
            f"Phase 6 vLLM returned {len(generated)} train candidates, "
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
        raise ValueError("Phase 6 train verification settings must be positive")
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
            "workload_id": TRAIN_WORKLOAD_ID,
            "model_role": mode,
            "generation_wall_time_seconds": generation_wall_time,
            "verification_wall_time_seconds": verification_wall_time,
            "run_wall_time_seconds": generation_wall_time + verification_wall_time,
        }
    )
    metadata = {
        "schema_version": PHASE6_TRAIN_RUN_SCHEMA_VERSION,
        "status": "passed" if summary["phase6_train_integrity_passed"] else "failed",
        "model_role": mode,
        "model": config.model,
        "reference_candidate": {
            "logical_id": REFERENCE_SFT_ID,
            "adapter": candidate["adapter"],
            "selection": candidate["selection"],
        },
        "adapter": None if adapter is None else adapter.metadata(),
        "dataset_schema_version": "mathlib-whole-proof-v1",
        "dataset_split": "train",
        "source_membership_workload_id": config.value["phase5_inputs"][
            "train_workload_id"
        ],
        "workload_id": TRAIN_WORKLOAD_ID,
        "selected_record_ids": selected_ids,
        "prompt_format_id": PROMPT_FORMAT_ID,
        "serialization_or_prompt_transformation": None,
        "exact_target_normalization": "line endings and trailing transport whitespace only",
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
    if not summary["phase6_train_integrity_passed"]:
        raise RuntimeError(
            "Phase 6 train evaluation failed integrity gates: "
            f"completeness={summary['completeness_errors']}, "
            f"infrastructure_errors={summary['infrastructure_error_count']}, "
            f"timeouts={summary['verifier_timeout_count']}, "
            f"exact_rejected={summary['exact_target_but_not_verified_count']}"
        )
    return metadata, results, summary


def run_phase6_minif2f_test(
    config: Phase6Config,
    benchmark_root: Path,
    candidate_manifest_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    mode: str,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[Any, list[CandidateResult], dict[str, Any]]:
    if mode not in {"base", "adapter"}:
        raise ValueError(f"unknown Phase 6 miniF2F model role: {mode}")
    candidate = load_reference_candidate(config, candidate_manifest_path, adapter_dir)
    phase1 = config.phase1_test_config()
    adapter = phase6_adapter_spec(config, adapter_dir) if mode == "adapter" else None
    if adapter is not None:
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
    metadata, results, summary = run_phase1_baseline(
        phase1,
        benchmark_root,
        str(config.value["minif2f_test"]["workload_id"]),
        output_dir,
        timeout_seconds=timeout,
        verification_workers=worker_count,
        adapter=adapter,
        result_schema_version=PHASE6_TEST_RUN_SCHEMA_VERSION,
    )
    metadata = replace(
        metadata,
        selected_adapter_binding={
            "logical_id": REFERENCE_SFT_ID,
            "artifact_id": config.adapter["artifact_id"],
            "hub_repository": config.adapter["hub_repository"],
            "hub_revision": config.adapter["hub_revision"],
            "selected_optimizer_step": config.adapter["selected_optimizer_step"],
            "training_artifact_sha256": config.adapter["training_artifact_sha256"],
        },
    )
    _write_json(output_dir / "run.json", metadata.to_dict())
    token_counts = [int(result.generated_token_count or 0) for result in results]
    expected_candidates = int(config.value["minif2f_test"]["expected_tasks"]) * int(
        config.value["minif2f_test"]["candidates_per_task"]
    )
    passed = bool(
        summary["complete"]
        and len(results) == expected_candidates
        and int(summary["infrastructure_error_count"]) == 0
        and int(summary["verifier_timeout_count"]) == 0
    )
    summary.update(
        {
            "schema_version": "phase6-minif2f-test-summary-v1",
            "phase6_minif2f_test_integrity_passed": passed,
            "model_role": mode,
            "reference_candidate_logical_id": candidate["logical_id"],
            "candidate_selection_predates_test": candidate["selection"][
                "selected_before_minif2f_test"
            ],
            "expected_tasks": config.value["minif2f_test"]["expected_tasks"],
            "expected_candidates": expected_candidates,
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
        raise RuntimeError(
            "Phase 6 miniF2F test evaluation failed integrity gates: "
            f"candidates={len(results)}, errors={summary['infrastructure_error_count']}, "
            f"timeouts={summary['verifier_timeout_count']}"
        )
    return metadata, results, summary
