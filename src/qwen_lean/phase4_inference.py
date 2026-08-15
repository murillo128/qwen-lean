from __future__ import annotations

import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from .baseline import (
    GeneratedCandidate,
    LoRAAdapterSpec,
    _generate_candidates,
    _local_cuda_runtime,
    run_phase1_baseline,
    vllm_engine_kwargs,
)
from .metrics import summarize_results
from .minif2f import Phase1Config
from .phase2_corpus import read_jsonl_records
from .phase2_extraction import Phase2Config
from .phase2_schema import MathlibProofRecord
from .phase2_verification import run_lean_source, validate_record_source_identity
from .phase3_verification import (
    _validate_phase2_environment,
    reconstruct_generated_proof,
)
from .phase4 import Phase4Config, load_phase4_workloads
from .schema import CandidateResult, TaskRecord
from .prompt import PROMPT_FORMAT_ID, normalize_transport


PHASE4_HELDOUT_RUN_SCHEMA_VERSION = "phase4-heldout-run-v1"
PHASE4_HELDOUT_COMPARISON_SCHEMA_VERSION = "phase4-heldout-comparison-v1"


def phase4_heldout_integrity_passed(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("complete")
        and int(summary.get("infrastructure_error_count", -1)) == 0
        and int(summary.get("verifier_timeout_count", -1)) == 0
    )


def _sanitized_adapter_metadata(
    value: dict[str, Any] | None, selected_step: int
) -> dict[str, Any] | None:
    if value is None:
        return None
    compact = {key: item for key, item in value.items() if key != "adapter_path"}
    compact["ignored_local_path"] = (
        f"artifacts/phase4/training/trainer-state/checkpoint-{selected_step}"
    )
    return compact


def _runtime_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "python",
            "torch",
            "torch_cuda_version",
            "inference_execution",
            "cuda_device_index",
            "cuda_device",
            "cuda_device_capability",
            "cuda_device_total_memory_bytes",
        )
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _phase1_config(config: Phase4Config) -> Phase1Config:
    project_root = config.path.parents[1]
    phase1 = Phase1Config.load(
        project_root / str(config.value["minif2f"]["phase1_config"])
    )
    expected_model = {
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
    }
    for key, expected in expected_model.items():
        if phase1.model[key] != expected:
            raise ValueError(f"Phase 1 {key} differs from Phase 4")
    expected_phase1_sampling = {
        "candidates_per_task": 8,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_new_tokens": 1024,
        "stop": "tokenizer_eos_or_token_limit",
        "seed": 0,
    }
    if phase1.sampling != expected_phase1_sampling:
        raise ValueError("Phase 1 sampling contract differs from Phase 4 miniF2F")
    if (
        str(phase1.engine["name"]) != "vllm"
        or str(phase1.engine["dtype"]) != "bfloat16"
        or int(phase1.engine["max_model_len"]) != 2048
    ):
        raise ValueError("Phase 1 inference engine differs from Phase 4 contract")
    return phase1


def phase4_adapter_spec(config: Phase4Config, adapter_dir: Path) -> LoRAAdapterSpec:
    return LoRAAdapterSpec(
        adapter_id=str(config.lora["artifact_id"]),
        path=adapter_dir.resolve(),
        rank=int(config.lora["r"]),
        base_model_id=str(config.model["model_id"]),
        base_model_revision=str(config.model["model_revision"]),
    )


def heldout_generation_request(
    config: Phase4Config, adapter_dir: Path | None
) -> dict[str, Any]:
    phase1 = _phase1_config(config)
    adapter = None if adapter_dir is None else phase4_adapter_spec(config, adapter_dir)
    return {
        "sampling": dict(config.value["heldout_generation"]),
        "engine": vllm_engine_kwargs(
            phase1, config.value["heldout_generation"], adapter
        ),
        "adapter": None if adapter is None else adapter.metadata(),
    }


def _validate_selected_adapter(
    config: Phase4Config, training_path: Path, adapter_dir: Path
) -> int:
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("status") != "passed":
        raise ValueError("Phase 4 post-selection evaluation requires passed training")
    selection = training.get("checkpoint_selection") or {}
    selected_step = int(selection.get("selected_optimizer_step", -1))
    if adapter_dir.name != f"checkpoint-{selected_step}":
        raise ValueError("evaluation adapter is not the validation-selected checkpoint")
    if bool(selection.get("heldout_or_minif2f_consulted", True)):
        raise ValueError(
            "Phase 4 checkpoint selection consulted post-selection results"
        )
    phase4_adapter_spec(config, adapter_dir).validate(_phase1_config(config))
    return selected_step


def _load_heldout_records(
    dataset_dir: Path, selected_record_ids: Sequence[str]
) -> list[MathlibProofRecord]:
    selected_ids = set(selected_record_ids)
    selected: dict[str, MathlibProofRecord] = {}
    for record in read_jsonl_records(dataset_dir / "heldout.jsonl"):
        if record.id not in selected_ids:
            continue
        if record.split != "heldout":
            raise ValueError(f"Phase 4 heldout record {record.id} has wrong split")
        if record.id in selected:
            raise ValueError(f"duplicate Phase 4 heldout record {record.id}")
        selected[record.id] = record
    missing = selected_ids - selected.keys()
    if missing:
        raise ValueError(
            "Phase 4 heldout records are missing from Phase 2: "
            + ", ".join(sorted(missing))
        )
    return [selected[record_id] for record_id in selected_record_ids]


def _heldout_candidate_result(
    generated: GeneratedCandidate,
    record: MathlibProofRecord,
    source: str,
    mathlib_root: Path,
    *,
    timeout_seconds: float,
) -> CandidateResult:
    if generated.generation_error is not None:
        return CandidateResult(
            task_id=record.id,
            candidate_id=f"model-{generated.candidate_index}",
            candidate_index=generated.candidate_index,
            candidate_text=generated.text,
            category="generation_error",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": generated.generation_error},
            generation_latency_seconds=generated.generation_latency_seconds,
            verification_latency_seconds=None,
            total_latency_seconds=generated.generation_latency_seconds,
            generated_token_count=generated.token_count,
            finish_reason=generated.finish_reason,
        )
    if not normalize_transport(generated.text):
        return CandidateResult(
            task_id=record.id,
            candidate_id=f"model-{generated.candidate_index}",
            candidate_index=generated.candidate_index,
            candidate_text=generated.text,
            category="empty_candidate",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": "empty generated continuation"},
            generation_latency_seconds=generated.generation_latency_seconds,
            verification_latency_seconds=0.0,
            total_latency_seconds=generated.generation_latency_seconds,
            generated_token_count=generated.token_count,
            finish_reason=generated.finish_reason,
        )
    try:
        reconstructed = reconstruct_generated_proof(source, record, generated.text)
        check = run_lean_source(
            reconstructed,
            mathlib_root,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:
        check = None
        category = "verifier_error"
        exit_code = None
        latency = None
        diagnostic = f"{type(error).__name__}: {error}"
    else:
        category = {
            "accepted": "verified",
            "rejected": "lean_rejected",
            "timeout": "verifier_timeout",
            "infrastructure_error": "verifier_error",
        }[check.status]
        exit_code = check.exit_code
        latency = check.latency_seconds
        diagnostic = check.diagnostic
    return CandidateResult(
        task_id=record.id,
        candidate_id=f"model-{generated.candidate_index}",
        candidate_index=generated.candidate_index,
        candidate_text=generated.text,
        category=category,  # type: ignore[arg-type]
        lean_exit_code=exit_code,
        diagnostics={"stdout": "", "stderr": diagnostic},
        generation_latency_seconds=generated.generation_latency_seconds,
        verification_latency_seconds=latency,
        total_latency_seconds=(
            generated.generation_latency_seconds + (0.0 if latency is None else latency)
        ),
        generated_token_count=generated.token_count,
        finish_reason=generated.finish_reason,
    )


def run_phase4_heldout(
    config: Phase4Config,
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
) -> tuple[dict[str, Any], list[CandidateResult], dict[str, Any]]:
    mode = "base" if adapter_dir is None else "adapter"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    selection = training.get("checkpoint_selection") or {}
    if training.get("status") != "passed" or not selection:
        raise ValueError("Phase 4 heldout evaluation requires frozen selection")
    if adapter_dir is not None:
        _validate_selected_adapter(config, training_path, adapter_dir)
    _validate_phase2_environment(phase2_config, dataset_dir, mathlib_root)
    workloads = load_phase4_workloads(workload_path, config)
    selected_ids = [item.record_id for item in workloads.heldout]
    records = _load_heldout_records(dataset_dir, selected_ids)
    for record, item in zip(records, workloads.heldout, strict=True):
        if record.declaration_name != item.declaration_name:
            raise ValueError("Phase 4 heldout declaration identity differs")
    sources = {
        record.id: validate_record_source_identity(record, mathlib_root)
        for record in records
    }
    phase1 = _phase1_config(config)
    adapter = None if adapter_dir is None else phase4_adapter_spec(config, adapter_dir)
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
    prompts = [item.prompt for item in workloads.heldout]
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
            f"Phase 4 vLLM returned {len(generated)} candidates, "
            f"expected {expected_candidates}"
        )
    record_by_id = {record.id: record for record in records}
    worker_count = int(
        config.value["verification"]["workers"]
        if verification_workers is None
        else verification_workers
    )
    timeout = float(
        config.value["verification"]["heldout_timeout_seconds"]
        if timeout_seconds is None
        else timeout_seconds
    )
    if worker_count < 1 or timeout <= 0:
        raise ValueError("Phase 4 heldout verification settings must be positive")
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(
                lambda item: _heldout_candidate_result(
                    item,
                    record_by_id[item.task.id],
                    sources[item.task.id],
                    mathlib_root,
                    timeout_seconds=timeout,
                ),
                generated,
            )
        )
    verification_wall_time = time.perf_counter() - verification_started
    summary = summarize_results(
        results,
        expected_task_ids=selected_ids,
        candidates_per_task=int(sampling["candidates_per_task"]),
        ks=(1, 4),
    )
    generated_token_counts = [
        int(result.generated_token_count or 0) for result in results
    ]
    summary.update(
        {
            "workload_id": config.workloads["heldout"]["id"],
            "model_role": mode,
            "generated_token_counts": {
                "total": sum(generated_token_counts),
                "mean": fmean(generated_token_counts),
                "minimum": min(generated_token_counts),
                "maximum": max(generated_token_counts),
            },
            "generation_wall_time_seconds": generation_wall_time,
            "verification_wall_time_seconds": verification_wall_time,
            "run_wall_time_seconds": (generation_wall_time + verification_wall_time),
        }
    )
    heldout_integrity_passed = phase4_heldout_integrity_passed(summary)
    summary["phase4_heldout_integrity_passed"] = heldout_integrity_passed
    metadata = {
        "schema_version": PHASE4_HELDOUT_RUN_SCHEMA_VERSION,
        "status": "passed" if heldout_integrity_passed else "failed",
        "model_role": mode,
        "model": config.model,
        "adapter": None if adapter is None else adapter.metadata(),
        "selected_optimizer_step": int(selection["selected_optimizer_step"]),
        "checkpoint_selection_metric": selection["metric"],
        "post_selection_evaluation": True,
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "dataset_split": "heldout",
        "workload_id": config.workloads["heldout"]["id"],
        "selected_record_ids": selected_ids,
        "prompt_format_id": PROMPT_FORMAT_ID,
        "serialization_or_prompt_transformation": None,
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
            stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    if not heldout_integrity_passed:
        raise RuntimeError(
            f"Phase 4 {mode} heldout evaluation failed integrity gates: "
            f"completeness={summary['completeness_errors']}, "
            f"infrastructure_errors={summary['infrastructure_error_count']}, "
            f"verifier_timeouts={summary['verifier_timeout_count']}"
        )
    return metadata, results, summary


def compare_phase4_heldout_runs(
    base_dir: Path, adapter_dir: Path, output: Path
) -> dict[str, Any]:
    base_run = json.loads((base_dir / "run.json").read_text(encoding="utf-8"))
    adapter_run = json.loads((adapter_dir / "run.json").read_text(encoding="utf-8"))
    base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    adapter_summary = json.loads(
        (adapter_dir / "summary.json").read_text(encoding="utf-8")
    )
    for key in (
        "model",
        "selected_optimizer_step",
        "dataset_schema_version",
        "dataset_split",
        "workload_id",
        "selected_record_ids",
        "prompt_format_id",
        "serialization_or_prompt_transformation",
        "generation_settings",
        "inference_engine",
        "inference_engine_version",
        "source_repository",
        "source_revision",
        "lean_toolchain",
        "verification",
    ):
        if base_run.get(key) != adapter_run.get(key):
            raise ValueError(f"Phase 4 heldout comparison differs in {key}")
    if base_run.get("model_role") != "base" or base_run.get("adapter") is not None:
        raise ValueError("Phase 4 base run unexpectedly enabled an adapter")
    if adapter_run.get("model_role") != "adapter" or not adapter_run.get(
        "adapter", {}
    ).get("enabled"):
        raise ValueError("Phase 4 adapter run did not enable the selected adapter")
    base_runtime = _runtime_identity(base_run["runtime"])
    adapter_runtime = _runtime_identity(adapter_run["runtime"])
    if base_runtime != adapter_runtime:
        raise ValueError("Phase 4 heldout comparison differs in local runtime identity")
    for role, summary in (("base", base_summary), ("adapter", adapter_summary)):
        if not phase4_heldout_integrity_passed(summary):
            raise ValueError(
                f"Phase 4 heldout {role} run has incomplete, infrastructure-error, "
                "or timeout results"
            )
        summary["phase4_heldout_integrity_passed"] = True
    base_metrics = base_summary["pass_at_k"]
    adapter_metrics = adapter_summary["pass_at_k"]
    comparison = {
        "schema_version": PHASE4_HELDOUT_COMPARISON_SCHEMA_VERSION,
        "status": "passed",
        "comparison_integrity_passed": True,
        "adapter_improvement_required": False,
        "workload_id": base_run["workload_id"],
        "selected_record_ids": base_run["selected_record_ids"],
        "selected_optimizer_step": base_run["selected_optimizer_step"],
        "evaluation_contract": {
            "model": base_run["model"],
            "dataset_schema_version": base_run["dataset_schema_version"],
            "dataset_split": base_run["dataset_split"],
            "prompt_format_id": base_run["prompt_format_id"],
            "serialization_or_prompt_transformation": base_run[
                "serialization_or_prompt_transformation"
            ],
            "generation_settings": base_run["generation_settings"],
            "inference_engine": {
                "name": base_run["inference_engine"],
                "version": base_run["inference_engine_version"],
            },
            "source_repository": base_run["source_repository"],
            "source_revision": base_run["source_revision"],
            "lean_toolchain": base_run["lean_toolchain"],
            "verification": base_run["verification"],
        },
        "runs": {
            "base": {
                "adapter": None,
                "runtime": base_runtime,
            },
            "adapter": {
                "adapter": _sanitized_adapter_metadata(
                    adapter_run["adapter"], int(base_run["selected_optimizer_step"])
                ),
                "runtime": adapter_runtime,
            },
        },
        "base": base_summary,
        "adapter": adapter_summary,
        "delta_adapter_minus_base": {
            key: float(adapter_metrics[key]) - float(base_metrics[key])
            for key in ("pass@1", "pass@4")
        },
        "raw_candidate_results_retained_outside_git": True,
    }
    _write_json(output, comparison)
    return comparison


def run_phase4_minif2f(
    config: Phase4Config,
    benchmark_root: Path,
    training_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    verification_workers: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[Any, Sequence[Any], dict[str, Any]]:
    selected_step = _validate_selected_adapter(config, training_path, adapter_dir)
    phase1 = _phase1_config(config)
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
        raise ValueError("Phase 4 miniF2F verification settings must be positive")
    metadata, results, summary = run_phase1_baseline(
        phase1,
        benchmark_root,
        str(config.value["minif2f"]["workload_id"]),
        output_dir,
        timeout_seconds=timeout,
        verification_workers=worker_count,
        adapter=phase4_adapter_spec(config, adapter_dir),
    )
    expected_candidates = 16 * int(phase1.sampling["candidates_per_task"])
    passed = bool(
        summary["complete"]
        and len(results) == expected_candidates
        and int(summary["infrastructure_error_count"]) == 0
        and int(summary["verifier_timeout_count"]) == 0
    )
    project_root = config.path.parents[1]
    accepted_base = json.loads(
        (project_root / "evidence/phase1/dev16/summary.json").read_text(
            encoding="utf-8"
        )
    )
    expected_base_metrics = {"pass@1": 0.0, "pass@4": 0.0, "pass@8": 0.0}
    if accepted_base.get("pass_at_k") != expected_base_metrics:
        raise ValueError("accepted Phase 1 dev16 base evidence has changed")
    summary.update(
        {
            "phase4_minif2f_comparison": True,
            "phase1_quality_comparable": True,
            "selected_optimizer_step": selected_step,
            "checkpoint_selection_influenced_by_minif2f": False,
            "adapter_enabled": True,
            "adapter_id": config.lora["artifact_id"],
            "expected_candidates": expected_candidates,
            "observed_candidates": len(results),
            "accepted_phase1_base_reference": {
                "path": "evidence/phase1/dev16/summary.json",
                "pass_at_k": expected_base_metrics,
                "regenerated": False,
            },
            "phase4_minif2f_passed": passed,
        }
    )
    _write_json(output_dir / "summary.json", summary)
    if not passed:
        raise RuntimeError(
            "Phase 4 miniF2F adapter evaluation failed integrity gates: "
            f"candidates={len(results)}, errors={summary['infrastructure_error_count']}, "
            f"timeouts={summary['verifier_timeout_count']}"
        )
    return metadata, results, summary
