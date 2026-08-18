from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

from .artifacts import read_artifacts
from .baseline import run_phase1_baseline
from .minif2f import Phase1Config


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
VLLM_VERSION = "0.23.0"
PREFLIGHT_WORKLOAD_ID = "qwen35-4b-preflight-v1"
DEV16_WORKLOAD_ID = "minif2f-valid-dev16-v1"
FULL_WORKLOAD_ID = "minif2f-valid-v1"
STRICT_SAMPLING = {
    "candidates_per_task": 4,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": -1,
    "max_new_tokens": 1024,
    "stop": "tokenizer_eos_or_token_limit",
    "seed": 0,
}


def load_assessment_config(path: Path) -> Phase1Config:
    config = Phase1Config.load(path)
    validate_assessment_config(config)
    return config


def validate_assessment_config(config: Phase1Config) -> None:
    required = [
        (("model", "model_id"), MODEL_ID),
        (("model", "model_revision"), MODEL_REVISION),
        (("model", "tokenizer_id"), MODEL_ID),
        (("model", "tokenizer_revision"), MODEL_REVISION),
        (("engine", "name"), "vllm"),
        (("engine", "version"), VLLM_VERSION),
        (("engine", "dtype"), "bfloat16"),
        (("engine", "tensor_parallel_size"), 1),
        (("engine", "max_model_len"), 2048),
        (("engine", "enforce_eager"), True),
        (("engine", "quantization"), None),
        (("engine", "language_model_only"), True),
        (("engine", "use_flashinfer_sampler"), False),
        (("engine", "require_gpu_memory_monitor"), True),
        (("engine", "expected_cuda_device_name_fragment"), "Ada"),
        (("verifier", "timeout_seconds"), 30.0),
        (("assessment", "prompt_format_id"), "whole-proof-v1"),
        (("assessment", "chat_template"), None),
        (("assessment", "proof_extraction"), False),
        (("assessment", "verifier_feedback"), False),
        (("assessment", "repair"), False),
        (("assessment", "preflight_workload_id"), PREFLIGHT_WORKLOAD_ID),
        (("assessment", "environment_probe_timeout_seconds"), 120.0),
    ]
    for path, expected in required:
        value: Any = config.value
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(
                    "missing Qwen3.5 assessment config field: " + ".".join(path)
                )
            value = value[key]
        if value != expected:
            raise ValueError(
                f"Qwen3.5 assessment requires {'.'.join(path)}={expected!r}, "
                f"got {value!r}"
            )

    if config.sampling != STRICT_SAMPLING:
        raise ValueError("Qwen3.5 assessment sampling differs from the strict contract")
    workloads = config.value.get("workloads", {})
    preflight = workloads.get(PREFLIGHT_WORKLOAD_ID, {})
    dev16 = workloads.get(DEV16_WORKLOAD_ID, {})
    full = workloads.get(FULL_WORKLOAD_ID, {})
    if preflight.get("expected_task_count") != 1:
        raise ValueError("Qwen3.5 preflight must contain exactly one fixed task")
    if preflight.get("selection") != "explicit_ids" or len(
        preflight.get("task_ids", [])
    ) != 1:
        raise ValueError("Qwen3.5 preflight task must be explicit and immutable")
    if dev16.get("expected_task_count") != 16:
        raise ValueError("Qwen3.5 dev gate must contain 16 tasks")
    if full.get("selection") != "all" or full.get("expected_task_count") != 244:
        raise ValueError("Qwen3.5 full workload must contain all 244 validation tasks")

    preflight_sampling = config.value["assessment"].get("preflight_sampling")
    expected_preflight_sampling = {**STRICT_SAMPLING, "candidates_per_task": 1}
    if preflight_sampling != expected_preflight_sampling:
        raise ValueError("Qwen3.5 preflight sampling differs from its frozen contract")


def run_preflight(
    config: Phase1Config,
    benchmark_root: Path,
    output_dir: Path,
    *,
    verification_workers: int,
) -> tuple[Any, Any, dict[str, Any]]:
    validate_assessment_config(config)
    _configure_pinned_runtime()
    return run_phase1_baseline(
        config,
        benchmark_root,
        PREFLIGHT_WORKLOAD_ID,
        output_dir,
        timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
        verification_workers=verification_workers,
        sampling_override=config.value["assessment"]["preflight_sampling"],
        report_progress=True,
        environment_probe_timeout_seconds=float(
            config.value["assessment"]["environment_probe_timeout_seconds"]
        ),
    )


def run_strict_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    workload_id: str,
    output_dir: Path,
    *,
    verification_workers: int,
) -> tuple[Any, Any, dict[str, Any]]:
    validate_assessment_config(config)
    _configure_pinned_runtime()
    if workload_id not in {DEV16_WORKLOAD_ID, FULL_WORKLOAD_ID}:
        raise ValueError(f"unsupported Qwen3.5 strict workload: {workload_id}")
    return run_phase1_baseline(
        config,
        benchmark_root,
        workload_id,
        output_dir,
        timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
        verification_workers=verification_workers,
        report_progress=True,
        environment_probe_timeout_seconds=float(
            config.value["assessment"]["environment_probe_timeout_seconds"]
        ),
    )


def write_compact_evidence(
    config: Phase1Config,
    *,
    preflight_dir: Path,
    dev16_dir: Path,
    full_dir: Path,
    reference_summary_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    validate_assessment_config(config)
    preflight = _compact_run(
        config,
        preflight_dir,
        workload_id=PREFLIGHT_WORKLOAD_ID,
        expected_tasks=1,
        expected_candidates_per_task=1,
    )
    dev16 = _compact_run(
        config,
        dev16_dir,
        workload_id=DEV16_WORKLOAD_ID,
        expected_tasks=16,
        expected_candidates_per_task=4,
    )
    full = _compact_run(
        config,
        full_dir,
        workload_id=FULL_WORKLOAD_ID,
        expected_tasks=244,
        expected_candidates_per_task=4,
    )
    reference_source = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    reference = reference_source.get("summary", reference_source)
    reference_metrics = reference.get("pass_at_k")
    if not isinstance(reference_metrics, dict):
        raise ValueError("reference-sft-v1 evidence has no pass@k metrics")
    for key in ("pass@1", "pass@4"):
        if key not in reference_metrics:
            raise ValueError(f"reference-sft-v1 evidence has no {key}")

    strict_metrics = full["pass_at_k"]
    comparison = {
        "schema_version": "qwen35-4b-casting-comparison-v1",
        "status": "passed",
        "workload_id": FULL_WORKLOAD_ID,
        "strict_model": MODEL_ID,
        "strict_model_revision": MODEL_REVISION,
        "reference_model": "reference-sft-v1",
        "candidate_budget_caveat": (
            "The strict Qwen3.5 lane uses the issue-mandated four candidates per "
            "task; accepted reference-sft-v1 metrics use eight. Both pass@k values "
            "use the same unbiased estimator and unchanged Lean verifier, but the "
            "finite sampling budgets differ."
        ),
        "metrics": {
            key: {
                "qwen35_4b": float(strict_metrics[key]),
                "reference_sft_v1": float(reference_metrics[key]),
                "delta_qwen35_minus_reference": float(strict_metrics[key])
                - float(reference_metrics[key]),
                "fraction_of_reference": (
                    float(strict_metrics[key]) / float(reference_metrics[key])
                    if float(reference_metrics[key])
                    else None
                ),
            }
            for key in ("pass@1", "pass@4")
        },
        "strict_execution_integrity": {
            "task_count": full["task_count"],
            "candidate_count": full["candidate_count"],
            "infrastructure_error_count": full["infrastructure_error_count"],
            "verifier_timeout_count": full["verifier_timeout_count"],
            "raw_continuation": True,
            "chat_template": None,
            "proof_extraction": False,
            "verifier_feedback": False,
            "repair": False,
        },
    }

    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("preflight.json", preflight),
        ("dev16.json", dev16),
        ("full.json", full),
        ("comparison.json", comparison),
    ):
        (evidence_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "README.md").write_text(
        _render_readme(config, preflight, dev16, full, comparison),
        encoding="utf-8",
    )
    return comparison


def _compact_run(
    config: Phase1Config,
    run_dir: Path,
    *,
    workload_id: str,
    expected_tasks: int,
    expected_candidates_per_task: int,
) -> dict[str, Any]:
    metadata, results = read_artifacts(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_candidates = expected_tasks * expected_candidates_per_task
    identity_checks = {
        "model_id": metadata.model_id == MODEL_ID,
        "model_revision": metadata.model_revision == MODEL_REVISION,
        "tokenizer_id": metadata.tokenizer_id == MODEL_ID,
        "tokenizer_revision": metadata.tokenizer_revision == MODEL_REVISION,
        "workload_id": metadata.workload_id == workload_id,
        "prompt_format_id": metadata.prompt_format_id == "whole-proof-v1",
        "inference_engine": metadata.inference_engine == "vllm",
        "inference_engine_version": metadata.inference_engine_version
        == str(config.engine["version"]),
        "candidate_budget": metadata.candidates_per_task
        == expected_candidates_per_task,
        "local_cuda": metadata.runtime.get("inference_execution") == "local_cuda",
        "ada_gpu": "Ada" in str(metadata.runtime.get("cuda_device", "")),
        "bf16": metadata.generation_settings.get("dtype") == "bfloat16"
        if metadata.generation_settings
        else False,
        "no_quantization": metadata.generation_settings.get("quantization") is None
        if metadata.generation_settings
        else False,
        "language_model_only": metadata.generation_settings.get(
            "language_model_only"
        )
        is True
        if metadata.generation_settings
        else False,
        "native_sampler": (
            metadata.generation_settings.get("use_flashinfer_sampler") is False
            and metadata.runtime.get("sampling_backend") == "vllm_pytorch_native"
        )
        if metadata.generation_settings
        else False,
        "raw_continuation": (
            metadata.generation_settings.get("chat_template") is None
            and metadata.generation_settings.get("prompt_transformation") is None
        )
        if metadata.generation_settings
        else False,
        "gpu_memory_observed": metadata.runtime.get("gpu_memory_monitoring")
        == "nvml_device_used_bytes",
    }
    if not all(identity_checks.values()):
        failed = [name for name, passed in identity_checks.items() if not passed]
        raise ValueError(f"Qwen3.5 run identity checks failed: {failed}")
    if not summary.get("complete"):
        raise ValueError(
            f"Qwen3.5 run is incomplete: {summary.get('completeness_errors')}"
        )
    if int(summary.get("task_count", -1)) != expected_tasks:
        raise ValueError("Qwen3.5 task count differs from the frozen workload")
    if len(results) != expected_candidates or int(
        summary.get("candidate_count", -1)
    ) != expected_candidates:
        raise ValueError("Qwen3.5 candidate count differs from the frozen workload")
    if int(summary.get("infrastructure_error_count", -1)) != 0:
        raise ValueError(
            "Qwen3.5 run contains generation/verifier infrastructure errors"
        )

    token_counts = [result.generated_token_count for result in results]
    if any(value is None for value in token_counts):
        raise ValueError("Qwen3.5 run is missing generated-token counts")
    concrete_counts = [int(value) for value in token_counts if value is not None]
    generation_wall = float(metadata.runtime["generation_wall_time_seconds"])
    run_wall = float(summary["run_wall_time_seconds"])
    solved_tasks = int(summary["tasks_with_verified_candidate"]["count"])
    total_tokens = sum(concrete_counts)
    return {
        "schema_version": "qwen35-4b-casting-run-evidence-v1",
        "status": "passed",
        "workload_id": workload_id,
        "model": {
            "id": metadata.model_id,
            "revision": metadata.model_revision,
            "tokenizer_id": metadata.tokenizer_id,
            "tokenizer_revision": metadata.tokenizer_revision,
            "license": config.value["assessment"]["model_license"],
        },
        "benchmark": {
            "repository": metadata.benchmark_repository,
            "revision": metadata.benchmark_revision,
            "split": metadata.benchmark_split,
            "lean_toolchain": metadata.lean_toolchain,
            "mathlib_revision": metadata.mathlib_revision,
            "verifier_environment": metadata.verifier_environment,
        },
        "runtime": {
            **metadata.runtime,
            "inference_engine": metadata.inference_engine,
            "inference_engine_version": metadata.inference_engine_version,
        },
        "generation_settings": metadata.generation_settings,
        "identity_checks": identity_checks,
        "task_count": summary["task_count"],
        "candidate_count": summary["candidate_count"],
        "candidates_per_task": summary["candidates_per_task"],
        "tasks_with_verified_candidate": summary["tasks_with_verified_candidate"],
        "pass_at_k": summary["pass_at_k"],
        "category_counts": summary["category_counts"],
        "finish_reason_counts": summary["finish_reason_counts"],
        "infrastructure_error_count": summary["infrastructure_error_count"],
        "verifier_timeout_count": summary["verifier_timeout_count"],
        "verifier_timeout_semantics": "unsuccessful_proof_outcome",
        "generated_token_counts": {
            "total": total_tokens,
            "minimum": min(concrete_counts),
            "mean": statistics.fmean(concrete_counts),
            "median": statistics.median(concrete_counts),
            "maximum": max(concrete_counts),
        },
        "timing_seconds": {
            **summary["timing_seconds"],
            "generation_wall": generation_wall,
            "verification_wall": float(
                metadata.runtime["verification_wall_time_seconds"]
            ),
            "run_wall": run_wall,
        },
        "throughput": {
            "generated_tokens_per_second": total_tokens / generation_wall,
            "candidates_per_generation_second": expected_candidates / generation_wall,
        },
        "compute_per_solved_task": {
            "run_wall_seconds": run_wall / solved_tasks if solved_tasks else None,
            "generated_tokens": total_tokens / solved_tasks if solved_tasks else None,
            "unavailable_reason": None
            if solved_tasks
            else "no task had a verified candidate",
        },
    }


def _render_readme(
    config: Phase1Config,
    preflight: dict[str, Any],
    dev16: dict[str, Any],
    full: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    metrics = comparison["metrics"]
    peak_gib = full["runtime"]["gpu_memory_peak_bytes"] / 1024**3
    preflight_peak_gib = preflight["runtime"]["gpu_memory_peak_bytes"] / 1024**3
    strict_pass = (
        f"{full['pass_at_k']['pass@1']:.6f}/{full['pass_at_k']['pass@4']:.6f}"
    )
    reference_pass = (
        f"{metrics['pass@1']['reference_sft_v1']:.6f}/"
        f"{metrics['pass@4']['reference_sft_v1']:.6f}"
    )
    reference_fraction = (
        f"{metrics['pass@1']['fraction_of_reference']:.2%}/"
        f"{metrics['pass@4']['fraction_of_reference']:.2%}"
    )
    dev_pass = f"{dev16['pass_at_k']['pass@1']:.6f}/{dev16['pass_at_k']['pass@4']:.6f}"
    return f"""# Qwen3.5-4B strict Lean casting assessment

**OBSERVED:** the post-trained `{MODEL_ID}` strict raw-continuation lane completed
all {full['task_count']} miniF2F validation tasks and {full['candidate_count']} candidates. It verified
{full['category_counts']['verified']} candidates across {full['tasks_with_verified_candidate']['count']}
tasks. pass@1/pass@4 were {strict_pass}; the accepted `reference-sft-v1`
values were {reference_pass}. The strict scores are {reference_fraction} of those
reference values.

The dev16 gate completed {dev16['candidate_count']} candidates with pass@1/pass@4 of
{dev_pass}. Its exact generated candidates were retained and reverified after
the original parallel run exposed and tests fixed a shared preamble-probe
synchronization defect. The accepted dev evidence has zero timeouts or
infrastructure errors. The real one-task BF16 compatibility preflight peaked at
{preflight_peak_gib:.2f} GiB device memory; the full run peaked at {peak_gib:.2f}
GiB on {full['runtime']['cuda_device']}.

**ACCEPTED:** the primary score uses exact `whole-proof-v1` raw continuation,
four candidates per task, temperature 0.8, top-p 0.95, no top-k, 1,024 new
tokens, seed 0, no chat template, no proof extraction, no verifier feedback,
and no repair. It ran in BF16 without quantization using vLLM
`{config.engine['version']}` on local project-controlled GPU compute. The supported
vLLM PyTorch-native sampler was frozen because FlashInfer 0.6.12's JIT headers
were incompatible with the available packaged CUDA compiler; this changes only
the implementation of the same temperature/top-p sampling contract.

The cold-start environment probe may use 120 seconds to load the pinned Lean
module graph; every generated candidate retains the unchanged 30-second
verifier timeout. The full run retained {full['verifier_timeout_count']} `verifier_timeout`
as an unsuccessful proof outcome and recorded zero infrastructure errors.

The model and tokenizer are pinned to `{MODEL_REVISION}`. The official model is
Apache-2.0; no weights or raw candidate corpus are committed. Compact JSON here
retains package/hardware identity, memory, counts, error and finish-reason
distributions, token totals, timing, throughput, and compute-per-solved-task.
Raw candidates and model caches remain outside Git.

`comparison.json` records the finite-budget caveat: this issue mandates four
candidates per task, while accepted reference evidence used eight. Both use the
same pass@k estimator and unchanged Lean verifier semantics.
"""


def _configure_pinned_runtime() -> None:
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
