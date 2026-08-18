from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .artifacts import read_artifacts
from .baseline import run_phase1_baseline
from .minif2f import Phase1Config
from .schema import RunMetadata


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BASE_MODEL_ID = "Qwen/Qwen3-8B-Base"
BASE_MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
REFERENCE_ID = "reference-sft-v1"
REFERENCE_ADAPTER_ID = "phase5-train-full-v1-lora"
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
        (("benchmark", "revision"), "f0a20e14c1eeccd859d51bb4c2b3ee487889c303"),
        (("benchmark", "expected_primary_task_count"), 244),
        (("benchmark", "lean_toolchain"), "leanprover/lean4:v4.27.0"),
        (("model", "model_id"), MODEL_ID),
        (("model", "model_revision"), MODEL_REVISION),
        (("model", "tokenizer_id"), MODEL_ID),
        (("model", "tokenizer_revision"), MODEL_REVISION),
        (("engine", "name"), "vllm"),
        (("engine", "version"), "0.10.2"),
        (("engine", "dtype"), "bfloat16"),
        (("engine", "tensor_parallel_size"), 1),
        (("engine", "max_model_len"), 2048),
        (("engine", "enforce_eager"), True),
        (("engine", "quantization"), None),
        (("engine", "expected_cuda_device_name_fragment"), "Ada"),
        (("verifier", "timeout_seconds"), 30.0),
        (("assessment", "prompt_format_id"), "whole-proof-v1"),
        (("assessment", "chat_template"), None),
        (("assessment", "proof_extraction"), False),
        (("assessment", "verifier_feedback"), False),
        (("assessment", "repair"), False),
        (("assessment", "native_mode_diagnostic"), False),
        (("assessment", "environment_probe_timeout_seconds"), 120.0),
    ]
    for field_path, expected in required:
        value: Any = config.value
        for key in field_path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(
                    "missing Qwen3-8B assessment config field: "
                    + ".".join(field_path)
                )
            value = value[key]
        if value != expected:
            raise ValueError(
                f"Qwen3-8B assessment requires {'.'.join(field_path)}={expected!r}, "
                f"got {value!r}"
            )

    if config.sampling != STRICT_SAMPLING:
        raise ValueError("Qwen3-8B assessment sampling differs from the strict contract")
    workloads = config.value.get("workloads", {})
    dev16 = workloads.get(DEV16_WORKLOAD_ID, {})
    full = workloads.get(FULL_WORKLOAD_ID, {})
    if (
        dev16.get("selection") != "explicit_ids"
        or dev16.get("expected_task_count") != 16
        or len(dev16.get("task_ids", [])) != 16
    ):
        raise ValueError("Qwen3-8B dev preflight must freeze exactly 16 task IDs")
    if full.get("selection") != "all" or full.get("expected_task_count") != 244:
        raise ValueError("Qwen3-8B full workload must contain all 244 validation tasks")


def run_strict_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    workload_id: str,
    output_dir: Path,
    *,
    verification_workers: int,
) -> tuple[Any, Any, dict[str, Any]]:
    validate_assessment_config(config)
    if workload_id not in {DEV16_WORKLOAD_ID, FULL_WORKLOAD_ID}:
        raise ValueError(f"unsupported Qwen3-8B strict workload: {workload_id}")
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
    dev16_dir: Path,
    full_dir: Path,
    base_dir: Path,
    reference_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    validate_assessment_config(config)
    dev16 = _compact_run(
        config,
        dev16_dir,
        workload_id=DEV16_WORKLOAD_ID,
        expected_tasks=16,
    )
    full = _compact_run(
        config,
        full_dir,
        workload_id=FULL_WORKLOAD_ID,
        expected_tasks=244,
    )
    base = _load_base_anchor(base_dir)
    reference = _load_reference_anchor(reference_path)

    strict_metrics = full["pass_at_k"]
    comparison = {
        "schema_version": "qwen3-8b-posttrained-comparison-v1",
        "status": "passed",
        "workload_id": FULL_WORKLOAD_ID,
        "strict_model": MODEL_ID,
        "strict_model_revision": MODEL_REVISION,
        "anchors_regenerated": False,
        "candidate_budget_caveat": (
            "The strict post-trained lane uses the issue-mandated four candidates "
            "per task; accepted Qwen3-8B-Base and reference-sft-v1 evidence use "
            "eight. All values use the same pass@k estimator and unchanged raw Lean "
            "verifier, but the finite sampling budgets differ."
        ),
        "metrics": {
            key: {
                "qwen3_8b_posttrained": float(strict_metrics[key]),
                "qwen3_8b_base": float(base["pass_at_k"][key]),
                "reference_sft_v1": float(reference["pass_at_k"][key]),
                "delta_posttrained_minus_base": float(strict_metrics[key])
                - float(base["pass_at_k"][key]),
                "delta_posttrained_minus_reference": float(strict_metrics[key])
                - float(reference["pass_at_k"][key]),
                "fraction_of_base": _ratio(
                    float(strict_metrics[key]), float(base["pass_at_k"][key])
                ),
                "fraction_of_reference": _ratio(
                    float(strict_metrics[key]), float(reference["pass_at_k"][key])
                ),
            }
            for key in ("pass@1", "pass@4")
        },
        "accepted_anchors": {
            "qwen3_8b_base": base,
            "reference_sft_v1": reference,
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
        ("dev16.json", dev16),
        ("full.json", full),
        ("comparison.json", comparison),
    ):
        (evidence_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "README.md").write_text(
        _render_readme(config, dev16, full, comparison), encoding="utf-8"
    )
    return comparison


def _compact_run(
    config: Phase1Config,
    run_dir: Path,
    *,
    workload_id: str,
    expected_tasks: int,
) -> dict[str, Any]:
    metadata, results = read_artifacts(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_candidates = expected_tasks * 4
    settings = metadata.generation_settings or {}
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
        "candidate_budget": metadata.candidates_per_task == 4,
        "strict_sampling": all(settings.get(key) == value for key, value in STRICT_SAMPLING.items()),
        "local_cuda": metadata.runtime.get("inference_execution") == "local_cuda",
        "ada_gpu": "Ada" in str(metadata.runtime.get("cuda_device", "")),
        "bf16": settings.get("dtype") == "bfloat16",
        "no_quantization": settings.get("quantization") is None,
        "raw_continuation": (
            settings.get("chat_template") is None
            and settings.get("prompt_transformation") is None
            and settings.get("adapter") is None
        ),
    }
    if not all(identity_checks.values()):
        failed = [name for name, passed in identity_checks.items() if not passed]
        raise ValueError(f"Qwen3-8B run identity checks failed: {failed}")
    if not summary.get("complete"):
        raise ValueError(
            f"Qwen3-8B run is incomplete: {summary.get('completeness_errors')}"
        )
    if int(summary.get("task_count", -1)) != expected_tasks:
        raise ValueError("Qwen3-8B task count differs from the frozen workload")
    if len(results) != expected_candidates or int(
        summary.get("candidate_count", -1)
    ) != expected_candidates:
        raise ValueError("Qwen3-8B candidate count differs from the frozen workload")
    if int(summary.get("infrastructure_error_count", -1)) != 0:
        raise ValueError("Qwen3-8B run contains generation/verifier errors")
    finish_reasons = summary.get("finish_reason_counts", {})
    if set(finish_reasons) - {"eos", "token_limit"} or sum(
        int(value) for value in finish_reasons.values()
    ) != expected_candidates:
        raise ValueError("Qwen3-8B run has incomplete or unknown finish reasons")
    pass_at_k = summary.get("pass_at_k")
    if not isinstance(pass_at_k, dict) or any(
        key not in pass_at_k for key in ("pass@1", "pass@4")
    ):
        raise ValueError("Qwen3-8B run has no complete pass@1/pass@4 metrics")

    token_counts = [result.generated_token_count for result in results]
    if any(value is None for value in token_counts):
        raise ValueError("Qwen3-8B run is missing generated-token counts")
    concrete_counts = [int(value) for value in token_counts if value is not None]
    generation_wall = float(metadata.runtime["generation_wall_time_seconds"])
    verification_wall = float(metadata.runtime["verification_wall_time_seconds"])
    run_wall = float(summary["run_wall_time_seconds"])
    solved_tasks = int(summary["tasks_with_verified_candidate"]["count"])
    total_tokens = sum(concrete_counts)
    return {
        "schema_version": "qwen3-8b-posttrained-run-evidence-v1",
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
        "pass_at_k": pass_at_k,
        "category_counts": summary["category_counts"],
        "finish_reason_counts": finish_reasons,
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
            "verification_wall": verification_wall,
            "run_wall": run_wall,
        },
        "throughput": {
            "generated_tokens_per_second": total_tokens / generation_wall,
            "candidates_per_generation_second": expected_candidates / generation_wall,
        },
        "compute_per_solved_task": {
            "run_wall_seconds": run_wall / solved_tasks if solved_tasks else None,
            "generation_wall_seconds": generation_wall / solved_tasks
            if solved_tasks
            else None,
            "generated_tokens": total_tokens / solved_tasks if solved_tasks else None,
            "unavailable_reason": None
            if solved_tasks
            else "no task had a verified candidate",
        },
    }


def _load_base_anchor(base_dir: Path) -> dict[str, Any]:
    metadata = RunMetadata.from_dict(
        json.loads((base_dir / "run.json").read_text(encoding="utf-8"))
    )
    summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        metadata.model_id != BASE_MODEL_ID
        or metadata.model_revision != BASE_MODEL_REVISION
        or metadata.adapter_enabled
    ):
        raise ValueError("accepted Qwen3-8B-Base anchor identity differs")
    _validate_anchor_summary(summary, "Qwen3-8B-Base")
    return {
        "id": BASE_MODEL_ID,
        "revision": BASE_MODEL_REVISION,
        "candidate_count": summary["candidate_count"],
        "candidates_per_task": summary["candidates_per_task"],
        "pass_at_k": {key: summary["pass_at_k"][key] for key in ("pass@1", "pass@4")},
        "source": "evidence/phase1/baseline",
    }


def _load_reference_anchor(reference_path: Path) -> dict[str, Any]:
    value = json.loads(reference_path.read_text(encoding="utf-8"))
    summary = value.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("reference-sft-v1 evidence has no summary")
    adapter = value.get("adapter", {})
    if (
        value.get("model_id") != BASE_MODEL_ID
        or value.get("model_revision") != BASE_MODEL_REVISION
        or adapter.get("artifact_id") != REFERENCE_ADAPTER_ID
        or summary.get("adapter_enabled") is not True
    ):
        raise ValueError("accepted reference-sft-v1 anchor identity differs")
    _validate_anchor_summary(summary, REFERENCE_ID)
    return {
        "id": REFERENCE_ID,
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "adapter_id": REFERENCE_ADAPTER_ID,
        "candidate_count": summary["candidate_count"],
        "candidates_per_task": summary["candidates_per_task"],
        "pass_at_k": {key: summary["pass_at_k"][key] for key in ("pass@1", "pass@4")},
        "source": "evidence/phase5/minif2f.json",
    }


def _validate_anchor_summary(summary: dict[str, Any], name: str) -> None:
    metrics = summary.get("pass_at_k")
    if (
        summary.get("complete") is not True
        or summary.get("task_count") != 244
        or summary.get("candidate_count") != 1952
        or summary.get("candidates_per_task") != 8
        or summary.get("infrastructure_error_count") != 0
        or not isinstance(metrics, dict)
        or any(key not in metrics for key in ("pass@1", "pass@4"))
    ):
        raise ValueError(f"accepted {name} anchor is incomplete or incompatible")


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _render_readme(
    config: Phase1Config,
    dev16: dict[str, Any],
    full: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    metrics = comparison["metrics"]
    strict_pass = (
        f"{full['pass_at_k']['pass@1']:.6f}/{full['pass_at_k']['pass@4']:.6f}"
    )
    base_pass = (
        f"{metrics['pass@1']['qwen3_8b_base']:.6f}/"
        f"{metrics['pass@4']['qwen3_8b_base']:.6f}"
    )
    reference_pass = (
        f"{metrics['pass@1']['reference_sft_v1']:.6f}/"
        f"{metrics['pass@4']['reference_sft_v1']:.6f}"
    )
    dev_pass = (
        f"{dev16['pass_at_k']['pass@1']:.6f}/"
        f"{dev16['pass_at_k']['pass@4']:.6f}"
    )
    compute = full["compute_per_solved_task"]
    if compute["run_wall_seconds"] is None:
        compute_text = "unavailable because no task had a verified candidate"
    else:
        compute_text = f"{compute['run_wall_seconds']:.2f} run-wall seconds"
    return f"""# Qwen3-8B official post-trained strict Lean casting assessment

**OBSERVED:** `{MODEL_ID}` completed all {full['task_count']} miniF2F validation
tasks and {full['candidate_count']} raw candidates. It verified
{full['category_counts']['verified']} candidates across
{full['tasks_with_verified_candidate']['count']} tasks. pass@1/pass@4 were
{strict_pass}, compared with {base_pass} for the accepted unchanged
`Qwen/Qwen3-8B-Base` anchor and {reference_pass} for `reference-sft-v1`.

The dev16 preflight completed {dev16['candidate_count']} candidates with
pass@1/pass@4 of {dev_pass}. Both accepted runs contain zero unresolved
generation/verifier infrastructure errors. The full run retained
{full['verifier_timeout_count']} `verifier_timeout` outcomes as unsuccessful
proofs. Compute per solved task was {compute_text}.

**ACCEPTED:** the score uses exact `whole-proof-v1` raw continuation, four
candidates per task, temperature 0.8, top-p 0.95, no top-k, 1,024 new tokens,
seed 0, no chat template, no proof extraction, no verifier feedback, and no
repair. It ran in BF16 without quantization using local vLLM
`{config.engine['version']}` on {full['runtime']['cuda_device']}.

The model and tokenizer are pinned to `{MODEL_REVISION}`. No optional native/chat
diagnostic was run. The official model is Apache-2.0; weights, caches, and raw
candidates remain outside Git. Compact JSON retains execution identity, counts,
category and finish-reason breakdowns, token/latency summaries, wall times,
throughput, and compute per solved task.

`comparison.json` records the finite-budget caveat: this issue mandates four
candidates per task, while both accepted anchors used eight. The anchors were
read from accepted repository evidence and were not rerun.
"""
