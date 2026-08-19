from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .baseline import (
    GeneratedCandidate,
    _generate_candidates,
    _GpuMemoryMonitor,
    _local_cuda_runtime,
    run_phase1_baseline,
)
from .minif2f import Phase1Config
from .prompt import PROMPT_FORMAT_ID
from .qwen3_posttrained_assessment import STRICT_SAMPLING, _compact_run
from .schema import TaskRecord


MODEL_ID = "allenai/Olmo-3-1025-7B"
MODEL_REVISION = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
VLLM_VERSION = "0.12.0"
DEV16_WORKLOAD_ID = "minif2f-valid-dev16-v1"
FULL_WORKLOAD_ID = "minif2f-valid-v1"
PREFLIGHT_SCHEMA_VERSION = "olmo3-7b-preflight-v1"
RUN_EVIDENCE_SCHEMA_VERSION = "olmo3-7b-run-evidence-v1"


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
        (("model", "chat_template"), None),
        (("engine", "name"), "vllm"),
        (("engine", "version"), VLLM_VERSION),
        (("engine", "dtype"), "bfloat16"),
        (("engine", "tensor_parallel_size"), 1),
        (("engine", "gpu_memory_utilization"), 0.95),
        (("engine", "max_model_len"), 2048),
        (("engine", "max_num_seqs"), 8),
        (("engine", "enforce_eager"), True),
        (("engine", "quantization"), None),
        (("engine", "use_flashinfer_sampler"), False),
        (("engine", "vllm_enable_v1_multiprocessing"), False),
        (("engine", "require_gpu_memory_monitor"), True),
        (("engine", "expected_cuda_device_name_fragment"), "Ada"),
        (("verifier", "timeout_seconds"), 30.0),
        (("assessment", "id"), "olmo3-7b-whole-proof-v1"),
        (("assessment", "prompt_format_id"), PROMPT_FORMAT_ID),
        (("assessment", "chat_template"), None),
        (("assessment", "proof_extraction"), False),
        (("assessment", "verifier_feedback"), False),
        (("assessment", "repair"), False),
        (("assessment", "environment_probe_timeout_seconds"), 120.0),
        (("assessment", "preflight_max_new_tokens"), 32),
        (("assessment", "model_license"), "Apache-2.0"),
        (("assessment", "runtime_isolation"), "tools/olmo3-assessment"),
    ]
    for field_path, expected in required:
        value: Any = config.value
        for key in field_path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(
                    "missing OLMo 3 assessment config field: "
                    + ".".join(field_path)
                )
            value = value[key]
        if value != expected:
            raise ValueError(
                f"OLMo 3 assessment requires {'.'.join(field_path)}={expected!r}, "
                f"got {value!r}"
            )

    if config.sampling != STRICT_SAMPLING:
        raise ValueError("OLMo 3 assessment sampling differs from the strict contract")
    required_packages = set(config.engine.get("required_runtime_packages", []))
    if required_packages != {
        "huggingface-hub",
        "nvidia-ml-py",
        "transformers",
        "vllm",
    }:
        raise ValueError("OLMo 3 assessment runtime package contract changed")
    workloads = config.value.get("workloads", {})
    dev16 = workloads.get(DEV16_WORKLOAD_ID, {})
    full = workloads.get(FULL_WORKLOAD_ID, {})
    if (
        dev16.get("selection") != "explicit_ids"
        or dev16.get("expected_task_count") != 16
        or len(dev16.get("task_ids", [])) != 16
    ):
        raise ValueError("OLMo 3 dev smoke must freeze exactly 16 tasks")
    if full.get("selection") != "all" or full.get("expected_task_count") != 244:
        raise ValueError("OLMo 3 full workload must contain all 244 validation tasks")


def run_preflight(config: Phase1Config, output_path: Path) -> dict[str, Any]:
    validate_assessment_config(config)
    _configure_runtime_environment(config)
    runtime = _local_cuda_runtime(config)
    task = TaskRecord(
        id="olmo3-compatibility-preflight",
        preamble="import Mathlib",
        declaration=(
            "theorem olmo3_compatibility_preflight (n : Nat) : n = n := by\n  "
        ),
        declaration_name="olmo3_compatibility_preflight",
    )
    sampling = {
        **config.sampling,
        "candidates_per_task": 1,
        "max_new_tokens": int(
            config.value["assessment"]["preflight_max_new_tokens"]
        ),
    }
    monitor = _GpuMemoryMonitor(
        int(runtime["cuda_device_index"]), required=True, interval_seconds=0.05
    )
    monitor.start()
    try:
        candidates, engine_version = _generate_candidates(
            config, [task], prompts=[task.declaration], sampling=sampling
        )
    finally:
        memory = monitor.stop()
    candidate = _require_successful_preflight(candidates)
    peak = int(memory["gpu_memory_peak_bytes"])
    total = int(runtime["cuda_device_total_memory_bytes"])
    evidence = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "model": dict(config.model),
        "prompt_format_id": PROMPT_FORMAT_ID,
        "raw_continuation": True,
        "chat_template": None,
        "runtime": {
            **runtime,
            **memory,
            "gpu_memory_headroom_at_peak_bytes": total - peak,
            "inference_engine": "vllm",
            "inference_engine_version": engine_version,
            "dtype": config.engine["dtype"],
            "quantization": config.engine["quantization"],
            "vllm_enable_v1_multiprocessing": False,
        },
        "probe": {
            "generated_token_count": candidate.token_count,
            "finish_reason": candidate.finish_reason,
        },
    }
    _validate_preflight(evidence, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def _require_successful_preflight(
    candidates: list[GeneratedCandidate],
) -> GeneratedCandidate:
    if len(candidates) != 1:
        raise RuntimeError("OLMo 3 preflight did not return exactly one candidate")
    candidate = candidates[0]
    if candidate.generation_error is not None:
        raise RuntimeError(candidate.generation_error)
    if candidate.token_count < 1:
        raise RuntimeError("OLMo 3 compatibility preflight generated no tokens")
    return candidate


def run_strict_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    workload_id: str,
    output_dir: Path,
    *,
    verification_workers: int,
) -> tuple[Any, Any, dict[str, Any]]:
    validate_assessment_config(config)
    _configure_runtime_environment(config)
    if workload_id not in {DEV16_WORKLOAD_ID, FULL_WORKLOAD_ID}:
        raise ValueError(f"unsupported OLMo 3 workload: {workload_id}")
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
    preflight_path: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    validate_assessment_config(config)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_preflight(preflight, config)
    dev16 = _compact_olmo_run(config, dev16_dir, DEV16_WORKLOAD_ID, 16)
    full = _compact_olmo_run(config, full_dir, FULL_WORKLOAD_ID, 244)
    outputs = {"preflight": preflight, "dev16": dev16, "full": full}
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (evidence_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "README.md").write_text(
        _render_readme(dev16, full), encoding="utf-8"
    )
    return outputs


def _validate_preflight(value: dict[str, Any], config: Phase1Config) -> None:
    if value.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("OLMo 3 preflight schema differs")
    if value.get("status") != "passed" or value.get("model") != config.model:
        raise ValueError("OLMo 3 preflight is incomplete or has the wrong model")
    if value.get("prompt_format_id") != PROMPT_FORMAT_ID:
        raise ValueError("OLMo 3 preflight prompt format changed")
    if value.get("raw_continuation") is not True or value.get("chat_template") is not None:
        raise ValueError("OLMo 3 preflight did not preserve raw continuation")
    runtime = value.get("runtime", {})
    expected = {
        "inference_execution": "local_cuda",
        "inference_engine": "vllm",
        "inference_engine_version": VLLM_VERSION,
        "dtype": "bfloat16",
        "quantization": None,
        "vllm_enable_v1_multiprocessing": False,
    }
    for key, expected_value in expected.items():
        if runtime.get(key) != expected_value:
            raise ValueError(f"OLMo 3 preflight runtime changed: {key}")
    if "Ada" not in str(runtime.get("cuda_device")):
        raise ValueError("OLMo 3 preflight did not use the project Ada GPU")
    if int(runtime.get("gpu_memory_peak_bytes", 0)) <= 0:
        raise ValueError("OLMo 3 preflight lacks peak GPU memory")
    if int(runtime.get("gpu_memory_headroom_at_peak_bytes", 0)) <= 0:
        raise ValueError("OLMo 3 BF16 preflight exceeded physical GPU memory")
    package_versions = runtime.get("package_versions", {})
    for package in config.engine["required_runtime_packages"]:
        if package not in package_versions:
            raise ValueError(f"OLMo 3 preflight lacks runtime package: {package}")


def _compact_olmo_run(
    config: Phase1Config,
    run_dir: Path,
    workload_id: str,
    expected_tasks: int,
) -> dict[str, Any]:
    value = _compact_run(
        config,
        run_dir,
        workload_id=workload_id,
        expected_tasks=expected_tasks,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        assessment_name="OLMo 3 7B",
        run_evidence_schema=RUN_EVIDENCE_SCHEMA_VERSION,
    )
    runtime = value["runtime"]
    peak = int(runtime.get("gpu_memory_peak_bytes", 0))
    total = int(runtime.get("cuda_device_total_memory_bytes", 0))
    if peak <= 0 or total <= peak:
        raise ValueError("OLMo 3 run lacks valid peak GPU-memory evidence")
    required_versions = runtime.get("package_versions", {})
    for package in config.engine["required_runtime_packages"]:
        if package not in required_versions:
            raise ValueError(f"OLMo 3 run lacks runtime package: {package}")
    runtime["gpu_memory_headroom_at_peak_bytes"] = total - peak
    runtime["vllm_enable_v1_multiprocessing"] = False
    return value


def _configure_runtime_environment(config: Phase1Config) -> None:
    enabled = bool(config.engine["vllm_enable_v1_multiprocessing"])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if enabled else "0"


def _render_readme(dev16: dict[str, Any], full: dict[str, Any]) -> str:
    def row(label: str, value: dict[str, Any]) -> str:
        return (
            f"| {label} | {value['task_count']} | {value['candidate_count']} | "
            f"{value['tasks_with_verified_candidate']['count']} | "
            f"{value['pass_at_k']['pass@1']:.6f} | "
            f"{value['pass_at_k']['pass@4']:.6f} | "
            f"{value['category_counts']['verified']} | "
            f"{value['infrastructure_error_count']} | "
            f"{value['verifier_timeout_count']} |"
        )

    runtime = full["runtime"]
    compute = full["compute_per_solved_task"]
    compute_text = (
        "unavailable because no task was solved"
        if compute["run_wall_seconds"] is None
        else f"{compute['run_wall_seconds']:.3f} run-wall seconds per solved task"
    )
    return f"""# OLMo 3 7B whole-proof assessment

**OBSERVED:** the official `{MODEL_ID}` Base checkpoint was evaluated under the
unchanged `whole-proof-v1` raw-continuation and Lean-verification contract. This
is independent model-assessment evidence, not a training or promotion result.

| Workload | Tasks | Candidates | Solved tasks | pass@1 | pass@4 | Verified candidates | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{row('dev16 smoke', dev16)}
{row('full validation', full)}

**ACCEPTED:** generation used temperature 0.8, top-p 0.95, no top-k, a 1,024
generated-token cap, seed 0, and four candidates per task. No chat template,
proof extraction, repair, Lean feedback, candidate regeneration, or hosted
inference was used.
`verifier_timeout` remains an unsuccessful proof outcome.

**OBSERVED:** the full run generated {full['generated_token_counts']['total']}
tokens; finish reasons were `{json.dumps(full['finish_reason_counts'], sort_keys=True)}`
and evaluator categories were `{json.dumps(full['category_counts'], sort_keys=True)}`.
Generation took {full['timing_seconds']['generation_wall']:.3f} seconds at
{full['throughput']['generated_tokens_per_second']:.3f} generated tokens/second;
end-to-end run time was {full['timing_seconds']['run_wall']:.3f} seconds. Compute
per solved task was {compute_text}.

**OBSERVED:** inference executed locally in BF16 without quantization using vLLM
{runtime['inference_engine_version']} on `{runtime['cuda_device']}`. Peak observed
GPU memory was {runtime['gpu_memory_peak_bytes']} of
{runtime['cuda_device_total_memory_bytes']} bytes. The Apache-2.0 model and
tokenizer are pinned to `{MODEL_REVISION}`. The isolated runtime package versions
are retained in the JSON evidence; weights, caches, raw candidates, and bulky
logs remain outside Git.
"""
