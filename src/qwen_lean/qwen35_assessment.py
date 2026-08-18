from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .artifacts import read_artifacts
from .baseline import (
    run_phase1_baseline,
    validate_minif2f_environment,
    vllm_engine_kwargs,
    vllm_sampling_kwargs,
)
from .minif2f import Phase1Config, materialize_benchmark_tasks
from .prompt import PROMPT_FORMAT_ID, render_prompt
from .schema import CandidateResult


PREFLIGHT_SCHEMA_VERSION = "qwen35-9b-preflight-v1"
EVIDENCE_SCHEMA_VERSION = "qwen35-9b-assessment-evidence-v1"
BF16_LANE = "bf16"
FALLBACK_LANE = "bitsandbytes-4bit"
SUPPORTED_LANES = {BF16_LANE, FALLBACK_LANE}


def validate_assessment_contract(config: Phase1Config) -> None:
    assessment = config.value.get("qwen35_assessment")
    if not isinstance(assessment, dict):
        raise ValueError("missing qwen35_assessment contract")
    expected = {
        ("model", "model_id"): "Qwen/Qwen3.5-9B",
        ("model", "tokenizer_id"): "Qwen/Qwen3.5-9B",
        ("sampling", "candidates_per_task"): 4,
        ("sampling", "temperature"): 0.8,
        ("sampling", "top_p"): 0.95,
        ("sampling", "top_k"): -1,
        ("sampling", "max_new_tokens"): 1024,
        ("sampling", "seed"): 0,
        ("engine", "name"): "vllm",
        ("engine", "dtype"): "bfloat16",
        ("engine", "max_model_len"): 2048,
        ("engine", "language_model_only"): True,
        ("engine", "flashinfer_sampler"): False,
        ("engine", "cpu_offload_gb"): 0.0,
        ("qwen35_assessment", "prompt_format_id"): PROMPT_FORMAT_ID,
        ("qwen35_assessment", "raw_continuation"): True,
        ("qwen35_assessment", "chat_template"): None,
        ("qwen35_assessment", "proof_extraction"): False,
        ("qwen35_assessment", "lean_guided_retry"): False,
    }
    for (section, key), value in expected.items():
        actual = config.value[section][key]
        if actual != value:
            raise ValueError(
                f"Qwen3.5 assessment contract changed at {section}.{key}: "
                f"expected {value!r}, got {actual!r}"
            )
    model_revision = str(config.model["model_revision"])
    tokenizer_revision = str(config.model["tokenizer_revision"])
    if len(model_revision) != 40 or tokenizer_revision != model_revision:
        raise ValueError("model and tokenizer must share one immutable 40-hex revision")
    bf16 = assessment["bf16_lane"]
    fallback = assessment["fallback_lane"]
    if bf16 != {
        "id": "bf16-local-vllm-v1",
        "dtype": "bfloat16",
        "quantization": None,
        "cpu_offload_gb": 0.0,
    }:
        raise ValueError("BF16 preflight lane differs from the frozen contract")
    if fallback != {
        "id": "bitsandbytes-inflight-fp4-w4a32-v1",
        "dtype": "bfloat16",
        "quantization": "bitsandbytes",
        "weight_bits": 4,
        "bnb_4bit_quant_type": "fp4",
        "bnb_4bit_compute_dtype": "float32",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_use_double_quant": False,
        "activation_dtype": "bfloat16",
        "source_checkpoint_dtype": "bfloat16",
        "loading": "inflight",
        "cpu_offload_gb": 0.0,
    }:
        raise ValueError("4-bit fallback lane differs from the frozen contract")


def config_for_lane(config: Phase1Config, lane: str) -> Phase1Config:
    validate_assessment_contract(config)
    if lane not in SUPPORTED_LANES:
        raise ValueError(f"unknown precision lane: {lane}")
    value = copy.deepcopy(config.value)
    lane_config = (
        value["qwen35_assessment"]["bf16_lane"]
        if lane == BF16_LANE
        else value["qwen35_assessment"]["fallback_lane"]
    )
    value["engine"]["dtype"] = lane_config["dtype"]
    value["engine"]["quantization"] = lane_config["quantization"]
    value["engine"]["cpu_offload_gb"] = lane_config["cpu_offload_gb"]
    return Phase1Config(path=config.path, value=value)


def run_precision_preflight(
    config: Phase1Config,
    benchmark_root: Path,
    output: Path,
    *,
    lane: str,
) -> dict[str, Any]:
    validate_assessment_contract(config)
    if lane not in SUPPORTED_LANES:
        raise ValueError(f"unknown precision lane: {lane}")
    state = _load_preflight_state(config, output, lane)
    selected_config = config_for_lane(config, lane)
    tasks = materialize_benchmark_tasks(selected_config, benchmark_root)
    prompt = render_prompt(
        selected_config.select_workload("minif2f-valid-dev16-v1", tasks)[0]
    )
    attempt = _attempt_vllm_preflight(selected_config, prompt, lane)
    state["attempts"].append(attempt)
    if attempt["status"] == "passed":
        state["status"] = "passed"
        state["selected_lane"] = lane
        state["selected_precision"] = attempt["precision"]
    else:
        attempt["failure_kind"] = _classify_preflight_failure(
            str(attempt.get("error", ""))
        )
        state["status"] = "failed"
        state["selected_lane"] = None
        state["selected_precision"] = None
    _write_json(output, state)
    return state


def run_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    preflight: Path,
    workload_id: str,
    output_dir: Path,
    *,
    verification_workers: int,
    timeout_seconds: float,
) -> tuple[Any, list[CandidateResult], dict[str, Any]]:
    validate_assessment_contract(config)
    state = json.loads(preflight.read_text(encoding="utf-8"))
    _validate_preflight_identity(config, state)
    if state.get("status") != "passed" or state.get("selected_lane") not in SUPPORTED_LANES:
        raise ValueError("Qwen3.5 assessment requires a passed precision preflight")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    _configure_cuda_home()
    selected = config_for_lane(config, str(state["selected_lane"]))
    return run_phase1_baseline(
        selected,
        benchmark_root,
        workload_id,
        output_dir,
        timeout_seconds=timeout_seconds,
        verification_workers=verification_workers,
    )


def write_compact_evidence(
    config: Phase1Config,
    preflight_path: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    validate_assessment_contract(config)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_preflight_identity(config, preflight)
    if preflight.get("status") != "passed":
        raise ValueError("cannot write evidence from a failed preflight")
    selected_config = config_for_lane(config, str(preflight["selected_lane"]))
    dev16 = _load_complete_run(
        selected_config, dev16_dir, "minif2f-valid-dev16-v1", 16
    )
    full = _load_complete_run(selected_config, full_dir, "minif2f-valid-v1", 244)
    reference_path = config.path.parents[1] / str(
        config.value["qwen35_assessment"]["reference_evidence"]
    )
    reference_evidence = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = reference_evidence["adapter"]
    current_metrics = full["summary"]["pass_at_k"]
    reference_metrics = reference["pass_at_k"]
    execution_limitations = [
        "Wall-time and candidate-timeout observations include periods of external "
        "non-batch Lean CPU and I/O contention on the shared host; all 10 candidate "
        "timeouts remain unsuccessful under the frozen protocol.",
        "The accepted full verification reuses the exact generated candidates after "
        "serializing the shared verifier preamble probe; model generation was not rerun.",
    ]
    full["execution_limitations"] = execution_limitations
    deltas = {
        key: float(current_metrics[key]) - float(reference_metrics[key])
        for key in ("pass@1", "pass@4")
    }
    if all(value > 0 for value in deltas.values()):
        result = "qwen35_generalist_higher_at_pass1_and_pass4"
    elif all(value < 0 for value in deltas.values()):
        result = "reference_sft_higher_at_pass1_and_pass4"
    else:
        result = "mixed_or_tied_pass1_pass4_result"
    comparison = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "OBSERVED",
        "assessment_id": config.value["qwen35_assessment"]["id"],
        "result": result,
        "strict_lane": {
            "model_id": config.model["model_id"],
            "model_revision": config.model["model_revision"],
            "precision_lane": preflight["selected_lane"],
            "precision": preflight["selected_precision"],
            "pass_at_k": current_metrics,
        },
        "reference": {
            "logical_id": config.value["qwen35_assessment"]["reference_logical_id"],
            "candidates_per_task": reference["candidates_per_task"],
            "pass_at_k": {
                key: reference_metrics[key] for key in ("pass@1", "pass@4")
            },
            "source": config.value["qwen35_assessment"]["reference_evidence"],
        },
        "delta_qwen35_minus_reference": deltas,
        "execution_limitations": execution_limitations,
        "comparison_limitations": [
            "The strict lane uses four candidates per task while the accepted reference estimator uses eight.",
            "A 4-bit selected lane is not precision-identical to the BF16 Qwen3 anchors.",
        ]
        if preflight["selected_lane"] == FALLBACK_LANE
        else [
            "The strict lane uses four candidates per task while the accepted reference estimator uses eight."
        ],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_dir / "preflight.json", _compact_preflight(preflight))
    _write_json(evidence_dir / "dev16.json", dev16)
    _write_json(evidence_dir / "full.json", full)
    _write_json(evidence_dir / "comparison.json", comparison)
    (evidence_dir / "README.md").write_text(
        _render_readme(preflight, dev16, full, comparison), encoding="utf-8"
    )
    return comparison


def _load_preflight_state(
    config: Phase1Config, output: Path, lane: str
) -> dict[str, Any]:
    identity = {
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "engine": config.engine["name"],
        "engine_version": config.engine["version"],
        "assessment_id": config.value["qwen35_assessment"]["id"],
    }
    if lane == BF16_LANE:
        if output.is_file():
            state = json.loads(output.read_text(encoding="utf-8"))
            _validate_preflight_identity(config, state)
            attempts = state.get("attempts", [])
            for attempt in attempts:
                if attempt.get("status") == "failed" and "failure_kind" not in attempt:
                    attempt["failure_kind"] = _classify_preflight_failure(
                        str(attempt.get("error", ""))
                    )
            if (
                state.get("selected_lane") is None
                and attempts
                and all(
                    attempt.get("lane") == BF16_LANE
                    and attempt.get("status") == "failed"
                    for attempt in attempts
                )
            ):
                return state
            raise ValueError("BF16 retry requires only prior failed BF16 attempts")
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "running",
            "identity": identity,
            "benchmark_revision": config.benchmark["revision"],
            "sampling": config.sampling,
            "engine_contract": config.engine,
            "attempts": [],
            "selected_lane": None,
            "selected_precision": None,
        }
    if not output.is_file():
        raise ValueError("4-bit fallback requires a recorded failed BF16 attempt")
    state = json.loads(output.read_text(encoding="utf-8"))
    _validate_preflight_identity(config, state)
    attempts = state.get("attempts", [])
    if (
        not attempts
        or any(
            attempt.get("lane") != BF16_LANE
            or attempt.get("status") != "failed"
            for attempt in attempts
        )
        or attempts[-1].get("failure_kind") != "memory_feasibility"
        or state.get("selected_lane") is not None
    ):
        raise ValueError(
            "4-bit fallback is allowed only after BF16 fails at the memory boundary"
        )
    return state


def _attempt_vllm_preflight(
    config: Phase1Config, prompt: str, lane: str
) -> dict[str, Any]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    _configure_cuda_home()
    started = time.perf_counter()
    monitor = _NvidiaMemoryMonitor()
    runtime = _runtime_versions()
    precision = _precision_metadata(config, lane)
    attempt: dict[str, Any] = {
        "lane": lane,
        "status": "failed",
        "precision": precision,
        "runtime": runtime,
        "prompt_count": 1,
        "candidates_per_prompt": int(config.sampling["candidates_per_task"]),
        "max_new_tokens": int(config.sampling["max_new_tokens"]),
        "max_model_len": int(config.engine["max_model_len"]),
        "generated_candidate_count": 0,
        "generated_token_count": 0,
        "finish_reason_counts": {},
        "error": None,
    }
    llm: Any | None = None
    monitor.start()
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        if vllm.__version__ != str(config.engine["version"]):
            raise RuntimeError(
                f"vLLM version mismatch: expected {config.engine['version']}, "
                f"got {vllm.__version__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("preflight requires a local CUDA GPU")
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        if str(config.engine["expected_cuda_device_name_fragment"]) not in properties.name:
            raise RuntimeError(f"preflight requires the project Ada GPU, got {properties.name}")
        if lane == FALLBACK_LANE:
            _validate_vllm_bnb_defaults(config)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        llm = LLM(**vllm_engine_kwargs(config, config.sampling, None))
        outputs = llm.generate(
            [prompt],
            SamplingParams(**vllm_sampling_kwargs(config.sampling)),
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        completions = outputs[0].outputs
        finish_reasons: dict[str, int] = {}
        for completion in completions:
            reason = "eos" if completion.finish_reason == "stop" else (
                "token_limit" if completion.finish_reason == "length" else str(completion.finish_reason)
            )
            finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
        attempt.update(
            {
                "status": "passed",
                "generated_candidate_count": len(completions),
                "generated_token_count": sum(len(item.token_ids) for item in completions),
                "finish_reason_counts": finish_reasons,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    except Exception as error:
        attempt["error"] = f"{type(error).__name__}: {error}"
        try:
            import torch

            if torch.cuda.is_available():
                attempt["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
                attempt["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        except Exception:
            pass
    finally:
        attempt["wall_time_seconds"] = time.perf_counter() - started
        monitor.stop()
        attempt["peak_nvidia_smi_used_memory_mib"] = monitor.peak_used_memory_mib
        if llm is not None:
            try:
                llm.llm_engine.shutdown()
            except (AttributeError, RuntimeError):
                pass
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    return attempt


def _classify_preflight_failure(message: str) -> str:
    normalized = message.lower()
    memory_markers = (
        "cuda out of memory",
        "free memory on device",
        "insufficient memory",
        "no available memory for the cache",
        "not enough memory",
        "out of memory",
    )
    return (
        "memory_feasibility"
        if any(marker in normalized for marker in memory_markers)
        else "compatibility_or_runtime"
    )


def _runtime_versions() -> dict[str, Any]:
    packages = {}
    for name in (
        "bitsandbytes",
        "huggingface-hub",
        "ninja",
        "nvidia-cuda-crt",
        "nvidia-cuda-nvcc",
        "torch",
        "transformers",
        "vllm",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "inference_execution": "local_cuda",
        "gpu_query": gpu.stdout.strip(),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "vllm_use_flashinfer_sampler": os.environ.get(
            "VLLM_USE_FLASHINFER_SAMPLER"
        ),
    }


def _configure_cuda_home() -> None:
    executable_directory = str(Path(sys.prefix) / "bin")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if executable_directory not in path_entries:
        os.environ["PATH"] = os.pathsep.join(
            [executable_directory, *path_entries]
        )
    if shutil.which("ninja") is None:
        raise RuntimeError("Qwen3.5 GDN JIT requires ninja on PATH")
    if os.environ.get("CUDA_HOME"):
        return
    namespace = importlib.util.find_spec("nvidia")
    if namespace is None or not namespace.submodule_search_locations:
        return
    for location in namespace.submodule_search_locations:
        candidate = Path(location) / "cu13"
        if (candidate / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = str(candidate)
            return


def _precision_metadata(config: Phase1Config, lane: str) -> dict[str, Any]:
    assessment = config.value["qwen35_assessment"]
    if lane == BF16_LANE:
        return copy.deepcopy(assessment["bf16_lane"])
    return copy.deepcopy(assessment["fallback_lane"])


def _validate_vllm_bnb_defaults(config: Phase1Config) -> None:
    from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig

    observed = BitsAndBytesConfig.from_config({})
    expected = config.value["qwen35_assessment"]["fallback_lane"]
    fields = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": expected["bnb_4bit_quant_type"],
        "bnb_4bit_compute_dtype": expected["bnb_4bit_compute_dtype"],
        "bnb_4bit_quant_storage": expected["bnb_4bit_quant_storage"],
        "bnb_4bit_use_double_quant": expected["bnb_4bit_use_double_quant"],
        "llm_int8_enable_fp32_cpu_offload": False,
    }
    for key, value in fields.items():
        if getattr(observed, key) != value:
            raise RuntimeError(
                f"vLLM BitsAndBytes default changed for {key}: "
                f"expected {value!r}, got {getattr(observed, key)!r}"
            )


def _validate_preflight_identity(config: Phase1Config, state: dict[str, Any]) -> None:
    if state.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unknown Qwen3.5 preflight schema")
    identity = state.get("identity", {})
    expected = {
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "engine": config.engine["name"],
        "engine_version": config.engine["version"],
        "assessment_id": config.value["qwen35_assessment"]["id"],
    }
    if identity != expected:
        raise ValueError("Qwen3.5 preflight identity differs from configuration")


def _load_complete_run(
    config: Phase1Config,
    run_dir: Path,
    workload_id: str,
    expected_tasks: int,
) -> dict[str, Any]:
    metadata, results = read_artifacts(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_candidates = expected_tasks * 4
    if not summary.get("complete"):
        raise ValueError(f"incomplete Qwen3.5 run: {workload_id}")
    if (
        summary.get("task_count") != expected_tasks
        or summary.get("candidate_count") != expected_candidates
        or len(results) != expected_candidates
    ):
        raise ValueError(f"wrong task/candidate denominator for {workload_id}")
    if summary.get("infrastructure_error_count") != 0:
        raise ValueError(f"unresolved infrastructure errors in {workload_id}")
    if metadata.workload_id != workload_id:
        raise ValueError(f"workload identity mismatch for {workload_id}")
    if (
        metadata.model_id != config.model["model_id"]
        or metadata.model_revision != config.model["model_revision"]
        or metadata.tokenizer_revision != config.model["tokenizer_revision"]
    ):
        raise ValueError(f"model identity mismatch for {workload_id}")
    settings = metadata.generation_settings or {}
    for key, expected in config.sampling.items():
        if settings.get(key) != expected:
            raise ValueError(f"sampling mismatch for {workload_id}: {key}")
    for key in (
        "dtype",
        "quantization",
        "cpu_offload_gb",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "enforce_eager",
        "language_model_only",
        "flashinfer_sampler",
    ):
        if settings.get(key) != config.engine[key]:
            raise ValueError(f"engine setting mismatch for {workload_id}: {key}")
    if (
        settings.get("chat_template") is not None
        or settings.get("prompt_transformation") is not None
    ):
        raise ValueError("strict lane applied an unapproved prompt transformation")
    generated_lengths = [result.generated_token_count for result in results]
    if any(value is None for value in generated_lengths):
        raise ValueError(f"missing generated token counts in {workload_id}")
    generated = [int(value) for value in generated_lengths if value is not None]
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    generation_wall = float(metadata.runtime["generation_wall_time_seconds"])
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "OBSERVED",
        "workload_id": workload_id,
        "model": {
            "model_id": metadata.model_id,
            "model_revision": metadata.model_revision,
            "tokenizer_id": metadata.tokenizer_id,
            "tokenizer_revision": metadata.tokenizer_revision,
        },
        "engine": {
            "name": metadata.inference_engine,
            "version": metadata.inference_engine_version,
            "generation_settings": settings,
            "runtime": metadata.runtime,
        },
        "summary": {
            key: summary[key]
            for key in (
                "complete",
                "task_count",
                "candidate_count",
                "candidates_per_task",
                "tasks_with_verified_candidate",
                "pass_at_k",
                "category_counts",
                "category_fractions",
                "finish_reason_counts",
                "verifier_timeout_count",
                "infrastructure_error_count",
                "timing_seconds",
                "run_wall_time_seconds",
            )
        },
        "generated_token_counts": _numeric_summary(generated),
        "latency_seconds": {
            "generation": _numeric_summary(
                value
                for value in (item.generation_latency_seconds for item in results)
                if value is not None
            ),
            "verification": _numeric_summary(
                value
                for value in (item.verification_latency_seconds for item in results)
                if value is not None
            ),
            "total": _numeric_summary(item.total_latency_seconds for item in results),
        },
        "compute_per_solved_task": {
            "tasks_with_verified_candidate": solved,
            "generation_wall_seconds_per_solved_task": (
                generation_wall / solved if solved else None
            ),
            "generation_gpu_hours_per_solved_task": (
                generation_wall / 3600 / solved if solved else None
            ),
        },
        "verifier_timeout_semantics": "unsuccessful_candidate_not_infrastructure_error",
        "raw_candidates_retained_outside_git": True,
    }


def _numeric_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    materialized = sorted(values)
    if not materialized:
        return {"count": 0, "total": 0, "minimum": None, "mean": None, "p50": None, "p95": None, "maximum": None}
    return {
        "count": len(materialized),
        "total": sum(materialized),
        "minimum": materialized[0],
        "mean": fmean(materialized),
        "p50": _percentile(materialized, 0.50),
        "p95": _percentile(materialized, 0.95),
        "maximum": materialized[-1],
    }


def _compact_preflight(state: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(state)
    for attempt in compact.get("attempts", []):
        error = attempt.get("error")
        if not isinstance(error, str) or len(error) <= 1000:
            continue
        attempt["error_sha256"] = hashlib.sha256(error.encode("utf-8")).hexdigest()
        attempt["error_original_character_count"] = len(error)
        attempt["error"] = (
            error[:500]
            + "\n...[truncated in committed evidence; raw artifact retained outside Git]...\n"
            + error[-500:]
        )
    return compact


def _percentile(values: list[float | int], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower]) * (1 - weight) + float(values[upper]) * weight


def _render_readme(
    preflight: dict[str, Any],
    dev16: dict[str, Any],
    full: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    dev = dev16["summary"]
    total = full["summary"]
    precision_note = (
        "The accepted result uses the frozen BitsAndBytes in-flight FP4 W4A32 fallback and is not precision-identical to the BF16 Qwen3 anchors."
        if preflight["selected_lane"] == FALLBACK_LANE
        else "The accepted result uses the BF16 lane."
    )
    return f"""# Qwen3.5-9B strict miniF2F casting assessment

**OBSERVED:** `{comparison['result']}`. Qwen3.5-9B strict pass@1/pass@4 were {total['pass_at_k']['pass@1']:.6f}/{total['pass_at_k']['pass@4']:.6f}; `reference-sft-v1` pass@1/pass@4 were {comparison['reference']['pass_at_k']['pass@1']:.6f}/{comparison['reference']['pass_at_k']['pass@4']:.6f}. {precision_note}

| Workload | Tasks | Candidates | pass@1 | pass@4 | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | {dev['task_count']} | {dev['candidate_count']} | {dev['pass_at_k']['pass@1']:.6f} | {dev['pass_at_k']['pass@4']:.6f} | {dev['infrastructure_error_count']} | {dev['verifier_timeout_count']} |
| full validation | {total['task_count']} | {total['candidate_count']} | {total['pass_at_k']['pass@1']:.6f} | {total['pass_at_k']['pass@4']:.6f} | {total['infrastructure_error_count']} | {total['verifier_timeout_count']} |

The strict lane preserves the raw `whole-proof-v1` continuation prompt with no chat template, proof extraction, or Lean-guided retry. Raw candidates, model weights, caches, and bulky logs remain outside Git; the JSON evidence records exact identities, precision, packages, GPU, counts, finish reasons, token lengths, latency, and compute summaries.

Execution limitations: {" ".join(comparison["execution_limitations"])}
"""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _NvidiaMemoryMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_memory_mib: int | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                try:
                    used = int(result.stdout.splitlines()[0].strip())
                    self.peak_used_memory_mib = max(
                        used, self.peak_used_memory_mib or 0
                    )
                except (IndexError, ValueError):
                    pass
            self._stop.wait(0.25)
