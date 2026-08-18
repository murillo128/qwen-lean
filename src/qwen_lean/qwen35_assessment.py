from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .artifacts import read_artifacts, write_artifacts
from .baseline import _generate_candidates, _local_cuda_runtime, run_phase1_baseline
from .minif2f import Phase1Config
from .prompt import PROMPT_FORMAT_ID
from .schema import CandidateResult, RunMetadata, TaskRecord


ASSESSMENT_SCHEMA_VERSION = "qwen35-2b-base-assessment-v1"
PREFLIGHT_SCHEMA_VERSION = "qwen35-2b-base-preflight-v1"
EVIDENCE_SCHEMA_VERSION = "qwen35-2b-base-evidence-v1"
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
MODEL_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
VLLM_VERSION = "0.17.0"
EXPECTED_WORKLOADS = {
    "minif2f-valid-dev16-v1": (16, 64),
    "minif2f-valid-v1": (244, 976),
}


def validate_assessment_config(config: Phase1Config) -> None:
    assessment = config.value.get("assessment", {})
    if assessment.get("schema_version") != ASSESSMENT_SCHEMA_VERSION:
        raise ValueError("unknown Qwen3.5 assessment contract")
    if assessment.get("artifact_namespace") != "qwen35-2b-base":
        raise ValueError("Qwen3.5 assessment must use its isolated artifact namespace")
    if assessment.get("model_license") != "Apache-2.0":
        raise ValueError("Qwen3.5 model license must be recorded")

    expected_model = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
    }
    if config.model != expected_model:
        raise ValueError("Qwen3.5 model and tokenizer identity must remain pinned")

    expected_sampling = {
        "candidates_per_task": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_new_tokens": 1024,
        "stop": "tokenizer_eos_or_token_limit",
        "seed": 0,
    }
    if config.sampling != expected_sampling:
        raise ValueError("Qwen3.5 strict-lane sampling contract changed")

    required_engine = {
        "name": "vllm",
        "version": VLLM_VERSION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
        "max_num_seqs": 32,
        "enforce_eager": True,
        "quantization": None,
        "expected_cuda_device_name_fragment": "Ada",
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    for key, expected in required_engine.items():
        if config.engine.get(key) != expected:
            raise ValueError(f"Qwen3.5 engine contract changed: {key}")
    required_benchmark = {
        "repository": "google-deepmind/miniF2F",
        "revision": "f0a20e14c1eeccd859d51bb4c2b3ee487889c303",
        "source_path": "MiniF2F/Valid.lean",
        "primary_task_manifest": "minif2f-valid-task-ids.txt",
        "split": "validation",
        "expected_primary_task_count": 244,
        "lean_toolchain": "leanprover/lean4:v4.27.0",
    }
    for key, expected in required_benchmark.items():
        if config.benchmark.get(key) != expected:
            raise ValueError(f"Qwen3.5 benchmark contract changed: {key}")
    verifier = config.value["verifier"]
    if float(verifier["timeout_seconds"]) != 30.0:
        raise ValueError("Qwen3.5 assessment must retain verifier timeout semantics")
    if verifier.get("known_valid_task_id") != "mathd_algebra_182":
        raise ValueError("Qwen3.5 verifier control task changed")
    if verifier.get("known_valid_candidate") != "ring":
        raise ValueError("Qwen3.5 verifier control proof changed")


class GpuMemoryMonitor:
    def __init__(self, device_index: int = 0, interval_seconds: float = 0.05):
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.total_bytes = 0
        self.driver_version = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        self._handle: Any = None

    def __enter__(self) -> GpuMemoryMonitor:
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError("Qwen3.5 assessment requires nvidia-ml-py") from error
        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.baseline_bytes = int(memory.used)
        self.peak_bytes = self.baseline_bytes
        self.total_bytes = int(memory.total)
        self.driver_version = str(pynvml.nvmlSystemGetDriverVersion())
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        assert self._pynvml is not None
        while not self._stop.wait(self.interval_seconds):
            used = int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            self.peak_bytes = max(self.peak_bytes, used)

    def __exit__(self, *_: object) -> None:
        assert self._pynvml is not None
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        used = int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        self.peak_bytes = max(self.peak_bytes, used)
        self._pynvml.nvmlShutdown()

    def evidence(self) -> dict[str, Any]:
        return {
            "gpu_driver_version": self.driver_version,
            "gpu_memory_used_baseline_bytes": self.baseline_bytes,
            "peak_gpu_memory_used_bytes": self.peak_bytes,
            "gpu_memory_total_bytes": self.total_bytes,
            "gpu_memory_headroom_at_peak_bytes": self.total_bytes - self.peak_bytes,
            "gpu_memory_sampling_interval_seconds": self.interval_seconds,
        }


def run_preflight(config: Phase1Config, output_path: Path) -> dict[str, Any]:
    validate_assessment_config(config)
    runtime = _local_cuda_runtime(config)
    task = TaskRecord(
        id="qwen35-compatibility-preflight",
        preamble="import Mathlib",
        declaration="theorem qwen35_compatibility_preflight (n : Nat) : n = n",
        declaration_name="qwen35_compatibility_preflight",
    )
    sampling = {
        **config.sampling,
        "candidates_per_task": 1,
        "max_new_tokens": 32,
    }
    started = time.perf_counter()
    with GpuMemoryMonitor() as monitor:
        candidates, engine_version = _generate_candidates(
            config, [task], sampling=sampling
        )
    wall_time = time.perf_counter() - started
    candidate = candidates[0]
    if candidate.generation_error is not None:
        raise RuntimeError(candidate.generation_error)
    if candidate.token_count < 1:
        raise RuntimeError("Qwen3.5 compatibility preflight generated no tokens")

    evidence = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "complete": True,
        "model": dict(config.model),
        "prompt_format_id": PROMPT_FORMAT_ID,
        "chat_template": None,
        "prompt_transformation": None,
        "inference": {
            "engine": str(config.engine["name"]),
            "engine_version": engine_version,
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "python_version": runtime["python"],
            "torch_cuda_version": runtime["torch_cuda_version"],
            "dtype": str(config.engine["dtype"]),
            "quantization": config.engine["quantization"],
            "text_only_multimodal_limits": dict(
                config.engine["limit_mm_per_prompt"]
            ),
            "execution": runtime["inference_execution"],
        },
        "gpu": {
            "device": runtime["cuda_device"],
            "device_index": runtime["cuda_device_index"],
            "device_capability": runtime["cuda_device_capability"],
            **monitor.evidence(),
        },
        "probe": {
            "generated_token_count": candidate.token_count,
            "finish_reason": candidate.finish_reason,
            "generation_wall_time_seconds": wall_time,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def run_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    workload_id: str,
    output_dir: Path,
    *,
    timeout_seconds: float,
    verification_workers: int,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    validate_assessment_config(config)
    if workload_id not in EXPECTED_WORKLOADS:
        raise ValueError(f"unsupported Qwen3.5 assessment workload: {workload_id}")
    if timeout_seconds != float(config.value["verifier"]["timeout_seconds"]):
        raise ValueError("Qwen3.5 assessment timeout must match the pinned contract")

    with GpuMemoryMonitor() as monitor:
        metadata, results, summary = run_phase1_baseline(
            config,
            benchmark_root,
            workload_id,
            output_dir,
            timeout_seconds=timeout_seconds,
            verification_workers=verification_workers,
        )

    runtime = {
        **metadata.runtime,
        "transformers": _package_version("transformers"),
        "vllm": _package_version("vllm"),
        **monitor.evidence(),
    }
    metadata = RunMetadata(**{**metadata.to_dict(), "runtime": runtime})
    summary = {
        **summary,
        "generated_tokens": generated_token_summary(results),
        "efficiency": _efficiency_from_metadata(metadata, summary, results),
    }
    write_artifacts(output_dir, metadata, results, summary=summary)
    return metadata, results, summary


def generated_token_summary(results: Iterable[CandidateResult]) -> dict[str, Any]:
    counts = [
        int(result.generated_token_count)
        for result in results
        if result.generated_token_count is not None
    ]
    if not counts:
        return {
            "count": 0,
            "total": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(counts),
        "total": sum(counts),
        "mean": statistics.fmean(counts),
        "median": statistics.median(counts),
        "min": min(counts),
        "max": max(counts),
    }


def _efficiency_from_metadata(
    metadata: RunMetadata,
    summary: dict[str, Any],
    results: list[CandidateResult],
) -> dict[str, Any]:
    generated = generated_token_summary(results)
    generation_wall_time = float(metadata.runtime["generation_wall_time_seconds"])
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    run_wall_time = float(summary["run_wall_time_seconds"])
    return {
        "generated_tokens_per_generation_wall_second": (
            generated["total"] / generation_wall_time
            if generation_wall_time > 0.0
            else None
        ),
        "generation_wall_time_seconds_per_solved_task": (
            generation_wall_time / solved if solved else None
        ),
        "run_wall_time_seconds_per_solved_task": (
            run_wall_time / solved if solved else None
        ),
        "compute_per_solved_task_available": solved > 0,
    }


def write_compact_evidence(
    config: Phase1Config,
    preflight_path: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, dict[str, Any]]:
    validate_assessment_config(config)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_preflight(preflight, config)
    dev16 = _compact_run(config, dev16_dir, "minif2f-valid-dev16-v1")
    full = _compact_run(config, full_dir, "minif2f-valid-v1")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"preflight": preflight, "dev16": dev16, "full": full}
    for name, value in outputs.items():
        (evidence_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "README.md").write_text(
        _render_readme(dev16, full), encoding="utf-8"
    )
    return outputs


def _validate_preflight(value: dict[str, Any], config: Phase1Config) -> None:
    if value.get("schema_version") != PREFLIGHT_SCHEMA_VERSION or not value.get(
        "complete"
    ):
        raise ValueError("Qwen3.5 compatibility preflight is incomplete")
    if value.get("model") != config.model:
        raise ValueError("Qwen3.5 preflight model identity differs from config")
    if value.get("prompt_format_id") != PROMPT_FORMAT_ID:
        raise ValueError("Qwen3.5 preflight prompt format changed")
    if value.get("chat_template") is not None:
        raise ValueError("Qwen3.5 strict lane cannot use a chat template")
    if value.get("prompt_transformation") is not None:
        raise ValueError("Qwen3.5 strict lane cannot transform the raw prompt")
    inference = value["inference"]
    expected_inference = {
        "engine": "vllm",
        "engine_version": VLLM_VERSION,
        "dtype": "bfloat16",
        "quantization": None,
        "text_only_multimodal_limits": {"image": 0, "video": 0},
        "execution": "local_cuda",
    }
    for key, expected in expected_inference.items():
        if inference.get(key) != expected:
            raise ValueError(f"Qwen3.5 preflight inference changed: {key}")
    if "Ada" not in str(value["gpu"].get("device")):
        raise ValueError("Qwen3.5 preflight did not use the project Ada GPU")
    if value["gpu"].get("peak_gpu_memory_used_bytes", 0) <= 0:
        raise ValueError("Qwen3.5 preflight lacks peak GPU memory")
    if value["gpu"].get("gpu_memory_headroom_at_peak_bytes", 0) <= 0:
        raise ValueError("Qwen3.5 preflight exceeded physical GPU memory")


def _compact_run(
    config: Phase1Config, artifact_dir: Path, workload_id: str
) -> dict[str, Any]:
    metadata, results = read_artifacts(artifact_dir)
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    expected_tasks, expected_candidates = EXPECTED_WORKLOADS[workload_id]
    if metadata.workload_id != workload_id:
        raise ValueError(f"unexpected workload in {artifact_dir}")
    if metadata.candidate_source != "model":
        raise ValueError(f"non-model candidates in {artifact_dir}")
    if metadata.model_id != MODEL_ID or metadata.model_revision != MODEL_REVISION:
        raise ValueError(f"Qwen3.5 model identity changed in {artifact_dir}")
    if metadata.tokenizer_id != MODEL_ID or metadata.tokenizer_revision != MODEL_REVISION:
        raise ValueError(f"Qwen3.5 tokenizer identity changed in {artifact_dir}")
    if metadata.prompt_format_id != PROMPT_FORMAT_ID:
        raise ValueError(f"prompt format changed in {artifact_dir}")
    if metadata.benchmark_repository != config.benchmark["repository"]:
        raise ValueError(f"benchmark repository changed in {artifact_dir}")
    if metadata.benchmark_revision != config.benchmark["revision"]:
        raise ValueError(f"benchmark revision changed in {artifact_dir}")
    if metadata.benchmark_split != config.benchmark["split"]:
        raise ValueError(f"benchmark split changed in {artifact_dir}")
    if metadata.verifier_timeout_seconds != float(
        config.value["verifier"]["timeout_seconds"]
    ):
        raise ValueError(f"verifier timeout changed in {artifact_dir}")
    if metadata.generation_settings is None:
        raise ValueError(f"missing generation settings in {artifact_dir}")
    for key, expected in config.sampling.items():
        if metadata.generation_settings.get(key) != expected:
            raise ValueError(f"sampling contract changed in {artifact_dir}: {key}")
    if metadata.generation_settings.get("chat_template") is not None:
        raise ValueError(f"chat template used in strict lane: {artifact_dir}")
    if metadata.generation_settings.get("prompt_transformation") is not None:
        raise ValueError(f"raw prompt was transformed in {artifact_dir}")
    if metadata.generation_settings.get("limit_mm_per_prompt") != {
        "image": 0,
        "video": 0,
    }:
        raise ValueError(f"Qwen3.5 text-only engine limits changed: {artifact_dir}")
    if metadata.inference_engine_version != VLLM_VERSION:
        raise ValueError(f"vLLM version changed in {artifact_dir}")
    if metadata.inference_engine != "vllm":
        raise ValueError(f"inference engine changed in {artifact_dir}")
    recorded_engine = {
        "dtype": metadata.generation_settings.get("dtype"),
        "tensor_parallel_size": metadata.generation_settings.get(
            "tensor_parallel_size"
        ),
        "gpu_memory_utilization": metadata.generation_settings.get(
            "gpu_memory_utilization"
        ),
        "max_model_len": metadata.generation_settings.get("max_model_len"),
        "max_num_seqs": metadata.generation_settings.get("max_num_seqs"),
        "enforce_eager": metadata.generation_settings.get("enforce_eager"),
        "quantization": metadata.generation_settings.get("quantization"),
    }
    for key, observed in recorded_engine.items():
        if observed != config.engine[key]:
            raise ValueError(f"engine setting changed in {artifact_dir}: {key}")
    if metadata.runtime.get("inference_execution") != "local_cuda":
        raise ValueError(f"inference was not local CUDA in {artifact_dir}")
    if "Ada" not in str(metadata.runtime.get("cuda_device")):
        raise ValueError(f"inference did not use the project Ada GPU: {artifact_dir}")
    if int(metadata.runtime.get("peak_gpu_memory_used_bytes", 0)) <= 0:
        raise ValueError(f"missing peak GPU memory in {artifact_dir}")
    if int(metadata.runtime.get("gpu_memory_headroom_at_peak_bytes", 0)) <= 0:
        raise ValueError(f"GPU memory preflight failed in {artifact_dir}")
    if summary.get("task_count") != expected_tasks:
        raise ValueError(f"task denominator changed in {artifact_dir}")
    if (
        summary.get("candidate_count") != expected_candidates
        or len(results) != expected_candidates
    ):
        raise ValueError(f"candidate denominator changed in {artifact_dir}")
    if summary.get("infrastructure_error_count") != 0:
        raise ValueError(f"unresolved infrastructure errors in {artifact_dir}")
    if not summary.get("complete"):
        raise ValueError(f"incomplete Qwen3.5 assessment: {artifact_dir}")
    if set(summary.get("pass_at_k", {})) != {"pass@1", "pass@4"}:
        raise ValueError(f"Qwen3.5 assessment must report only pass@1/pass@4: {artifact_dir}")

    token_summary = generated_token_summary(results)
    if token_summary["count"] != expected_candidates:
        raise ValueError(f"missing generated-token counts in {artifact_dir}")
    if sum(summary["category_counts"].values()) != expected_candidates:
        raise ValueError(f"category counts do not cover all candidates: {artifact_dir}")
    if sum(summary["finish_reason_counts"].values()) != expected_candidates:
        raise ValueError(f"finish reasons do not cover all candidates: {artifact_dir}")
    if not set(summary["finish_reason_counts"]).issubset({"eos", "token_limit"}):
        raise ValueError(f"unexpected finish reason in {artifact_dir}")

    efficiency = _efficiency_from_metadata(metadata, summary, results)
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "complete": True,
        "workload_id": workload_id,
        "contract": {
            "model": {
                "id": metadata.model_id,
                "revision": metadata.model_revision,
            },
            "tokenizer": {
                "id": metadata.tokenizer_id,
                "revision": metadata.tokenizer_revision,
            },
            "benchmark": {
                "repository": metadata.benchmark_repository,
                "revision": metadata.benchmark_revision,
                "split": metadata.benchmark_split,
            },
            "prompt_format_id": metadata.prompt_format_id,
            "sampling": dict(config.sampling),
            "model_license": config.value["assessment"]["model_license"],
            "verifier_timeout_seconds": metadata.verifier_timeout_seconds,
            "verifier_timeout_semantics": "unsuccessful_proof_attempt",
            "verifier_timeout_is_infrastructure_error": False,
        },
        "runtime": {
            "inference_engine": metadata.inference_engine,
            "inference_engine_version": metadata.inference_engine_version,
            "python": metadata.runtime.get("python"),
            "torch": metadata.runtime.get("torch"),
            "transformers": metadata.runtime.get("transformers"),
            "torch_cuda_version": metadata.runtime.get("torch_cuda_version"),
            "cuda_device": metadata.runtime.get("cuda_device"),
            "cuda_device_capability": metadata.runtime.get("cuda_device_capability"),
            "gpu_driver_version": metadata.runtime.get("gpu_driver_version"),
            "gpu_memory_total_bytes": metadata.runtime.get("gpu_memory_total_bytes"),
            "peak_gpu_memory_used_bytes": metadata.runtime.get(
                "peak_gpu_memory_used_bytes"
            ),
            "gpu_memory_headroom_at_peak_bytes": metadata.runtime.get(
                "gpu_memory_headroom_at_peak_bytes"
            ),
            "dtype": metadata.generation_settings["dtype"],
            "quantization": metadata.generation_settings["quantization"],
            "inference_execution": metadata.runtime.get("inference_execution"),
        },
        "results": {
            "task_count": summary["task_count"],
            "candidate_count": summary["candidate_count"],
            "candidates_per_task": summary["candidates_per_task"],
            "tasks_with_verified_candidate": summary[
                "tasks_with_verified_candidate"
            ],
            "pass_at_k": summary["pass_at_k"],
            "category_counts": summary["category_counts"],
            "finish_reason_counts": summary["finish_reason_counts"],
            "verifier_timeout_count": summary["verifier_timeout_count"],
            "infrastructure_error_count": summary["infrastructure_error_count"],
            "generated_tokens": token_summary,
        },
        "timing": {
            "generation_wall_time_seconds": metadata.runtime.get(
                "generation_wall_time_seconds"
            ),
            "verification_wall_time_seconds": metadata.runtime.get(
                "verification_wall_time_seconds"
            ),
            "run_wall_time_seconds": summary["run_wall_time_seconds"],
            "candidate_latency_seconds": summary["timing_seconds"],
            **efficiency,
        },
        "verifier_environment": metadata.verifier_environment,
        "candidate_generation_projection_sha256": _candidate_projection_sha256(
            results
        ),
    }


def _candidate_projection_sha256(results: list[CandidateResult]) -> str:
    projection = [
        {
            "task_id": result.task_id,
            "candidate_id": result.candidate_id,
            "candidate_index": result.candidate_index,
            "candidate_text": result.candidate_text,
            "generated_token_count": result.generated_token_count,
            "finish_reason": result.finish_reason,
        }
        for result in results
    ]
    serialized = json.dumps(
        projection, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _render_readme(dev16: dict[str, Any], full: dict[str, Any]) -> str:
    def row(label: str, value: dict[str, Any]) -> str:
        results = value["results"]
        return (
            f"| {label} | {results['task_count']} | {results['candidate_count']} | "
            f"{results['tasks_with_verified_candidate']['count']} | "
            f"{results['pass_at_k']['pass@1']:.6f} | "
            f"{results['pass_at_k']['pass@4']:.6f} | "
            f"{results['infrastructure_error_count']} | "
            f"{results['verifier_timeout_count']} |"
        )

    full_results = full["results"]
    full_timing = full["timing"]
    full_runtime = full["runtime"]
    compute = (
        "unavailable because no task was solved"
        if not full_timing["compute_per_solved_task_available"]
        else (
            f"{full_timing['generation_wall_time_seconds_per_solved_task']:.3f} "
            "generation seconds per solved task"
        )
    )
    return f"""# Qwen3.5-2B-Base whole-proof assessment

**OBSERVED:** the official `Qwen/Qwen3.5-2B-Base` foundation was evaluated independently under the unchanged `whole-proof-v1` raw-continuation and Lean-verification contract. This is model-assessment evidence, not a training or automatic-promotion result.

| Workload | Tasks | Candidates | Solved tasks | pass@1 | pass@4 | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{row('dev16 gate', dev16)}
{row('full validation', full)}

**ACCEPTED:** generation used temperature 0.8, top-p 0.95, no top-k, a 1,024 generated-token cap, seed 0, and four candidates per task. No chat template, extraction, repair, Lean feedback, candidate regeneration, or hosted inference was used. `verifier_timeout` remains an unsuccessful proof outcome rather than an infrastructure error.

**OBSERVED:** the full run generated {full_results['generated_tokens']['total']} tokens; finish reasons were `{json.dumps(full_results['finish_reason_counts'], sort_keys=True)}` and evaluator categories were `{json.dumps(full_results['category_counts'], sort_keys=True)}`. Generation took {full_timing['generation_wall_time_seconds']:.3f} seconds at {full_timing['generated_tokens_per_generation_wall_second']:.3f} generated tokens/second; end-to-end run time was {full_timing['run_wall_time_seconds']:.3f} seconds. Compute per solved task was {compute}.

**OBSERVED:** inference executed locally in BF16 with vLLM {full_runtime['inference_engine_version']} on `{full_runtime['cuda_device']}`. Peak observed GPU memory was {full_runtime['peak_gpu_memory_used_bytes']} of {full_runtime['gpu_memory_total_bytes']} bytes. The Apache-2.0 model and tokenizer were pinned to `{MODEL_REVISION}`; raw candidates, weights, caches, and bulky logs remain outside Git.
"""


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)
