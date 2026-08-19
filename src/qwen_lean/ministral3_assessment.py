from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import read_artifacts, write_artifacts
from .baseline import GeneratedCandidate, validate_minif2f_environment
from .metrics import summarize_results
from .minif2f import PHASE1_CONFIG_SCHEMA_VERSION, Phase1Config, materialize_benchmark_tasks
from .prompt import PROMPT_FORMAT_ID, render_prompt
from .schema import PHASE1_RESULT_SCHEMA_VERSION, CandidateResult, RunMetadata, TaskRecord
from .verifier import LeanVerifier


CONFIG_SCHEMA_VERSION = "ministral3-8b-base-assessment-config-v1"
PREFLIGHT_SCHEMA_VERSION = "ministral3-8b-base-preflight-v1"
EVIDENCE_SCHEMA_VERSION = "ministral3-8b-base-evidence-v1"
MODEL_ID = "mistralai/Ministral-3-8B-Base-2512"
MODEL_REVISION = "d4883f9b36aa2e5d775730d3fdba3d30de51a8ef"
WORKLOADS = ("minif2f-valid-dev16-v1", "minif2f-valid-v1")
ENVIRONMENT_PROBE_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class Ministral3AssessmentConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Ministral3AssessmentConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        config = cls(path=path.resolve(), value=value)
        config.validate()
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.value["sampling"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.value["runtime"]

    @property
    def verifier(self) -> dict[str, Any]:
        return self.value["verifier"]

    def lane(self, lane_id: str) -> dict[str, Any]:
        for key in ("bf16_lane", "fallback_lane"):
            lane = self.value[key]
            if lane["lane_id"] == lane_id:
                return lane
        raise ValueError(f"unknown assessment lane: {lane_id}")

    def phase1_config(self) -> Phase1Config:
        return Phase1Config(
            path=self.path,
            value={
                "schema_version": PHASE1_CONFIG_SCHEMA_VERSION,
                "benchmark": self.value["benchmark"],
                "workloads": self.value["workloads"],
                "model": self.model,
                "sampling": self.sampling,
                "engine": self.value["bf16_lane"],
                "verifier": self.verifier,
            },
        )

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def validate(self) -> None:
        if self.value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("unknown Ministral 3 assessment config schema")
        expected_model = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
        }
        if self.model != expected_model:
            raise ValueError("model/tokenizer identity differs from issue #52")
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
        if self.sampling != expected_sampling:
            raise ValueError("sampling differs from the strict issue #52 contract")
        if set(self.value["workloads"]) != set(WORKLOADS):
            raise ValueError("assessment workloads differ from issue #52")
        benchmark = self.value["benchmark"]
        if (
            benchmark["split"] != "validation"
            or int(benchmark["expected_primary_task_count"]) != 244
        ):
            raise ValueError("assessment must use all 244 miniF2F validation tasks")
        if float(self.verifier["timeout_seconds"]) != 30.0:
            raise ValueError("verifier timeout must remain 30 seconds")
        if int(self.verifier["verification_workers"]) < 1:
            raise ValueError("verification worker count must be positive")
        runtime = self.runtime
        expected_runtime = {
            "inference_engine": "vllm",
            "inference_engine_version": "0.23.0",
            "torch_version": "2.11.0+cu130",
            "transformers_version": "5.15.0",
            "bitsandbytes_version": "0.49.1",
            "mistral_common_version": "1.11.7",
            "cuda_toolkit_source": "isolated-python-runtime",
            "cuda_linker_layout": "python-wheel-lib64-compat-v1",
            "expected_cuda_device_name": "NVIDIA RTX 4000 Ada Generation",
            "vllm_enable_v1_multiprocessing": False,
            "vllm_worker_multiproc_method": "spawn",
        }
        if runtime != expected_runtime:
            raise ValueError("runtime differs from the frozen local Ada stack")
        self._validate_lane(self.value["bf16_lane"], fallback=False)
        self._validate_lane(self.value["fallback_lane"], fallback=True)

    @staticmethod
    def _validate_lane(lane: dict[str, Any], *, fallback: bool) -> None:
        required = {
            "dtype": "bfloat16",
            "language_model_only": True,
            "cpu_offload_gb": 0,
            "tensor_parallel_size": 1,
            "max_model_len": 2048,
            "max_num_seqs": 16,
            "enforce_eager": True,
            "tokenizer_mode": "mistral",
        }
        if any(lane.get(key) != value for key, value in required.items()):
            raise ValueError(f"invalid frozen lane: {lane.get('lane_id')}")
        if fallback:
            if (
                lane.get("lane_id") != "bitsandbytes-nf4-online-v1"
                or lane.get("quantization") != "bitsandbytes"
                or lane.get("load_format") != "bitsandbytes"
            ):
                raise ValueError("fallback must be the frozen bitsandbytes NF4 lane")
            expected = {
                "bits": 4,
                "quant_type": "nf4",
                "double_quantization": True,
                "compute_dtype": "bfloat16",
                "conversion": "vllm online from pinned BF16 safetensors",
                "prequantized_checkpoint": False,
            }
            if lane.get("quantization_metadata") != expected:
                raise ValueError("fallback quantization metadata differs from frozen NF4")
        elif lane.get("quantization") is not None or lane.get("load_format") != "auto":
            raise ValueError("BF16 preflight lane must be unquantized")


class _DeviceMemorySampler:
    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        self.peak_used_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _DeviceMemorySampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            self._sample()

    def _sample(self) -> None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.device_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            return
        try:
            value = int(completed.stdout.strip())
        except ValueError:
            return
        self.peak_used_mib = value if self.peak_used_mib is None else max(
            self.peak_used_mib, value
        )


def validate_model_snapshot(config: Ministral3AssessmentConfig, snapshot: Path) -> Path:
    resolved = snapshot.resolve()
    if resolved.name != MODEL_REVISION:
        raise ValueError(
            f"model snapshot must resolve to pinned revision {MODEL_REVISION}: {resolved}"
        )
    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (resolved / name).is_file())
    if missing:
        raise ValueError(f"model snapshot is incomplete: {missing}")
    index = json.loads(
        (resolved / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (resolved / name).is_file()]
    if missing_shards:
        raise ValueError(f"model snapshot is missing weight shards: {missing_shards}")
    return resolved


def vllm_engine_kwargs(
    config: Ministral3AssessmentConfig,
    snapshot: Path,
    lane: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": str(snapshot),
        "tokenizer": str(snapshot),
        "runner": "generate",
        "tokenizer_mode": str(lane["tokenizer_mode"]),
        "trust_remote_code": False,
        "dtype": str(lane["dtype"]),
        "quantization": lane["quantization"],
        "load_format": str(lane["load_format"]),
        "language_model_only": bool(lane["language_model_only"]),
        "cpu_offload_gb": float(lane["cpu_offload_gb"]),
        "tensor_parallel_size": int(lane["tensor_parallel_size"]),
        "gpu_memory_utilization": float(lane["gpu_memory_utilization"]),
        "max_model_len": int(lane["max_model_len"]),
        "max_num_seqs": int(lane["max_num_seqs"]),
        "enforce_eager": bool(lane["enforce_eager"]),
        "seed": int(config.sampling["seed"]),
        "generation_config": "vllm",
        "mm_processor_cache_gb": 0,
        "disable_log_stats": True,
    }


def vllm_sampling_kwargs(sampling: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": int(sampling["candidates_per_task"]),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": int(sampling["top_k"]),
        "max_tokens": int(sampling["max_new_tokens"]),
        "seed": int(sampling["seed"]),
        "ignore_eos": False,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
    }


def run_preflight(
    config: Ministral3AssessmentConfig,
    benchmark_root: Path,
    model_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    _configure_vllm_environment(config)
    snapshot = validate_model_snapshot(config, model_snapshot)
    phase1 = config.phase1_config()
    environment = validate_minif2f_environment(
        phase1,
        benchmark_root,
        timeout_seconds=ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    )
    tasks = phase1.select_workload(
        "minif2f-valid-dev16-v1",
        materialize_benchmark_tasks(phase1, benchmark_root),
    )
    runtime = _local_runtime(config)
    bf16 = _generation_attempt(
        config, snapshot, config.value["bf16_lane"], [render_prompt(tasks[0])]
    )
    fallback: dict[str, Any] | None = None
    accepted_lane: str | None = None
    status = "failed"
    if bf16["status"] == "passed":
        accepted_lane = str(config.value["bf16_lane"]["lane_id"])
        status = "passed"
    elif bf16["memory_failure"]:
        fallback = _generation_attempt(
            config, snapshot, config.value["fallback_lane"], [render_prompt(tasks[0])]
        )
        if fallback["status"] == "passed":
            accepted_lane = str(config.value["fallback_lane"]["lane_id"])
            status = "passed"
    evidence = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "model": config.model,
        "model_snapshot": str(snapshot),
        "prompt_format_id": PROMPT_FORMAT_ID,
        "chat_template": None,
        "prompt_transformation": None,
        "benchmark_environment": environment,
        "runtime": runtime,
        "bf16_attempt": bf16,
        "fallback_attempt": fallback,
        "accepted_lane": accepted_lane,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def _generation_attempt(
    config: Ministral3AssessmentConfig,
    snapshot: Path,
    lane: Mapping[str, Any],
    prompts: list[str],
) -> dict[str, Any]:
    import torch

    _configure_vllm_environment(config)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    llm: Any | None = None
    outputs: list[Any] = []
    load_seconds: float | None = None
    generation_seconds: float | None = None
    error: BaseException | None = None
    sampler = _DeviceMemorySampler(torch.cuda.current_device())
    try:
        import vllm
        from vllm import LLM, SamplingParams

        _validate_runtime_versions(config, vllm)
        with sampler:
            load_started = time.perf_counter()
            llm = LLM(**vllm_engine_kwargs(config, snapshot, lane))
            load_seconds = time.perf_counter() - load_started
            generation_started = time.perf_counter()
            outputs = llm.generate(
                prompts,
                SamplingParams(**vllm_sampling_kwargs(config.sampling)),
                use_tqdm=True,
            )
            generation_seconds = time.perf_counter() - generation_started
    except BaseException as caught:
        error = caught
    finally:
        del llm
        gc.collect()
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache()

    message = None if error is None else f"{type(error).__name__}: {error}"
    memory_failure = False if message is None else _is_memory_failure(error, message)
    finish_reasons: Counter[str] = Counter()
    token_counts: list[int] = []
    candidate_digests: list[str] = []
    for request in outputs:
        for item in request.outputs:
            finish_reasons[_finish_reason(item.finish_reason)] += 1
            token_counts.append(len(item.token_ids))
            candidate_digests.append(hashlib.sha256(item.text.encode()).hexdigest())
    expected = len(prompts) * int(config.sampling["candidates_per_task"])
    passed = error is None and len(token_counts) == expected
    return {
        "lane": dict(lane),
        "status": "passed" if passed else "failed",
        "memory_failure": memory_failure,
        "error": message,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "prompt_count": len(prompts),
        "expected_candidate_count": expected,
        "candidate_count": len(token_counts),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "generated_token_count": sum(token_counts),
        "candidate_sha256": candidate_digests,
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "peak_device_memory_used_mib": sampler.peak_used_mib,
    }


def _is_memory_failure(error: BaseException | None, message: str) -> bool:
    try:
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            return True
    except ImportError:
        pass
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda oom",
            "free memory on device",
            "no available memory for the cache blocks",
            "memory profiling",
        )
    )


def run_assessment(
    config: Ministral3AssessmentConfig,
    benchmark_root: Path,
    model_snapshot: Path,
    preflight_path: Path,
    workload_id: str,
    output_dir: Path,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    _configure_vllm_environment(config)
    if workload_id not in WORKLOADS:
        raise ValueError(f"unknown assessment workload: {workload_id}")
    snapshot = validate_model_snapshot(config, model_snapshot)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_preflight(config, preflight, snapshot)
    lane = config.lane(str(preflight["accepted_lane"]))
    phase1 = config.phase1_config()
    tasks = phase1.select_workload(
        workload_id, materialize_benchmark_tasks(phase1, benchmark_root)
    )
    environment_validation = validate_minif2f_environment(
        phase1,
        benchmark_root,
        timeout_seconds=ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    )
    runtime = _local_runtime(config)
    generated, engine_metrics = _generate_candidates(config, snapshot, lane, tasks)
    runtime.update(engine_metrics)

    verifier = LeanVerifier(
        benchmark_root,
        timeout_seconds=float(config.verifier["timeout_seconds"]),
    )
    post_generation_probe_started = time.perf_counter()
    probe_failure = verifier.prime_preamble(
        tasks[0].preamble,
        timeout_seconds=ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    )
    post_generation_probe_seconds = time.perf_counter() - post_generation_probe_started
    if probe_failure is not None:
        diagnostics = (
            probe_failure.diagnostics["stdout"] + probe_failure.diagnostics["stderr"]
        )
        raise RuntimeError(
            "post-generation verifier environment probe failed as "
            f"{probe_failure.category}: {diagnostics}"
        )
    runtime["post_generation_environment_probe_timeout_seconds"] = (
        ENVIRONMENT_PROBE_TIMEOUT_SECONDS
    )
    runtime["post_generation_environment_probe_wall_time_seconds"] = (
        post_generation_probe_seconds
    )
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=int(config.verifier["verification_workers"])
    ) as executor:
        results = list(
            executor.map(lambda item: _verify_candidate(verifier, item), generated)
        )
    verification_seconds = time.perf_counter() - verification_started
    runtime["verification_wall_time_seconds"] = verification_seconds
    runtime["verification_workers"] = int(config.verifier["verification_workers"])

    summary = summarize_results(
        results,
        expected_task_ids=[task.id for task in tasks],
        candidates_per_task=int(config.sampling["candidates_per_task"]),
    )
    summary["workload_id"] = workload_id
    summary["generated_tokens"] = _token_statistics(results)
    summary["generation_wall_time_seconds"] = engine_metrics[
        "generation_wall_time_seconds"
    ]
    summary["engine_load_time_seconds"] = engine_metrics["engine_load_time_seconds"]
    summary["run_wall_time_seconds"] = (
        float(engine_metrics["engine_load_time_seconds"])
        + float(engine_metrics["generation_wall_time_seconds"])
        + post_generation_probe_seconds
        + verification_seconds
    )
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    generated_tokens = int(summary["generated_tokens"]["total"])
    generation_seconds = float(engine_metrics["generation_wall_time_seconds"])
    summary["throughput"] = {
        "generated_tokens_per_second": (
            generated_tokens / generation_seconds if generation_seconds else None
        ),
        "candidates_per_second": (
            len(results) / generation_seconds if generation_seconds else None
        ),
    }
    summary["compute_per_solved_task"] = {
        "generated_tokens": generated_tokens / solved if solved else None,
        "generation_gpu_seconds": generation_seconds / solved if solved else None,
    }

    environment = environment_validation["verifier_environment"]
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source=(
            f"{phase1.benchmark['repository']}@{phase1.benchmark['revision']}:"
            f"{phase1.benchmark['source_path']}"
        ),
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=str(environment["lean_toolchain"]),
        mathlib_revision=str(environment["dependencies"]["mathlib"]),
        verifier_timeout_seconds=float(config.verifier["timeout_seconds"]),
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=workload_id,
        benchmark_split="validation",
        benchmark_repository=str(phase1.benchmark["repository"]),
        benchmark_revision=str(phase1.benchmark["revision"]),
        verifier_environment=environment,
        candidates_per_task=int(config.sampling["candidates_per_task"]),
        inference_engine=str(config.runtime["inference_engine"]),
        inference_engine_version=str(config.runtime["inference_engine_version"]),
        generation_settings={
            **config.sampling,
            "lane_id": lane["lane_id"],
            "dtype": lane["dtype"],
            "quantization": lane["quantization"],
            "quantization_metadata": lane.get("quantization_metadata"),
            "load_format": lane["load_format"],
            "language_model_only": lane["language_model_only"],
            "cpu_offload_gb": lane["cpu_offload_gb"],
            "tensor_parallel_size": lane["tensor_parallel_size"],
            "gpu_memory_utilization": lane["gpu_memory_utilization"],
            "max_model_len": lane["max_model_len"],
            "max_num_seqs": lane["max_num_seqs"],
            "enforce_eager": lane["enforce_eager"],
            "tokenizer_mode": lane["tokenizer_mode"],
            "chat_template": None,
            "prompt_transformation": None,
            "proof_extraction": None,
            "lean_feedback": None,
            "retry": None,
        },
        runtime={
            **runtime,
            "assessment_id": config.value["assessment_id"],
            "config_sha256": config.digest(),
            "accepted_preflight": str(preflight_path.resolve()),
            "model_snapshot": str(snapshot),
            "inference_execution": "local_cuda",
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
    return metadata, results, summary


def reverify_assessment(
    config: Ministral3AssessmentConfig,
    benchmark_root: Path,
    input_dir: Path,
    output_dir: Path,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    metadata, original_results = read_artifacts(input_dir)
    if metadata.model_id != MODEL_ID or metadata.model_revision != MODEL_REVISION:
        raise ValueError("reverification model identity differs from issue #52")
    if metadata.tokenizer_id != MODEL_ID or metadata.tokenizer_revision != MODEL_REVISION:
        raise ValueError("reverification tokenizer identity differs from issue #52")
    if metadata.prompt_format_id != PROMPT_FORMAT_ID:
        raise ValueError("reverification prompt format differs from whole-proof-v1")
    if metadata.runtime.get("config_sha256") != config.digest():
        raise ValueError("reverification config digest differs from current config")
    workload_id = str(metadata.workload_id)
    if workload_id not in WORKLOADS:
        raise ValueError(f"unknown assessment workload: {workload_id}")
    phase1 = config.phase1_config()
    tasks = phase1.select_workload(
        workload_id, materialize_benchmark_tasks(phase1, benchmark_root)
    )
    tasks_by_id = {task.id: task for task in tasks}
    validate_minif2f_environment(
        phase1,
        benchmark_root,
        timeout_seconds=ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    )
    generated = [
        GeneratedCandidate(
            task=tasks_by_id[result.task_id],
            candidate_index=result.candidate_index,
            text=result.candidate_text,
            token_count=int(result.generated_token_count or 0),
            finish_reason=str(result.finish_reason or "unknown"),
            generation_latency_seconds=float(result.generation_latency_seconds or 0.0),
        )
        for result in original_results
    ]
    verifier = LeanVerifier(
        benchmark_root,
        timeout_seconds=float(config.verifier["timeout_seconds"]),
    )
    probe_started = time.perf_counter()
    probe_failure = verifier.prime_preamble(
        tasks[0].preamble,
        timeout_seconds=ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    )
    probe_seconds = time.perf_counter() - probe_started
    if probe_failure is not None:
        diagnostics = (
            probe_failure.diagnostics["stdout"] + probe_failure.diagnostics["stderr"]
        )
        raise RuntimeError(
            "reverification environment probe failed as "
            f"{probe_failure.category}: {diagnostics}"
        )
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=int(config.verifier["verification_workers"])
    ) as executor:
        results = list(
            executor.map(lambda item: _verify_candidate(verifier, item), generated)
        )
    verification_seconds = time.perf_counter() - verification_started
    runtime = dict(metadata.runtime)
    runtime.update(
        {
            "candidate_generation_reused": True,
            "previous_verification_wall_time_seconds": runtime.get(
                "verification_wall_time_seconds"
            ),
            "post_generation_environment_probe_timeout_seconds": (
                ENVIRONMENT_PROBE_TIMEOUT_SECONDS
            ),
            "post_generation_environment_probe_wall_time_seconds": probe_seconds,
            "verification_wall_time_seconds": verification_seconds,
            "verification_workers": int(config.verifier["verification_workers"]),
        }
    )
    updated_metadata = RunMetadata(**{**metadata.to_dict(), "runtime": runtime})
    summary = summarize_results(
        results,
        expected_task_ids=[task.id for task in tasks],
        candidates_per_task=int(config.sampling["candidates_per_task"]),
    )
    summary["workload_id"] = workload_id
    summary["generated_tokens"] = _token_statistics(results)
    generation_seconds = float(runtime["generation_wall_time_seconds"])
    summary["generation_wall_time_seconds"] = generation_seconds
    summary["engine_load_time_seconds"] = float(runtime["engine_load_time_seconds"])
    summary["run_wall_time_seconds"] = generation_seconds + probe_seconds + verification_seconds
    generated_tokens = int(summary["generated_tokens"]["total"])
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    summary["throughput"] = {
        "generated_tokens_per_second": generated_tokens / generation_seconds,
        "candidates_per_second": len(results) / generation_seconds,
    }
    summary["compute_per_solved_task"] = {
        "generated_tokens": generated_tokens / solved if solved else None,
        "generation_gpu_seconds": generation_seconds / solved if solved else None,
    }
    write_artifacts(output_dir, updated_metadata, results, summary=summary)
    return updated_metadata, results, summary


def _validate_preflight(
    config: Ministral3AssessmentConfig,
    preflight: dict[str, Any],
    snapshot: Path,
) -> None:
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unknown or missing Ministral 3 preflight")
    if preflight.get("status") != "passed":
        raise ValueError("accepted generation requires a passing preflight")
    if preflight.get("config_sha256") != config.digest():
        raise ValueError("preflight config digest differs from current config")
    if preflight.get("model") != config.model:
        raise ValueError("preflight model identity differs from current config")
    if Path(str(preflight.get("model_snapshot"))).resolve() != snapshot:
        raise ValueError("preflight model snapshot differs from requested snapshot")
    lane = config.lane(str(preflight.get("accepted_lane")))
    if lane["lane_id"] == config.value["fallback_lane"]["lane_id"]:
        if not preflight["bf16_attempt"].get("memory_failure"):
            raise ValueError("4-bit fallback requires a recorded BF16 memory failure")
        if preflight.get("fallback_attempt", {}).get("status") != "passed":
            raise ValueError("4-bit fallback requires a passing compatibility attempt")
    elif preflight["bf16_attempt"].get("status") != "passed":
        raise ValueError("BF16 lane requires a passing BF16 attempt")


def _generate_candidates(
    config: Ministral3AssessmentConfig,
    snapshot: Path,
    lane: Mapping[str, Any],
    tasks: list[TaskRecord],
) -> tuple[list[GeneratedCandidate], dict[str, Any]]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    _configure_vllm_environment(config)
    _validate_runtime_versions(config, vllm)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler = _DeviceMemorySampler(torch.cuda.current_device())
    llm: Any | None = None
    try:
        with sampler:
            load_started = time.perf_counter()
            llm = LLM(**vllm_engine_kwargs(config, snapshot, lane))
            load_seconds = time.perf_counter() - load_started
            prompts = [render_prompt(task) for task in tasks]
            generation_started = time.perf_counter()
            outputs = llm.generate(
                prompts,
                SamplingParams(**vllm_sampling_kwargs(config.sampling)),
                use_tqdm=True,
            )
            generation_seconds = time.perf_counter() - generation_started
        generated = _convert_outputs(
            tasks,
            prompts,
            outputs,
            generation_seconds,
            candidates_per_task=int(config.sampling["candidates_per_task"]),
        )
        metrics = {
            "engine_load_time_seconds": load_seconds,
            "generation_wall_time_seconds": generation_seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_device_memory_used_mib": sampler.peak_used_mib,
        }
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return generated, metrics


def _convert_outputs(
    tasks: list[TaskRecord],
    prompts: list[str],
    outputs: list[Any],
    generation_seconds: float,
    *,
    candidates_per_task: int,
) -> list[GeneratedCandidate]:
    if len(outputs) != len(tasks):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} requests for {len(tasks)} tasks"
        )
    fallback_latency = generation_seconds / len(tasks) if tasks else generation_seconds
    generated: list[GeneratedCandidate] = []
    for task, prompt, request in zip(tasks, prompts, outputs, strict=True):
        completions = sorted(request.outputs, key=lambda item: item.index)
        indices = [item.index for item in completions]
        if request.prompt != prompt or indices != list(range(candidates_per_task)):
            raise RuntimeError(
                f"invalid vLLM output for {task.id}: "
                f"prompt_match={request.prompt == prompt}, indices={indices}"
            )
        metrics = request.metrics
        latency = fallback_latency
        if (
            metrics is not None
            and metrics.finished_time is not None
            and metrics.finished_time >= metrics.arrival_time
        ):
            latency = metrics.finished_time - metrics.arrival_time
        generated.extend(
            GeneratedCandidate(
                task=task,
                candidate_index=item.index,
                text=item.text,
                token_count=len(item.token_ids),
                finish_reason=_finish_reason(item.finish_reason),
                generation_latency_seconds=latency,
            )
            for item in completions
        )
    return generated


def _finish_reason(value: str | None) -> str:
    if value == "stop":
        return "eos"
    if value == "length":
        return "token_limit"
    return "unknown" if value is None else value


def _verify_candidate(
    verifier: LeanVerifier, generated: GeneratedCandidate
) -> CandidateResult:
    try:
        outcome = verifier.verify(generated.task, generated.text)
        return CandidateResult(
            task_id=generated.task.id,
            candidate_id=f"model-{generated.candidate_index}",
            candidate_index=generated.candidate_index,
            candidate_text=generated.text,
            category=outcome.category,
            lean_exit_code=outcome.lean_exit_code,
            diagnostics=outcome.diagnostics,
            generation_latency_seconds=generated.generation_latency_seconds,
            verification_latency_seconds=outcome.latency_seconds,
            total_latency_seconds=(
                generated.generation_latency_seconds + outcome.latency_seconds
            ),
            generated_token_count=generated.token_count,
            finish_reason=generated.finish_reason,
        )
    except Exception as error:
        return CandidateResult(
            task_id=generated.task.id,
            candidate_id=f"model-{generated.candidate_index}",
            candidate_index=generated.candidate_index,
            candidate_text=generated.text,
            category="verifier_error",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": f"{type(error).__name__}: {error}"},
            generation_latency_seconds=generated.generation_latency_seconds,
            verification_latency_seconds=None,
            total_latency_seconds=generated.generation_latency_seconds,
            generated_token_count=generated.token_count,
            finish_reason=generated.finish_reason,
        )


def write_compact_evidence(
    config: Ministral3AssessmentConfig,
    preflight_path: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    snapshot = Path(str(preflight.get("model_snapshot"))).resolve()
    _validate_preflight(config, preflight, snapshot)
    dev16 = _compact_run(
        config,
        dev16_dir,
        expected_workload="minif2f-valid-dev16-v1",
        expected_tasks=16,
    )
    full = _compact_run(
        config,
        full_dir,
        expected_workload="minif2f-valid-v1",
        expected_tasks=244,
    )
    if dev16["lane_id"] != preflight["accepted_lane"]:
        raise ValueError("dev16 lane differs from the accepted preflight")
    if full["lane_id"] != preflight["accepted_lane"]:
        raise ValueError("full lane differs from the accepted preflight")
    compact_preflight = {
        key: value for key, value in preflight.items() if key != "model_snapshot"
    }
    compact_preflight["model_snapshot"] = {
        "revision": MODEL_REVISION,
        "local_cache_used": True,
        "path_committed": False,
    }
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "assessment_id": config.value["assessment_id"],
        "model": config.model,
        "config_sha256": config.digest(),
        "preflight": compact_preflight,
        "dev16": dev16,
        "full": full,
        "limitations": [
            "The accepted quantized lane is not precision-identical to BF16."
            if preflight["accepted_lane"] == "bitsandbytes-nf4-online-v1"
            else "The accepted lane is BF16.",
            "verifier_timeout is an unsuccessful proof outcome, not an infrastructure error.",
            "Raw candidate continuations remain outside Git.",
        ],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("preflight.json", compact_preflight),
        ("dev16.json", dev16),
        ("full.json", full),
    ):
        (evidence_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "README.md").write_text(
        _render_evidence_readme(payload), encoding="utf-8"
    )
    return payload


def _compact_run(
    config: Ministral3AssessmentConfig,
    artifact_dir: Path,
    *,
    expected_workload: str,
    expected_tasks: int,
) -> dict[str, Any]:
    metadata, results = read_artifacts(artifact_dir)
    stored_summary = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    if metadata.workload_id != expected_workload:
        raise ValueError(f"unexpected workload in {artifact_dir}: {metadata.workload_id}")
    if metadata.model_id != MODEL_ID or metadata.model_revision != MODEL_REVISION:
        raise ValueError("artifact model identity differs from issue #52")
    if (
        metadata.tokenizer_id != MODEL_ID
        or metadata.tokenizer_revision != MODEL_REVISION
    ):
        raise ValueError("artifact tokenizer identity differs from issue #52")
    if metadata.prompt_format_id != PROMPT_FORMAT_ID:
        raise ValueError("artifact prompt format differs from whole-proof-v1")
    if metadata.verifier_timeout_seconds != 30.0:
        raise ValueError("artifact verifier timeout differs from 30 seconds")
    if metadata.candidates_per_task != 4:
        raise ValueError("artifact candidate budget differs from four")
    if metadata.runtime.get("config_sha256") != config.digest():
        raise ValueError("artifact config digest differs from current config")
    settings = metadata.generation_settings or {}
    for key, value in config.sampling.items():
        if settings.get(key) != value:
            raise ValueError(f"artifact sampling mismatch for {key}")
    if any(
        settings.get(key) is not None
        for key in (
            "chat_template",
            "prompt_transformation",
            "proof_extraction",
            "lean_feedback",
            "retry",
        )
    ):
        raise ValueError("artifact applied a forbidden prompt/proof transformation")
    expected_ids = list(config.value["workloads"][expected_workload].get("task_ids", []))
    if not expected_ids:
        expected_ids = [str(item["task_id"]) for item in stored_summary["per_task"]]
    recomputed = summarize_results(
        results, expected_task_ids=expected_ids, candidates_per_task=4
    )
    if len(expected_ids) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, got {len(expected_ids)}")
    for key in (
        "complete",
        "completeness_errors",
        "task_count",
        "candidate_count",
        "candidates_per_task",
        "tasks_with_verified_candidate",
        "pass_at_k",
        "category_counts",
        "finish_reason_counts",
        "verifier_timeout_count",
        "infrastructure_error_count",
        "per_task",
    ):
        if recomputed[key] != stored_summary[key]:
            raise ValueError(f"stored summary differs from raw results for {key}")
    if not recomputed["complete"] or recomputed["infrastructure_error_count"]:
        raise ValueError("assessment artifacts are incomplete or infrastructure-failed")
    expected_candidates = expected_tasks * 4
    if len(results) != expected_candidates:
        raise ValueError(
            f"expected {expected_candidates} candidates, got {len(results)}"
        )
    token_statistics = _token_statistics(results)
    if token_statistics != stored_summary.get("generated_tokens"):
        raise ValueError("stored generated-token summary differs from raw results")
    return {
        "workload_id": expected_workload,
        "task_count": expected_tasks,
        "candidate_count": expected_candidates,
        "verified_candidate_count": recomputed["category_counts"]["verified"],
        "tasks_with_verified_candidate": recomputed["tasks_with_verified_candidate"],
        "pass_at_k": recomputed["pass_at_k"],
        "category_counts": recomputed["category_counts"],
        "finish_reason_counts": recomputed["finish_reason_counts"],
        "verifier_timeout_count": recomputed["verifier_timeout_count"],
        "infrastructure_error_count": recomputed["infrastructure_error_count"],
        "generated_tokens": token_statistics,
        "timing_seconds": stored_summary["timing_seconds"],
        "engine_load_time_seconds": stored_summary["engine_load_time_seconds"],
        "generation_wall_time_seconds": stored_summary[
            "generation_wall_time_seconds"
        ],
        "run_wall_time_seconds": stored_summary["run_wall_time_seconds"],
        "throughput": stored_summary["throughput"],
        "compute_per_solved_task": stored_summary["compute_per_solved_task"],
        "lane_id": settings["lane_id"],
        "precision": settings["dtype"],
        "quantization": settings["quantization"],
        "quantization_metadata": settings["quantization_metadata"],
        "runtime": {
            key: value
            for key, value in metadata.runtime.items()
            if key not in {"model_snapshot", "accepted_preflight"}
        },
        "verifier_environment": metadata.verifier_environment,
        "results_jsonl_sha256": _file_sha256(artifact_dir / "results.jsonl"),
        "candidate_text_sha256": _candidate_text_digest(results),
    }


def _token_statistics(results: list[CandidateResult]) -> dict[str, Any]:
    counts = [int(result.generated_token_count or 0) for result in results]
    ordered = sorted(counts)
    return {
        "total": sum(counts),
        "minimum": min(counts) if counts else None,
        "maximum": max(counts) if counts else None,
        "mean": statistics.fmean(counts) if counts else None,
        "median": statistics.median(counts) if counts else None,
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
    }


def _nearest_rank(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    index = max(0, int(fraction * len(values) + 0.999999999) - 1)
    return values[min(index, len(values) - 1)]


def _candidate_text_digest(results: list[CandidateResult]) -> str:
    digest = hashlib.sha256()
    for result in results:
        digest.update(result.task_id.encode())
        digest.update(b"\0")
        digest.update(str(result.candidate_index).encode())
        digest.update(b"\0")
        digest.update(result.candidate_text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_evidence_readme(payload: dict[str, Any]) -> str:
    full = payload["full"]
    dev16 = payload["dev16"]
    preflight = payload["preflight"]
    lane = preflight["accepted_lane"]
    return (
        "# Ministral 3 8B Base strict miniF2F assessment\n\n"
        f"The immutable `{MODEL_ID}` snapshot `{MODEL_REVISION}` was evaluated "
        "locally on the project RTX 4000 Ada under the unchanged raw "
        "`whole-proof-v1` contract. No chat template, image input, proof extraction, "
        "repair, Lean feedback, or retry was applied.\n\n"
        f"The accepted precision lane was `{lane}`. The BF16 feasibility attempt "
        f"reported status `{preflight['bf16_attempt']['status']}` and memory failure "
        f"`{preflight['bf16_attempt']['memory_failure']}`.\n\n"
        "| Workload | Tasks | Candidates | Solved tasks | Verified candidates | pass@1 | pass@4 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| dev16 | {dev16['task_count']} | {dev16['candidate_count']} | "
        f"{dev16['tasks_with_verified_candidate']['count']} | "
        f"{dev16['verified_candidate_count']} | {dev16['pass_at_k']['pass@1']:.10f} | "
        f"{dev16['pass_at_k']['pass@4']:.10f} |\n"
        f"| full validation | {full['task_count']} | {full['candidate_count']} | "
        f"{full['tasks_with_verified_candidate']['count']} | "
        f"{full['verified_candidate_count']} | {full['pass_at_k']['pass@1']:.10f} | "
        f"{full['pass_at_k']['pass@4']:.10f} |\n\n"
        f"The full run generated {full['generated_tokens']['total']} tokens in "
        f"{full['generation_wall_time_seconds']:.2f} generation seconds. It retained "
        f"{full['verifier_timeout_count']} verifier-timeout proof outcomes and "
        f"{full['infrastructure_error_count']} unresolved infrastructure errors. "
        "Raw candidates and model/cache artifacts remain outside Git.\n"
    )


def _configure_vllm_environment(config: Ministral3AssessmentConfig) -> None:
    _configure_cuda_toolkit(config)
    expected_mp = "0" if not config.runtime["vllm_enable_v1_multiprocessing"] else "1"
    requested = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": expected_mp,
        "VLLM_WORKER_MULTIPROC_METHOD": str(
            config.runtime["vllm_worker_multiproc_method"]
        ),
    }
    for name, expected in requested.items():
        current = os.environ.get(name)
        if current is not None and current != expected:
            raise RuntimeError(
                f"{name}={current!r} conflicts with frozen runtime value {expected!r}"
            )
        os.environ[name] = expected


def _configure_cuda_toolkit(config: Ministral3AssessmentConfig) -> None:
    if config.runtime["cuda_toolkit_source"] != "isolated-python-runtime":
        raise RuntimeError("unsupported CUDA toolkit source")
    import sysconfig

    cuda_home = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"isolated runtime CUDA compiler is missing: {nvcc}")
    _ensure_cuda_linker_layout(cuda_home, config)
    requested = str(cuda_home.resolve())
    existing = os.environ.get("CUDA_HOME")
    if existing is not None and Path(existing).resolve() != cuda_home.resolve():
        raise RuntimeError(
            f"CUDA_HOME={existing!r} conflicts with isolated runtime {requested!r}"
        )
    os.environ["CUDA_HOME"] = requested
    bin_path = str((cuda_home / "bin").resolve())
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_path not in path_entries:
        os.environ["PATH"] = os.pathsep.join([bin_path, *path_entries])
    if shutil.which("nvcc") != str(nvcc.resolve()):
        raise RuntimeError("isolated runtime nvcc is not first on PATH")


def _ensure_cuda_linker_layout(
    cuda_home: Path, config: Ministral3AssessmentConfig
) -> None:
    if config.runtime["cuda_linker_layout"] != "python-wheel-lib64-compat-v1":
        raise RuntimeError("unsupported CUDA linker layout")
    runtime_library = cuda_home / "lib" / "libcudart.so.13"
    if not runtime_library.is_file():
        raise RuntimeError(
            f"isolated runtime CUDA library is missing: {runtime_library}"
        )
    linker_dir = cuda_home / "lib64"
    linker_dir.mkdir(exist_ok=True)
    linker_library = linker_dir / "libcudart.so"
    if linker_library.is_symlink():
        if linker_library.resolve() != runtime_library.resolve():
            raise RuntimeError(
                f"isolated runtime CUDA linker path conflicts: {linker_library}"
            )
    elif linker_library.exists():
        raise RuntimeError(
            f"isolated runtime CUDA linker path conflicts: {linker_library}"
        )
    else:
        linker_library.symlink_to(runtime_library)


def _validate_runtime_versions(
    config: Ministral3AssessmentConfig, vllm_module: Any
) -> None:
    import bitsandbytes
    import mistral_common
    import torch
    import transformers

    actual = {
        "inference_engine_version": str(vllm_module.__version__),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "bitsandbytes_version": str(bitsandbytes.__version__),
        "mistral_common_version": str(mistral_common.__version__),
    }
    expected = {key: str(config.runtime[key]) for key in actual}
    if actual != expected:
        raise RuntimeError(f"runtime package mismatch: expected {expected}, got {actual}")


def _local_runtime(config: Ministral3AssessmentConfig) -> dict[str, Any]:
    import bitsandbytes
    import mistral_common
    import torch
    import transformers
    import vllm

    _validate_runtime_versions(config, vllm)
    if not torch.cuda.is_available():
        raise RuntimeError("issue #52 requires the project local CUDA GPU")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    if properties.name != config.runtime["expected_cuda_device_name"]:
        raise RuntimeError(
            "issue #52 requires NVIDIA RTX 4000 Ada Generation; "
            f"detected {properties.name!r}"
        )
    nvcc = subprocess.run(
        ["nvcc", "--version"], text=True, capture_output=True, check=False
    )
    if nvcc.returncode != 0:
        raise RuntimeError(f"nvcc version probe failed: {nvcc.stderr.strip()}")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "mistral_common": mistral_common.__version__,
        "cuda_toolkit_source": config.runtime["cuda_toolkit_source"],
        "cuda_linker_layout": config.runtime["cuda_linker_layout"],
        "nvcc_version": nvcc.stdout.strip().splitlines()[-1],
        "cuda_device_index": index,
        "cuda_device": properties.name,
        "cuda_device_capability": [properties.major, properties.minor],
        "cuda_device_total_memory_bytes": properties.total_memory,
        "vllm_enable_v1_multiprocessing": bool(
            config.runtime["vllm_enable_v1_multiprocessing"]
        ),
        "vllm_worker_multiproc_method": str(
            config.runtime["vllm_worker_multiproc_method"]
        ),
        "environment_probe_timeout_seconds": ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    }
