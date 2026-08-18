from __future__ import annotations

import gc
import json
import os
import platform
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import read_artifacts, write_artifacts
from .metrics import summarize_results
from .minif2f import (
    Phase1Config,
    materialize_benchmark_tasks,
    verifier_environment_metadata,
)
from .prompt import PROMPT_FORMAT_ID, render_prompt
from .schema import (
    PHASE1_RESULT_SCHEMA_VERSION,
    CandidateResult,
    RunMetadata,
    TaskRecord,
)
from .verifier import LeanVerifier


@dataclass(frozen=True)
class GeneratedCandidate:
    task: TaskRecord
    candidate_index: int
    text: str
    token_count: int
    finish_reason: str
    generation_latency_seconds: float
    generation_error: str | None = None


@dataclass(frozen=True)
class LoRAAdapterSpec:
    adapter_id: str
    path: Path
    rank: int
    base_model_id: str
    base_model_revision: str

    def validate(self, config: Phase1Config) -> None:
        if not self.adapter_id or self.adapter_id == self.base_model_id:
            raise ValueError("adapter identity must be distinct from the base model")
        if self.rank < 1:
            raise ValueError("adapter rank must be positive")
        if self.base_model_id != str(config.model["model_id"]):
            raise ValueError("adapter base model differs from the inference config")
        if self.base_model_revision != str(config.model["model_revision"]):
            raise ValueError("adapter base revision differs from the inference config")
        if not (self.path / "adapter_config.json").is_file():
            raise ValueError(f"adapter config does not exist: {self.path}")

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "adapter_id": self.adapter_id,
            "adapter_path": str(self.path.resolve()),
            "adapter_rank": self.rank,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "merged": False,
        }


def validate_minif2f_environment(
    config: Phase1Config,
    benchmark_root: Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    tasks = materialize_benchmark_tasks(config, benchmark_root)
    environment = verifier_environment_metadata(config, benchmark_root)
    verifier = LeanVerifier(benchmark_root, timeout_seconds=timeout_seconds)
    known_task_id = str(config.value["verifier"]["known_valid_task_id"])
    known_candidate = str(config.value["verifier"]["known_valid_candidate"])
    try:
        known_task = next(task for task in tasks if task.id == known_task_id)
    except StopIteration as error:
        raise ValueError(
            f"known-valid task not materialized: {known_task_id}"
        ) from error
    outcome = verifier.verify(known_task, known_candidate)
    if outcome.category != "verified":
        diagnostics = outcome.diagnostics["stdout"] + outcome.diagnostics["stderr"]
        raise RuntimeError(
            f"known-valid miniF2F candidate failed as {outcome.category}: {diagnostics}"
        )
    placeholder_outcome = verifier.verify(known_task, "sorry")
    if placeholder_outcome.category != "lean_rejected":
        raise RuntimeError(
            "miniF2F verifier accepted a placeholder candidate as "
            f"{placeholder_outcome.category}"
        )
    return {
        "benchmark_split": str(config.benchmark["split"]),
        "primary_task_count": len(tasks),
        "preamble": tasks[0].preamble,
        "verifier_environment": environment,
        "known_valid_task_id": known_task_id,
        "known_valid_candidate_category": outcome.category,
        "known_valid_candidate_lean_exit_code": outcome.lean_exit_code,
        "placeholder_candidate_category": placeholder_outcome.category,
        "placeholder_candidate_lean_exit_code": placeholder_outcome.lean_exit_code,
    }


def run_phase1_baseline(
    config: Phase1Config,
    benchmark_root: Path,
    workload_id: str,
    output_dir: Path,
    *,
    timeout_seconds: float,
    verification_workers: int,
    sampling_override: Mapping[str, Any] | None = None,
    adapter: LoRAAdapterSpec | None = None,
    result_schema_version: str = PHASE1_RESULT_SCHEMA_VERSION,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    all_tasks = materialize_benchmark_tasks(config, benchmark_root)
    tasks = config.select_workload(workload_id, all_tasks)
    environment_validation = validate_minif2f_environment(
        config,
        benchmark_root,
        timeout_seconds=timeout_seconds,
    )
    environment = environment_validation["verifier_environment"]

    runtime = _local_cuda_runtime(config)
    sampling = dict(config.sampling if sampling_override is None else sampling_override)
    if adapter is not None:
        adapter.validate(config)
    generation_started = time.perf_counter()
    generated, engine_version = _generate_candidates(
        config, tasks, sampling=sampling, adapter=adapter
    )
    generation_wall_time = time.perf_counter() - generation_started
    runtime["generation_wall_time_seconds"] = generation_wall_time

    verifier = LeanVerifier(benchmark_root, timeout_seconds=timeout_seconds)
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=verification_workers) as executor:
        results = list(
            executor.map(lambda item: _verify_candidate(verifier, item), generated)
        )
    verification_wall_time = time.perf_counter() - verification_started
    runtime["verification_wall_time_seconds"] = verification_wall_time
    runtime["verification_workers"] = verification_workers

    summary = summarize_results(
        results,
        expected_task_ids=[task.id for task in tasks],
        candidates_per_task=int(sampling["candidates_per_task"]),
    )
    summary["workload_id"] = workload_id
    summary["run_wall_time_seconds"] = generation_wall_time + verification_wall_time

    metadata = RunMetadata(
        schema_version=result_schema_version,
        candidate_source="model",
        task_source=(
            f"{config.benchmark['repository']}@{config.benchmark['revision']}:"
            f"{config.benchmark['source_path']}"
        ),
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=str(environment["lean_toolchain"]),
        mathlib_revision=str(environment["dependencies"]["mathlib"]),
        verifier_timeout_seconds=timeout_seconds,
        model_id=str(config.model["model_id"]),
        tokenizer_id=str(config.model["tokenizer_id"]),
        model_revision=str(config.model["model_revision"]),
        tokenizer_revision=str(config.model["tokenizer_revision"]),
        workload_id=workload_id,
        benchmark_split=str(config.benchmark["split"]),
        benchmark_repository=str(config.benchmark["repository"]),
        benchmark_revision=str(config.benchmark["revision"]),
        verifier_environment=environment,
        candidates_per_task=int(sampling["candidates_per_task"]),
        inference_engine=str(config.engine["name"]),
        inference_engine_version=engine_version,
        adapter_enabled=adapter is not None,
        adapter_id=None if adapter is None else adapter.adapter_id,
        adapter_path=None if adapter is None else str(adapter.path.resolve()),
        adapter_rank=None if adapter is None else adapter.rank,
        generation_settings={
            **sampling,
            "dtype": str(config.engine["dtype"]),
            "tensor_parallel_size": int(config.engine["tensor_parallel_size"]),
            "max_model_len": int(config.engine["max_model_len"]),
            "max_num_seqs": int(config.engine["max_num_seqs"]),
            "gpu_memory_utilization": float(config.engine["gpu_memory_utilization"]),
            "enforce_eager": bool(config.engine["enforce_eager"]),
            "quantization": config.engine["quantization"],
            **(
                {"language_model_only": bool(config.engine["language_model_only"])}
                if "language_model_only" in config.engine
                else {}
            ),
            **(
                {"sampler_backend": "native"}
                if config.engine.get("use_flashinfer_sampler") is False
                else {}
            ),
            **(
                {"add_special_tokens": bool(config.model["add_special_tokens"])}
                if "add_special_tokens" in config.model
                else {}
            ),
            "chat_template": config.model.get("chat_template"),
            "prompt_transformation": None,
            "adapter": None if adapter is None else adapter.metadata(),
        },
        runtime=runtime,
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
    return metadata, results, summary


def reverify_phase1_artifacts(
    config: Phase1Config,
    benchmark_root: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    timeout_seconds: float,
    verification_workers: int,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    metadata, original_results = read_artifacts(input_dir)
    if metadata.schema_version != PHASE1_RESULT_SCHEMA_VERSION:
        raise ValueError(f"cannot reverify schema {metadata.schema_version}")
    if metadata.workload_id is None:
        raise ValueError("Phase 1 artifacts have no workload id")
    if metadata.model_revision != str(config.model["model_revision"]):
        raise ValueError("model revision differs from the Phase 1 configuration")
    if metadata.benchmark_revision != str(config.benchmark["revision"]):
        raise ValueError("benchmark revision differs from the Phase 1 configuration")

    all_tasks = materialize_benchmark_tasks(config, benchmark_root)
    tasks = config.select_workload(metadata.workload_id, all_tasks)
    tasks_by_id = {task.id: task for task in tasks}
    validate_minif2f_environment(
        config,
        benchmark_root,
        timeout_seconds=timeout_seconds,
    )
    generated = [
        GeneratedCandidate(
            task=tasks_by_id[result.task_id],
            candidate_index=result.candidate_index,
            text=result.candidate_text,
            token_count=result.generated_token_count or 0,
            finish_reason=result.finish_reason or "unknown",
            generation_latency_seconds=result.generation_latency_seconds or 0.0,
            generation_error=(
                result.diagnostics["stderr"]
                if result.category == "generation_error"
                else None
            ),
        )
        for result in original_results
    ]

    verifier = LeanVerifier(benchmark_root, timeout_seconds=timeout_seconds)
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=verification_workers) as executor:
        results = list(
            executor.map(lambda item: _verify_candidate(verifier, item), generated)
        )
    verification_wall_time = time.perf_counter() - verification_started

    runtime = dict(metadata.runtime)
    previous_verification_wall_time = runtime.get("verification_wall_time_seconds")
    runtime.update(
        {
            "candidate_generation_reused": True,
            "previous_verification_wall_time_seconds": previous_verification_wall_time,
            "verification_wall_time_seconds": verification_wall_time,
            "verification_workers": verification_workers,
        }
    )
    updated_metadata = RunMetadata(
        **{
            **metadata.to_dict(),
            "verifier_timeout_seconds": timeout_seconds,
            "runtime": runtime,
        }
    )
    summary = summarize_results(
        results,
        expected_task_ids=[task.id for task in tasks],
        candidates_per_task=int(config.sampling["candidates_per_task"]),
    )
    summary["workload_id"] = metadata.workload_id
    summary["run_wall_time_seconds"] = (
        float(runtime.get("generation_wall_time_seconds", 0.0)) + verification_wall_time
    )
    write_artifacts(output_dir, updated_metadata, results, summary=summary)
    return updated_metadata, results, summary


def _local_cuda_runtime(config: Phase1Config) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Phase 1 requires the baseline/model optional dependencies"
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 baseline generation requires a local CUDA GPU")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    expected_fragment = str(config.engine["expected_cuda_device_name_fragment"])
    if expected_fragment not in properties.name:
        raise RuntimeError(
            f"Phase 1 requires the project Ada GPU; detected {properties.name!r}"
        )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "inference_execution": "local_cuda",
        "cuda_device_index": device_index,
        "cuda_device": properties.name,
        "cuda_device_capability": [properties.major, properties.minor],
        "cuda_device_total_memory_bytes": properties.total_memory,
    }


def _generate_candidates(
    config: Phase1Config,
    tasks: list[TaskRecord],
    *,
    prompts: list[str] | None = None,
    sampling: Mapping[str, Any] | None = None,
    adapter: LoRAAdapterSpec | None = None,
) -> tuple[list[GeneratedCandidate], str]:
    if config.engine.get("use_flashinfer_sampler") is False:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "Phase 1 requires the baseline optional dependencies"
        ) from error

    engine = config.engine
    configured_version = str(engine["version"])
    if vllm.__version__ != configured_version:
        raise RuntimeError(
            f"vLLM version mismatch: expected {configured_version}, got {vllm.__version__}"
        )

    sampling = dict(config.sampling if sampling is None else sampling)
    prompts = [render_prompt(task) for task in tasks] if prompts is None else prompts
    if len(prompts) != len(tasks):
        raise ValueError("vLLM prompt count differs from task count")
    started = time.perf_counter()
    try:
        llm = LLM(**vllm_engine_kwargs(config, sampling, adapter))
        generate_kwargs: dict[str, Any] = {"use_tqdm": True}
        if adapter is not None:
            from vllm.lora.request import LoRARequest

            generate_kwargs["lora_request"] = LoRARequest(
                adapter.adapter_id, 1, str(adapter.path.resolve())
            )
        outputs = llm.generate(
            prompts,
            SamplingParams(**vllm_sampling_kwargs(sampling)),
            **generate_kwargs,
        )
    except Exception as error:
        latency = time.perf_counter() - started
        message = f"{type(error).__name__}: {error}"
        return _generation_error_records(
            tasks, sampling, message, latency
        ), vllm.__version__

    wall_time = time.perf_counter() - started
    try:
        generated = _convert_vllm_outputs(
            tasks, prompts, outputs, wall_time, sampling=sampling
        )
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    return generated, vllm.__version__


def _convert_vllm_outputs(
    tasks: list[TaskRecord],
    prompts: list[str],
    outputs: list[Any],
    wall_time: float,
    *,
    sampling: Mapping[str, Any],
) -> list[GeneratedCandidate]:
    n = int(sampling["candidates_per_task"])
    if len(outputs) != len(tasks):
        message = f"vLLM returned {len(outputs)} requests for {len(tasks)} tasks"
        return _generation_error_records(tasks, sampling, message, wall_time)

    generated: list[GeneratedCandidate] = []
    fallback_latency = wall_time / len(tasks) if tasks else wall_time
    for task, prompt, request_output in zip(tasks, prompts, outputs, strict=True):
        completions = sorted(request_output.outputs, key=lambda item: item.index)
        indices = [item.index for item in completions]
        if request_output.prompt != prompt or indices != list(range(n)):
            message = (
                f"invalid vLLM output for {task.id}: prompt_match="
                f"{request_output.prompt == prompt}, indices={indices}"
            )
            generated.extend(
                _generation_error_records([task], sampling, message, fallback_latency)
            )
            continue
        latency = _request_latency(request_output, fallback_latency)
        generated.extend(
            GeneratedCandidate(
                task=task,
                candidate_index=completion.index,
                text=completion.text,
                token_count=len(completion.token_ids),
                finish_reason=_finish_reason(completion.finish_reason),
                generation_latency_seconds=latency,
            )
            for completion in completions
        )
    return generated


def _generation_error_records(
    tasks: list[TaskRecord],
    sampling: Mapping[str, Any],
    message: str,
    latency: float,
) -> list[GeneratedCandidate]:
    n = int(sampling["candidates_per_task"])
    return [
        GeneratedCandidate(
            task=task,
            candidate_index=index,
            text="",
            token_count=0,
            finish_reason="generation_error",
            generation_latency_seconds=latency,
            generation_error=message,
        )
        for task in tasks
        for index in range(n)
    ]


def vllm_engine_kwargs(
    config: Phase1Config,
    sampling: Mapping[str, Any],
    adapter: LoRAAdapterSpec | None,
) -> dict[str, Any]:
    engine = config.engine
    kwargs: dict[str, Any] = {
        "model": str(config.model["model_id"]),
        "revision": str(config.model["model_revision"]),
        "tokenizer": str(config.model["tokenizer_id"]),
        "tokenizer_revision": str(config.model["tokenizer_revision"]),
        "dtype": str(engine["dtype"]),
        "tensor_parallel_size": int(engine["tensor_parallel_size"]),
        "gpu_memory_utilization": float(engine["gpu_memory_utilization"]),
        "max_model_len": int(engine["max_model_len"]),
        "max_num_seqs": int(engine["max_num_seqs"]),
        "enforce_eager": bool(engine["enforce_eager"]),
        "quantization": engine["quantization"],
        "seed": int(sampling["seed"]),
        "trust_remote_code": False,
    }
    if "language_model_only" in engine:
        kwargs["language_model_only"] = bool(engine["language_model_only"])
    if adapter is not None:
        adapter.validate(config)
        kwargs.update(
            {
                "enable_lora": True,
                "max_lora_rank": adapter.rank,
                "max_loras": 1,
            }
        )
    return kwargs


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


def _request_latency(request_output: Any, fallback: float) -> float:
    metrics = request_output.metrics
    if (
        metrics is not None
        and metrics.finished_time is not None
        and metrics.finished_time >= metrics.arrival_time
    ):
        return metrics.finished_time - metrics.arrival_time
    return fallback


def _finish_reason(value: str | None) -> str:
    if value == "stop":
        return "eos"
    if value == "length":
        return "token_limit"
    return "unknown" if value is None else value


def _verify_candidate(
    verifier: LeanVerifier, generated: GeneratedCandidate
) -> CandidateResult:
    if generated.generation_error is not None:
        return CandidateResult(
            task_id=generated.task.id,
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
        message = f"{type(error).__name__}: {error}"
        return CandidateResult(
            task_id=generated.task.id,
            candidate_id=f"model-{generated.candidate_index}",
            candidate_index=generated.candidate_index,
            candidate_text=generated.text,
            category="verifier_error",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": message},
            generation_latency_seconds=generated.generation_latency_seconds,
            verification_latency_seconds=None,
            total_latency_seconds=generated.generation_latency_seconds,
            generated_token_count=generated.token_count,
            finish_reason=generated.finish_reason,
        )


def write_environment_validation(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
