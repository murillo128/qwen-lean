from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import _GpuMemoryMonitor
from .dataset_v3 import first_proof_construct, normalized_proof_structure
from .schema import TaskRecord
from .verifier import LeanVerifier, VerificationOutcome

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
VLLM_VERSION = "0.27.2rc1.dev203+g41f179b57"
MATHIA_FREEZE_ID = (
    "frontier_assisted_intuition_corpus_"
    "26b3030e14ab4df8694b9c14ab30a297214f33f7a7ebdb8686ccd2ffae37849f"
)
WORKLOADS = ("minif2f-valid-clean-v2", "fresh-composition-valid-v2")
ARMS = ("t0", "t1")
EXPECTED_COUNTS = {
    "minif2f-valid-clean-v2": 223,
    "fresh-composition-valid-v2": 388,
}
FROZEN_USER_TEMPLATE = (
    "Complete the Lean proof below.\n"
    "Use the mathematical intuition as high-level guidance for the proof.\n"
    "In the final answer, output only the Lean proof body that continues after "
    "the existing `:= by`.\n"
    "Do not repeat the theorem declaration or `by`.\n"
    "Do not include Markdown, explanations, comments, or any natural language "
    "in the final answer.\n"
    "Do not use `sorry` or `admit`.\n"
    "\n"
    "Mathematical intuition:\n"
    "<FROZEN_INTUITION_BYTES>\n"
    "\n"
    "Lean theorem:\n"
    "<UNCHANGED_DECLARATION> := by\n"
)
FROZEN_SAMPLING = {
    "candidates_per_task": 4,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 4096,
    "seed": 0,
    "candidate_seed_mapping": "seed_plus_candidate_index",
    "stop": "tokenizer_eos_or_token_limit",
}
GENERATION_RECORD_SCHEMA = "qwen35-native-thinking-generation-v1"
VERIFICATION_RECORD_SCHEMA = "qwen35-native-thinking-verification-v1"


@dataclass(frozen=True)
class NativeThinkingConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> NativeThinkingConfig:
        config = cls(
            path=path.resolve(), value=json.loads(path.read_text(encoding="utf-8"))
        )
        validate_config(config)
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def mathia(self) -> dict[str, Any]:
        return self.value["mathia"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.value["sampling"]

    @property
    def engine(self) -> dict[str, Any]:
        return self.value["engine"]

    @property
    def verifier(self) -> dict[str, Any]:
        return self.value["verifier"]


@dataclass(frozen=True)
class MathiaTask:
    task_id: str
    workload: str
    preamble: str
    declaration: str
    declaration_name: str
    intuition: str
    intuition_sha256: str
    theorem_sha256: str

    def verifier_task(self) -> TaskRecord:
        return TaskRecord(
            id=self.task_id,
            preamble=self.preamble,
            declaration=self.declaration,
            declaration_name=self.declaration_name,
        )


@dataclass(frozen=True)
class CandidateRequest:
    task: MathiaTask
    arm: str
    enable_thinking: bool
    candidate_index: int
    candidate_seed: int
    user_message: str
    user_message_sha256: str
    rendered_prompt: str
    rendered_prompt_sha256: str
    rendered_prompt_token_count: int
    generation_config_sha256: str
    candidate_id: str


def validate_config(config: NativeThinkingConfig) -> None:
    required = [
        (("schema_version",), "qwen35-native-thinking-ab-config-v1"),
        (("model", "model_id"), MODEL_ID),
        (("model", "model_revision"), MODEL_REVISION),
        (("model", "tokenizer_id"), MODEL_ID),
        (("model", "tokenizer_revision"), MODEL_REVISION),
        (("mathia", "freeze_id"), MATHIA_FREEZE_ID),
        (("mathia", "expected_counts"), EXPECTED_COUNTS),
        (("prompt", "roles"), ["user"]),
        (("prompt", "user_template"), FROZEN_USER_TEMPLATE),
        (
            ("arms",),
            {"t0": {"enable_thinking": False}, "t1": {"enable_thinking": True}},
        ),
        (("engine", "name"), "vllm"),
        (("engine", "version"), VLLM_VERSION),
        (("engine", "reasoning_parser"), "qwen3"),
        (("engine", "dtype"), "bfloat16"),
        (("engine", "tensor_parallel_size"), 1),
        (("engine", "enforce_eager"), True),
        (("engine", "quantization"), None),
        (("engine", "language_model_only"), True),
        (("engine", "resolve_pinned_snapshot"), True),
        (("engine", "use_flashinfer_sampler"), False),
    ]
    for path, expected in required:
        observed: Any = config.value
        for key in path:
            if not isinstance(observed, dict) or key not in observed:
                raise ValueError(
                    "missing native-thinking config field: " + ".".join(path)
                )
            observed = observed[key]
        if observed != expected:
            raise ValueError(
                f"native-thinking config requires {'.'.join(path)}={expected!r}, "
                f"got {observed!r}"
            )
    if config.sampling != FROZEN_SAMPLING:
        raise ValueError("native-thinking sampling differs from the frozen contract")
    if int(config.engine["max_model_len"]) <= int(config.sampling["max_new_tokens"]):
        raise ValueError("max_model_len must leave room for the frozen prompt")
    if set(config.verifier["workloads"]) != set(WORKLOADS):
        raise ValueError("native-thinking verifier workload set changed")


def render_user_message(task: MathiaTask) -> str:
    if FROZEN_USER_TEMPLATE.count("<FROZEN_INTUITION_BYTES>") != 1 or (
        FROZEN_USER_TEMPLATE.count("<UNCHANGED_DECLARATION>") != 1
    ):
        raise AssertionError("frozen prompt placeholders are ambiguous")
    return FROZEN_USER_TEMPLATE.replace(
        "<FROZEN_INTUITION_BYTES>", task.intuition
    ).replace("<UNCHANGED_DECLARATION>", task.declaration)


def load_mathia_tasks(
    config: NativeThinkingConfig, mathia_root: Path
) -> tuple[list[MathiaTask], dict[str, Any]]:
    corpus_root = _corpus_root(config, mathia_root)
    freeze_path = corpus_root / "freeze.json"
    accepted_path = corpus_root / "accepted_intuitions.jsonl"
    source_path = corpus_root / "source_tasks.jsonl"
    expected_hashes = {
        freeze_path: str(config.mathia["freeze_sha256"]),
        accepted_path: str(config.mathia["accepted_intuitions_sha256"]),
        source_path: str(config.mathia["source_tasks_sha256"]),
    }
    for path, expected in expected_hashes.items():
        observed = _file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"Mathia frozen input hash differs for {path.name}: {observed}"
            )

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("freeze_id") != MATHIA_FREEZE_ID:
        raise ValueError("Mathia freeze identity changed")
    if freeze.get("decision") != "FRONTIER_ASSISTED_INTUITION_CORPUS_READY":
        raise ValueError("Mathia corpus is not frozen ready evidence")

    accepted = _read_jsonl(accepted_path)
    sources = _read_jsonl(source_path)
    source_by_id = {str(row["task_id"]): row for row in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("Mathia source tasks contain duplicate ids")

    tasks: list[MathiaTask] = []
    seen: set[str] = set()
    for intuition in accepted:
        task_id = str(intuition["task_id"])
        workload = str(intuition["workload"])
        if task_id in seen:
            raise ValueError(f"duplicate accepted Mathia task: {task_id}")
        seen.add(task_id)
        if workload not in WORKLOADS:
            raise ValueError(f"unexpected accepted Mathia workload: {workload}")
        source = source_by_id.get(task_id)
        if source is None:
            raise ValueError(f"accepted intuition has no source task: {task_id}")
        if source.get("workload") != workload:
            raise ValueError(f"Mathia workload mismatch for {task_id}")
        if source.get("model_visible_theorem_sha256") != intuition.get(
            "model_visible_theorem_sha256"
        ):
            raise ValueError(f"Mathia theorem binding mismatch for {task_id}")
        text = str(intuition["text"])
        if _sha256_text(text) != intuition.get("text_sha256"):
            raise ValueError(f"Mathia intuition text hash mismatch for {task_id}")
        if intuition.get("semantic_boundary_decision") != "accepted_intuition":
            raise ValueError(f"Mathia task is not boundary accepted: {task_id}")
        tasks.append(
            MathiaTask(
                task_id=task_id,
                workload=workload,
                preamble=str(source["public_context"]),
                declaration=str(source["declaration"]),
                declaration_name=str(source["declaration_name"]),
                intuition=text,
                intuition_sha256=str(intuition["text_sha256"]),
                theorem_sha256=str(source["model_visible_theorem_sha256"]),
            )
        )

    counts = Counter(task.workload for task in tasks)
    if dict(counts) != EXPECTED_COUNTS or len(tasks) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Mathia accepted population changed: {dict(counts)}")
    membership = {
        workload: _sha256_json(
            [task.task_id for task in tasks if task.workload == workload]
        )
        for workload in WORKLOADS
    }
    return tasks, {
        "corpus_root": str(corpus_root.resolve()),
        "freeze_id": str(freeze["freeze_id"]),
        "file_sha256": {path.name: digest for path, digest in expected_hashes.items()},
        "counts": dict(counts),
        "ordered_task_ids_sha256": membership,
    }


def candidate_identity(
    config: NativeThinkingConfig,
    *,
    arm: str,
    task: MathiaTask,
    prompt_sha256: str,
    candidate_index: int,
) -> tuple[str, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown native-thinking arm: {arm}")
    if candidate_index not in range(int(config.sampling["candidates_per_task"])):
        raise ValueError(f"invalid candidate index: {candidate_index}")
    generation_hash = generation_config_sha256(config)
    payload = {
        "arm": arm,
        "enable_thinking": bool(config.value["arms"][arm]["enable_thinking"]),
        "workload": task.workload,
        "task_id": task.task_id,
        "prompt_sha256": prompt_sha256,
        "candidate_index": candidate_index,
        "seed": int(config.sampling["seed"]) + candidate_index,
        "model_revision": MODEL_REVISION,
        "generation_config_sha256": generation_hash,
    }
    return "native-thinking-" + _sha256_json(payload)[:32], payload


def generation_config_sha256(config: NativeThinkingConfig) -> str:
    return _sha256_json(
        {
            "model": config.model,
            "sampling": config.sampling,
            "engine": {
                key: config.engine[key]
                for key in (
                    "name",
                    "version",
                    "reasoning_parser",
                    "dtype",
                    "tensor_parallel_size",
                    "max_model_len",
                    "enforce_eager",
                    "quantization",
                    "language_model_only",
                    "resolve_pinned_snapshot",
                    "use_flashinfer_sampler",
                )
            },
        }
    )


def run_generation(
    config: NativeThinkingConfig,
    mathia_root: Path,
    arm: str,
    output_dir: Path,
    *,
    task_ids: set[str] | None = None,
    candidate_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    if arm not in ARMS:
        raise ValueError(f"unknown native-thinking arm: {arm}")
    tasks, binding = load_mathia_tasks(config, mathia_root)
    if task_ids is not None:
        tasks = [task for task in tasks if task.task_id in task_ids]
        if {task.task_id for task in tasks} != task_ids:
            missing = sorted(task_ids - {task.task_id for task in tasks})
            raise ValueError(f"unknown requested Mathia task ids: {missing}")
    indices = tuple(
        range(int(config.sampling["candidates_per_task"]))
        if candidate_indices is None
        else candidate_indices
    )
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("candidate indices must be unique and non-empty")
    if any(
        index not in range(int(config.sampling["candidates_per_task"]))
        for index in indices
    ):
        raise ValueError("candidate index falls outside frozen budget")

    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = output_dir / "generations.jsonl"
    prior = load_generation_records(generation_path)
    completed = {str(row["candidate_id"]) for row in prior}
    if any(row["arm"] != arm for row in prior if task_ids is None):
        # A shared full-run directory intentionally contains both arms.
        pass

    snapshot_path = _resolve_model_snapshot(config)
    tokenizer = _load_tokenizer(config, snapshot_path=snapshot_path)
    enable_thinking = bool(config.value["arms"][arm]["enable_thinking"])
    requests: list[CandidateRequest] = []
    prompt_counts: list[int] = []
    for task in tasks:
        user_message = render_user_message(task)
        user_hash = _sha256_text(user_message)
        rendered, token_count = _render_native_prompt(
            tokenizer, user_message, enable_thinking=enable_thinking
        )
        prompt_counts.append(token_count)
        if token_count + int(config.sampling["max_new_tokens"]) > int(
            config.engine["max_model_len"]
        ):
            raise ValueError(
                f"frozen prompt plus generation budget exceeds max_model_len for "
                f"{task.task_id}: {token_count} + {config.sampling['max_new_tokens']}"
            )
        for candidate_index in indices:
            candidate_id, identity = candidate_identity(
                config,
                arm=arm,
                task=task,
                prompt_sha256=user_hash,
                candidate_index=candidate_index,
            )
            if candidate_id in completed:
                continue
            requests.append(
                CandidateRequest(
                    task=task,
                    arm=arm,
                    enable_thinking=enable_thinking,
                    candidate_index=candidate_index,
                    candidate_seed=int(identity["seed"]),
                    user_message=user_message,
                    user_message_sha256=user_hash,
                    rendered_prompt=rendered,
                    rendered_prompt_sha256=_sha256_text(rendered),
                    rendered_prompt_token_count=token_count,
                    generation_config_sha256=str(identity["generation_config_sha256"]),
                    candidate_id=candidate_id,
                )
            )

    if not requests:
        return {
            "status": "already_complete",
            "arm": arm,
            "requested_candidates": len(tasks) * len(indices),
            "new_candidates": 0,
            "prompt_token_count": _distribution(prompt_counts),
            "mathia_binding": binding,
        }

    _configure_runtime()
    runtime = _local_runtime(config)
    monitor = _GpuMemoryMonitor(int(runtime["cuda_device_index"]), required=True)
    segment_started = time.perf_counter()
    monitor.start()
    status = "failed"
    new_records: list[dict[str, Any]] = []
    error_text: str | None = None
    try:
        new_records = asyncio.run(
            _run_async_generation(
                config,
                tokenizer,
                requests,
                generation_path,
                snapshot_path,
            )
        )
        status = "completed"
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        runtime.update(monitor.stop())
        runtime["segment_wall_time_seconds"] = time.perf_counter() - segment_started
        runtime["status"] = status
        runtime["arm"] = arm
        runtime["requested_candidate_count"] = len(requests)
        runtime["persisted_candidate_count"] = len(new_records)
        runtime["error"] = error_text
        runtime["package_versions"] = _package_versions()
        runtime["model_snapshot_revision"] = snapshot_path.name
        _append_jsonl(output_dir / "generation-segments.jsonl", runtime)

    return {
        "status": status,
        "arm": arm,
        "requested_candidates": len(tasks) * len(indices),
        "new_candidates": len(new_records),
        "prompt_token_count": _distribution(prompt_counts),
        "mathia_binding": binding,
        "runtime": runtime,
    }


async def _run_async_generation(
    config: NativeThinkingConfig,
    tokenizer: Any,
    requests: Sequence[CandidateRequest],
    generation_path: Path,
    snapshot_path: Path,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(snapshot_path),
        tokenizer=str(snapshot_path),
        revision=str(config.model["model_revision"]),
        tokenizer_revision=str(config.model["tokenizer_revision"]),
        dtype=str(config.engine["dtype"]),
        tensor_parallel_size=int(config.engine["tensor_parallel_size"]),
        gpu_memory_utilization=float(config.engine["gpu_memory_utilization"]),
        max_model_len=int(config.engine["max_model_len"]),
        max_num_seqs=int(config.engine["max_num_seqs"]),
        enforce_eager=bool(config.engine["enforce_eager"]),
        quantization=config.engine["quantization"],
        language_model_only=bool(config.engine["language_model_only"]),
        reasoning_parser=str(config.engine["reasoning_parser"]),
        generation_config="vllm",
        enable_log_requests=False,
        disable_log_stats=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    queue: asyncio.Queue[CandidateRequest] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    persisted: list[dict[str, Any]] = []
    persisted_lock = asyncio.Lock()
    progress_every = max(1, min(100, len(requests) // 10 or 1))

    async def worker() -> None:
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            started = time.perf_counter()
            last_output: Any | None = None
            params = SamplingParams(
                n=1,
                temperature=float(config.sampling["temperature"]),
                top_p=float(config.sampling["top_p"]),
                top_k=int(config.sampling["top_k"]),
                min_p=float(config.sampling["min_p"]),
                presence_penalty=float(config.sampling["presence_penalty"]),
                repetition_penalty=float(config.sampling["repetition_penalty"]),
                max_tokens=int(config.sampling["max_new_tokens"]),
                seed=request.candidate_seed,
                skip_special_tokens=False,
            )
            try:
                async for output in engine.generate(
                    request.rendered_prompt,
                    params,
                    request_id=request.candidate_id,
                    prompt_text=request.rendered_prompt,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": {
                            "enable_thinking": request.enable_thinking
                        }
                    },
                ):
                    last_output = output
                if last_output is None or not last_output.finished:
                    raise RuntimeError(
                        f"vLLM request did not complete: {request.candidate_id}"
                    )
                if len(last_output.outputs) != 1:
                    raise RuntimeError(
                        f"vLLM returned {len(last_output.outputs)} completions for n=1"
                    )
                record = _generation_record(
                    config,
                    tokenizer,
                    request,
                    last_output.outputs[0],
                    latency_seconds=time.perf_counter() - started,
                )
                async with persisted_lock:
                    _append_jsonl(generation_path, record)
                    persisted.append(record)
                    if len(persisted) % progress_every == 0 or len(persisted) == len(
                        requests
                    ):
                        print(
                            json.dumps(
                                {
                                    "phase": "generation",
                                    "arm": request.arm,
                                    "completed_candidates": len(persisted),
                                    "total_candidates": len(requests),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(min(int(config.engine["max_in_flight_requests"]), len(requests)))
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        engine.shutdown()
        gc.collect()
    return persisted


def _generation_record(
    config: NativeThinkingConfig,
    tokenizer: Any,
    request: CandidateRequest,
    completion: Any,
    *,
    latency_seconds: float,
) -> dict[str, Any]:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.reasoning import ReasoningParserManager

    token_ids = [int(value) for value in completion.token_ids]
    raw_text = str(completion.text)
    parser_class = ReasoningParserManager.get_reasoning_parser(
        str(config.engine["reasoning_parser"])
    )
    parser = parser_class(
        tokenizer,
        chat_template_kwargs={"enable_thinking": request.enable_thinking},
    )
    parser_request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": request.user_message}],
        max_tokens=int(config.sampling["max_new_tokens"]),
        temperature=float(config.sampling["temperature"]),
        top_p=float(config.sampling["top_p"]),
        include_reasoning=True,
    )
    reasoning, final_content = parser.extract_reasoning(raw_text, parser_request)
    if not request.enable_thinking:
        reasoning_token_count = 0
        final_token_count = len(token_ids)
    else:
        reasoning_token_count = int(parser.count_reasoning_tokens(token_ids))
        final_token_count = (
            0 if final_content is None else len(parser.extract_content_ids(token_ids))
        )
    raw_finish_reason = (
        None if completion.finish_reason is None else str(completion.finish_reason)
    )
    finish_reason = _finish_reason(raw_finish_reason)
    final_exact = (
        final_content == raw_text
        if not request.enable_thinking
        else final_content is None or raw_text.endswith(final_content)
    )
    identity_payload = {
        "arm": request.arm,
        "enable_thinking": request.enable_thinking,
        "workload": request.task.workload,
        "task_id": request.task.task_id,
        "prompt_sha256": request.user_message_sha256,
        "candidate_index": request.candidate_index,
        "seed": request.candidate_seed,
        "model_revision": MODEL_REVISION,
        "generation_config_sha256": request.generation_config_sha256,
    }
    expected_id = "native-thinking-" + _sha256_json(identity_payload)[:32]
    if expected_id != request.candidate_id:
        raise AssertionError("candidate identity changed during generation")
    return {
        "schema_version": GENERATION_RECORD_SCHEMA,
        "candidate_id": request.candidate_id,
        **identity_payload,
        "intuition_sha256": request.task.intuition_sha256,
        "theorem_sha256": request.task.theorem_sha256,
        "rendered_prompt_sha256": request.rendered_prompt_sha256,
        "rendered_prompt_token_count": request.rendered_prompt_token_count,
        "raw_response_text": raw_text,
        "raw_response_sha256": _sha256_text(raw_text),
        "raw_response_token_ids": token_ids,
        "raw_response_token_count": len(token_ids),
        "reasoning_content": reasoning,
        "reasoning_content_sha256": (
            None if reasoning is None else _sha256_text(reasoning)
        ),
        "reasoning_token_count": reasoning_token_count,
        "final_content": final_content,
        "final_content_sha256": (
            None if final_content is None else _sha256_text(final_content)
        ),
        "final_token_count": final_token_count,
        "parser_final_content_is_exact_raw_suffix": final_exact,
        "finish_reason": finish_reason,
        "raw_finish_reason": raw_finish_reason,
        "generation_latency_seconds": latency_seconds,
        "request_id": request.candidate_id,
    }


def load_generation_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    seen: set[str] = set()
    for record in records:
        if record.get("schema_version") != GENERATION_RECORD_SCHEMA:
            raise ValueError("unknown native-thinking generation schema")
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate persisted generation: {candidate_id}")
        seen.add(candidate_id)
        if _sha256_text(str(record["raw_response_text"])) != record.get(
            "raw_response_sha256"
        ):
            raise ValueError(f"raw response hash mismatch: {candidate_id}")
        final = record.get("final_content")
        if final is not None and _sha256_text(str(final)) != record.get(
            "final_content_sha256"
        ):
            raise ValueError(f"final content hash mismatch: {candidate_id}")
        # Parser metadata from an interrupted pre-checkpoint implementation may
        # have counted an all-reasoning token-limit response as final tokens.
        # The retained raw response is immutable; normalize this derived count
        # deterministically without regenerating the completed candidate.
        if final is None:
            record["final_token_count"] = 0
    return records


def validate_lean_environments(
    config: NativeThinkingConfig,
    tasks: Sequence[MathiaTask],
    project_roots: Mapping[str, Path],
) -> dict[str, Any]:
    if set(project_roots) != set(WORKLOADS):
        raise ValueError("project roots must cover both frozen workloads")
    evidence: dict[str, Any] = {}
    for workload in WORKLOADS:
        project_root = project_roots[workload].resolve()
        expected = config.verifier["workloads"][workload]
        toolchain = (
            (project_root / "lean-toolchain").read_text(encoding="utf-8").strip()
        )
        manifest = json.loads(
            (project_root / "lake-manifest.json").read_text(encoding="utf-8")
        )
        packages = {
            str(item["name"]): str(item["rev"]) for item in manifest["packages"]
        }
        if toolchain != expected["lean_toolchain"]:
            raise ValueError(f"Lean toolchain differs for {workload}: {toolchain}")
        if packages.get("mathlib") != expected["mathlib_revision"]:
            raise ValueError(
                f"mathlib revision differs for {workload}: {packages.get('mathlib')}"
            )
        exemplar = next(task for task in tasks if task.workload == workload)
        control = TaskRecord(
            id=f"{workload}-environment-control",
            preamble=exemplar.preamble,
            declaration="theorem native_thinking_environment_control : True",
            declaration_name="native_thinking_environment_control",
        )
        verifier = LeanVerifier(
            project_root,
            timeout_seconds=float(config.verifier["environment_probe_timeout_seconds"]),
        )
        valid = verifier.verify(control, "exact True.intro")
        placeholder = verifier.verify(control, "sorry")
        if valid.category != "verified" or placeholder.category != "lean_rejected":
            raise RuntimeError(
                f"Lean environment controls failed for {workload}: "
                f"{valid.category}/{placeholder.category}"
            )
        evidence[workload] = {
            "project_root": str(project_root),
            "project_head": _git_revision(project_root),
            "lean_toolchain": toolchain,
            "mathlib_revision": packages["mathlib"],
            "known_valid_control": valid.category,
            "placeholder_control": placeholder.category,
        }
    return evidence


def run_verification(
    config: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    *,
    project_roots: Mapping[str, Path],
    workers: int | None = None,
) -> dict[str, Any]:
    tasks, _ = load_mathia_tasks(config, mathia_root)
    tasks_by_id = {task.task_id: task for task in tasks}
    environments = validate_lean_environments(config, tasks, project_roots)
    generation_records = load_generation_records(artifact_dir / "generations.jsonl")
    verification_path = artifact_dir / "verifications.jsonl"
    prior = load_verification_records(verification_path)
    completed = {str(record["candidate_id"]) for record in prior}
    pending = [
        record
        for record in generation_records
        if str(record["candidate_id"]) not in completed
    ]
    if not pending:
        return {
            "status": "already_complete",
            "generation_candidates": len(generation_records),
            "new_verifications": 0,
            "environments": environments,
        }
    verifiers = {
        workload: LeanVerifier(
            project_roots[workload],
            timeout_seconds=float(config.verifier["timeout_seconds"]),
        )
        for workload in WORKLOADS
    }
    worker_count = int(config.verifier["workers"] if workers is None else workers)
    started = time.perf_counter()
    new_count = 0
    progress_every = max(1, min(100, len(pending) // 10 or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _verify_generation_record,
                record,
                tasks_by_id[str(record["task_id"])],
                verifiers[str(record["workload"])],
            ): str(record["candidate_id"])
            for record in pending
        }
        for future in as_completed(futures):
            record = future.result()
            _append_jsonl(verification_path, record)
            new_count += 1
            if new_count % progress_every == 0 or new_count == len(pending):
                print(
                    json.dumps(
                        {
                            "phase": "verification",
                            "completed_candidates": new_count,
                            "total_candidates": len(pending),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    segment = {
        "status": "completed",
        "candidate_count": new_count,
        "workers": worker_count,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _append_jsonl(artifact_dir / "verification-segments.jsonl", segment)
    return {
        "status": "completed",
        "generation_candidates": len(generation_records),
        "new_verifications": new_count,
        "environments": environments,
        "runtime": segment,
    }


def _verify_generation_record(
    generation: dict[str, Any], task: MathiaTask, verifier: LeanVerifier
) -> dict[str, Any]:
    if generation["task_id"] != task.task_id:
        raise ValueError("generation/task mismatch during verification")
    final_content = generation.get("final_content")
    if final_content is None or final_content == "":
        outcome = VerificationOutcome(
            category="empty_candidate",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": "native final content is empty"},
            latency_seconds=0.0,
        )
    else:
        # Insert the native final channel byte-for-byte after the accepted theorem
        # boundary. No stripping, extraction, repair, or line filtering occurs.
        source = f"{task.preamble}\n\n{task.declaration} := by\n  {final_content}\n"
        outcome = verifier._run_source(source)
    return {
        "schema_version": VERIFICATION_RECORD_SCHEMA,
        "candidate_id": generation["candidate_id"],
        "arm": generation["arm"],
        "workload": generation["workload"],
        "task_id": generation["task_id"],
        "candidate_index": generation["candidate_index"],
        "seed": generation["seed"],
        "prompt_sha256": generation["prompt_sha256"],
        "generation_config_sha256": generation["generation_config_sha256"],
        "final_content_sha256": generation["final_content_sha256"],
        "final_content_submitted_without_repair": True,
        "category": outcome.category,
        "lean_exit_code": outcome.lean_exit_code,
        "diagnostics": outcome.diagnostics,
        "verification_latency_seconds": outcome.latency_seconds,
    }


def load_verification_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    seen: set[str] = set()
    for record in records:
        if record.get("schema_version") != VERIFICATION_RECORD_SCHEMA:
            raise ValueError("unknown native-thinking verification schema")
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate persisted verification: {candidate_id}")
        seen.add(candidate_id)
    return records


def run_preflight(
    config: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    output_path: Path,
    *,
    project_roots: Mapping[str, Path],
) -> dict[str, Any]:
    tasks, binding = load_mathia_tasks(config, mathia_root)
    tokenizer = _load_tokenizer(config)
    prompt_audit = _prompt_audit(config, tokenizer, tasks)
    parser_integrity = _parser_integrity(config, tokenizer)
    environments = validate_lean_environments(config, tasks, project_roots)
    smoke_task = next(
        task for task in tasks if task.workload == "minif2f-valid-clean-v2"
    )
    smoke_ids = {smoke_task.task_id}
    for arm in ARMS:
        run_generation(
            config,
            mathia_root,
            arm,
            artifact_dir,
            task_ids=smoke_ids,
            candidate_indices=(0,),
        )
    run_verification(
        config,
        mathia_root,
        artifact_dir,
        project_roots=project_roots,
        workers=1,
    )
    generations = load_generation_records(artifact_dir / "generations.jsonl")
    verifications = load_verification_records(artifact_dir / "verifications.jsonl")
    if len(generations) != 2 or {row["arm"] for row in generations} != set(ARMS):
        raise RuntimeError(
            "preflight smoke does not contain exactly one candidate per arm"
        )
    if len(verifications) != 2:
        raise RuntimeError("preflight smoke verification is incomplete")
    t0 = next(row for row in generations if row["arm"] == "t0")
    if t0["reasoning_content"] not in {None, ""}:
        raise RuntimeError("T0 native smoke unexpectedly produced reasoning content")
    if not all(row["parser_final_content_is_exact_raw_suffix"] for row in generations):
        raise RuntimeError("native parser changed final content bytes")
    segments = _read_jsonl(artifact_dir / "generation-segments.jsonl")
    successful_segments = [row for row in segments if row.get("status") == "completed"]
    if not successful_segments:
        raise RuntimeError("preflight recorded no successful generation segment")
    peak = max(int(row["gpu_memory_peak_bytes"]) for row in successful_segments)
    total_memory = max(
        int(row["cuda_device_total_memory_bytes"]) for row in successful_segments
    )
    if peak <= 0 or peak > total_memory:
        raise RuntimeError("preflight GPU memory evidence is invalid")
    evidence = {
        "schema_version": "qwen35-native-thinking-pre-inference-v1",
        "status": "passed",
        "experiment_id": config.value["experiment_id"],
        "model": config.model,
        "runtime": {
            "engine": config.engine,
            "package_versions": _package_versions(),
            "generation_segments": [
                _compact_generation_segment(row) for row in successful_segments
            ],
            "gpu_memory_peak_bytes": peak,
            "gpu_memory_total_bytes": total_memory,
        },
        "mathia_binding": _compact_mathia_binding(binding),
        "prompt_audit": prompt_audit,
        "candidate_identity": {
            "generation_config_sha256": generation_config_sha256(config),
            "candidate_seed_mapping": config.sampling["candidate_seed_mapping"],
            "candidate_seeds": [
                int(config.sampling["seed"]) + index
                for index in range(int(config.sampling["candidates_per_task"]))
            ],
            "prospective_candidate_count": sum(EXPECTED_COUNTS.values())
            * int(config.sampling["candidates_per_task"])
            * len(ARMS),
        },
        "parser_integrity": parser_integrity,
        "lean_environments": _compact_lean_environments(environments),
        "smoke": {
            "task_id": smoke_task.task_id,
            "operational_only": True,
            "quality_not_used_for_contract_changes": True,
            "generation": {
                row["arm"]: {
                    "candidate_id": row["candidate_id"],
                    "finish_reason": row["finish_reason"],
                    "raw_response_sha256": row["raw_response_sha256"],
                    "reasoning_present": bool(row["reasoning_content"]),
                    "reasoning_token_count": row["reasoning_token_count"],
                    "final_content_sha256": row["final_content_sha256"],
                    "final_token_count": row["final_token_count"],
                }
                for row in generations
            },
            "verification": {row["arm"]: row["category"] for row in verifications},
            "execution_status": "completed_persisted",
            "verification_status": "completed_persisted",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, evidence)
    return evidence


def _prompt_audit(
    config: NativeThinkingConfig, tokenizer: Any, tasks: Sequence[MathiaTask]
) -> dict[str, Any]:
    user_hashes: dict[str, str] = {}
    chat_hashes: dict[str, dict[str, str]] = {}
    token_counts: dict[str, list[int]] = {arm: [] for arm in ARMS}
    suffix_differences: Counter[str] = Counter()
    common_prefix_bytes: list[int] = []
    for task in tasks:
        message = render_user_message(task)
        user_hashes[task.task_id] = _sha256_text(message)
        rendered: dict[str, str] = {}
        for arm in ARMS:
            prompt, count = _render_native_prompt(
                tokenizer,
                message,
                enable_thinking=bool(config.value["arms"][arm]["enable_thinking"]),
            )
            rendered[arm] = prompt
            token_counts[arm].append(count)
        chat_hashes[task.task_id] = {arm: _sha256_text(rendered[arm]) for arm in ARMS}
        prefix = os.path.commonprefix([rendered["t0"], rendered["t1"]])
        common_prefix_bytes.append(len(prefix.encode("utf-8")))
        suffix_differences[
            _sha256_json(
                {
                    "t0_suffix": rendered["t0"][len(prefix) :],
                    "t1_suffix": rendered["t1"][len(prefix) :],
                }
            )
        ] += 1
    return {
        "task_count": len(tasks),
        "single_user_message_only": True,
        "user_message_identical_between_arms": True,
        "ordered_user_message_hashes_sha256": _sha256_json(
            [[task.task_id, user_hashes[task.task_id]] for task in tasks]
        ),
        "ordered_native_chat_hashes_sha256": _sha256_json(
            [[task.task_id, chat_hashes[task.task_id]] for task in tasks]
        ),
        "native_template_call_difference": "chat_template_kwargs.enable_thinking only",
        "native_template_suffix_difference_classes": len(suffix_differences),
        "native_template_suffix_difference_counts": dict(suffix_differences),
        "native_template_common_prefix_bytes": _distribution(common_prefix_bytes),
        "prompt_token_counts": {
            arm: _distribution(values) for arm, values in token_counts.items()
        },
        "all_prompts_fit_frozen_budget": all(
            value + int(config.sampling["max_new_tokens"])
            <= int(config.engine["max_model_len"])
            for values in token_counts.values()
            for value in values
        ),
    }


def _parser_integrity(config: NativeThinkingConfig, tokenizer: Any) -> dict[str, Any]:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager

    parser_class = ReasoningParserManager.get_reasoning_parser("qwen3")
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "deterministic parser probe"}],
        include_reasoning=True,
    )
    t0_raw = "exact True.intro\n"
    t1_reasoning = "deterministic reasoning bytes\n"
    t1_final = "exact True.intro\n"
    t1_raw = f"<think>{t1_reasoning}</think>{t1_final}"
    t0_parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": False})
    t1_parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
    t0_reasoning, t0_final = t0_parser.extract_reasoning(t0_raw, request)
    parsed_reasoning, parsed_final = t1_parser.extract_reasoning(t1_raw, request)
    checks = {
        "t0_reasoning_absent": t0_reasoning is None,
        "t0_final_exact": t0_final == t0_raw,
        "t1_reasoning_separated": parsed_reasoning == t1_reasoning,
        "t1_final_exact": parsed_final == t1_final,
        "raw_evidence_retained": _sha256_text(t1_raw)
        != _sha256_text(t1_reasoning + t1_final),
        "parser_registered_as_qwen3": "qwen3"
        in ReasoningParserManager.list_registered(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Qwen3 reasoning parser integrity failed: {checks}")
    return {
        "status": "passed",
        "parser": config.engine["reasoning_parser"],
        "checks": checks,
        "t0_raw_sha256": _sha256_text(t0_raw),
        "t1_raw_sha256": _sha256_text(t1_raw),
        "t1_reasoning_sha256": _sha256_text(t1_reasoning),
        "t1_final_sha256": _sha256_text(t1_final),
    }


def write_final_evidence(
    config: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    preflight_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    tasks, binding = load_mathia_tasks(config, mathia_root)
    generations = load_generation_records(artifact_dir / "generations.jsonl")
    verifications = load_verification_records(artifact_dir / "verifications.jsonl")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise ValueError("pre-inference gate did not pass")
    expected_candidates = len(tasks) * int(config.sampling["candidates_per_task"]) * 2
    if len(generations) != expected_candidates:
        raise ValueError(
            f"full generation is incomplete: {len(generations)}/{expected_candidates}"
        )
    if len(verifications) != expected_candidates:
        raise ValueError(
            f"full verification is incomplete: {len(verifications)}/{expected_candidates}"
        )
    generation_by_id = {row["candidate_id"]: row for row in generations}
    verification_by_id = {row["candidate_id"]: row for row in verifications}
    if set(generation_by_id) != set(verification_by_id):
        raise ValueError("generation and verification candidate sets differ")
    if any(row["category"] == "verifier_error" for row in verifications):
        raise ValueError("full run contains verifier infrastructure errors")
    if any(not row["parser_final_content_is_exact_raw_suffix"] for row in generations):
        raise ValueError("full run contains parser final-byte integrity failures")
    if any(
        verification_by_id[candidate_id]["final_content_sha256"]
        != generation["final_content_sha256"]
        for candidate_id, generation in generation_by_id.items()
    ):
        raise ValueError("verified final content differs from generated final content")

    joined = [
        {**generation, "verification": verification_by_id[generation["candidate_id"]]}
        for generation in generations
    ]
    analysis = analyze_results(config, tasks, joined)
    segments = _read_jsonl(artifact_dir / "generation-segments.jsonl")
    verification_segments = _read_jsonl(artifact_dir / "verification-segments.jsonl")
    cost = _cost_metrics(joined, analysis, segments, verification_segments)
    result = {
        "schema_version": "qwen35-native-thinking-final-evidence-v1",
        "status": "complete",
        "experiment_id": config.value["experiment_id"],
        "model": config.model,
        "mathia_binding": _compact_mathia_binding(binding),
        "causal_contract": {
            "user_message_identical_between_arms": True,
            "model_revision_identical": True,
            "sampling_identical": True,
            "candidate_seed_mapping_identical": True,
            "verifier_semantics_identical": True,
            "only_intended_variable": "chat_template_kwargs.enable_thinking",
            "no_output_repair": True,
            "reasoning_never_submitted_to_lean": True,
        },
        "analysis": analysis,
        "cost": cost,
        "interpretation": _interpretation(analysis),
        "limitations": [
            "Thinking intentionally consumes additional test-time tokens; the arms are not compute matched.",
            "Natural-language contamination is a deterministic heuristic diagnostic only.",
            "The accepted frozen workload metadata does not expose a source-proof-free direct/branching/deep label, so that optional breakdown is unavailable.",
        ],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(evidence_dir / "results.json", result)
    (evidence_dir / "README.md").write_text(
        _render_final_readme(result), encoding="utf-8"
    )
    return result


def analyze_results(
    config: NativeThinkingConfig,
    tasks: Sequence[MathiaTask],
    joined: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for workload in (*WORKLOADS, "combined"):
        task_ids = [
            task.task_id
            for task in tasks
            if workload == "combined" or task.workload == workload
        ]
        task_set = set(task_ids)
        subset = [row for row in joined if row["task_id"] in task_set]
        arms = {
            arm: _arm_metrics([row for row in subset if row["arm"] == arm], task_ids)
            for arm in ARMS
        }
        paired = _paired_metrics(
            config,
            task_ids,
            [row for row in subset if row["arm"] == "t0"],
            [row for row in subset if row["arm"] == "t1"],
        )
        result[workload] = {
            "task_count": len(task_ids),
            "arms": arms,
            "paired": paired,
            "direct_branching_deep": {
                "available": False,
                "reason": "frozen source-proof-free Mathia projection has no structural class field",
            },
        }
    return result


def _arm_metrics(
    rows: Sequence[dict[str, Any]], task_ids: Sequence[str]
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    if set(by_task) != set(task_ids) or any(
        len(by_task[task_id]) != 4 for task_id in task_ids
    ):
        raise ValueError("arm result set is incomplete")
    verified_per_task = {
        task_id: sum(
            row["verification"]["category"] == "verified" for row in by_task[task_id]
        )
        for task_id in task_ids
    }
    pass_at_1 = statistics.fmean(value / 4 for value in verified_per_task.values())
    pass_at_4 = statistics.fmean(value > 0 for value in verified_per_task.values())
    verified_count = sum(verified_per_task.values())
    candidate_count = len(rows)
    reasoning_counts = [int(row["reasoning_token_count"]) for row in rows]
    final_counts = [int(row["final_token_count"]) for row in rows]
    ratios = [
        reasoning / final
        for reasoning, final in zip(reasoning_counts, final_counts, strict=True)
        if final > 0
    ]
    category_counts = Counter(row["verification"]["category"] for row in rows)
    finish_counts = Counter(row["finish_reason"] for row in rows)
    final_texts = [str(row.get("final_content") or "") for row in rows]
    normalized = [normalized_proof_structure(text) for text in final_texts]
    verified_normalized = [
        value
        for value, row in zip(normalized, rows, strict=True)
        if row["verification"]["category"] == "verified"
    ]
    diagnostics = {
        "reasoning_present": _count_fraction(
            sum(bool(row.get("reasoning_content")) for row in rows), candidate_count
        ),
        "reasoning_tokens": _distribution(reasoning_counts),
        "final_tokens": _distribution(final_counts),
        "reasoning_to_final_token_ratio": _distribution(ratios),
        "token_limit_before_usable_final": _count_fraction(
            sum(
                row["finish_reason"] == "token_limit"
                and not (row.get("final_content") or "")
                for row in rows
            ),
            candidate_count,
        ),
        "empty_or_missing_final": _count_fraction(
            sum(not (row.get("final_content") or "") for row in rows),
            candidate_count,
        ),
        "markdown_fence": _count_fraction(
            sum("```" in text for text in final_texts), candidate_count
        ),
        "repeated_declaration_or_by": _count_fraction(
            sum(_repeats_declaration_or_by(text) for text in final_texts),
            candidate_count,
        ),
        "sorry_or_admit": _count_fraction(
            sum(bool(re.search(r"\b(?:sorry|admit)\b", text)) for text in final_texts),
            candidate_count,
        ),
        "apparent_natural_language": {
            **_count_fraction(
                sum(_apparent_natural_language(text) for text in final_texts),
                candidate_count,
            ),
            "heuristic_id": "final-line-four-words-v1",
            "used_for_filtering_or_repair": False,
        },
    }
    return {
        "candidate_count": candidate_count,
        "verified_candidate": _count_fraction(verified_count, candidate_count),
        "solved_within_4": _count_fraction(
            sum(value > 0 for value in verified_per_task.values()), len(task_ids)
        ),
        "pass_at_k": {"pass@1": pass_at_1, "pass@4": pass_at_4},
        "verifier_result_counts": dict(sorted(category_counts.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "interface_diagnostics": diagnostics,
        "diversity": {
            "normalization": "dataset-v3 normalized_proof_structure without source dependencies",
            "all_final_outputs": {
                "unique": len(set(normalized)),
                "fraction": len(set(normalized)) / candidate_count,
            },
            "verified_final_outputs": {
                "unique": len(set(verified_normalized)),
                "count": len(verified_normalized),
                "fraction": (
                    len(set(verified_normalized)) / len(verified_normalized)
                    if verified_normalized
                    else None
                ),
            },
            "first_lean_construct_counts": dict(
                sorted(
                    Counter(first_proof_construct(text) for text in final_texts).items()
                )
            ),
        },
    }


def _paired_metrics(
    config: NativeThinkingConfig,
    task_ids: Sequence[str],
    t0_rows: Sequence[dict[str, Any]],
    t1_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    def task_values(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["task_id"])].append(row)
        return {
            task_id: (
                sum(
                    row["verification"]["category"] == "verified"
                    for row in grouped[task_id]
                )
                / 4,
                float(
                    any(
                        row["verification"]["category"] == "verified"
                        for row in grouped[task_id]
                    )
                ),
            )
            for task_id in task_ids
        }

    t0 = task_values(t0_rows)
    t1 = task_values(t1_rows)
    both = sum(t0[task_id][1] == 1 and t1[task_id][1] == 1 for task_id in task_ids)
    t0_only = sum(t0[task_id][1] == 1 and t1[task_id][1] == 0 for task_id in task_ids)
    t1_only = sum(t0[task_id][1] == 0 and t1[task_id][1] == 1 for task_id in task_ids)
    neither = len(task_ids) - both - t0_only - t1_only
    deltas = {
        "pass@1": [t1[task_id][0] - t0[task_id][0] for task_id in task_ids],
        "pass@4": [t1[task_id][1] - t0[task_id][1] for task_id in task_ids],
    }
    bootstrap = {
        key: _paired_bootstrap(
            values,
            resamples=int(config.value["analysis"]["bootstrap_resamples"]),
            seed=int(config.value["analysis"]["bootstrap_seed"]),
            percentiles=tuple(config.value["analysis"]["bootstrap_percentiles"]),
        )
        for key, values in deltas.items()
    }
    t0_verified: dict[str, set[str]] = defaultdict(set)
    t1_verified: dict[str, set[str]] = defaultdict(set)
    for rows, target in ((t0_rows, t0_verified), (t1_rows, t1_verified)):
        for row in rows:
            if row["verification"]["category"] == "verified":
                target[str(row["task_id"])].add(
                    normalized_proof_structure(str(row.get("final_content") or ""))
                )
    shared_verified_proof_tasks = sum(
        bool(t0_verified[task_id] & t1_verified[task_id]) for task_id in task_ids
    )
    return {
        "solved_at_4_overlap": {
            "t0_only": t0_only,
            "t1_only": t1_only,
            "both": both,
            "neither": neither,
        },
        "paired_delta_t1_minus_t0": {
            key: statistics.fmean(values) for key, values in deltas.items()
        },
        "mcnemar_exact_two_sided": {
            "discordant_t0_only": t0_only,
            "discordant_t1_only": t1_only,
            "p_value": _mcnemar_exact(t0_only, t1_only),
        },
        "paired_bootstrap_t1_minus_t0": bootstrap,
        "task_level_verified_proof_overlap": {
            "tasks_with_shared_normalized_verified_proof": shared_verified_proof_tasks,
            "fraction": shared_verified_proof_tasks / len(task_ids),
        },
    }


def _paired_bootstrap(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    percentiles: tuple[float, float],
) -> dict[str, Any]:
    rng = random.Random(seed)
    size = len(values)
    samples = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    )
    return {
        "method": "paired-task-percentile-bootstrap",
        "resamples": resamples,
        "seed": seed,
        "estimate": statistics.fmean(values),
        "interval": [
            _percentile(samples, percentiles[0]),
            _percentile(samples, percentiles[1]),
        ],
        "interval_percentiles": list(percentiles),
    }


def _mcnemar_exact(t0_only: int, t1_only: int) -> float:
    discordant = t0_only + t1_only
    if discordant == 0:
        return 1.0
    smaller = min(t0_only, t1_only)
    lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * lower_tail)


def _cost_metrics(
    joined: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    generation_segments: Sequence[dict[str, Any]],
    verification_segments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in joined if row["arm"] == arm]
        segments = [
            row
            for row in generation_segments
            if row.get("arm") == arm and row.get("status") == "completed"
        ]
        reasoning = sum(int(row["reasoning_token_count"]) for row in rows)
        final = sum(int(row["final_token_count"]) for row in rows)
        raw = sum(int(row["raw_response_token_count"]) for row in rows)
        wall = sum(float(row["segment_wall_time_seconds"]) for row in segments)
        verified = int(analysis["combined"]["arms"][arm]["verified_candidate"]["count"])
        solved = int(analysis["combined"]["arms"][arm]["solved_within_4"]["count"])
        result[arm] = {
            "reasoning_tokens": reasoning,
            "final_tokens": final,
            "raw_generated_tokens": raw,
            "generation_wall_time_seconds": wall,
            "raw_generated_tokens_per_second": raw / wall if wall else None,
            "peak_gpu_memory_bytes": max(
                (int(row["gpu_memory_peak_bytes"]) for row in segments), default=None
            ),
            "generated_tokens_per_verified_candidate": (
                raw / verified if verified else None
            ),
            "generated_tokens_per_solved_task": raw / solved if solved else None,
        }
    result["verification_wall_time_seconds"] = sum(
        float(row["wall_time_seconds"])
        for row in verification_segments
        if row.get("status") == "completed"
    )
    result["compute_matched"] = False
    return result


def _interpretation(analysis: dict[str, Any]) -> dict[str, Any]:
    combined = analysis["combined"]
    delta = float(combined["paired"]["paired_delta_t1_minus_t0"]["pass@4"])
    t0 = combined["arms"]["t0"]
    t1 = combined["arms"]["t1"]
    if delta > 0:
        category = "t1_improves_lean_success"
    elif delta < 0 and (
        t1["interface_diagnostics"]["token_limit_before_usable_final"]["count"]
        > t0["interface_diagnostics"]["token_limit_before_usable_final"]["count"]
        or t1["interface_diagnostics"]["apparent_natural_language"]["count"]
        > t0["interface_diagnostics"]["apparent_natural_language"]["count"]
    ):
        category = "t1_hurts_with_token_or_format_behavior"
    elif delta == 0 and (
        t1["diversity"]["all_final_outputs"]["unique"]
        != t0["diversity"]["all_final_outputs"]["unique"]
    ):
        category = "t1_changes_diversity_without_coverage_gain"
    else:
        category = "t1_neutral_or_ordinary_lean_failure"
    return {
        "category": category,
        "automatic_architecture_change_authorized": False,
        "external_planner_issue_changed": False,
    }


def _render_final_readme(result: dict[str, Any]) -> str:
    combined = result["analysis"]["combined"]
    t0 = combined["arms"]["t0"]
    t1 = combined["arms"]["t1"]
    paired = combined["paired"]
    t1_budget_exhausted = t1["interface_diagnostics"]["token_limit_before_usable_final"]
    return f"""# Qwen3.5-4B native thinking A/B

**OBSERVED:** both frozen native-chat arms completed all 611 Mathia-guided tasks
and 2,444 candidates per arm. T0 solved {t0["solved_within_4"]["count"]} tasks
within four candidates; T1 solved {t1["solved_within_4"]["count"]}. Combined
pass@1/pass@4 were {t0["pass_at_k"]["pass@1"]:.6f}/{t0["pass_at_k"]["pass@4"]:.6f}
for T0 and {t1["pass_at_k"]["pass@1"]:.6f}/{t1["pass_at_k"]["pass@4"]:.6f}
for T1. The paired solved@4 delta (T1 minus T0) was
{paired["paired_delta_t1_minus_t0"]["pass@4"]:.6f}; exact two-sided McNemar
p={paired["mcnemar_exact_two_sided"]["p_value"]:.6g}.

**ACCEPTED:** model/tokenizer revision `{MODEL_REVISION}`, Mathia freeze
`{MATHIA_FREEZE_ID}`, user-visible prompt bytes, temperature 0.6, top-p 0.95,
top-k 20, four candidates, seed mapping, 4,096-token output budget, BF16,
native Qwen chat template, and Lean verifier semantics were matched. The only
intended variable was `chat_template_kwargs.enable_thinking`. The vLLM `qwen3`
parser separated native reasoning from final content; only exact final-channel
bytes were submitted to Lean, without extraction, sanitization, repair, or
verifier-driven retry.

**OBSERVED:** the deterministic interpretation category is
`{result["interpretation"]["category"]}`. T1 hit the shared token limit before
usable final content on {t1_budget_exhausted["count"]}/{t1["candidate_count"]} candidates
({t1_budget_exhausted["fraction"]:.2%});
reasoning-budget exhaustion, rather than final-channel contamination, was the
dominant observed interface failure. Thinking is not compute matched: cost and
token totals for each arm are retained in `results.json`. This result does not
change the external-planner design or authorize a training architecture.
"""


def _render_native_prompt(
    tokenizer: Any, user_message: str, *, enable_thinking: bool
) -> tuple[str, int]:
    messages = [{"role": "user", "content": user_message}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    token_ids = tokenized["input_ids"] if isinstance(tokenized, Mapping) else tokenized
    return str(rendered), len(token_ids)


def _load_tokenizer(
    config: NativeThinkingConfig, *, snapshot_path: Path | None = None
) -> Any:
    from transformers import AutoTokenizer

    snapshot = (
        _resolve_model_snapshot(config) if snapshot_path is None else snapshot_path
    )
    return AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )


def _resolve_model_snapshot(config: NativeThinkingConfig) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=str(config.model["model_id"]),
            revision=str(config.model["model_revision"]),
            local_files_only=True,
        )
    ).resolve()
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError(
            f"resolved model snapshot differs from frozen revision: {snapshot.name}"
        )
    for required in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        if not (snapshot / required).is_file():
            raise RuntimeError(f"pinned model snapshot is incomplete: {required}")
    return snapshot


def _configure_runtime() -> None:
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _local_runtime(config: NativeThinkingConfig) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("native-thinking generation requires torch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("native-thinking generation requires local CUDA")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    expected = str(config.engine["expected_cuda_device_name_fragment"])
    if expected not in name:
        raise RuntimeError(
            f"native-thinking generation requires {expected} GPU, got {name}"
        )
    properties = torch.cuda.get_device_properties(device)
    return {
        "inference_execution": "local_cuda",
        "cuda_device_index": int(device),
        "cuda_device": name,
        "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _package_versions() -> dict[str, str]:
    packages = ("nvidia-ml-py", "torch", "transformers", "vllm")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required runtime package is missing: {package}"
            ) from error
    if versions["vllm"] != VLLM_VERSION:
        raise RuntimeError(
            f"native-thinking vLLM version differs: {versions['vllm']} != {VLLM_VERSION}"
        )
    return versions


def _compact_generation_segment(segment: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "arm",
        "inference_execution",
        "cuda_device_index",
        "cuda_device",
        "cuda_device_capability",
        "cuda_device_total_memory_bytes",
        "torch_cuda_version",
        "gpu_memory_monitoring",
        "gpu_memory_before_bytes",
        "gpu_memory_peak_bytes",
        "gpu_memory_peak_delta_bytes",
        "segment_wall_time_seconds",
        "persisted_candidate_count",
        "package_versions",
        "model_snapshot_revision",
    )
    return {key: segment[key] for key in keys}


def _compact_mathia_binding(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        key: binding[key]
        for key in (
            "freeze_id",
            "file_sha256",
            "counts",
            "ordered_task_ids_sha256",
        )
    }


def _compact_lean_environments(
    environments: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    keys = (
        "lean_toolchain",
        "mathlib_revision",
        "known_valid_control",
        "placeholder_control",
    )
    return {
        workload: {key: environments[workload][key] for key in keys}
        for workload in WORKLOADS
    }


def _finish_reason(value: str | None) -> str:
    if value in {"stop", "eos"}:
        return "eos"
    if value in {"length", "token_limit"}:
        return "token_limit"
    return "unknown" if value is None else value


def _repeats_declaration_or_by(value: str) -> bool:
    stripped = value.lstrip()
    return bool(
        re.match(r"(?:theorem|lemma)\b", stripped) or re.match(r"by(?:\s|$)", stripped)
    )


def _apparent_natural_language(value: str) -> bool:
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("--", "/-", "*")):
            continue
        words = re.findall(r"[A-Za-z]{3,}", stripped)
        if len(words) >= 4 and (
            stripped.endswith((".", "!", "?"))
            or re.search(
                r"\b(?:therefore|because|hence|thus|we\s+(?:need|have|show)|the proof)\b",
                stripped,
                flags=re.IGNORECASE,
            )
        ):
            return True
    return False


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    concrete = [float(value) for value in values]
    ordered = sorted(concrete)
    return {
        "count": len(concrete),
        "min": min(concrete),
        "mean": statistics.fmean(concrete),
        "median": statistics.median(concrete),
        "p90": _percentile(ordered, 90.0),
        "p95": _percentile(ordered, 95.0),
        "max": max(concrete),
    }


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    if not ordered:
        raise ValueError("percentile requires non-empty values")
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _count_fraction(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "fraction": count / total if total else None}


def _corpus_root(config: NativeThinkingConfig, mathia_root: Path) -> Path:
    direct = mathia_root.resolve()
    if (direct / "freeze.json").is_file():
        return direct
    return direct / str(config.mathia["artifact_subdirectory"])


def _git_revision(path: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
