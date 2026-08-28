from __future__ import annotations

import asyncio
import gc
import inspect
import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import _GpuMemoryMonitor
from .native_thinking_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    VLLM_VERSION,
    WORKLOADS,
    MathiaTask,
    NativeThinkingConfig,
    _append_jsonl,
    _atomic_write_json,
    _configure_runtime,
    _file_sha256,
    _finish_reason,
    _load_tokenizer,
    _local_runtime,
    _package_versions,
    _render_native_prompt,
    _resolve_model_snapshot,
    _sha256_json,
    _sha256_text,
    load_mathia_tasks,
    render_user_message,
)

SCALING_CONFIG_SCHEMA = "qwen35-thinking-budget-scaling-config-v1"
GENERATION_SCHEMA = "qwen35-thinking-budget-scaling-generation-v1"
GATE_SCHEMA = "qwen35-thinking-budget-runtime-gate-v1"
LEAN_WRAPPER_NORMALIZATION = "lean-wrapper-normalization-v1"
EXPECTED_STAGE1_TARGET = "cbd93de8a96bba9c93fac8afb95e8a8d12205715"
EXPECTED_STAGE1_RESULTS_SHA256 = (
    "5733df8939418fd4c841124e993832169cd05bab546c060972398f7250a163fe"
)
FROZEN_SCALING_SAMPLING = {
    "candidates_per_task": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 0,
    "stop": "tokenizer_eos_or_token_limit",
}
FROZEN_SCALING_ARMS = {
    "B4": {
        "max_reasoning_tokens": 4096,
        "total_output_ceiling": 8192,
        "nominal_final_allowance": 4096,
    },
    "B8": {
        "max_reasoning_tokens": 8192,
        "total_output_ceiling": 12288,
        "nominal_final_allowance": 4096,
    },
    "B16": {
        "max_reasoning_tokens": 16384,
        "total_output_ceiling": 20480,
        "nominal_final_allowance": 4096,
    },
}


def lean_wrapper_normalization_v1(
    parsed_final: str | None,
) -> tuple[str | None, bool]:
    """Remove one exact leading Lean `by` wrapper, and nothing else."""

    if parsed_final is None:
        return None, False
    match = re.match(r"\A\s*by(?![\w'])", parsed_final)
    if match is None:
        return parsed_final, False
    return parsed_final[match.end() :], True


@dataclass(frozen=True)
class ThinkingBudgetScalingConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ThinkingBudgetScalingConfig:
        config = cls(
            path=path.resolve(), value=json.loads(path.read_text(encoding="utf-8"))
        )
        validate_scaling_config(config)
        return config

    @property
    def stage1(self) -> dict[str, Any]:
        return self.value["stage1"]

    @property
    def runtime_gate(self) -> dict[str, Any]:
        return self.value["runtime_gate"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.value["selection"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.value["sampling"]

    @property
    def arms(self) -> dict[str, Any]:
        return self.value["arms"]


@dataclass(frozen=True)
class SelectedTask:
    task: MathiaTask
    frozen_global_index: int
    frozen_workload_index: int
    user_message: str
    user_message_sha256: str
    rendered_prompt: str
    rendered_prompt_sha256: str
    rendered_prompt_token_count: int


@dataclass(frozen=True)
class ScalingRequest:
    selected: SelectedTask
    arm: str
    max_reasoning_tokens: int
    total_output_ceiling: int
    candidate_index: int
    candidate_seed: int
    generation_config_sha256: str
    candidate_id: str
    is_runtime_gate: bool = False


def validate_scaling_config(config: ThinkingBudgetScalingConfig) -> None:
    required = {
        "schema_version": SCALING_CONFIG_SCHEMA,
        "experiment_id": "qwen35-4b-thinking-budget-scaling-v1",
    }
    for key, expected in required.items():
        if config.value.get(key) != expected:
            raise ValueError(f"thinking-budget scaling requires {key}={expected!r}")
    if config.stage1.get("reviewed_target") != EXPECTED_STAGE1_TARGET:
        raise ValueError("Stage 1 reviewed target changed")
    if config.stage1.get("results_sha256") != EXPECTED_STAGE1_RESULTS_SHA256:
        raise ValueError("Stage 1 result binding changed")
    if config.runtime_gate.get("vllm_version") != VLLM_VERSION:
        raise ValueError("Stage 2 must use the exact Stage 1 vLLM version")
    if config.selection != {
        "method": (
            "first-per-workload-in-frozen-stage1-order-with-t1-prompt-at-most-ceiling"
        ),
        "tasks_per_workload": 8,
        "prompt_token_ceiling": 4096,
        "outcome_fields_forbidden": True,
    }:
        raise ValueError("Stage 2 task selection differs from the frozen amendment")
    if config.sampling != FROZEN_SCALING_SAMPLING:
        raise ValueError("Stage 2 sampling differs from the frozen amendment")
    if config.arms != FROZEN_SCALING_ARMS:
        raise ValueError("Stage 2 arms differ from the frozen amendment")
    gate = config.runtime_gate
    if gate.get("probe_workload") != WORKLOADS[0]:
        raise ValueError("runtime gate workload changed")
    if gate.get("probe_final_allowance") != 128:
        raise ValueError("runtime gate final allowance changed")
    if gate.get("reasoning_budgets") != [0, 8, 32]:
        raise ValueError("runtime gate probes changed")
    source_hash = str(gate.get("thinking_budget_state_sha256", ""))
    if len(source_hash) != 64:
        raise ValueError("runtime gate source hash is not SHA-256")


def validate_stage1_binding(
    config: ThinkingBudgetScalingConfig,
    stage1_config: NativeThinkingConfig,
    *,
    stage1_results_path: Path | None = None,
) -> None:
    if _file_sha256(stage1_config.path) != config.stage1["config_sha256"]:
        raise ValueError("Stage 1 config bytes changed")
    if stage1_results_path is not None and (
        _file_sha256(stage1_results_path) != config.stage1["results_sha256"]
    ):
        raise ValueError("Stage 1 compact results changed")
    maximum = max(int(arm["total_output_ceiling"]) for arm in config.arms.values())
    if int(config.selection["prompt_token_ceiling"]) + maximum > int(
        stage1_config.engine["max_model_len"]
    ):
        raise ValueError("Stage 2 prompt plus B16 ceiling exceeds model context")


def select_scaling_tasks(
    config: ThinkingBudgetScalingConfig,
    stage1_config: NativeThinkingConfig,
    tasks: Sequence[MathiaTask],
    tokenizer: Any,
) -> tuple[list[SelectedTask], dict[str, Any]]:
    ceiling = int(config.selection["prompt_token_ceiling"])
    per_workload = int(config.selection["tasks_per_workload"])
    workload_seen = Counter()
    workload_selected = Counter()
    selected: list[SelectedTask] = []
    eligible_counts = Counter()

    for global_index, task in enumerate(tasks):
        workload_index = workload_seen[task.workload]
        workload_seen[task.workload] += 1
        user_message = render_user_message(task)
        rendered, token_count = _render_native_prompt(
            tokenizer, user_message, enable_thinking=True
        )
        if token_count > ceiling:
            continue
        eligible_counts[task.workload] += 1
        if workload_selected[task.workload] >= per_workload:
            continue
        selected.append(
            SelectedTask(
                task=task,
                frozen_global_index=global_index,
                frozen_workload_index=workload_index,
                user_message=user_message,
                user_message_sha256=_sha256_text(user_message),
                rendered_prompt=rendered,
                rendered_prompt_sha256=_sha256_text(rendered),
                rendered_prompt_token_count=token_count,
            )
        )
        workload_selected[task.workload] += 1

    expected = {workload: per_workload for workload in WORKLOADS}
    if dict(workload_selected) != expected:
        raise ValueError(
            f"insufficient Stage 2 eligible tasks: {dict(workload_selected)}"
        )
    maximum = max(int(arm["total_output_ceiling"]) for arm in config.arms.values())
    if any(
        row.rendered_prompt_token_count + maximum
        > int(stage1_config.engine["max_model_len"])
        for row in selected
    ):
        raise ValueError("selected Stage 2 task exceeds B16 context")
    rows = [
        {
            "workload": row.task.workload,
            "task_id": row.task.task_id,
            "frozen_global_index": row.frozen_global_index,
            "frozen_workload_index": row.frozen_workload_index,
            "prompt_sha256": row.user_message_sha256,
            "rendered_prompt_sha256": row.rendered_prompt_sha256,
            "rendered_prompt_token_count": row.rendered_prompt_token_count,
        }
        for row in selected
    ]
    return selected, {
        "method": config.selection["method"],
        "outcome_blind": bool(config.selection["outcome_fields_forbidden"]),
        "prompt_token_ceiling": ceiling,
        "eligible_task_counts": dict(eligible_counts),
        "selected_task_counts": dict(workload_selected),
        "selected_tasks": rows,
        "ordered_selection_sha256": _sha256_json(rows),
    }


def _gate_candidate_identity(
    config: ThinkingBudgetScalingConfig,
    selected: SelectedTask,
    reasoning_budget: int,
) -> tuple[str, dict[str, Any]]:
    total = reasoning_budget + int(config.runtime_gate["probe_final_allowance"])
    payload = {
        "arm": f"gate-B{reasoning_budget}",
        "enable_thinking": True,
        "workload": selected.task.workload,
        "task_id": selected.task.task_id,
        "prompt_sha256": selected.user_message_sha256,
        "candidate_index": 0,
        "seed": int(config.sampling["seed"]),
        "model_revision": MODEL_REVISION,
        "max_reasoning_tokens": reasoning_budget,
        "total_output_ceiling": total,
        "generation_config_sha256": _sha256_json(
            {
                "stage1_target": config.stage1["reviewed_target"],
                "runtime_gate": config.runtime_gate,
                "sampling": config.sampling,
            }
        ),
    }
    return "thinking-budget-gate-" + _sha256_json(payload)[:32], payload


def reasoning_exit_category(
    reasoning_token_count: int,
    max_reasoning_tokens: int,
    final_content: str | None,
    *,
    reasoning_end_position_token_count: int | None = None,
) -> str:
    if final_content is None or final_content == "":
        return "no_final_transition"
    if reasoning_end_position_token_count is not None:
        if (
            max_reasoning_tokens == 0 and reasoning_end_position_token_count == 1
        ) or reasoning_end_position_token_count == max_reasoning_tokens:
            return "forced_at_budget"
        if reasoning_end_position_token_count < max_reasoning_tokens:
            return "natural_before_budget"
        return "budget_exceeded"
    if reasoning_token_count < max_reasoning_tokens:
        return "natural_before_budget"
    if reasoning_token_count == max_reasoning_tokens:
        return "forced_at_budget"
    return "budget_exceeded"


def _request_from_payload(
    selected: SelectedTask,
    payload: Mapping[str, Any],
    candidate_id: str,
    *,
    is_runtime_gate: bool,
) -> ScalingRequest:
    return ScalingRequest(
        selected=selected,
        arm=str(payload["arm"]),
        max_reasoning_tokens=int(payload["max_reasoning_tokens"]),
        total_output_ceiling=int(payload["total_output_ceiling"]),
        candidate_index=int(payload["candidate_index"]),
        candidate_seed=int(payload["seed"]),
        generation_config_sha256=str(payload["generation_config_sha256"]),
        candidate_id=candidate_id,
        is_runtime_gate=is_runtime_gate,
    )


def runtime_support_audit(
    config: ThinkingBudgetScalingConfig,
    stage1_config: NativeThinkingConfig,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.v1.sample import thinking_budget_state

    packages = _package_versions()
    sampling_has_budget = (
        "thinking_token_budget" in inspect.signature(SamplingParams).parameters
    )
    chat_has_budget = "thinking_token_budget" in ChatCompletionRequest.model_fields
    engine_has_reasoning_config = (
        "reasoning_config" in inspect.signature(AsyncEngineArgs).parameters
    )
    source_path = Path(str(thinking_budget_state.__file__)).resolve()
    source_sha256 = _file_sha256(source_path)
    checks = {
        "exact_vllm_version": packages["vllm"] == config.runtime_gate["vllm_version"],
        "sampling_params_supports_thinking_token_budget": sampling_has_budget,
        "chat_protocol_supports_thinking_token_budget": chat_has_budget,
        "engine_supports_reasoning_config": engine_has_reasoning_config,
        "thinking_budget_source_matches_frozen_runtime": source_sha256
        == config.runtime_gate["thinking_budget_state_sha256"],
        "qwen3_reasoning_parser_preserved": stage1_config.engine["reasoning_parser"]
        == "qwen3",
    }
    return {
        "package_versions": packages,
        "thinking_budget_state_module": ("vllm/v1/sample/thinking_budget_state.py"),
        "thinking_budget_state_sha256": source_sha256,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_runtime_gate(
    config: ThinkingBudgetScalingConfig,
    stage1_config: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    output_path: Path,
    *,
    stage1_results_path: Path | None = None,
) -> dict[str, Any]:
    validate_stage1_binding(
        config, stage1_config, stage1_results_path=stage1_results_path
    )
    audit = runtime_support_audit(config, stage1_config)
    if not audit["passed"]:
        evidence = {
            "schema_version": GATE_SCHEMA,
            "status": "failed",
            "conclusion": "budget_control_runtime_or_parser_not_usable",
            "stage1_reviewed_target": config.stage1["reviewed_target"],
            "runtime_support": audit,
            "scientific_generation_authorized": False,
        }
        _write_gate_evidence(output_path, evidence)
        return evidence

    tasks, mathia_binding = load_mathia_tasks(stage1_config, mathia_root)
    snapshot = _resolve_model_snapshot(stage1_config)
    tokenizer = _load_tokenizer(stage1_config, snapshot_path=snapshot)
    selected, selection = select_scaling_tasks(config, stage1_config, tasks, tokenizer)
    probe = next(
        row
        for row in selected
        if row.task.workload == config.runtime_gate["probe_workload"]
    )
    requests: list[ScalingRequest] = []
    for budget in config.runtime_gate["reasoning_budgets"]:
        candidate_id, payload = _gate_candidate_identity(config, probe, int(budget))
        requests.append(
            _request_from_payload(probe, payload, candidate_id, is_runtime_gate=True)
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_path = artifact_dir / "runtime-gate-generations.jsonl"
    prior = load_scaling_generation_records(generation_path)
    completed = {str(row["candidate_id"]) for row in prior}
    pending = [row for row in requests if row.candidate_id not in completed]
    runtime: dict[str, Any] | None = None
    segment_path = artifact_dir / "runtime-gate-segments.jsonl"
    if pending:
        runtime = _execute_requests(
            config,
            stage1_config,
            tokenizer,
            pending,
            generation_path,
            snapshot,
            segment_path,
            segment_kind="runtime_gate",
        )
    elif segment_path.exists():
        runtime = _read_jsonl(segment_path)[-1]
    records = load_scaling_generation_records(generation_path)
    by_id = {str(row["candidate_id"]): row for row in records}
    probe_records = [by_id[row.candidate_id] for row in requests]
    audited_probe_records = [
        _audit_probe_record(row, tokenizer) for row in probe_records
    ]
    checks = {
        "all_probe_records_present": len(probe_records) == len(requests),
        "reasoning_bounded": all(
            int(row["reasoning_token_count"]) <= int(row["max_reasoning_tokens"])
            for row in probe_records
        ),
        "budget_limit_can_transition_to_final": all(
            row["reasoning_exit_audit"] == "forced_at_budget"
            and bool(row["final_content"])
            for row in audited_probe_records
        ),
        "parser_final_is_exact_raw_suffix": all(
            row["parser_final_content_is_exact_raw_suffix"] for row in probe_records
        ),
        "reasoning_end_marker_observed": all(
            row["raw_has_reasoning_end_marker"] for row in probe_records
        ),
        "transition_markers_absent_from_final": all(
            not row["final_has_reasoning_marker"] for row in probe_records
        ),
        "final_generation_continued": all(
            int(row["final_token_count"]) > 0 for row in probe_records
        ),
        "raw_token_ids_retained": all(
            len(row["raw_response_token_ids"]) == int(row["raw_response_token_count"])
            for row in probe_records
        ),
    }
    passed = all(checks.values())
    evidence = {
        "schema_version": GATE_SCHEMA,
        "status": "passed" if passed else "failed",
        "conclusion": (
            "runtime_gate_passed"
            if passed
            else "budget_control_runtime_or_parser_not_usable"
        ),
        "stage1_reviewed_target": config.stage1["reviewed_target"],
        "stage1_config_sha256": _file_sha256(stage1_config.path),
        "runtime_support": audit,
        "mathia_binding": {
            key: value for key, value in mathia_binding.items() if key != "corpus_root"
        },
        "selection_binding": {
            "ordered_selection_sha256": selection["ordered_selection_sha256"],
            "probe_task_id": probe.task.task_id,
            "probe_workload": probe.task.workload,
            "prompt_sha256": probe.user_message_sha256,
            "rendered_prompt_sha256": probe.rendered_prompt_sha256,
            "rendered_prompt_token_count": probe.rendered_prompt_token_count,
        },
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "probes": [_compact_generation(row) for row in audited_probe_records],
        "runtime": runtime,
        "raw_artifacts": {
            "generation_path": _repository_relative_path(generation_path),
            "generation_sha256": _file_sha256(generation_path),
        },
        "scientific_generation_authorized": passed,
        "scientific_generation_started": False,
        "scientific_generation_candidate_count": 0,
    }
    _write_gate_evidence(output_path, evidence)
    return evidence


def _execute_requests(
    config: ThinkingBudgetScalingConfig,
    stage1_config: NativeThinkingConfig,
    tokenizer: Any,
    requests: Sequence[ScalingRequest],
    generation_path: Path,
    snapshot_path: Path,
    segment_path: Path,
    *,
    segment_kind: str,
) -> dict[str, Any]:
    _configure_runtime()
    runtime = _local_runtime(stage1_config)
    monitor = _GpuMemoryMonitor(int(runtime["cuda_device_index"]), required=True)
    started = time.perf_counter()
    monitor.start()
    status = "failed"
    persisted: list[dict[str, Any]] = []
    error_text: str | None = None
    try:
        persisted = asyncio.run(
            _run_async_requests(
                config,
                stage1_config,
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
        runtime.update(
            {
                "schema_version": "qwen35-thinking-budget-runtime-segment-v1",
                "segment_kind": segment_kind,
                "status": status,
                "arm": requests[0].arm
                if len({row.arm for row in requests}) == 1
                else None,
                "requested_candidate_count": len(requests),
                "persisted_candidate_count": len(persisted),
                "segment_wall_time_seconds": time.perf_counter() - started,
                "error": error_text,
                "package_versions": _package_versions(),
                "model_snapshot_revision": snapshot_path.name,
            }
        )
        _append_jsonl(segment_path, runtime)
    return runtime


async def _run_async_requests(
    config: ThinkingBudgetScalingConfig,
    stage1_config: NativeThinkingConfig,
    tokenizer: Any,
    requests: Sequence[ScalingRequest],
    generation_path: Path,
    snapshot_path: Path,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=str(snapshot_path),
        tokenizer=str(snapshot_path),
        revision=str(stage1_config.model["model_revision"]),
        tokenizer_revision=str(stage1_config.model["tokenizer_revision"]),
        dtype=str(stage1_config.engine["dtype"]),
        tensor_parallel_size=int(stage1_config.engine["tensor_parallel_size"]),
        gpu_memory_utilization=float(stage1_config.engine["gpu_memory_utilization"]),
        max_model_len=int(stage1_config.engine["max_model_len"]),
        max_num_seqs=int(stage1_config.engine["max_num_seqs"]),
        enforce_eager=bool(stage1_config.engine["enforce_eager"]),
        quantization=stage1_config.engine["quantization"],
        language_model_only=bool(stage1_config.engine["language_model_only"]),
        reasoning_parser=str(stage1_config.engine["reasoning_parser"]),
        generation_config="vllm",
        enable_log_requests=False,
        disable_log_stats=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    queue: asyncio.Queue[ScalingRequest] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    persisted: list[dict[str, Any]] = []
    persisted_lock = asyncio.Lock()
    progress_every = max(1, len(requests) // 8)

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
                max_tokens=request.total_output_ceiling,
                thinking_token_budget=request.max_reasoning_tokens,
                seed=request.candidate_seed,
                skip_special_tokens=False,
            )
            try:
                async for output in engine.generate(
                    request.selected.rendered_prompt,
                    params,
                    request_id=request.candidate_id,
                    prompt_text=request.selected.rendered_prompt,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": {"enable_thinking": True}
                    },
                ):
                    last_output = output
                if last_output is None or not last_output.finished:
                    raise RuntimeError(
                        f"vLLM request did not complete: {request.candidate_id}"
                    )
                if len(last_output.outputs) != 1:
                    raise RuntimeError("Stage 2 vLLM request returned n != 1")
                record = _generation_record(
                    config,
                    stage1_config,
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
                                    "phase": "thinking_budget_generation",
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
        for _ in range(
            min(int(stage1_config.engine["max_in_flight_requests"]), len(requests))
        )
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        engine.shutdown()
        gc.collect()
    return persisted


def _generation_record(
    config: ThinkingBudgetScalingConfig,
    stage1_config: NativeThinkingConfig,
    tokenizer: Any,
    request: ScalingRequest,
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
        str(stage1_config.engine["reasoning_parser"])
    )
    parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
    parser_request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": request.selected.user_message}],
        max_tokens=request.total_output_ceiling,
        thinking_token_budget=request.max_reasoning_tokens,
        temperature=float(config.sampling["temperature"]),
        top_p=float(config.sampling["top_p"]),
        include_reasoning=True,
    )
    reasoning, final_content = parser.extract_reasoning(raw_text, parser_request)
    normalized_final, normalization_applied = lean_wrapper_normalization_v1(
        final_content
    )
    normalized_replay, _ = lean_wrapper_normalization_v1(normalized_final)
    reasoning_token_count = int(parser.count_reasoning_tokens(token_ids))
    final_token_count = (
        0 if final_content is None else len(parser.extract_content_ids(token_ids))
    )
    parsed_final_token_count = (
        0
        if final_content is None
        else len(tokenizer.encode(final_content, add_special_tokens=False))
    )
    normalized_final_token_count = (
        0
        if normalized_final is None
        else len(tokenizer.encode(normalized_final, add_special_tokens=False))
    )
    if reasoning_token_count > request.max_reasoning_tokens:
        raise RuntimeError(
            f"thinking budget exceeded for {request.candidate_id}: "
            f"{reasoning_token_count} > {request.max_reasoning_tokens}"
        )
    if len(token_ids) > request.total_output_ceiling:
        raise RuntimeError(f"total output ceiling exceeded: {request.candidate_id}")
    end_marker = str(parser.reasoning_end_str)
    start_marker = str(parser.reasoning_start_str)
    end_token_ids = tokenizer.encode(end_marker, add_special_tokens=False)
    end_positions = _sequence_positions(token_ids, end_token_ids)
    first_end_position = (
        None if not end_positions else end_positions[0] + len(end_token_ids)
    )
    final_has_marker = bool(
        final_content
        and (
            (start_marker and start_marker in final_content)
            or (end_marker and end_marker in final_content)
        )
    )
    identity = {
        "arm": request.arm,
        "enable_thinking": True,
        "workload": request.selected.task.workload,
        "task_id": request.selected.task.task_id,
        "prompt_sha256": request.selected.user_message_sha256,
        "candidate_index": request.candidate_index,
        "seed": request.candidate_seed,
        "model_revision": MODEL_REVISION,
        "max_reasoning_tokens": request.max_reasoning_tokens,
        "total_output_ceiling": request.total_output_ceiling,
        "generation_config_sha256": request.generation_config_sha256,
    }
    prefix = (
        "thinking-budget-gate-"
        if request.is_runtime_gate
        else "thinking-budget-scaling-"
    )
    if request.candidate_id != prefix + _sha256_json(identity)[:32]:
        raise AssertionError("Stage 2 candidate identity changed during generation")
    raw_finish_reason = (
        None if completion.finish_reason is None else str(completion.finish_reason)
    )
    return {
        "schema_version": GENERATION_SCHEMA,
        "candidate_id": request.candidate_id,
        **identity,
        "is_runtime_gate": request.is_runtime_gate,
        "intuition_sha256": request.selected.task.intuition_sha256,
        "theorem_sha256": request.selected.task.theorem_sha256,
        "rendered_prompt_sha256": request.selected.rendered_prompt_sha256,
        "rendered_prompt_token_count": request.selected.rendered_prompt_token_count,
        "raw_response_text": raw_text,
        "raw_response_sha256": _sha256_text(raw_text),
        "raw_response_token_ids": token_ids,
        "raw_response_token_ids_sha256": _sha256_json(token_ids),
        "raw_response_token_count": len(token_ids),
        "reasoning_content": reasoning,
        "reasoning_content_sha256": (
            None if reasoning is None else _sha256_text(reasoning)
        ),
        "reasoning_token_count": reasoning_token_count,
        "reasoning_exit": reasoning_exit_category(
            reasoning_token_count,
            request.max_reasoning_tokens,
            final_content,
            reasoning_end_position_token_count=first_end_position,
        ),
        "reasoning_end_marker_token_positions": end_positions,
        "reasoning_end_position_token_count": first_end_position,
        "final_content": final_content,
        "final_content_sha256": (
            None if final_content is None else _sha256_text(final_content)
        ),
        "final_token_count": final_token_count,
        "parsed_final_exact": final_content,
        "parsed_final_sha256": (
            None if final_content is None else _sha256_text(final_content)
        ),
        "parsed_final_token_count": parsed_final_token_count,
        "normalized_final_exact": normalized_final,
        "normalized_final_sha256": (
            None if normalized_final is None else _sha256_text(normalized_final)
        ),
        "normalized_final_token_count": normalized_final_token_count,
        "normalization_id": LEAN_WRAPPER_NORMALIZATION,
        "normalization_applied": normalization_applied,
        "normalization_pass_count": 1,
        "normalization_idempotent": normalized_replay == normalized_final,
        "parser_final_content_is_exact_raw_suffix": (
            final_content is None or raw_text.endswith(final_content)
        ),
        "raw_has_reasoning_end_marker": bool(end_marker and end_marker in raw_text),
        "final_has_reasoning_marker": final_has_marker,
        "finish_reason": _finish_reason(raw_finish_reason),
        "raw_finish_reason": raw_finish_reason,
        "generation_latency_seconds": latency_seconds,
        "request_id": request.candidate_id,
    }


def load_scaling_generation_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    seen: set[str] = set()
    for record in records:
        if record.get("schema_version") != GENERATION_SCHEMA:
            raise ValueError("unknown thinking-budget generation schema")
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate thinking-budget generation: {candidate_id}")
        seen.add(candidate_id)
        if _sha256_text(str(record["raw_response_text"])) != record.get(
            "raw_response_sha256"
        ):
            raise ValueError(f"raw response hash mismatch: {candidate_id}")
        token_ids = [int(value) for value in record["raw_response_token_ids"]]
        if _sha256_json(token_ids) != record.get("raw_response_token_ids_sha256"):
            raise ValueError(f"raw token-id hash mismatch: {candidate_id}")
        if len(token_ids) != int(record["raw_response_token_count"]):
            raise ValueError(f"raw token count mismatch: {candidate_id}")
        final = record.get("final_content")
        if final is not None and _sha256_text(str(final)) != record.get(
            "final_content_sha256"
        ):
            raise ValueError(f"final response hash mismatch: {candidate_id}")
        identity = {
            key: record[key]
            for key in (
                "arm",
                "enable_thinking",
                "workload",
                "task_id",
                "prompt_sha256",
                "candidate_index",
                "seed",
                "model_revision",
                "max_reasoning_tokens",
                "total_output_ceiling",
                "generation_config_sha256",
            )
        }
        prefix = (
            "thinking-budget-gate-"
            if record.get("is_runtime_gate")
            else "thinking-budget-scaling-"
        )
        if candidate_id != prefix + _sha256_json(identity)[:32]:
            raise ValueError(f"candidate identity mismatch: {candidate_id}")
    return records


def _compact_generation(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "arm",
        "workload",
        "task_id",
        "seed",
        "max_reasoning_tokens",
        "total_output_ceiling",
        "raw_response_sha256",
        "raw_response_token_ids_sha256",
        "raw_response_token_count",
        "reasoning_content_sha256",
        "reasoning_token_count",
        "reasoning_exit_audit",
        "reasoning_end_marker_token_positions_audit",
        "reasoning_end_position_token_count_audit",
        "final_content_sha256",
        "final_token_count",
        "parser_final_content_is_exact_raw_suffix",
        "raw_has_reasoning_end_marker",
        "final_has_reasoning_marker",
        "finish_reason",
        "generation_latency_seconds",
    )
    return {key: row[key] for key in keys}


def _audit_probe_record(record: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    from vllm.reasoning import ReasoningParserManager

    parser_class = ReasoningParserManager.get_reasoning_parser("qwen3")
    parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
    end_token_ids = tokenizer.encode(
        str(parser.reasoning_end_str), add_special_tokens=False
    )
    positions = _sequence_positions(
        [int(value) for value in record["raw_response_token_ids"]], end_token_ids
    )
    first_end_position = None if not positions else positions[0] + len(end_token_ids)
    audited = dict(record)
    audited.update(
        {
            "reasoning_end_marker_token_positions_audit": positions,
            "reasoning_end_position_token_count_audit": first_end_position,
            "reasoning_exit_audit": reasoning_exit_category(
                int(record["reasoning_token_count"]),
                int(record["max_reasoning_tokens"]),
                record.get("final_content"),
                reasoning_end_position_token_count=first_end_position,
            ),
        }
    )
    return audited


def _sequence_positions(values: Sequence[int], target: Sequence[int]) -> list[int]:
    if not target:
        return []
    return [
        index
        for index in range(len(values) - len(target) + 1)
        if list(values[index : index + len(target)]) == list(target)
    ]


def _repository_relative_path(path: Path) -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(repository_root))
    except ValueError:
        return path.name


def _write_gate_evidence(output_path: Path, evidence: Mapping[str, Any]) -> None:
    _atomic_write_json(output_path, dict(evidence))
    output_path.with_name("README.md").write_text(
        _render_gate_readme(evidence), encoding="utf-8"
    )


def _render_gate_readme(evidence: Mapping[str, Any]) -> str:
    if evidence.get("status") == "passed":
        finding = (
            "All bounded probes preserved an exact reasoning-to-final split and "
            "the 48-generation scaling probe is authorized."
        )
    else:
        finding = (
            "All three probes bounded reasoning and continued into a non-empty "
            "final channel. The B32 probe emitted a duplicate reasoning-end marker; "
            "the pinned `qwen3` parser absorbed it but returned final content with "
            "one additional leading newline, so the final was not an exact raw "
            "suffix. This violates the Stage 2a parser-integrity gate."
        )
    return "\n".join(
        [
            "# Qwen3.5-4B thinking-budget runtime gate",
            "",
            (
                "**OBSERVED:** the exact Stage 1 runtime exposes native "
                "`thinking_token_budget` with the frozen `qwen3` parser."
            ),
            "",
            finding,
            "",
            f"**OBSERVED conclusion:** `{evidence['conclusion']}`.",
            "",
            (
                "The outcome-bearing 48-generation B4/B8/B16 scaling probe was not "
                "started. The durable probe JSONL was not regenerated or rewritten, "
                "and the independently reviewed Stage 1 target remains unchanged. "
                "A runtime upgrade or custom two-stage forcing mechanism requires an "
                "explicit bounded amendment before further scientific generation."
            ),
            "",
        ]
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
