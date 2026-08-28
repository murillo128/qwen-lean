"""Full-context diagnostic forks for the frozen failed T1 trajectory.

This module deliberately writes only to its own artifact and evidence paths.  The
reviewed issue-92 result is an immutable comparison input, never an output.
"""

from __future__ import annotations

import asyncio
import gc
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import _GpuMemoryMonitor
from .counterfactual_forking_assessment import (
    EXPECTED_RUNTIME_VERSIONS,
    MATHEMATICAL_VERIFIER_CATEGORIES,
    CounterfactualForkingConfig,
    ForkRequest,
    ParentTrajectory,
    _assert_no_other_compute_process,
    _configure_fork_runtime,
    _fork_generation_record,
    _fork_local_runtime,
    _restart_safe_jsonl_lines,
    _validate_fork_generation_record,
    _validated_package_versions,
    _verify_fork_generation_record,
    latest_verifications,
    load_fork_verification_records,
    materialize_parent_trajectories,
)
from .native_thinking_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    _append_jsonl,
    _atomic_write_json,
    _file_sha256,
    _load_tokenizer,
    _resolve_model_snapshot,
    _sha256_json,
    load_mathia_tasks,
    validate_lean_environments,
)
from .verifier import LeanVerifier

CONFIG_SCHEMA = "qwen35-full-context-forking-config-v1"
CALIBRATION_PROBE_SCHEMA = "qwen35-full-context-calibration-probe-v1"
CALIBRATION_EVENT_SCHEMA = "qwen35-full-context-calibration-event-v1"
CALIBRATION_EVIDENCE_SCHEMA = "qwen35-full-context-calibration-v1"
CHECKPOINT_REVIEW_SCHEMA = "qwen35-full-context-calibration-review-v1"
GENERATION_SEGMENT_SCHEMA = "qwen35-full-context-generation-segment-v1"
BRANCH_ATTEMPT_SCHEMA = "qwen35-full-context-branch-attempt-v1"
FINAL_EVIDENCE_SCHEMA = "qwen35-full-context-forking-results-v1"
REVIEWED_TARGET_COMMIT = "6f6838041ef517518d3fcfa68889ea2988074c83"
HANDOFF_COMMIT = "bcd72d5203d82e27d50e42ec6d2d2afa061c2504"
COUNTERFACTUAL_RESULTS_SHA256 = (
    "844ad2f31503c995a78498812e778932966767907b3e7d8b8b135f4dc4aab56d"
)
TARGET_WORKLOAD = "minif2f-valid-clean-v2"
TARGET_TASK_ID = "amc12a_2019_p21"
TARGET_PARENT_ID = "native-thinking-752f2a6728b47b5a89263445496c7b80"
SCIENTIFIC_STATES = ("P0", "P15", "P30", "P45", "P60", "P75", "P90")
SCIENTIFIC_SEEDS = tuple(range(100, 106))
KNOWN_CONTEXT_LENGTH = 24_576
NATIVE_CONTEXT_LENGTH = 262_144
GPU_MEMORY_UTILIZATION = 0.89


@dataclass(frozen=True)
class FullContextForkingConfig:
    path: Path
    value: dict[str, Any]
    counterfactual: CounterfactualForkingConfig
    repository_root: Path

    @classmethod
    def load(cls, path: Path) -> FullContextForkingConfig:
        resolved = path.resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        repository_root = resolved.parents[1]
        counterfactual = CounterfactualForkingConfig.load(
            repository_root / str(value.get("counterfactual_config_path", ""))
        )
        config = cls(resolved, value, counterfactual, repository_root)
        validate_full_context_config(config)
        return config

    @property
    def reviewed_target(self) -> dict[str, Any]:
        return self.value["reviewed_target"]

    @property
    def diagnostic_target(self) -> dict[str, Any]:
        return self.value["diagnostic_target"]

    @property
    def calibration(self) -> dict[str, Any]:
        return self.value["calibration"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.value["execution"]

    @property
    def branches(self) -> dict[str, Any]:
        return self.value["branches"]

    @property
    def classification(self) -> dict[str, Any]:
        return self.value["classification"]


def validate_full_context_config(config: FullContextForkingConfig) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "counterfactual_config_path": "config/qwen35-counterfactual-forking.json",
        "reviewed_target": {
            "pr": 95,
            "commit": REVIEWED_TARGET_COMMIT,
            "handoff_commit": HANDOFF_COMMIT,
            "counterfactual_results_path": (
                "evidence/qwen35-counterfactual-forking/results.json"
            ),
            "counterfactual_results_sha256": COUNTERFACTUAL_RESULTS_SHA256,
        },
        "diagnostic_target": {
            "workload": TARGET_WORKLOAD,
            "task_id": TARGET_TASK_ID,
            "parent_candidate_id": TARGET_PARENT_ID,
            "parent_ordinal": 0,
        },
        "calibration": {
            "known_reviewed_context_length": KNOWN_CONTEXT_LENGTH,
            "native_context_length": NATIVE_CONTEXT_LENGTH,
            "progressive_context_lengths": [
                32_768,
                49_152,
                65_536,
                98_304,
                131_072,
                196_608,
                262_144,
            ],
            "refinement_granularity": 1024,
            "required_successes_at_selected_length": 2,
            "probe_state": "P90",
            "probe_max_new_tokens": 1,
            "seed_base": 9800,
        },
        "execution": {
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": 1,
            "max_in_flight_requests": 1,
        },
        "branches": {
            "states": list(SCIENTIFIC_STATES),
            "seeds": list(SCIENTIFIC_SEEDS),
            "expected_branch_count": 42,
        },
        "classification": {
            "multiple_branches_minimum": 2,
            "great_majority_numerator": 5,
            "great_majority_denominator": 6,
            "state_dependent_means_any_exact_rate_difference": True,
        },
    }
    if config.value != expected:
        raise ValueError("full-context diagnostic config differs from issue #98")
    if _file_sha256(
        config.repository_root
        / str(config.reviewed_target["counterfactual_results_path"])
    ) != str(config.reviewed_target["counterfactual_results_sha256"]):
        raise ValueError("reviewed issue-92 compact results changed")


def full_context_config_sha256(config: FullContextForkingConfig) -> str:
    return _sha256_json(config.value)


def _target_parent(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    tokenizer: Any,
    *,
    parent_release_package_path: Path | None,
) -> tuple[ParentTrajectory, dict[str, Any]]:
    parents, integrity = materialize_parent_trajectories(
        config.counterfactual,
        mathia_root,
        parent_generations_path,
        tokenizer,
        release_package_path=parent_release_package_path,
    )
    parent = parents[int(config.diagnostic_target["parent_ordinal"])]
    observed = {
        "workload": parent.task.workload,
        "task_id": parent.task.task_id,
        "parent_candidate_id": parent.handoff["candidate_id"],
        "parent_ordinal": parent.ordinal,
    }
    if observed != config.diagnostic_target:
        raise ValueError(f"issue-98 diagnostic target changed: {observed}")
    return parent, integrity


def _nvml_memory_snapshot(device_index: int) -> dict[str, int]:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_memory_total_bytes": int(info.total),
            "gpu_memory_free_bytes": int(info.free),
            "gpu_memory_used_bytes": int(info.used),
        }
    finally:
        pynvml.nvmlShutdown()


def _calibration_engine_args(
    config: FullContextForkingConfig, snapshot_path: Path, context_length: int
) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs

    native = config.counterfactual.native
    return AsyncEngineArgs(
        model=str(snapshot_path),
        tokenizer=str(snapshot_path),
        revision=str(native.model["model_revision"]),
        tokenizer_revision=str(native.model["tokenizer_revision"]),
        dtype=str(native.engine["dtype"]),
        tensor_parallel_size=int(native.engine["tensor_parallel_size"]),
        gpu_memory_utilization=float(config.execution["gpu_memory_utilization"]),
        max_model_len=context_length,
        max_num_seqs=int(config.execution["max_num_seqs"]),
        enforce_eager=bool(native.engine["enforce_eager"]),
        quantization=native.engine["quantization"],
        language_model_only=bool(native.engine["language_model_only"]),
        reasoning_parser=str(native.engine["reasoning_parser"]),
        generation_config="vllm",
        enable_log_requests=False,
        disable_log_stats=False,
    )


async def _run_calibration_continuation(
    config: FullContextForkingConfig,
    snapshot_path: Path,
    parent: ParentTrajectory,
    *,
    context_length: int,
    seed: int,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    states = {state.label: state for state in parent.states}
    state = states[str(config.calibration["probe_state"])]
    prompt_token_ids = (
        parent.rendered_prompt_token_ids
        + parent.raw_response_token_ids[: state.prefix_len]
    )
    if len(prompt_token_ids) >= context_length:
        raise RuntimeError("calibration probe input does not fit candidate context")
    engine = AsyncLLM.from_engine_args(
        _calibration_engine_args(config, snapshot_path, context_length)
    )
    try:
        params = SamplingParams(
            n=1,
            temperature=float(config.counterfactual.native.sampling["temperature"]),
            top_p=float(config.counterfactual.native.sampling["top_p"]),
            top_k=int(config.counterfactual.native.sampling["top_k"]),
            min_p=float(config.counterfactual.native.sampling["min_p"]),
            presence_penalty=float(
                config.counterfactual.native.sampling["presence_penalty"]
            ),
            repetition_penalty=float(
                config.counterfactual.native.sampling["repetition_penalty"]
            ),
            max_tokens=int(config.calibration["probe_max_new_tokens"]),
            seed=seed,
            skip_special_tokens=False,
        )
        last_output: Any | None = None
        request_id = f"issue98-calibration-{context_length}-{seed}"
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=list(prompt_token_ids)),
            params,
            request_id=request_id,
            reasoning_ended=False,
            reasoning_parser_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
        ):
            last_output = output
        if last_output is None or not last_output.finished:
            raise RuntimeError("calibration continuation did not finish")
        if len(last_output.outputs) != 1:
            raise RuntimeError("calibration continuation count changed")
        completion = last_output.outputs[0]
        token_ids = [int(value) for value in completion.token_ids]
        if not token_ids:
            raise RuntimeError("calibration continuation emitted no token ID")
        return {
            "request_id": request_id,
            "state": state.label,
            "fork_prefix_len": state.prefix_len,
            "prompt_token_count": len(prompt_token_ids),
            "prompt_token_ids_sha256": _sha256_json(list(prompt_token_ids)),
            "continuation_token_count": len(token_ids),
            "continuation_token_ids_sha256": _sha256_json(token_ids),
            "finished": True,
            "raw_finish_reason": (
                None
                if completion.finish_reason is None
                else str(completion.finish_reason)
            ),
        }
    finally:
        engine.shutdown()
        gc.collect()


def run_calibration_probe(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    output_path: Path,
    *,
    context_length: int,
    seed: int,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    if not KNOWN_CONTEXT_LENGTH <= context_length <= NATIVE_CONTEXT_LENGTH:
        raise ValueError("calibration context length is outside the frozen bounds")
    _configure_fork_runtime()
    snapshot_path = _resolve_model_snapshot(config.counterfactual.native)
    tokenizer = _load_tokenizer(
        config.counterfactual.native, snapshot_path=snapshot_path
    )
    parent, integrity = _target_parent(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        parent_release_package_path=parent_release_package_path,
    )
    runtime = _fork_local_runtime(config.counterfactual)
    device_index = int(runtime["cuda_device_index"])
    _assert_no_other_compute_process(device_index)
    versions = _validated_package_versions()
    before = _nvml_memory_snapshot(device_index)
    monitor = _GpuMemoryMonitor(device_index, required=True)
    monitor.start()
    started = time.perf_counter()
    status = "failed"
    continuation: dict[str, Any] | None = None
    error_text: str | None = None
    try:
        continuation = asyncio.run(
            _run_calibration_continuation(
                config,
                snapshot_path,
                parent,
                context_length=context_length,
                seed=seed,
            )
        )
        status = "passed"
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        memory = monitor.stop()
        result = {
            "schema_version": CALIBRATION_PROBE_SCHEMA,
            "status": status,
            "context_length": context_length,
            "seed": seed,
            "config_sha256": full_context_config_sha256(config),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_snapshot_revision": snapshot_path.name,
            "dtype": config.counterfactual.native.engine["dtype"],
            "reasoning_parser": config.counterfactual.native.engine["reasoning_parser"],
            "gpu_memory_utilization": config.execution["gpu_memory_utilization"],
            "max_num_seqs": config.execution["max_num_seqs"],
            "context_extension": None,
            "runtime": runtime,
            "package_versions": versions,
            "memory_before": before,
            "memory_observed": memory,
            "wall_time_seconds": time.perf_counter() - started,
            "continuation": continuation,
            "error": error_text,
            "parent_integrity": {
                "handoff_commit": integrity["handoff_commit"],
                "handoff_manifest_sha256": integrity["handoff_manifest_sha256"],
                "parser_parity_failures": integrity["parser_parity_failures"],
            },
        }
        _atomic_write_json(output_path, result)
    return result


def _completed_calibration_attempts(
    events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    starts: dict[int, dict[str, Any]] = {}
    completed: dict[int, dict[str, Any]] = {}
    for event in events:
        index = int(event["attempt_index"])
        if event["event"] == "started":
            if index in starts:
                raise ValueError(f"duplicate calibration start: {index}")
            starts[index] = event
        elif event["event"] in {"completed", "interrupted"}:
            if index not in starts or index in completed:
                raise ValueError(f"invalid calibration terminal event: {index}")
            completed[index] = event
        else:
            raise ValueError("unknown calibration event")
    return [
        completed[index]
        for index in sorted(completed)
        if completed[index]["event"] == "completed"
    ]


def _next_calibration_action(
    config: FullContextForkingConfig,
    attempts: Sequence[dict[str, Any]],
) -> tuple[str, int] | None:
    successes: Counter[int] = Counter()
    failed: set[int] = set()
    for attempt in attempts:
        length = int(attempt["context_length"])
        if attempt["status"] == "passed":
            successes[length] += 1
        else:
            failed.add(length)

    progressive = [
        int(value) for value in config.calibration["progressive_context_lengths"]
    ]
    encountered_failure = False
    for length in progressive:
        if length in failed:
            encountered_failure = True
            break
        if successes[length] == 0:
            return "progressive", length
    if not encountered_failure and all(successes[length] for length in progressive):
        selected = progressive[-1]
    else:
        clean_passes = [
            length
            for length, count in successes.items()
            if count and length not in failed
        ]
        lower = max([KNOWN_CONTEXT_LENGTH, *clean_passes])
        upper_candidates = [length for length in failed if length > lower]
        if not upper_candidates:
            raise RuntimeError("calibration journal has no valid failure bracket")
        upper = min(upper_candidates)
        granularity = int(config.calibration["refinement_granularity"])
        while upper - lower > granularity:
            steps = (upper - lower) // granularity
            candidate = lower + max(1, steps // 2) * granularity
            if candidate >= upper:
                candidate = upper - granularity
            if candidate in failed:
                upper = candidate
            elif successes[candidate]:
                lower = candidate
            else:
                return "refinement", candidate
        selected = lower

    required = int(config.calibration["required_successes_at_selected_length"])
    if selected in failed:
        raise AssertionError("failed context length cannot be selected")
    if successes[selected] < required:
        return "confirmation", selected
    return None


def _read_calibration_events(
    path: Path, config: FullContextForkingConfig
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in _restart_safe_jsonl_lines(path):
        event = json.loads(line)
        if event.get("schema_version") != CALIBRATION_EVENT_SCHEMA:
            raise ValueError("unknown calibration journal schema")
        if event.get("config_sha256") != full_context_config_sha256(config):
            raise ValueError("calibration journal config binding changed")
        if int(event.get("attempt_index", -1)) < 0:
            raise ValueError("invalid calibration attempt index")
        events.append(event)
    _completed_calibration_attempts(events)
    return events


def _calibration_failure_diagnostic(
    probes_dir: Path, attempt: Mapping[str, Any]
) -> dict[str, Any]:
    stderr_path = probes_dir / f"attempt-{int(attempt['attempt_index']):03d}.stderr.log"
    if not stderr_path.exists():
        return {"stderr_log_sha256": None, "failure_category": None}
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    estimated_match = re.search(r"estimated maximum model length is (\d+)", stderr_text)
    if attempt["status"] == "passed":
        failure_category = None
    elif "larger than the available KV cache memory" in stderr_text:
        failure_category = "insufficient_kv_cache_capacity"
    elif "out of memory" in stderr_text.lower():
        failure_category = "cuda_out_of_memory"
    else:
        failure_category = "engine_initialization_failure"
    return {
        "stderr_log_sha256": _file_sha256(stderr_path),
        "failure_category": failure_category,
        "vllm_estimated_max_model_length": (
            int(estimated_match.group(1)) if estimated_match else None
        ),
    }


def _compact_calibration_evidence(
    config: FullContextForkingConfig,
    events: Sequence[dict[str, Any]],
    probes_dir: Path,
) -> dict[str, Any]:
    attempts = _completed_calibration_attempts(events)
    if _next_calibration_action(config, attempts) is not None:
        raise RuntimeError("calibration is not complete")
    passed = [attempt for attempt in attempts if attempt["status"] == "passed"]
    failed_lengths = {
        int(attempt["context_length"])
        for attempt in attempts
        if attempt["status"] != "passed"
    }
    selected = max(
        int(attempt["context_length"])
        for attempt in passed
        if int(attempt["context_length"]) not in failed_lengths
    )
    required = int(config.calibration["required_successes_at_selected_length"])
    selected_passes = sum(
        1 for attempt in passed if int(attempt["context_length"]) == selected
    )
    if selected_passes < required:
        raise AssertionError("selected calibration length lacks confirmations")
    terminal_attempts = {
        int(event["attempt_index"])
        for event in events
        if event["event"] in {"completed", "interrupted"}
    }
    explicitly_interrupted = [
        event for event in events if event["event"] == "interrupted"
    ]
    incomplete_started = [
        event
        for event in events
        if event["event"] == "started"
        and int(event["attempt_index"]) not in terminal_attempts
    ]
    compact_attempts = []
    for attempt in attempts:
        compact_attempts.append(
            {
                **{
                    key: attempt.get(key)
                    for key in (
                        "attempt_index",
                        "stage",
                        "context_length",
                        "seed",
                        "status",
                        "returncode",
                        "engine_initialized_and_real_continuation_finished",
                        "gpu_memory_total_bytes",
                        "gpu_memory_free_before_bytes",
                        "gpu_memory_peak_bytes",
                        "gpu_memory_peak_delta_bytes",
                        "wall_time_seconds",
                        "probe_result_sha256",
                        "error",
                    )
                },
                **_calibration_failure_diagnostic(probes_dir, attempt),
            }
        )
    return {
        "schema_version": CALIBRATION_EVIDENCE_SCHEMA,
        "status": "passed",
        "config_sha256": full_context_config_sha256(config),
        "reviewed_target": config.reviewed_target,
        "diagnostic_target": config.diagnostic_target,
        "calibration_method": {
            "search": "progressive_then_binary_refinement",
            "known_reviewed_context_length": KNOWN_CONTEXT_LENGTH,
            "native_context_length": NATIVE_CONTEXT_LENGTH,
            "refinement_granularity": config.calibration["refinement_granularity"],
            "passing_probe_requires": (
                "engine initialization and finished real direct-token-ID continuation"
            ),
            "required_successes_at_selected_length": required,
            "probe_max_new_tokens": config.calibration["probe_max_new_tokens"],
            "probe_identities_are_excluded_from_scientific_branches": True,
            "context_extension": None,
        },
        "selected_max_context_length": selected,
        "selected_length_success_count": selected_passes,
        "attempts": compact_attempts,
        "interrupted_attempt_count": len(explicitly_interrupted)
        + len(incomplete_started),
        "incomplete_started_attempt_count": len(incomplete_started),
        "runtime_contract": {
            "inference_execution": "project_controlled_local_cuda",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dtype": config.counterfactual.native.engine["dtype"],
            "reasoning_parser": config.counterfactual.native.engine["reasoning_parser"],
            "gpu_memory_utilization": config.execution["gpu_memory_utilization"],
            "max_num_seqs": config.execution["max_num_seqs"],
            "max_in_flight_requests": config.execution["max_in_flight_requests"],
            "package_versions": EXPECTED_RUNTIME_VERSIONS,
        },
    }


def run_context_calibration(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    evidence_path: Path,
    *,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = artifact_dir / "calibration-probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    journal_path = artifact_dir / "calibration-events.jsonl"
    events = _read_calibration_events(journal_path, config)
    while True:
        attempts = _completed_calibration_attempts(events)
        action = _next_calibration_action(config, attempts)
        if action is None:
            evidence = _compact_calibration_evidence(config, events, probes_dir)
            _atomic_write_json(evidence_path, evidence)
            return evidence
        stage, context_length = action
        attempt_index = 1 + max(
            (int(event["attempt_index"]) for event in events), default=-1
        )
        seed = int(config.calibration["seed_base"]) + attempt_index
        base_event = {
            "schema_version": CALIBRATION_EVENT_SCHEMA,
            "attempt_index": attempt_index,
            "stage": stage,
            "context_length": context_length,
            "seed": seed,
            "config_sha256": full_context_config_sha256(config),
        }
        started_event = {**base_event, "event": "started"}
        _append_jsonl(journal_path, started_event)
        events.append(started_event)
        result_path = probes_dir / f"attempt-{attempt_index:03d}.json"
        stdout_path = probes_dir / f"attempt-{attempt_index:03d}.stdout.log"
        stderr_path = probes_dir / f"attempt-{attempt_index:03d}.stderr.log"
        command = [
            sys.executable,
            "-c",
            "from qwen_lean.cli import main; raise SystemExit(main())",
            "qwen35-full-context-calibration-probe",
            "--config",
            str(config.path),
            "--mathia-root",
            str(mathia_root),
            "--parent-generations",
            str(parent_generations_path),
            "--context-length",
            str(context_length),
            "--seed",
            str(seed),
            "--output",
            str(result_path),
        ]
        if parent_release_package_path is not None:
            command.extend(["--parent-package", str(parent_release_package_path)])
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                completed_process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False
                )
        except BaseException as error:
            interrupted_event = {
                **base_event,
                "event": "interrupted",
                "status": "interrupted",
                "error": f"{type(error).__name__}: {error}",
            }
            _append_jsonl(journal_path, interrupted_event)
            events.append(interrupted_event)
            raise
        probe_result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        )
        passed = bool(
            completed_process.returncode == 0
            and probe_result is not None
            and probe_result.get("status") == "passed"
            and probe_result.get("continuation", {}).get("finished") is True
        )
        memory_before = (probe_result or {}).get("memory_before", {})
        memory_observed = (probe_result or {}).get("memory_observed", {})
        completed_event = {
            **base_event,
            "event": "completed",
            "status": "passed" if passed else "failed",
            "returncode": completed_process.returncode,
            "engine_initialized_and_real_continuation_finished": passed,
            "gpu_memory_total_bytes": memory_before.get("gpu_memory_total_bytes"),
            "gpu_memory_free_before_bytes": memory_before.get("gpu_memory_free_bytes"),
            "gpu_memory_peak_bytes": memory_observed.get("gpu_memory_peak_bytes"),
            "gpu_memory_peak_delta_bytes": memory_observed.get(
                "gpu_memory_peak_delta_bytes"
            ),
            "wall_time_seconds": (probe_result or {}).get("wall_time_seconds"),
            "probe_result_sha256": (
                _file_sha256(result_path) if result_path.exists() else None
            ),
            "error": (probe_result or {}).get("error")
            or (
                f"probe process exited {completed_process.returncode}"
                if completed_process.returncode
                else None
            ),
        }
        _append_jsonl(journal_path, completed_event)
        events.append(completed_event)
        print(
            json.dumps(
                {
                    "phase": "context_calibration",
                    "attempt_index": attempt_index,
                    "stage": stage,
                    "context_length": context_length,
                    "status": completed_event["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _load_calibration_evidence(
    config: FullContextForkingConfig, path: Path
) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != CALIBRATION_EVIDENCE_SCHEMA:
        raise ValueError("unknown calibration evidence schema")
    if evidence.get("status") != "passed":
        raise ValueError("context calibration did not pass")
    if evidence.get("config_sha256") != full_context_config_sha256(config):
        raise ValueError("calibration evidence config binding changed")
    selected = int(evidence.get("selected_max_context_length", 0))
    if not KNOWN_CONTEXT_LENGTH <= selected <= NATIVE_CONTEXT_LENGTH:
        raise ValueError("selected calibrated context is outside frozen bounds")
    required = int(config.calibration["required_successes_at_selected_length"])
    if int(evidence.get("selected_length_success_count", 0)) < required:
        raise ValueError("selected context lacks repeat confirmation")
    return evidence


def _validate_checkpoint_review(
    calibration_evidence_path: Path, review_path: Path
) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema_version") != CHECKPOINT_REVIEW_SCHEMA:
        raise ValueError("unknown calibration checkpoint review schema")
    if review.get("verdict") != "PASS":
        raise RuntimeError("calibration checkpoint review is not PASS")
    if review.get("calibration_evidence_sha256") != _file_sha256(
        calibration_evidence_path
    ):
        raise ValueError("checkpoint review targets different calibration evidence")
    reviewed_commit = str(review.get("reviewed_commit", ""))
    if len(reviewed_commit) != 40:
        raise ValueError("checkpoint review lacks an exact commit target")
    if not review.get("published_review_url"):
        raise ValueError("checkpoint review was not published")
    return review


def _scientific_generation_hash(
    config: FullContextForkingConfig,
    calibration_evidence_path: Path,
    selected_context_length: int,
) -> str:
    return _sha256_json(
        {
            "config_sha256": full_context_config_sha256(config),
            "calibration_evidence_sha256": _file_sha256(calibration_evidence_path),
            "selected_max_context_length": selected_context_length,
        }
    )


def full_context_requests(
    config: FullContextForkingConfig,
    parent: ParentTrajectory,
    *,
    selected_context_length: int,
    calibration_evidence_path: Path,
) -> list[ForkRequest]:
    generation_hash = _scientific_generation_hash(
        config, calibration_evidence_path, selected_context_length
    )
    states = {state.label: state for state in parent.states}
    requests: list[ForkRequest] = []
    rendered_prompt_count = len(parent.rendered_prompt_token_ids)
    for state_label in SCIENTIFIC_STATES:
        state = states[state_label]
        max_tokens = selected_context_length - rendered_prompt_count - state.prefix_len
        if max_tokens < 1:
            raise ValueError(f"no continuation budget remains for {state_label}")
        for seed in SCIENTIFIC_SEEDS:
            identity = {
                "phase": "full_context",
                "parent_candidate_id": parent.handoff["candidate_id"],
                "workload": parent.task.workload,
                "task_id": parent.task.task_id,
                "fork_state": state.label,
                "fork_fraction": state.fraction,
                "fork_prefix_len": state.prefix_len,
                "branch_seed": seed,
                "max_tokens": max_tokens,
                "model_revision": MODEL_REVISION,
                "fork_generation_config_sha256": generation_hash,
                "interval_id": None,
                "interval_side": None,
            }
            branch_id = "full-context-fork-" + _sha256_json(identity)[:32]
            prefix = parent.raw_response_token_ids[: state.prefix_len]
            requests.append(
                ForkRequest(
                    phase="full_context",
                    parent=parent,
                    state=state,
                    seed=seed,
                    max_tokens=max_tokens,
                    branch_id=branch_id,
                    generation_config_sha256=generation_hash,
                    fork_prompt_token_ids=parent.rendered_prompt_token_ids + prefix,
                )
            )
    if len(requests) != int(config.branches["expected_branch_count"]):
        raise AssertionError("full-context scientific branch count changed")
    if len({request.branch_id for request in requests}) != len(requests):
        raise AssertionError("full-context branch identities are not unique")
    return requests


def _full_context_engine_args(
    config: FullContextForkingConfig,
    snapshot_path: Path,
    selected_context_length: int,
) -> Any:
    return _calibration_engine_args(config, snapshot_path, selected_context_length)


def _reasoning_transition_index(
    tokenizer: Any, combined_ids: Sequence[int]
) -> int | None:
    from vllm.reasoning import ReasoningParserManager

    parser_class = ReasoningParserManager.get_reasoning_parser("qwen3")
    parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
    end_token_id = int(parser.end_token_id)
    try:
        return list(combined_ids).index(end_token_id) + 1
    except ValueError:
        return None


def _enrich_full_context_record(
    record: dict[str, Any],
    request: ForkRequest,
    tokenizer: Any,
    *,
    selected_context_length: int,
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    combined_ids = [
        *request.parent.raw_response_token_ids[: request.state.prefix_len],
        *(int(value) for value in record["suffix_response_token_ids"]),
    ]
    record.update(
        {
            "diagnostic_kind": "scoring_excluded_full_context",
            "selected_max_context_length": selected_context_length,
            "max_new_tokens_formula": "M - rendered_prompt_tokens - frozen_prefix_tokens",
            "max_new_tokens_rendered_prompt_count": len(
                request.parent.rendered_prompt_token_ids
            ),
            "max_new_tokens_frozen_prefix_count": request.state.prefix_len,
            "total_generated_token_count": record["suffix_response_token_count"],
            "total_context_consumed_token_count": len(request.fork_prompt_token_ids)
            + int(record["suffix_response_token_count"]),
            "reasoning_to_final_transition_token_index": _reasoning_transition_index(
                tokenizer, combined_ids
            ),
            "reasoning_to_final_transition_index_semantics": (
                "zero-based combined-response index of first token after </think>"
            ),
            "branch_gpu_memory": dict(memory),
        }
    )
    _validate_full_context_generation_record(
        record, request, selected_context_length=selected_context_length
    )
    return record


def _validate_full_context_generation_record(
    record: dict[str, Any],
    request: ForkRequest,
    *,
    selected_context_length: int,
) -> None:
    _validate_fork_generation_record(record, request)
    expected = {
        "diagnostic_kind": "scoring_excluded_full_context",
        "selected_max_context_length": selected_context_length,
        "max_new_tokens_rendered_prompt_count": len(
            request.parent.rendered_prompt_token_ids
        ),
        "max_new_tokens_frozen_prefix_count": request.state.prefix_len,
        "total_generated_token_count": record.get("suffix_response_token_count"),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"full-context record field changed for {request.branch_id}: {field}"
            )
    if request.max_tokens != (
        selected_context_length
        - len(request.parent.rendered_prompt_token_ids)
        - request.state.prefix_len
    ):
        raise ValueError("full-context max-new-token formula changed")
    generated = int(record["suffix_response_token_count"])
    if int(record.get("total_context_consumed_token_count", -1)) != (
        len(request.fork_prompt_token_ids) + generated
    ):
        raise ValueError("full-context consumed-token count changed")
    transition = record.get("reasoning_to_final_transition_token_index")
    combined_count = int(record["combined_response_token_count"])
    if transition is not None and not 1 <= int(transition) <= combined_count:
        raise ValueError("invalid reasoning-to-final transition index")
    memory = record.get("branch_gpu_memory")
    if not isinstance(memory, dict) or memory.get("gpu_memory_peak_bytes") is None:
        raise ValueError("full-context branch lacks measured peak GPU memory")


def load_full_context_generation_records(
    path: Path,
    expected_requests: Sequence[ForkRequest],
    *,
    selected_context_length: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    expected = {request.branch_id: request for request in expected_requests}
    records_by_id: dict[str, dict[str, Any]] = {}
    for line in _restart_safe_jsonl_lines(path):
        record = json.loads(line)
        branch_id = str(record.get("branch_id"))
        request = expected.get(branch_id)
        if request is None:
            raise ValueError(f"unknown persisted full-context branch: {branch_id}")
        if branch_id in records_by_id:
            raise ValueError(f"duplicate persisted full-context branch: {branch_id}")
        _validate_full_context_generation_record(
            record, request, selected_context_length=selected_context_length
        )
        records_by_id[branch_id] = record
    return [
        records_by_id[request.branch_id]
        for request in expected_requests
        if request.branch_id in records_by_id
    ]


def _branch_attempt_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    for line in _restart_safe_jsonl_lines(path):
        event = json.loads(line)
        if event.get("schema_version") != BRANCH_ATTEMPT_SCHEMA:
            raise ValueError("unknown branch-attempt schema")
        if event.get("event") == "started":
            counts[str(event["branch_id"])] += 1
    return counts


async def _run_full_context_branches(
    config: FullContextForkingConfig,
    tokenizer: Any,
    pending: Sequence[ForkRequest],
    generation_path: Path,
    attempt_path: Path,
    snapshot_path: Path,
    *,
    selected_context_length: int,
    device_index: int,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(
        _full_context_engine_args(config, snapshot_path, selected_context_length)
    )
    persisted: list[dict[str, Any]] = []
    attempt_counts = _branch_attempt_counts(attempt_path)
    try:
        for request in pending:
            attempt_index = attempt_counts[request.branch_id]
            started_event = {
                "schema_version": BRANCH_ATTEMPT_SCHEMA,
                "event": "started",
                "branch_id": request.branch_id,
                "attempt_index": attempt_index,
                "fork_state": request.state.label,
                "branch_seed": request.seed,
                "max_tokens": request.max_tokens,
            }
            _append_jsonl(attempt_path, started_event)
            attempt_counts[request.branch_id] += 1
            monitor = _GpuMemoryMonitor(device_index, required=True)
            monitor.start()
            started = time.perf_counter()
            try:
                params = SamplingParams(
                    n=1,
                    temperature=float(
                        config.counterfactual.native.sampling["temperature"]
                    ),
                    top_p=float(config.counterfactual.native.sampling["top_p"]),
                    top_k=int(config.counterfactual.native.sampling["top_k"]),
                    min_p=float(config.counterfactual.native.sampling["min_p"]),
                    presence_penalty=float(
                        config.counterfactual.native.sampling["presence_penalty"]
                    ),
                    repetition_penalty=float(
                        config.counterfactual.native.sampling["repetition_penalty"]
                    ),
                    max_tokens=request.max_tokens,
                    seed=request.seed,
                    skip_special_tokens=False,
                )
                last_output: Any | None = None
                async for output in engine.generate(
                    TokensPrompt(prompt_token_ids=list(request.fork_prompt_token_ids)),
                    params,
                    request_id=request.branch_id,
                    reasoning_ended=False,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": {"enable_thinking": True}
                    },
                ):
                    last_output = output
                if last_output is None or not last_output.finished:
                    raise RuntimeError(
                        f"full-context branch did not finish: {request.branch_id}"
                    )
                if len(last_output.outputs) != 1:
                    raise RuntimeError("unexpected full-context completion count")
                memory = monitor.stop()
                record = _fork_generation_record(
                    config.counterfactual,
                    tokenizer,
                    request,
                    last_output.outputs[0],
                    latency_seconds=time.perf_counter() - started,
                )
                _enrich_full_context_record(
                    record,
                    request,
                    tokenizer,
                    selected_context_length=selected_context_length,
                    memory=memory,
                )
                _append_jsonl(generation_path, record)
                _append_jsonl(
                    attempt_path,
                    {
                        **started_event,
                        "event": "persisted",
                        "generation_record_sha256": _sha256_json(record),
                    },
                )
                persisted.append(record)
                print(
                    json.dumps(
                        {
                            "phase": "full_context_generation",
                            "completed_in_segment": len(persisted),
                            "pending_in_segment": len(pending),
                            "branch_id": request.branch_id,
                            "fork_state": request.state.label,
                            "branch_seed": request.seed,
                            "generated_tokens": record["suffix_response_token_count"],
                            "final_production_status": record[
                                "final_production_status"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except BaseException as error:
                memory = monitor.stop()
                _append_jsonl(
                    attempt_path,
                    {
                        **started_event,
                        "event": "interrupted"
                        if isinstance(
                            error, (KeyboardInterrupt, asyncio.CancelledError)
                        )
                        else "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "branch_gpu_memory": memory,
                    },
                )
                raise
    finally:
        engine.shutdown()
        gc.collect()
    return persisted


def run_full_context_generation(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    calibration_evidence_path: Path,
    checkpoint_review_path: Path,
    *,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    calibration = _load_calibration_evidence(config, calibration_evidence_path)
    review = _validate_checkpoint_review(
        calibration_evidence_path, checkpoint_review_path
    )
    selected = int(calibration["selected_max_context_length"])
    snapshot_path = _resolve_model_snapshot(config.counterfactual.native)
    tokenizer = _load_tokenizer(
        config.counterfactual.native, snapshot_path=snapshot_path
    )
    parent, integrity = _target_parent(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        parent_release_package_path=parent_release_package_path,
    )
    expected = full_context_requests(
        config,
        parent,
        selected_context_length=selected,
        calibration_evidence_path=calibration_evidence_path,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_path = artifact_dir / "generations.jsonl"
    prior = load_full_context_generation_records(
        generation_path, expected, selected_context_length=selected
    )
    completed = {str(record["branch_id"]) for record in prior}
    pending = [request for request in expected if request.branch_id not in completed]
    if not pending:
        return {
            "status": "already_complete",
            "expected_branches": len(expected),
            "new_branches": 0,
            "selected_max_context_length": selected,
            "integrity": integrity,
        }

    _configure_fork_runtime()
    runtime = _fork_local_runtime(config.counterfactual)
    runtime["engine_gpu_memory_utilization"] = config.execution[
        "gpu_memory_utilization"
    ]
    runtime["selected_max_context_length"] = selected
    runtime["package_versions"] = _validated_package_versions()
    device_index = int(runtime["cuda_device_index"])
    _assert_no_other_compute_process(device_index)
    started = time.perf_counter()
    status = "failed"
    error_text: str | None = None
    new_records: list[dict[str, Any]] = []
    try:
        new_records = asyncio.run(
            _run_full_context_branches(
                config,
                tokenizer,
                pending,
                generation_path,
                artifact_dir / "branch-attempts.jsonl",
                snapshot_path,
                selected_context_length=selected,
                device_index=device_index,
            )
        )
        status = "completed"
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        _append_jsonl(
            artifact_dir / "generation-segments.jsonl",
            {
                "schema_version": GENERATION_SEGMENT_SCHEMA,
                "status": status,
                "requested_branch_count": len(pending),
                "persisted_branch_count": len(new_records),
                "segment_wall_time_seconds": time.perf_counter() - started,
                "error": error_text,
                "selected_max_context_length": selected,
                "calibration_evidence_sha256": _file_sha256(calibration_evidence_path),
                "checkpoint_reviewed_commit": review["reviewed_commit"],
                "runtime": runtime,
            },
        )
    return {
        "status": status,
        "expected_branches": len(expected),
        "new_branches": len(new_records),
        "selected_max_context_length": selected,
        "integrity": integrity,
        "runtime": runtime,
    }


def run_full_context_verification(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    calibration_evidence_path: Path,
    *,
    project_roots: Mapping[str, Path],
    workers: int | None = None,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    calibration = _load_calibration_evidence(config, calibration_evidence_path)
    selected = int(calibration["selected_max_context_length"])
    snapshot_path = _resolve_model_snapshot(config.counterfactual.native)
    tokenizer = _load_tokenizer(
        config.counterfactual.native, snapshot_path=snapshot_path
    )
    parent, integrity = _target_parent(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        parent_release_package_path=parent_release_package_path,
    )
    expected = full_context_requests(
        config,
        parent,
        selected_context_length=selected,
        calibration_evidence_path=calibration_evidence_path,
    )
    generations = load_full_context_generation_records(
        artifact_dir / "generations.jsonl",
        expected,
        selected_context_length=selected,
    )
    if len(generations) != len(expected):
        raise RuntimeError(
            f"full-context generation is incomplete: {len(generations)}/{len(expected)}"
        )
    environment_tasks, _ = load_mathia_tasks(config.counterfactual.native, mathia_root)
    environments = validate_lean_environments(
        config.counterfactual.native, environment_tasks, project_roots
    )
    verification_path = artifact_dir / "verifications.jsonl"
    prior = load_fork_verification_records(verification_path, expected)
    latest = latest_verifications(prior)
    pending = [
        generation
        for generation in generations
        if str(generation["branch_id"]) not in latest
        or str(latest[str(generation["branch_id"])]["category"])
        not in MATHEMATICAL_VERIFIER_CATEGORIES
    ]
    if not pending:
        return {
            "status": "already_complete",
            "generation_branches": len(generations),
            "new_verifications": 0,
            "integrity": integrity,
            "environments": environments,
        }
    verifier = LeanVerifier(
        project_roots[TARGET_WORKLOAD],
        timeout_seconds=float(config.counterfactual.native.verifier["timeout_seconds"]),
    )
    worker_count = int(
        config.counterfactual.native.verifier["workers"] if workers is None else workers
    )
    if worker_count < 1:
        raise ValueError("verification worker count must be positive")
    attempts = Counter(str(record["branch_id"]) for record in prior)
    new_count = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _verify_fork_generation_record,
                generation,
                parent.task,
                verifier,
                attempt_index=attempts[str(generation["branch_id"])],
            ): str(generation["branch_id"])
            for generation in pending
        }
        for future in as_completed(futures):
            record = future.result()
            _append_jsonl(verification_path, record)
            new_count += 1
            print(
                json.dumps(
                    {
                        "phase": "full_context_verification",
                        "completed": new_count,
                        "pending": len(pending),
                        "branch_id": record["branch_id"],
                        "category": record["category"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "status": "completed",
        "generation_branches": len(generations),
        "new_verifications": new_count,
        "verification_wall_time_seconds": time.perf_counter() - started,
        "integrity": integrity,
        "environments": environments,
    }


def _branch_attempt_summary(path: Path) -> dict[str, Any]:
    starts: Counter[str] = Counter()
    terminal: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    if path.exists():
        for line in _restart_safe_jsonl_lines(path):
            event = json.loads(line)
            if event.get("schema_version") != BRANCH_ATTEMPT_SCHEMA:
                raise ValueError("unknown branch-attempt schema")
            branch_id = str(event["branch_id"])
            event_kind = str(event["event"])
            if event_kind == "started":
                starts[branch_id] += 1
            elif event_kind in {"persisted", "failed", "interrupted"}:
                terminal[branch_id] += 1
                status_counts[event_kind] += 1
            else:
                raise ValueError("unknown branch-attempt event")
    incomplete_starts = sum(
        max(0, starts[branch_id] - terminal[branch_id]) for branch_id in starts
    )
    retried_branch_ids = sorted(
        branch_id for branch_id, count in starts.items() if count > 1
    )
    return {
        "started_attempt_count": sum(starts.values()),
        "terminal_attempt_count": sum(terminal.values()),
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "incomplete_started_attempt_count": incomplete_starts,
        "retried_in_flight_branch_count": len(retried_branch_ids),
        "retried_in_flight_branch_ids_sha256": _sha256_json(retried_branch_ids),
    }


def _classify_outcome(
    config: FullContextForkingConfig,
    per_state: Sequence[dict[str, Any]],
    generations: Sequence[dict[str, Any]],
    verifications: Sequence[dict[str, Any]],
) -> str:
    nonempty = sum(
        1 for record in generations if record["final_production_status"] == "nonempty"
    )
    verified = sum(1 for record in verifications if record["category"] == "verified")
    multiple = int(config.classification["multiple_branches_minimum"])
    f_values = {float(row["F"]) for row in per_state}
    v_values = {float(row["V"]) for row in per_state}
    state_dependent = len(f_values) > 1 or len(v_values) > 1
    if max(nonempty, verified) >= multiple:
        return (
            "context_limited_signal"
            if state_dependent
            else "context_limited_but_no_fork_signal"
        )
    empty_or_exhausted = sum(
        1
        for record in generations
        if record["final_production_status"] == "empty"
        and record["finish_reason"] == "length"
    )
    numerator = int(config.classification["great_majority_numerator"])
    denominator = int(config.classification["great_majority_denominator"])
    if empty_or_exhausted * denominator >= len(generations) * numerator:
        return "persistent_thinking_attractor"
    return "mixed_or_inconclusive"


def _baseline_issue92_rows(
    config: FullContextForkingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = config.repository_root / str(
        config.reviewed_target["counterfactual_results_path"]
    )
    if _file_sha256(path) != COUNTERFACTUAL_RESULTS_SHA256:
        raise ValueError("issue-92 result changed before comparison")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in evidence["analysis"]["discovery"]["per_prefix"]
        if row["task_id"] == TARGET_TASK_ID
        and row["parent_candidate_id"] == TARGET_PARENT_ID
    ]
    if [row["fork_state"] for row in rows] != list(SCIENTIFIC_STATES):
        raise ValueError("issue-92 target prefix order changed")
    if not all(
        row["branch_count"] == 6
        and row["nonempty_final_branches"] == 0
        and row["verified_branches"] == 0
        and row["F"] == 0.0
        and row["V_op"] == 0.0
        for row in rows
    ):
        raise ValueError("issue-92 all-zero comparison changed")
    compact = [
        {
            "fork_state": row["fork_state"],
            "fork_prefix_len": row["fork_prefix_len"],
            "branch_count": row["branch_count"],
            "F": row["F"],
            "V": row["V_op"],
        }
        for row in rows
    ]
    return compact, {
        "path": str(config.reviewed_target["counterfactual_results_path"]),
        "sha256": COUNTERFACTUAL_RESULTS_SHA256,
        "mutation": "none_read_only_comparison",
    }


def write_full_context_evidence(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    calibration_evidence_path: Path,
    checkpoint_review_path: Path,
    output_path: Path,
    *,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    calibration = _load_calibration_evidence(config, calibration_evidence_path)
    checkpoint_review = _validate_checkpoint_review(
        calibration_evidence_path, checkpoint_review_path
    )
    selected = int(calibration["selected_max_context_length"])
    snapshot_path = _resolve_model_snapshot(config.counterfactual.native)
    tokenizer = _load_tokenizer(
        config.counterfactual.native, snapshot_path=snapshot_path
    )
    parent, integrity = _target_parent(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        parent_release_package_path=parent_release_package_path,
    )
    requests = full_context_requests(
        config,
        parent,
        selected_context_length=selected,
        calibration_evidence_path=calibration_evidence_path,
    )
    generations = load_full_context_generation_records(
        artifact_dir / "generations.jsonl",
        requests,
        selected_context_length=selected,
    )
    if len(generations) != len(requests):
        raise RuntimeError("cannot write evidence from incomplete generation")
    verification_records = load_fork_verification_records(
        artifact_dir / "verifications.jsonl", requests
    )
    latest = latest_verifications(verification_records)
    if len(latest) != len(requests) or any(
        record["category"] not in MATHEMATICAL_VERIFIER_CATEGORIES
        for record in latest.values()
    ):
        raise RuntimeError("cannot write evidence from incomplete verification")
    ordered_verifications = [latest[request.branch_id] for request in requests]
    baseline_rows, baseline_binding = _baseline_issue92_rows(config)
    per_state: list[dict[str, Any]] = []
    branch_diagnostics: list[dict[str, Any]] = []
    for state_label in SCIENTIFIC_STATES:
        state_generations = [
            record for record in generations if record["fork_state"] == state_label
        ]
        state_verifications = [
            record
            for record in ordered_verifications
            if record["fork_state"] == state_label
        ]
        nonempty = sum(
            1
            for record in state_generations
            if record["final_production_status"] == "nonempty"
        )
        verified = sum(
            1 for record in state_verifications if record["category"] == "verified"
        )
        request = next(
            request for request in requests if request.state.label == state_label
        )
        per_state.append(
            {
                "fork_state": state_label,
                "fork_fraction": request.state.fraction,
                "fork_prefix_len": request.state.prefix_len,
                "rendered_prompt_token_count": len(parent.rendered_prompt_token_ids),
                "max_new_tokens": request.max_tokens,
                "branch_count": len(state_generations),
                "nonempty_final_branches": nonempty,
                "verified_branches": verified,
                "F": nonempty / len(state_generations),
                "V": verified / len(state_verifications),
                "issue92_F": 0.0,
                "issue92_V": 0.0,
            }
        )
    for generation, verification in zip(
        generations, ordered_verifications, strict=True
    ):
        memory = generation["branch_gpu_memory"]
        branch_diagnostics.append(
            {
                "branch_id": generation["branch_id"],
                "fork_state": generation["fork_state"],
                "branch_seed": generation["branch_seed"],
                "max_new_tokens": generation["max_tokens"],
                "generated_token_count": generation["total_generated_token_count"],
                "reasoning_token_count": generation["reasoning_token_count"],
                "final_token_count": generation["final_token_count"],
                "reasoning_to_final_transition_token_index": generation[
                    "reasoning_to_final_transition_token_index"
                ],
                "finish_reason": generation["finish_reason"],
                "verifier_category": verification["category"],
                "generation_latency_seconds": generation["generation_latency_seconds"],
                "gpu_memory_peak_bytes": memory["gpu_memory_peak_bytes"],
            }
        )
    classification = _classify_outcome(
        config, per_state, generations, ordered_verifications
    )
    generation_sha = _file_sha256(artifact_dir / "generations.jsonl")
    verification_sha = _file_sha256(artifact_dir / "verifications.jsonl")
    evidence = {
        "schema_version": FINAL_EVIDENCE_SCHEMA,
        "status": "complete_scoring_excluded_diagnostic",
        "classification": classification,
        "reviewed_target": config.reviewed_target,
        "diagnostic_target": config.diagnostic_target,
        "calibration": {
            "evidence_path": str(
                calibration_evidence_path.relative_to(config.repository_root)
            ),
            "evidence_sha256": _file_sha256(calibration_evidence_path),
            "selected_max_context_length": selected,
            "selected_length_success_count": calibration[
                "selected_length_success_count"
            ],
            "checkpoint_review": checkpoint_review,
        },
        "scientific_contract": {
            "branch_count": len(requests),
            "states": list(SCIENTIFIC_STATES),
            "seeds": list(SCIENTIFIC_SEEDS),
            "max_new_tokens_formula": (
                "M - rendered_prompt_token_count - frozen_reasoning_prefix_count"
            ),
            "direct_stored_token_id_prefixes": True,
            "prefix_positions_reused_from_issue92": True,
            "final_channel_only_lean_submission": True,
            "repair": None,
            "context_extension": None,
            "hosted_inference": False,
            "only_scientific_variable": "context_and_generation_budget",
            "classification_rule": config.classification,
        },
        "comparison": {
            "issue92": baseline_binding,
            "issue92_per_state": baseline_rows,
            "full_context_per_state": per_state,
        },
        "branch_diagnostics": branch_diagnostics,
        "branch_identity": {
            "ordered_branch_ids_sha256": _sha256_json(
                [request.branch_id for request in requests]
            ),
            "fork_generation_config_sha256": requests[0].generation_config_sha256,
            "raw_generations_sha256": generation_sha,
            "raw_verifications_sha256": verification_sha,
        },
        "restart_safety": _branch_attempt_summary(
            artifact_dir / "branch-attempts.jsonl"
        ),
        "outcomes": {
            "nonempty_final_branch_count": sum(
                1
                for generation in generations
                if generation["final_production_status"] == "nonempty"
            ),
            "verified_branch_count": sum(
                1
                for verification in ordered_verifications
                if verification["category"] == "verified"
            ),
            "verifier_category_counts": dict(
                sorted(
                    Counter(
                        str(record["category"]) for record in ordered_verifications
                    ).items()
                )
            ),
        },
        "integrity": {
            "parent": integrity,
            "reviewed_issue92_results_unchanged": True,
            "reviewed_issue92_results_sha256": COUNTERFACTUAL_RESULTS_SHA256,
        },
        "limitations": [
            "This is a one-task scoring-excluded diagnostic and does not replace issue-92 results.",
            "The classification localizes this frozen trajectory only; it is not a population estimate.",
            "Raw generations and verifier diagnostics remain outside Git; hashes bind them here.",
        ],
    }
    _atomic_write_json(output_path, evidence)
    return evidence
