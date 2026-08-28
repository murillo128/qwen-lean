from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .native_thinking_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    WORKLOADS,
    MathiaTask,
    NativeThinkingConfig,
    _apparent_natural_language,
    _append_jsonl,
    _atomic_write_json,
    _distribution,
    _file_sha256,
    _load_tokenizer,
    _repeats_declaration_or_by,
    _sha256_json,
    _sha256_text,
    load_mathia_tasks,
    validate_lean_environments,
)
from .thinking_budget_scaling import (
    FROZEN_SCALING_ARMS,
    LEAN_WRAPPER_NORMALIZATION,
    ScalingRequest,
    SelectedTask,
    ThinkingBudgetScalingConfig,
    _audit_probe_record,
    _execute_requests,
    _repository_relative_path,
    _request_from_payload,
    lean_wrapper_normalization_v1,
    load_scaling_generation_records,
    runtime_support_audit,
    select_scaling_tasks,
    validate_stage1_binding,
)
from .verifier import LeanVerifier, VerificationOutcome

CONTINUATION_CONFIG_SCHEMA = "qwen35-thinking-budget-continuation-config-v1"
CONTINUATION_GATE_SCHEMA = "qwen35-thinking-budget-continuation-gate-v1"
CONTINUATION_VERIFICATION_SCHEMA = "qwen35-thinking-budget-continuation-verification-v1"
CONTINUATION_RESULTS_SCHEMA = "qwen35-thinking-budget-continuation-results-v1"
EXPECTED_BASE_CONFIG_SHA256 = (
    "cf9222dbdfb334aa6c439c8b59b8fbd369259f2319678cbea97add9e0d89b8b6"
)
EXPECTED_HISTORICAL_GATE_TARGET = "034c96fb54761e843ff68fb21d688b22a2a645ef"
EXPECTED_HISTORICAL_GATE_SHA256 = (
    "302cfeb1a348d61387e83764a5a142975069fcec84708fb146bb7a3fab871f0e"
)
CONTINUATION_ARMS = ("B4", "B8", "B16")


@dataclass(frozen=True)
class ThinkingBudgetContinuationConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ThinkingBudgetContinuationConfig:
        config = cls(
            path=path.resolve(), value=json.loads(path.read_text(encoding="utf-8"))
        )
        validate_continuation_config(config)
        return config

    @property
    def base_scaling_config(self) -> dict[str, Any]:
        return self.value["base_scaling_config"]

    @property
    def historical_gate(self) -> dict[str, Any]:
        return self.value["historical_gate"]

    @property
    def canonical_output(self) -> dict[str, Any]:
        return self.value["canonical_output"]

    @property
    def continuation_gate(self) -> dict[str, Any]:
        return self.value["continuation_gate"]


def validate_continuation_config(config: ThinkingBudgetContinuationConfig) -> None:
    if config.value.get("schema_version") != CONTINUATION_CONFIG_SCHEMA:
        raise ValueError("unknown thinking-budget continuation config schema")
    if (
        config.value.get("experiment_id")
        != "qwen35-4b-thinking-budget-scaling-continuation-v1"
    ):
        raise ValueError("thinking-budget continuation experiment id changed")
    if config.base_scaling_config != {
        "path": "config/qwen35-thinking-budget-scaling.json",
        "sha256": EXPECTED_BASE_CONFIG_SHA256,
    }:
        raise ValueError("historical scaling config binding changed")
    if config.historical_gate != {
        "reviewed_target": EXPECTED_HISTORICAL_GATE_TARGET,
        "evidence_path": ("evidence/qwen35-thinking-budget-scaling/runtime-gate.json"),
        "evidence_sha256": EXPECTED_HISTORICAL_GATE_SHA256,
        "preserve_unchanged": True,
    }:
        raise ValueError("historical runtime-gate binding changed")
    if config.canonical_output != {
        "parser": "qwen3",
        "parser_source_sha256": (
            "962f8f55210eb0a431cb9c78b013e35f7a7dd58d06d6fb2e7fcff1b457356f8e"
        ),
        "canonical_answer": "parsed_final_exact",
        "normalization": LEAN_WRAPPER_NORMALIZATION,
        "raw_suffix_identity_is_diagnostic": True,
    }:
        raise ValueError("canonical parsed-output contract changed")
    if config.continuation_gate != {
        "probe_workload": WORKLOADS[0],
        "probe_final_allowance": 128,
        "reasoning_budgets": [32, 64, 128],
    }:
        raise ValueError("continuation-gate probes changed")


def validate_continuation_binding(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    *,
    stage1_results_path: Path | None = None,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    expected_scaling_path = repository_root / continuation.base_scaling_config["path"]
    if scaling.path != expected_scaling_path.resolve():
        raise ValueError("continuation loaded a different scaling config path")
    if _file_sha256(scaling.path) != continuation.base_scaling_config["sha256"]:
        raise ValueError("historical scaling config bytes changed")
    historical_gate_path = (
        repository_root / continuation.historical_gate["evidence_path"]
    )
    if (
        _file_sha256(historical_gate_path)
        != continuation.historical_gate["evidence_sha256"]
    ):
        raise ValueError("historical runtime-gate evidence changed")
    validate_stage1_binding(scaling, stage1, stage1_results_path=stage1_results_path)


def continuation_generation_config_sha256(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    arm: str,
) -> str:
    if arm not in CONTINUATION_ARMS:
        raise ValueError(f"unknown continuation arm: {arm}")
    return _sha256_json(
        {
            "continuation_config_sha256": _file_sha256(continuation.path),
            "base_scaling_config_sha256": _file_sha256(scaling.path),
            "stage1_model": stage1.model,
            "stage1_prompt": stage1.value["prompt"],
            "enable_thinking": True,
            "sampling": scaling.sampling,
            "arm": {"name": arm, **scaling.arms[arm]},
            "canonical_output": continuation.canonical_output,
            "engine": {
                key: stage1.engine[key]
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


def continuation_candidate_identity(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    selected: SelectedTask,
    arm: str,
) -> tuple[str, dict[str, Any]]:
    if arm not in CONTINUATION_ARMS:
        raise ValueError(f"unknown continuation arm: {arm}")
    arm_config = scaling.arms[arm]
    payload = {
        "arm": arm,
        "enable_thinking": True,
        "workload": selected.task.workload,
        "task_id": selected.task.task_id,
        "prompt_sha256": selected.user_message_sha256,
        "candidate_index": 0,
        "seed": int(scaling.sampling["seed"]),
        "model_revision": MODEL_REVISION,
        "max_reasoning_tokens": int(arm_config["max_reasoning_tokens"]),
        "total_output_ceiling": int(arm_config["total_output_ceiling"]),
        "generation_config_sha256": continuation_generation_config_sha256(
            continuation, scaling, stage1, arm
        ),
    }
    return "thinking-budget-scaling-" + _sha256_json(payload)[:32], payload


def _continuation_gate_candidate_identity(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    selected: SelectedTask,
    reasoning_budget: int,
) -> tuple[str, dict[str, Any]]:
    total = reasoning_budget + int(
        continuation.continuation_gate["probe_final_allowance"]
    )
    payload = {
        "arm": f"continuation-gate-B{reasoning_budget}",
        "enable_thinking": True,
        "workload": selected.task.workload,
        "task_id": selected.task.task_id,
        "prompt_sha256": selected.user_message_sha256,
        "candidate_index": 0,
        "seed": int(scaling.sampling["seed"]),
        "model_revision": MODEL_REVISION,
        "max_reasoning_tokens": reasoning_budget,
        "total_output_ceiling": total,
        "generation_config_sha256": _sha256_json(
            {
                "continuation_config_sha256": _file_sha256(continuation.path),
                "continuation_gate": continuation.continuation_gate,
                "canonical_output": continuation.canonical_output,
                "sampling": scaling.sampling,
            }
        ),
    }
    return "thinking-budget-gate-" + _sha256_json(payload)[:32], payload


def run_continuation_gate(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    output_path: Path,
    *,
    stage1_results_path: Path | None = None,
) -> dict[str, Any]:
    validate_continuation_binding(
        continuation,
        scaling,
        stage1,
        stage1_results_path=stage1_results_path,
    )
    runtime = runtime_support_audit(scaling, stage1)
    if not runtime["passed"]:
        evidence = _failed_gate_evidence(continuation, runtime)
        _write_continuation_gate_evidence(output_path, evidence)
        return evidence

    tasks, mathia_binding = load_mathia_tasks(stage1, mathia_root)
    tokenizer = _load_tokenizer(stage1)
    selected, selection = select_scaling_tasks(scaling, stage1, tasks, tokenizer)
    probe = next(
        row
        for row in selected
        if row.task.workload == continuation.continuation_gate["probe_workload"]
    )
    requests: list[ScalingRequest] = []
    for budget in continuation.continuation_gate["reasoning_budgets"]:
        candidate_id, payload = _continuation_gate_candidate_identity(
            continuation, scaling, probe, int(budget)
        )
        requests.append(
            _request_from_payload(probe, payload, candidate_id, is_runtime_gate=True)
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_path = artifact_dir / "continuation-gate-generations.jsonl"
    segment_path = artifact_dir / "continuation-gate-segments.jsonl"
    prior = load_continuation_generation_records(generation_path)
    completed = {str(row["candidate_id"]) for row in prior}
    pending = [row for row in requests if row.candidate_id not in completed]
    runtime_segment: dict[str, Any] | None = None
    if pending:
        runtime_segment = _execute_requests(
            scaling,
            stage1,
            tokenizer,
            pending,
            generation_path,
            _resolve_snapshot(stage1),
            segment_path,
            segment_kind="continuation_runtime_gate",
        )
    elif segment_path.exists():
        runtime_segment = _read_jsonl(segment_path)[-1]

    records = load_continuation_generation_records(generation_path)
    by_id = {str(row["candidate_id"]): row for row in records}
    probe_records = [by_id[row.candidate_id] for row in requests]
    audits = [
        _continuation_replay_audit(row, probe, tokenizer, scaling, stage1)
        for row in probe_records
    ]
    checks = {
        "all_probe_records_present": len(probe_records) == len(requests),
        "reasoning_bounded": all(
            int(row["reasoning_token_count"]) <= int(row["max_reasoning_tokens"])
            for row in probe_records
        ),
        "forced_boundaries_continue_to_nonempty_final": all(
            row["reasoning_exit_audit"] == "forced_at_budget"
            and bool(row["parsed_final_exact"])
            for row in audits
        ),
        "pinned_parser_replay_deterministic": all(
            row["parser_replay_matches_stored"] for row in audits
        ),
        "reasoning_markers_absent_from_parsed_final": all(
            not row["final_has_reasoning_marker"] for row in probe_records
        ),
        "normalization_replay_deterministic": all(
            row["normalization_replay_matches_stored"] for row in audits
        ),
        "normalization_idempotent": all(
            row["normalization_idempotent"] for row in probe_records
        ),
        "exact_parsed_and_normalized_hashes_retained": all(
            _record_output_hashes_match(row) for row in probe_records
        ),
        "exactly_one_frozen_normalization_pass": all(
            row["normalization_id"] == LEAN_WRAPPER_NORMALIZATION
            and int(row["normalization_pass_count"]) == 1
            for row in probe_records
        ),
        "raw_token_ids_retained": all(
            len(row["raw_response_token_ids"]) == int(row["raw_response_token_count"])
            for row in probe_records
        ),
    }
    passed = all(checks.values())
    evidence = {
        "schema_version": CONTINUATION_GATE_SCHEMA,
        "status": "passed" if passed else "failed",
        "conclusion": (
            "continuation_runtime_gate_passed"
            if passed
            else "budget_control_runtime_or_parser_not_usable"
        ),
        "historical_gate": continuation.historical_gate,
        "continuation_config_sha256": _file_sha256(continuation.path),
        "base_scaling_config_sha256": _file_sha256(scaling.path),
        "stage1_config_sha256": _file_sha256(stage1.path),
        "runtime_support": runtime,
        "canonical_output_contract": continuation.canonical_output,
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
        "probes": [_compact_continuation_generation(row) for row in audits],
        "raw_suffix_identity": {
            "blocking": False,
            "mismatch_count": sum(
                not row["parser_final_content_is_exact_raw_suffix"]
                for row in probe_records
            ),
        },
        "runtime": runtime_segment,
        "raw_artifacts": {
            "generation_path": _repository_relative_path(generation_path),
            "generation_sha256": _file_sha256(generation_path),
            "segment_path": _repository_relative_path(segment_path),
            "segment_sha256": _file_sha256(segment_path),
        },
        "scientific_generation_authorized": passed,
        "scientific_generation_started": False,
        "scientific_generation_candidate_count": 0,
    }
    _write_continuation_gate_evidence(output_path, evidence)
    return evidence


def _failed_gate_evidence(
    continuation: ThinkingBudgetContinuationConfig, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": CONTINUATION_GATE_SCHEMA,
        "status": "failed",
        "conclusion": "budget_control_runtime_or_parser_not_usable",
        "historical_gate": continuation.historical_gate,
        "runtime_support": dict(runtime),
        "scientific_generation_authorized": False,
        "scientific_generation_started": False,
        "scientific_generation_candidate_count": 0,
    }


def _resolve_snapshot(stage1: NativeThinkingConfig) -> Path:
    from .native_thinking_assessment import _resolve_model_snapshot

    return _resolve_model_snapshot(stage1)


def run_continuation_generation(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    arm: str,
    artifact_dir: Path,
    gate_path: Path,
    *,
    stage1_results_path: Path | None = None,
) -> dict[str, Any]:
    if arm not in CONTINUATION_ARMS:
        raise ValueError(f"unknown continuation arm: {arm}")
    validate_continuation_binding(
        continuation,
        scaling,
        stage1,
        stage1_results_path=stage1_results_path,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != CONTINUATION_GATE_SCHEMA or not gate.get(
        "scientific_generation_authorized"
    ):
        raise RuntimeError("Stage 2 continuation gate has not passed")
    if gate.get("continuation_config_sha256") != _file_sha256(continuation.path):
        raise RuntimeError("Stage 2 continuation gate used another config")

    tasks, mathia_binding = load_mathia_tasks(stage1, mathia_root)
    snapshot = _resolve_snapshot(stage1)
    tokenizer = _load_tokenizer(stage1, snapshot_path=snapshot)
    selected, selection = select_scaling_tasks(scaling, stage1, tasks, tokenizer)
    requests: list[ScalingRequest] = []
    for row in selected:
        candidate_id, payload = continuation_candidate_identity(
            continuation, scaling, stage1, row, arm
        )
        requests.append(
            _request_from_payload(row, payload, candidate_id, is_runtime_gate=False)
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_path = artifact_dir / "generations.jsonl"
    prior = load_continuation_generation_records(generation_path)
    completed = {str(row["candidate_id"]) for row in prior}
    pending = [row for row in requests if row.candidate_id not in completed]
    if not pending:
        return {
            "status": "already_complete",
            "arm": arm,
            "requested_candidates": len(requests),
            "new_candidates": 0,
            "selection": selection,
        }
    runtime = _execute_requests(
        scaling,
        stage1,
        tokenizer,
        pending,
        generation_path,
        snapshot,
        artifact_dir / "generation-segments.jsonl",
        segment_kind="continuation_scaling_probe",
    )
    return {
        "status": "completed",
        "arm": arm,
        "requested_candidates": len(requests),
        "new_candidates": len(pending),
        "selection": selection,
        "mathia_binding": {
            key: value for key, value in mathia_binding.items() if key != "corpus_root"
        },
        "runtime": runtime,
    }


def load_continuation_generation_records(path: Path) -> list[dict[str, Any]]:
    records = load_scaling_generation_records(path)
    for record in records:
        candidate_id = str(record["candidate_id"])
        required = (
            "parsed_final_exact",
            "parsed_final_sha256",
            "parsed_final_token_count",
            "normalized_final_exact",
            "normalized_final_sha256",
            "normalized_final_token_count",
            "normalization_id",
            "normalization_applied",
            "normalization_pass_count",
            "normalization_idempotent",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"continuation generation lacks {missing}: {candidate_id}")
        if record["parsed_final_exact"] != record.get("final_content"):
            raise ValueError(f"canonical parsed final mismatch: {candidate_id}")
        if record["parsed_final_sha256"] != record.get("final_content_sha256"):
            raise ValueError(f"canonical parsed-final hash mismatch: {candidate_id}")
        if not _record_output_hashes_match(record):
            raise ValueError(f"continuation output hash mismatch: {candidate_id}")
        normalized_replay, applied = lean_wrapper_normalization_v1(
            record["parsed_final_exact"]
        )
        if normalized_replay != record["normalized_final_exact"] or applied != bool(
            record["normalization_applied"]
        ):
            raise ValueError(f"normalization replay mismatch: {candidate_id}")
        if record["normalization_id"] != LEAN_WRAPPER_NORMALIZATION:
            raise ValueError(f"normalization id mismatch: {candidate_id}")
        if int(record["normalization_pass_count"]) != 1:
            raise ValueError(f"normalization pass count mismatch: {candidate_id}")
        second_pass, _ = lean_wrapper_normalization_v1(record["normalized_final_exact"])
        if second_pass != record["normalized_final_exact"]:
            raise ValueError(f"normalization is not idempotent: {candidate_id}")
    return records


def _record_output_hashes_match(record: Mapping[str, Any]) -> bool:
    parsed = record.get("parsed_final_exact")
    normalized = record.get("normalized_final_exact")
    return record.get("parsed_final_sha256") == (
        None if parsed is None else _sha256_text(str(parsed))
    ) and record.get("normalized_final_sha256") == (
        None if normalized is None else _sha256_text(str(normalized))
    )


def _continuation_replay_audit(
    record: Mapping[str, Any],
    selected: SelectedTask,
    tokenizer: Any,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
) -> dict[str, Any]:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.reasoning import ReasoningParserManager

    if record["task_id"] != selected.task.task_id:
        raise ValueError("parser replay task binding changed")
    parser_class = ReasoningParserManager.get_reasoning_parser(
        str(stage1.engine["reasoning_parser"])
    )
    parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": selected.user_message}],
        max_tokens=int(record["total_output_ceiling"]),
        thinking_token_budget=int(record["max_reasoning_tokens"]),
        temperature=float(scaling.sampling["temperature"]),
        top_p=float(scaling.sampling["top_p"]),
        include_reasoning=True,
    )
    replay_reasoning, replay_final = parser.extract_reasoning(
        str(record["raw_response_text"]), request
    )
    replay_normalized, replay_applied = lean_wrapper_normalization_v1(replay_final)
    parsed_token_count = (
        0
        if replay_final is None
        else len(tokenizer.encode(replay_final, add_special_tokens=False))
    )
    normalized_token_count = (
        0
        if replay_normalized is None
        else len(tokenizer.encode(replay_normalized, add_special_tokens=False))
    )
    audited = _audit_probe_record(record, tokenizer)
    audited.update(
        {
            "parser_replay_matches_stored": (
                replay_reasoning == record.get("reasoning_content")
                and replay_final == record.get("parsed_final_exact")
            ),
            "normalization_replay_matches_stored": (
                replay_normalized == record.get("normalized_final_exact")
                and replay_applied == bool(record.get("normalization_applied"))
            ),
            "parser_replay_reasoning_sha256": (
                None if replay_reasoning is None else _sha256_text(replay_reasoning)
            ),
            "parser_replay_final_sha256": (
                None if replay_final is None else _sha256_text(replay_final)
            ),
            "parsed_final_token_count_stored": int(record["parsed_final_token_count"]),
            "normalized_final_token_count_stored": int(
                record["normalized_final_token_count"]
            ),
            "parsed_final_token_count": parsed_token_count,
            "normalized_final_token_count": normalized_token_count,
        }
    )
    return audited


def run_continuation_verification(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    *,
    project_roots: Mapping[str, Path],
    workers: int | None = None,
) -> dict[str, Any]:
    validate_continuation_binding(continuation, scaling, stage1)
    tasks, _ = load_mathia_tasks(stage1, mathia_root)
    tasks_by_id = {task.task_id: task for task in tasks}
    environments = validate_lean_environments(stage1, tasks, project_roots)
    generations = load_continuation_generation_records(
        artifact_dir / "generations.jsonl"
    )
    verification_path = artifact_dir / "verifications.jsonl"
    prior = load_continuation_verification_records(verification_path)
    completed = {str(record["candidate_id"]) for record in prior}
    pending = [
        record for record in generations if str(record["candidate_id"]) not in completed
    ]
    if not pending:
        return {
            "status": "already_complete",
            "generation_candidates": len(generations),
            "new_verifications": 0,
            "environments": environments,
        }
    verifiers = {
        workload: LeanVerifier(
            project_roots[workload],
            timeout_seconds=float(stage1.verifier["timeout_seconds"]),
        )
        for workload in WORKLOADS
    }
    worker_count = int(stage1.verifier["workers"] if workers is None else workers)
    if worker_count < 1:
        raise ValueError("verification workers must be positive")
    started = time.perf_counter()
    new_count = 0
    progress_every = max(1, len(pending) // 8)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _verify_dual_record,
                record,
                tasks_by_id[str(record["task_id"])],
                verifiers[str(record["workload"])],
            ): str(record["candidate_id"])
            for record in pending
        }
        for future in as_completed(futures):
            _append_jsonl(verification_path, future.result())
            new_count += 1
            if new_count % progress_every == 0 or new_count == len(pending):
                print(
                    json.dumps(
                        {
                            "phase": "thinking_budget_continuation_verification",
                            "completed_candidates": new_count,
                            "total_candidates": len(pending),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    segment = {
        "schema_version": (
            "qwen35-thinking-budget-continuation-verification-segment-v1"
        ),
        "status": "completed",
        "candidate_count": new_count,
        "workers": worker_count,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _append_jsonl(artifact_dir / "verification-segments.jsonl", segment)
    return {
        "status": "completed",
        "generation_candidates": len(generations),
        "new_verifications": new_count,
        "environments": environments,
        "runtime": segment,
    }


def _verify_dual_record(
    generation: dict[str, Any], task: MathiaTask, verifier: LeanVerifier
) -> dict[str, Any]:
    if generation["task_id"] != task.task_id:
        raise ValueError("generation/task mismatch during continuation verification")
    parsed = generation.get("parsed_final_exact")
    normalized = generation.get("normalized_final_exact")
    strict = _verify_exact_final(parsed, task, verifier)
    shared = parsed == normalized
    deployed = strict if shared else _verify_exact_final(normalized, task, verifier)
    return {
        "schema_version": CONTINUATION_VERIFICATION_SCHEMA,
        "candidate_id": generation["candidate_id"],
        "arm": generation["arm"],
        "workload": generation["workload"],
        "task_id": generation["task_id"],
        "candidate_index": generation["candidate_index"],
        "seed": generation["seed"],
        "prompt_sha256": generation["prompt_sha256"],
        "generation_config_sha256": generation["generation_config_sha256"],
        "parsed_final_sha256": generation["parsed_final_sha256"],
        "normalized_final_sha256": generation["normalized_final_sha256"],
        "normalization_id": generation["normalization_id"],
        "normalization_applied": generation["normalization_applied"],
        "strict_parsed_interface": strict,
        "deployed_normalized_interface": deployed,
        "shared_identical_submission": shared,
        "lean_invocation_count": 1 if shared else 2,
        "verification_outcome_changed_by_normalization": (
            strict["category"] != deployed["category"]
        ),
    }


def _verify_exact_final(
    content: str | None, task: MathiaTask, verifier: LeanVerifier
) -> dict[str, Any]:
    if content is None or content == "":
        outcome = VerificationOutcome(
            category="empty_candidate",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": "parsed final content is empty"},
            latency_seconds=0.0,
        )
    else:
        source = f"{task.preamble}\n\n{task.declaration} := by\n  {content}\n"
        outcome = verifier._run_source(source)
    return {
        "submitted_sha256": (None if content is None else _sha256_text(content)),
        "submitted_exactly_once": True,
        "category": outcome.category,
        "lean_exit_code": outcome.lean_exit_code,
        "diagnostics": outcome.diagnostics,
        "verification_latency_seconds": outcome.latency_seconds,
    }


def load_continuation_verification_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    seen: set[str] = set()
    for record in records:
        if record.get("schema_version") != CONTINUATION_VERIFICATION_SCHEMA:
            raise ValueError("unknown continuation verification schema")
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate continuation verification: {candidate_id}")
        seen.add(candidate_id)
        if (
            record["strict_parsed_interface"]["submitted_sha256"]
            != record["parsed_final_sha256"]
        ):
            raise ValueError(f"strict submitted hash mismatch: {candidate_id}")
        if (
            record["deployed_normalized_interface"]["submitted_sha256"]
            != record["normalized_final_sha256"]
        ):
            raise ValueError(f"normalized submitted hash mismatch: {candidate_id}")
        expected_invocations = 1 if record["shared_identical_submission"] else 2
        if int(record["lean_invocation_count"]) != expected_invocations:
            raise ValueError(f"Lean invocation count mismatch: {candidate_id}")
    return records


def write_continuation_evidence(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    gate_path: Path,
    evidence_dir: Path,
    *,
    project_roots: Mapping[str, Path],
) -> dict[str, Any]:
    validate_continuation_binding(continuation, scaling, stage1)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("schema_version") != CONTINUATION_GATE_SCHEMA
        or gate.get("status") != "passed"
    ):
        raise RuntimeError("cannot render continuation evidence from a failed gate")
    tasks, mathia_binding = load_mathia_tasks(stage1, mathia_root)
    tokenizer = _load_tokenizer(stage1)
    selected, selection = select_scaling_tasks(scaling, stage1, tasks, tokenizer)
    selected_by_id = {row.task.task_id: row for row in selected}
    generations = load_continuation_generation_records(
        artifact_dir / "generations.jsonl"
    )
    verifications = load_continuation_verification_records(
        artifact_dir / "verifications.jsonl"
    )
    expected_ids = {
        continuation_candidate_identity(
            continuation, scaling, stage1, selected_task, arm
        )[0]
        for selected_task in selected
        for arm in CONTINUATION_ARMS
    }
    generation_by_id = {str(row["candidate_id"]): row for row in generations}
    verification_by_id = {str(row["candidate_id"]): row for row in verifications}
    if set(generation_by_id) != expected_ids or len(generations) != 48:
        raise RuntimeError("continuation generation set is incomplete or mixed")
    if set(verification_by_id) != expected_ids or len(verifications) != 48:
        raise RuntimeError("continuation verification set is incomplete or mixed")

    parser_audits = [
        _continuation_replay_audit(
            generation,
            selected_by_id[str(generation["task_id"])],
            tokenizer,
            scaling,
            stage1,
        )
        for generation in generations
    ]
    parser_audit_by_id = {str(row["candidate_id"]): row for row in parser_audits}
    if not all(row["parser_replay_matches_stored"] for row in parser_audits):
        raise RuntimeError("scientific parser replay is not deterministic")
    if not all(row["normalization_replay_matches_stored"] for row in parser_audits):
        raise RuntimeError("scientific normalization replay is not deterministic")
    if any(row["final_has_reasoning_marker"] for row in generations):
        raise RuntimeError("reasoning marker leaked into parsed final")
    if any(not row["normalization_idempotent"] for row in generations):
        raise RuntimeError("scientific normalization is not idempotent")
    if any(
        verification_by_id[candidate_id]["parsed_final_sha256"]
        != generation["parsed_final_sha256"]
        or verification_by_id[candidate_id]["normalized_final_sha256"]
        != generation["normalized_final_sha256"]
        for candidate_id, generation in generation_by_id.items()
    ):
        raise RuntimeError("verified output hashes differ from generation records")
    if any(
        interface["category"] == "verifier_error"
        for row in verifications
        for interface in (
            row["strict_parsed_interface"],
            row["deployed_normalized_interface"],
        )
    ):
        raise RuntimeError("continuation contains verifier infrastructure errors")

    environments = validate_lean_environments(stage1, tasks, project_roots)
    joined = [
        {
            **generation,
            "parsed_final_token_count_stored": generation["parsed_final_token_count"],
            "normalized_final_token_count_stored": generation[
                "normalized_final_token_count"
            ],
            "parsed_final_token_count": parser_audit_by_id[
                str(generation["candidate_id"])
            ]["parsed_final_token_count"],
            "normalized_final_token_count": parser_audit_by_id[
                str(generation["candidate_id"])
            ]["normalized_final_token_count"],
            "verification": verification_by_id[str(generation["candidate_id"])],
        }
        for generation in generations
    ]
    generation_segments = _read_jsonl(artifact_dir / "generation-segments.jsonl")
    verification_segments = _read_jsonl(artifact_dir / "verification-segments.jsonl")
    summaries = {
        arm: {
            workload: _continuation_summary(
                [
                    row
                    for row in joined
                    if row["arm"] == arm and row["workload"] == workload
                ],
                [
                    segment
                    for segment in generation_segments
                    if segment.get("arm") == arm
                ],
                include_segment_cost=False,
            )
            for workload in WORKLOADS
        }
        for arm in CONTINUATION_ARMS
    }
    for arm in CONTINUATION_ARMS:
        arm_rows = [row for row in joined if row["arm"] == arm]
        summaries[arm]["combined"] = _continuation_summary(
            arm_rows,
            [segment for segment in generation_segments if segment.get("arm") == arm],
            include_segment_cost=True,
        )
    matched_table = _matched_task_table(
        continuation, scaling, stage1, selected, generation_by_id, verification_by_id
    )
    conclusion = _continuation_conclusion(summaries, matched_table)
    result = {
        "schema_version": CONTINUATION_RESULTS_SCHEMA,
        "status": "complete",
        "experiment_id": continuation.value["experiment_id"],
        "historical_targets": {
            "stage1_reviewed_target": scaling.stage1["reviewed_target"],
            "stage2a_reviewed_target": continuation.historical_gate["reviewed_target"],
            "preserved_unchanged": True,
            "historical_candidates_regenerated": 0,
        },
        "config_bindings": {
            "continuation_config_sha256": _file_sha256(continuation.path),
            "base_scaling_config_sha256": _file_sha256(scaling.path),
            "stage1_config_sha256": _file_sha256(stage1.path),
        },
        "model": stage1.model,
        "engine": stage1.engine,
        "sampling": scaling.sampling,
        "arms": FROZEN_SCALING_ARMS,
        "canonical_output_contract": continuation.canonical_output,
        "continuation_gate": {
            "status": gate["status"],
            "checks": gate["checks"],
            "evidence_sha256": _file_sha256(gate_path),
        },
        "mathia_binding": {
            key: value for key, value in mathia_binding.items() if key != "corpus_root"
        },
        "selection": selection,
        "lean_environments": {
            workload: {
                key: value
                for key, value in environment.items()
                if key not in {"project_root", "project_head"}
            }
            for workload, environment in environments.items()
        },
        "generation_candidates": len(generations),
        "verification_candidates": len(verifications),
        "parser_replay": {
            "candidate_count": len(parser_audits),
            "all_exact": True,
            "canonical_output_token_count_method": (
                "pinned tokenizer encode of exact parsed/normalized bytes"
            ),
            "stored_legacy_parser_token_count_mismatch_count": sum(
                int(row["parsed_final_token_count_stored"])
                != int(row["parsed_final_token_count"])
                for row in parser_audits
            ),
            "raw_suffix_mismatch_count": sum(
                not row["parser_final_content_is_exact_raw_suffix"]
                for row in generations
            ),
            "raw_suffix_identity_blocking": False,
        },
        "arm_workload_summaries": summaries,
        "matched_task_table": matched_table,
        "conclusion": conclusion,
        "cost": {
            "verification_wall_time_seconds": sum(
                float(row["wall_time_seconds"]) for row in verification_segments
            ),
            "generation_segment_count": len(generation_segments),
            "verification_segment_count": len(verification_segments),
        },
        "artifact_integrity": {
            "generations_jsonl_sha256": _file_sha256(
                artifact_dir / "generations.jsonl"
            ),
            "verifications_jsonl_sha256": _file_sha256(
                artifact_dir / "verifications.jsonl"
            ),
            "generation_segments_jsonl_sha256": _file_sha256(
                artifact_dir / "generation-segments.jsonl"
            ),
            "verification_segments_jsonl_sha256": _file_sha256(
                artifact_dir / "verification-segments.jsonl"
            ),
            "raw_artifacts_git_ignored": True,
        },
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(evidence_dir / "continuation-results.json", result)
    (evidence_dir / "CONTINUATION.md").write_text(
        _render_continuation_readme(result), encoding="utf-8"
    )
    return result


def _continuation_summary(
    rows: Sequence[dict[str, Any]],
    segments: Sequence[dict[str, Any]],
    *,
    include_segment_cost: bool,
) -> dict[str, Any]:
    count = len(rows)
    parsed = [str(row.get("parsed_final_exact") or "") for row in rows]
    normalized = [str(row.get("normalized_final_exact") or "") for row in rows]
    strict_categories = Counter(
        row["verification"]["strict_parsed_interface"]["category"] for row in rows
    )
    deployed_categories = Counter(
        row["verification"]["deployed_normalized_interface"]["category"] for row in rows
    )
    total_raw_tokens = sum(int(row["raw_response_token_count"]) for row in rows)
    candidate_latency = sum(float(row["generation_latency_seconds"]) for row in rows)
    result = {
        "candidate_count": count,
        "configured_reasoning_tokens": sorted(
            {int(row["max_reasoning_tokens"]) for row in rows}
        ),
        "actual_reasoning_tokens": _distribution(
            [int(row["reasoning_token_count"]) for row in rows]
        ),
        "reasoning_exit_counts": dict(
            sorted(Counter(str(row["reasoning_exit"]) for row in rows).items())
        ),
        "nonempty_parsed_final": _count_fraction(
            sum(bool(value) for value in parsed), count
        ),
        "parsed_final_tokens": _distribution(
            [int(row["parsed_final_token_count"]) for row in rows]
        ),
        "normalized_final_tokens": _distribution(
            [int(row["normalized_final_token_count"]) for row in rows]
        ),
        "strict_parsed_interface": {
            "verified": _count_fraction(strict_categories["verified"], count),
            "category_counts": dict(sorted(strict_categories.items())),
        },
        "deployed_normalized_interface": {
            "verified": _count_fraction(deployed_categories["verified"], count),
            "category_counts": dict(sorted(deployed_categories.items())),
        },
        "wrapper_normalization_changed": _count_fraction(
            sum(bool(row["normalization_applied"]) for row in rows), count
        ),
        "verification_outcome_changed_by_normalization": _count_fraction(
            sum(
                bool(
                    row["verification"]["verification_outcome_changed_by_normalization"]
                )
                for row in rows
            ),
            count,
        ),
        "finish_reason_counts": dict(
            sorted(Counter(str(row["finish_reason"]) for row in rows).items())
        ),
        "format_diagnostics": {
            "parsed": _format_diagnostics(parsed),
            "normalized": _format_diagnostics(normalized),
        },
        "total_reasoning_tokens": sum(
            int(row["reasoning_token_count"]) for row in rows
        ),
        "total_parsed_final_tokens": sum(
            int(row["parsed_final_token_count"]) for row in rows
        ),
        "total_normalized_final_tokens": sum(
            int(row["normalized_final_token_count"]) for row in rows
        ),
        "total_raw_tokens": total_raw_tokens,
        "summed_candidate_latency_seconds": candidate_latency,
        "throughput_raw_tokens_per_summed_candidate_second": (
            total_raw_tokens / candidate_latency if candidate_latency else None
        ),
    }
    if include_segment_cost:
        wall_time = sum(float(row["segment_wall_time_seconds"]) for row in segments)
        result["generation_wall_time_seconds"] = wall_time
        result["throughput_raw_tokens_per_wall_second"] = (
            total_raw_tokens / wall_time if wall_time else None
        )
        result["peak_gpu_memory_bytes"] = max(
            (int(row["gpu_memory_peak_bytes"]) for row in segments), default=None
        )
    return result


def _format_diagnostics(values: Sequence[str]) -> dict[str, int]:
    return {
        "markdown_fence": sum("```" in value for value in values),
        "repeated_declaration_or_by": sum(
            _repeats_declaration_or_by(value) for value in values
        ),
        "sorry_or_admit": sum(
            bool(re.search(r"\b(?:sorry|admit)\b", value, flags=re.IGNORECASE))
            for value in values
        ),
        "apparent_natural_language": sum(
            _apparent_natural_language(value) for value in values
        ),
    }


def _count_fraction(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "fraction": count / total if total else 0.0}


def _matched_task_table(
    continuation: ThinkingBudgetContinuationConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    selected: Sequence[SelectedTask],
    generation_by_id: Mapping[str, dict[str, Any]],
    verification_by_id: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for selected_task in selected:
        arms: dict[str, Any] = {}
        for arm in CONTINUATION_ARMS:
            candidate_id, _ = continuation_candidate_identity(
                continuation, scaling, stage1, selected_task, arm
            )
            generation = generation_by_id[candidate_id]
            verification = verification_by_id[candidate_id]
            arms[arm] = {
                "candidate_id": candidate_id,
                "configured_reasoning_tokens": generation["max_reasoning_tokens"],
                "actual_reasoning_tokens": generation["reasoning_token_count"],
                "reasoning_exit": generation["reasoning_exit"],
                "parsed_final_nonempty": bool(generation["parsed_final_exact"]),
                "parsed_final_tokens": generation["parsed_final_token_count"],
                "normalized_final_tokens": generation["normalized_final_token_count"],
                "normalization_applied": generation["normalization_applied"],
                "strict_lean_category": verification["strict_parsed_interface"][
                    "category"
                ],
                "deployed_lean_category": verification["deployed_normalized_interface"][
                    "category"
                ],
                "finish_reason": generation["finish_reason"],
                "raw_tokens": generation["raw_response_token_count"],
                "generation_latency_seconds": generation["generation_latency_seconds"],
            }
        table.append(
            {
                "workload": selected_task.task.workload,
                "task_id": selected_task.task.task_id,
                "frozen_global_index": selected_task.frozen_global_index,
                "frozen_workload_index": selected_task.frozen_workload_index,
                "rendered_prompt_token_count": (
                    selected_task.rendered_prompt_token_count
                ),
                "arms": arms,
            }
        )
    return table


def _continuation_conclusion(
    summaries: Mapping[str, Mapping[str, dict[str, Any]]],
    matched_table: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    b4_verified = {
        row["task_id"]
        for row in matched_table
        if row["arms"]["B4"]["deployed_lean_category"] == "verified"
    }
    longer_verified = {
        row["task_id"]
        for row in matched_table
        if any(
            row["arms"][arm]["deployed_lean_category"] == "verified"
            for arm in ("B8", "B16")
        )
    }
    longer_unique = sorted(longer_verified - b4_verified)
    all_final = all(
        int(summaries[arm]["combined"]["nonempty_parsed_final"]["count"])
        == int(summaries[arm]["combined"]["candidate_count"])
        for arm in CONTINUATION_ARMS
    )
    final_gain = max(
        int(summaries[arm]["combined"]["nonempty_parsed_final"]["count"])
        for arm in ("B8", "B16")
    ) > int(summaries["B4"]["combined"]["nonempty_parsed_final"]["count"])
    longer_costs_more = all(
        int(summaries[arm]["combined"]["total_raw_tokens"])
        > int(summaries["B4"]["combined"]["total_raw_tokens"])
        for arm in ("B8", "B16")
    )
    if longer_unique or final_gain:
        category = "budget_control_works_and_longer_thinking_is_promising"
    elif all_final and longer_costs_more:
        category = "budget_control_works_but_longer_thinking_only_adds_cost"
    else:
        category = "budget_control_works_but_no_quality_signal_yet"
    return {
        "category": category,
        "stage1_no_final_regime_eliminated": all_final,
        "all_48_produced_nonempty_parsed_final": all_final,
        "longer_budget_unique_deployed_verified_tasks": longer_unique,
        "longer_budget_final_production_gain_over_B4": final_gain,
        "longer_budgets_used_more_raw_tokens_than_B4": longer_costs_more,
        "preflight_not_powered_for_pass_at_k": True,
        "larger_experiment_authorized": False,
    }


def _compact_continuation_generation(row: Mapping[str, Any]) -> dict[str, Any]:
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
        "parsed_final_sha256",
        "parsed_final_token_count",
        "parsed_final_token_count_stored",
        "normalized_final_sha256",
        "normalized_final_token_count",
        "normalized_final_token_count_stored",
        "normalization_applied",
        "normalization_idempotent",
        "parser_replay_matches_stored",
        "normalization_replay_matches_stored",
        "parser_final_content_is_exact_raw_suffix",
        "final_has_reasoning_marker",
        "finish_reason",
        "generation_latency_seconds",
    )
    return {key: row[key] for key in keys}


def _write_continuation_gate_evidence(
    output_path: Path, evidence: Mapping[str, Any]
) -> None:
    _atomic_write_json(output_path, dict(evidence))
    output_path.with_suffix(".md").write_text(
        _render_continuation_gate_readme(evidence), encoding="utf-8"
    )


def _render_continuation_gate_readme(evidence: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Qwen3.5-4B thinking-budget continuation gate",
            "",
            (
                f"**OBSERVED:** `{evidence['status']}` under the pinned `qwen3` "
                "canonical parsed-output contract and "
                "`lean-wrapper-normalization-v1`."
            ),
            "",
            (
                "The 32/64/128-token bounded probes retain raw token IDs, replay "
                "the parser deterministically, preserve exact parsed/normalized "
                "bytes and hashes, and treat raw-suffix identity only as a "
                "diagnostic."
            ),
            "",
            (
                "Scientific generation is authorized only when every continuation "
                "gate check passes. Historical Stage 1 and Stage 2a artifacts remain "
                "unchanged."
            ),
            "",
        ]
    )


def _render_continuation_readme(result: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3.5-4B thinking-budget scaling continuation",
        "",
        (
            "**OBSERVED:** the revised canonical parsed-output gate passed and all "
            "48 frozen B4/B8/B16 candidates were generated and verified under both "
            "the strict parsed and deployed normalized interfaces."
        ),
        "",
        "| Arm | Non-empty final | Strict verified | Deployed verified | Wrapper changed | Raw tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in CONTINUATION_ARMS:
        summary = result["arm_workload_summaries"][arm]["combined"]
        lines.append(
            f"| {arm} | {summary['nonempty_parsed_final']['count']}/16 | "
            f"{summary['strict_parsed_interface']['verified']['count']}/16 | "
            f"{summary['deployed_normalized_interface']['verified']['count']}/16 | "
            f"{summary['wrapper_normalization_changed']['count']}/16 | "
            f"{summary['total_raw_tokens']} |"
        )
    lines.extend(
        [
            "",
            f"**OBSERVED conclusion:** `{result['conclusion']['category']}`.",
            "",
            (
                "This remains an operational preflight, not a powered pass@k study. "
                "It does not alter the historical Stage 1/Stage 2a targets or "
                "authorize a larger experiment. Raw artifacts remain Git-ignored."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
