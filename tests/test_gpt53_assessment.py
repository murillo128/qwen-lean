import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from qwen_lean.gpt53_assessment import (
    ALLOWED_REASONING_EFFORTS,
    API_KEY_ENVIRONMENT_VARIABLES,
    MODEL_ID,
    REASONING_EFFORT,
    GPT53Config,
    ProgressLogger,
    audit_jsonl_text,
    build_child_argv,
    candidate_records_manifest_sha256,
    compact_workload_evidence,
    is_retryable_infrastructure_failure,
    load_existing_candidate,
    load_prior_retryable_attempts,
    paired_outcome_comparison,
    read_final_message,
    render_codex_prompt,
    sanitize_child_environment,
    should_retry_result_category,
    validate_arm_artifact_path,
    validate_child_argv,
    validate_isolated_workdir,
)
from qwen_lean.schema import TaskRecord

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/gpt53-assessment.json"
LOW_CONFIG_PATH = ROOT / "config/gpt53-low-assessment.json"


def _valid_jsonl() -> str:
    events = [
        {"type": "thread.started", "thread_id": "thread-test"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": ""}},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "exact h"},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
    ]
    return "".join(json.dumps(event) + "\n" for event in events)


def test_config_and_argv_freeze_exact_model_and_reasoning_pin(tmp_path: Path) -> None:
    config = GPT53Config.load(CONFIG_PATH)
    argv = build_child_argv(
        config,
        codex_binary=tmp_path / "codex",
        final_message_path=tmp_path / "final.txt",
    )

    model_index = argv.index("--model")
    assert argv[model_index + 1] == MODEL_ID == "gpt-5.3-codex-spark"
    assert f'model_reasoning_effort="{REASONING_EFFORT}"' in argv
    assert "--ephemeral" in argv
    assert "--json" in argv
    assert "--output-last-message" in argv
    assert argv[-1] == "-"
    assert "resume" not in argv
    assert argv.count("--model") == 1
    assert all("theorem" not in value for value in argv)


def test_low_config_changes_only_effort_and_uses_exact_low_argv(tmp_path: Path) -> None:
    xhigh = GPT53Config.load(CONFIG_PATH)
    low = GPT53Config.load(LOW_CONFIG_PATH)
    argv = build_child_argv(
        low,
        codex_binary=tmp_path / "codex",
        final_message_path=tmp_path / "final.txt",
    )

    assert ALLOWED_REASONING_EFFORTS == {"low", "xhigh"}
    assert xhigh.reasoning_effort == REASONING_EFFORT == "xhigh"
    assert low.reasoning_effort == "low"
    assert low.value["model"]["id"] == xhigh.value["model"]["id"]
    assert low.value["benchmark"] == xhigh.value["benchmark"]
    assert low.value["workloads"] == xhigh.value["workloads"]
    assert low.value["prompt"] == xhigh.value["prompt"]
    assert low.value["codex"] == xhigh.value["codex"]
    assert argv.count("--model") == 1
    assert argv[argv.index("--model") + 1] == "gpt-5.3-codex-spark"
    assert 'model_reasoning_effort="low"' in argv
    assert 'model_reasoning_effort="xhigh"' not in argv


@pytest.mark.parametrize("effort", [None, "", "medium", "high", "LOW"])
def test_config_rejects_unapproved_or_implicit_reasoning_effort(
    tmp_path: Path, effort: str | None
) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if effort is None:
        value["model"].pop("reasoning_effort")
    else:
        value["model"]["reasoning_effort"] = effort
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly selected"):
        GPT53Config.load(path)


def test_child_argv_cannot_omit_or_change_explicit_model() -> None:
    with pytest.raises(ValueError, match="omits"):
        validate_child_argv(["codex", "exec", "--json"])
    with pytest.raises(ValueError, match="exactly gpt-5.3-codex-spark"):
        validate_child_argv(["codex", "exec", "--model", "gpt-5.3-codex"])
    with pytest.raises(ValueError, match="required low override"):
        validate_child_argv(
            [
                "codex",
                "exec",
                "--model",
                MODEL_ID,
                "-c",
                'model_reasoning_effort="xhigh"',
            ],
            reasoning_effort="low",
        )


def test_config_rejects_any_model_default_or_substitution(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["model"]["id"] = "gpt-5.3-codex"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly gpt-5.3-codex-spark"):
        GPT53Config.load(path)


def test_low_arm_cannot_use_or_overwrite_xhigh_artifact_paths() -> None:
    config = GPT53Config.load(LOW_CONFIG_PATH)

    validate_arm_artifact_path(config, ROOT / "artifacts/gpt53-spark-low/full")
    validate_arm_artifact_path(config, ROOT / "evidence/gpt53-spark-low", evidence=True)
    with pytest.raises(ValueError, match="gpt53-spark-low"):
        validate_arm_artifact_path(config, ROOT / "artifacts/gpt53-spark/full")
    with pytest.raises(ValueError, match="gpt53-spark-low"):
        validate_arm_artifact_path(config, ROOT / "evidence/gpt53-spark", evidence=True)


def test_low_arm_rejects_namespace_symlink_to_xhigh(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = project_root / "config/low.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        LOW_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    xhigh_root = project_root / "artifacts/gpt53-spark"
    xhigh_root.mkdir(parents=True)
    low_root = project_root / "artifacts/gpt53-spark-low"
    low_root.symlink_to(xhigh_root, target_is_directory=True)
    config = GPT53Config.load(config_path)

    with pytest.raises(ValueError, match="gpt53-spark-low"):
        validate_arm_artifact_path(config, low_root / "full")


def test_child_environment_removes_api_keys_and_lean_path(tmp_path: Path) -> None:
    forbidden = tmp_path / "forbidden"
    safe = tmp_path / "safe"
    forbidden.mkdir()
    safe.mkdir()
    lean = forbidden / "lean"
    lean.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    lean.chmod(0o755)
    source = {
        "PATH": os.pathsep.join((str(forbidden), str(safe))),
        "OPENAI_API_KEY": "must-not-pass",
        "CODEX_API_KEY": "must-not-pass",
        "SAFE_VALUE": "retained",
    }

    environment = sanitize_child_environment(source)

    assert API_KEY_ENVIRONMENT_VARIABLES.isdisjoint(environment)
    assert environment["PATH"] == str(safe)
    assert environment["SAFE_VALUE"] == "retained"


def test_candidate_working_directory_must_be_empty_and_external(tmp_path: Path) -> None:
    project = tmp_path / "project"
    benchmark = tmp_path / "benchmark"
    isolated = tmp_path / "isolated"
    project.mkdir()
    benchmark.mkdir()
    isolated.mkdir()

    validate_isolated_workdir(isolated, project_root=project, benchmark_root=benchmark)
    with pytest.raises(ValueError, match="inside qwen-lean"):
        validate_isolated_workdir(
            project / "child", project_root=project, benchmark_root=benchmark
        )
    (isolated / "unexpected").write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        validate_isolated_workdir(
            isolated, project_root=project, benchmark_root=benchmark
        )


def test_jsonl_audit_accepts_reasoning_and_final_message_only() -> None:
    audit = audit_jsonl_text(_valid_jsonl())

    assert audit.valid
    assert audit.tool_event_count == 0
    assert audit.thread_id == "thread-test"
    assert audit.usage == {"input_tokens": 10, "output_tokens": 20}


@pytest.mark.parametrize(
    "event_type",
    ["command_execution", "file_change", "web_search", "mcp_tool_call"],
)
def test_jsonl_tool_event_invalidates_execution(event_type: str) -> None:
    tool_event = json.dumps({"type": "item.completed", "item": {"type": event_type}})
    audit = audit_jsonl_text(_valid_jsonl() + tool_event + "\n")

    assert not audit.valid
    assert audit.tool_event_count == 1
    assert any("external-tool" in violation for violation in audit.violations)


@pytest.mark.parametrize(
    "marker",
    [
        "selected gpt-5.3-codex",
        "selected gpt-5.6-codex",
        "falling back to another model",
        "model migration is active",
        "model substitution occurred",
    ],
)
def test_model_fallback_migration_and_gpt56_markers_fail_closed(marker: str) -> None:
    audit = audit_jsonl_text(_valid_jsonl(), marker)

    assert not audit.valid
    assert any("integrity marker" in violation for violation in audit.violations)


def test_other_effective_model_in_jsonl_fails_closed() -> None:
    events = _valid_jsonl() + json.dumps(
        {"type": "metadata", "effective_model": "some-other-model"}
    )
    audit = audit_jsonl_text(events)

    assert not audit.valid
    assert any("unexpected model value" in violation for violation in audit.violations)


def test_only_output_last_message_file_becomes_raw_candidate(tmp_path: Path) -> None:
    path = tmp_path / "final-message.txt"
    raw = "```lean\nby\n  exact h\n```\nprose is not repaired\n"
    path.write_text(raw, encoding="utf-8")

    assert read_final_message(path) == raw
    assert "agent_message" in _valid_jsonl()


def test_lean_rejection_never_triggers_infrastructure_retry() -> None:
    assert not should_retry_result_category("lean_rejected")
    assert not should_retry_result_category("verifier_timeout")
    assert should_retry_result_category("verifier_error")
    audit = audit_jsonl_text(_valid_jsonl())
    assert is_retryable_infrastructure_failure(
        exit_code=1,
        stderr_text="connection reset by peer",
        audit=audit,
    )
    tool_audit = audit_jsonl_text(
        _valid_jsonl()
        + json.dumps({"type": "item.completed", "item": {"type": "command_execution"}})
        + "\n"
    )
    assert not is_retryable_infrastructure_failure(
        exit_code=1,
        stderr_text="connection reset by peer",
        audit=tool_audit,
    )


def test_context_window_exhaustion_is_retryable_infrastructure() -> None:
    failed_jsonl = "".join(
        json.dumps(event) + "\n"
        for event in (
            {"type": "thread.started", "thread_id": "thread-test"},
            {"type": "turn.started"},
            {
                "type": "error",
                "message": (
                    "stream disconnected before completion: Incomplete response "
                    "returned, reason: max_output_tokens"
                ),
            },
            {
                "type": "turn.failed",
                "error": {
                    "message": ("Codex ran out of room in the model's context window.")
                },
            },
        )
    )
    audit = audit_jsonl_text(failed_jsonl)

    assert is_retryable_infrastructure_failure(
        exit_code=1,
        stderr_text=failed_jsonl,
        audit=audit,
    )


def test_model_capacity_is_retryable_infrastructure() -> None:
    failed_jsonl = "".join(
        json.dumps(event) + "\n"
        for event in (
            {"type": "thread.started", "thread_id": "thread-test"},
            {"type": "turn.started"},
            {
                "type": "error",
                "message": "Selected model is at capacity. Please try a different model.",
            },
            {
                "type": "turn.failed",
                "error": {
                    "message": (
                        "Selected model is at capacity. Please try a different model."
                    )
                },
            },
        )
    )
    audit = audit_jsonl_text(failed_jsonl)

    assert is_retryable_infrastructure_failure(
        exit_code=1,
        stderr_text=failed_jsonl,
        audit=audit,
    )


def test_resume_preserves_and_reclassifies_retryable_failed_attempt(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    attempt_dir = output_dir / "candidates" / "task" / "attempt-1"
    attempt_dir.mkdir(parents=True)
    events = "".join(
        json.dumps(event) + "\n"
        for event in (
            {"type": "thread.started", "thread_id": "thread-test"},
            {"type": "turn.started"},
            {
                "type": "error",
                "message": "Incomplete response returned, reason: max_output_tokens",
            },
            {"type": "turn.failed", "error": {"message": "context exhausted"}},
        )
    )
    (attempt_dir / "events.jsonl").write_text(events, encoding="utf-8")
    (attempt_dir / "stderr.log").write_text("", encoding="utf-8")
    failure_path = attempt_dir.parent / "failure.json"
    attempt = {
        "accepted": False,
        "exit_code": 1,
        "stdout_jsonl_path": "candidates/task/attempt-1/events.jsonl",
        "stderr_path": "candidates/task/attempt-1/stderr.log",
        "integrity_errors": ["turn.failed"],
        "audit": {"tool_event_count": 0},
    }
    failure_path.write_text(
        json.dumps(
            {
                "schema_version": "gpt53-spark-assessment-run-v1",
                "status": "execution_contract_failure",
                "contract_fingerprint": "contract",
                "workload_id": "workload",
                "task_id": "task",
                "candidate_index": 0,
                "prompt_sha256": "prompt",
                "attempts": [attempt],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_prior_retryable_attempts(
        failure_path,
        output_dir=output_dir,
        contract_fingerprint="contract",
        workload_id="workload",
        task_id="task",
        prompt_sha256="prompt",
    )

    assert loaded == [attempt]


def test_resume_preserves_verifier_timeout_as_final_proof_failure(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    candidate_dir = output_dir / "candidates" / "task"
    attempt_dir = candidate_dir / "attempt-1"
    attempt_dir.mkdir(parents=True)
    raw = "exact h\n"
    raw_path = attempt_dir / "final-message.txt"
    raw_path.write_text(raw, encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw.encode()).hexdigest()
    result = {
        "task_id": "task",
        "candidate_id": "gpt53-spark-0",
        "candidate_index": 0,
        "candidate_text": raw,
        "category": "verifier_timeout",
        "lean_exit_code": None,
        "diagnostics": {"stdout": "", "stderr": ""},
        "generation_latency_seconds": 1.0,
        "verification_latency_seconds": 30.0,
        "total_latency_seconds": 31.0,
        "generated_token_count": None,
        "finish_reason": "turn_completed",
    }
    record_path = candidate_dir / "candidate.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": "gpt53-spark-assessment-run-v1",
                "status": "accepted",
                "contract_fingerprint": "contract",
                "workload_id": "workload",
                "task_id": "task",
                "candidate_index": 0,
                "prompt_sha256": "prompt",
                "requested_model": MODEL_ID,
                "requested_reasoning_effort": REASONING_EFFORT,
                "attempt_count": 1,
                "attempts": [
                    {
                        "accepted": True,
                        "audit": {"valid": True, "tool_event_count": 0},
                    }
                ],
                "accepted_attempt": 1,
                "raw_candidate_path": "candidates/task/attempt-1/final-message.txt",
                "raw_candidate_sha256": raw_sha256,
                "verification": {
                    "category": "verifier_timeout",
                    "lean_exit_code": None,
                    "latency_seconds": 30.0,
                },
                "result": result,
            }
        ),
        encoding="utf-8",
    )

    _, loaded = load_existing_candidate(
        record_path,
        output_dir=output_dir,
        contract_fingerprint="contract",
        task=TaskRecord(
            id="task",
            preamble="import Init",
            declaration="theorem task (p : Prop) (h : p) : p",
            declaration_name="task",
        ),
        prompt_sha256="prompt",
    )

    assert loaded.category == "verifier_timeout"
    assert loaded.candidate_text == raw
    assert not should_retry_result_category(loaded.category)


def test_codex_prompt_retains_whole_proof_prefix_and_raw_continuation_contract() -> (
    None
):
    config = GPT53Config.load(CONFIG_PATH)
    task = TaskRecord(
        id="identity",
        preamble="import Init",
        declaration="theorem identity (p : Prop) (h : p) : p",
        declaration_name="identity",
    )

    prompt = render_codex_prompt(config, task)

    assert prompt.endswith(f"{task.declaration} := by\n  ")
    assert "only Lean code continuing after `by`" in prompt
    assert "Do not call or use any external tools" in prompt
    assert "sorry" in prompt and "admit" in prompt


def test_progress_log_exposes_started_heartbeat_and_completion_state(
    tmp_path: Path,
) -> None:
    console = io.StringIO()
    path = tmp_path / "run-log.jsonl"
    logger = ProgressLogger(path, console=console)
    common = {
        "workload_id": "minif2f-valid-dev16-v1",
        "task_id": "identity",
        "candidate_index": 0,
        "total": 16,
        "pid": 123,
    }
    logger.emit(
        {
            **common,
            "event": "candidate_started",
            "completed": 0,
            "requested_model": MODEL_ID,
            "requested_reasoning_effort": REASONING_EFFORT,
            "argv": "codex exec --model gpt-5.3-codex-spark",
        }
    )
    logger.emit(
        {
            **common,
            "event": "candidate_heartbeat",
            "completed": 0,
            "elapsed_seconds": 15.0,
            "event_count": 2,
        }
    )
    logger.emit(
        {
            **common,
            "event": "candidate_completed",
            "completed": 1,
            "exit_code": 0,
            "elapsed_seconds": 20.0,
            "tool_event_count": 0,
            "usage": {"output_tokens": 20},
            "final_message_sha256": "abc",
        }
    )

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "candidate_started",
        "candidate_heartbeat",
        "candidate_completed",
    ]
    assert events[0]["requested_model"] == MODEL_ID
    assert events[0]["requested_reasoning_effort"] == REASONING_EFFORT
    assert events[1]["pid"] == 123 and events[1]["event_count"] == 2
    assert events[2]["completed"] == 1 and events[2]["tool_event_count"] == 0
    assert "running pid=123" in console.getvalue()


def test_compact_evidence_freezes_timeout_as_proof_failure(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps({"verifier_timeout_seconds": 30.0}), encoding="utf-8"
    )
    (tmp_path / "run-log.jsonl").write_text(
        json.dumps(
            {
                "event": "candidate_completed",
                "task_id": "task",
                "completed": 1,
                "total": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    execution_integrity = {
        "accepted_candidate_execution_count": 1,
        "total_child_attempt_count": 2,
        "child_failure_count": 1,
        "retry_count": 1,
    }
    evidence = compact_workload_evidence(
        {
            "workload_id": "workload",
            "task_count": 1,
            "candidate_count": 1,
            "candidates_per_task": 1,
            "tasks_with_verified_candidate": {"count": 0, "fraction": 0.0},
            "pass_at_k": {"pass@1": 0.0},
            "category_counts": {"verifier_timeout": 1},
            "verifier_timeout_count": 1,
            "infrastructure_error_count": 0,
            "execution_integrity": execution_integrity,
        },
        tmp_path,
    )

    assert evidence["verifier_policy"] == {
        "timeout_seconds": 30.0,
        "verifier_timeout_semantics": "unsuccessful_proof_attempt",
        "verifier_timeout_is_infrastructure_error": False,
        "verifier_timeout_triggers_candidate_regeneration": False,
        "verifier_timeout_triggers_verification_retry": False,
    }
    assert evidence["child_process_retry_accounting"] == execution_integrity


def test_paired_comparison_requires_exact_ids_and_uses_discordant_pairs() -> None:
    task_ids = ["both", "xhigh-only", "low-only", "neither"]
    low = {
        "both": "verified",
        "xhigh-only": "lean_rejected",
        "low-only": "verified",
        "neither": "verifier_timeout",
    }
    xhigh = {
        "both": "verified",
        "xhigh-only": "verified",
        "low-only": "lean_rejected",
        "neither": "empty_candidate",
    }

    comparison = paired_outcome_comparison(
        low,
        xhigh,
        expected_task_ids=task_ids,
        xhigh_pass_at_1=0.5,
    )

    assert comparison["paired_outcome_table"] == {
        "low_fail_xhigh_fail": 1,
        "low_fail_xhigh_verified": 1,
        "low_verified_xhigh_fail": 1,
        "low_verified_xhigh_verified": 1,
    }
    assert comparison["paired_outcomes"] == {
        "solved_by_both": 1,
        "solved_only_by_xhigh": 1,
        "solved_only_by_low": 1,
        "solved_by_neither": 1,
    }
    assert comparison["paired_binary_test"] == {
        "method": "exact_two_sided_mcnemar_binomial",
        "discordant_pair_count": 2,
        "p_value": 1.0,
        "interpretation": "descriptive_uncertainty_not_a_hard_success_gate",
    }

    with pytest.raises(ValueError, match="task IDs differ"):
        paired_outcome_comparison(
            {key: value for key, value in low.items() if key != "neither"},
            xhigh,
            expected_task_ids=task_ids,
            xhigh_pass_at_1=0.5,
        )


def test_candidate_record_manifest_hash_changes_with_records(tmp_path: Path) -> None:
    record_path = tmp_path / "candidates/task/candidate.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text('{"status":"accepted"}\n', encoding="utf-8")
    original = candidate_records_manifest_sha256(tmp_path)

    record_path.write_text('{"status":"changed"}\n', encoding="utf-8")

    assert candidate_records_manifest_sha256(tmp_path) != original
