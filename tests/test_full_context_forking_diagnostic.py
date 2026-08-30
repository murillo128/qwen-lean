from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.counterfactual_forking_assessment import (
    ParentTrajectory,
    fork_states,
)
from qwen_lean.full_context_forking_diagnostic import (
    BRANCH_ATTEMPT_SCHEMA,
    CHECKPOINT_REVIEW_SCHEMA,
    COUNTERFACTUAL_RESULTS_SHA256,
    FullContextForkingConfig,
    _branch_attempt_counts,
    _branch_attempt_summary,
    _classify_outcome,
    _load_attempt_recovery,
    _next_calibration_action,
    _prepare_attempt_recovery,
    _reasoning_transition_index,
    _validate_checkpoint_review,
    _validate_scientific_runtime_hardware,
    full_context_requests,
)
from qwen_lean.native_thinking_assessment import MathiaTask, _file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-full-context-forking.json"
CALIBRATION_PATH = ROOT / "evidence/qwen35-full-context-forking/calibration.json"
CALIBRATION_REVIEW_PATH = (
    ROOT / "evidence/qwen35-full-context-forking/calibration-review.json"
)
ATTEMPT_RECOVERY_PATH = (
    ROOT / "evidence/qwen35-full-context-forking/attempt-recovery.json"
)


def _config() -> FullContextForkingConfig:
    return FullContextForkingConfig.load(CONFIG_PATH)


def _attempt(index: int, length: int, status: str) -> dict[str, object]:
    return {
        "attempt_index": index,
        "context_length": length,
        "status": status,
    }


def _parent() -> ParentTrajectory:
    task = MathiaTask(
        task_id="amc12a_2019_p21",
        workload="minif2f-valid-clean-v2",
        preamble="import Mathlib",
        declaration="theorem amc12a_2019_p21 : True",
        declaration_name="amc12a_2019_p21",
        intuition="trivial",
        intuition_sha256="i" * 64,
        theorem_sha256="t" * 64,
    )
    return ParentTrajectory(
        ordinal=0,
        task=task,
        handoff={
            "candidate_id": "native-thinking-752f2a6728b47b5a89263445496c7b80",
            "raw_response_sha256": "r" * 64,
            "raw_response_token_ids_sha256": "p" * 64,
        },
        record={},
        record_sha256="g" * 64,
        raw_response_token_ids=tuple(range(4096)),
        rendered_prompt_token_ids=tuple(range(253)),
        rendered_prompt_sha256="q" * 64,
        states=fork_states(4096),
        parser_parity={"status": "passed"},
    )


def test_config_binds_reviewed_issue92_without_mutating_it() -> None:
    config = _config()
    issue92 = ROOT / config.reviewed_target["counterfactual_results_path"]
    assert _file_sha256(issue92) == COUNTERFACTUAL_RESULTS_SHA256
    assert config.diagnostic_target == {
        "workload": "minif2f-valid-clean-v2",
        "task_id": "amc12a_2019_p21",
        "parent_candidate_id": "native-thinking-752f2a6728b47b5a89263445496c7b80",
        "parent_ordinal": 0,
    }


def test_calibration_progresses_then_refines_and_repeat_confirms() -> None:
    config = _config()
    attempts: list[dict[str, object]] = []
    assert _next_calibration_action(config, attempts) == ("progressive", 32768)
    attempts.append(_attempt(0, 32768, "passed"))
    assert _next_calibration_action(config, attempts) == ("progressive", 49152)
    attempts.append(_attempt(1, 49152, "failed"))
    assert _next_calibration_action(config, attempts) == ("refinement", 40960)
    attempts.extend(
        [
            _attempt(2, 40960, "passed"),
            _attempt(3, 45056, "passed"),
            _attempt(4, 47104, "passed"),
            _attempt(5, 48128, "failed"),
        ]
    )
    assert _next_calibration_action(config, attempts) == ("confirmation", 47104)
    attempts.append(_attempt(6, 47104, "passed"))
    assert _next_calibration_action(config, attempts) is None


def test_failed_confirmation_cannot_remain_selected() -> None:
    config = _config()
    attempts = [
        _attempt(0, 32768, "passed"),
        _attempt(1, 49152, "failed"),
        _attempt(2, 40960, "passed"),
        _attempt(3, 45056, "passed"),
        _attempt(4, 47104, "passed"),
        _attempt(5, 48128, "failed"),
        _attempt(6, 47104, "failed"),
    ]
    assert _next_calibration_action(config, attempts) == ("refinement", 46080)


def test_scientific_requests_are_exactly_42_and_use_m_minus_q_minus_p(
    tmp_path: Path,
) -> None:
    config = _config()
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}\n", encoding="utf-8")
    parent = _parent()
    selected = 47104
    requests = full_context_requests(
        config,
        parent,
        selected_context_length=selected,
        calibration_evidence_path=calibration,
    )
    assert len(requests) == 42
    assert [(request.state.label, request.seed) for request in requests[:7]] == [
        ("P0", 100),
        ("P0", 101),
        ("P0", 102),
        ("P0", 103),
        ("P0", 104),
        ("P0", 105),
        ("P15", 100),
    ]
    assert all(
        request.max_tokens
        == selected - len(parent.rendered_prompt_token_ids) - request.state.prefix_len
        for request in requests
    )
    assert requests[-1].state.label == "P90"
    assert requests[-1].seed == 105


def test_generation_requires_published_pass_review_of_exact_calibration(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"selected_max_context_length": 47104}\n', encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_REVIEW_SCHEMA,
                "verdict": "PASS",
                "reviewed_commit": "a" * 40,
                "calibration_evidence_sha256": _file_sha256(calibration),
                "published_review_url": "https://github.example/review",
            }
        ),
        encoding="utf-8",
    )
    assert _validate_checkpoint_review(calibration, review)["verdict"] == "PASS"
    value = json.loads(review.read_text(encoding="utf-8"))
    value["verdict"] = "FAIL"
    review.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not PASS"):
        _validate_checkpoint_review(calibration, review)


def test_scientific_generation_requires_exact_calibrated_gpu_identity() -> None:
    calibration = {
        "attempts": [
            {"gpu_memory_total_bytes": 12_878_610_432},
            {"gpu_memory_total_bytes": 12_878_610_432},
        ]
    }
    expected_runtime = {
        "cuda_device": "NVIDIA GeForce RTX 4070 Ti",
        "nvml_gpu_memory_total_bytes": 12_878_610_432,
    }
    assert _validate_scientific_runtime_hardware(expected_runtime, calibration) == {
        "gpu_name_fragment": "RTX 4070 Ti",
        "gpu_memory_total_bytes": 12_878_610_432,
        "status": "matched",
    }

    with pytest.raises(RuntimeError, match="exact calibrated RTX 4070 Ti"):
        _validate_scientific_runtime_hardware(
            {
                "cuda_device": "NVIDIA RTX 4000 Ada Generation",
                "nvml_gpu_memory_total_bytes": 21_469_052_928,
            },
            calibration,
        )
    with pytest.raises(RuntimeError, match="exact calibrated RTX 4070 Ti"):
        _validate_scientific_runtime_hardware(
            {
                "cuda_device": "NVIDIA GeForce RTX 4070 Ti",
                "nvml_gpu_memory_total_bytes": 12_000_000_000,
            },
            calibration,
        )


def _recovery_requests():
    return full_context_requests(
        _config(),
        _parent(),
        selected_context_length=64_512,
        calibration_evidence_path=CALIBRATION_PATH,
    )


def test_committed_attempt_recovery_binds_exact_published_review(
    tmp_path: Path,
) -> None:
    config = _config()
    requests = _recovery_requests()
    recovery = _load_attempt_recovery(
        config,
        CALIBRATION_PATH,
        CALIBRATION_REVIEW_PATH,
        ATTEMPT_RECOVERY_PATH,
        requests,
    )
    assert recovery["scope"] == {
        "branch_id": "full-context-fork-8398e5a881d6a1dc5e4ec1a6532f02cc",
        "parent_candidate_id": "native-thinking-752f2a6728b47b5a89263445496c7b80",
        "workload": "minif2f-valid-clean-v2",
        "task_id": "amc12a_2019_p21",
        "fork_state": "P0",
        "fork_fraction": 0.0,
        "fork_prefix_len": 0,
        "branch_seed": 100,
        "max_new_tokens": 64_259,
        "fork_generation_config_sha256": (
            "ee3ce5f11b2846a0a582e73c2ed9e47d54376548a2e34922727512cb1ffe5b33"
        ),
    }
    assert recovery["attempt_recovery"] == {
        "failed_attempt_index": 0,
        "failed_attempt_terminal_status": "failed",
        "next_attempt_index": 1,
        "persisted_scientific_record_count_at_review": 0,
        "raw_journal_recovered": False,
        "mechanism": "published_independent_transition_review",
    }

    tampered = json.loads(ATTEMPT_RECOVERY_PATH.read_text(encoding="utf-8"))
    tampered["attempt_recovery"]["next_attempt_index"] = 0
    tampered_path = tmp_path / "attempt-recovery.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from reviewed P0/100 state"):
        _load_attempt_recovery(
            config,
            CALIBRATION_PATH,
            CALIBRATION_REVIEW_PATH,
            tampered_path,
            requests,
        )

    tampered_review = tmp_path / "calibration-review.json"
    tampered_review.write_text(
        CALIBRATION_REVIEW_PATH.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transition review evidence bytes changed"):
        _load_attempt_recovery(
            config,
            CALIBRATION_PATH,
            tampered_review,
            ATTEMPT_RECOVERY_PATH,
            requests,
        )


def test_attempt_recovery_starts_at_one_without_synthetic_attempt_zero(
    tmp_path: Path,
) -> None:
    recovery = _load_attempt_recovery(
        _config(),
        CALIBRATION_PATH,
        CALIBRATION_REVIEW_PATH,
        ATTEMPT_RECOVERY_PATH,
        _recovery_requests(),
    )
    attempt_path = tmp_path / "branch-attempts.jsonl"
    provenance, recovered_counts = _prepare_attempt_recovery(
        attempt_path,
        ATTEMPT_RECOVERY_PATH,
        recovery,
        persisted_branch_ids=[],
    )
    branch_id = str(recovery["scope"]["branch_id"])
    events = [
        json.loads(line)
        for line in attempt_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["recovery_checkpoint_applied"]
    assert "attempt_index" not in events[0]
    assert provenance["raw_journal_recovered"] is False
    assert recovered_counts == {branch_id: 1}
    assert _branch_attempt_counts(attempt_path, recovered_counts)[branch_id] == 1

    with attempt_path.open("a", encoding="utf-8") as handle:
        for event_kind in ("started", "failed"):
            handle.write(
                json.dumps(
                    {
                        "schema_version": BRANCH_ATTEMPT_SCHEMA,
                        "event": event_kind,
                        "branch_id": branch_id,
                        "attempt_index": 1,
                    }
                )
                + "\n"
            )
    assert _branch_attempt_counts(attempt_path, recovered_counts)[branch_id] == 2
    summary = _branch_attempt_summary(attempt_path)
    assert summary["started_attempt_count"] == 1
    assert summary["recovered_failed_attempt_count"] == 1
    assert summary["total_disclosed_attempt_count"] == 2
    assert summary["attempt_recovery_disclosure_count"] == 1
    assert summary["retried_in_flight_branch_count"] == 1


def test_attempt_recovery_fails_closed_on_unjournaled_records_and_conflicts(
    tmp_path: Path,
) -> None:
    recovery = _load_attempt_recovery(
        _config(),
        CALIBRATION_PATH,
        CALIBRATION_REVIEW_PATH,
        ATTEMPT_RECOVERY_PATH,
        _recovery_requests(),
    )
    branch_id = str(recovery["scope"]["branch_id"])
    unjournaled = tmp_path / "unjournaled.jsonl"
    with pytest.raises(ValueError, match="unjournaled scientific records"):
        _prepare_attempt_recovery(
            unjournaled,
            ATTEMPT_RECOVERY_PATH,
            recovery,
            persisted_branch_ids=[branch_id],
        )
    assert not unjournaled.exists()

    conflict = tmp_path / "conflict.jsonl"
    with conflict.open("w", encoding="utf-8") as handle:
        for event_kind in ("started", "persisted"):
            handle.write(
                json.dumps(
                    {
                        "schema_version": BRANCH_ATTEMPT_SCHEMA,
                        "event": event_kind,
                        "branch_id": branch_id,
                        "attempt_index": 0,
                    }
                )
                + "\n"
            )
    with pytest.raises(ValueError, match="conflicts with reviewed P0/100 failure"):
        _prepare_attempt_recovery(
            conflict,
            ATTEMPT_RECOVERY_PATH,
            recovery,
            persisted_branch_ids=[],
        )


def test_transition_index_uses_exact_qwen3_vocab_without_parser_internals() -> None:
    class Tokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"</think>": 42}

    tokenizer = Tokenizer()
    assert _reasoning_transition_index(tokenizer, [1, 2, 42, 3, 4]) == 3
    assert _reasoning_transition_index(tokenizer, [1, 2, 3]) is None
    assert _reasoning_transition_index(tokenizer, [1, 2, 42]) == 3


def test_classification_rules_are_deterministic() -> None:
    config = _config()
    per_state = [
        {"F": 0.0, "V": 0.0},
        {"F": 1 / 6, "V": 0.0},
    ]
    generations = [
        {"final_production_status": "nonempty", "finish_reason": "stop"},
        {"final_production_status": "nonempty", "finish_reason": "stop"},
    ]
    verifications = [{"category": "lean_rejected"}] * 2
    assert (
        _classify_outcome(config, per_state, generations, verifications)
        == "context_limited_signal"
    )

    per_state = [{"F": 0.0, "V": 0.0}] * 7
    generations = [{"final_production_status": "empty", "finish_reason": "length"}] * 42
    verifications = [{"category": "empty_candidate"}] * 42
    assert (
        _classify_outcome(config, per_state, generations, verifications)
        == "persistent_thinking_attractor"
    )
