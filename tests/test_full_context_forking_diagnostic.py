from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.counterfactual_forking_assessment import (
    ParentTrajectory,
    fork_states,
)
from qwen_lean.full_context_forking_diagnostic import (
    CHECKPOINT_REVIEW_SCHEMA,
    COUNTERFACTUAL_RESULTS_SHA256,
    FullContextForkingConfig,
    _classify_outcome,
    _next_calibration_action,
    _reasoning_transition_index,
    _validate_checkpoint_review,
    full_context_requests,
)
from qwen_lean.native_thinking_assessment import MathiaTask, _file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-full-context-forking.json"


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
        rendered_prompt_token_ids=tuple(range(100)),
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
