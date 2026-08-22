import pytest

from qwen_lean.metrics import pass_at_k, summarize_results
from qwen_lean.schema import CandidateResult


def _result(
    task_id: str,
    candidate_index: int,
    category: str,
    *,
    finish_reason: str = "eos",
) -> CandidateResult:
    return CandidateResult(
        task_id=task_id,
        candidate_id=f"model-{candidate_index}",
        candidate_index=candidate_index,
        candidate_text="exact h",
        category=category,  # type: ignore[arg-type]
        lean_exit_code=0 if category == "verified" else 1,
        diagnostics={"stdout": "", "stderr": ""},
        generation_latency_seconds=0.2,
        verification_latency_seconds=0.1,
        total_latency_seconds=0.3,
        generated_token_count=2,
        finish_reason=finish_reason,
    )


@pytest.mark.parametrize("k", [1, 4, 8])
def test_pass_at_k_zero_and_all_success_cases(k: int) -> None:
    assert pass_at_k(8, 0, k) == 0.0
    assert pass_at_k(8, 8, k) == 1.0


def test_pass_at_k_hand_checkable_one_success_case() -> None:
    assert pass_at_k(8, 1, 1) == pytest.approx(1 / 8)
    assert pass_at_k(8, 1, 4) == pytest.approx(1 / 2)
    assert pass_at_k(8, 1, 8) == 1.0


def test_summary_uses_distinct_per_task_candidate_indices() -> None:
    results = [
        _result("task-a", index, "verified" if index == 0 else "lean_rejected")
        for index in range(8)
    ]

    summary = summarize_results(
        results, expected_task_ids=["task-a"], candidates_per_task=8
    )

    assert summary["complete"] is True
    assert summary["candidate_count"] == 8
    assert summary["pass_at_k"] == pytest.approx(
        {"pass@1": 1 / 8, "pass@4": 1 / 2, "pass@8": 1.0}
    )
    assert summary["tasks_solved_within_k"] == {
        "solved@1": 1,
        "solved@4": 1,
        "solved@8": 1,
    }
    assert summary["finish_reason_counts"] == {"eos": 8, "token_limit": 0}


def test_summary_omits_pass_at_k_larger_than_the_candidate_budget() -> None:
    summary = summarize_results(
        [_result("task-a", 0, "lean_rejected")],
        expected_task_ids=["task-a"],
        candidates_per_task=1,
    )

    assert summary["complete"] is True
    assert summary["pass_at_k"] == {"pass@1": 0.0}
    assert summary["tasks_solved_within_k"] == {"solved@1": 0}


def test_infrastructure_error_makes_summary_incomplete_without_headline_metrics() -> (
    None
):
    results = [
        _result("task-a", index, "generation_error" if index == 3 else "lean_rejected")
        for index in range(8)
    ]

    summary = summarize_results(
        results, expected_task_ids=["task-a"], candidates_per_task=8
    )

    assert summary["complete"] is False
    assert summary["pass_at_k"] is None
    assert summary["tasks_solved_within_k"] is None
    assert summary["infrastructure_error_count"] == 1


def test_summary_reports_complete_extended_search_curve() -> None:
    results = [
        _result(
            "task-a",
            index,
            "verified" if index in {9, 47} else "lean_rejected",
        )
        for index in range(64)
    ]

    summary = summarize_results(
        results,
        expected_task_ids=["task-a"],
        candidates_per_task=64,
        ks=(1, 2, 4, 8, 16, 32, 64),
    )

    assert list(summary["pass_at_k"]) == [
        "pass@1",
        "pass@2",
        "pass@4",
        "pass@8",
        "pass@16",
        "pass@32",
        "pass@64",
    ]
    assert summary["tasks_solved_within_k"] == {
        "solved@1": 0,
        "solved@2": 0,
        "solved@4": 0,
        "solved@8": 0,
        "solved@16": 1,
        "solved@32": 1,
        "solved@64": 1,
    }
