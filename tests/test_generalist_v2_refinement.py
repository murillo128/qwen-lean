from __future__ import annotations

from qwen_lean.generalist_v2_refinement import (
    _aggregate_tasks,
    _classify_failure,
    _density_uncertainty,
    _overlap,
    _screening_lane_counts,
    _solvability_label,
)
from qwen_lean.generalist_v2_evaluation import _ordered_ids_digest


def test_refinement_solvability_partitions_are_exact() -> None:
    assert _solvability_label(64) == "robust"
    assert _solvability_label(16) == "robust"
    assert _solvability_label(15) == "search-sensitive"
    assert _solvability_label(2) == "search-sensitive"
    assert _solvability_label(1) == "lottery"
    assert _solvability_label(0) == "dead-zone"


def test_refinement_failure_classes_use_stable_priority() -> None:
    assert _classify_failure({"category": "verifier_timeout"}) == "timeout"
    assert _classify_failure(
        {
            "category": "lean_rejected",
            "diagnostics": {
                "stdout": "Unknown identifier `missing`\nerror: unsolved goals",
                "stderr": "",
            },
        }
    ) == "unknown-or-mismatched-lemma"
    assert _classify_failure(
        {
            "category": "lean_rejected",
            "diagnostics": {"stdout": "error: unsolved goals", "stderr": ""},
        }
    ) == "unsolved-goals"


def test_refinement_overlap_retains_exact_task_ids() -> None:
    overlap = _overlap(
        {"a": 1, "b": 1, "c": 0, "d": 0},
        {"a": 1, "b": 0, "c": 1, "d": 0},
        candidate_only_label="q4-only",
        reference_only_label="q0-only",
    )

    assert overlap["counts"] == {
        "both_solved": 1,
        "q4-only": 1,
        "q0-only": 1,
        "solved_by_neither": 1,
    }
    assert overlap["task_ids"]["q4-only"] == ["c"]


def test_refinement_reconstructs_legacy_q0_task_outcomes_from_bound_order() -> None:
    task_ids = ["a", "b"]
    counts = _screening_lane_counts(
        {
            "task_count": 2,
            "candidate_count": 16,
            "verified_counts": [0, 3],
            "ordered_task_ids_sha256": _ordered_ids_digest(task_ids),
        },
        task_ids,
        lane_label="Q0",
    )

    assert counts == {"a": 0, "b": 3}


def test_refinement_group_metrics_expose_density_and_duplication() -> None:
    verified = {
        "category": "verified",
        "normalized_proof_sha256": "proof-a",
    }
    rejected = {
        "category": "lean_rejected",
        "diagnostics": {"stdout": "error: unsolved goals", "stderr": ""},
    }
    raw = {
        "a": [verified, verified, *([rejected] * 62)],
        "b": [rejected] * 64,
    }
    properties = {
        task_id: {
            "declaration_chars": 10,
            "declaration_lines": 1,
            "named_hypothesis_count": 0,
            "binder_group_count": 0,
            "coercion_marker_count": 0,
            "oracle_proof": {"available": False},
        }
        for task_id in ("a", "b")
    }

    summary = _aggregate_tasks(
        ["a", "b"],
        {"a": 2, "b": 0},
        {"a": 1, "b": 0},
        raw,
        properties,
    )

    assert summary["verified_candidate_density"] == 2 / 128
    assert summary["unique_verified_proof_density"] == 1 / 128
    assert summary["verified_duplication_fraction"] == 0.5
    assert summary["solvability_partition_counts"] == {
        "dead-zone": 1,
        "search-sensitive": 1,
    }
    assert summary["failure_classes"]["counts"] == {"unsolved-goals": 126}
    assert summary["verified_candidate_density_uncertainty"]["candidate_slots"] == 128
    assert summary["verified_candidate_density_uncertainty"][
        "task_cluster_normal_95_interval"
    ] is not None


def test_refinement_uncertainty_does_not_treat_candidates_as_independent() -> None:
    summary = _density_uncertainty([64, 0])

    assert summary["density"] == 0.5
    assert summary["task_count"] == 2
    assert summary["candidate_slots"] == 128
    assert summary["task_cluster_standard_error"] == 0.5
