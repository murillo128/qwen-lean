from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from qwen_lean.mathia_prompt_ab_regressions import (
    ANALYSIS_SCHEMA_VERSION,
    RAW_B_SCHEMA_VERSION,
    _obvious_local_premise_identifiers,
    _qualified_lean_identifiers,
    _write_jsonl,
    candidate_diagnostic_observations,
    candidate_format_markers,
    mechanical_variants,
    reconstruct_regression_tasks,
    render_regression_analysis,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reconstruct_regression_tasks_uses_frozen_q0_and_all_b_slots() -> None:
    manifest = {
        "tasks": [
            {
                "task_id": "regression",
                "q0_verified_candidate_count": 2,
                "candidate_slots": {"B": ["b-0", "b-1"]},
            },
            {
                "task_id": "arm-b-pass",
                "q0_verified_candidate_count": 1,
                "candidate_slots": {"B": ["b-2", "b-3"]},
            },
            {
                "task_id": "q0-fail",
                "q0_verified_candidate_count": 0,
                "candidate_slots": {"B": ["b-4", "b-5"]},
            },
        ]
    }
    results = {
        "b-0": {"category": "lean_rejected"},
        "b-1": {"category": "lean_rejected"},
        "b-2": {"category": "lean_rejected"},
        "b-3": {"category": "verified"},
        "b-4": {"category": "lean_rejected"},
        "b-5": {"category": "lean_rejected"},
    }

    assert [
        task["task_id"] for task in reconstruct_regression_tasks(manifest, results)
    ] == ["regression"]


def test_mechanical_variants_cover_each_permitted_wrapper_without_repair() -> None:
    declaration = "theorem example : True"
    cases = {
        "by\n  trivial": ("strip_leading_duplicated_by", "\n  trivial"),
        "```lean\ntrivial\n```": ("unwrap_markdown_fence", "trivial"),
        "theorem example : True := by\n  trivial": (
            "remove_exact_repeated_theorem_declaration",
            "\n  trivial",
        ),
        "begin\n  trivial\nend": ("unwrap_lean3_begin_end", "\n  trivial\n"),
        "Here is the Lean proof:\ntrivial": (
            "remove_whitelisted_natural_language_prefix",
            "trivial",
        ),
        "trivial\nThis completes the proof.": (
            "remove_whitelisted_natural_language_suffix",
            "trivial",
        ),
    }

    for raw_text, (transform, transformed_text) in cases.items():
        variants = mechanical_variants(raw_text, declaration)
        assert any(
            variant.transform_sequence == (transform,)
            and variant.transformed_text == transformed_text
            for variant in variants
        )

    assert mechanical_variants("exact True.intro", declaration) == []


def test_mechanical_variants_compose_only_recorded_wrapper_removals() -> None:
    raw_text = (
        "Here's the Lean proof:\n"
        "```lean\n"
        "by\n"
        "  trivial\n"
        "```\n"
        "QED."
    )

    variants = mechanical_variants(raw_text, "theorem example : True")

    recovered = next(variant for variant in variants if variant.transformed_text == "\n  trivial")
    assert len(recovered.transform_sequence) == 4
    assert set(recovered.transform_sequence) == {
        "remove_whitelisted_natural_language_prefix",
        "remove_whitelisted_natural_language_suffix",
        "unwrap_markdown_fence",
        "strip_leading_duplicated_by",
    }


def test_candidate_format_markers_do_not_treat_arbitrary_code_as_wrapper() -> None:
    marked = candidate_format_markers(
        "by\n```lean\nbegin\n  sorry\nend\n```\nThis proves the proof."
    )
    assert marked == {
        "duplicated_by": True,
        "theorem_repetition": False,
        "markdown_fence": True,
        "lean3_begin": False,
        "natural_language_contamination": True,
        "sorry_or_admit": True,
    }
    assert candidate_format_markers("by_contra h") == {
        "duplicated_by": False,
        "theorem_repetition": False,
        "markdown_fence": False,
        "lean3_begin": False,
        "natural_language_contamination": False,
        "sorry_or_admit": False,
    }


def test_candidate_diagnostic_observations_use_only_explicit_lean_messages() -> None:
    result = {
        "diagnostics": {
            "stdout": (
                "Candidate.lean:1:1: error(lean.unknownIdentifier): "
                "Unknown identifier `missingLemma`\n"
                "Candidate.lean:2:1: error(lean.unknownIdentifier): "
                "Unknown constant `Missing.api`\n"
                "Candidate.lean:3:1: error: unsolved goals\n"
                "Candidate.lean:4:1: error: Type mismatch\n"
                "Candidate.lean:5:1: error: failed to synthesize OfNat X 1\n"
                "Candidate.lean:6:1: error: unexpected token 'by'; expected tactic\n"
                "Candidate.lean:7:1: error: Tactic `apply` failed\n"
                "Candidate.lean:8:1: error: declaration uses `sorry`\n"
            ),
            "stderr": "",
        }
    }

    observed = candidate_diagnostic_observations(result)

    assert observed["diagnostic_categories"] == [
        "unknown_identifier",
        "unknown_constant",
        "unsolved_goals",
        "type_mismatch",
        "elaboration_error",
        "syntax_error",
        "tactic_failure",
        "sorry_or_admit_rejected",
    ]
    assert observed["unknown_references"] == [
        {"kind": "constant", "name": "Missing.api"},
        {"kind": "identifier", "name": "missingLemma"},
    ]
    assert len(observed["diagnostic_text_sha256"]) == 64


def test_direct_text_features_are_conservative_and_lexical() -> None:
    declaration = (
        "theorem example (x : Nat) (h₀ : x = 0) (hi : x ≤ 1) "
        "(helper : Nat) : True"
    )

    assert _obvious_local_premise_identifiers(declaration) == {"h₀", "hi"}
    assert _qualified_lean_identifiers(
        "simpa using Nat.add_zero x; exact _root_.True.intro; have z := 3.14"
    ) == {"Nat.add_zero", "_root_.True.intro"}


def test_raw_jsonl_round_trip_preserves_exact_continuation_and_hash(
    tmp_path: Path,
) -> None:
    raw_text = "by\r\n  have café : True := by trivial\r\n  exact café\n"
    raw_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    output = tmp_path / "raw-b-candidates.jsonl"
    row = {
        "schema_version": RAW_B_SCHEMA_VERSION,
        "workload": "fixture",
        "task_id": "fixture-task",
        "candidate_index": 0,
        "candidate_id": "fixture-candidate",
        "raw_text": raw_text,
        "raw_sha256": raw_sha256,
        "finish_reason": "eos",
        "generated_token_count": 17,
        "official_verifier_classification": "lean_rejected",
    }

    payload_sha256 = _write_jsonl(output, [row])
    parsed = json.loads(output.read_bytes())

    assert parsed["raw_text"] == raw_text
    assert parsed["raw_sha256"] == raw_sha256
    assert hashlib.sha256(parsed["raw_text"].encode("utf-8")).hexdigest() == raw_sha256
    assert hashlib.sha256(output.read_bytes()).hexdigest() == payload_sha256


def test_committed_regression_evidence_is_complete_and_self_consistent() -> None:
    evidence_root = ROOT / "evidence/mathia-prompt-ab"
    subset_root = evidence_root / "q0-b-regressions"

    def read_jsonl(name: str) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (subset_root / name).read_bytes().splitlines()
            if line.strip()
        ]

    raw_b = read_jsonl("raw-b-candidates.jsonl")
    q0_verified = read_jsonl("q0-verified-candidates.jsonl")
    transformed = read_jsonl("transformed-b-candidates.jsonl")
    analysis = json.loads(
        (evidence_root / "q0-b-regression-analysis.json").read_bytes()
    )

    assert len(raw_b) == 184
    assert len(q0_verified) == 31
    raw_tasks = {(row["workload"], row["task_id"]) for row in raw_b}
    assert len(raw_tasks) == 23
    assert Counter(workload for workload, _ in raw_tasks) == {
        "minif2f-valid-clean-v2": 17,
        "fresh-composition-valid-v2": 6,
    }

    b_by_task: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for row in raw_b:
        b_by_task[(row["workload"], row["task_id"])].append(row)
        assert row["official_verifier_classification"] == "lean_rejected"
        assert hashlib.sha256(str(row["raw_text"]).encode("utf-8")).hexdigest() == row[
            "raw_sha256"
        ]
        assert row["finish_reason"] in {"eos", "token_limit"}
        assert isinstance(row["generated_token_count"], int)
    assert len({row["candidate_id"] for row in raw_b}) == 184
    assert all(
        sorted(int(row["candidate_index"]) for row in rows) == list(range(8))
        for rows in b_by_task.values()
    )

    assert {(row["workload"], row["task_id"]) for row in q0_verified} == raw_tasks
    assert len({row["candidate_id"] for row in q0_verified}) == len(q0_verified)
    q0_by_task: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for row in q0_verified:
        q0_by_task[(row["workload"], row["task_id"])].append(row)
        assert row["authoritative_q0_classification"] == "verified"
        assert hashlib.sha256(str(row["raw_text"]).encode("utf-8")).hexdigest() == row[
            "raw_sha256"
        ]
        assert row["source_generation_file_sha256"]
        assert row["source_result_file_sha256"]
        assert row["source_generation_row_sha256"]
        assert row["source_result_row_sha256"]
        assert row["q0_evidence_sha256"] == analysis["source_bindings"][
            "q0_evidence_sha256"
        ]
        assert row["q0_raw_recovery_archive_sha256"] == analysis["source_bindings"][
            "q0_raw_recovery_archive_sha256"
        ]

    raw_sha256s = {row["raw_sha256"] for row in raw_b}
    for row in transformed:
        assert row["source_raw_sha256"] in raw_sha256s
        assert row["scoring_excluded"] is True
        assert hashlib.sha256(
            str(row["transformed_text"]).encode("utf-8")
        ).hexdigest() == row["transformed_sha256"]
        assert row["transform_sequence"]

    committed = analysis["committed_evidence"]
    for key, name, rows in (
        ("raw_b_candidates", "raw-b-candidates.jsonl", raw_b),
        ("q0_verified_candidates", "q0-verified-candidates.jsonl", q0_verified),
        ("transformed_b_candidates", "transformed-b-candidates.jsonl", transformed),
    ):
        path = subset_root / name
        assert committed[key]["path"] == f"evidence/mathia-prompt-ab/q0-b-regressions/{name}"
        assert committed[key]["row_count"] == len(rows)
        assert committed[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    assert analysis["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert analysis["aggregate"]["total_regressions"] == 23
    assert sum(analysis["aggregate"]["classification_counts"].values()) == 23
    assert {(row["workload"], row["task_id"]) for row in analysis["tasks"]} == raw_tasks
    assert "training_questions" not in analysis["aggregate"]
    assert "primary_owner" not in json.dumps(analysis, sort_keys=True)
    for task in analysis["tasks"]:
        task_key = (task["workload"], task["task_id"])
        assert len(task["b_candidate_diagnostics"]) == 8
        assert {
            candidate["candidate_index"]
            for candidate in task["b_candidate_diagnostics"]
        } == set(range(8))
        features = task["observable_comparison_features"]
        assert len(features["q0_verified_candidate_lengths"]) == task[
            "q0_verified_candidate_count"
        ]
        expected_q0_lengths = {
            (
                row["candidate_index"],
                row["raw_sha256"],
                row["generated_token_count"],
                len(str(row["raw_text"])),
                len(str(row["raw_text"]).encode("utf-8")),
            )
            for row in q0_by_task[task_key]
        }
        assert {
            (
                row["candidate_index"],
                row["raw_sha256"],
                row["generated_token_count"],
                row["character_count"],
                row["utf8_byte_count"],
            )
            for row in features["q0_verified_candidate_lengths"]
        } == expected_q0_lengths
        assert features["shortest_q0_verified_by_generated_tokens"][
            "generated_token_count"
        ] >= 0
        assert features["shortest_q0_verified_by_character_count"][
            "character_count"
        ] >= 0
        assert features["direct_text_comparison_scope"]
    facts = analysis["aggregate"]["observed_diagnostic_facts"]
    assert facts["final_output_formatting_recovery_lower_bound"]["task_count"] == 3
    assert facts["final_output_formatting_recovery_lower_bound"][
        "unique_candidate_proof_count"
    ] == 4
    assert facts["content_or_search_after_mechanical_cleanup"]["task_count"] == 20
    assert facts["inconclusive_task_count"] == 0
    assert facts["arm_b_candidate_count"] == 184
    assert analysis["official_86_results_modified"] is False
    assert analysis["model_inference_or_regeneration_performed"] is False
    report = (
        evidence_root / "Q0_B_REGRESSION_ANALYSIS.md"
    ).read_text(encoding="utf-8")
    assert report == render_regression_analysis(analysis)
    assert "## Observed diagnostic facts" in report
    assert "## Training questions" not in report
    assert "Prioritize" not in report
