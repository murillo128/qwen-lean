from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.generalist_v2_q4_canary import (
    Q4_ADAPTER_MODEL_SHA256,
    Q4_CANARY_EVIDENCE_SCHEMA_VERSION,
    Q4_CANARY_GATE_ID,
    Q4_CANARY_REQUIRED_GATES,
    _known_positive_selection,
    q4_failure_signals,
    q4_failure_template,
    q4_primary_failure_category,
    validate_q4_canary_gate,
)


def test_known_positive_selection_is_balanced_and_density_ranked() -> None:
    metadata = {}
    results = []
    expected = []
    for class_index, structural_class in enumerate(
        ("direct", "branching", "deep")
    ):
        for rank in range(6):
            task_id = f"{class_index}-{rank}"
            metadata[task_id] = {"structural_class": structural_class}
            verified_count = 6 - rank
            expected.append((structural_class, verified_count, task_id))
            for candidate_index in range(6):
                results.append(
                    {
                        "task_id": task_id,
                        "candidate_index": candidate_index,
                        "candidate_text": f"exact proof_{task_id}",
                        "category": (
                            "verified"
                            if candidate_index < verified_count
                            else "lean_rejected"
                        ),
                    }
                )

    selected, observations, result_count = _known_positive_selection(
        results, metadata
    )

    assert result_count == 108
    assert selected == [
        "0-0",
        "0-1",
        "0-2",
        "0-3",
        "1-0",
        "1-1",
        "1-2",
        "1-3",
        "2-0",
        "2-1",
        "2-2",
        "2-3",
    ]
    assert [observations[task_id]["prior_verified_candidate_count"] for task_id in selected] == [
        6,
        5,
        4,
        3,
    ] * 3


@pytest.mark.parametrize(
    ("diagnostics", "candidate", "primary"),
    [
        (
            "Candidate.lean:1:1: error: unexpected end of input; expected '⟩'",
            "exact ⟨foo",
            "premature_eos_or_incomplete_proof",
        ),
        (
            "Candidate.lean:1:1: error: unexpected token ')'",
            "exact foo )",
            "syntax_or_parser",
        ),
        (
            "error: Invalid `⟨...⟩` notation: The expected type is not an inductive type",
            "exact ⟨@Foo.bar, @Foo.baz⟩",
            "structurally_wrong_proof_or_template",
        ),
        (
            "error(lean.unknownIdentifier): Unknown constant `Foo.bar`",
            "exact @Foo.bar",
            "unknown_or_mismatched_lemma",
        ),
        ("error: unsolved goals", "constructor", "unsolved_goals"),
        (
            "error: Application type mismatch",
            "exact foo",
            "type_or_elaboration",
        ),
        ("error: tactic failed", "omega", "other_lean_rejection"),
    ],
)
def test_failure_classification_has_stable_precedence(
    diagnostics: str, candidate: str, primary: str
) -> None:
    signals = q4_failure_signals(diagnostics, candidate)
    assert q4_primary_failure_category(signals) == primary


def test_failure_template_exposes_cross_task_constructor_collapse() -> None:
    first = "exact ⟨@Foo.long_name.{0, 1}, @Bar.other.{2}⟩"
    second = "exact ⟨@Different.a.{4, 5}, @Another.b.{6}⟩"

    assert q4_failure_template(first) == "exact ⟨@CONST, @CONST⟩"
    assert q4_failure_template(second) == q4_failure_template(first)


def test_q4_canary_gate_rejects_partial_or_wrong_adapter(tmp_path: Path) -> None:
    path = tmp_path / "canary.json"
    value = {
        "schema_version": Q4_CANARY_EVIDENCE_SCHEMA_VERSION,
        "gate_id": Q4_CANARY_GATE_ID,
        "status": "passed",
        "adapter_model_sha256": Q4_ADAPTER_MODEL_SHA256,
        "requirements": {name: True for name in Q4_CANARY_REQUIRED_GATES},
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    assert validate_q4_canary_gate(path)["status"] == "passed"

    value["requirements"]["hf_q4_forward_effect"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed"):
        validate_q4_canary_gate(path)

    value["requirements"]["hf_q4_forward_effect"] = True
    value["adapter_model_sha256"] = "different"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed"):
        validate_q4_canary_gate(path)
