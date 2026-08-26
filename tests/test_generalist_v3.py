from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from qwen_lean.dataset_v3_schema import DATASET_V3_VIEW_SCHEMA_VERSION, DerivedExampleRef
from qwen_lean.generalist_v3 import (
    GeneralistV3Config,
    anchor_schedule,
    choose_exact_mass_reference,
    context_for_maximum,
    deterministic_stream_references,
    evaluate_collapse_gates,
    positive_500_step_gate,
    select_checkpoint,
    summarize_canary_candidates,
    tokenize_materialized_example,
)
from qwen_lean.generalist_v3_training import base_forward_kl
from qwen_lean.phase3 import IGNORE_INDEX


ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    eos_token_id = 999

    def encode(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return [ord(value) for value in text]


def _reference(identifier: str, numerator: int, denominator: int) -> DerivedExampleRef:
    return DerivedExampleRef(
        schema_version=DATASET_V3_VIEW_SCHEMA_VERSION,
        example_id=identifier,
        statement_id="statement",
        proof_variant_id="proof",
        kind="whole",
        boundary_id=None,
        mass_numerator=numerator,
        mass_denominator=denominator,
    )


def test_generalist_v3_config_freezes_issue_contract() -> None:
    config = GeneralistV3Config.load(ROOT / "config/qwen35-4b-generalist-v3.json")
    assert config.training["configurations"]["C1"] == {
        "learning_rate": 3e-5,
        "base_kl_lambda": 0.1,
        "eligible": True,
    }
    assert config.training["canonical_context_tokens"] == 262144
    assert config.training["resolved_context_tokens"] == 65536
    assert config.training["execution_view"]["expected_quarantined_examples"] == 6
    assert config.training["sample_theorem_by_structural_multiplier"] is True
    assert config.preservation["anchor_count"] == 512
    assert config.evaluation["interfaces"] == ["whole", "incremental"]


def test_v3_tokenization_masks_prompt_and_supervises_one_eos() -> None:
    example = tokenize_materialized_example(
        {
            "statement_id": "s",
            "example_id": "e",
            "task_kind": "continuation",
            "model_input": "theorem t : True := by\n  trivial\n  ",
            "target": "exact True.intro\n",
        },
        TinyTokenizer(),
    )
    assert example.labels[: example.prompt_tokens] == (IGNORE_INDEX,) * example.prompt_tokens
    assert example.labels[-1] == 999
    assert example.input_ids[-1] == 999
    assert example.completion == "exact True.intro"


def test_exact_mass_choice_and_stream_are_deterministic() -> None:
    references = (
        _reference("a", 1, 4),
        _reference("b", 3, 4),
    )
    assert choose_exact_mass_reference(references, Fraction(0, 1)).example_id == "a"
    assert choose_exact_mass_reference(references, Fraction(1, 4)).example_id == "b"
    first = [
        item.example_id
        for item in deterministic_stream_references(
            {"statement": references}, microbatches=20
        )
    ]
    second = [
        item.example_id
        for item in deterministic_stream_references(
            {"statement": references}, microbatches=20
        )
    ]
    assert first == second
    assert set(first) == {"a", "b"}


def test_context_and_anchor_schedule_boundaries() -> None:
    assert context_for_maximum(4096) == 4096
    assert context_for_maximum(4097) == 8192
    assert context_for_maximum(157034) == 262144
    with pytest.raises(ValueError, match="no supported context"):
        context_for_maximum(262145)
    schedule = anchor_schedule(512, 1025)
    assert set(schedule[:512]) == set(range(512))
    assert set(schedule[512:1024]) == set(range(512))
    assert schedule == anchor_schedule(512, 1025)


def _candidates(*, repeated: bool = False):
    rows = []
    for interface in ("whole", "incremental"):
        task_id = f"t:{interface}"
        for index in range(8):
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_text": "exact bad" if repeated else f"exact proof_{index}",
                    "category": "verified" if index == 0 and not repeated else "lean_rejected",
                    "generated_token_count": 8,
                    "finish_reason": "eos",
                }
            )
    return rows


def test_canary_summary_collapse_gate_and_positive_gate() -> None:
    config = GeneralistV3Config.load(ROOT / "config/qwen35-4b-generalist-v3.json")
    task_ids = ["t:whole", "t:incremental"]
    base = summarize_canary_candidates(_candidates(), expected_task_ids=task_ids)
    collapsed = summarize_canary_candidates(
        _candidates(repeated=True), expected_task_ids=task_ids
    )
    gates = evaluate_collapse_gates(
        config, collapsed, base, retained_base_solved=0
    )
    assert gates["repeated_template_collapse"] is True
    assert gates["short_eos_collapse"] is True
    assert gates["eligible"] is False
    assert positive_500_step_gate(config, collapsed, base, gates) is False


def test_selection_is_lexicographic_and_excludes_control() -> None:
    def candidate(identifier: str, solved: int, step: int):
        return {
            "configuration_id": identifier,
            "optimizer_step": step,
            "learning_rate": 1e-5,
            "retained_base_solved": 1,
            "mean_anchor_kl": 0.1,
            "collapse_gates": {"eligible": True},
            "summary": {
                "combined": {
                    "solved_at_8": solved,
                    "verified_density": 0.1,
                    "normalized_template_diversity": 0.5,
                },
                "whole": {"solved_at_8": solved},
            },
        }

    selected = select_checkpoint(
        [candidate("C0", 10, 100), candidate("C1", 2, 500), candidate("C2", 3, 500)]
    )
    assert selected["configuration_id"] == "C2"


def test_base_forward_kl_is_zero_for_identical_logits() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert base_forward_kl(logits, logits).item() == pytest.approx(0.0, abs=1e-7)
    assert base_forward_kl(logits, torch.tensor([3.0, 2.0, 1.0])).item() > 0
