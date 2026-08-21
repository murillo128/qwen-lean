from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import fmean

import pytest

from qwen_lean.generalist_v2 import (
    GENERALIST_SERIALIZATION_ID,
    CheckpointValidation,
    GeneralistProofVariant,
    GeneralistV2Config,
    build_weighted_training_examples,
    compare_paired_verified_counts,
    compute_training_weights,
    final_evaluation_plan,
    materialize_fresh_riemann_views,
    normalized_example_loss_scales,
    normalized_riemann_domain_tags,
    one_pass_trajectory,
    select_context_length,
    select_generalist_checkpoint,
    tokenize_generalist_variant,
    validation_evaluation_plan,
)

ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def _record(
    statement: int,
    variant: int = 1,
    *,
    source_kind: str = "real",
    split: str = "train",
    optimizer_eligible: bool = True,
    family: str | None = None,
    composition_class: str | None = None,
    derivation_family: str | None = None,
    tags: tuple[str, ...] = (),
) -> GeneralistProofVariant:
    return GeneralistProofVariant(
        statement_id=f"statement-{statement:03d}",
        proof_variant_id=f"statement-{statement:03d}-proof-{variant:02d}",
        declaration_name=f"fixture_{statement}",
        declaration=f"theorem fixture_{statement} : True",
        completion="trivial" if variant == 1 else "exact True.intro",
        preamble="import Mathlib\n\nopen scoped BigOperators",
        split=split,
        optimizer_eligible=optimizer_eligible,
        source_kind=source_kind,
        generator_family=family,
        composition_class=composition_class,
        derivation_family_id=derivation_family,
        domain_tags=tags,
    )


def _synthetic(
    statement: int,
    *,
    split: str = "train",
    optimizer_eligible: bool = True,
    family: str = "chain",
    composition_class: str = "direct",
    derivation_family: str | None = None,
    tags: tuple[str, ...] = (),
) -> GeneralistProofVariant:
    return _record(
        statement,
        source_kind="synthetic",
        split=split,
        optimizer_eligible=optimizer_eligible,
        family=family,
        composition_class=composition_class,
        derivation_family=derivation_family or f"derivation-{statement:03d}",
        tags=tags,
    )


def test_generalist_config_freezes_issue_78_contract() -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")

    assert config.model["model_revision"] == (
        "1001bb4d826a52d1f399e183466143f4da7b741b"
    )
    assert config.serialization["id"] == GENERALIST_SERIALIZATION_ID
    assert config.weighting["domain_multipliers"] == {}
    assert config.training["context_choices"] == [4096, 8192, 16384, 32768]
    assert config.training["resolved_context_tokens"] == 32768
    assert config.weighting["resolved_synthetic_statement_multiplier"] == 4.0
    assert config.evaluation["sampling"]["temperature"] == 0.8
    assert config.value["riemann"]["training_stream"] is None


def test_config_rejects_riemann_specific_weighting() -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    changed = GeneralistV2Config(
        path=config.path,
        value={
            **config.value,
            "weighting": {
                **config.weighting,
                "domain_multipliers": {"riemann-core": 2.0},
            },
        },
    )

    with pytest.raises(ValueError, match="domain_multipliers"):
        changed.validate()


def test_serialization_masks_prompt_and_supervises_one_terminal_eos() -> None:
    record = _record(1)
    example = tokenize_generalist_variant(record, _Tokenizer(), example_weight=0.5)

    assert example.prompt == (
        "import Mathlib\n\nopen scoped BigOperators\n\n"
        "/- Complete the proof below.\n"
        "Return only Lean code continuing after `by`; do not use `sorry` or `admit`. -/\n"
        "theorem fixture_1 : True := by\n  "
    )
    assert example.labels[: example.prompt_tokens] == (-100,) * example.prompt_tokens
    assert example.labels[example.prompt_tokens : -1] == tuple(
        ord(character) for character in "trivial"
    )
    assert example.input_ids[-1] == example.labels[-1] == 999
    assert example.input_ids.count(999) == 1


def test_serialization_rejects_in_band_eos_and_never_truncates() -> None:
    class EosTokenizer(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            values = super().encode(text, add_special_tokens=add_special_tokens)
            return values + [999] if text == "trivial" else values

    with pytest.raises(ValueError, match="already contains EOS"):
        tokenize_generalist_variant(_record(1), EosTokenizer(), example_weight=1.0)

    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_generalist_variant(
            _record(1),
            _Tokenizer(),
            example_weight=1.0,
            maximum_sequence_tokens=8,
        )


def test_placeholder_check_ignores_comments_but_rejects_active_code() -> None:
    commented = replace(
        _record(1),
        completion=(
            "/- This proof does not admit a shortcut. -/\n"
            "-- sorry was considered here\n"
            "trivial"
        ),
    )
    commented.validate()

    for completion in ("sorry", "by\n  admit"):
        with pytest.raises(ValueError, match="placeholder"):
            replace(_record(1), completion=completion).validate()


def test_statement_weights_normalize_variants_and_target_ten_percent_synthetic_mass() -> (
    None
):
    records = [_record(index) for index in range(1, 10)]
    records[0] = _record(1, 1)
    records.insert(1, _record(1, 2))
    records.append(_synthetic(10, family="branch", composition_class="deep"))

    weights = compute_training_weights(records)

    assert weights.synthetic_base_multiplier == pytest.approx(1.0)
    assert weights.synthetic_mass_fraction == pytest.approx(0.1)
    assert weights.variant_weights["statement-001-proof-01"] == pytest.approx(0.5)
    assert weights.variant_weights["statement-001-proof-02"] == pytest.approx(0.5)
    assert sum(
        weights.variant_weights[item.proof_variant_id]
        for item in records
        if item.statement_id == "statement-001"
    ) == pytest.approx(1.0)


def test_synthetic_balance_caps_individual_statements_and_preserves_total_mass() -> (
    None
):
    records = [_record(index) for index in range(1, 101)]
    records.extend(
        [
            _synthetic(101, family="prolific", composition_class="direct"),
            _synthetic(102, family="prolific", composition_class="direct"),
            _synthetic(103, family="rare", composition_class="deep"),
        ]
    )

    weights = compute_training_weights(records)

    assert weights.synthetic_base_multiplier == pytest.approx(100 / 27)
    assert weights.synthetic_mass_fraction == pytest.approx(0.1)
    assert weights.maximum_statement_weight <= 4.0
    assert (
        weights.statement_weights["statement-103"]
        > weights.statement_weights["statement-101"]
    )


def test_synthetic_multiplier_cap_is_honest_when_ten_percent_is_unreachable() -> None:
    records = [_record(index) for index in range(1, 101)] + [_synthetic(101)]

    weights = compute_training_weights(records)

    assert weights.synthetic_base_multiplier == 4.0
    assert weights.synthetic_mass_fraction == pytest.approx(4 / 104)
    assert weights.maximum_statement_weight == 4.0


def test_all_optimizer_variants_are_serialized_once_and_eval_proofs_are_rejected() -> (
    None
):
    records = [_record(1, 1), _record(1, 2), _record(2), _synthetic(3)]
    examples, weights = build_weighted_training_examples(records, _Tokenizer())

    assert [item.proof_variant_id for item in examples] == [
        item.proof_variant_id for item in records
    ]
    assert len(weights.variant_weights) == len(records)
    with pytest.raises(ValueError, match="optimizer-visible"):
        build_weighted_training_examples(
            records,
            _Tokenizer(),
            forbidden_proof_variant_ids={"statement-001-proof-02"},
        )


def test_loss_scales_preserve_weights_with_micro_batch_one() -> None:
    scales = normalized_example_loss_scales([0.25, 0.75, 2.0])

    assert scales == pytest.approx((0.25, 0.75, 2.0))
    assert fmean(scales) == pytest.approx(1.0)
    assert len(set(scales)) == 3


def test_context_selection_uses_smallest_fitting_bucket_and_rejects_over_32k() -> None:
    assert select_context_length([200, 4096])["selected_context_tokens"] == 4096
    assert select_context_length([4097, 7000])["selected_context_tokens"] == 8192
    assert select_context_length([16385])["selected_context_tokens"] == 32768

    with pytest.raises(ValueError, match="maximum supported"):
        select_context_length([32769])


def test_one_pass_trajectory_contains_every_variant_once_and_q1_q4() -> None:
    records = [_record(index) for index in range(1, 33)]
    records.extend(_record(index, 2) for index in range(1, 5))
    trajectory = one_pass_trajectory(records)

    assert trajectory["optimizer_visible_variants"] == 36
    assert trajectory["unique_optimizer_visible_variants"] == 36
    assert trajectory["optimizer_steps"] == 5
    assert trajectory["final_optimizer_update_rows"] == 4
    assert trajectory["duplicate_final_batch_fill"] is False
    assert trajectory["checkpoint_optimizer_steps"] == {
        "Q1": 2,
        "Q2": 3,
        "Q3": 4,
        "Q4": 5,
    }


def test_evaluation_plans_keep_test_and_riemann_out_of_selection() -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    validation = validation_evaluation_plan(config)

    assert set(validation["checkpoints"]) == {"Q0", "Q1", "Q2", "Q3", "Q4"}
    assert validation["test_workloads_consulted"] is False
    assert validation["riemann_used_for_selection"] is False
    assert "dataset-v2-train-probe" in validation["checkpoints"]["Q2"]["workloads"]
    assert "dataset-v2-train-probe" not in validation["checkpoints"]["Q3"]["workloads"]

    final = final_evaluation_plan(config, selected_checkpoint="Q3")
    assert final["selected_checkpoint_frozen"] == "Q3"
    assert final["test_workloads_consulted_after_checkpoint_freeze"] is True
    assert final["historical_riemann"]["clean_generalization"] is False
    assert final["historical_riemann"]["candidates_per_task"] == 4
    assert final["generic_candidates_per_task"] == 8
    assert final["fresh_riemann_candidates_per_task"] == 8
    with pytest.raises(ValueError, match="frozen Q1-Q4"):
        final_evaluation_plan(config, selected_checkpoint="Q0")


def test_checkpoint_selection_uses_frozen_rule_and_paired_math() -> None:
    evaluations = {
        "Q0": CheckpointValidation((1, 0, 0, 0), (1, 1, 1, 1)),
        "Q1": CheckpointValidation((1, 0, 0, 0), (1, 1, 1, 1)),
        "Q2": CheckpointValidation((1, 1, 0, 0), (1, 1, 0, 0)),
        "Q3": CheckpointValidation((1, 1, 0, 0), (1, 1, 1, 0)),
        "Q4": CheckpointValidation((1, 1, 1, 0), (0, 0, 0, 0)),
    }

    selection = select_generalist_checkpoint(evaluations, resamples=200, seed=0)

    assert selection["selected_checkpoint"] == "Q3"
    assert "Q4" not in selection["eligible_checkpoints"]
    assert selection["test_or_riemann_outcomes_consulted"] is False
    paired = compare_paired_verified_counts(
        [0, 1, 0, 1], [1, 1, 0, 0], resamples=100, seed=0
    )
    assert paired["paired_outcomes"] == {
        "both_solved": 1,
        "candidate_only": 1,
        "reference_only": 1,
        "neither_solved": 1,
        "paired_wins_candidate_minus_reference": 0,
    }
    assert paired["exact_two_sided_mcnemar_p"] == 1.0


def test_fresh_riemann_views_are_deterministic_and_leakage_free() -> None:
    records = [
        _synthetic(
            20,
            split="validation",
            optimizer_eligible=False,
            family="chain",
            composition_class="branching",
            derivation_family="fresh-valid-family",
            tags=("zeta",),
        ),
        _synthetic(
            21,
            split="test",
            optimizer_eligible=False,
            family="chain",
            composition_class="deep",
            derivation_family="fresh-test-family",
            tags=("PNT+", "arithmetic_functions"),
        ),
        _synthetic(
            22,
            split="test",
            optimizer_eligible=False,
            family="logic",
            composition_class="direct",
            derivation_family="unrelated-family",
            tags=("topology",),
        ),
    ]

    first = materialize_fresh_riemann_views(
        records,
        training_statement_ids={"statement-001"},
        training_derivation_family_ids={"training-family"},
    )
    second = materialize_fresh_riemann_views(
        reversed(records),
        training_statement_ids={"statement-001"},
        training_derivation_family_ids={"training-family"},
    )

    assert first == second
    assert first["views"]["riemann-fresh-valid-v2"]["statement_ids"] == [
        "statement-020"
    ]
    assert first["views"]["riemann-fresh-test-v2"]["domain_breakdown"] == {
        "arithmetic-functions": 1,
        "pnt-plus": 1,
    }

    leaking = replace(records[0], derivation_family_id="training-family")
    with pytest.raises(ValueError, match="derivation family leaks"):
        materialize_fresh_riemann_views(
            [leaking, records[1]],
            training_statement_ids=set(),
            training_derivation_family_ids={"training-family"},
        )


def test_dataset_v2_namespaced_prime_tags_expand_to_frozen_riemann_tags() -> None:
    assert normalized_riemann_domain_tags(
        (
            "prime-family:zeta-analytic-number-theory",
            "prime-family:prime-counting-pnt",
            "prime-family:pnt-plus",
            "prime-family:arithmetic-functions",
            "prime-family:prime-arithmetic-divisibility",
            "prime-family:riemann-core-bubble",
        )
    ) == {
        "zeta",
        "analytic-number-theory",
        "prime-counting",
        "pnt",
        "pnt-plus",
        "arithmetic-functions",
        "prime-arithmetic",
        "divisibility",
        "riemann-core",
        "riemann-bubble",
    }
