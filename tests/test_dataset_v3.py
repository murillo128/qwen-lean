from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from qwen_lean.dataset_v2_contract import statement_fingerprint_v2
from qwen_lean.dataset_v2_schema import EnvironmentContext, ProofVerification
from qwen_lean.dataset_v3 import (
    build_boundaries,
    convert_v2_record,
    dataset_v3_proof_variant_id,
    dataset_v3_statement_id,
    materialize_example,
    plan_optimizer_examples,
    read_records,
    source_expression_in_pinned_file,
    structural_boundary_offsets,
    structural_proof_fingerprint,
    validate_no_placeholders,
    validate_split_isolation,
    write_records,
)
from qwen_lean.dataset_v3_composition import (
    _balanced_role_selection,
    render_source_preserving_proof,
)
from qwen_lean.dataset_v2_composition import CompositionPlan, CompositionSource
from qwen_lean.dataset_v3_schema import (
    DATASET_V3_SCHEMA_VERSION,
    DatasetV3ProofVariant,
    DatasetV3Record,
)


def _environment() -> EnvironmentContext:
    return EnvironmentContext(
        environment_id="env-v3",
        lean_toolchain="leanprover/lean4:v4.32.2",
        repository="https://example.invalid/source",
        revision="a" * 40,
        mathlib_revision="b" * 40,
        file_path="Mathlib/Test.lean",
        module="Mathlib.Test",
        imports=("Mathlib",),
        source_span=None,
        context_kind="source-position",
        target_compatibility="verified-target-environment",
    )


def _record(
    name: str = "fixture",
    *,
    proof: str = "by\n  intro h\n  constructor\n  · exact h\n  · exact h",
    role: str = "training",
    proposition: str = "True → True ∧ True",
) -> DatasetV3Record:
    declaration = f"theorem {name} : {proposition}"
    statement_id = dataset_v3_statement_id(declaration)
    proof_variant_id = dataset_v3_proof_variant_id(
        statement_id,
        proof,
        source_repository="https://example.invalid/source",
        source_revision="a" * 40,
        source_file="Mathlib/Test.lean",
    )
    variant = DatasetV3ProofVariant(
        proof_variant_id=proof_variant_id,
        source_expression=proof,
        proof_text=proof,
        proof_form="tactic",
        transformation_kind="none",
        transformation_reason=None,
        exact_text_fingerprint=hashlib.sha256(proof.encode("utf-8")).hexdigest(),
        structural_fingerprint=structural_proof_fingerprint(proof),
        resolved_dependencies=(),
        boundaries=build_boundaries(proof, proof_variant_id),
        verification=ProofVerification(
            status="accepted",
            environment_id="env-v3",
            method="fixture",
            evidence_id="fixture",
        ),
        source_declaration_name=name,
        source_repository="https://example.invalid/source",
        source_revision="a" * 40,
        source_file="Mathlib/Test.lean",
        source_expression_verified=True,
    )
    record = DatasetV3Record(
        schema_version=DATASET_V3_SCHEMA_VERSION,
        statement_id=statement_id,
        canonical_declaration=declaration,
        normalized_statement_fingerprint=statement_fingerprint_v2(declaration),
        role=role,  # type: ignore[arg-type]
        theorem_mass_numerator=1,
        theorem_mass_denominator=1,
        provenance="real-mathlib",
        environment=_environment(),
        proof_variants=(variant,),
        topic_tags=("domain:generic",),
        memberships=(),
    )
    record.validate()
    return record


def test_layout_boundaries_only_cut_at_complete_top_level_tactics() -> None:
    proof = (
        "by\n"
        "  intro h\n"
        "  have nested : True := by\n"
        "    exact True.intro\n"
        "  constructor\n"
        "  · exact nested\n"
        "  · exact h"
    )
    offsets = structural_boundary_offsets(proof)
    continuations = [proof[offset:].lstrip() for offset, _ in offsets]

    assert len(offsets) == 4
    assert continuations[0].startswith("have nested")
    assert continuations[1].startswith("constructor")
    assert continuations[2].startswith("· exact nested")
    assert continuations[3].startswith("· exact h")
    assert all(proof[:offset] + proof[offset:] == proof for offset, _ in offsets)


def test_layout_boundary_extractor_refuses_term_and_single_tactic_proofs() -> None:
    assert structural_boundary_offsets("True.intro") == []
    assert structural_boundary_offsets("by\n  exact True.intro") == []


def test_structural_fingerprint_collapses_transport_parentheses_and_aliases() -> None:
    dependency = "Example.Namespace.proof"
    term = structural_proof_fingerprint("Example.Namespace.proof", (dependency,))
    wrapped = structural_proof_fingerprint("by exact ((proof))", (dependency,))
    assert term == wrapped


def test_statement_mass_is_exact_and_independent_of_boundary_count() -> None:
    record = _record()
    examples = plan_optimizer_examples(record, whole_mass=Fraction(1, 2))

    assert {item.kind for item in examples} == {"whole", "continuation"}
    assert sum(
        (Fraction(item.mass_numerator, item.mass_denominator) for item in examples),
        Fraction(),
    ) == Fraction(1, 1)
    assert sum(
        (
            Fraction(item.mass_numerator, item.mass_denominator)
            for item in examples
            if item.kind == "whole"
        ),
        Fraction(),
    ) == Fraction(1, 2)


def test_materialized_continuations_reconstruct_the_verified_whole_proof() -> None:
    record = _record()
    examples = plan_optimizer_examples(record, whole_mass=Fraction(1, 2))
    continuation = next(item for item in examples if item.kind == "continuation")
    materialized = materialize_example(record, continuation)

    assert (
        materialized["proof_prefix"] + materialized["target"]
        == record.proof_variants[0].proof_text
    )
    assert materialized["model_input"].endswith(materialized["proof_prefix"])


def test_materializer_refuses_silent_context_drop() -> None:
    record = _record()
    whole = next(
        item
        for item in plan_optimizer_examples(record, whole_mass=Fraction(1, 2))
        if item.kind == "whole"
    )
    with pytest.raises(ValueError, match="refusing to truncate or omit"):
        materialize_example(record, whole, max_utf8_bytes=1)


def test_v2_term_transport_is_recovered_to_raw_source_form() -> None:
    declaration = "theorem source_term : True"
    value = {
        "canonical_declaration": declaration,
        "role": "training",
        "provenance": "real-mathlib",
        "environment": {
            **_environment().__dict__,
            "imports": ["Mathlib"],
            "source_span": None,
        },
        "proof_variants": [
            {
                "source_expression": "True.intro",
                "canonical_proof": "by\n  exact (True.intro)",
                "transformation_kind": "term-to-exact",
                "resolved_dependencies": ["True.intro"],
                "verification": {
                    "status": "accepted",
                    "environment_id": "env-v3",
                    "method": "fixture",
                    "evidence_id": "fixture",
                },
                "source_declaration_name": "source_term",
                "source_repository": "https://example.invalid/source",
                "source_revision": "a" * 40,
                "source_file": "Mathlib/Test.lean",
            }
        ],
        "topic_tags": ["domain:generic"],
        "memberships": [],
    }
    record = convert_v2_record(value, source_expression_verifier=lambda _: True)

    assert record is not None
    assert record.proof_variants[0].proof_text == "True.intro"
    assert record.proof_variants[0].proof_form == "term"
    assert record.proof_variants[0].transformation_kind == "none"


def test_pinned_source_expression_check_uses_original_text(tmp_path: Path) -> None:
    source = tmp_path / "Test.lean"
    source.write_text("theorem source : True := by\n  exact True.intro\n")
    assert source_expression_in_pinned_file("by\n  exact True.intro", source)
    assert not source_expression_in_pinned_file("by\n  trivial", source)


def test_split_leakage_detects_structurally_cosmetic_proof_overlap() -> None:
    training = _record("training")
    heldout = _record("heldout", role="test")
    with pytest.raises(ValueError, match="split leakage"):
        validate_split_isolation([training, heldout])


def test_placeholder_gate_applies_only_to_optimizer_membership() -> None:
    record = _record(proof="by\n  exact True.intro")
    assert validate_no_placeholders([record])["placeholders"] == 0


def test_record_packaging_is_byte_deterministic(tmp_path: Path) -> None:
    record = _record()
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    assert write_records(first, [record]) == write_records(second, [record])
    assert first.read_bytes() == second.read_bytes()
    assert list(read_records(first)) == [record]


def test_generated_composition_proofs_use_multiple_top_level_tactics() -> None:
    sources = tuple(
        CompositionSource(
            statement_id=str(index),
            declaration_name=f"source_{index}",
            source_module="Mathlib.Test",
            topic_tags=("domain:generic",),
            domain_family="generic",
        )
        for index in range(2)
    )
    plan = CompositionPlan(
        synthetic_name="synthetic",
        source_lemmas=sources,
        domain_family="generic",
        structural_class="direct",
        generator_family="direct-source-and-tactic-v3",
        normalized_proof_dag="and(leaf:0,leaf:1);path=2",
        derivation_family_fingerprint="f" * 64,
        relation_edges=(("source_0", "source_1", "fixture"),),
    )
    proof = render_source_preserving_proof(plan)
    assert proof.startswith("by\n  ")
    assert "source_0" in proof and "source_1" in proof
    assert structural_boundary_offsets(proof)


def test_composition_role_assignment_reserves_every_stratum() -> None:
    plans = []
    for structural, arity in (("direct", 2), ("branching", 3), ("deep", 4)):
        for logic in ("and", "iff"):
            for index in range(6):
                sources = tuple(
                    CompositionSource(
                        statement_id=f"{structural}-{logic}-{index}-{source_index}",
                        declaration_name=(
                            f"source_{structural}_{logic}_{index}_{source_index}"
                        ),
                        source_module="Mathlib.Test",
                        topic_tags=("domain:generic",),
                        domain_family="generic",
                    )
                    for source_index in range(arity)
                )
                plans.append(
                    CompositionPlan(
                        synthetic_name=f"synthetic_{structural}_{logic}_{index}",
                        source_lemmas=sources,
                        domain_family="generic",
                        structural_class=structural,
                        generator_family=(
                            f"final-only:{structural}-source-iff-tactic-v3"
                            if logic == "iff"
                            else f"{structural}-source-and-tactic-v3"
                        ),
                        normalized_proof_dag=f"{structural}-{logic}-{index}",
                        derivation_family_fingerprint=(
                            f"{structural}-{logic}-{index}"
                        ),
                        relation_edges=tuple(
                            (
                                sources[source_index].declaration_name,
                                sources[source_index + 1].declaration_name,
                                "fixture",
                            )
                            for source_index in range(arity - 1)
                        ),
                    )
                )

    assignments = _balanced_role_selection(
        plans,
        role_counts={"training": 12, "validation": 12, "test": 12},
        seed="fixture",
    )
    by_name = {plan.synthetic_name: plan for plan in plans}
    for role in ("training", "validation", "test"):
        observed = {
            (
                by_name[name].structural_class,
                "iff"
                if by_name[name].generator_family.startswith("final-only:")
                else "and",
            )
            for name, assigned in assignments.items()
            if assigned == role
        }
        assert observed == {
            (structural, logic)
            for structural in ("direct", "branching", "deep")
            for logic in ("and", "iff")
        }
