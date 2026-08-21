from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qwen_lean.dataset_v2 import (
    assign_synthetic_roles,
    iter_optimizer_examples,
    merge_statement_records,
    plan_training_examples,
    read_membership_view,
    validate_prime_coverage,
    validate_role_isolation,
    validate_synthetic_source_resolvability,
    write_records,
    write_membership_view,
)
from qwen_lean.dataset_v2_contract import (
    derivation_family_fingerprint,
    proof_fingerprint,
    proof_variant_id,
    statement_fingerprint_v2,
    statement_id,
)
from qwen_lean.dataset_v2_pipeline import (
    GENERAL_TRAIN_VIEW,
    RIEMANN_TRAIN_VIEW,
    build_training_view_memberships,
)
from qwen_lean.dataset_v2_schema import (
    DATASET_V2_SCHEMA_VERSION,
    DatasetV2Record,
    EnvironmentContext,
    LengthMetadata,
    ProofVariant,
    ProofVerification,
)


def _record(
    name: str,
    *,
    role: str = "training",
    proof: str = "by\n  exact True.intro",
    provenance: str = "real-mathlib",
    family: str | None = None,
    structural: str | None = None,
    utf8_bytes: int = 100,
    proposition: str = "True",
) -> DatasetV2Record:
    declaration = f"theorem {name} : {proposition}"
    sid = statement_id(declaration)
    variant = ProofVariant(
        proof_variant_id=proof_variant_id(sid, proof),
        source_expression=proof,
        canonical_proof=proof,
        completion=proof[2:].lstrip(),
        transformation_kind="none",
        proof_fingerprint=proof_fingerprint(proof),
        resolved_dependencies=("True.intro",),
        verification=ProofVerification(
            status="accepted",
            environment_id="env-v2",
            method="fixture",
            evidence_id="fixture-evidence",
        ),
        source_declaration_name=name,
        source_repository="https://example.invalid/source",
        source_revision="a" * 40,
        source_file="Mathlib/Test.lean",
    )
    derivation = (
        None
        if family is None
        else derivation_family_fingerprint(
            [f"lemma-{name}", "lemma-shared"],
            normalized_proof_dag=f"root({structural})",
            generator_family=family,
        )
    )
    return DatasetV2Record(
        schema_version=DATASET_V2_SCHEMA_VERSION,
        statement_id=sid,
        canonical_declaration=declaration,
        normalized_statement_fingerprint=statement_fingerprint_v2(declaration),
        role=role,  # type: ignore[arg-type]
        sampling_group_id=sid,
        provenance=provenance,  # type: ignore[arg-type]
        environment=EnvironmentContext(
            environment_id="env-v2",
            lean_toolchain="leanprover/lean4:v4.32.2",
            repository="https://example.invalid/source",
            revision="a" * 40,
            mathlib_revision="b" * 40,
            file_path="Mathlib/Test.lean",
            module="Mathlib.Test",
            imports=("Mathlib",),
            source_span=None,
            context_kind="generated-module",
            target_compatibility="verified-target-environment",
        ),
        proof_variants=(variant,),
        topic_tags=("domain:generic",),
        memberships=(),
        length=LengthMetadata(20, len(proof), len(proof) - 2, 1, 2, utf8_bytes),
        derivation_family_fingerprint=derivation,
        generator_family=family,
        structural_class=structural,
        normalized_proof_dag=None if family is None else f"root({structural})",
        source_lemma_ids=() if family is None else (f"lemma-{name}", "lemma-shared"),
        source_relation_edges=(
            ()
            if family is None
            else ((f"lemma-{name}", "lemma-shared", "fixture-relevance"),)
        ),
        shortcut_checks=() if role == "training" else ("assumption:failed",),
    )


def test_loader_samples_one_variant_per_statement_without_weight_multiplication() -> None:
    first = _record("same", proof="by\n  exact True.intro")
    second = _record("renamed", proof="by\n  trivial")
    merged = merge_statement_records([first, second])

    assert len(merged) == 1
    assert len(merged[0].proof_variants) == 2
    planned = plan_training_examples(merged)
    assert len(planned) == 1
    assert len(planned[0]["proof_variant_ids"]) == 2
    assert len(list(iter_optimizer_examples(merged, variant_seed="seed"))) == 1


def test_merge_preserves_cross_source_real_variants_and_longest_length() -> None:
    first = _record("first", proof="by\n  exact True.intro")
    second_variant = _record(
        "second",
        proof="by\n  have witness : True := True.intro\n  exact witness",
        provenance="external-lean",
    )
    merged = merge_statement_records([first, second_variant])

    assert len(merged) == 1
    assert merged[0].provenance == "mixed-real"
    assert len(merged[0].proof_variants) == 2
    assert merged[0].length.proof_chars == len(
        second_variant.proof_variants[0].canonical_proof
    )
    assert merged[0].length.utf8_bytes == len(
        (
            merged[0].canonical_declaration
            + " := "
            + second_variant.proof_variants[0].canonical_proof
        ).encode("utf-8")
    )


def test_loader_refuses_to_truncate_or_silently_skip_long_statement() -> None:
    record = _record("long", utf8_bytes=10_000)
    assert plan_training_examples([record], max_utf8_bytes=512)[0]["context_status"] == "long-context"
    with pytest.raises(ValueError, match="refusing to truncate or omit"):
        list(iter_optimizer_examples([record], variant_seed="seed", max_utf8_bytes=512))


def test_gzip_corpus_packaging_is_byte_deterministic(tmp_path: Path) -> None:
    record = _record("deterministic")
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    assert write_records(first, [record]) == write_records(second, [record])
    assert first.read_bytes() == second.read_bytes()


def test_membership_view_is_id_only_deterministic_and_resolvable(tmp_path: Path) -> None:
    records = [
        _record("first"),
        _record("second", proposition="False → False"),
        _record("evaluation", role="test", proposition="1 = 1"),
    ]
    selected = {records[0].statement_id, records[1].statement_id}
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    assert write_membership_view(first, records, selected) == write_membership_view(
        second, records, reversed(sorted(selected))
    )
    assert first.read_bytes() == second.read_bytes()
    assert {
        record.statement_id for record in read_membership_view(first, records)
    } == selected

    with pytest.raises(ValueError, match="non-training"):
        write_membership_view(first, records, [records[2].statement_id])


def test_training_views_cover_prime_specialist_and_support_rows() -> None:
    support = _record("support")
    prime = _record("prime", proposition="False → False")
    prime = replace(
        prime,
        topic_tags=("domain:prime-number-theory", "prime-family:prime-counting-pnt"),
        memberships=("number-theory-wide-v1",),
        proof_variants=(
            replace(
                prime.proof_variants[0],
                resolved_dependencies=("support",),
            ),
        ),
    )
    evaluation = _record("evaluation", role="validation", proposition="1 = 1")
    views, validation = build_training_view_memberships(
        [support, prime, evaluation]
    )

    assert views[GENERAL_TRAIN_VIEW] == {support.statement_id, prime.statement_id}
    assert views[RIEMANN_TRAIN_VIEW] == {support.statement_id, prime.statement_id}
    assert validation["riemann_subset_general"] is True
    assert validation["prime_training_in_riemann"] == 1
    assert validation["historical_membership_coverage"]["number-theory-wide-v1"] == {
        "training_statements": 1,
        "riemann_view_statements": 1,
    }


def test_riemann_view_requires_and_includes_synthetic_prime_source_support() -> None:
    support = _record("synthetic_support")
    synthetic = _record(
        "synthetic_prime",
        provenance="synthetic",
        family="direct-family",
        structural="direct",
        proposition="False → False",
    )
    synthetic = replace(
        synthetic,
        topic_tags=("domain:prime-number-theory", "prime-family:pnt-plus"),
        source_lemma_ids=(support.statement_id,),
        source_relation_edges=(),
    )
    views, validation = build_training_view_memberships([support, synthetic])

    assert views[RIEMANN_TRAIN_VIEW] == {
        support.statement_id,
        synthetic.statement_id,
    }
    assert validation["support_neighborhood"] == {
        "dependency_name_references": 1,
        "resolved_dependency_statements": 0,
        "synthetic_source_lemma_statements": 1,
        "synthetic_prime_source_lemma_statements": 1,
        "missing_source_lemma_statements": 0,
    }

    unresolved = replace(synthetic, source_lemma_ids=("missing-statement-id",))
    with pytest.raises(ValueError, match="source lemma support is absent"):
        build_training_view_memberships([support, unresolved])


def test_synthetic_roles_keep_statement_family_and_proof_out_of_other_roles() -> None:
    records = [
        _record(
            f"synthetic_{index}",
            provenance="synthetic",
            family=f"family-{index}",
            structural=("direct", "branching", "deep")[index % 3],
            proposition=f"{index} = {index}",
        )
        for index in range(30)
    ]
    assigned = assign_synthetic_roles(records, seed="dataset-v2-split-v1")

    assert {record.role for record in assigned} == {"training", "validation", "test"}
    assert validate_role_isolation(assigned) == {
        "cross_role_statements": 0,
        "cross_role_derivation_families": 0,
        "cross_role_proofs": 0,
    }


def test_synthetic_source_lemmas_must_resolve_to_canonical_training() -> None:
    source = _record("source")
    synthetic = _record(
        "synthetic",
        provenance="synthetic",
        family="direct-family",
        structural="direct",
        proposition="False → False",
    )
    synthetic = replace(
        synthetic,
        source_lemma_ids=(source.statement_id, source.statement_id),
        source_relation_edges=(
            (source.statement_id, source.statement_id, "fixture-relevance"),
        ),
    )
    assert validate_synthetic_source_resolvability([source, synthetic]) == {
        "synthetic_records": 1,
        "source_lemma_references": 2,
        "missing_source_lemma_references": 0,
        "missing_source_statement_ids": 0,
    }

    with pytest.raises(ValueError, match="do not resolve"):
        validate_synthetic_source_resolvability([synthetic])


def test_validation_proof_variant_cannot_reappear_in_training() -> None:
    validation = _record("validation", role="validation")
    training = replace(_record("training"), proof_variants=validation.proof_variants)
    with pytest.raises(ValueError, match="role leakage"):
        validate_role_isolation([validation, training])


def test_prime_coverage_rejects_eligible_omission() -> None:
    record = _record("prime_training")
    included = {
        "coverage_id": "prime-a",
        "disposition": "included-training",
        "verified_lean": True,
        "legally_usable": True,
        "target_compatible": True,
        "statement_ids": [record.statement_id],
    }
    assert validate_prime_coverage([included], [record])[
        "verified_legal_target_compatible_omissions"
    ] == 0

    omitted = {
        **included,
        "coverage_id": "prime-b",
        "disposition": "source-integrity-blocked",
        "statement_ids": [],
    }
    with pytest.raises(ValueError, match="prime omissions"):
        validate_prime_coverage([included, omitted], [record])
