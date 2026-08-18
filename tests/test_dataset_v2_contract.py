from qwen_lean.dataset_v2_contract import (
    canonicalize_proof_expression,
    derivation_family_fingerprint,
    proof_fingerprint,
)


def test_canonicalize_existing_by_proof_preserves_completion() -> None:
    value = canonicalize_proof_expression("by\r\n  simpa using h\r\n")

    assert value.source_expression == "by\n  simpa using h"
    assert value.canonical_proof == "by\n  simpa using h"
    assert value.completion == "simpa using h"
    assert value.transformation == "none"
    assert value.requires_lean_verification


def test_canonicalize_term_proof_wraps_exact_without_claiming_verification() -> None:
    value = canonicalize_proof_expression("fun h => h")

    assert value.source_expression == "fun h => h"
    assert value.canonical_proof == "by\n  exact (\n    fun h => h\n  )"
    assert value.completion == "exact (\n    fun h => h\n  )"
    assert value.transformation == "term-to-exact"
    assert value.requires_lean_verification


def test_proof_fingerprint_uses_lean_lexical_normalization() -> None:
    assert proof_fingerprint("by\r\n  exact h -- transport comment\r\n") == proof_fingerprint(
        "by\n exact h\n"
    )
    assert proof_fingerprint("by\n  exact h") != proof_fingerprint("by\n  simpa using h")


def test_derivation_family_is_order_independent_for_source_lemmas() -> None:
    left = derivation_family_fingerprint(
        ["lemma-b", "lemma-a", "lemma-a"],
        normalized_proof_dag="root(branch(a,b))",
        generator_family="branching-v1",
    )
    right = derivation_family_fingerprint(
        ["lemma-a", "lemma-b"],
        normalized_proof_dag="root(branch(a,b))",
        generator_family="branching-v1",
    )

    assert left == right


def test_derivation_family_changes_with_dag_or_generator_family() -> None:
    baseline = derivation_family_fingerprint(
        ["lemma-a", "lemma-b"],
        normalized_proof_dag="root(branch(a,b))",
        generator_family="branching-v1",
    )

    assert baseline != derivation_family_fingerprint(
        ["lemma-a", "lemma-b"],
        normalized_proof_dag="root(chain(a,b))",
        generator_family="branching-v1",
    )
    assert baseline != derivation_family_fingerprint(
        ["lemma-a", "lemma-b"],
        normalized_proof_dag="root(branch(a,b))",
        generator_family="branching-v2",
    )
