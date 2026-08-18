from qwen_lean.dataset_v2_contract import (
    canonicalize_equation_clauses,
    canonicalize_proof_expression,
    derivation_family_fingerprint,
    proof_fingerprint,
    proof_variant_id,
    statement_fingerprint_v2,
    statement_id,
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


def test_equation_clauses_are_wrapped_as_an_explicit_function() -> None:
    value = canonicalize_equation_clauses("| 0 => by simp\n| n + 1 => by omega")

    assert value.source_expression.startswith("| 0")
    assert "exact (\n    @fun\n      | 0 => by simp" in value.canonical_proof
    assert value.transformation == "equations-to-fun-exact"


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


def test_statement_identity_ignores_name_kind_comments_and_binder_alpha_renaming() -> None:
    left = "theorem first (x : Nat) (h : x = x) : x = x"
    right = "lemma second (value : Nat) (proof : value = value) : value = value -- same"

    assert statement_fingerprint_v2(left) == statement_fingerprint_v2(right)
    assert statement_id(left) == statement_id(right)


def test_proof_variant_identity_is_scoped_to_statement() -> None:
    proof = "by\n  exact h"
    assert proof_variant_id("statement-a", proof) == proof_variant_id("statement-a", proof)
    assert proof_variant_id("statement-a", proof) != proof_variant_id("statement-b", proof)


def test_statement_identity_normalizes_quantified_and_lambda_binders() -> None:
    left = "theorem first : (∀ x y : Nat, x = y → y = x) → (fun z => z) = id"
    right = "lemma second : (∀ a b : Nat, a = b → b = a) → (fun value => value) = id"

    assert statement_fingerprint_v2(left) == statement_fingerprint_v2(right)


def test_statement_identity_is_scope_safe_for_shadowed_binders() -> None:
    left = "theorem first (x : Nat) : (∀ x : Nat, x = x) ∧ x = x"
    right = "lemma second (y : Nat) : (∀ z : Nat, z = z) ∧ y = y"
    changed = "lemma third (y : Nat) : (∀ z : Nat, z = y) ∧ y = y"

    assert statement_fingerprint_v2(left) == statement_fingerprint_v2(right)
    assert statement_fingerprint_v2(left) != statement_fingerprint_v2(changed)


def test_statement_identity_normalizes_explicit_universe_names() -> None:
    left = "theorem first.{u, v} (α : Type u) (β : Type v) : α = α"
    right = "lemma second.{x, y} (A : Type x) (B : Type y) : A = A"

    assert statement_fingerprint_v2(left) == statement_fingerprint_v2(right)
