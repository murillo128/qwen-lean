from __future__ import annotations

from dataclasses import dataclass

import pytest

from qwen_lean.dataset_v2_extraction import (
    candidate_from_traced_theorem,
    select_candidates,
    split_whole_declaration,
    substitute_proofs,
)


@dataclass(frozen=True)
class _Pos:
    line_nb: int
    column_nb: int


class _ProofNode:
    def __init__(self, start: _Pos, end: _Pos, dependencies: tuple[str, ...]) -> None:
        self._start = start
        self._end = end
        self._dependencies = dependencies

    def get_closure(self):
        return self._start, self._end

    def traverse_preorder(self, callback, node_cls=None):
        del node_cls
        for dependency in self._dependencies:
            callback(type("Ident", (), {"full_name": dependency})(), [])


class CommandTheoremNode:
    pass


class _Theorem:
    def __init__(self, *, proof_start: _Pos, proof_end: _Pos, private: bool = False) -> None:
        self.start = _Pos(1, 1)
        self.end = proof_end
        self.is_private = private
        self.theorem = type("Theorem", (), {"full_name": "identity"})()
        self.ast = CommandTheoremNode()
        self._proof = _ProofNode(proof_start, proof_end, ("id", "Eq.refl"))

    def get_proof_node(self):
        return self._proof


def _candidate(source: str):
    line = source.splitlines()[0]
    marker = line.index(":=") + 4
    theorem = _Theorem(
        proof_start=_Pos(1, marker),
        proof_end=_Pos(1, len(line) + 1),
    )
    return candidate_from_traced_theorem(
        theorem,
        source=source,
        file_path="Mathlib/Test.lean",
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="a" * 40,
        provenance="real-mathlib",
    )


def test_term_candidate_uses_complete_source_expression_and_actual_proof_dependencies() -> None:
    candidate = _candidate("theorem identity (P : Prop) : P → P := fun h => h")

    assert candidate.source_expression == "fun h => h"
    assert candidate.transformation_kind == "term-to-exact"
    assert candidate.canonical_proof == "by\n  exact (\n    fun h => h\n  )"
    assert candidate.verification_status == "pending"
    assert candidate.resolved_dependencies == ("Eq.refl", "id")


def test_by_candidate_preserves_source_semantics() -> None:
    candidate = _candidate("theorem identity (P : Prop) (h : P) : P := by exact h")

    assert candidate.source_expression == "by exact h"
    assert candidate.canonical_proof == "by exact h"
    assert candidate.completion == "exact h"
    assert candidate.transformation_kind == "none"
    assert candidate.verification_status == "accepted"


def test_placeholder_candidate_is_classified_before_training_eligibility() -> None:
    with pytest.raises(ValueError, match="proof-placeholder"):
        _candidate("theorem identity : True := by sorry")


def test_substitution_reconstructs_term_proof_at_declared_source_position() -> None:
    source = "theorem identity (P : Prop) : P → P := fun h => h\n"
    candidate = _candidate(source)
    reconstructed = substitute_proofs(source, [candidate])

    assert reconstructed == (
        "theorem identity (P : Prop) : P → P := by\n"
        "  exact (\n"
        "    fun h => h\n"
        "  )\n"
    )


def test_deterministic_selection_requires_requested_transformation_population() -> None:
    term = _candidate("theorem identity (P : Prop) : P → P := fun h => h")
    selected = select_candidates(
        [term], count=1, seed="preflight", transformation_kind="term-to-exact"
    )
    assert selected == [term]
    with pytest.raises(ValueError, match="requires 2, found 1"):
        select_candidates(
            [term], count=2, seed="preflight", transformation_kind="term-to-exact"
        )


def test_split_whole_declaration_recovers_outer_calc_proof() -> None:
    declaration, proof = split_whole_declaration(
        "theorem renamed (n : Nat) : n = n := calc\n  n = n := by rfl"
    )
    assert declaration == "theorem renamed (n : Nat) : n = n"
    assert proof == "calc\n  n = n := by rfl"


def test_split_whole_declaration_skips_comment_assignments() -> None:
    declaration, proof = split_whole_declaration(
        "lemma named : True /- := ignored -/ := by\n  trivial"
    )
    assert declaration == "lemma named : True /- := ignored -/"
    assert proof == "by\n  trivial"
