from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from qwen_lean.dataset_v2_extraction import (
    candidate_from_traced_theorem,
    select_candidates,
    split_whole_declaration,
    substitute_proofs,
    verify_transformed_candidates,
)
from qwen_lean import dataset_v2_extraction


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


def test_equation_clause_theorem_is_recovered_as_a_verified_candidate_shape() -> None:
    source = (
        "theorem equation : ∀ n : Nat, n = n\n"
        "  | 0 => by rfl\n"
        "  | n + 1 => by rfl\n"
    )
    theorem = _Theorem(
        proof_start=_Pos(2, 3),
        proof_end=_Pos(3, len("  | n + 1 => by rfl") + 1),
    )
    candidate = candidate_from_traced_theorem(
        theorem,
        source=source,
        file_path="Mathlib/Test.lean",
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="a" * 40,
        provenance="real-mathlib",
    )

    assert candidate.transformation_kind == "equations-to-fun-exact"
    assert candidate.source_expression.startswith("| 0 =>")
    reconstructed = substitute_proofs(source, [candidate])
    assert ":= by\n  exact (\n    @fun\n      | 0 => by rfl" in reconstructed


def test_where_theorem_is_recovered_as_an_explicit_structure_proof() -> None:
    source = (
        "theorem identity (P : Prop) : P ↔ P where\n"
        "  mp h := h\n"
        "  mpr h := h\n"
    )
    theorem = _Theorem(
        proof_start=_Pos(2, 3),
        proof_end=_Pos(3, len("  mpr h := h") + 1),
    )
    candidate = candidate_from_traced_theorem(
        theorem,
        source=source,
        file_path="Mathlib/Test.lean",
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="a" * 40,
        provenance="real-mathlib",
    )

    assert candidate.declaration == "theorem identity (P : Prop) : P ↔ P"
    assert candidate.transformation_kind == "where-to-structure-exact"
    assert candidate.source_expression.startswith("where\n")
    reconstructed = substitute_proofs(source, [candidate])
    assert reconstructed.startswith(
        "theorem identity (P : Prop) : P ↔ P := by\n  exact {\n"
    )
    assert "mp h := h" in reconstructed
    assert "mpr h := h" in reconstructed


def test_where_theorem_recovery_crosses_comments_before_first_field() -> None:
    source = (
        "theorem identity (P : Prop) : P ↔ P where\n"
        "  -- the trace starts at the first field, after this comment\n"
        "  mp h := h\n"
        "  mpr h := h\n"
    )
    theorem = _Theorem(
        proof_start=_Pos(3, 3),
        proof_end=_Pos(4, len("  mpr h := h") + 1),
    )
    candidate = candidate_from_traced_theorem(
        theorem,
        source=source,
        file_path="Mathlib/Test.lean",
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="a" * 40,
        provenance="real-mathlib",
    )

    assert candidate.transformation_kind == "where-to-structure-exact"
    assert "-- the trace starts" in candidate.source_expression
    assert candidate.canonical_proof.startswith("by\n  exact {")


def test_outer_assignment_recovers_complete_term_when_trace_points_inside_it() -> None:
    source = (
        "theorem identity : Function.Injective (fun n : Nat => n) :=\n"
        "  Function.Injective.of_comp\n"
        "    (f := fun n : Nat => n)\n"
        "    Function.injective_id\n"
    )
    theorem = _Theorem(
        proof_start=_Pos(4, 5),
        proof_end=_Pos(4, len("    Function.injective_id") + 1),
    )
    candidate = candidate_from_traced_theorem(
        theorem,
        source=source,
        file_path="Mathlib/Test.lean",
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="a" * 40,
        provenance="real-mathlib",
    )

    assert candidate.transformation_kind == "term-to-exact"
    assert candidate.source_expression.startswith("Function.Injective.of_comp")
    assert "(f := fun n : Nat => n)" in candidate.source_expression
    reconstructed = substitute_proofs(source, [candidate])
    assert reconstructed.startswith(
        "theorem identity : Function.Injective (fun n : Nat => n) :=\n  by\n"
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


def _verify_candidate(
    candidate,
    *,
    source_root: Path,
    cache_dir: Path | None = None,
):
    return verify_transformed_candidates(
        [candidate],
        source_roots={
            (candidate.source_repository, candidate.source_revision): source_root
        },
        target_root=source_root,
        environment_id="test-environment",
        evidence_id="test-evidence",
        workers=1,
        timeout_seconds=1,
        group_cache_dir=cache_dir,
    )


def test_verification_rejects_candidate_when_unchanged_source_is_not_reconstructible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "theorem identity (P : Prop) : P → P := fun h => h\n"
    candidate = _candidate(source)
    source_path = tmp_path / candidate.file_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    calls = 0

    def reject_source(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return "rejected", 1, 0.01, "unrelated source failure"

    monkeypatch.setattr(dataset_v2_extraction, "_run_reconstructed_file", reject_source)
    verified, files = _verify_candidate(candidate, source_root=tmp_path)

    assert calls == 2
    assert verified[0].verification_status == "baseline-rejected"
    assert verified[0].verification_method == "source-file-baseline-verification"
    assert files[0].status == "baseline-rejected"
    assert len(files[0].rejected_candidate_ids) == 1


def test_verification_group_cache_survives_across_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "theorem identity (P : Prop) : P → P := fun h => h\n"
    candidate = _candidate(source)
    source_path = tmp_path / candidate.file_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(
        dataset_v2_extraction,
        "_run_reconstructed_file",
        lambda *args, **kwargs: ("accepted", 0, 0.01, ""),
    )
    first = _verify_candidate(candidate, source_root=tmp_path, cache_dir=cache_dir)

    def unexpected_verification(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cached group was verified again")

    monkeypatch.setattr(
        dataset_v2_extraction,
        "_run_reconstructed_file",
        unexpected_verification,
    )
    second = _verify_candidate(candidate, source_root=tmp_path, cache_dir=cache_dir)

    assert second == first
    assert len(list(cache_dir.glob("*.pkl"))) == 1


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
