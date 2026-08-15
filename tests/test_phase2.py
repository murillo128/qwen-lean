from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from qwen_lean.phase2_corpus import (
    ExtractedRecord,
    RawTheorem,
    build_file_components,
    derive_completion,
    evaluate_raw_theorem,
    exclude_ambiguous_record_identities,
    exclude_contamination,
    finalize_records,
    load_phase2_dataset,
    minif2f_statement_fingerprints,
    statement_fingerprint,
    validate_record_source_text,
    validate_split_hygiene,
    write_jsonl_splits,
)
from qwen_lean.phase2_extraction import Phase2Config
from qwen_lean.phase2_schema import (
    PHASE2_DATASET_SCHEMA_VERSION,
    PHASE2_MANIFEST_SCHEMA_VERSION,
    MathlibProofRecord,
    SourcePosition,
    SourceSpan,
)
from qwen_lean.phase2_verification import run_lean_source

SOURCE_REPOSITORY = "https://github.com/leanprover-community/mathlib4"
SOURCE_REVISION = "81a5d257c8e410db227a6665ed08f64fea08e997"
PROPORTIONS = {"train": 0.9, "validation": 0.05, "heldout": 0.05}
ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        assert add_special_tokens is False
        return text.split()


@pytest.mark.parametrize(
    ("diagnostic", "expected_status"),
    [
        (
            "Reconstructed.lean:1:1: error(lean.unknownIdentifier): Unknown identifier",
            "rejected",
        ),
        ("error: external command 'git' exited with code 128", "infrastructure_error"),
    ],
)
def test_lean_check_distinguishes_source_rejection_from_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic: str,
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        "qwen_lean.phase2_verification.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr=diagnostic
        ),
    )

    outcome = run_lean_source(
        "theorem example : True := by trivial", tmp_path, timeout_seconds=1
    )

    assert outcome.status == expected_status


def test_phase2_config_matches_project_source_and_tokenizer_pins() -> None:
    config = Phase2Config.load(ROOT / "config/phase2-mathlib.json")

    config.validate_project_pins()


def _span(line: int, start: int, end: int) -> SourceSpan:
    return SourceSpan(SourcePosition(line, start), SourcePosition(line, end))


def _raw(
    name: str,
    *,
    file_path: str = "Mathlib/Test.lean",
    statement: str | None = None,
    proof: str | None = "by\n  rfl",
    private: bool = False,
    kind: str = "theorem",
) -> RawTheorem:
    declaration = statement or f"{kind} {name} (x : Nat) : x = x := "
    return RawTheorem(
        file_path=file_path,
        declaration_name=name,
        declaration_kind=kind,
        source_span=_span(1, 1, 60),
        declaration_span=_span(1, 1, 42),
        proof_span=_span(1, 42, 60),
        declaration=declaration,
        proof=proof,
        premises=("Eq.refl",),
        is_private=private,
    )


def _extracted(
    name: str,
    *,
    file_path: str,
    proposition: str = "x = x",
    proof: str = "by\n  rfl",
) -> ExtractedRecord:
    result = evaluate_raw_theorem(
        _raw(
            name,
            file_path=file_path,
            statement=f"theorem {name} (x : Nat) : {proposition} := ",
            proof=proof,
        ),
        source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_REVISION,
    )
    assert result.filter_reason is None
    assert result.record is not None
    return result.record


def _corpus(count: int = 20) -> list[ExtractedRecord]:
    return [
        _extracted(
            f"test_{index}",
            file_path=f"Mathlib/Synthetic/File{index:03d}.lean",
            proposition=f"x + {index} = x + {index}",
        )
        for index in range(count)
    ]


def test_phase2_record_schema_round_trips_without_losing_required_data() -> None:
    record = finalize_records(
        _corpus(), WordTokenizer(), PROPORTIONS, seed="schema-test"
    )[0]

    restored = MathlibProofRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.source_revision == SOURCE_REVISION
    assert restored.declaration
    assert restored.proof.startswith("by")
    assert restored.split in PROPORTIONS
    assert restored.token_lengths.declaration_and_completion > 0


def test_record_source_spans_resolve_to_the_retained_declaration_and_proof() -> None:
    source = "theorem source_bound (x : Nat) : x = x := by\n  rfl\n"
    declaration = "theorem source_bound (x : Nat) : x = x :="
    proof = "by\n  rfl"
    result = evaluate_raw_theorem(
        RawTheorem(
            file_path="Mathlib/Synthetic/Bound.lean",
            declaration_name="source_bound",
            declaration_kind="theorem",
            source_span=SourceSpan(SourcePosition(1, 1), SourcePosition(2, 6)),
            declaration_span=SourceSpan(SourcePosition(1, 1), SourcePosition(1, 43)),
            proof_span=SourceSpan(SourcePosition(1, 43), SourcePosition(2, 6)),
            declaration=declaration,
            proof=proof,
        ),
        source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_REVISION,
    )
    assert result.record is not None

    validate_record_source_text(result.record, source)

    with pytest.raises(ValueError, match="source proof span differs"):
        validate_record_source_text(result.record, source.replace("rfl", "simp"))


def test_by_proof_completion_removes_only_delimiter_and_separator_whitespace() -> None:
    proof = "by\n    simpa [Nat.add_comm] using h  "

    assert derive_completion(proof) == "simpa [Nat.add_comm] using h  "
    assert proof == "by\n    simpa [Nat.add_comm] using h  "


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (_raw("empty", proof="by   "), "empty_completion"),
        (_raw("term", proof="fun x => x"), "non_by_proof"),
        (_raw("private_decl", private=True), "private"),
        (
            _raw(
                "private_syntax",
                statement="private theorem private_syntax : True := ",
            ),
            "private",
        ),
        (_raw("placeholder", proof="by\n  sorry"), "proof_placeholder"),
        (_raw("admitted", proof="by\n  admit"), "proof_placeholder"),
        (
            _raw("where_proof", statement="theorem where_proof : True where"),
            "unsupported_proof_delimiter",
        ),
    ],
)
def test_ineligible_examples_have_explicit_filter_reasons(
    raw: RawTheorem, reason: str
) -> None:
    result = evaluate_raw_theorem(
        raw,
        source_repository=SOURCE_REPOSITORY,
        source_revision=SOURCE_REVISION,
    )

    assert result.record is None
    assert result.filter_reason == reason


def test_statement_fingerprint_ignores_name_comments_and_insignificant_whitespace() -> (
    None
):
    first = "@[simp] theorem alpha (x : Nat) : ∀ y, x ≠ y → x = x"
    second = "@[simp]\n theorem   beta(x:Nat):∀ y,x≠y→x=x -- transport comment"
    lemma = "@[simp] lemma delta (x : Nat) : ∀ y, x ≠ y → x = x"
    changed = "@[simp] theorem gamma (x : Nat) : x + 1 = x"

    assert statement_fingerprint(first) == statement_fingerprint(second)
    assert statement_fingerprint(first) == statement_fingerprint(lemma)
    assert statement_fingerprint(first) != statement_fingerprint(changed)


def test_statement_fingerprint_preserves_explicit_universes_after_the_name() -> None:
    first = "theorem alpha.{u} (x : Type u) : x = x"
    second = "theorem beta.{u} (x : Type u) : x = x"
    changed = "theorem gamma.{u, v} (x : Type u) : x = x"

    assert statement_fingerprint(first) == statement_fingerprint(second)
    assert statement_fingerprint(first) != statement_fingerprint(changed)


def test_colliding_source_identities_are_all_excluded_as_ambiguous() -> None:
    first = _extracted(
        "collision", file_path="Mathlib/Synthetic/Collision.lean", proposition="x = x"
    )
    second = _extracted(
        "collision",
        file_path="Mathlib/Synthetic/Collision.lean",
        proposition="x + 1 = x + 1",
    )

    retained, excluded, ambiguous_ids = exclude_ambiguous_record_identities(
        [first, second]
    )

    assert retained == []
    assert excluded == [first, second]
    assert ambiguous_ids == {first.id}


def test_same_file_records_are_assigned_to_one_split() -> None:
    records = _corpus(18) + [
        _extracted("same_a", file_path="Mathlib/Synthetic/Together.lean"),
        _extracted(
            "same_b",
            file_path="Mathlib/Synthetic/Together.lean",
            proposition="x + 1 = x + 1",
        ),
    ]
    finalized = finalize_records(
        records, WordTokenizer(), PROPORTIONS, seed="same-file-test"
    )

    together = {
        record.split
        for record in finalized
        if record.file_path == "Mathlib/Synthetic/Together.lean"
    }
    assert len(together) == 1
    assert validate_split_hygiene(finalized)["cross_split_files"] == 0


def test_duplicate_statements_connect_different_source_files() -> None:
    records = [
        _extracted("alpha", file_path="Mathlib/Synthetic/A.lean"),
        _extracted("beta", file_path="Mathlib/Synthetic/B.lean"),
        _extracted(
            "gamma",
            file_path="Mathlib/Synthetic/C.lean",
            proposition="x + 1 = x + 1",
        ),
    ]

    file_components, _ = build_file_components(records)

    assert (
        file_components["Mathlib/Synthetic/A.lean"]
        == file_components["Mathlib/Synthetic/B.lean"]
    )
    assert (
        file_components["Mathlib/Synthetic/A.lean"]
        != file_components["Mathlib/Synthetic/C.lean"]
    )


def test_split_assignment_is_deterministic_and_matches_synthetic_targets() -> None:
    records = _corpus(20)

    first = finalize_records(records, WordTokenizer(), PROPORTIONS, seed="stable")
    second = finalize_records(
        list(reversed(records)), WordTokenizer(), PROPORTIONS, seed="stable"
    )

    assert {record.id: record.split for record in first} == {
        record.id: record.split for record in second
    }
    assert {
        split: sum(record.split == split for record in first) for split in PROPORTIONS
    } == {
        "train": 18,
        "validation": 1,
        "heldout": 1,
    }


def test_exact_minif2f_statement_match_is_excluded_regardless_of_name() -> None:
    source = """\
import MiniF2F.ProblemImports
theorem benchmark_name (x : Nat) : x = x := by
  sorry
"""
    fingerprints = minif2f_statement_fingerprints([source])
    matching = _extracted("different_name", file_path="Mathlib/Synthetic/Match.lean")
    unrelated = _extracted(
        "unrelated",
        file_path="Mathlib/Synthetic/Other.lean",
        proposition="x + 1 = x + 1",
    )

    retained, excluded = exclude_contamination(
        [matching, unrelated], set(fingerprints.values())
    )

    assert [record.declaration_name for record in excluded] == ["different_name"]
    assert [record.declaration_name for record in retained] == ["unrelated"]


def test_equal_proof_text_does_not_connect_unrelated_statements() -> None:
    records = [
        _extracted("alpha", file_path="Mathlib/Synthetic/A.lean", proposition="x = x"),
        _extracted(
            "beta",
            file_path="Mathlib/Synthetic/B.lean",
            proposition="x + 1 = x + 1",
        ),
    ]
    assert records[0].proof == records[1].proof

    file_components, _ = build_file_components(records)

    assert (
        file_components[records[0].file_path] != file_components[records[1].file_path]
    )


def test_jsonl_splits_load_through_hugging_face_datasets(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    records = finalize_records(
        _corpus(), WordTokenizer(), PROPORTIONS, seed="loader-test"
    )
    counts = write_jsonl_splits(records, tmp_path)
    manifest = {
        "schema_version": PHASE2_MANIFEST_SCHEMA_VERSION,
        "dataset_schema_version": PHASE2_DATASET_SCHEMA_VERSION,
        "splits": {split: {"records": count} for split, count in counts.items()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = load_phase2_dataset(tmp_path)

    assert {split: len(dataset[split]) for split in dataset} == counts
    assert set(dataset["train"].column_names) == set(
        MathlibProofRecord.__dataclass_fields__
    )
    assert all(value == "heldout" for value in dataset["heldout"]["split"])
