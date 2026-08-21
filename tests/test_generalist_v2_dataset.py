from __future__ import annotations

from pathlib import Path

from qwen_lean.dataset_v2_schema import DatasetV2Record
from qwen_lean.generalist_v2_dataset import (
    dataset_record_preamble,
    generalist_variants,
    read_training_membership,
)


def _record() -> DatasetV2Record:
    return DatasetV2Record.from_dict(
        {
            "schema_version": "lean-whole-proof-v2",
            "statement_id": "statement-1",
            "canonical_declaration": "theorem fixture : True",
            "normalized_statement_fingerprint": "fingerprint",
            "role": "training",
            "sampling_group_id": "statement-1",
            "provenance": "synthetic",
            "environment": {
                "environment_id": "environment",
                "lean_toolchain": "leanprover/lean4:v4.32.2",
                "repository": "https://example.invalid/repository",
                "revision": "revision",
                "mathlib_revision": "mathlib-revision",
                "file_path": "Fixture.lean",
                "module": "Fixture",
                "imports": ["Mathlib", "Mathlib"],
                "source_span": None,
                "context_kind": "persisted-imports",
                "target_compatibility": "verified-target-environment",
            },
            "proof_variants": [
                {
                    "proof_variant_id": "proof-1",
                    "source_expression": "by trivial",
                    "canonical_proof": "by\n  trivial",
                    "completion": "trivial",
                    "transformation_kind": "none",
                    "proof_fingerprint": "proof-fingerprint",
                    "resolved_dependencies": [],
                    "verification": {
                        "status": "accepted",
                        "environment_id": "environment",
                        "method": "fixture",
                        "evidence_id": "fixture",
                    },
                    "source_declaration_name": "fixture",
                    "source_repository": "https://example.invalid/repository",
                    "source_revision": "revision",
                    "source_file": "Fixture.lean",
                }
            ],
            "topic_tags": ["prime-family:pnt-plus"],
            "memberships": [],
            "length": {
                "declaration_chars": 22,
                "proof_chars": 12,
                "completion_chars": 7,
                "declaration_lines": 1,
                "proof_lines": 2,
                "utf8_bytes": 34,
            },
            "derivation_family_fingerprint": "family-1",
            "generator_family": "direct-graph-logic-v2",
            "structural_class": "direct",
            "normalized_proof_dag": "dag",
            "source_lemma_ids": ["source-1"],
            "source_relation_edges": [],
            "shortcut_retrieval_ids": [],
            "shortcut_retrieval_index": [],
            "shortcut_checks": [],
        }
    )


def test_dataset_record_maps_to_target_only_generalist_variant() -> None:
    record = _record()

    assert dataset_record_preamble(record) == "import Mathlib"
    variants = generalist_variants(record)

    assert len(variants) == 1
    assert variants[0].statement_id == "statement-1"
    assert variants[0].proof_variant_id == "proof-1"
    assert variants[0].split == "train"
    assert variants[0].optimizer_eligible is True
    assert variants[0].source_kind == "synthetic"
    assert variants[0].domain_tags == ("prime-family:pnt-plus",)


def test_training_membership_rejects_duplicate_variants(tmp_path: Path) -> None:
    path = tmp_path / "membership.jsonl"
    path.write_text(
        '{"statement_id":"a","proof_variant_ids":["proof"]}\n'
        '{"statement_id":"b","proof_variant_ids":["proof"]}\n',
        encoding="utf-8",
    )

    try:
        read_training_membership(path)
    except ValueError as error:
        assert "repeats proof variants" in str(error)
    else:
        raise AssertionError("duplicate proof variant was accepted")
