from __future__ import annotations

import json
from pathlib import Path

from qwen_lean.phase2_corpus import read_jsonl_records, statement_fingerprint
from qwen_lean.riemann_data import (
    RiemannAtlasConfig,
    RiemannDataConfig,
    build_atlas,
    build_internal_graph,
    build_specialist_partitions,
    extract_lean_declaration,
    materialize_riemann_data,
    validate_materialized_riemann_data,
)


def _record(
    index: int,
    *,
    split: str,
    file_path: str,
    name: str,
    premises: list[str] | None = None,
    component: str | None = None,
) -> dict:
    declaration = f"theorem {name} : {index} = {index}"
    return {
        "schema_version": "mathlib-whole-proof-v1",
        "id": f"record-{index:03d}",
        "source_repository": "https://github.com/leanprover-community/mathlib4",
        "source_revision": "fixture-revision",
        "file_path": file_path,
        "declaration_name": name,
        "declaration_kind": "theorem",
        "source_span": {"start": {"line": index + 1, "column": 1}, "end": {"line": index + 2, "column": 10}},
        "declaration_span": {"start": {"line": index + 1, "column": 1}, "end": {"line": index + 1, "column": len(declaration) + 1}},
        "proof_span": {"start": {"line": index + 1, "column": len(declaration) + 1}, "end": {"line": index + 2, "column": 10}},
        "declaration": declaration,
        "proof": "by\n  rfl",
        "completion": "rfl",
        "premises": premises or [],
        "file_group": file_path,
        "component_id": component or f"component-{index:03d}",
        "split": split,
        "statement_fingerprint": statement_fingerprint(declaration),
        "token_lengths": {"declaration": 5, "proof": 3, "completion": 1, "declaration_and_proof": 9, "declaration_and_completion": 7},
    }


def _config() -> RiemannDataConfig:
    return RiemannDataConfig(
        {
            "schema_version": "riemann-data-config-v1",
            "snapshot_date": "2026-08-17",
            "phase2_source": {
                "dataset_schema_version": "mathlib-whole-proof-v1",
                "repository": "https://github.com/leanprover-community/mathlib4",
                "revision": "fixture-revision",
                "lean_toolchain": "fixture-toolchain",
            },
            "graph": {
                "max_dependency_depth": 2,
                "max_neighbors_per_frontier": 3,
                "max_ring_nodes_per_direction": 20,
                "max_selected_proportion": 1.0,
                "source_neighbor_window": 1,
                "max_source_neighborhood_nodes": 20,
                "number_theory_control_prefix": "Mathlib/NumberTheory/",
                "seed_families": [
                    {
                        "id": "zeta",
                        "paths": ["Mathlib/NumberTheory/LSeries/RiemannZeta.lean"],
                        "declaration_patterns": ["Seed"],
                        "reason": "fixture seed",
                        "atlas_target": "riemann-zeta-function",
                    }
                ],
            },
            "corpora": {
                "number_theory_wide": "number-theory-wide-v1",
                "riemann_bubble": "riemann-bubble-v1",
                "riemann_core": "riemann-core-v1",
                "specialist_validation": "riemann-specialist-validation-v1",
                "riemann_near_holdout": "riemann-near-holdout-v1",
                "number_theory_far_holdout": "number-theory-far-holdout-v1",
                "external_lean": "riemann-external-lean-v1",
            },
            "validation": {"require_nonempty_corpora": True},
            "external_discovery": {
                "scope_note": "fixture",
                "projects": [
                    {
                        "repository": "https://example.test/pnt",
                        "revision": "fixture-external",
                        "license": "Apache-2.0",
                        "lean_toolchain": "fixture-lean",
                        "mathlib_revision": "fixture-mathlib",
                        "status": "accepted",
                        "require_git_revision": False,
                        "native_verification": {
                            "status": "passed",
                            "project_revision": "fixture-external",
                            "build_command": "lake build",
                            "build_jobs": 1,
                            "checked_on": "2026-08-17",
                            "axiom_audit_command": "lake env lean Audit.lean",
                        },
                        "selected_declarations": [
                            {
                                "file_path": "PNT.lean",
                                "local_name": "ExternalPNT",
                                "full_name": "ExternalPNT",
                                "axioms": ["propext", "Classical.choice", "Quot.sound"],
                                "topic_tags": ["PNT"],
                                "atlas_target": "prime-number-theorem",
                            }
                        ],
                    }
                ],
            },
        }
    )


def _atlas_config() -> RiemannAtlasConfig:
    return RiemannAtlasConfig(
        {
            "schema_version": "riemann-atlas-config-v1",
            "census_scope": "fixture",
            "sources": [
                {"id": "fixture-source", "title": "Fixture", "url": "https://example.test", "kind": "test"},
                {"id": "mathlib-phase2", "title": "Mathlib", "url": "https://example.test/mathlib", "kind": "formal-source"},
                {"id": "pnt-plus", "title": "PNT", "url": "https://example.test/pnt", "kind": "formal-source"},
            ],
            "entries": [
                {
                    "id": "riemann-hypothesis",
                    "canonical_name": "Riemann hypothesis",
                    "display_name": "Riemann hypothesis",
                    "aliases": ["RH"],
                    "statement": "All nontrivial zeros lie on the critical line.",
                    "authors": [],
                    "year": None,
                    "source_ids": ["fixture-source"],
                    "relationship_to_rh": {"class": "definition/foundation", "direction": "identity", "source_ids": ["fixture-source"]},
                    "prerequisites": [],
                    "topic_tags": ["RH"],
                    "formalization_status": "literature-only",
                    "formalization": None,
                    "verification_status": "open",
                    "candidate_role": "knowledge-only",
                    "kind": "conjecture",
                    "notes": "fixture",
                },
                {
                    "id": "riemann-zeta-function",
                    "canonical_name": "Riemann zeta function",
                    "display_name": "Riemann zeta function",
                    "aliases": [],
                    "statement": "Zeta foundation.",
                    "authors": [],
                    "year": None,
                    "source_ids": ["fixture-source"],
                    "relationship_to_rh": {"class": "prerequisite", "direction": "foundation", "source_ids": ["fixture-source"]},
                    "prerequisites": [],
                    "topic_tags": ["zeta"],
                    "formalization_status": "mathlib",
                    "formalization": {},
                    "verification_status": "verified",
                    "candidate_role": "training-source",
                    "kind": "theorem-family",
                    "notes": "fixture",
                },
                {
                    "id": "prime-number-theorem",
                    "canonical_name": "Prime number theorem",
                    "display_name": "Prime number theorem",
                    "aliases": ["PNT"],
                    "statement": "pi(x) is asymptotic to x/log x.",
                    "authors": [],
                    "year": None,
                    "source_ids": ["fixture-source"],
                    "relationship_to_rh": {"class": "partial-progress", "direction": "weaker", "source_ids": ["fixture-source"]},
                    "prerequisites": [],
                    "topic_tags": ["PNT"],
                    "formalization_status": "external-lean",
                    "formalization": {},
                    "verification_status": "verified",
                    "candidate_role": "training-source",
                    "kind": "theorem",
                    "notes": "fixture",
                },
            ],
            "relationships": [
                {"source": "riemann-zeta-function", "target": "riemann-hypothesis", "type": "prerequisite-for", "source_ids": ["fixture-source"]}
            ],
        }
    )


def _records() -> list[dict]:
    seed_file = "Mathlib/NumberTheory/LSeries/RiemannZeta.lean"
    return [
        _record(1, split="train", file_path=seed_file, name="Seed", premises=["PremiseOne"]),
        _record(2, split="train", file_path="Mathlib/Analysis/P1.lean", name="PremiseOne", premises=["PremiseTwo"]),
        _record(3, split="train", file_path="Mathlib/Analysis/P2.lean", name="PremiseTwo"),
        _record(4, split="train", file_path="Mathlib/Analysis/U1.lean", name="UserOne", premises=["Seed"]),
        _record(5, split="train", file_path="Mathlib/Analysis/U2.lean", name="UserTwo", premises=["UserOne"]),
        _record(6, split="train", file_path=seed_file, name="SourceNeighbor"),
        _record(7, split="train", file_path="Mathlib/NumberTheory/FarTrain.lean", name="FarTrain"),
        _record(8, split="validation", file_path="Mathlib/NumberTheory/Validation.lean", name="Validation"),
        _record(9, split="heldout", file_path="Mathlib/Analysis/Near.lean", name="Near", premises=["Seed"]),
        _record(10, split="heldout", file_path="Mathlib/NumberTheory/FarHeldout.lean", name="FarHeldout"),
    ]


def _write_phase2_fixture(path: Path, records: list[dict]) -> None:
    path.mkdir(parents=True)
    for split in ("train", "validation", "heldout"):
        with (path / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                if record["split"] == split:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_schema_version": "mathlib-whole-proof-v1",
                "source": {"repository": "https://github.com/leanprover-community/mathlib4", "revision": "fixture-revision", "lean_toolchain": "fixture-toolchain"},
                "counts": {"final_records": len(records)},
                "split_hygiene": {"cross_split_components": 0, "cross_split_statement_fingerprints": 0},
                "contamination": {"remaining_exact_statement_matches": 0},
            }
        ),
        encoding="utf-8",
    )


def test_graph_builds_deterministic_bounded_rings_and_separate_edge_types() -> None:
    records = _records()
    first = build_internal_graph(records, _config())
    second = build_internal_graph(records, _config())
    assert first["seed_manifest"] == second["seed_manifest"]
    assert first["edges"] == second["edges"]
    assert first["classes"]["record-001"] == "core"
    assert first["classes"]["record-002"] == "premise-1"
    assert first["classes"]["record-003"] == "premise-2"
    assert first["classes"]["record-004"] == "user-1"
    assert first["classes"]["record-005"] == "user-2"
    assert first["diagnostics"]["bounds"]["premise"]["max_depth"] == 2
    assert {edge[0] for edge in first["edges"]} == {"premise", "user", "source-neighborhood"}


def test_partitions_preserve_phase2_identity_and_no_forbidden_leakage() -> None:
    records = _records()
    graph = build_internal_graph(records, _config())
    partitions = build_specialist_partitions(records, graph, _config())
    memberships = partitions["memberships"]
    assert all(
        next(record for record in records if record["id"] == record_id)["split"] == "train"
        for name in ("number-theory-wide-v1", "riemann-bubble-v1", "riemann-core-v1")
        for record_id in memberships[name]
    )
    assert memberships["riemann-near-holdout-v1"] == {"record-009"}
    assert memberships["number-theory-far-holdout-v1"] == {"record-010"}
    assert all(
        value == 0
        for pair in partitions["diagnostics"]["forbidden_leakage"].values()
        for value in pair.values()
    )


def test_external_extraction_rejects_placeholders_and_preserves_source() -> None:
    source = "theorem ExternalPNT : True := by\n  trivial\n\ntheorem Next : True := by\n  trivial\n"
    value = extract_lean_declaration(source, "ExternalPNT")
    assert value["statement"] == "theorem ExternalPNT : True"
    assert value["proof"] == "by\n  trivial"
    assert "theorem Next" not in value["source_text"]

    bad = "theorem ExternalPNT : True := by\n  sorry\n"
    try:
        extract_lean_declaration(bad, "ExternalPNT")
    except ValueError as error:
        assert "placeholder" in str(error)
    else:
        raise AssertionError("placeholder proof was accepted")


def test_atlas_is_deterministic_and_math_edges_are_not_premise_edges() -> None:
    records = _records()
    graph = build_internal_graph(records, _config())
    external = [
        {
            "id": "external-id",
            "declaration_name": "ExternalPNT",
            "statement": "theorem ExternalPNT : True",
            "repository": "https://example.test/pnt",
            "revision": "fixture-external",
            "file_path": "PNT.lean",
            "topic_tags": ["PNT"],
            "atlas_target": "prime-number-theorem",
        }
    ]
    first = build_atlas(_atlas_config(), graph, records, external, _config())
    second = build_atlas(_atlas_config(), graph, records, external, _config())
    assert first["entries"] == second["entries"]
    assert first["relationships"] == second["relationships"]
    assert all(edge["type"] != "premise" for edge in first["relationships"])
    assert len({entry["id"] for entry in first["entries"]}) == len(first["entries"])


def test_materialization_is_reproducible_and_generic_phase2_loader_unchanged(tmp_path: Path) -> None:
    records = _records()
    phase2 = tmp_path / "phase2"
    external = tmp_path / "external"
    _write_phase2_fixture(phase2, records)
    external.mkdir()
    (external / "PNT.lean").write_text(
        "theorem ExternalPNT : True := by\n  trivial\n", encoding="utf-8"
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = materialize_riemann_data(
        phase2, first, _config(), _atlas_config(), external_root=external
    )
    second_summary = materialize_riemann_data(
        phase2, second, _config(), _atlas_config(), external_root=external
    )
    assert first_summary["complete"] is True
    assert second_summary["complete"] is True
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert validate_materialized_riemann_data(first)["complete"] is True

    loaded = read_jsonl_records(phase2 / "train.jsonl")
    assert [record.to_dict() for record in loaded] == [
        record for record in records if record["split"] == "train"
    ]

    external_record = json.loads(
        (first / "external/riemann-external-lean-v1.jsonl").read_text(encoding="utf-8")
    )
    assert external_record["native_verification"]["status"] == "verified"
    assert external_record["repository"] == "https://example.test/pnt"
