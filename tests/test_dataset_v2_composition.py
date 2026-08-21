from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from qwen_lean.dataset_v2_composition import (
    AUDIT_PREFIX,
    CompositionAudit,
    CompositionSource,
    LeanCompositionRun,
    build_composition_plans,
    composition_imports,
    find_non_roundtripping_compositions,
    lean_name_key,
    parse_audits,
    records_from_compositions,
    render_constant_audit_source,
    render_constant_presence_source,
    render_composition_source,
    render_composition_roundtrip_source,
    render_persisted_synthetic_context_source,
    render_shortcut_gate_source,
    run_shortcut_gate_source,
    validate_composition_audits,
    verify_persisted_synthetic_contexts,
)


def _pool(domain: str, size: int = 16) -> list[CompositionSource]:
    return [
        CompositionSource(
            statement_id=f"statement-{domain}-{index}",
            declaration_name=f"Source.{domain}_{index}",
            source_module="Mathlib.Test",
            topic_tags=(f"prime-family:{domain}",),
            domain_family=domain,
            type_head="iff" if index % 2 == 0 else "other",
        )
        for index in range(size)
    ]


def test_composition_plans_are_unique_and_cover_structural_classes() -> None:
    plans = build_composition_plans(
        {"generic": _pool("generic")},
        {"generic": 12},
        seed="pilot",
    )

    assert len(plans) == 12
    assert len({tuple(sorted(item.statement_id for item in plan.source_lemmas)) for plan in plans}) == 12
    assert {plan.structural_class for plan in plans} == {"direct", "branching", "deep"}
    assert any(plan.generator_family.startswith("final-only:") for plan in plans)


def test_lean_name_key_matches_quoted_keyword_audit_output() -> None:
    assert lean_name_key("Sigma.«exists»") == "Sigma.exists"


def test_rendered_composition_uses_graph_grounded_iff_and_explicit_oracle() -> None:
    plans = build_composition_plans(
        {"pnt-plus": _pool("pnt")}, {"pnt-plus": 3}, seed="pilot"
    )
    source = render_composition_source(plans)

    assert "source_type% Source.pnt_" in source
    assert " ↔ " in source
    assert "Nonempty" not in source
    assert "exact ⟨fun _ =>" in source
    assert "exact ⟨fun _ => @Source.pnt_" in source
    assert "datasetV2Audit" in source
    assert all(plan.relation_edges for plan in plans)


def test_synthetic_record_persists_its_verified_import_context() -> None:
    generic = build_composition_plans(
        {"generic": _pool("generic")}, {"generic": 1}, seed="imports"
    )[0]
    audit = CompositionAudit(
        generic.synthetic_name,
        "True ∧ True",
        tuple(item.declaration_name for item in generic.source_lemmas),
        (),
    )
    verified = ("Mathlib.Extra", "Mathlib.Test")
    record = records_from_compositions(
        [generic],
        {generic.synthetic_name: audit},
        environment={
            "environment_id": "fixture-env",
            "lean_toolchain": "leanprover/lean4:v4.32.2",
            "mathlib_revision": "a" * 40,
        },
        verification_evidence_id="fixture-evidence",
        shortcut_status={},
        verified_imports={generic.synthetic_name: verified},
    )[0]

    assert record.environment.imports == verified
    assert "PrimeNumberTheoremAnd" not in record.environment.imports
    rendered = render_persisted_synthetic_context_source([record])
    assert rendered.startswith("import Mathlib.Extra\nimport Mathlib.Test\n")
    assert f"theorem {generic.synthetic_name} :" in rendered
    assert record.canonical_declaration in rendered
    assert "source_type%" not in rendered
    assert rendered.index(record.canonical_declaration) < rendered.index(
        "def datasetV2StoredAudit"
    )
    assert record.proof_variants[0].canonical_proof in rendered

    with pytest.raises(ValueError, match="omit source modules"):
        records_from_compositions(
            [generic],
            {generic.synthetic_name: audit},
            environment={
                "environment_id": "fixture-env",
                "lean_toolchain": "leanprover/lean4:v4.32.2",
                "mathlib_revision": "a" * 40,
            },
            verification_evidence_id="fixture-evidence",
            shortcut_status={},
            verified_imports={generic.synthetic_name: ("Mathlib.Extra",)},
        )


def test_pnt_composition_uses_pinned_umbrella_import() -> None:
    plan = build_composition_plans(
        {"pnt-plus": _pool("pnt-plus")}, {"pnt-plus": 1}, seed="pnt-imports"
    )[0]
    plan = replace(
        plan,
        source_lemmas=tuple(
            replace(source, source_module="PrimeNumberTheoremAnd.Test")
            for source in plan.source_lemmas
        ),
    )
    assert composition_imports([plan]) == ("PrimeNumberTheoremAnd",)


def test_persisted_context_gate_covers_every_record(tmp_path, monkeypatch) -> None:
    plans = build_composition_plans(
        {"generic": _pool("generic")}, {"generic": 2}, seed="context-gate"
    )
    audits = {
        plan.synthetic_name: CompositionAudit(
            plan.synthetic_name,
            "True ∧ True",
            tuple(item.declaration_name for item in plan.source_lemmas),
            (),
        )
        for plan in plans
    }
    imports = {
        plans[0].synthetic_name: ("Mathlib.Test",),
        plans[1].synthetic_name: ("Mathlib.Other", "Mathlib.Test"),
    }
    records = records_from_compositions(
        plans,
        audits,
        environment={
            "environment_id": "fixture-env",
            "lean_toolchain": "leanprover/lean4:v4.32.2",
            "mathlib_revision": "a" * 40,
        },
        verification_evidence_id="fixture-evidence",
        shortcut_status={},
        verified_imports=imports,
    )

    def accepted(source_path, **kwargs):
        names = re.findall(
            r"`(DatasetV2PersistedContext\.[^,\]]+)",
            source_path.read_text(encoding="utf-8"),
        )
        return LeanCompositionRun(
            "accepted",
            0,
            0.01,
            "",
            tuple(CompositionAudit(name, "True ∧ True", (), ()) for name in names),
        )

    monkeypatch.setattr(
        "qwen_lean.dataset_v2_composition.run_composition_source", accepted
    )

    evidence = verify_persisted_synthetic_contexts(
        records,
        output_dir=tmp_path,
        target_root=tmp_path,
        workers=2,
    )

    assert evidence["synthetic_records"] == 2
    assert evidence["accepted_records"] == 2
    assert evidence["context_groups"] == 2
    assert evidence["context_failures"] == 0


def test_persisted_context_gate_rejects_invalid_stored_declaration(
    tmp_path, monkeypatch
) -> None:
    plan = build_composition_plans(
        {"generic": _pool("generic")}, {"generic": 1}, seed="invalid-context"
    )[0]
    audit = CompositionAudit(
        plan.synthetic_name,
        "True ∧ True",
        tuple(item.declaration_name for item in plan.source_lemmas),
        (),
    )
    record = records_from_compositions(
        [plan],
        {plan.synthetic_name: audit},
        environment={
            "environment_id": "fixture-env",
            "lean_toolchain": "leanprover/lean4:v4.32.2",
            "mathlib_revision": "a" * 40,
        },
        verification_evidence_id="fixture-evidence",
        shortcut_status={},
    )[0]
    invalid = replace(
        record,
        canonical_declaration=(
            f"theorem {plan.synthetic_name} : DatasetV2DefinitelyMissing"
        ),
    )

    def rejected(source_path, **kwargs):
        source = source_path.read_text(encoding="utf-8")
        assert invalid.canonical_declaration in source
        assert "source_type%" not in source
        return LeanCompositionRun("rejected", 1, 0.01, "unknown identifier", ())

    monkeypatch.setattr(
        "qwen_lean.dataset_v2_composition.run_composition_source", rejected
    )

    with pytest.raises(RuntimeError, match="persisted synthetic context verification failed"):
        verify_persisted_synthetic_contexts(
            [invalid], output_dir=tmp_path, target_root=tmp_path
        )


def test_composition_roundtrip_gate_bisects_invalid_audited_statement(
    tmp_path, monkeypatch
) -> None:
    plans = build_composition_plans(
        {"generic": _pool("generic")}, {"generic": 3}, seed="roundtrip"
    )
    bad_name = plans[1].synthetic_name
    audits = {
        plan.synthetic_name: CompositionAudit(
            plan.synthetic_name,
            "DatasetV2DefinitelyMissing" if plan.synthetic_name == bad_name else "True",
            tuple(item.declaration_name for item in plan.source_lemmas),
            (),
        )
        for plan in plans
    }
    rendered = render_composition_roundtrip_source(plans, audits)
    assert "source_type%" not in rendered
    assert f"theorem {bad_name} : DatasetV2DefinitelyMissing" in rendered

    def compile_if_valid(source_path, **kwargs):
        source = source_path.read_text(encoding="utf-8")
        if "DatasetV2DefinitelyMissing" in source:
            return LeanCompositionRun("rejected", 1, 0.01, "unknown identifier", ())
        names = re.findall(
            r"`(DatasetV2CompositionRoundtrip\.[^,\]]+)", source
        )
        return LeanCompositionRun(
            "accepted",
            0,
            0.01,
            "",
            tuple(CompositionAudit(name, "", (), ()) for name in names),
        )

    monkeypatch.setattr(
        "qwen_lean.dataset_v2_composition.run_composition_source",
        compile_if_valid,
    )
    rejected = find_non_roundtripping_compositions(
        plans,
        audits,
        source_path=tmp_path / "Roundtrip.lean",
        target_root=tmp_path,
    )

    assert rejected == (bad_name,)


def test_constant_audit_imports_target_and_indexes_names() -> None:
    source = render_constant_audit_source(["One", "Namespace.Two"])
    assert source.startswith("import PrimeNumberTheoremAnd")
    assert "datasetV2Audit #[`One, `Namespace.Two]" in source


def test_constant_presence_audit_uses_exact_mathlib_modules() -> None:
    sources = _pool("generic")[:2]
    source = render_constant_presence_source(sources)
    assert source.startswith("import Mathlib.Test")
    assert "DATASET_V2_MISSING" in source


def test_constant_presence_audit_batches_large_name_arrays() -> None:
    source = render_constant_presence_source(_pool("generic", size=513))

    assert source.count("run_cmd datasetV2Presence") == 3


def test_rendered_composition_uses_audited_explicit_universes() -> None:
    plan = build_composition_plans(
        {"generic": _pool("generic")}, {"generic": 1}, seed="pilot"
    )[0]
    sources = tuple(
        replace(source, universe_arguments=("0", "0"))
        for source in plan.source_lemmas
    )
    source = render_composition_source([replace(plan, source_lemmas=sources)])

    assert ".{0, 0}" in source
    assert "`pp.all true" in source
    assert "`pp.privateNames false" in source
    assert 'map fun _ => "0"' in render_constant_presence_source(_pool("generic")[:1])


def test_audit_validation_requires_every_planned_source_in_elaborated_proof() -> None:
    plan = build_composition_plans(
        {"generic": _pool("generic")}, {"generic": 1}, seed="pilot"
    )[0]
    dependencies = [item.declaration_name for item in plan.source_lemmas]
    line = AUDIT_PREFIX + json.dumps(
        {
            "name": plan.synthetic_name,
            "type": "True ∧ True",
            "dependencies": dependencies,
            "level_parameters": [],
        }
    )
    audits = parse_audits(line)
    assert validate_composition_audits([plan], audits)["actual_dependency_failures"] == 0

    missing = CompositionAudit(plan.synthetic_name, "True ∧ True", (), ())
    try:
        validate_composition_audits([plan], [missing])
    except ValueError as error:
        assert "omitted source lemmas" in str(error)
    else:
        raise AssertionError("missing actual dependencies were accepted")


def test_shortcut_gate_checks_frozen_suite_and_indexed_sources() -> None:
    plan = build_composition_plans(
        {"generic": _pool("generic")}, {"generic": 1}, seed="pilot"
    )[0]
    audit = CompositionAudit(
        plan.synthetic_name,
        "True ∧ True",
        tuple(item.declaration_name for item in plan.source_lemmas),
        (),
    )
    rendered, line_map = render_shortcut_gate_source(
        [plan], {plan.synthetic_name: audit}
    )

    assert "fail_if_success assumption" in rendered
    assert "fail_if_success rfl" in rendered
    assert "fail_if_success (solve | simp)" in rendered
    assert "fail_if_success (exact Source.generic_" in rendered
    assert "fail_if_success (simpa using Source.generic_" in rendered
    assert set(line_map.values()) == {plan.synthetic_name}
    assert plan.retrieval_lemmas
    assert any("dependency-relevance-neighborhood" in origin for _, origin in plan.retrieval_index)
    assert any("type-head:iff" in origin for _, origin in plan.retrieval_index)
    assert any(
        source.declaration_name not in {
            item.declaration_name for item in plan.source_lemmas
        }
        and f"exact {source.declaration_name}" in rendered
        for source in plan.retrieval_lemmas
    )


def test_shortcut_gate_maps_lean_error_lines(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "Shortcut.lean"
    source_path.write_text("example : True := by trivial\n", encoding="utf-8")

    class Completed:
        returncode = 1
        stdout = ""
        stderr = f"{source_path.resolve()}:1:24: error: tactic succeeded\n"

    monkeypatch.setattr(
        "qwen_lean.dataset_v2_composition.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )
    result = run_shortcut_gate_source(
        source_path, target_root=tmp_path, line_to_name={1: "synthetic_0"}
    )
    assert result.status == "rejected-shortcuts"
    assert result.rejected_names == ("synthetic_0",)
