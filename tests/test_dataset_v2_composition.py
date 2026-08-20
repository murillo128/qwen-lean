from __future__ import annotations

import json

from qwen_lean.dataset_v2_composition import (
    AUDIT_PREFIX,
    CompositionAudit,
    CompositionSource,
    build_composition_plans,
    parse_audits,
    render_constant_audit_source,
    render_constant_presence_source,
    render_composition_source,
    render_shortcut_gate_source,
    run_shortcut_gate_source,
    validate_composition_audits,
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


def test_rendered_composition_uses_graph_grounded_iff_and_explicit_oracle() -> None:
    plans = build_composition_plans(
        {"pnt-plus": _pool("pnt")}, {"pnt-plus": 3}, seed="pilot"
    )
    source = render_composition_source(plans)

    assert "source_type% Source.pnt_" in source
    assert " ↔ " in source
    assert "Nonempty" not in source
    assert "exact ⟨fun _ =>" in source
    assert "by simpa only using (@Source.pnt_" in source
    assert "datasetV2Audit" in source
    assert all(plan.relation_edges for plan in plans)


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
