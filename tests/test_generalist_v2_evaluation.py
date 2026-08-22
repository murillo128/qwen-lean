from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_generalist_v2_dataset import _record

from qwen_lean.generalist_v2 import GeneralistV2Config
from qwen_lean.generalist_v2_dataset import generalist_variants
from qwen_lean.generalist_v2_evaluation import (
    _checkpoint_adapter_spec,
    _compact_checkpoint_workload,
    _compact_extended_workload,
    _source_position_verification_task,
    _synthetic_task,
    compact_extended_validation_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_evaluation_uses_persisted_import_context() -> None:
    record = _record()
    task = _synthetic_task(record)

    assert task.id == record.statement_id
    assert task.preamble == "import Mathlib"
    assert task.declaration == record.canonical_declaration
    assert generalist_variants(record)[0].completion == "trivial"


def test_source_position_verifier_preserves_exact_source_prefix(
    tmp_path: Path,
) -> None:
    source = (
        "import Mathlib\n\n"
        "namespace Fixture\n\n"
        "variable (localContext : True)\n\n"
        "theorem fixture : True := by\n"
        "  exact localContext\n"
    )
    source_path = tmp_path / ".lake/packages/mathlib/Fixture.lean"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    value = _record().to_dict()
    value["provenance"] = "real-mathlib"
    value["derivation_family_fingerprint"] = None
    value["generator_family"] = None
    value["structural_class"] = None
    value["normalized_proof_dag"] = None
    value["source_lemma_ids"] = []
    value["environment"]["repository"] = (
        "https://github.com/leanprover-community/mathlib4"
    )
    value["environment"]["revision"] = "revision"
    value["environment"]["file_path"] = "Fixture.lean"
    value["environment"]["source_span"] = {
        "start": {"line": 7, "column": 1},
        "end": {"line": 8, "column": 21},
    }
    record = type(_record()).from_dict(value)

    task = _source_position_verification_task(record, tmp_path)

    assert task.preamble.endswith("variable (localContext : True)")
    assert "theorem fixture" not in task.preamble
    assert task.declaration == "theorem fixture : True"


def test_checkpoint_adapter_spec_requires_pinned_unmerged_identity(
    tmp_path: Path,
) -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-4B-Base",
                "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                "r": 16,
                "target_modules": config.lora["target_regex"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")

    adapter = _checkpoint_adapter_spec(config, "Q2", tmp_path)

    assert adapter.adapter_id == "qwen-lean-generalist-v2-q2"
    assert adapter.base_model_revision == config.model["model_revision"]
    with pytest.raises(ValueError, match="must be Q1-Q4"):
        _checkpoint_adapter_spec(config, "Q0", tmp_path)


def test_compact_checkpoint_workload_binds_adapter_and_task_order(
    tmp_path: Path,
) -> None:
    adapter = {
        "adapter_id": "qwen-lean-generalist-v2-q2",
        "adapter_rank": 16,
        "base_model_id": "Qwen/Qwen3.5-4B-Base",
        "base_model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "adapter_config_sha256": "config-hash",
        "adapter_model_sha256": "adapter-hash",
    }
    (tmp_path / "generation-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-checkpoint-generation-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "ordered_task_ids_sha256": "task-hash",
                "candidate_count": 8,
                "generation_error_count": 0,
                "adapter": adapter,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run.json").write_text(
        json.dumps({"selected_adapter_binding": adapter}), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-checkpoint-verification-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "task_count": 1,
                "candidate_count": 8,
                "complete": True,
                "infrastructure_error_count": 0,
                "tasks_with_verified_candidate": 1,
                "pass_at_k": {"pass@1": 0.5, "pass@4": 1.0, "pass@8": 1.0},
                "category_counts": {"verified": 4, "lean_rejected": 4},
                "finish_reason_counts": {"eos": 8},
                "exact_target_candidate_count": 1,
                "exact_target_task_count": 1,
                "per_task": [{"verified_candidate_count": 4}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "results.jsonl").write_text("result\n", encoding="utf-8")
    (tmp_path / "generations.jsonl").write_text("generation\n", encoding="utf-8")

    compact = _compact_checkpoint_workload(
        tmp_path, "Q2", "fixture", 1, "task-hash", "adapter-hash"
    )

    assert compact["verified_counts"] == [4]
    assert compact["adapter_model_sha256"] == "adapter-hash"
    with pytest.raises(ValueError, match="selection workload is incomplete"):
        _compact_checkpoint_workload(
            tmp_path, "Q2", "fixture", 1, "different-task-hash", "adapter-hash"
        )


def test_compact_extended_workload_reports_curve_and_marginal_gains(
    tmp_path: Path,
) -> None:
    adapter = {"adapter_model_sha256": "adapter-hash"}
    (tmp_path / "generation-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-extended-generation-v1",
                "evaluation_profile": "extended-search-budget-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "ordered_task_ids_sha256": "task-hash",
                "candidate_count": 64,
                "generation_error_count": 0,
                "adapter": adapter,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run.json").write_text(
        json.dumps({"selected_adapter_binding": adapter, "candidates_per_task": 64}),
        encoding="utf-8",
    )
    pass_at_k = {
        "pass@1": 0.1,
        "pass@2": 0.2,
        "pass@4": 0.3,
        "pass@8": 0.4,
        "pass@16": 0.55,
        "pass@32": 0.7,
        "pass@64": 1.0,
    }
    solved = {
        "solved@1": 0,
        "solved@2": 0,
        "solved@4": 0,
        "solved@8": 0,
        "solved@16": 1,
        "solved@32": 1,
        "solved@64": 1,
    }
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-extended-verification-v1",
                "evaluation_profile": "extended-search-budget-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "task_count": 1,
                "candidate_count": 64,
                "complete": True,
                "infrastructure_error_count": 0,
                "pass_at_k": pass_at_k,
                "tasks_solved_within_k": solved,
                "category_counts": {"verified": 2, "lean_rejected": 62},
                "finish_reason_counts": {"eos": 64},
                "per_task": [{"verified_candidate_count": 2}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "results.jsonl").write_text("result\n", encoding="utf-8")
    (tmp_path / "generations.jsonl").write_text("generation\n", encoding="utf-8")

    compact = _compact_extended_workload(
        tmp_path, "Q2", "fixture", 1, "task-hash", "adapter-hash"
    )

    assert compact["marginal_pass_at_k"] == pytest.approx(
        {"delta_8_to_16": 0.15, "delta_16_to_32": 0.15, "delta_32_to_64": 0.3}
    )
    assert compact["marginal_tasks_solved"] == {
        "delta_8_to_16": 1,
        "delta_16_to_32": 0,
        "delta_32_to_64": 0,
    }


def test_extended_validation_freeze_uses_only_screened_finalists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screening = {
        "schema_version": "generalist-v2-checkpoint-selection-v1",
        "status": "frozen",
        "selection": {
            "selected_checkpoint": "Q3",
            "strongest_runner_up": "Q2",
        },
        "checkpoints": {
            checkpoint: {
                "adapter_model_sha256": f"{checkpoint}-hash",
                "workloads": {
                    workload: {"ordered_task_ids_sha256": f"{workload}-hash"}
                    for workload in (
                        "fresh-composition-valid-v2",
                        "minif2f-valid-clean-v2",
                    )
                },
            }
            for checkpoint in ("Q1", "Q2", "Q3", "Q4")
        },
    }
    screening_path = tmp_path / "screening.json"
    screening_path.write_text(json.dumps(screening), encoding="utf-8")

    monkeypatch.setattr(
        "qwen_lean.generalist_v2_evaluation._compact_extended_workload",
        lambda *args, **kwargs: {"pass_at_k": {"pass@64": 0.5}},
    )
    output = tmp_path / "extended.json"
    evidence = compact_extended_validation_evidence(
        GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json"),
        screening_path,
        tmp_path / "runs",
        output,
        final_checkpoint="Q2",
        decision_rationale="Q2 has materially higher clean pass@64 coverage.",
    )

    assert list(evidence["controls_and_finalists"]) == ["Q0", "Q3", "Q2"]
    assert evidence["final_checkpoint"] == "Q2"
    assert evidence["extended_validation_refined_selection"] is True
    assert evidence["test_workloads_consulted_before_final_freeze"] is False
    with pytest.raises(ValueError, match="two screened finalists"):
        compact_extended_validation_evidence(
            GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json"),
            screening_path,
            tmp_path / "runs",
            output,
            final_checkpoint="Q1",
            decision_rationale="not a finalist",
        )
