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
    _source_position_verification_task,
    _synthetic_task,
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
