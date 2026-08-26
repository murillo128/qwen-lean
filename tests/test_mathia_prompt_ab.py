from __future__ import annotations

import json
from pathlib import Path

import pytest

import qwen_lean.mathia_prompt_ab as prompt_ab
from qwen_lean.mathia_prompt_ab import (
    BoundTask,
    PromptABConfig,
    _exact_two_sided_mcnemar,
    _generation_shard_path,
    _sha256_text,
    _write_once_json,
    candidate_identity,
    inventory_generations,
    render_arm_prompt,
)
from qwen_lean.schema import TaskRecord


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config/mathia-prompt-ab.json"
MANIFEST_PATH = (
    REPOSITORY_ROOT / "evidence/mathia-prompt-ab/execution-manifest.json"
)


def _bound_task() -> BoundTask:
    return BoundTask(
        ordinal=0,
        workload_id="minif2f-valid-clean-v2",
        task=TaskRecord(
            id="example",
            preamble="import Mathlib",
            declaration="theorem example : True",
            declaration_name="example",
        ),
        intuition_id="intuition-example",
        intuition_text="Use the canonical inhabitant of True.",
        intuition_sha256=_sha256_text("Use the canonical inhabitant of True."),
        model_visible_theorem_sha256="theorem-hash",
        q0_verified_candidate_count=0,
        metadata={},
    )


def test_prompt_templates_are_exact_and_differ_only_by_frozen_instruction() -> None:
    config = PromptABConfig.load(CONFIG_PATH)
    bound = _bound_task()
    arm_a = render_arm_prompt(config, bound, "A")
    arm_b = render_arm_prompt(config, bound, "B")
    assert arm_a == (
        "import Mathlib\n\n"
        "/- Mathematical intuition:\n"
        "Use the canonical inhabitant of True.\n"
        "-/\n\n"
        "theorem example : True := by\n  "
    )
    instruction = (
        "Complete the Lean proof below.\n"
        "Use the mathematical intuition as high-level guidance for the proof.\n"
        "Return only Lean code continuing after `by`.\n"
        "Do not use `sorry` or `admit`."
    )
    assert arm_b == arm_a.replace(
        "/- Mathematical intuition:\n",
        f"/- {instruction}\n\nMathematical intuition:\n",
        1,
    )


def test_candidate_identity_is_stable_and_binds_every_scientific_field() -> None:
    fields = {
        "arm_id": "A",
        "workload_id": "minif2f-valid-clean-v2",
        "task_id": "example",
        "prompt_sha256": "prompt",
        "candidate_index": 0,
        "sampling_seed": 0,
        "model_revision": "revision",
        "generation_config_sha256": "generation",
    }
    first = candidate_identity(**fields)
    assert candidate_identity(**fields) == first
    for key, replacement in {
        "arm_id": "B",
        "workload_id": "fresh-composition-valid-v2",
        "task_id": "other",
        "prompt_sha256": "other-prompt",
        "candidate_index": 1,
        "sampling_seed": 1,
        "model_revision": "other-revision",
        "generation_config_sha256": "other-generation",
    }.items():
        assert candidate_identity(**{**fields, key: replacement}) != first


def test_config_rejects_prompt_wording_drift(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["arms"]["B"]["comment_prefix"] = value["arms"]["B"][
        "comment_prefix"
    ].replace("high-level guidance", "guidance")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="arm B prompt wording differs"):
        PromptABConfig.load(path)


def test_atomic_write_once_rejects_nonidentical_replacement(tmp_path: Path) -> None:
    path = tmp_path / "immutable.json"
    _write_once_json(path, {"value": 1})
    _write_once_json(path, {"value": 1})
    with pytest.raises(ValueError, match="immutable artifact differs"):
        _write_once_json(path, {"value": 2})


def test_generation_inventory_skips_one_complete_atomic_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prompt_ab, "EXPECTED_TASKS", 1)
    manifest_sha256 = "manifest"
    slots = {
        arm_id: [
            {
                "candidate_id": f"{arm_id}-{index}",
                "candidate_index": index,
                "sampling_seed": 0,
            }
            for index in range(8)
        ]
        for arm_id in ("A", "B")
    }
    task = {
        "ordinal": 0,
        "workload_id": "minif2f-valid-clean-v2",
        "task_id": "example",
        "prompt_sha256": {"A": "prompt-a", "B": "prompt-b"},
        "candidate_slots": slots,
    }
    manifest = {"tasks": [task]}
    candidates = [
        {
            "candidate_id": f"A-{index}",
            "candidate_index": index,
            "sampling_seed": 0,
            "raw_continuation": "trivial",
            "raw_continuation_sha256": _sha256_text("trivial"),
            "token_count": 1,
            "finish_reason": "eos",
            "generation_latency_seconds": 0.1,
            "generation_error": None,
        }
        for index in range(8)
    ]
    shard = {
        "schema_version": prompt_ab.GENERATION_SHARD_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "arm_id": "A",
        "workload_id": task["workload_id"],
        "task_id": task["task_id"],
        "task_ordinal": 0,
        "prompt_sha256": "prompt-a",
        "candidates": candidates,
    }
    _write_once_json(_generation_shard_path(tmp_path, "A", task), shard)
    inventory = inventory_generations(manifest, tmp_path, manifest_sha256)
    assert inventory["completed_candidate_count"] == 8
    assert inventory["completed_tasks_by_arm"] == {"A": [0], "B": []}
    assert inventory["generation_failure_count"] == 0
    assert set(inventory["candidates_by_id"]) == {f"A-{index}" for index in range(8)}


def test_exact_mcnemar_handles_ties_and_two_sided_tail() -> None:
    assert _exact_two_sided_mcnemar(0, 0) == 1.0
    assert _exact_two_sided_mcnemar(4, 0) == pytest.approx(0.125)
    assert _exact_two_sided_mcnemar(2, 2) == 1.0


def test_committed_execution_manifest_has_every_unique_candidate_slot() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == prompt_ab.MANIFEST_SCHEMA_VERSION
    assert manifest["task_count"] == 611
    assert manifest["task_counts"] == {
        "minif2f-valid-clean-v2": 223,
        "fresh-composition-valid-v2": 388,
    }
    assert manifest["prompt_integrity_gate"]["passed"] is True
    candidate_ids = [
        slot["candidate_id"]
        for task in manifest["tasks"]
        for arm_id in ("A", "B")
        for slot in task["candidate_slots"][arm_id]
    ]
    assert len(candidate_ids) == 9_776
    assert len(set(candidate_ids)) == 9_776
