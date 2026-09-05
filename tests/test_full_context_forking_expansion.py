from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.counterfactual_forking_assessment import (
    ParentTrajectory,
    fork_states,
)
from qwen_lean.full_context_forking_diagnostic import FullContextForkingConfig
from qwen_lean.full_context_forking_expansion import (
    EXPANSION_BRANCH_COUNT,
    EXPANSION_MANIFEST_SCHEMA,
    EXPANSION_PARENT_COUNT,
    EXPANSION_PHASE,
    build_expansion_manifest_payload,
    expansion_requests,
)
from qwen_lean.native_thinking_assessment import MathiaTask

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-full-context-forking.json"
CALIBRATION_PATH = (
    ROOT / "evidence/qwen35-full-context-forking/calibration-rtx4000.json"
)
DIAGNOSTIC_EVIDENCE_PATH = (
    ROOT / "evidence/qwen35-full-context-forking/results.json"
)


def _config() -> FullContextForkingConfig:
    return FullContextForkingConfig.load(CONFIG_PATH)


def _parent(ordinal: int) -> ParentTrajectory:
    task_id = f"frozen_task_{ordinal:02d}"
    candidate_id = f"native-thinking-parent-{ordinal:02d}"
    task = MathiaTask(
        task_id=task_id,
        workload="minif2f-valid-clean-v2",
        preamble="import Mathlib",
        declaration=f"theorem {task_id} : True",
        declaration_name=task_id,
        intuition="trivial",
        intuition_sha256=f"{ordinal:064x}",
        theorem_sha256=f"{ordinal + 100:064x}",
    )
    handoff = {
        "candidate_id": candidate_id,
        "candidate_index": 0,
        "seed": 0,
        "prompt_sha256": f"{ordinal + 200:064x}",
        "rendered_prompt_sha256": f"{ordinal + 300:064x}",
        "raw_generation_record_sha256": f"{ordinal + 400:064x}",
        "raw_response_sha256": f"{ordinal + 500:064x}",
        "raw_response_token_ids_sha256": f"{ordinal + 600:064x}",
        "raw_generation_jsonl_line_number": ordinal * 4 + 1,
        "raw_generation_artifact_path": "artifacts/frozen/generations.jsonl",
    }
    return ParentTrajectory(
        ordinal=ordinal,
        task=task,
        handoff=handoff,
        record={},
        record_sha256=handoff["raw_generation_record_sha256"],
        raw_response_token_ids=tuple(range(4096)),
        rendered_prompt_token_ids=tuple(range(200 + ordinal)),
        rendered_prompt_sha256=handoff["rendered_prompt_sha256"],
        states=fork_states(4096),
        parser_parity={"status": "passed"},
    )


def test_manifest_freezes_ordinals_one_through_twenty_nine_without_quality() -> None:
    config = _config()
    parents = [_parent(ordinal) for ordinal in range(1, 30)]
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    diagnostic = json.loads(
        DIAGNOSTIC_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    manifest = build_expansion_manifest_payload(
        config,
        parents,
        {
            "handoff_commit": config.reviewed_target["handoff_commit"],
            "handoff_manifest_sha256": "a" * 64,
        },
        calibration,
        CALIBRATION_PATH,
        diagnostic,
        DIAGNOSTIC_EVIDENCE_PATH,
    )
    assert manifest["schema_version"] == EXPANSION_MANIFEST_SCHEMA
    assert manifest["selection"]["selected_parent_count"] == (
        EXPANSION_PARENT_COUNT
    )
    assert [
        row["frozen_parent_ordinal"]
        for row in manifest["selection"]["ordered_parents"]
    ] == list(range(1, 30))
    assert manifest["request_plan"]["planned_branch_count"] == (
        EXPANSION_BRANCH_COUNT
    )
    assert manifest["request_plan"]["unfittable_request_count"] == 0
    forbidden = {"verifier_category", "finish_reason", "final_token_count"}
    assert all(
        forbidden.isdisjoint(row)
        for row in manifest["selection"]["ordered_parents"]
    )


def test_expansion_requests_are_1218_and_preserve_parent_state_seed_order(
    tmp_path: Path,
) -> None:
    config = _config()
    parents = [_parent(ordinal) for ordinal in range(1, 30)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    requests, unfittable = expansion_requests(
        config,
        parents,
        selected_context_length=262_144,
        calibration_evidence_path=CALIBRATION_PATH,
        manifest_path=manifest_path,
    )
    assert len(requests) == EXPANSION_BRANCH_COUNT
    assert unfittable == []
    assert len({request.branch_id for request in requests}) == len(requests)
    assert all(request.phase == EXPANSION_PHASE for request in requests)
    assert [
        (request.parent.ordinal, request.state.label, request.seed)
        for request in requests[:7]
    ] == [
        (1, "P0", 100),
        (1, "P0", 101),
        (1, "P0", 102),
        (1, "P0", 103),
        (1, "P0", 104),
        (1, "P0", 105),
        (1, "P15", 100),
    ]
    assert requests[42].parent.ordinal == 2
    assert requests[-1].parent.ordinal == 29
    assert requests[-1].state.label == "P90"
    assert requests[-1].seed == 105
    assert all(
        request.max_tokens
        == 262_144
        - len(request.parent.rendered_prompt_token_ids)
        - request.state.prefix_len
        for request in requests
    )


def test_manifest_rejects_non_frozen_parent_order() -> None:
    config = _config()
    parents = [_parent(ordinal) for ordinal in range(1, 30)]
    parents[0], parents[1] = parents[1], parents[0]
    with pytest.raises(ValueError, match="frozen ordinals"):
        build_expansion_manifest_payload(
            config,
            parents,
            {
                "handoff_commit": config.reviewed_target["handoff_commit"],
                "handoff_manifest_sha256": "a" * 64,
            },
            json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")),
            CALIBRATION_PATH,
            json.loads(
                DIAGNOSTIC_EVIDENCE_PATH.read_text(encoding="utf-8")
            ),
            DIAGNOSTIC_EVIDENCE_PATH,
        )
