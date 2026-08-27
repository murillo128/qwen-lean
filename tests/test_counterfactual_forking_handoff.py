from __future__ import annotations

import hashlib
import json

import pytest

from qwen_lean.counterfactual_forking_handoff import build_handoff_manifest
from qwen_lean.native_thinking_assessment import GENERATION_RECORD_SCHEMA


def _located(task_id: str, index: int, line_number: int) -> dict[str, object]:
    token_ids = [line_number, index]
    raw_text = f"raw-{task_id}-{index}"
    record = {
        "schema_version": GENERATION_RECORD_SCHEMA,
        "arm": "t1",
        "candidate_id": f"candidate-{task_id}-{index}",
        "workload": "workload",
        "task_id": task_id,
        "candidate_index": index,
        "seed": index,
        "prompt_sha256": f"prompt-{task_id}",
        "rendered_prompt_sha256": f"rendered-{task_id}",
        "raw_response_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "raw_response_token_ids": token_ids,
        "raw_response_text": raw_text,
        "raw_response_token_count": len(token_ids),
        "reasoning_content": raw_text,
        "reasoning_token_count": len(token_ids),
        "final_content": None,
        "final_token_count": 0,
        "finish_reason": "token_limit",
    }
    return {
        "record": record,
        "jsonl_line_number": line_number,
        "generation_record_sha256": f"line-{line_number}",
    }


def test_selects_first_frozen_fully_covered_tasks_and_exact_token_locator() -> None:
    task_order = [("workload", task_id) for task_id in ("a", "b", "c")]
    source = [
        *[_located("b", index, 10 + index) for index in range(4)],
        *[_located("a", index, 20 + index) for index in range(4)],
        _located("c", 0, 30),
    ]
    manifest = build_handoff_manifest(
        task_order=task_order,
        generation_source=source,
        artifact_location="artifacts/run/generations.jsonl",
        task_count=2,
        candidates_per_task=4,
        verifier_categories={"candidate-a-0": "verified"},
    )

    assert [row["task_id"] for row in manifest["selection"]["selected_tasks"]] == [
        "a",
        "b",
    ]
    assert [row["candidate_id"] for row in manifest["candidates"][:4]] == [
        f"candidate-a-{index}" for index in range(4)
    ]
    first = manifest["candidates"][0]
    assert first["raw_generation_jsonl_line_number"] == 20
    assert first["raw_generation_record_lookup"] == {
        "field": "candidate_id",
        "value": "candidate-a-0",
    }
    assert (
        first["raw_response_token_ids_sha256"]
        == hashlib.sha256(
            json.dumps([20, 0], separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert first["verifier_category"] == "verified"
    assert manifest["verification"] == {
        "status": "partial",
        "verified_candidate_count": 1,
        "unverified_candidate_count": 7,
        "selection_independent_of_verification": True,
    }


def test_requires_full_candidate_coverage() -> None:
    source = [_located("a", index, index + 1) for index in range(3)]
    with pytest.raises(ValueError, match="need 1 fully covered"):
        build_handoff_manifest(
            task_order=[("workload", "a")],
            generation_source=source,
            artifact_location="generations.jsonl",
            task_count=1,
            candidates_per_task=4,
        )
