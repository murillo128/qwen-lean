"""Build a compact, read-only handoff for counterfactual trajectory forks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qwen_lean.native_thinking_assessment import (
    GENERATION_RECORD_SCHEMA,
    NativeThinkingConfig,
    generation_config_sha256,
    load_mathia_tasks,
)

HANDOFF_SCHEMA = "qwen35-counterfactual-forking-handoff-v1"
RAW_PAYLOAD_FIELDS = (
    "raw_response_token_ids",
    "raw_response_text",
    "reasoning_content",
    "final_content",
)
REQUIRED_MANIFEST_FIELDS = (
    "candidate_id",
    "workload",
    "task_id",
    "candidate_index",
    "seed",
    "prompt_sha256",
    "rendered_prompt_sha256",
    "raw_response_sha256",
    "raw_response_token_count",
    "reasoning_token_count",
    "final_token_count",
    "finish_reason",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return _sha256_text(payload)


def _read_generation_source(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.removesuffix("\n")
            if not raw_line:
                continue
            record = json.loads(raw_line)
            if record.get("schema_version") != GENERATION_RECORD_SCHEMA:
                raise ValueError(f"unknown generation schema on line {line_number}")
            candidate_id = str(record["candidate_id"])
            if candidate_id in seen:
                raise ValueError(f"duplicate generation candidate: {candidate_id}")
            seen.add(candidate_id)
            if _sha256_text(str(record["raw_response_text"])) != record.get(
                "raw_response_sha256"
            ):
                raise ValueError(f"raw response hash mismatch: {candidate_id}")
            token_ids = record.get("raw_response_token_ids")
            if not isinstance(token_ids, list) or len(token_ids) != int(
                record["raw_response_token_count"]
            ):
                raise ValueError(f"raw token ids/count mismatch: {candidate_id}")
            for field in RAW_PAYLOAD_FIELDS:
                if field not in record:
                    raise ValueError(f"generation lacks {field}: {candidate_id}")
            records.append(
                {
                    "record": record,
                    "jsonl_line_number": line_number,
                    "generation_record_sha256": _sha256_text(raw_line),
                }
            )
    return records


def _read_verifier_categories(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    categories: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            candidate_id = str(record["candidate_id"])
            if candidate_id in categories:
                raise ValueError(f"duplicate verification candidate: {candidate_id}")
            categories[candidate_id] = str(record["category"])
    return categories


def build_handoff_manifest(
    *,
    task_order: Sequence[tuple[str, str]],
    generation_source: Sequence[dict[str, Any]],
    artifact_location: str,
    task_count: int,
    candidates_per_task: int,
    verifier_categories: dict[str, str] | None = None,
    generation_config_sha256_value: str | None = None,
    mathia_freeze_id: str | None = None,
) -> dict[str, Any]:
    """Select the first fully covered T1 tasks in frozen task order."""
    categories = verifier_categories or {}
    expected_indices = set(range(candidates_per_task))
    by_task: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    t1_count = 0
    for located in generation_source:
        record = located["record"]
        if record.get("arm") != "t1":
            continue
        t1_count += 1
        key = (str(record["workload"]), str(record["task_id"]))
        index = int(record["candidate_index"])
        task_candidates = by_task.setdefault(key, {})
        if index in task_candidates:
            raise ValueError(f"duplicate T1 task/candidate index: {key} #{index}")
        task_candidates[index] = located

    selected_tasks = [
        key for key in task_order if set(by_task.get(key, {})) == expected_indices
    ][:task_count]
    if len(selected_tasks) != task_count:
        raise ValueError(
            f"need {task_count} fully covered T1 tasks; found {len(selected_tasks)}"
        )

    candidates: list[dict[str, Any]] = []
    for key in selected_tasks:
        for candidate_index in range(candidates_per_task):
            located = by_task[key][candidate_index]
            record = located["record"]
            missing = [
                field for field in REQUIRED_MANIFEST_FIELDS if field not in record
            ]
            if missing:
                raise ValueError(
                    f"generation lacks manifest fields {missing}: {record['candidate_id']}"
                )
            candidate = {field: record[field] for field in REQUIRED_MANIFEST_FIELDS}
            candidate.update(
                {
                    "raw_response_token_ids_sha256": _sha256_json(
                        record["raw_response_token_ids"]
                    ),
                    "raw_generation_artifact_path": artifact_location,
                    "raw_generation_jsonl_line_number": located["jsonl_line_number"],
                    "raw_generation_record_sha256": located["generation_record_sha256"],
                    "raw_generation_record_lookup": {
                        "field": "candidate_id",
                        "value": record["candidate_id"],
                    },
                    "verifier_category": categories.get(str(record["candidate_id"])),
                }
            )
            candidates.append(candidate)

    verified_count = sum(row["verifier_category"] is not None for row in candidates)
    return {
        "schema_version": HANDOFF_SCHEMA,
        "purpose": "non-invasive intermediate handoff for counterfactual forking",
        "selection": {
            "arm": "t1",
            "policy": "first_frozen_task_order_with_all_candidates_durable",
            "quality_independent": True,
            "task_count": task_count,
            "candidates_per_task": candidates_per_task,
            "candidate_count": len(candidates),
            "selected_tasks": [
                {"workload": workload, "task_id": task_id}
                for workload, task_id in selected_tasks
            ],
        },
        "source": {
            "durable_generation_artifact_path": artifact_location,
            "durable_t1_candidate_count_at_export": t1_count,
            "durable_t1_distinct_task_count_at_export": len(by_task),
            "generation_config_sha256": generation_config_sha256_value,
            "mathia_freeze_id": mathia_freeze_id,
            "record_location": "one-based JSONL line plus exact candidate_id lookup",
            "required_original_payload_fields": list(RAW_PAYLOAD_FIELDS),
            "token_branching_integrity": (
                "branch from raw_response_token_ids in the located original record; "
                "do not re-tokenize text"
            ),
        },
        "verification": {
            "status": "complete" if verified_count == len(candidates) else "partial",
            "verified_candidate_count": verified_count,
            "unverified_candidate_count": len(candidates) - verified_count,
            "selection_independent_of_verification": True,
        },
        "candidates": candidates,
    }


def write_handoff(
    *,
    config_path: Path,
    mathia_root: Path,
    generations_path: Path,
    artifact_location: str,
    output_path: Path,
    task_count: int = 30,
    verifications_path: Path | None = None,
) -> dict[str, Any]:
    config = NativeThinkingConfig.load(config_path)
    tasks, binding = load_mathia_tasks(config, mathia_root)
    generation_source = _read_generation_source(generations_path)
    manifest = build_handoff_manifest(
        task_order=[(task.workload, task.task_id) for task in tasks],
        generation_source=generation_source,
        artifact_location=artifact_location,
        task_count=task_count,
        candidates_per_task=int(config.sampling["candidates_per_task"]),
        verifier_categories=_read_verifier_categories(verifications_path),
        generation_config_sha256_value=generation_config_sha256(config),
        mathia_freeze_id=str(binding["freeze_id"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mathia-root", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--artifact-location", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=30)
    parser.add_argument("--verifications", type=Path)
    args = parser.parse_args()
    manifest = write_handoff(
        config_path=args.config,
        mathia_root=args.mathia_root,
        generations_path=args.generations,
        artifact_location=args.artifact_location,
        output_path=args.output,
        task_count=args.task_count,
        verifications_path=args.verifications,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_count": manifest["selection"]["task_count"],
                "candidate_count": manifest["selection"]["candidate_count"],
                "verification": manifest["verification"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
