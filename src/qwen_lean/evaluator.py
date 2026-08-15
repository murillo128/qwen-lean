from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import write_artifacts
from .prompt import PROMPT_FORMAT_ID
from .schema import CandidateResult, RunMetadata, TaskRecord
from .verifier import LeanVerifier


LEAN_TOOLCHAIN = "leanprover/lean4:v4.32.0"
MATHLIB_REVISION = "v4.32.0"


@dataclass(frozen=True)
class ProvidedCandidate:
    task_id: str
    candidate_id: str
    candidate_text: str
    expected_category: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProvidedCandidate:
        return cls(
            task_id=str(value["task_id"]),
            candidate_id=str(value["candidate_id"]),
            candidate_text=str(value["candidate_text"]),
            expected_category=value.get("expected_category"),
        )


def load_fixture_set(
    path: Path,
) -> tuple[str, list[TaskRecord], list[ProvidedCandidate]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        str(payload["fixture_id"]),
        [TaskRecord.from_dict(item) for item in payload["tasks"]],
        [ProvidedCandidate.from_dict(item) for item in payload["candidates"]],
    )


def evaluate_provided_candidates(
    tasks: Iterable[TaskRecord],
    candidates: Iterable[ProvidedCandidate],
    verifier: LeanVerifier,
) -> list[CandidateResult]:
    tasks_by_id = {task.id: task for task in tasks}
    results: list[CandidateResult] = []
    for index, candidate in enumerate(candidates):
        started = time.perf_counter()
        try:
            task = tasks_by_id[candidate.task_id]
            outcome = verifier.verify(task, candidate.candidate_text)
            result = CandidateResult(
                task_id=candidate.task_id,
                candidate_id=candidate.candidate_id,
                candidate_index=index,
                candidate_text=candidate.candidate_text,
                category=outcome.category,
                lean_exit_code=outcome.lean_exit_code,
                diagnostics=outcome.diagnostics,
                generation_latency_seconds=None,
                verification_latency_seconds=outcome.latency_seconds,
                total_latency_seconds=time.perf_counter() - started,
            )
        except Exception as error:  # Keep later candidates isolated from one bad record.
            result = CandidateResult(
                task_id=candidate.task_id,
                candidate_id=candidate.candidate_id,
                candidate_index=index,
                candidate_text=candidate.candidate_text,
                category="verifier_error",
                lean_exit_code=None,
                diagnostics={"stdout": "", "stderr": f"{type(error).__name__}: {error}"},
                generation_latency_seconds=None,
                verification_latency_seconds=None,
                total_latency_seconds=time.perf_counter() - started,
            )
        results.append(result)
    return results


def run_fixture_evaluation(
    fixture_path: Path,
    output_dir: Path,
    project_root: Path,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[RunMetadata, list[CandidateResult], list[str]]:
    fixture_id, tasks, candidates = load_fixture_set(fixture_path)
    verifier = LeanVerifier(project_root, timeout_seconds=timeout_seconds)
    results = evaluate_provided_candidates(tasks, candidates, verifier)
    metadata = RunMetadata(
        candidate_source="fixture",
        task_source=fixture_id,
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=LEAN_TOOLCHAIN,
        mathlib_revision=MATHLIB_REVISION,
        verifier_timeout_seconds=timeout_seconds,
    )
    write_artifacts(output_dir, metadata, results)
    mismatches = [
        f"{candidate.candidate_id}: expected {candidate.expected_category}, got {result.category}"
        for candidate, result in zip(candidates, results, strict=True)
        if candidate.expected_category is not None
        and candidate.expected_category != result.category
    ]
    return metadata, results, mismatches
