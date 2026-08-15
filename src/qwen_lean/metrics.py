from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import fmean
from typing import Any, Iterable

from .schema import CandidateResult, RESULT_CATEGORIES


SUMMARY_SCHEMA_VERSION = "phase1-summary-v1"
INFRASTRUCTURE_ERROR_CATEGORIES = {"generation_error", "verifier_error"}


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n:
        raise ValueError(f"verified candidate count must be between 0 and {n}: {c}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}: {k}")
    failures = n - c
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(n, k)


def summarize_results(
    results: Iterable[CandidateResult],
    *,
    expected_task_ids: list[str],
    candidates_per_task: int,
    ks: tuple[int, ...] = (1, 4, 8),
) -> dict[str, Any]:
    materialized = list(results)
    by_task: dict[str, list[CandidateResult]] = defaultdict(list)
    for result in materialized:
        by_task[result.task_id].append(result)

    completeness_errors: list[str] = []
    expected_id_set = set(expected_task_ids)
    unexpected = sorted(set(by_task) - expected_id_set)
    missing = sorted(expected_id_set - set(by_task))
    if unexpected:
        completeness_errors.append(f"unexpected task ids: {unexpected}")
    if missing:
        completeness_errors.append(f"missing task ids: {missing}")

    per_task: list[dict[str, Any]] = []
    for task_id in expected_task_ids:
        task_results = by_task.get(task_id, [])
        indices = sorted(result.candidate_index for result in task_results)
        expected_indices = list(range(candidates_per_task))
        if indices != expected_indices:
            completeness_errors.append(
                f"{task_id}: expected candidate indices {expected_indices}, got {indices}"
            )
        per_task.append(
            {
                "task_id": task_id,
                "candidate_count": len(task_results),
                "verified_candidate_count": sum(
                    result.category == "verified" for result in task_results
                ),
            }
        )

    observed_category_counts = Counter(result.category for result in materialized)
    category_counts = {
        category: observed_category_counts[category]
        for category in sorted(RESULT_CATEGORIES)
    }
    infrastructure_error_count = sum(
        category_counts[category] for category in INFRASTRUCTURE_ERROR_CATEGORIES
    )
    if infrastructure_error_count:
        completeness_errors.append(
            f"infrastructure error candidates: {infrastructure_error_count}"
        )

    complete = not completeness_errors
    pass_metrics: dict[str, float] | None = None
    if complete:
        pass_metrics = {
            f"pass@{k}": fmean(
                pass_at_k(
                    candidates_per_task,
                    item["verified_candidate_count"],
                    k,
                )
                for item in per_task
            )
            for k in ks
        }

    candidate_count = len(materialized)
    tasks_with_success = sum(item["verified_candidate_count"] > 0 for item in per_task)
    observed_finish_reason_counts = Counter(
        result.finish_reason for result in materialized if result.finish_reason is not None
    )
    finish_reason_counts = {
        reason: observed_finish_reason_counts[reason]
        for reason in sorted({"eos", "token_limit"} | set(observed_finish_reason_counts))
    }

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "complete": complete,
        "completeness_errors": completeness_errors,
        "task_count": len(expected_task_ids),
        "candidate_count": candidate_count,
        "candidates_per_task": candidates_per_task,
        "tasks_with_verified_candidate": {
            "count": tasks_with_success,
            "fraction": tasks_with_success / len(expected_task_ids)
            if expected_task_ids
            else 0.0,
        },
        "pass_at_k": pass_metrics,
        "category_counts": category_counts,
        "category_fractions": {
            category: count / candidate_count if candidate_count else 0.0
            for category, count in category_counts.items()
        },
        "finish_reason_counts": finish_reason_counts,
        "verifier_timeout_count": category_counts["verifier_timeout"],
        "infrastructure_error_count": infrastructure_error_count,
        "timing_seconds": {
            "generation_candidate_latency_sum": _sum_optional(
                result.generation_latency_seconds for result in materialized
            ),
            "generation_candidate_latency_mean": _mean_optional(
                result.generation_latency_seconds for result in materialized
            ),
            "verification_latency_sum": _sum_optional(
                result.verification_latency_seconds for result in materialized
            ),
            "verification_latency_mean": _mean_optional(
                result.verification_latency_seconds for result in materialized
            ),
            "candidate_total_latency_sum": sum(
                result.total_latency_seconds for result in materialized
            ),
            "candidate_total_latency_mean": fmean(
                result.total_latency_seconds for result in materialized
            )
            if materialized
            else None,
        },
        "per_task": per_task,
    }


def _sum_optional(values: Iterable[float | None]) -> float:
    return sum(value for value in values if value is not None)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None
