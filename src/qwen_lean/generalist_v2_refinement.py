from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .generalist_v2_dataset import sha256_file
from .generalist_v2_evaluation import (
    _compact_final_run,
    _load_canonical_subset,
    _ordered_ids_digest,
    _read_json,
    materialize_q0_workload,
)


ANALYSIS_WORKLOADS = (
    "minif2f-valid-clean-v2",
    "fresh-composition-valid-v2",
)
SOLVABILITY_PARTITIONS = ("robust", "search-sensitive", "lottery", "dead-zone")
FAILURE_CLASS_RULES = {
    "syntax-or-parser": "parser/syntax diagnostics",
    "unknown-or-mismatched-lemma": "unknown identifier/constant or invalid field",
    "type-or-elaboration": "type mismatch, failed synthesis, or elaboration diagnostics",
    "unsolved-goals": "Lean reports remaining or unexpectedly absent goals",
    "timeout": "candidate verification timeout",
    "other-lean-rejection": "other Lean rejection",
    "infrastructure-error": "non-semantic evaluator failure",
}


def _solvability_label(verified_count: int) -> str:
    if verified_count < 0 or verified_count > 64:
        raise ValueError("n=64 verified count is outside [0, 64]")
    if verified_count >= 16:
        return "robust"
    if verified_count >= 2:
        return "search-sensitive"
    if verified_count == 1:
        return "lottery"
    return "dead-zone"


def _classify_failure(row: dict[str, Any]) -> str:
    category = str(row.get("category", ""))
    if category == "verified":
        return "verified"
    if category == "verifier_timeout":
        return "timeout"
    if category in {"generation_error", "verifier_error"}:
        return "infrastructure-error"
    if category == "empty_candidate":
        return "syntax-or-parser"
    diagnostics = row.get("diagnostics") or {}
    text = "\n".join(
        str(diagnostics.get(key, "")) for key in ("stdout", "stderr")
    ).lower()
    if re.search(r"unexpected token|unexpected end|parser|syntax|invalid command", text):
        return "syntax-or-parser"
    if re.search(
        r"unknown identifier|unknown constant|invalid field|invalid projection",
        text,
    ):
        return "unknown-or-mismatched-lemma"
    if re.search(
        r"type mismatch|application type mismatch|failed to synthesize|"
        r"cannot synthesize|invalid argument name|elaboration",
        text,
    ):
        return "type-or-elaboration"
    if "unsolved goals" in text or "no goals to be solved" in text:
        return "unsolved-goals"
    return "other-lean-rejection"


def _read_raw_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("extended raw candidate row is not an object")
            rows.append(value)
    return rows


def _source_family(workload_id: str, task_id: str, metadata: dict[str, Any]) -> str:
    if workload_id == "fresh-composition-valid-v2":
        return str(metadata.get("generator_family") or "unknown-generator")
    lowered = task_id.lower()
    for prefix in (
        "mathd_numbertheory",
        "mathd_combinatorics",
        "mathd_geometry",
        "mathd_algebra",
        "induction",
        "numbertheory",
        "algebra",
        "amc12a",
        "aime",
        "imo",
    ):
        if lowered.startswith(prefix):
            return prefix
    return lowered.split("_", 1)[0]


def _broad_domains(
    task_id: str, declaration: str, metadata: dict[str, Any]
) -> list[str]:
    lowered_id = task_id.lower()
    domains: set[str] = set()
    if "numbertheory" in lowered_id:
        domains.add("number-theory")
    if "geometry" in lowered_id:
        domains.add("geometry")
    if "combinatorics" in lowered_id:
        domains.add("combinatorics")
    if lowered_id.startswith("induction"):
        domains.add("sequences-recurrences-induction")
    if "algebra" in lowered_id:
        domains.add("algebra")
    tags = " ".join(str(value).lower() for value in metadata.get("topic_tags", []))
    text = f"{declaration} {tags}"
    if "prime-number-theory" in tags or "prime-family:" in tags:
        domains.add("number-theory")
    if re.search(r"\bComplex\b|ℂ", text):
        domains.add("complex-numbers")
    if re.search(r"Nat\.Prime|Nat\.gcd|ModEq|\bdvd\b|∣|%", text):
        domains.add("number-theory")
    if re.search(r"SimpleGraph|Finset|Fintype\.card|Multiset", text):
        domains.add("combinatorics")
    if re.search(r"Euclidean|Affine|Geometry|angle|dist", text, re.IGNORECASE):
        domains.add("geometry")
    if re.search(r"sequence|recurr|\bNat\.rec\b|\bFunction\.iterate\b", text, re.I):
        domains.add("sequences-recurrences-induction")
    if re.search(r"(?<![-=])(?:<|>|≤|≥)(?!=)", declaration):
        domains.add("inequalities-order")
    if re.search(
        r"MeasureTheory|Topology|Filter|Continuous|Real\.(?:sqrt|log|sin|cos)|"
        r"deriv|integral",
        text,
        re.IGNORECASE,
    ):
        domains.add("analysis-topology")
    if re.search(r"CategoryTheory|Functor|Morphism", text):
        domains.add("category-theory")
    if re.search(
        r"Monoid|Group|Ring|Field|Module|LinearMap|Polynomial|Submodule|Submonoid",
        text,
    ):
        domains.add("algebra")
    if re.search(r"Set|Subtype|Iff|And|Or|Function", text):
        domains.add("logic-sets-functions")
    return sorted(domains) or ["other-or-generic"]


def _carriers(declaration: str) -> list[str]:
    patterns = (
        ("Nat", r"\bNat\b|ℕ"),
        ("Int", r"\bInt\b|ℤ"),
        ("Rat", r"\bRat\b|ℚ"),
        ("Real", r"\bReal\b|ℝ"),
        ("Complex", r"\bComplex\b|ℂ"),
        ("Finset", r"\bFinset\b"),
        ("Set", r"(?<!Fin)\bSet\b"),
        ("functions", r"→|\bFunction\b"),
        (
            "coercion-pattern",
            r"↑|\b(?:Nat|Int|Rat|Real)\.cast\b|\balgebraMap\b|"
            r"\b(?:norm_cast|exact_mod_cast|push_cast)\b",
        ),
    )
    values = [name for name, pattern in patterns if re.search(pattern, declaration)]
    return values or ["other"]


def _logical_shapes(task_id: str, declaration: str) -> list[str]:
    values: list[str] = []
    checks = (
        ("universal", r"∀|\bforall\b"),
        ("existential", r"∃"),
        ("negation", r"¬|\bNot\b"),
        ("iff", r"↔|\bIff\b"),
        ("equality", r"(?<![<>=!])=(?!=|>)"),
        ("uniqueness-or-minimality", r"∃!|IsLeast|IsGreatest|\bUnique\b"),
    )
    for name, pattern in checks:
        if re.search(pattern, declaration):
            values.append(name)
    if re.search(r"induct|recurr|a \(n -", f"{task_id} {declaration}", re.I):
        values.append("induction-or-recurrence")
    return values or ["other"]


def _oracle_properties(completion: str) -> dict[str, Any]:
    tactic_patterns = (
        ("induction", r"\binduction\b|\binduction'\b"),
        ("omega", r"\bomega\b"),
        ("linarith-or-nlinarith", r"\bn?linarith\b"),
        ("ring", r"\bring(?:_nf)?\b"),
        ("field-simp", r"\bfield_simp\b"),
        ("simp", r"\bsimp(?:_all)?\b|\bsimpa\b"),
        ("norm-num", r"\bnorm_num\b"),
        ("finset", r"\bFinset\b|\bfin_cases\b"),
        ("divisibility-or-modular", r"\bdvd\b|∣|\bmod_cases\b|\bomega\b"),
        ("rewrite-heavy", r"\brw\b|\brewrite\b"),
    )
    return {
        "available": True,
        "completion_chars": len(completion),
        "completion_lines": completion.count("\n") + 1,
        "tactic_or_lemma_families": [
            name for name, pattern in tactic_patterns if re.search(pattern, completion)
        ],
    }


def _task_properties(
    workload_id: str,
    tasks: Sequence[Any],
    metadata: dict[str, dict[str, Any]],
    records: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_metadata = metadata.get(task.id, {})
        record = records.get(task.id)
        oracle = (
            _oracle_properties(record.proof_variants[0].completion)
            if record is not None
            else {
                "available": False,
                "reason": "clean-miniF2F source contains sorry placeholders, not oracle proofs",
            }
        )
        output[task.id] = {
            "source_family": _source_family(workload_id, task.id, task_metadata),
            "broad_domain": _broad_domains(
                task.id, task.declaration, task_metadata
            ),
            "carriers": _carriers(task.declaration),
            "logical_shapes": _logical_shapes(task.id, task.declaration),
            "declaration_chars": len(task.declaration),
            "declaration_lines": task.declaration.count("\n") + 1,
            "named_hypothesis_count": len(
                re.findall(r"\bh[₀-₉0-9A-Za-z_']*\s*:", task.declaration)
            ),
            "binder_group_count": len(
                re.findall(r"[({\[][^)}\]]*:[^)}\]]*[)}\]]", task.declaration)
            ),
            "coercion_marker_count": len(
                re.findall(
                    r"↑|\b(?:Nat|Int|Rat|Real)\.cast\b|\balgebraMap\b|"
                    r"\b(?:norm_cast|exact_mod_cast|push_cast)\b",
                    task.declaration,
                )
            ),
            "oracle_proof": oracle,
        }
    return output


def _numeric_summary(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _density_uncertainty(
    success_counts: Sequence[int], *, candidates_per_task: int = 64
) -> dict[str, Any]:
    """Describe uncertainty across tasks without treating samples as independent."""

    task_count = len(success_counts)
    candidate_slots = task_count * candidates_per_task
    density = (
        0.0 if candidate_slots == 0 else sum(success_counts) / candidate_slots
    )
    if task_count < 2:
        standard_error = None
        interval = None
    else:
        task_densities = [value / candidates_per_task for value in success_counts]
        variance = sum((value - density) ** 2 for value in task_densities) / (
            task_count - 1
        )
        standard_error = math.sqrt(variance / task_count)
        radius = 1.959963984540054 * standard_error
        interval = [max(0.0, density - radius), min(1.0, density + radius)]
    return {
        "task_count": task_count,
        "candidate_slots": candidate_slots,
        "density": density,
        "task_cluster_standard_error": standard_error,
        "task_cluster_normal_95_interval": interval,
        "method": "normal interval over per-task candidate densities",
    }


def _aggregate_tasks(
    task_ids: Sequence[str],
    counts: dict[str, int],
    unique_counts: dict[str, int],
    raw_by_task: dict[str, list[dict[str, Any]]],
    properties: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    verified = sum(counts[task_id] for task_id in task_ids)
    unique = sum(unique_counts[task_id] for task_id in task_ids)
    candidate_slots = len(task_ids) * 64
    verified_uncertainty = _density_uncertainty(
        [counts[task_id] for task_id in task_ids]
    )
    unique_uncertainty = _density_uncertainty(
        [unique_counts[task_id] for task_id in task_ids]
    )
    failures = Counter(
        _classify_failure(row)
        for task_id in task_ids
        for row in raw_by_task[task_id]
        if str(row.get("category")) != "verified"
    )
    rejected = sum(failures.values())
    oracle_available = [
        properties[task_id]["oracle_proof"]
        for task_id in task_ids
        if properties[task_id]["oracle_proof"]["available"]
    ]
    return {
        "task_count": len(task_ids),
        "small_group_under_10_tasks": len(task_ids) < 10,
        "candidate_slots": candidate_slots,
        "verified_candidate_count": verified,
        "verified_candidate_density": 0.0 if not candidate_slots else verified / candidate_slots,
        "verified_candidate_density_uncertainty": verified_uncertainty,
        "unique_verified_proof_count": unique,
        "unique_verified_proof_density": 0.0 if not candidate_slots else unique / candidate_slots,
        "unique_verified_proof_density_uncertainty": unique_uncertainty,
        "verified_duplication_fraction": (
            None if verified == 0 else 1.0 - (unique / verified)
        ),
        "tasks_solved_within_64": sum(counts[task_id] > 0 for task_id in task_ids),
        "solvability_partition_counts": dict(
            sorted(Counter(_solvability_label(counts[item]) for item in task_ids).items())
        ),
        "failure_classes": {
            "rejected_candidate_count": rejected,
            "counts": dict(sorted(failures.items())),
            "fractions_of_rejected_candidates": {
                key: value / rejected for key, value in sorted(failures.items())
            } if rejected else {},
        },
        "complexity": {
            "declaration_chars": _numeric_summary(
                [properties[item]["declaration_chars"] for item in task_ids]
            ),
            "declaration_lines": _numeric_summary(
                [properties[item]["declaration_lines"] for item in task_ids]
            ),
            "named_hypothesis_count": _numeric_summary(
                [properties[item]["named_hypothesis_count"] for item in task_ids]
            ),
            "binder_group_count": _numeric_summary(
                [properties[item]["binder_group_count"] for item in task_ids]
            ),
            "coercion_marker_count": _numeric_summary(
                [properties[item]["coercion_marker_count"] for item in task_ids]
            ),
        },
        "oracle_proof_coverage": {
            "available_task_count": len(oracle_available),
            "unavailable_task_count": len(task_ids) - len(oracle_available),
            "completion_chars": _numeric_summary(
                [int(value["completion_chars"]) for value in oracle_available]
            ),
            "tactic_or_lemma_family_task_counts": dict(
                sorted(
                    Counter(
                        family
                        for value in oracle_available
                        for family in value["tactic_or_lemma_families"]
                    ).items()
                )
            ),
        },
    }


def _dimension_breakdown(
    dimension: str,
    task_ids: Sequence[str],
    counts: dict[str, int],
    unique_counts: dict[str, int],
    raw_by_task: dict[str, list[dict[str, Any]]],
    properties: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        value = properties[task_id][dimension]
        values = value if isinstance(value, list) else [value]
        for item in values:
            groups[str(item)].append(task_id)
    return {
        value: _aggregate_tasks(
            members, counts, unique_counts, raw_by_task, properties
        )
        for value, members in sorted(groups.items())
    }


def _representative_examples(
    task_ids: Iterable[str],
    counts: dict[str, int],
    unique_counts: dict[str, int],
    properties: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(task_ids, key=lambda item: (-counts[item], item))[:3]
    return [
        {
            "task_id": task_id,
            "verified_candidate_count": counts[task_id],
            "unique_verified_proof_count": unique_counts[task_id],
            "source_family": properties[task_id]["source_family"],
            "broad_domain": properties[task_id]["broad_domain"],
            "carriers": properties[task_id]["carriers"],
            "logical_shapes": properties[task_id]["logical_shapes"],
        }
        for task_id in ordered
    ]


def _overlap(
    reference: dict[str, int],
    candidate: dict[str, int],
    *,
    candidate_only_label: str,
    reference_only_label: str,
) -> dict[str, Any]:
    if list(reference) != list(candidate):
        raise ValueError("paired refinement task identities differ")
    buckets = {
        "both_solved": [],
        candidate_only_label: [],
        reference_only_label: [],
        "solved_by_neither": [],
    }
    for task_id in reference:
        left = reference[task_id] > 0
        right = candidate[task_id] > 0
        key = (
            "both_solved"
            if left and right
            else candidate_only_label
            if right
            else reference_only_label
            if left
            else "solved_by_neither"
        )
        buckets[key].append(task_id)
    return {
        "task_count": len(reference),
        "counts": {key: len(value) for key, value in buckets.items()},
        "task_ids": buckets,
    }


def _rank_density_groups(breakdowns: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = [
        {
            "dimension": dimension,
            "value": value,
            "task_count": group["task_count"],
            "verified_candidate_density": group["verified_candidate_density"],
            "verified_duplication_fraction": group["verified_duplication_fraction"],
        }
        for dimension, values in breakdowns.items()
        for value, group in values.items()
        if group["task_count"] >= 10
    ]
    return {
        "highest_density_groups": sorted(
            rows,
            key=lambda item: (
                -item["verified_candidate_density"],
                item["dimension"],
                item["value"],
            ),
        )[:8],
        "lowest_density_groups": sorted(
            rows,
            key=lambda item: (
                item["verified_candidate_density"],
                item["dimension"],
                item["value"],
            ),
        )[:8],
        "small_groups_excluded_from_ranking": True,
    }


def _screening_lane_counts(
    value: dict[str, Any], task_ids: list[str], *, lane_label: str
) -> dict[str, int]:
    counts = value.get("verified_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != len(task_ids)
        or value.get("task_count") != len(task_ids)
        or value.get("candidate_count") != len(task_ids) * 8
        or value.get("ordered_task_ids_sha256") != _ordered_ids_digest(task_ids)
    ):
        raise ValueError(f"refinement {lane_label} screening identity differs")
    per_task = value.get("per_task")
    if per_task is not None and (
        [str(item["task_id"]) for item in per_task] != task_ids
        or [int(item["verified_candidate_count"]) for item in per_task]
        != [int(item) for item in counts]
    ):
        raise ValueError(f"refinement {lane_label} per-task outcomes differ")
    return dict(zip(task_ids, (int(item) for item in counts), strict=True))


def compact_refinement_evidence(
    extended_path: Path,
    final_path: Path,
    q0_path: Path,
    selection_path: Path,
    deepseek_root: Path,
    extended_root: Path,
    package_root: Path,
    view_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Analyze Q4 validation capability gaps without changing frozen metrics."""

    extended = _read_json(extended_path)
    final = _read_json(final_path)
    q0 = _read_json(q0_path)
    selection = _read_json(selection_path)
    selected = str(extended.get("screening_selected_checkpoint", ""))
    adapter_hash = str(
        extended.get("evaluated_checkpoint", {}).get("adapter_model_sha256", "")
    )
    if (
        extended.get("schema_version") != "generalist-v2-extended-validation-v1"
        or extended.get("status")
        != "selected-checkpoint-extended-validation-complete"
        or selected not in {"Q1", "Q2", "Q3", "Q4"}
        or not adapter_hash
        or final.get("schema_version") != "generalist-v2-final-assessment-v1"
        or final.get("status") != "complete"
        or final.get("selected_checkpoint") != selected
        or final.get("selected_adapter_model_sha256") != adapter_hash
        or q0.get("schema_version") != "generalist-v2-q0-evidence-v1"
        or selection.get("schema_version")
        != "generalist-v2-checkpoint-selection-v1"
        or selection.get("status") != "frozen"
        or selection.get("selection", {}).get("selected_checkpoint") != selected
        or selection.get("selected_checkpoint", {}).get("adapter_model_sha256")
        != adapter_hash
    ):
        raise ValueError("refinement analysis requires complete frozen evidence")

    workload_evidence: dict[str, Any] = {}
    cross_workload_failure_counts: Counter[str] = Counter()
    ranked_groups: dict[str, Any] = {}
    for workload_id in ANALYSIS_WORKLOADS:
        compact = extended["evaluated_checkpoint"]["workloads"][workload_id]
        raw_path = extended_root / selected / workload_id / "raw-candidates.jsonl.gz"
        if sha256_file(raw_path) != compact["raw_candidate_evidence"]["sha256"]:
            raise ValueError(f"refinement raw evidence hash differs for {workload_id}")
        raw = _read_raw_candidates(raw_path)
        tasks, _, _, metadata = materialize_q0_workload(
            workload_id, package_root, view_dir
        )
        task_ids = [str(task.id) for task in tasks]
        task_id_set = set(task_ids)
        if len(raw) != len(task_ids) * 64:
            raise ValueError(f"refinement raw candidate count differs for {workload_id}")
        raw_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw:
            task_id = str(row.get("task_id", ""))
            if (
                row.get("schema_version") != "generalist-v2-extended-candidate-v1"
                or row.get("checkpoint_id") != selected
                or row.get("workload_id") != workload_id
                or row.get("adapter_model_sha256") != adapter_hash
                or task_id not in task_id_set
            ):
                raise ValueError(f"refinement raw candidate identity differs for {workload_id}")
            raw_by_task[task_id].append(row)
        if any(
            sorted(int(row["candidate_index"]) for row in raw_by_task[task_id])
            != list(range(64))
            for task_id in task_ids
        ):
            raise ValueError(f"refinement n=64 candidate indices differ for {workload_id}")
        counts = {
            task_id: sum(
                str(row.get("category")) == "verified"
                for row in raw_by_task[task_id]
            )
            for task_id in task_ids
        }
        unique_counts = {
            task_id: len(
                {
                    str(row["normalized_proof_sha256"])
                    for row in raw_by_task[task_id]
                    if row.get("category") == "verified"
                }
            )
            for task_id in task_ids
        }
        compact_per_task = compact["raw_candidate_evidence"]["per_task"]
        if [str(item["task_id"]) for item in compact_per_task] != task_ids or any(
            counts[task_id] != int(item["verified_candidate_count"])
            or unique_counts[task_id] != int(item["unique_verified_proof_count"])
            for task_id, item in zip(task_ids, compact_per_task, strict=True)
        ):
            raise ValueError(f"refinement raw aggregates differ for {workload_id}")

        records = (
            _load_canonical_subset(package_root, task_ids)
            if workload_id == "fresh-composition-valid-v2"
            else {}
        )
        properties = _task_properties(workload_id, tasks, metadata, records)
        partitions: dict[str, Any] = {}
        for partition in SOLVABILITY_PARTITIONS:
            members = [
                task_id
                for task_id in task_ids
                if _solvability_label(counts[task_id]) == partition
            ]
            partitions[partition] = {
                **_aggregate_tasks(
                    members, counts, unique_counts, raw_by_task, properties
                ),
                "representative_examples": _representative_examples(
                    members, counts, unique_counts, properties
                ),
            }
        breakdowns = {
            dimension: _dimension_breakdown(
                dimension,
                task_ids,
                counts,
                unique_counts,
                raw_by_task,
                properties,
            )
            for dimension in (
                "source_family",
                "broad_domain",
                "carriers",
                "logical_shapes",
            )
        }
        oracle_groups: dict[str, list[str]] = defaultdict(list)
        for task_id in task_ids:
            for family in properties[task_id]["oracle_proof"].get(
                "tactic_or_lemma_families", []
            ):
                oracle_groups[str(family)].append(task_id)
        breakdowns["oracle_tactic_or_lemma_families"] = {
            family: _aggregate_tasks(
                members, counts, unique_counts, raw_by_task, properties
            )
            for family, members in sorted(oracle_groups.items())
        }

        lane_counts = {
            "base": _screening_lane_counts(
                q0["workloads"][workload_id], task_ids, lane_label="Q0"
            ),
            "selected": _screening_lane_counts(
                selection["checkpoints"][selected]["workloads"][workload_id],
                task_ids,
                lane_label=selected,
            ),
        }
        q0_overlap = _overlap(
            lane_counts["base"],
            lane_counts["selected"],
            candidate_only_label="q4-only",
            reference_only_label="q0-only",
        )
        paired_n8_overlap = {"q0_vs_q4": q0_overlap}
        deepseek_only_summary: dict[str, Any] = {
            "available": False,
            "reason": (
                "no accepted same-task DeepSeek validation evidence existed before "
                "the issue amendment; additional validation generation is prohibited"
            ),
        }
        comparison_sources: dict[str, Any] = {
            "q0_evidence_sha256": sha256_file(q0_path),
            "selection_evidence_sha256": sha256_file(selection_path),
        }
        if workload_id == "minif2f-valid-clean-v2":
            deepseek = _compact_final_run(
                deepseek_root / "deepseek" / workload_id,
                model_label="deepseek",
                selected_checkpoint=selected,
                workload_id=workload_id,
                expected_task_count=len(task_ids),
            )
            if [str(item["task_id"]) for item in deepseek["per_task"]] != task_ids:
                raise ValueError(
                    f"refinement DeepSeek task identities differ for {workload_id}"
                )
            lane_counts["deepseek"] = {
                str(item["task_id"]): int(item["verified_candidate_count"])
                for item in deepseek["per_task"]
            }
            deepseek_overlap = _overlap(
                lane_counts["deepseek"],
                lane_counts["selected"],
                candidate_only_label="q4-only",
                reference_only_label="deepseek-only",
            )
            paired_n8_overlap["q4_vs_deepseek"] = deepseek_overlap
            deepseek_only = deepseek_overlap["task_ids"]["deepseek-only"]
            deepseek_only_properties = {
                dimension: _dimension_breakdown(
                    dimension,
                    deepseek_only,
                    counts,
                    unique_counts,
                    raw_by_task,
                    properties,
                )
                for dimension in (
                    "source_family",
                    "broad_domain",
                    "carriers",
                    "logical_shapes",
                )
            }
            deepseek_only_summary = {
                "available": True,
                **_aggregate_tasks(
                    deepseek_only, counts, unique_counts, raw_by_task, properties
                ),
                "q4_n64_partition_counts": dict(
                    sorted(
                        Counter(
                            _solvability_label(counts[item])
                            for item in deepseek_only
                        ).items()
                    )
                ),
                "q4_rescued_by_n64_after_missing_at_n8": sum(
                    counts[item] > 0 for item in deepseek_only
                ),
                "properties": deepseek_only_properties,
                "representative_examples": _representative_examples(
                    deepseek_only, counts, unique_counts, properties
                ),
            }
            comparison_sources["deepseek"] = {
                "generation_sha256": deepseek["generation_sha256"],
                "results_sha256": deepseek["results_sha256"],
                "ordered_task_ids_sha256": deepseek[
                    "ordered_task_ids_sha256"
                ],
            }
        workload_failures = Counter(
            _classify_failure(row)
            for row in raw
            if str(row.get("category")) != "verified"
        )
        cross_workload_failure_counts.update(workload_failures)
        ranked_groups[workload_id] = _rank_density_groups(breakdowns)
        workload_evidence[workload_id] = {
            "task_count": len(task_ids),
            "candidates_per_task": 64,
            "overall": _aggregate_tasks(
                task_ids, counts, unique_counts, raw_by_task, properties
            ),
            "raw_candidate_evidence": {
                "path_role": "retained-outside-git",
                "sha256": sha256_file(raw_path),
                "candidate_count": len(raw),
            },
            "exact_c_i_distribution": {
                str(value): count
                for value, count in sorted(Counter(counts.values()).items())
            },
            "per_task_empirical_solvability": [
                {
                    "task_id": task_id,
                    "verified_candidate_count_c_i": counts[task_id],
                    "unique_normalized_verified_proof_count": unique_counts[task_id],
                    "partition": _solvability_label(counts[task_id]),
                }
                for task_id in task_ids
            ],
            "partitions": partitions,
            "property_breakdowns": breakdowns,
            "complexity_by_partition": {
                key: value["complexity"] for key, value in partitions.items()
            },
            "oracle_proof_analysis_by_partition": {
                key: value["oracle_proof_coverage"]
                for key, value in partitions.items()
            },
            "paired_n8_overlap": {
                **paired_n8_overlap,
                "compute_contract": "same tasks, n=8, temperature 0.8, top-p 0.95, seed 0",
            },
            "deepseek_only_capability_gap": deepseek_only_summary,
            "comparison_sources": comparison_sources,
            "ranked_systematic_patterns": ranked_groups[workload_id],
        }

    evidence = {
        "schema_version": "generalist-v2-refinement-conclusions-v1",
        "status": "complete",
        "selected_checkpoint": selected,
        "selected_adapter_model_sha256": adapter_hash,
        "extended_validation_sha256": sha256_file(extended_path),
        "final_assessment_sha256": sha256_file(final_path),
        "q0_evidence_sha256": sha256_file(q0_path),
        "selection_evidence_sha256": sha256_file(selection_path),
        "analysis_scope": {
            "workloads": list(ANALYSIS_WORKLOADS),
            "validation_only": True,
            "test_workloads_inspected_for_refinement": False,
            "checkpoint_or_metrics_changed": False,
            "additional_training_authorized": False,
            "mathia_or_cross_model_intervention_executed": False,
            "riemann_evidence_used": False,
            "deepseek_validation_workloads_used": ["minif2f-valid-clean-v2"],
            "missing_deepseek_validation_lanes_regenerated": False,
        },
        "definitions": {
            "robust": "verified candidate count c_i >= 16 of 64",
            "search-sensitive": "2 <= c_i <= 15 of 64",
            "lottery": "c_i == 1 of 64",
            "dead-zone": "c_i == 0 of 64",
            "verified_candidate_density": "verified candidates / (tasks * 64)",
            "unique_verified_proof_density": "unique normalized verified proofs / (tasks * 64)",
            "verified_duplication_fraction": "1 - unique normalized verified proofs / verified candidates",
            "failure_classification": FAILURE_CLASS_RULES,
            "uncertainty_method": "normal 95% intervals over per-task candidate densities; candidate samples within a theorem are not treated as independent",
            "property_group_membership": "broad-domain, carrier, logical-shape, and oracle-tactic groups may overlap",
            "oracle_variant_policy": "the sole canonical validation proof variant is analyzed where available",
        },
        "workloads": workload_evidence,
        "conclusions": {
            "strengths_and_weak_spots": ranked_groups,
            "executor_or_knowledge_deficiency_signal": {
                "interpretation": "dead-zone tasks plus unknown-lemma, type/elaboration, and unsolved-goal diagnostics indicate skills or Lean knowledge not recovered by 64 samples",
                "dead_zone_task_counts": {
                    workload_id: value["partitions"]["dead-zone"]["task_count"]
                    for workload_id, value in workload_evidence.items()
                },
                "dead_zone_task_fractions": {
                    workload_id: (
                        value["partitions"]["dead-zone"]["task_count"]
                        / value["task_count"]
                    )
                    for workload_id, value in workload_evidence.items()
                },
                "cross_workload_failure_class_counts": dict(
                    sorted(cross_workload_failure_counts.items())
                ),
            },
            "search_or_diversity_deficiency_signal": {
                "interpretation": "search-sensitive/lottery task mass and high verified-proof duplication identify gains available from search or proof diversity without relabeling dead-zone tasks as solved",
                "partition_task_counts": {
                    workload_id: {
                        partition: value["task_count"]
                        for partition, value in evidence_workload["partitions"].items()
                    }
                    for workload_id, evidence_workload in workload_evidence.items()
                },
                "search_sensitive_or_lottery_task_counts": {
                    workload_id: (
                        value["partitions"]["search-sensitive"]["task_count"]
                        + value["partitions"]["lottery"]["task_count"]
                    )
                    for workload_id, value in workload_evidence.items()
                },
                "verified_duplication_fraction_by_workload": {
                    workload_id: value["overall"][
                        "verified_duplication_fraction"
                    ]
                    for workload_id, value in workload_evidence.items()
                },
            },
            "actionable_future_work": [
                "source new independent training theorems matching the weakest sufficiently sized domain/carrier/logical-shape groups; never add failed validation theorems themselves",
                "increase independently sourced proof-structure diversity for search-sensitive groups while preserving statement-normalized weighting",
                "target unknown-lemma and elaboration-heavy gaps with independently sourced verified premise-use examples",
                "track planner, Mathia-assisted rescue, and other cross-model interventions in separate issues",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence
