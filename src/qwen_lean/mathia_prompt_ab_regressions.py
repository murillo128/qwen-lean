from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .dataset_v2 import sha256_file
from .mathia_prompt_ab import (
    EXPECTED_CANDIDATES_PER_TASK,
    EXPECTED_CANDIDATES_TOTAL,
    VERIFIER_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    WORKLOAD_IDS,
    BoundTask,
    PromptABConfig,
    _atomic_write,
    _canonical_json_bytes,
    _load_q0_reference,
    _sha256_json,
    _sha256_text,
    bind_tasks,
    inventory_generations,
    inventory_verifications,
    validate_execution_manifest,
    verifier_environment_identities,
)
from .schema import TaskRecord
from .verifier import LeanVerifier, VerificationOutcome


ANALYSIS_SCHEMA_VERSION = "mathia-q0-b-regression-analysis-v1"
RAW_B_SCHEMA_VERSION = "mathia-q0-b-regression-raw-b-candidate-v1"
RAW_Q0_SCHEMA_VERSION = "mathia-q0-b-regression-q0-verified-candidate-v1"
TRANSFORMED_SCHEMA_VERSION = "mathia-q0-b-regression-transformed-candidate-v1"
EXPECTED_REGRESSION_TASKS = 23
EXPECTED_REGRESSION_COUNTS = {
    "minif2f-valid-clean-v2": 17,
    "fresh-composition-valid-v2": 6,
}
RECOVERY_ARCHIVE_SHA256 = (
    "aeac05f215c9882456a712de341593a19a7a7253da7e4cebff64e015301d9182"
)

_LEADING_BY = re.compile(r"\A(?P<leading>[ \t\r\n]*)by(?=\s|$)")
_MARKDOWN_FENCE = re.compile(
    r"\A\s*```(?:lean|lean4)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_LEAN3_BEGIN_END = re.compile(
    r"\A(?P<leading>\s*)begin(?=\s)(?P<body>.*?)\bend(?P<trailing>\s*)\Z",
    re.DOTALL,
)
_NATURAL_LANGUAGE_PREFIX = re.compile(
    r"\A(?P<leading>[ \t]*)(?:Here(?:'s| is) (?:the )?(?:Lean )?proof|"
    r"The (?:Lean )?proof is|Proof|Answer):?[ \t]*\r?\n",
    re.IGNORECASE,
)
_NATURAL_LANGUAGE_SUFFIX = re.compile(
    r"\r?\n[ \t]*(?:This (?:completes|proves) the proof\.?|QED\.?|"
    r"Explanation:.*|Note:.*)[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)
_SORRY_ADMIT = re.compile(r"\b(?:sorry|admit)\b")
_THEOREM_REPETITION = re.compile(r"(?m)^\s*(?:theorem|lemma)\s+[A-Za-z0-9_'.]+")
_UNKNOWN_IDENTIFIER = re.compile(
    r"unknown(?:Identifier| constant| identifier| declaration)", re.IGNORECASE
)

_TACTIC_FAMILIES = (
    "aesop",
    "apply",
    "by_contra",
    "constructor",
    "decide",
    "exact",
    "field_simp",
    "fin_cases",
    "have",
    "induction",
    "interval_cases",
    "intro",
    "linarith",
    "native_decide",
    "nlinarith",
    "norm_num",
    "omega",
    "positivity",
    "ring",
    "ring_nf",
    "rw",
    "simp",
)


@dataclass(frozen=True)
class MechanicalVariant:
    transform_sequence: tuple[str, ...]
    transformed_text: str
    transformed_sha256: str


def _iter_jsonl_bytes(path: Path) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            yield line_number, line, value


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = _jsonl_payload(rows)
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def reconstruct_regression_tasks(
    manifest: Mapping[str, Any],
    verification_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        if int(task["q0_verified_candidate_count"]) <= 0:
            continue
        categories = [
            str(verification_by_id[str(candidate_id)]["category"])
            for candidate_id in task["candidate_slots"]["B"]
        ]
        if "verified" not in categories:
            regressions.append(dict(task))
    return regressions


def candidate_format_markers(raw_text: str) -> dict[str, bool]:
    return {
        "duplicated_by": _LEADING_BY.search(raw_text) is not None,
        "theorem_repetition": _THEOREM_REPETITION.search(raw_text) is not None,
        "markdown_fence": "```" in raw_text,
        "lean3_begin": re.match(r"\A\s*begin(?=\s)", raw_text) is not None,
        "natural_language_contamination": (
            _NATURAL_LANGUAGE_PREFIX.search(raw_text) is not None
            or _NATURAL_LANGUAGE_SUFFIX.search(raw_text) is not None
        ),
        "sorry_or_admit": _SORRY_ADMIT.search(raw_text) is not None,
    }


def _one_step_transforms(raw_text: str, declaration: str) -> list[tuple[str, str]]:
    transformed: list[tuple[str, str]] = []

    match = _LEADING_BY.search(raw_text)
    if match is not None:
        transformed.append(
            (
                "strip_leading_duplicated_by",
                match.group("leading") + raw_text[match.end() :],
            )
        )

    match = _MARKDOWN_FENCE.fullmatch(raw_text)
    if match is not None:
        transformed.append(("unwrap_markdown_fence", match.group("body")))

    stripped = raw_text.lstrip()
    repeated_prefix = f"{declaration} := by"
    if stripped.startswith(repeated_prefix):
        transformed.append(
            (
                "remove_exact_repeated_theorem_declaration",
                raw_text[: len(raw_text) - len(stripped)]
                + stripped[len(repeated_prefix) :],
            )
        )

    match = _LEAN3_BEGIN_END.fullmatch(raw_text)
    if match is not None:
        transformed.append(
            (
                "unwrap_lean3_begin_end",
                match.group("leading")
                + match.group("body")
                + match.group("trailing"),
            )
        )

    match = _NATURAL_LANGUAGE_PREFIX.search(raw_text)
    if match is not None:
        transformed.append(
            (
                "remove_whitelisted_natural_language_prefix",
                match.group("leading") + raw_text[match.end() :],
            )
        )

    match = _NATURAL_LANGUAGE_SUFFIX.search(raw_text)
    if match is not None:
        transformed.append(
            ("remove_whitelisted_natural_language_suffix", raw_text[: match.start()])
        )

    return [(name, text) for name, text in transformed if text != raw_text]


def mechanical_variants(raw_text: str, declaration: str) -> list[MechanicalVariant]:
    queue: deque[tuple[tuple[str, ...], str]] = deque([((), raw_text)])
    seen_text = {raw_text}
    variants: list[MechanicalVariant] = []
    while queue:
        sequence, current = queue.popleft()
        if len(sequence) >= 4:
            continue
        for transform_name, transformed_text in _one_step_transforms(
            current, declaration
        ):
            if transform_name in sequence or transformed_text in seen_text:
                continue
            seen_text.add(transformed_text)
            transformed_sequence = (*sequence, transform_name)
            variants.append(
                MechanicalVariant(
                    transform_sequence=transformed_sequence,
                    transformed_text=transformed_text,
                    transformed_sha256=_sha256_text(transformed_text),
                )
            )
            queue.append((transformed_sequence, transformed_text))
    return sorted(
        variants,
        key=lambda variant: (
            len(variant.transform_sequence),
            variant.transform_sequence,
            variant.transformed_sha256,
        ),
    )


def _tactic_families(text: str) -> set[str]:
    return {
        tactic
        for tactic in _TACTIC_FAMILIES
        if re.search(rf"\b{re.escape(tactic)}\b", text)
    }


def _load_q0_candidates(
    q0_root: Path,
    q0_evidence: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, dict[str, str]],
]:
    candidates: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for workload_id in WORKLOAD_IDS:
        workload_evidence = q0_evidence["workloads"][workload_id]
        workload_root = q0_root / workload_id
        generation_path = workload_root / "generations.jsonl"
        result_path = workload_root / "results.jsonl"
        expected_generation_sha = str(workload_evidence["generation_sha256"])
        expected_results_sha = str(workload_evidence["results_sha256"])
        if sha256_file(generation_path) != expected_generation_sha:
            raise ValueError(f"Q0 generation bytes differ for {workload_id}")
        if sha256_file(result_path) != expected_results_sha:
            raise ValueError(f"Q0 result bytes differ for {workload_id}")
        source_hashes[workload_id] = {
            "generations_sha256": expected_generation_sha,
            "results_sha256": expected_results_sha,
            "generation_metadata_sha256": sha256_file(
                workload_root / "generation-metadata.json"
            ),
            "run_sha256": sha256_file(workload_root / "run.json"),
            "summary_sha256": sha256_file(workload_root / "summary.json"),
        }
        for line_number, raw_line, generation in _iter_jsonl_bytes(generation_path):
            task = generation.get("task", {})
            key = (
                workload_id,
                str(task.get("id")),
                int(generation["candidate_index"]),
            )
            if key in candidates:
                raise ValueError(f"duplicate Q0 generation candidate: {key}")
            raw_text = generation.get("text")
            if not isinstance(raw_text, str):
                raise ValueError(f"Q0 raw continuation is not text: {key}")
            candidates[key] = {
                "generation": generation,
                "generation_line_number": line_number,
                "generation_row_sha256": hashlib.sha256(raw_line).hexdigest(),
                "raw_sha256": _sha256_text(raw_text),
            }
        result_count = 0
        for line_number, raw_line, result in _iter_jsonl_bytes(result_path):
            key = (
                workload_id,
                str(result["task_id"]),
                int(result["candidate_index"]),
            )
            candidate = candidates.get(key)
            if candidate is None or "result" in candidate:
                raise ValueError(f"Q0 result identity differs: {key}")
            generation = candidate["generation"]
            if (
                result.get("candidate_text") != generation.get("text")
                or result.get("finish_reason") != generation.get("finish_reason")
                or int(result.get("generated_token_count", -1))
                != int(generation.get("token_count", -2))
                or result.get("category")
                not in {
                    "verified",
                    "lean_rejected",
                    "verifier_timeout",
                    "verifier_error",
                }
            ):
                raise ValueError(f"Q0 generation/result bytes differ: {key}")
            candidate.update(
                {
                    "result": result,
                    "result_line_number": line_number,
                    "result_row_sha256": hashlib.sha256(raw_line).hexdigest(),
                }
            )
            result_count += 1
        expected_candidates = int(workload_evidence["candidate_count"])
        observed = [key for key in candidates if key[0] == workload_id]
        if len(observed) != expected_candidates or result_count != expected_candidates:
            raise ValueError(f"Q0 candidate count differs for {workload_id}")
    if any("result" not in candidate for candidate in candidates.values()):
        raise ValueError("Q0 candidate without authoritative classification")
    return candidates, source_hashes


def _raw_b_rows(
    regressions: Sequence[Mapping[str, Any]],
    generation_inventory: Mapping[str, Any],
    verification_inventory: Mapping[str, Any],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    shard_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in generation_inventory["shards"]
    }
    verification_hashes = dict(verification_inventory["file_hashes"])
    rows: list[dict[str, Any]] = []
    for task in regressions:
        for candidate_id_value in task["candidate_slots"]["B"]:
            candidate_id = str(candidate_id_value)
            generation = generation_inventory["candidates_by_id"][candidate_id]
            verification = verification_inventory["results_by_id"][candidate_id]
            raw_text = generation["raw_continuation"]
            if not isinstance(raw_text, str):
                raise ValueError(f"Arm-B raw continuation is not text: {candidate_id}")
            verification_path = str(
                (
                    Path("verifications")
                    / "B"
                    / str(task["workload_id"])
                    / f"{candidate_id}.json"
                )
            )
            rows.append(
                {
                    "schema_version": RAW_B_SCHEMA_VERSION,
                    "workload": task["workload_id"],
                    "task_id": task["task_id"],
                    "task_ordinal": task["ordinal"],
                    "candidate_index": generation["candidate_index"],
                    "candidate_id": candidate_id,
                    "raw_text": raw_text,
                    "raw_sha256": generation["raw_continuation_sha256"],
                    "finish_reason": generation["finish_reason"],
                    "generated_token_count": generation["token_count"],
                    "official_verifier_classification": verification["category"],
                    "official_lean_exit_code": verification["lean_exit_code"],
                    "official_verifier_environment_sha256": verification[
                        "verifier_environment_sha256"
                    ],
                    "official_verification_result_sha256": verification_hashes[
                        verification_path
                    ],
                    "source_generation_shard": generation["generation_shard"],
                    "source_generation_shard_sha256": shard_hashes[
                        generation["generation_shard"]
                    ],
                    "manifest_sha256": manifest_sha256,
                }
            )
    if len(rows) != EXPECTED_REGRESSION_TASKS * EXPECTED_CANDIDATES_PER_TASK:
        raise ValueError("regression Arm-B raw candidate count differs")
    if any(row["raw_sha256"] != _sha256_text(row["raw_text"]) for row in rows):
        raise ValueError("regression Arm-B raw text hash differs")
    return rows


def _raw_q0_rows(
    regressions: Sequence[Mapping[str, Any]],
    q0_candidates: Mapping[tuple[str, str, int], Mapping[str, Any]],
    q0_evidence_sha256: str,
    q0_source_hashes: Mapping[str, Mapping[str, str]],
    model_revision: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in regressions:
        workload_id = str(task["workload_id"])
        task_id = str(task["task_id"])
        selected = [
            candidate
            for (workload, observed_task, _), candidate in q0_candidates.items()
            if workload == workload_id
            and observed_task == task_id
            and candidate["result"]["category"] == "verified"
        ]
        selected.sort(key=lambda candidate: candidate["generation"]["candidate_index"])
        if len(selected) != int(task["q0_verified_candidate_count"]):
            raise ValueError(f"Q0 verified count differs for regression task: {task_id}")
        for candidate in selected:
            generation = candidate["generation"]
            result = candidate["result"]
            candidate_index = int(generation["candidate_index"])
            stable_identity = "q0-candidate-" + _sha256_json(
                {
                    "workload": workload_id,
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "model_revision": model_revision,
                    "generation_sha256": q0_source_hashes[workload_id][
                        "generations_sha256"
                    ],
                }
            )
            rows.append(
                {
                    "schema_version": RAW_Q0_SCHEMA_VERSION,
                    "workload": workload_id,
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "candidate_id": stable_identity,
                    "source_candidate_id": result["candidate_id"],
                    "source_model_revision": model_revision,
                    "raw_text": generation["text"],
                    "raw_sha256": candidate["raw_sha256"],
                    "finish_reason": generation["finish_reason"],
                    "generated_token_count": generation["token_count"],
                    "authoritative_q0_classification": result["category"],
                    "authoritative_lean_exit_code": result["lean_exit_code"],
                    "authoritative_diagnostics": result["diagnostics"],
                    "task": generation["task"],
                    "source_generation_file_sha256": q0_source_hashes[workload_id][
                        "generations_sha256"
                    ],
                    "source_result_file_sha256": q0_source_hashes[workload_id][
                        "results_sha256"
                    ],
                    "source_generation_line_number": candidate[
                        "generation_line_number"
                    ],
                    "source_generation_row_sha256": candidate[
                        "generation_row_sha256"
                    ],
                    "source_result_line_number": candidate["result_line_number"],
                    "source_result_row_sha256": candidate["result_row_sha256"],
                    "q0_evidence_sha256": q0_evidence_sha256,
                    "q0_raw_recovery_archive_sha256": RECOVERY_ARCHIVE_SHA256,
                }
            )
    if any(row["raw_sha256"] != _sha256_text(row["raw_text"]) for row in rows):
        raise ValueError("regression Q0 raw text hash differs")
    return rows


def _verify_variants(
    variant_jobs: Sequence[dict[str, Any]],
    tasks_by_id: Mapping[tuple[str, str], BoundTask],
    config: PromptABConfig,
    lean_project_roots: Mapping[str, Path],
    *,
    workers: int,
) -> dict[str, VerificationOutcome]:
    verifiers = {
        workload_id: LeanVerifier(
            lean_project_roots[workload_id],
            timeout_seconds=float(config.verifier["timeout_seconds"]),
        )
        for workload_id in WORKLOAD_IDS
    }
    variant_task_keys = {
        (str(job["workload"]), str(job["task_id"])) for job in variant_jobs
    }
    for workload_id in WORKLOAD_IDS:
        preambles = {
            bound.task.preamble
            for key, bound in tasks_by_id.items()
            if key in variant_task_keys and key[0] == workload_id
        }
        for preamble in sorted(preambles):
            failure = verifiers[workload_id].prime_preamble(
                preamble,
                timeout_seconds=VERIFIER_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
            )
            if failure is not None:
                raise RuntimeError(
                    f"regression verifier environment probe failed: {workload_id}: "
                    f"{failure.category}"
                )

    outcomes: dict[str, VerificationOutcome] = {}

    def verify(job: Mapping[str, Any]) -> tuple[str, VerificationOutcome]:
        key = (str(job["workload"]), str(job["task_id"]))
        return str(job["variant_id"]), verifiers[key[0]].verify(
            tasks_by_id[key].task, str(job["transformed_text"])
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verify, job) for job in variant_jobs]
        for future in as_completed(futures):
            variant_id, outcome = future.result()
            outcomes[variant_id] = outcome
    return outcomes


def _build_variant_jobs(
    regressions: Sequence[Mapping[str, Any]],
    raw_b_rows: Sequence[Mapping[str, Any]],
    tasks_by_id: Mapping[tuple[str, str], BoundTask],
) -> list[dict[str, Any]]:
    raw_by_id = {str(row["candidate_id"]): row for row in raw_b_rows}
    jobs: list[dict[str, Any]] = []
    for task in regressions:
        key = (str(task["workload_id"]), str(task["task_id"]))
        declaration = tasks_by_id[key].task.declaration
        for candidate_id_value in task["candidate_slots"]["B"]:
            candidate_id = str(candidate_id_value)
            raw = raw_by_id[candidate_id]
            for variant in mechanical_variants(str(raw["raw_text"]), declaration):
                variant_id = "mechanical-variant-" + _sha256_json(
                    {
                        "source_raw_sha256": raw["raw_sha256"],
                        "transform_sequence": variant.transform_sequence,
                        "transformed_sha256": variant.transformed_sha256,
                    }
                )
                jobs.append(
                    {
                        "variant_id": variant_id,
                        "workload": key[0],
                        "task_id": key[1],
                        "candidate_id": candidate_id,
                        "candidate_index": raw["candidate_index"],
                        "source_raw_sha256": raw["raw_sha256"],
                        "transform_sequence": list(variant.transform_sequence),
                        "transformed_text": variant.transformed_text,
                        "transformed_sha256": variant.transformed_sha256,
                    }
                )
    jobs.sort(
        key=lambda job: (
            next(
                int(task["ordinal"])
                for task in regressions
                if task["task_id"] == job["task_id"]
                and task["workload_id"] == job["workload"]
            ),
            int(job["candidate_index"]),
            tuple(job["transform_sequence"]),
            str(job["variant_id"]),
        )
    )
    return jobs


def _diagnostic_tags(
    raw_rows: Sequence[Mapping[str, Any]],
    official_results: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    intuition_text: str,
) -> tuple[list[str], dict[str, Any]]:
    q0_family_sets = [
        _tactic_families(str(row["raw_text"])) for row in q0_rows
    ]
    q0_families = set().union(*q0_family_sets)
    b_family_sets = [_tactic_families(str(row["raw_text"])) for row in raw_rows]
    b_families = set().union(*b_family_sets)
    intuition_families = _tactic_families(intuition_text)
    diagnostic_texts = [
        str(result.get("diagnostics", {}).get("stdout", ""))
        + str(result.get("diagnostics", {}).get("stderr", ""))
        for result in official_results
    ]
    unknown_identifier_candidate_count = sum(
        _UNKNOWN_IDENTIFIER.search(text) is not None for text in diagnostic_texts
    )
    tags: list[str] = []
    if any(row["finish_reason"] == "token_limit" for row in raw_rows):
        tags.append("incomplete_or_token_limited")
    if unknown_identifier_candidate_count:
        tags.append("hallucinated_lemma_or_api")
    if q0_families and b_families:
        overlap = len(q0_families & b_families) / len(q0_families)
        if overlap < 0.5:
            tags.append("different_proof_family_from_q0")
    shortest_q0_index = min(
        range(len(q0_rows)),
        key=lambda index: (
            int(q0_rows[index]["generated_token_count"]),
            int(q0_rows[index]["candidate_index"]),
        ),
    )
    shortest_q0_tokens = int(
        q0_rows[shortest_q0_index]["generated_token_count"]
    )
    shortest_q0_families = q0_family_sets[shortest_q0_index]
    if (
        shortest_q0_tokens <= 64
        and shortest_q0_families
        and not any(
            shortest_q0_families <= b_families
            for b_families in b_family_sets
        )
    ):
        tags.append("lost_simple_q0_strategy")
    if intuition_families:
        if intuition_families & b_families:
            tags.append("guidance_followed_but_formalization_failed")
        else:
            tags.append("guidance_seems_ignored")
    if not tags:
        tags.append("cannot_determine")
    features = {
        "q0_tactic_families": sorted(q0_families),
        "q0_tactic_families_by_verified_candidate": [
            sorted(families) for families in q0_family_sets
        ],
        "b_tactic_families": sorted(b_families),
        "intuition_explicit_tactic_families": sorted(intuition_families),
        "shortest_q0_verified_generated_tokens": shortest_q0_tokens,
        "shortest_q0_verified_tactic_families": sorted(shortest_q0_families),
        "q0_b_tactic_family_overlap_fraction": (
            len(q0_families & b_families) / len(q0_families)
            if q0_families
            else None
        ),
        "official_unknown_identifier_candidate_count": (
            unknown_identifier_candidate_count
        ),
    }
    return tags, features


def _task_analysis(
    regressions: Sequence[Mapping[str, Any]],
    raw_b_rows: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    transformed_rows: Sequence[Mapping[str, Any]],
    tasks_by_id: Mapping[tuple[str, str], BoundTask],
    verification_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_b_by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    q0_by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    transformed_by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for rows, target in (
        (raw_b_rows, raw_b_by_task),
        (q0_rows, q0_by_task),
        (transformed_rows, transformed_by_task),
    ):
        for row in rows:
            target.setdefault((str(row["workload"]), str(row["task_id"])), []).append(
                row
            )
    analyses: list[dict[str, Any]] = []
    for task in regressions:
        key = (str(task["workload_id"]), str(task["task_id"]))
        bound = tasks_by_id[key]
        raw_rows = raw_b_by_task[key]
        q0_task_rows = q0_by_task[key]
        variants = transformed_by_task.get(key, [])
        recoveries = [
            row for row in variants if row["diagnostic_verifier_classification"] == "verified"
        ]
        inconclusive = [
            row
            for row in variants
            if row["diagnostic_verifier_classification"]
            in {"verifier_timeout", "verifier_error"}
        ]
        if recoveries:
            classification = "FORMAT_ONLY"
        elif inconclusive:
            classification = "INCONCLUSIVE"
        else:
            classification = "CONTENT_OR_SEARCH"
        official_results = [
            verification_by_id[str(candidate_id)]
            for candidate_id in task["candidate_slots"]["B"]
        ]
        tags: list[str] = []
        features: dict[str, Any] = {}
        if classification == "CONTENT_OR_SEARCH":
            tags, features = _diagnostic_tags(
                raw_rows,
                official_results,
                q0_task_rows,
                bound.intuition_text,
            )
        category_counts = Counter(str(row["official_verifier_classification"]) for row in raw_rows)
        finish_counts = Counter(str(row["finish_reason"]) for row in raw_rows)
        if classification == "FORMAT_ONLY":
            note = (
                f"{len(recoveries)} of 8 raw B candidates become Lean-valid only after "
                "a recorded superficial wrapper transform; official raw classifications "
                "remain unchanged."
            )
        elif classification == "INCONCLUSIVE":
            note = (
                f"No transformed candidate verified, but {len(inconclusive)} strict "
                "mechanical variant result is infrastructurally inconclusive."
            )
        else:
            observations: list[str] = []
            if "incomplete_or_token_limited" in tags:
                observations.append(
                    f"{finish_counts['token_limit']}/8 candidates hit the token limit"
                )
            if "hallucinated_lemma_or_api" in tags:
                observations.append(
                    f"{features['official_unknown_identifier_candidate_count']}/8 "
                    "official diagnostics report an unknown identifier/API"
                )
            if "different_proof_family_from_q0" in tags:
                observations.append(
                    "the union of B tactic families covers only "
                    f"{features['q0_b_tactic_family_overlap_fraction']:.0%} of the "
                    "verified-Q0 families"
                )
            if "lost_simple_q0_strategy" in tags:
                observations.append(
                    f"the shortest Q0 proof uses {features['shortest_q0_verified_generated_tokens']} "
                    "tokens and its complete tactic-family set is absent from every B "
                    "candidate"
                )
            if "guidance_followed_but_formalization_failed" in tags:
                observations.append(
                    "B attempts the tactic family named explicitly by the intuition, "
                    "but every attempt is Lean-rejected"
                )
            if "guidance_seems_ignored" in tags:
                observations.append(
                    "the tactic family named explicitly by the intuition is absent from B"
                )
            if "cannot_determine" in tags:
                observations.append(
                    "no narrower deterministic observable tag is supportable"
                )
            note = (
                f"All 8 raw B candidates and all {len(variants)} applicable strict "
                "wrapper variants fail under the frozen verifier; "
                + "; ".join(observations)
                + "."
            )
        analyses.append(
            {
                "ordinal": task["ordinal"],
                "workload": key[0],
                "task_id": key[1],
                "declaration_name": task["declaration_name"],
                "public_context": bound.task.preamble,
                "public_context_sha256": task["public_context_sha256"],
                "declaration": bound.task.declaration,
                "declaration_sha256": task["declaration_sha256"],
                "frozen_mathia_intuition": bound.intuition_text,
                "frozen_mathia_intuition_sha256": bound.intuition_sha256,
                "q0_verified_candidate_count": len(q0_task_rows),
                "q0_verified_candidate_raw_sha256": [
                    row["raw_sha256"] for row in q0_task_rows
                ],
                "b_candidate_result_distribution": dict(sorted(category_counts.items())),
                "b_finish_reason_distribution": dict(sorted(finish_counts.items())),
                "b_token_limit_candidate_count": finish_counts["token_limit"],
                "b_candidate_markers": [
                    {
                        "candidate_index": row["candidate_index"],
                        "candidate_id": row["candidate_id"],
                        "raw_sha256": row["raw_sha256"],
                        **candidate_format_markers(str(row["raw_text"])),
                    }
                    for row in raw_rows
                ],
                "mechanical_variants_tested": len(variants),
                "recoveries": [
                    {
                        key: row[key]
                        for key in (
                            "candidate_index",
                            "candidate_id",
                            "source_raw_sha256",
                            "transform_sequence",
                            "transformed_sha256",
                            "diagnostic_verifier_classification",
                            "diagnostic_lean_exit_code",
                            "verifier_environment_sha256",
                        )
                    }
                    for row in recoveries
                ],
                "primary_classification": classification,
                "diagnostic_tags": tags,
                "observable_comparison_features": features,
                "note": note,
            }
        )
    return analyses


def _aggregate(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classification_counts = Counter(
        str(row["primary_classification"]) for row in task_rows
    )
    recoveries = [recovery for row in task_rows for recovery in row["recoveries"]]
    recoverable_candidates = {
        (str(row["workload"]), str(row["task_id"]), str(recovery["candidate_id"]))
        for row in task_rows
        for recovery in row["recoveries"]
    }
    format_types = Counter(
        transform
        for recovery in recoveries
        for transform in recovery["transform_sequence"]
    )
    tag_counts = Counter(
        tag
        for row in task_rows
        if row["primary_classification"] == "CONTENT_OR_SEARCH"
        for tag in row["diagnostic_tags"]
    )
    total = len(task_rows)
    format_only = classification_counts["FORMAT_ONLY"]
    content = classification_counts["CONTENT_OR_SEARCH"]
    lost_simple_examples = [
        str(row["task_id"])
        for row in task_rows
        if "lost_simple_q0_strategy" in row["diagnostic_tags"]
    ]
    translation_examples = [
        str(row["task_id"])
        for row in task_rows
        if "guidance_followed_but_formalization_failed" in row["diagnostic_tags"]
    ]
    return {
        "total_regressions": total,
        "workload_counts": dict(sorted(Counter(str(row["workload"]) for row in task_rows).items())),
        "classification_counts": {
            key: classification_counts[key]
            for key in ("FORMAT_ONLY", "CONTENT_OR_SEARCH", "INCONCLUSIVE")
        },
        "classification_fractions": {
            key: classification_counts[key] / total
            for key in ("FORMAT_ONLY", "CONTENT_OR_SEARCH", "INCONCLUSIVE")
        },
        "tasks_with_mechanically_recoverable_b_proof": sum(
            bool(row["recoveries"]) for row in task_rows
        ),
        "mechanically_recoverable_candidate_proofs": len(recoverable_candidates),
        "verified_mechanical_variants": len(recoveries),
        "format_error_type_counts_among_recoveries": dict(sorted(format_types.items())),
        "content_or_search_diagnostic_tag_counts": dict(sorted(tag_counts.items())),
        "training_questions": {
            "1_output_protocol_attribution": (
                f"Confirmed FORMAT_ONLY failures account for {format_only}/{total} "
                f"tasks ({format_only / total:.1%})."
            ),
            "2_residual_search_or_formalization": (
                f"After removing confirmed format-only cases, {content}/{total} "
                f"tasks ({content / total:.1%}) remain CONTENT_OR_SEARCH; causal harm "
                "from guidance is not established by stochastic n=8 samples."
            ),
            "3_training_priority": (
                "Prioritize a combination: output-protocol SFT addresses the confirmed "
                "serialization failures, while the larger residual motivates proof-search "
                "and theorem+guidance conditioning work. This retrospective cannot choose "
                "between guidance gating and planner training."
            ),
            "4_guidance_away_from_simple_q0_examples": lost_simple_examples,
            "4_guidance_away_interpretation": (
                "These are observable short-Q0/B tactic-family divergences only; the "
                "stochastic comparison cannot establish that Mathia guidance caused them."
            ),
            "5_guidance_useful_but_lean_translation_failed_examples": translation_examples,
            "5_guidance_translation_interpretation": (
                "In these tasks B visibly attempts a tactic family named by the frozen "
                "intuition, but all eight candidates still fail Lean."
            ),
        },
    }


def render_regression_analysis(analysis: Mapping[str, Any]) -> str:
    aggregate = analysis["aggregate"]
    counts = aggregate["classification_counts"]
    bindings = analysis["source_bindings"]
    lines = [
        "# Q0-pass / Arm-B-fail regression retrospective",
        "",
        "**OBSERVED, scoring-excluded:** This diagnostic reuses frozen #78 Q0 and "
        "#86 Arm-B candidates. It does not regenerate, repair for scoring, or modify "
        "any official classification or metric.",
        "",
        f"The exact regression set contains {aggregate['total_regressions']} tasks: "
        f"{aggregate['workload_counts']['minif2f-valid-clean-v2']} MiniF2F and "
        f"{aggregate['workload_counts']['fresh-composition-valid-v2']} fresh composition.",
        "",
        "| primary classification | tasks | fraction |",
        "| --- | ---: | ---: |",
    ]
    for classification in ("FORMAT_ONLY", "CONTENT_OR_SEARCH", "INCONCLUSIVE"):
        lines.append(
            f"| {classification} | {counts[classification]} | "
            f"{aggregate['classification_fractions'][classification]:.1%} |"
        )
    lines.extend(
        [
            "",
            "Mechanically recoverable: "
            f"{aggregate['tasks_with_mechanically_recoverable_b_proof']} tasks and "
            f"{aggregate['mechanically_recoverable_candidate_proofs']} unique candidate "
            f"proofs ({aggregate['verified_mechanical_variants']} verified transform "
            "variants).",
            "",
            "## Method and provenance",
            "",
            "The regression membership is reconstructed from the frozen manifest and "
            "all official Arm-B verification records. Only the six declared superficial "
            "wrapper removals (and deterministic compositions of at most four removals) "
            "are tested. A task is `FORMAT_ONLY` only when Lean accepts at least one "
            "transformed candidate in the exact workload environment.",
            "",
            f"- execution manifest SHA-256: `{bindings['manifest_sha256']}`",
            f"- #86 verification result-set SHA-256: "
            f"`{bindings['issue_86_verification_result_set_sha256']}`",
            f"- Q0 compact evidence SHA-256: `{bindings['q0_evidence_sha256']}`",
            f"- Q0 recovery archive SHA-256: "
            f"`{bindings['q0_raw_recovery_archive_sha256']}`",
            f"- verifier environment-set SHA-256: "
            f"`{bindings['verifier_environment_set_sha256']}`",
            "",
            "## Per-task evidence",
            "",
            "| workload | task | Q0 verified | B finish eos/token limit | variants | classification | tags |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in analysis["tasks"]:
        finish = row["b_finish_reason_distribution"]
        lines.append(
            f"| {row['workload']} | `{row['task_id']}` | "
            f"{row['q0_verified_candidate_count']} | {finish.get('eos', 0)}/"
            f"{finish.get('token_limit', 0)} | {row['mechanical_variants_tested']} | "
            f"{row['primary_classification']} | "
            f"{', '.join(row['diagnostic_tags']) or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Training questions",
            "",
            f"1. {aggregate['training_questions']['1_output_protocol_attribution']}",
            f"2. {aggregate['training_questions']['2_residual_search_or_formalization']}",
            f"3. {aggregate['training_questions']['3_training_priority']}",
            "4. Observable short-Q0/B tactic-family divergences: "
            + (", ".join(
                f"`{task_id}`"
                for task_id in aggregate["training_questions"][
                    "4_guidance_away_from_simple_q0_examples"
                ]
            ) or "none under the deterministic tag rule")
            + ". "
            + aggregate["training_questions"]["4_guidance_away_interpretation"],
            "5. Cases where an explicit intuition tactic family appears "
            "in B but formalization still fails: "
            + (", ".join(
                f"`{task_id}`"
                for task_id in aggregate["training_questions"][
                    "5_guidance_useful_but_lean_translation_failed_examples"
                ]
            ) or "none under the deterministic tag rule")
            + ". "
            + aggregate["training_questions"][
                "5_guidance_translation_interpretation"
            ],
            "",
            "The Q0-pass/B-fail selection compares two stochastic n=8 samples from "
            "different prompt-conditioned distributions. FORMAT_ONLY is directly "
            "confirmed by Lean; CONTENT_OR_SEARCH establishes an observable distribution "
            "change after strict wrapper transforms, not causal harm from the intuition.",
            "",
            "## Permanent raw evidence boundary",
            "",
            "`q0-b-regressions/raw-b-candidates.jsonl` contains exactly 184 untouched "
            "Arm-B continuations. `q0-b-regressions/q0-verified-candidates.jsonl` "
            f"contains all {analysis['committed_evidence']['q0_verified_candidates']['row_count']} "
            "authoritative verified Q0 continuations for the same tasks. "
            "All transformed candidates are separate in "
            "`q0-b-regressions/transformed-b-candidates.jsonl` and reference their "
            "source raw SHA-256.",
            "",
            "A fresh checkout can audit the committed subset with "
            "`uv run pytest -q tests/test_mathia_prompt_ab_regressions.py`. Recomputing "
            "the Lean diagnostic additionally requires the frozen #86 artifact root, "
            "both frozen Lean projects, and the hash-matching Q0 recovery archive; the "
            "complete CLI is available through "
            "`python -m qwen_lean.mathia_prompt_ab_regressions --help`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_regression_analysis(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
    manifest_path: Path,
    artifact_root: Path,
    q0_root: Path,
    lean_project_roots: Mapping[str, Path],
    raw_b_output: Path,
    q0_verified_output: Path,
    transformed_output: Path,
    analysis_output: Path,
    readme_output: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    manifest, bound_tasks = validate_execution_manifest(
        config, dataset_root, mathia_root, repository_root, manifest_path
    )
    manifest_sha256 = sha256_file(manifest_path)
    generation_inventory = inventory_generations(
        manifest, artifact_root, manifest_sha256
    )
    if generation_inventory["completed_candidate_count"] != EXPECTED_CANDIDATES_TOTAL:
        raise ValueError("#86 generation inventory is incomplete")
    environment_bundle = verifier_environment_identities(config, lean_project_roots)
    verification_inventory = inventory_verifications(
        manifest,
        artifact_root,
        manifest_sha256,
        environment_bundle["environment_sha256_by_workload"],
        generation_inventory,
    )
    if verification_inventory["completed_verification_count"] != EXPECTED_CANDIDATES_TOTAL:
        raise ValueError("#86 verification inventory is incomplete")
    regressions = reconstruct_regression_tasks(
        manifest, verification_inventory["results_by_id"]
    )
    counts = Counter(str(task["workload_id"]) for task in regressions)
    if len(regressions) != EXPECTED_REGRESSION_TASKS or counts != Counter(
        EXPECTED_REGRESSION_COUNTS
    ):
        raise ValueError(
            f"Q0-pass/B-fail regression population differs: {len(regressions)} {counts}"
        )

    q0_evidence, q0_evidence_bytes = _load_q0_reference(config, repository_root)
    q0_evidence_sha256 = hashlib.sha256(q0_evidence_bytes).hexdigest()
    q0_candidates, q0_source_hashes = _load_q0_candidates(q0_root, q0_evidence)
    raw_b_rows = _raw_b_rows(
        regressions,
        generation_inventory,
        verification_inventory,
        manifest_sha256,
    )
    q0_rows = _raw_q0_rows(
        regressions,
        q0_candidates,
        q0_evidence_sha256,
        q0_source_hashes,
        str(config.model["model_revision"]),
    )
    raw_b_sha256 = _write_jsonl(raw_b_output, raw_b_rows)
    q0_verified_sha256 = _write_jsonl(q0_verified_output, q0_rows)

    tasks_by_id = {
        (bound.workload_id, bound.task.id): bound for bound in bound_tasks
    }
    variant_jobs = _build_variant_jobs(
        regressions, raw_b_rows, tasks_by_id
    )
    outcomes = _verify_variants(
        variant_jobs,
        tasks_by_id,
        config,
        lean_project_roots,
        workers=workers,
    )
    transformed_rows = []
    for job in variant_jobs:
        outcome = outcomes[str(job["variant_id"])]
        transformed_rows.append(
            {
                "schema_version": TRANSFORMED_SCHEMA_VERSION,
                **job,
                "diagnostic_verifier_classification": outcome.category,
                "diagnostic_lean_exit_code": outcome.lean_exit_code,
                "diagnostic_diagnostics": outcome.diagnostics,
                "verifier_environment_sha256": environment_bundle[
                    "environment_sha256_by_workload"
                ][str(job["workload"])],
                "scoring_excluded": True,
            }
        )
    transformed_sha256 = _write_jsonl(transformed_output, transformed_rows)
    task_rows = _task_analysis(
        regressions,
        raw_b_rows,
        q0_rows,
        transformed_rows,
        tasks_by_id,
        verification_inventory["results_by_id"],
    )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "decision_marker": "OBSERVED",
        "scoring_excluded": True,
        "official_86_results_modified": False,
        "model_inference_or_regeneration_performed": False,
        "sampling_caveat": (
            "Q0-pass/B-fail compares stochastic n=8 samples from different "
            "prompt-conditioned distributions. It identifies confirmed format failures "
            "and observable search/formalization changes, not causal harm from guidance."
        ),
        "source_bindings": {
            "manifest_sha256": manifest_sha256,
            "issue_86_results_sha256": sha256_file(
                repository_root / "evidence/mathia-prompt-ab/results.json"
            ),
            "issue_86_format_diagnostic_sha256": sha256_file(
                repository_root
                / "evidence/mathia-prompt-ab/format-contamination-diagnostic.json"
            ),
            "issue_86_verification_result_set_sha256": verification_inventory[
                "result_set_sha256"
            ],
            "q0_evidence_sha256": q0_evidence_sha256,
            "q0_raw_recovery_archive_sha256": RECOVERY_ARCHIVE_SHA256,
            "q0_workloads": q0_source_hashes,
            "verifier_environment_set_sha256": environment_bundle[
                "environment_set_sha256"
            ],
            "verifier_environment_sha256_by_workload": environment_bundle[
                "environment_sha256_by_workload"
            ],
        },
        "committed_evidence": {
            "raw_b_candidates": {
                "path": str(
                    raw_b_output.resolve().relative_to(repository_root.resolve())
                ),
                "row_count": len(raw_b_rows),
                "sha256": raw_b_sha256,
            },
            "q0_verified_candidates": {
                "path": str(
                    q0_verified_output.resolve().relative_to(repository_root.resolve())
                ),
                "row_count": len(q0_rows),
                "sha256": q0_verified_sha256,
            },
            "transformed_b_candidates": {
                "path": str(
                    transformed_output.resolve().relative_to(repository_root.resolve())
                ),
                "row_count": len(transformed_rows),
                "sha256": transformed_sha256,
            },
        },
        "mechanical_transform_contract": {
            "maximum_composed_transforms": 4,
            "transforms": [
                "strip_leading_duplicated_by",
                "unwrap_markdown_fence",
                "remove_exact_repeated_theorem_declaration",
                "unwrap_lean3_begin_end",
                "remove_whitelisted_natural_language_prefix",
                "remove_whitelisted_natural_language_suffix",
            ],
            "semantic_repair_permitted": False,
        },
        "diagnostic_tag_rules": {
            "incomplete_or_token_limited": "at least one B candidate hit token_limit",
            "hallucinated_lemma_or_api": "official Lean diagnostics report an unknown identifier/constant/declaration",
            "different_proof_family_from_q0": "B covers under half of the tactic families observed in verified Q0 candidates",
            "lost_simple_q0_strategy": "the shortest verified Q0 proof uses at most 64 generated tokens and no B candidate contains all of its tactic families",
            "guidance_followed_but_formalization_failed": "an explicit tactic family named in the intuition also occurs in B",
            "guidance_seems_ignored": "the intuition names a tactic family absent from B",
            "cannot_determine": "none of the deterministic observable rules above applies",
        },
        "aggregate": _aggregate(task_rows),
        "tasks": task_rows,
    }
    _atomic_write(analysis_output, _canonical_json_bytes(analysis, pretty=True))
    _atomic_write(readme_output, render_regression_analysis(analysis).encode("utf-8"))
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scoring-excluded Q0-pass/Arm-B-fail retrospective for issue #93"
    )
    parser.add_argument("--config", type=Path, default=Path("config/mathia-prompt-ab.json"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mathia-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--q0-root", type=Path, required=True)
    parser.add_argument("--minif2f-project-root", type=Path, required=True)
    parser.add_argument("--fresh-project-root", type=Path, required=True)
    parser.add_argument("--raw-b-output", type=Path, required=True)
    parser.add_argument("--q0-verified-output", type=Path, required=True)
    parser.add_argument("--transformed-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--readme-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = run_regression_analysis(
        PromptABConfig.load(args.config),
        args.dataset_root,
        args.mathia_root,
        args.repository_root.resolve(),
        args.manifest,
        args.artifacts,
        args.q0_root,
        {
            "minif2f-valid-clean-v2": args.minif2f_project_root,
            "fresh-composition-valid-v2": args.fresh_project_root,
        },
        args.raw_b_output,
        args.q0_verified_output,
        args.transformed_output,
        args.analysis_output,
        args.readme_output,
        workers=args.workers,
    )
    print(json.dumps(analysis["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
