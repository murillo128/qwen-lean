from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import write_artifacts
from .baseline import GeneratedCandidate, _generate_candidates, _verify_candidate
from .dataset_v2_schema import DatasetV2Record
from .generalist_v2 import MODEL_ID, MODEL_REVISION, GeneralistV2Config
from .generalist_v2_dataset import (
    _iter_jsonl,
    _read_json,
    dataset_record_preamble,
    sha256_file,
)
from .metrics import summarize_results
from .minif2f import Phase1Config
from .phase2_corpus import position_offset
from .prompt import normalize_transport
from .schema import CandidateResult, RunMetadata, TaskRecord
from .verifier import LeanVerifier

Q0_GENERATION_SCHEMA_VERSION = "generalist-v2-q0-generation-v1"
Q0_VERIFICATION_SCHEMA_VERSION = "generalist-v2-q0-verification-v1"
Q0_WORKLOADS = {
    "fresh-composition-valid-v2": 406,
    "minif2f-valid-clean-v2": 244,
    "dataset-v2-train-probe": 256,
    "riemann-fresh-valid-v2": 100,
}


def _ordered_ids_digest(ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"not a readable pinned Git checkout: {path}")
    return completed.stdout.strip()


def _load_id_view(path: Path) -> list[str]:
    ids = [str(value["statement_id"]) for value in _iter_jsonl(path)]
    if not ids or ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"generalist-v2 view is not sorted and unique: {path}")
    return ids


def _train_probe_ids(path: Path) -> list[str]:
    value = _read_json(path)
    if value.get("id") != "dataset-v2-train-probe":
        raise ValueError("generalist-v2 train probe has the wrong identity")
    ids = [str(item) for rows in value["strata"].values() for item in rows]
    if len(ids) != 256 or len(set(ids)) != 256:
        raise ValueError("generalist-v2 train probe must contain 256 unique tasks")
    return sorted(ids)


def _load_canonical_subset(
    package_root: Path, statement_ids: list[str]
) -> dict[str, DatasetV2Record]:
    wanted = set(statement_ids)
    selected: dict[str, DatasetV2Record] = {}
    for value in _iter_jsonl(package_root / "records.jsonl.gz"):
        statement_id = str(value["statement_id"])
        if statement_id in wanted:
            selected[statement_id] = DatasetV2Record.from_dict(value)
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"evaluation view has {len(missing)} unresolved statements")
    return selected


def _synthetic_task(record: DatasetV2Record) -> TaskRecord:
    if record.provenance != "synthetic" or len(record.proof_variants) != 1:
        raise ValueError("fresh-composition evaluation requires one synthetic proof")
    variant = record.proof_variants[0]
    return TaskRecord(
        id=record.statement_id,
        preamble=dataset_record_preamble(record),
        declaration=record.canonical_declaration,
        declaration_name=variant.source_declaration_name,
    )


def _source_position_verification_task(
    record: DatasetV2Record, lean_project_root: Path
) -> TaskRecord:
    """Reconstruct a real declaration in its exact source-prefix context."""

    if record.provenance not in {"real-mathlib", "external-lean", "mixed-real"}:
        raise ValueError("source-position verification requires a real record")
    source_span = record.environment.source_span
    if source_span is None:
        raise ValueError(f"real record lacks a source span: {record.statement_id}")
    if record.environment.repository.endswith("/mathlib4"):
        source_path = (
            lean_project_root / ".lake/packages/mathlib" / record.environment.file_path
        )
    else:
        source_path = lean_project_root / record.environment.file_path
    source = source_path.read_text(encoding="utf-8")
    source_start = position_offset(source, source_span.start)
    source_end = position_offset(source, source_span.end)
    if source_start >= source_end:
        raise ValueError(f"source span is empty or reversed for {record.statement_id}")
    source_declaration = source[source_start:source_end]
    variants = [
        item
        for item in record.proof_variants
        if item.source_file == record.environment.file_path
        and item.source_revision == record.environment.revision
        and item.source_declaration_name.removeprefix("_root_.").rsplit(".", 1)[-1]
        in source_declaration
    ]
    if len(variants) != 1:
        raise ValueError(
            f"source span resolves {len(variants)} proof variants for {record.statement_id}"
        )
    variant = variants[0]
    return TaskRecord(
        id=record.statement_id,
        preamble=source[:source_start].rstrip(),
        declaration=record.canonical_declaration,
        declaration_name=variant.source_declaration_name,
    )


def _clean_minif2f_tasks(path: Path, expected: int) -> list[TaskRecord]:
    tasks = [
        TaskRecord(
            id=str(value["task_id"]),
            preamble=str(value["preamble"]),
            declaration=str(value["declaration"]),
            declaration_name=str(value["declaration_name"]),
        )
        for value in _iter_jsonl(path)
    ]
    if len(tasks) != expected or len({item.id for item in tasks}) != expected:
        raise ValueError("clean miniF2F task count/identity differs")
    return tasks


def materialize_q0_workload(
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    *,
    lean_project_root: Path | None = None,
) -> tuple[
    list[TaskRecord],
    dict[str, TaskRecord],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, Any]],
]:
    if workload_id not in Q0_WORKLOADS:
        raise ValueError(f"unknown generalist-v2 Q0 workload: {workload_id}")
    expected = Q0_WORKLOADS[workload_id]
    if workload_id == "minif2f-valid-clean-v2":
        tasks = _clean_minif2f_tasks(
            package_root / "minif2f-valid-clean-v2.jsonl", expected
        )
        return tasks, {item.id: item for item in tasks}, {}, {}
    if workload_id == "dataset-v2-train-probe":
        ids = _train_probe_ids(package_root / "train-probe.json")
    else:
        ids = _load_id_view(view_dir / f"{workload_id}.jsonl")
    records = _load_canonical_subset(package_root, ids)
    tasks: list[TaskRecord] = []
    verification: dict[str, TaskRecord] = {}
    targets: dict[str, tuple[str, ...]] = {}
    task_metadata: dict[str, dict[str, Any]] = {}
    for statement_id in ids:
        record = records[statement_id]
        if workload_id == "dataset-v2-train-probe":
            task = TaskRecord(
                id=record.statement_id,
                preamble=dataset_record_preamble(record),
                declaration=record.canonical_declaration,
                declaration_name=record.proof_variants[0].source_declaration_name,
            )
            verification_task = (
                _synthetic_task(record)
                if record.provenance == "synthetic"
                else (
                    task
                    if lean_project_root is None
                    else _source_position_verification_task(record, lean_project_root)
                )
            )
            target_variants = sorted(
                record.proof_variants,
                key=lambda item: item.source_declaration_name
                != verification_task.declaration_name,
            )
            targets[statement_id] = tuple(
                normalize_transport(item.completion) for item in target_variants
            )
        else:
            task = _synthetic_task(record)
            verification_task = task
        tasks.append(task)
        verification[statement_id] = verification_task
        task_metadata[statement_id] = {
            "provenance": record.provenance,
            "structural_class": record.structural_class,
            "generator_family": record.generator_family,
            "topic_tags": list(record.topic_tags),
            "derivation_family_fingerprint": record.derivation_family_fingerprint,
        }
    if len(tasks) != expected:
        raise ValueError(f"{workload_id} expected {expected} tasks, got {len(tasks)}")
    return tasks, verification, targets, task_metadata


def _evaluation_phase1_config(
    path: Path, generalist: GeneralistV2Config
) -> Phase1Config:
    base = Phase1Config.load(path)
    expected_model = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
    }
    if base.model != expected_model:
        raise ValueError("Q0 evaluation config does not bind the pinned base")
    engine = {
        **base.engine,
        "max_model_len": int(generalist.training["resolved_context_tokens"]),
        "max_num_seqs": min(16, int(base.engine["max_num_seqs"])),
    }
    return Phase1Config(path=base.path, value={**base.value, "engine": engine})


def _sampling(config: GeneralistV2Config) -> dict[str, Any]:
    return {
        "candidates_per_task": int(config.evaluation["candidates_per_task"]),
        **config.evaluation["sampling"],
    }


def run_q0_generation(
    config: GeneralistV2Config,
    base_evaluation_config: Path,
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config.validate()
    tasks, _, _, task_metadata = materialize_q0_workload(
        workload_id, package_root, view_dir
    )
    phase1 = _evaluation_phase1_config(base_evaluation_config, config)
    sampling = _sampling(config)
    started = time.perf_counter()
    generated, engine_version = _generate_candidates(
        phase1, tasks, sampling=sampling
    )
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = output_dir / "generations.jsonl"
    with generation_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in generated:
            value = asdict(item)
            value["task"] = item.task.to_dict()
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    metadata = {
        "schema_version": Q0_GENERATION_SCHEMA_VERSION,
        "checkpoint_id": "Q0",
        "workload_id": workload_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_format_id": "lean-sft-v2-raw-whole-proof",
        "task_count": len(tasks),
        "ordered_task_ids_sha256": _ordered_ids_digest([item.id for item in tasks]),
        "candidate_count": len(generated),
        "sampling": sampling,
        "engine": phase1.engine,
        "engine_version": engine_version,
        "inference_execution": "project-controlled-local-cuda",
        "generation_wall_time_seconds": elapsed,
        "generation_sha256": sha256_file(generation_path),
        "generation_error_count": sum(
            item.generation_error is not None for item in generated
        ),
        "task_metadata": task_metadata,
    }
    (output_dir / "generation-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if metadata["generation_error_count"]:
        raise RuntimeError("Q0 generation contains infrastructure errors")
    return metadata


def _read_generations(
    path: Path, tasks: dict[str, TaskRecord]
) -> list[GeneratedCandidate]:
    generated: list[GeneratedCandidate] = []
    for value in _iter_jsonl(path):
        task_id = str(value["task"]["id"])
        task = tasks[task_id]
        generated.append(
            GeneratedCandidate(
                task=task,
                candidate_index=int(value["candidate_index"]),
                text=str(value["text"]),
                token_count=int(value["token_count"]),
                finish_reason=str(value["finish_reason"]),
                generation_latency_seconds=float(value["generation_latency_seconds"]),
                generation_error=(
                    None
                    if value.get("generation_error") is None
                    else str(value["generation_error"])
                ),
            )
        )
    return generated


def _prime_verifiers(
    verification_tasks: dict[str, TaskRecord],
    lean_project_root: Path,
    *,
    candidate_timeout_seconds: float,
    workers: int,
    targets: dict[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, LeanVerifier], set[str]]:
    representative_tasks: dict[str, TaskRecord] = {}
    for task in verification_tasks.values():
        representative_tasks.setdefault(task.preamble, task)
    preambles = sorted(representative_tasks)

    def prime(preamble: str) -> tuple[str, LeanVerifier, str | None]:
        verifier = LeanVerifier(
            lean_project_root, timeout_seconds=candidate_timeout_seconds
        )
        task = representative_tasks[preamble]
        if targets and task.id in targets:
            outcome = verifier.prime_task(
                task,
                targets[task.id][0],
                timeout_seconds=candidate_timeout_seconds,
            )
            validated_task_id: str | None = task.id
        else:
            outcome = verifier.prime_preamble(
                preamble,
                timeout_seconds=max(120.0, candidate_timeout_seconds),
            )
            validated_task_id = None
        if outcome is not None:
            diagnostics = outcome.diagnostics["stdout"] + outcome.diagnostics["stderr"]
            raise RuntimeError(
                f"Q0 verification context does not elaborate: {diagnostics}"
            )
        return preamble, verifier, validated_task_id

    with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
        primed = list(executor.map(prime, preambles))
    return (
        {preamble: verifier for preamble, verifier, _ in primed},
        {task_id for _, _, task_id in primed if task_id is not None},
    )


def _validate_oracles(
    verifiers: dict[str, LeanVerifier],
    verification_tasks: dict[str, TaskRecord],
    targets: dict[str, tuple[str, ...]],
    *,
    workers: int,
    already_validated: set[str] | None = None,
) -> None:
    def validate(item: tuple[str, tuple[str, ...]]) -> None:
        task_id, completions = item
        if already_validated and task_id in already_validated:
            return
        task = verification_tasks[task_id]
        outcome = verifiers[task.preamble].verify(task, completions[0])
        if outcome.category != "verified":
            diagnostics = outcome.diagnostics["stdout"] + outcome.diagnostics["stderr"]
            raise RuntimeError(
                f"train-probe oracle does not verify for {task_id}: {diagnostics}"
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(validate, targets.items()))


def run_q0_verification(
    config: GeneralistV2Config,
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    output_dir: Path,
    lean_project_root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    config.validate()
    tasks, verification_tasks, targets, task_metadata = materialize_q0_workload(
        workload_id,
        package_root,
        view_dir,
        lean_project_root=lean_project_root,
    )
    task_by_id = {item.id: item for item in tasks}
    generation_path = output_dir / "generations.jsonl"
    generation_metadata = _read_json(output_dir / "generation-metadata.json")
    if generation_metadata["generation_sha256"] != sha256_file(generation_path):
        raise ValueError("Q0 generation artifact hash differs")
    generated = _read_generations(generation_path, task_by_id)
    candidates_per_task = int(config.evaluation["candidates_per_task"])
    if len(generated) != len(tasks) * candidates_per_task:
        raise ValueError("Q0 generation artifact is incomplete")
    expected_project_revision = (
        "f0a20e14c1eeccd859d51bb4c2b3ee487889c303"
        if workload_id == "minif2f-valid-clean-v2"
        else "7715064f690d0689f30889846f4e2c5e7ec0c47e"
    )
    if _git_revision(lean_project_root) != expected_project_revision:
        raise ValueError("Q0 verifier project revision differs from Dataset-v2")
    mathlib_revision = _git_revision(lean_project_root / ".lake/packages/mathlib")
    started = time.perf_counter()
    verifiers, validated_oracles = _prime_verifiers(
        verification_tasks,
        lean_project_root,
        candidate_timeout_seconds=float(
            config.evaluation["verifier_timeout_seconds"]
        ),
        workers=workers,
        targets=targets,
    )
    if workload_id == "dataset-v2-train-probe":
        _validate_oracles(
            verifiers,
            verification_tasks,
            targets,
            workers=workers,
            already_validated=validated_oracles,
        )

    def verify(item: GeneratedCandidate) -> CandidateResult:
        verification_task = verification_tasks[item.task.id]
        return _verify_candidate(
            verifiers[verification_task.preamble],
            GeneratedCandidate(
                task=verification_task,
                candidate_index=item.candidate_index,
                text=item.text,
                token_count=item.token_count,
                finish_reason=item.finish_reason,
                generation_latency_seconds=item.generation_latency_seconds,
                generation_error=item.generation_error,
            ),
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(verify, generated))
    elapsed = time.perf_counter() - started
    summary = summarize_results(
        results,
        expected_task_ids=[item.id for item in tasks],
        candidates_per_task=candidates_per_task,
    )
    exact_candidates = 0
    exact_tasks: set[str] = set()
    if targets:
        for result in results:
            if normalize_transport(result.candidate_text) in targets[result.task_id]:
                exact_candidates += 1
                exact_tasks.add(result.task_id)
    metadata = RunMetadata(
        candidate_source="model",
        task_source=f"lean-whole-proof-v2/{workload_id}",
        prompt_format_id="lean-sft-v2-raw-whole-proof",
        lean_toolchain=(lean_project_root / "lean-toolchain")
        .read_text(encoding="utf-8")
        .strip(),
        mathlib_revision=mathlib_revision,
        verifier_timeout_seconds=float(
            config.evaluation["verifier_timeout_seconds"]
        ),
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=workload_id,
        candidates_per_task=candidates_per_task,
        inference_engine=str(generation_metadata["engine"]["name"]),
        inference_engine_version=str(generation_metadata["engine_version"]),
        adapter_enabled=False,
        generation_settings=generation_metadata["sampling"],
        runtime={
            "inference_execution": "project-controlled-local-cuda",
            "generation_wall_time_seconds": generation_metadata[
                "generation_wall_time_seconds"
            ],
            "verification_wall_time_seconds": elapsed,
            "verification_workers": workers,
            "verification_context_count": len(verifiers),
            "oracle_validation_count": len(targets),
            "lean_project_root_identity": (
                "miniF2F@f0a20e14c1eeccd859d51bb4c2b3ee487889c303"
                if workload_id == "minif2f-valid-clean-v2"
                else "PrimeNumberTheoremAnd@7715064f690d0689f30889846f4e2c5e7ec0c47e"
            ),
        },
    )
    summary.update(
        {
            "schema_version": Q0_VERIFICATION_SCHEMA_VERSION,
            "checkpoint_id": "Q0",
            "workload_id": workload_id,
            "exact_target_candidate_count": exact_candidates,
            "exact_target_task_count": len(exact_tasks),
            "task_metadata": task_metadata,
        }
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
    if not summary["complete"] or summary["infrastructure_error_count"]:
        raise RuntimeError("Q0 verification is incomplete or has infrastructure errors")
    return summary


def compact_q0_evidence(
    config: GeneralistV2Config,
    evaluation_root: Path,
    output: Path,
) -> dict[str, Any]:
    config.validate()
    workloads: dict[str, Any] = {}
    for workload_id, expected_tasks in Q0_WORKLOADS.items():
        root = evaluation_root / workload_id
        metadata = _read_json(root / "run.json")
        summary = _read_json(root / "summary.json")
        if summary["task_count"] != expected_tasks:
            raise ValueError(f"Q0 task count differs for {workload_id}")
        if (
            not summary["complete"]
            or summary["infrastructure_error_count"] != 0
            or metadata["model_revision"] != MODEL_REVISION
            or metadata["candidates_per_task"] != 8
        ):
            raise ValueError(f"Q0 workload contract failed for {workload_id}")
        ordered_counts = [
            int(item["verified_candidate_count"]) for item in summary["per_task"]
        ]
        workloads[workload_id] = {
            "task_count": summary["task_count"],
            "candidate_count": summary["candidate_count"],
            "tasks_with_verified_candidate": summary[
                "tasks_with_verified_candidate"
            ],
            "pass_at_k": summary["pass_at_k"],
            "category_counts": summary["category_counts"],
            "finish_reason_counts": summary["finish_reason_counts"],
            "exact_target_candidate_count": summary[
                "exact_target_candidate_count"
            ],
            "exact_target_task_count": summary["exact_target_task_count"],
            "verified_counts": ordered_counts,
            "results_sha256": sha256_file(root / "results.jsonl"),
            "generation_sha256": sha256_file(root / "generations.jsonl"),
            "timing_seconds": metadata["runtime"],
        }
    evidence = {
        "schema_version": "generalist-v2-q0-evidence-v1",
        "checkpoint_id": "Q0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "candidates_per_task": 8,
        "selection_test_workloads_consulted": False,
        "riemann_used_for_selection": False,
        "workloads": workloads,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence
