from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .artifacts import write_artifacts
from .baseline import (
    GeneratedCandidate,
    LoRAAdapterSpec,
    _generate_candidates,
    _verify_candidate,
    render_prompt,
)
from .dataset_v2_schema import DatasetV2Record
from .generalist_v2 import (
    MODEL_ID,
    MODEL_REVISION,
    CheckpointValidation,
    GeneralistV2Config,
    compare_paired_verified_counts,
    select_generalist_checkpoint,
)
from .generalist_v2_dataset import (
    _iter_jsonl,
    _read_json,
    dataset_record_preamble,
    sha256_file,
)
from .metrics import pass_at_k as estimate_pass_at_k
from .metrics import summarize_results
from .minif2f import Phase1Config
from .phase2_corpus import _lex_lean, position_offset
from .prompt import normalize_transport
from .schema import RESULT_CATEGORIES, CandidateResult, RunMetadata, TaskRecord
from .verifier import LeanVerifier

Q0_GENERATION_SCHEMA_VERSION = "generalist-v2-q0-generation-v1"
Q0_VERIFICATION_SCHEMA_VERSION = "generalist-v2-q0-verification-v1"
CHECKPOINT_GENERATION_SCHEMA_VERSION = "generalist-v2-checkpoint-generation-v1"
CHECKPOINT_VERIFICATION_SCHEMA_VERSION = "generalist-v2-checkpoint-verification-v1"
EXTENDED_GENERATION_SCHEMA_VERSION = "generalist-v2-extended-generation-v1"
EXTENDED_VERIFICATION_SCHEMA_VERSION = "generalist-v2-extended-verification-v1"
FINAL_GENERATION_SCHEMA_VERSION = "generalist-v2-final-generation-v1"
FINAL_VERIFICATION_SCHEMA_VERSION = "generalist-v2-final-verification-v1"
EXTENDED_SEARCH_KS = (1, 2, 4, 8, 16, 32, 64)
DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"
DEEPSEEK_MODEL_REVISION = "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b"
Q0_WORKLOADS = {
    "fresh-composition-valid-v2": 406,
    "minif2f-valid-clean-v2": 244,
    "dataset-v2-train-probe": 256,
    "riemann-fresh-valid-v2": 100,
}
EXTENDED_SEARCH_WORKLOADS = {
    "fresh-composition-valid-v2": 406,
    "minif2f-valid-clean-v2": 244,
}
FINAL_TEST_WORKLOADS = {
    "fresh-composition-test-v2": 415,
    "minif2f-test-clean-v2": 244,
    "riemann-fresh-test-v2": 104,
}


def _ordered_ids_digest(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalized_verified_proof_sha256(candidate_text: str) -> str:
    """Hash the verified Lean token stream after transport/comment normalization."""

    tokens = _lex_lean(normalize_transport(candidate_text))
    canonical = json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
    return _text_sha256(canonical)


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
    known_workloads = {
        **Q0_WORKLOADS,
        **EXTENDED_SEARCH_WORKLOADS,
        **FINAL_TEST_WORKLOADS,
    }
    if workload_id not in known_workloads:
        raise ValueError(f"unknown generalist-v2 evaluation workload: {workload_id}")
    expected = known_workloads[workload_id]
    if workload_id.startswith("minif2f-"):
        tasks = _clean_minif2f_tasks(package_root / f"{workload_id}.jsonl", expected)
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
                key=lambda item: (
                    item.source_declaration_name != verification_task.declaration_name
                ),
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


def _final_phase1_config(
    path: Path, generalist: GeneralistV2Config, *, model_label: str
) -> Phase1Config:
    base = Phase1Config.load(path)
    expected = (
        (DEEPSEEK_MODEL_ID, DEEPSEEK_MODEL_REVISION)
        if model_label == "deepseek"
        else (MODEL_ID, MODEL_REVISION)
    )
    if (
        base.model.get("model_id"),
        base.model.get("model_revision"),
        base.model.get("tokenizer_id"),
        base.model.get("tokenizer_revision"),
    ) != (expected[0], expected[1], expected[0], expected[1]):
        raise ValueError(f"final {model_label} evaluation model binding differs")
    if model_label == "deepseek":
        census = base.value.get("assessment", {}).get("context_census", {})
        expected_sampling = {
            "candidates_per_task": 8,
            **generalist.evaluation["sampling"],
        }
        if (
            base.value.get("assessment", {}).get("id")
            != "deepseek-prover-v2-7b-generalist-v2-final-v1"
            or base.sampling != expected_sampling
            or census.get("max_prompt_plus_generation_tokens") != 21041
            or int(base.engine.get("max_model_len", 0)) < 21041
            or float(base.engine.get("cpu_offload_gb", 0.0)) != 8.0
            or base.engine.get("dtype") != "bfloat16"
            or base.engine.get("quantization") is not None
        ):
            raise ValueError("final DeepSeek context or BF16 offload contract differs")
        engine = dict(base.engine)
    else:
        engine = {
            **base.engine,
            "max_model_len": int(generalist.training["resolved_context_tokens"]),
            "max_num_seqs": min(16, int(base.engine["max_num_seqs"])),
        }
    return Phase1Config(path=base.path, value={**base.value, "engine": engine})


def _sampling(
    config: GeneralistV2Config, *, candidates_per_task: int | None = None
) -> dict[str, Any]:
    return {
        "candidates_per_task": (
            int(config.evaluation["candidates_per_task"])
            if candidates_per_task is None
            else candidates_per_task
        ),
        **config.evaluation["sampling"],
    }


def _checkpoint_adapter_spec(
    config: GeneralistV2Config, checkpoint_id: str, adapter_dir: Path
) -> LoRAAdapterSpec:
    if checkpoint_id not in {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("generalist-v2 adapter checkpoint must be Q1-Q4")
    adapter_config = _read_json(adapter_dir / "adapter_config.json")
    if (
        adapter_config.get("base_model_name_or_path") != MODEL_ID
        or adapter_config.get("revision") != MODEL_REVISION
        or int(adapter_config.get("r", -1)) != int(config.lora["r"])
        or adapter_config.get("target_modules") != str(config.lora["target_regex"])
        or not (adapter_dir / "adapter_model.safetensors").is_file()
    ):
        raise ValueError(f"generalist-v2 {checkpoint_id} adapter identity differs")
    return LoRAAdapterSpec(
        adapter_id=f"qwen-lean-generalist-v2-{checkpoint_id.lower()}",
        path=adapter_dir.resolve(),
        rank=int(config.lora["r"]),
        base_model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
    )


def run_checkpoint_generation(
    config: GeneralistV2Config,
    base_evaluation_config: Path,
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    output_dir: Path,
    *,
    checkpoint_id: str,
    adapter_dir: Path | None,
    candidates_per_task: int | None = None,
) -> dict[str, Any]:
    config.validate()
    if (checkpoint_id == "Q0") != (adapter_dir is None):
        raise ValueError("Q0 must omit an adapter and Q1-Q4 must provide one")
    adapter = (
        None
        if adapter_dir is None
        else _checkpoint_adapter_spec(config, checkpoint_id, adapter_dir)
    )
    tasks, _, _, task_metadata = materialize_q0_workload(
        workload_id, package_root, view_dir
    )
    phase1 = _evaluation_phase1_config(base_evaluation_config, config)
    resolved_candidates = (
        int(config.evaluation["candidates_per_task"])
        if candidates_per_task is None
        else candidates_per_task
    )
    if resolved_candidates not in {8, 64}:
        raise ValueError("generalist-v2 generation budget must be 8 or 64")
    sampling = _sampling(config, candidates_per_task=resolved_candidates)
    started = time.perf_counter()
    generated, engine_version = _generate_candidates(
        phase1, tasks, sampling=sampling, adapter=adapter
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
        "schema_version": (
            EXTENDED_GENERATION_SCHEMA_VERSION
            if resolved_candidates == 64
            else (
                Q0_GENERATION_SCHEMA_VERSION
                if checkpoint_id == "Q0"
                else CHECKPOINT_GENERATION_SCHEMA_VERSION
            )
        ),
        "evaluation_profile": (
            "extended-search-budget-v1"
            if resolved_candidates == 64
            else "checkpoint-screening-n8"
        ),
        "checkpoint_id": checkpoint_id,
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
        "generate_all_candidates_without_early_stop": resolved_candidates == 64,
        "adapter": (
            None
            if adapter is None
            else {
                "adapter_id": adapter.adapter_id,
                "adapter_rank": adapter.rank,
                "base_model_id": adapter.base_model_id,
                "base_model_revision": adapter.base_model_revision,
                "adapter_config_sha256": sha256_file(
                    adapter.path / "adapter_config.json"
                ),
                "adapter_model_sha256": sha256_file(
                    adapter.path / "adapter_model.safetensors"
                ),
            }
        ),
        "task_metadata": task_metadata,
    }
    (output_dir / "generation-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if metadata["generation_error_count"]:
        raise RuntimeError(
            f"generalist-v2 {checkpoint_id} generation contains infrastructure errors"
        )
    return metadata


def run_final_generation(
    config: GeneralistV2Config,
    evaluation_config: Path,
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    output_dir: Path,
    *,
    model_label: str,
    selected_checkpoint: str,
    adapter_dir: Path | None,
) -> dict[str, Any]:
    """Generate one frozen n=8 final lane without overwriting prior evidence."""

    config.validate()
    if model_label not in {"base", "selected", "deepseek"}:
        raise ValueError("final model label must be base, selected, or deepseek")
    if selected_checkpoint not in {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("final evaluation requires a frozen Q1-Q4 checkpoint")
    if (model_label == "selected") != (adapter_dir is not None):
        raise ValueError("only the selected final lane may provide an adapter")
    if (output_dir / "generation-metadata.json").exists():
        existing = _read_json(output_dir / "generation-metadata.json")
        if existing.get("generation_error_count") == 0:
            raise FileExistsError(
                f"refusing to overwrite frozen final generation: {output_dir}"
            )
    adapter = (
        None
        if adapter_dir is None
        else _checkpoint_adapter_spec(config, selected_checkpoint, adapter_dir)
    )
    tasks, _, _, task_metadata = materialize_q0_workload(
        workload_id, package_root, view_dir
    )
    phase1 = _final_phase1_config(evaluation_config, config, model_label=model_label)
    sampling = _sampling(config, candidates_per_task=8)
    started = time.perf_counter()
    generated, engine_version = _generate_candidates(
        phase1, tasks, sampling=sampling, adapter=adapter
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
        "schema_version": FINAL_GENERATION_SCHEMA_VERSION,
        "evaluation_profile": "postselection-final-n8",
        "model_label": model_label,
        "selected_checkpoint": selected_checkpoint,
        "workload_id": workload_id,
        "model_id": str(phase1.model["model_id"]),
        "model_revision": str(phase1.model["model_revision"]),
        "tokenizer_id": str(phase1.model["tokenizer_id"]),
        "tokenizer_revision": str(phase1.model["tokenizer_revision"]),
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
        "final_only_workload": workload_id in FINAL_TEST_WORKLOADS,
        "first_complete_result_overwrite_protected": True,
        "adapter": (
            None
            if adapter is None
            else {
                "adapter_id": adapter.adapter_id,
                "adapter_rank": adapter.rank,
                "base_model_id": adapter.base_model_id,
                "base_model_revision": adapter.base_model_revision,
                "adapter_config_sha256": sha256_file(
                    adapter.path / "adapter_config.json"
                ),
                "adapter_model_sha256": sha256_file(
                    adapter.path / "adapter_model.safetensors"
                ),
            }
        ),
        "task_metadata": task_metadata,
    }
    (output_dir / "generation-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if metadata["generation_error_count"]:
        raise RuntimeError(
            f"generalist-v2 final {model_label} generation contains "
            "infrastructure errors"
        )
    return metadata


def run_final_deepseek_preflight(
    config: GeneralistV2Config,
    evaluation_config: Path,
    selection_path: Path,
    package_root: Path,
    view_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Load and generate from the longest validation prompt in the BF16 lane."""

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from .qwen35_4b_base_assessment import _GpuMemorySampler

    config.validate()
    selection = _read_json(selection_path)
    selected_checkpoint = str(
        selection.get("selection", {}).get("selected_checkpoint", "")
    )
    if (
        selection.get("status") != "frozen"
        or selected_checkpoint not in {"Q1", "Q2", "Q3", "Q4"}
    ):
        raise ValueError("DeepSeek final preflight requires a frozen checkpoint")
    phase1 = _final_phase1_config(
        evaluation_config, config, model_label="deepseek"
    )
    census = phase1.value["assessment"]["context_census"]
    maximum = census["maximum_task"]
    workload_id = str(maximum["workload_id"])
    task_id = str(maximum["task_id"])
    tasks, _, _, _ = materialize_q0_workload(
        workload_id, package_root, view_dir
    )
    selected = [task for task in tasks if task.id == task_id]
    if len(selected) != 1:
        raise ValueError("DeepSeek preflight maximum validation task differs")
    snapshot = Path(
        snapshot_download(
            repo_id=DEEPSEEK_MODEL_ID,
            revision=DEEPSEEK_MODEL_REVISION,
            local_files_only=True,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompt_tokens = len(
        tokenizer.encode(render_prompt(selected[0]), add_special_tokens=False)
    )
    max_new_tokens = int(config.evaluation["sampling"]["max_new_tokens"])
    if prompt_tokens + max_new_tokens != int(
        census["max_prompt_plus_generation_tokens"]
    ):
        raise ValueError("DeepSeek preflight tokenizer context census differs")
    sampling = {
        **_sampling(config, candidates_per_task=8),
        "candidates_per_task": 1,
    }
    started = time.perf_counter()
    with _GpuMemorySampler() as memory:
        generated, engine_version = _generate_candidates(
            phase1, selected, sampling=sampling
        )
    elapsed = time.perf_counter() - started
    if (
        len(generated) != 1
        or generated[0].generation_error is not None
        or not memory.samples_mib
    ):
        raise RuntimeError("DeepSeek final local-GPU context preflight failed")
    evidence = {
        "schema_version": "generalist-v2-deepseek-final-preflight-v1",
        "status": "passed",
        "model_id": DEEPSEEK_MODEL_ID,
        "model_revision": DEEPSEEK_MODEL_REVISION,
        "tokenizer_revision": DEEPSEEK_MODEL_REVISION,
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "selected_checkpoint_frozen": selected_checkpoint,
        "validation_only_before_final_test": True,
        "workload_id": workload_id,
        "task_id": task_id,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "max_prompt_plus_generation_tokens": prompt_tokens + max_new_tokens,
        "engine": phase1.engine,
        "engine_version": engine_version,
        "inference_execution": "project-controlled-local-cuda",
        "generation_wall_time_seconds": elapsed,
        "generated_token_count": generated[0].token_count,
        "finish_reason": generated[0].finish_reason,
        "candidate_text_sha256": _text_sha256(generated[0].text),
        "peak_device_memory_used_mib": max(memory.samples_mib),
        "gpu_memory_sample_count": len(memory.samples_mib),
        "candidate_budget_role": "single-candidate-infrastructure-preflight-only",
        "semantic_outcome_used_for_selection_or_tuning": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def run_q0_generation(
    config: GeneralistV2Config,
    base_evaluation_config: Path,
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return run_checkpoint_generation(
        config,
        base_evaluation_config,
        workload_id,
        package_root,
        view_dir,
        output_dir,
        checkpoint_id="Q0",
        adapter_dir=None,
    )


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


def _summarize_extended_raw_candidate_evidence(
    path: Path,
    *,
    checkpoint_id: str,
    workload_id: str,
    expected_task_ids: list[str],
    expected_adapter_model_sha256: str,
    sampling_seed: int,
) -> dict[str, Any]:
    """Validate the primary n=64 candidate artifact and derive density metrics."""

    expected_task_id_set = set(expected_task_ids)
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    verified_by_task: dict[str, int] = defaultdict(int)
    verified_indices_by_task: dict[str, list[int]] = defaultdict(list)
    unique_verified_by_task: dict[str, set[str]] = defaultdict(set)
    candidate_count = 0
    for row in _iter_jsonl(path):
        task_id = str(row.get("task_id", ""))
        candidate_text = str(row.get("candidate_text", ""))
        category = str(row.get("category", ""))
        normalized_hash = row.get("normalized_proof_sha256")
        if (
            row.get("schema_version") != "generalist-v2-extended-candidate-v1"
            or row.get("checkpoint_id") != checkpoint_id
            or row.get("workload_id") != workload_id
            or row.get("model_id") != MODEL_ID
            or row.get("model_revision") != MODEL_REVISION
            or row.get("adapter_model_sha256") != expected_adapter_model_sha256
            or int(row.get("sampling_seed", -1)) != sampling_seed
            or task_id not in expected_task_id_set
            or category not in RESULT_CATEGORIES
            or row.get("candidate_text_sha256") != _text_sha256(candidate_text)
            or not isinstance(row.get("diagnostics"), dict)
        ):
            raise ValueError("extended raw candidate identity or payload differs")
        candidate_index = int(row.get("candidate_index", -1))
        if row.get("candidate_id") != f"model-{candidate_index}":
            raise ValueError("extended raw candidate identifier differs")
        indices_by_task[task_id].append(candidate_index)
        if category == "verified":
            expected_normalized = _normalized_verified_proof_sha256(candidate_text)
            if normalized_hash != expected_normalized:
                raise ValueError("extended verified proof hash differs")
            verified_by_task[task_id] += 1
            verified_indices_by_task[task_id].append(candidate_index)
            unique_verified_by_task[task_id].add(expected_normalized)
        elif normalized_hash is not None:
            raise ValueError("non-verified extended candidate has a proof identity")
        candidate_count += 1

    for task_id in expected_task_ids:
        if sorted(indices_by_task[task_id]) != list(range(64)):
            raise ValueError(
                f"extended raw candidates are incomplete for task {task_id}"
            )
    if candidate_count != len(expected_task_ids) * 64:
        raise ValueError("extended raw candidate count differs")

    per_task = [
        {
            "task_id": task_id,
            "verified_candidate_count": verified_by_task[task_id],
            "unique_verified_proof_count": len(unique_verified_by_task[task_id]),
            "verified_candidate_indices": sorted(verified_indices_by_task[task_id]),
        }
        for task_id in expected_task_ids
    ]
    bucket_counts = {key: 0 for key in ("0", "1", "2-4", "5-15", "16-31", "32+")}
    for item in per_task:
        count = int(item["verified_candidate_count"])
        bucket = (
            "0"
            if count == 0
            else (
                "1"
                if count == 1
                else (
                    "2-4"
                    if count <= 4
                    else "5-15"
                    if count <= 15
                    else "16-31"
                    if count <= 31
                    else "32+"
                )
            )
        )
        bucket_counts[bucket] += 1
    verified_count = sum(verified_by_task.values())
    unique_verified_count = sum(
        len(values) for values in unique_verified_by_task.values()
    )
    solved = [item for item in per_task if item["verified_candidate_count"]]
    return {
        "raw_candidate_count": candidate_count,
        "verified_candidate_count": verified_count,
        "verified_rate": verified_count / candidate_count if candidate_count else 0.0,
        "unique_verified_proof_count": unique_verified_count,
        "unique_verified_rate": (
            unique_verified_count / candidate_count if candidate_count else 0.0
        ),
        "unique_verified_identity": "task_id plus normalized Lean proof-token hash",
        "verified_rate_denominator": "all generated candidates",
        "unique_verified_rate_denominator": "all generated candidates",
        "verified_duplication_fraction": (
            1.0 - unique_verified_count / verified_count if verified_count else None
        ),
        "verified_candidates_per_task_buckets": bucket_counts,
        "mean_verified_candidates_per_solved_task": (
            fmean(int(item["verified_candidate_count"]) for item in solved)
            if solved
            else None
        ),
        "mean_unique_verified_proofs_per_solved_task": (
            fmean(int(item["unique_verified_proof_count"]) for item in solved)
            if solved
            else None
        ),
        "per_task": per_task,
    }


def _write_extended_raw_candidate_evidence(
    path: Path,
    generated: list[GeneratedCandidate],
    results: list[CandidateResult],
    generation_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Persist every n=64 generation and Lean outcome without early stopping."""

    if len(generated) != len(results):
        raise ValueError("extended generations and results differ in length")
    adapter = generation_metadata.get("adapter")
    if not isinstance(adapter, dict):
        raise TypeError("extended validation requires the frozen adapter")
    checkpoint_id = str(generation_metadata["checkpoint_id"])
    workload_id = str(generation_metadata["workload_id"])
    sampling_seed = int(generation_metadata["sampling"]["seed"])
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed_handle:
            with io.TextIOWrapper(
                compressed_handle, encoding="utf-8", newline="\n"
            ) as handle:
                for item, result in zip(generated, results, strict=True):
                    if (
                        item.task.id != result.task_id
                        or item.candidate_index != result.candidate_index
                        or item.text != result.candidate_text
                    ):
                        raise ValueError(
                            "extended generation/result candidate order differs"
                        )
                    value = {
                        "schema_version": "generalist-v2-extended-candidate-v1",
                        "checkpoint_id": checkpoint_id,
                        "workload_id": workload_id,
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "adapter_model_sha256": adapter["adapter_model_sha256"],
                        "task_id": result.task_id,
                        "candidate_id": result.candidate_id,
                        "candidate_index": result.candidate_index,
                        "sampling_seed": sampling_seed,
                        "candidate_text": result.candidate_text,
                        "candidate_text_sha256": _text_sha256(result.candidate_text),
                        "generated_token_count": result.generated_token_count,
                        "finish_reason": result.finish_reason,
                        "generation_latency_seconds": result.generation_latency_seconds,
                        "category": result.category,
                        "lean_exit_code": result.lean_exit_code,
                        "diagnostics": result.diagnostics,
                        "verification_latency_seconds": (
                            result.verification_latency_seconds
                        ),
                        "total_latency_seconds": result.total_latency_seconds,
                        "normalized_proof_sha256": (
                            _normalized_verified_proof_sha256(result.candidate_text)
                            if result.category == "verified"
                            else None
                        ),
                    }
                    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
    expected_task_ids = list(dict.fromkeys(item.task.id for item in generated))
    return _summarize_extended_raw_candidate_evidence(
        path,
        checkpoint_id=checkpoint_id,
        workload_id=workload_id,
        expected_task_ids=expected_task_ids,
        expected_adapter_model_sha256=str(adapter["adapter_model_sha256"]),
        sampling_seed=sampling_seed,
    )


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


def run_checkpoint_verification(
    config: GeneralistV2Config,
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    output_dir: Path,
    lean_project_root: Path,
    *,
    checkpoint_id: str,
    workers: int,
    candidates_per_task: int | None = None,
    final_model_label: str | None = None,
) -> dict[str, Any]:
    config.validate()
    if final_model_label is None and checkpoint_id not in {
        "Q0",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }:
        raise ValueError("generalist-v2 verification checkpoint must be Q0-Q4")
    if final_model_label is not None and (
        final_model_label not in {"base", "selected", "deepseek"}
        or checkpoint_id not in {"Q1", "Q2", "Q3", "Q4"}
    ):
        raise ValueError("final verification needs a model lane and frozen checkpoint")
    tasks, verification_tasks, targets, task_metadata = materialize_q0_workload(
        workload_id,
        package_root,
        view_dir,
        lean_project_root=lean_project_root,
    )
    task_by_id = {item.id: item for item in tasks}
    generation_path = output_dir / "generations.jsonl"
    generation_metadata = _read_json(output_dir / "generation-metadata.json")
    resolved_candidates = (
        int(config.evaluation["candidates_per_task"])
        if candidates_per_task is None
        else candidates_per_task
    )
    if resolved_candidates not in {8, 64}:
        raise ValueError("generalist-v2 verification budget must be 8 or 64")
    if final_model_label is not None and resolved_candidates != 8:
        raise ValueError("final assessment retains the frozen n=8 budget")
    expected_generation_schema = (
        FINAL_GENERATION_SCHEMA_VERSION
        if final_model_label is not None
        else (
            EXTENDED_GENERATION_SCHEMA_VERSION
            if resolved_candidates == 64
            else (
                Q0_GENERATION_SCHEMA_VERSION
                if checkpoint_id == "Q0"
                else CHECKPOINT_GENERATION_SCHEMA_VERSION
            )
        )
    )
    adapter_metadata = generation_metadata.get("adapter")
    expected_model = (
        (DEEPSEEK_MODEL_ID, DEEPSEEK_MODEL_REVISION)
        if final_model_label == "deepseek"
        else (MODEL_ID, MODEL_REVISION)
    )
    final_identity_differs = final_model_label is not None and (
        generation_metadata.get("model_label") != final_model_label
        or generation_metadata.get("selected_checkpoint") != checkpoint_id
        or ((final_model_label == "selected") != (adapter_metadata is not None))
        or generation_metadata.get("model_id") != expected_model[0]
        or generation_metadata.get("model_revision") != expected_model[1]
        or generation_metadata.get("candidate_count") != len(tasks) * 8
        or generation_metadata.get("first_complete_result_overwrite_protected")
        is not True
    )
    checkpoint_identity_differs = final_model_label is None and (
        generation_metadata.get("checkpoint_id") != checkpoint_id
        or ((checkpoint_id == "Q0") != (adapter_metadata is None))
    )
    if (
        generation_metadata.get("schema_version") != expected_generation_schema
        or final_identity_differs
        or checkpoint_identity_differs
    ):
        raise ValueError("generalist-v2 generation checkpoint identity differs")
    if generation_metadata["generation_sha256"] != sha256_file(generation_path):
        raise ValueError("Q0 generation artifact hash differs")
    generated = _read_generations(generation_path, task_by_id)
    if len(generated) != len(tasks) * resolved_candidates:
        raise ValueError("Q0 generation artifact is incomplete")
    expected_project_revision = (
        "f0a20e14c1eeccd859d51bb4c2b3ee487889c303"
        if workload_id.startswith("minif2f-")
        else "7715064f690d0689f30889846f4e2c5e7ec0c47e"
    )
    if _git_revision(lean_project_root) != expected_project_revision:
        raise ValueError("Q0 verifier project revision differs from Dataset-v2")
    mathlib_revision = _git_revision(lean_project_root / ".lake/packages/mathlib")
    started = time.perf_counter()
    verifiers, validated_oracles = _prime_verifiers(
        verification_tasks,
        lean_project_root,
        candidate_timeout_seconds=float(config.evaluation["verifier_timeout_seconds"]),
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
        results: list[CandidateResult] = []
        for completed, result in enumerate(executor.map(verify, generated), start=1):
            results.append(result)
            if completed % 128 == 0 or completed == len(generated):
                print(
                    f"verified {completed}/{len(generated)} candidates for "
                    f"{checkpoint_id}/{workload_id}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    summary = summarize_results(
        results,
        expected_task_ids=[item.id for item in tasks],
        candidates_per_task=resolved_candidates,
        ks=(EXTENDED_SEARCH_KS if resolved_candidates == 64 else (1, 4, 8)),
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
        verifier_timeout_seconds=float(config.evaluation["verifier_timeout_seconds"]),
        model_id=str(generation_metadata.get("model_id", MODEL_ID)),
        tokenizer_id=str(generation_metadata.get("tokenizer_id", MODEL_ID)),
        model_revision=str(generation_metadata.get("model_revision", MODEL_REVISION)),
        tokenizer_revision=str(
            generation_metadata.get("tokenizer_revision", MODEL_REVISION)
        ),
        workload_id=workload_id,
        candidates_per_task=resolved_candidates,
        inference_engine=str(generation_metadata["engine"]["name"]),
        inference_engine_version=str(generation_metadata["engine_version"]),
        adapter_enabled=adapter_metadata is not None,
        adapter_id=(
            None if adapter_metadata is None else str(adapter_metadata["adapter_id"])
        ),
        adapter_rank=(
            None if adapter_metadata is None else int(adapter_metadata["adapter_rank"])
        ),
        selected_adapter_binding=adapter_metadata,
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
                if workload_id.startswith("minif2f-")
                else "PrimeNumberTheoremAnd@7715064f690d0689f30889846f4e2c5e7ec0c47e"
            ),
        },
    )
    summary.update(
        {
            "schema_version": (
                FINAL_VERIFICATION_SCHEMA_VERSION
                if final_model_label is not None
                else (
                    EXTENDED_VERIFICATION_SCHEMA_VERSION
                    if resolved_candidates == 64
                    else (
                        Q0_VERIFICATION_SCHEMA_VERSION
                        if checkpoint_id == "Q0"
                        else CHECKPOINT_VERIFICATION_SCHEMA_VERSION
                    )
                )
            ),
            "evaluation_profile": (
                "postselection-final-n8"
                if final_model_label is not None
                else (
                    "extended-search-budget-v1"
                    if resolved_candidates == 64
                    else "checkpoint-screening-n8"
                )
            ),
            "checkpoint_id": checkpoint_id,
            "model_label": final_model_label,
            "workload_id": workload_id,
            "exact_target_candidate_count": exact_candidates,
            "exact_target_task_count": len(exact_tasks),
            "task_metadata": task_metadata,
        }
    )
    if resolved_candidates == 64:
        raw_path = output_dir / "raw-candidates.jsonl.gz"
        density = _write_extended_raw_candidate_evidence(
            raw_path, generated, results, generation_metadata
        )
        summary["all_candidates_verified_without_early_stop"] = True
        summary["extended_candidate_evidence"] = {
            "artifact": "raw-candidates.jsonl.gz",
            "sha256": sha256_file(raw_path),
            **density,
        }
    write_artifacts(output_dir, metadata, results, summary=summary)
    if not summary["complete"] or summary["infrastructure_error_count"]:
        raise RuntimeError(
            f"generalist-v2 {checkpoint_id} verification is incomplete or has "
            "infrastructure errors"
        )
    return summary


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
    return run_checkpoint_verification(
        config,
        workload_id,
        package_root,
        view_dir,
        output_dir,
        lean_project_root,
        checkpoint_id="Q0",
        workers=workers,
    )


def run_final_verification(
    config: GeneralistV2Config,
    workload_id: str,
    package_root: Path,
    view_dir: Path,
    output_dir: Path,
    lean_project_root: Path,
    *,
    model_label: str,
    selected_checkpoint: str,
    workers: int,
) -> dict[str, Any]:
    if (output_dir / "summary.json").exists():
        existing = _read_json(output_dir / "summary.json")
        if (
            existing.get("complete") is True
            and existing.get("infrastructure_error_count") == 0
        ):
            raise FileExistsError(
                f"refusing to overwrite complete final verification: {output_dir}"
            )
    return run_checkpoint_verification(
        config,
        workload_id,
        package_root,
        view_dir,
        output_dir,
        lean_project_root,
        checkpoint_id=selected_checkpoint,
        workers=workers,
        candidates_per_task=8,
        final_model_label=model_label,
    )


def compact_q0_evidence(
    config: GeneralistV2Config,
    evaluation_root: Path,
    output: Path,
) -> dict[str, Any]:
    config.validate()
    workloads: dict[str, Any] = {}
    for workload_id, expected_tasks in Q0_WORKLOADS.items():
        root = evaluation_root / workload_id
        generation_metadata = _read_json(root / "generation-metadata.json")
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
        per_task = [
            {
                "task_id": str(item["task_id"]),
                "verified_candidate_count": int(item["verified_candidate_count"]),
            }
            for item in summary["per_task"]
        ]
        workloads[workload_id] = {
            "task_count": summary["task_count"],
            "candidate_count": summary["candidate_count"],
            "tasks_with_verified_candidate": summary["tasks_with_verified_candidate"],
            "pass_at_k": summary["pass_at_k"],
            "category_counts": summary["category_counts"],
            "finish_reason_counts": summary["finish_reason_counts"],
            "exact_target_candidate_count": summary["exact_target_candidate_count"],
            "exact_target_task_count": summary["exact_target_task_count"],
            "verified_counts": ordered_counts,
            "per_task": per_task,
            "ordered_task_ids_sha256": generation_metadata["ordered_task_ids_sha256"],
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


def _compact_checkpoint_workload(
    root: Path,
    checkpoint_id: str,
    workload_id: str,
    expected_task_count: int,
    expected_task_ids_sha256: str,
    expected_adapter_model_sha256: str,
) -> dict[str, Any]:
    generation = _read_json(root / "generation-metadata.json")
    run = _read_json(root / "run.json")
    summary = _read_json(root / "summary.json")
    adapter = generation.get("adapter", {})
    if (
        generation.get("schema_version") != CHECKPOINT_GENERATION_SCHEMA_VERSION
        or generation.get("checkpoint_id") != checkpoint_id
        or generation.get("workload_id") != workload_id
        or generation.get("ordered_task_ids_sha256") != expected_task_ids_sha256
        or generation.get("candidate_count") != expected_task_count * 8
        or generation.get("generation_error_count") != 0
        or generation.get("generation_sha256")
        != sha256_file(root / "generations.jsonl")
        or adapter.get("adapter_model_sha256") != expected_adapter_model_sha256
        or run.get("selected_adapter_binding") != adapter
        or summary.get("schema_version") != CHECKPOINT_VERIFICATION_SCHEMA_VERSION
        or summary.get("checkpoint_id") != checkpoint_id
        or summary.get("workload_id") != workload_id
        or summary.get("task_count") != expected_task_count
        or summary.get("candidate_count") != expected_task_count * 8
        or summary.get("complete") is not True
        or summary.get("infrastructure_error_count") != 0
    ):
        raise ValueError(
            f"generalist-v2 selection workload is incomplete: "
            f"{checkpoint_id}/{workload_id}"
        )
    verified_counts = [
        int(item["verified_candidate_count"]) for item in summary["per_task"]
    ]
    per_task = [
        {
            "task_id": str(item["task_id"]),
            "verified_candidate_count": int(item["verified_candidate_count"]),
        }
        for item in summary["per_task"]
    ]
    if len(verified_counts) != expected_task_count:
        raise ValueError("generalist-v2 checkpoint per-task outcomes are incomplete")
    return {
        "task_count": expected_task_count,
        "candidate_count": expected_task_count * 8,
        "tasks_with_verified_candidate": summary["tasks_with_verified_candidate"],
        "pass_at_k": summary["pass_at_k"],
        "category_counts": summary["category_counts"],
        "finish_reason_counts": summary["finish_reason_counts"],
        "exact_target_candidate_count": summary["exact_target_candidate_count"],
        "exact_target_task_count": summary["exact_target_task_count"],
        "verified_counts": verified_counts,
        "per_task": per_task,
        "ordered_task_ids_sha256": expected_task_ids_sha256,
        "results_sha256": sha256_file(root / "results.jsonl"),
        "generation_sha256": sha256_file(root / "generations.jsonl"),
        "adapter_model_sha256": expected_adapter_model_sha256,
    }


def compact_checkpoint_selection_evidence(
    config: GeneralistV2Config,
    q0_evidence_path: Path,
    training_run_path: Path,
    evaluation_root: Path,
    output: Path,
) -> dict[str, Any]:
    config.validate()
    q0 = _read_json(q0_evidence_path)
    training = _read_json(training_run_path)
    if (
        q0.get("schema_version") != "generalist-v2-q0-evidence-v1"
        or q0.get("checkpoint_id") != "Q0"
        or training.get("schema_version") != "generalist-v2-full-training-v1"
        or training.get("status") != "passed"
    ):
        raise ValueError("generalist-v2 selection needs complete Q0 and training")
    required = {
        "fresh-composition-valid-v2": 406,
        "minif2f-valid-clean-v2": 244,
        "riemann-fresh-valid-v2": 100,
    }
    evaluations: dict[str, CheckpointValidation] = {
        "Q0": CheckpointValidation(
            fresh_composition_verified_counts=tuple(
                q0["workloads"]["fresh-composition-valid-v2"]["verified_counts"]
            ),
            minif2f_verified_counts=tuple(
                q0["workloads"]["minif2f-valid-clean-v2"]["verified_counts"]
            ),
        )
    }
    checkpoints: dict[str, Any] = {}
    for checkpoint_id in ("Q1", "Q2", "Q3", "Q4"):
        trained_checkpoint = training["checkpoints"][checkpoint_id]
        checkpoint_workloads: dict[str, Any] = {}
        for workload_id, task_count in required.items():
            checkpoint_workloads[workload_id] = _compact_checkpoint_workload(
                evaluation_root / checkpoint_id / workload_id,
                checkpoint_id,
                workload_id,
                task_count,
                str(q0["workloads"][workload_id]["ordered_task_ids_sha256"]),
                str(trained_checkpoint["adapter_model_sha256"]),
            )
        if checkpoint_id in {"Q2", "Q4"}:
            workload_id = "dataset-v2-train-probe"
            checkpoint_workloads[workload_id] = _compact_checkpoint_workload(
                evaluation_root / checkpoint_id / workload_id,
                checkpoint_id,
                workload_id,
                256,
                str(q0["workloads"][workload_id]["ordered_task_ids_sha256"]),
                str(trained_checkpoint["adapter_model_sha256"]),
            )
        evaluations[checkpoint_id] = CheckpointValidation(
            fresh_composition_verified_counts=tuple(
                checkpoint_workloads["fresh-composition-valid-v2"]["verified_counts"]
            ),
            minif2f_verified_counts=tuple(
                checkpoint_workloads["minif2f-valid-clean-v2"]["verified_counts"]
            ),
        )
        checkpoints[checkpoint_id] = {
            "optimizer_step": trained_checkpoint["optimizer_step"],
            "adapter_model_sha256": trained_checkpoint["adapter_model_sha256"],
            "workloads": checkpoint_workloads,
        }
    selection = select_generalist_checkpoint(
        evaluations,
        resamples=int(config.evaluation["bootstrap_resamples"]),
        seed=int(config.evaluation["bootstrap_seed"]),
    )
    selected = str(selection["selected_checkpoint"])
    evidence = {
        "schema_version": "generalist-v2-checkpoint-selection-v1",
        "status": "frozen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "training_run_sha256": sha256_file(training_run_path),
        "q0_evidence_sha256": sha256_file(q0_evidence_path),
        "checkpoints": checkpoints,
        "selection": selection,
        "selected_checkpoint": {
            "checkpoint_id": selected,
            **training["checkpoints"][selected],
        },
        "test_workloads_consulted_before_freeze": False,
        "riemann_validation_used_for_selection": False,
        "training_probe_used_for_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def _compact_extended_workload(
    root: Path,
    checkpoint_id: str,
    workload_id: str,
    expected_task_count: int,
    expected_task_ids_sha256: str,
    expected_adapter_model_sha256: str,
) -> dict[str, Any]:
    generation = _read_json(root / "generation-metadata.json")
    run = _read_json(root / "run.json")
    summary = _read_json(root / "summary.json")
    adapter = generation.get("adapter")
    adapter_matches = (
        isinstance(adapter, dict)
        and adapter.get("adapter_model_sha256") == expected_adapter_model_sha256
    )
    required_pass_metrics = {f"pass@{k}" for k in EXTENDED_SEARCH_KS}
    required_solved_metrics = {f"solved@{k}" for k in EXTENDED_SEARCH_KS}
    if (
        generation.get("schema_version") != EXTENDED_GENERATION_SCHEMA_VERSION
        or generation.get("evaluation_profile") != "extended-search-budget-v1"
        or generation.get("checkpoint_id") != checkpoint_id
        or generation.get("workload_id") != workload_id
        or generation.get("ordered_task_ids_sha256") != expected_task_ids_sha256
        or generation.get("candidate_count") != expected_task_count * 64
        or generation.get("generation_error_count") != 0
        or generation.get("generate_all_candidates_without_early_stop") is not True
        or not adapter_matches
        or run.get("selected_adapter_binding") != adapter
        or run.get("candidates_per_task") != 64
        or summary.get("schema_version") != EXTENDED_VERIFICATION_SCHEMA_VERSION
        or summary.get("evaluation_profile") != "extended-search-budget-v1"
        or summary.get("checkpoint_id") != checkpoint_id
        or summary.get("workload_id") != workload_id
        or summary.get("task_count") != expected_task_count
        or summary.get("candidate_count") != expected_task_count * 64
        or summary.get("complete") is not True
        or summary.get("infrastructure_error_count") != 0
        or summary.get("all_candidates_verified_without_early_stop") is not True
        or set(summary.get("pass_at_k", {})) != required_pass_metrics
        or set(summary.get("tasks_solved_within_k", {})) != required_solved_metrics
    ):
        raise ValueError(
            "generalist-v2 extended workload is incomplete: "
            f"{checkpoint_id}/{workload_id}"
        )
    pass_at_k = summary["pass_at_k"]
    solved_within_k = summary["tasks_solved_within_k"]
    verified_counts = [
        int(item["verified_candidate_count"]) for item in summary["per_task"]
    ]
    if len(verified_counts) != expected_task_count or any(
        count < 0 or count > 64 for count in verified_counts
    ):
        raise ValueError("generalist-v2 extended per-task outcomes are incomplete")
    raw_path = root / "raw-candidates.jsonl.gz"
    density = _summarize_extended_raw_candidate_evidence(
        raw_path,
        checkpoint_id=checkpoint_id,
        workload_id=workload_id,
        expected_task_ids=[str(item["task_id"]) for item in summary["per_task"]],
        expected_adapter_model_sha256=expected_adapter_model_sha256,
        sampling_seed=int(generation["sampling"]["seed"]),
    )
    raw_evidence = {
        "artifact": "raw-candidates.jsonl.gz",
        "sha256": sha256_file(raw_path),
        **density,
    }
    raw_pass_at_k = {
        f"pass@{k}": fmean(
            estimate_pass_at_k(64, int(item["verified_candidate_count"]), k)
            for item in density["per_task"]
        )
        for k in EXTENDED_SEARCH_KS
    }
    raw_solved_within_k = {
        f"solved@{k}": sum(
            any(index < k for index in item["verified_candidate_indices"])
            for item in density["per_task"]
        )
        for k in EXTENDED_SEARCH_KS
    }
    if (
        summary.get("extended_candidate_evidence") != raw_evidence
        or pass_at_k != raw_pass_at_k
        or solved_within_k != raw_solved_within_k
        or verified_counts
        != [int(item["verified_candidate_count"]) for item in density["per_task"]]
        or summary["category_counts"]["verified"] != density["verified_candidate_count"]
    ):
        raise ValueError("extended aggregate evidence differs from raw candidates")
    return {
        "task_count": expected_task_count,
        "candidate_count": expected_task_count * 64,
        "pass_at_k": pass_at_k,
        "marginal_pass_at_k": {
            "delta_8_to_16": pass_at_k["pass@16"] - pass_at_k["pass@8"],
            "delta_16_to_32": pass_at_k["pass@32"] - pass_at_k["pass@16"],
            "delta_32_to_64": pass_at_k["pass@64"] - pass_at_k["pass@32"],
        },
        "tasks_solved_within_k": solved_within_k,
        "marginal_tasks_solved": {
            "delta_8_to_16": solved_within_k["solved@16"] - solved_within_k["solved@8"],
            "delta_16_to_32": solved_within_k["solved@32"]
            - solved_within_k["solved@16"],
            "delta_32_to_64": solved_within_k["solved@64"]
            - solved_within_k["solved@32"],
        },
        "verified_counts": verified_counts,
        "category_counts": summary["category_counts"],
        "finish_reason_counts": summary["finish_reason_counts"],
        "ordered_task_ids_sha256": expected_task_ids_sha256,
        "results_sha256": sha256_file(root / "results.jsonl"),
        "generation_sha256": sha256_file(root / "generations.jsonl"),
        "raw_candidate_evidence": raw_evidence,
        "adapter_model_sha256": expected_adapter_model_sha256,
    }


def compact_extended_validation_evidence(
    config: GeneralistV2Config,
    screening_selection_path: Path,
    evaluation_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Compact n=64 validation evidence for the already-frozen checkpoint."""

    config.validate()
    screening = _read_json(screening_selection_path)
    selection = screening.get("selection", {})
    selected = str(selection.get("selected_checkpoint", ""))
    if (
        screening.get("schema_version") != "generalist-v2-checkpoint-selection-v1"
        or screening.get("status") != "frozen"
        or selected not in {"Q1", "Q2", "Q3", "Q4"}
    ):
        raise ValueError("extended validation needs complete n=8 screening evidence")

    workload_counts = {
        "fresh-composition-valid-v2": 406,
        "minif2f-valid-clean-v2": 244,
    }
    checkpoint_workloads: dict[str, Any] = {}
    for workload_id, task_count in workload_counts.items():
        screening_workload = screening["checkpoints"][selected]["workloads"][
            workload_id
        ]
        checkpoint_workloads[workload_id] = _compact_extended_workload(
            evaluation_root / selected / workload_id,
            selected,
            workload_id,
            task_count,
            str(screening_workload["ordered_task_ids_sha256"]),
            str(screening["checkpoints"][selected]["adapter_model_sha256"]),
        )

    evidence = {
        "schema_version": "generalist-v2-extended-validation-v1",
        "status": "selected-checkpoint-extended-validation-complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "screening_selection_sha256": sha256_file(screening_selection_path),
        "candidates_per_task": 64,
        "reported_k": list(EXTENDED_SEARCH_KS),
        "screening_selected_checkpoint": selected,
        "evaluated_checkpoint": {
            "checkpoint_id": selected,
            "adapter_model_sha256": screening["checkpoints"][selected][
                "adapter_model_sha256"
            ],
            "workloads": checkpoint_workloads,
        },
        "base_control_evaluated_at_n64": False,
        "runner_up_evaluated_at_n64": False,
        "test_workloads_evaluated_at_n64": False,
        "test_workloads_consulted_before_checkpoint_freeze": False,
        "riemann_used_for_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


FINAL_ASSESSMENT_WORKLOADS = {
    "minif2f-valid-clean-v2": 244,
    "minif2f-test-clean-v2": 244,
    "fresh-composition-valid-v2": 406,
    "fresh-composition-test-v2": 415,
    "riemann-fresh-valid-v2": 100,
    "riemann-fresh-test-v2": 104,
}


def _compact_final_run(
    root: Path,
    *,
    model_label: str,
    selected_checkpoint: str,
    workload_id: str,
    expected_task_count: int,
) -> dict[str, Any]:
    generation = _read_json(root / "generation-metadata.json")
    run = _read_json(root / "run.json")
    summary = _read_json(root / "summary.json")
    expected_model = (
        (DEEPSEEK_MODEL_ID, DEEPSEEK_MODEL_REVISION)
        if model_label == "deepseek"
        else (MODEL_ID, MODEL_REVISION)
    )
    adapter = generation.get("adapter")
    if (
        generation.get("schema_version") != FINAL_GENERATION_SCHEMA_VERSION
        or generation.get("evaluation_profile") != "postselection-final-n8"
        or generation.get("model_label") != model_label
        or generation.get("selected_checkpoint") != selected_checkpoint
        or generation.get("workload_id") != workload_id
        or generation.get("model_id") != expected_model[0]
        or generation.get("model_revision") != expected_model[1]
        or generation.get("task_count") != expected_task_count
        or generation.get("candidate_count") != expected_task_count * 8
        or generation.get("generation_error_count") != 0
        or generation.get("generation_sha256")
        != sha256_file(root / "generations.jsonl")
        or generation.get("first_complete_result_overwrite_protected") is not True
        or generation.get("final_only_workload")
        != (workload_id in FINAL_TEST_WORKLOADS)
        or ((model_label == "selected") != isinstance(adapter, dict))
        or run.get("model_id") != expected_model[0]
        or run.get("model_revision") != expected_model[1]
        or run.get("selected_adapter_binding") != adapter
        or run.get("candidates_per_task") != 8
        or summary.get("schema_version") != FINAL_VERIFICATION_SCHEMA_VERSION
        or summary.get("evaluation_profile") != "postselection-final-n8"
        or summary.get("model_label") != model_label
        or summary.get("checkpoint_id") != selected_checkpoint
        or summary.get("workload_id") != workload_id
        or summary.get("task_count") != expected_task_count
        or summary.get("candidate_count") != expected_task_count * 8
        or summary.get("complete") is not True
        or summary.get("infrastructure_error_count") != 0
    ):
        raise ValueError(
            f"final assessment lane is incomplete: {model_label}/{workload_id}"
        )
    per_task = [
        {
            "task_id": str(item["task_id"]),
            "verified_candidate_count": int(item["verified_candidate_count"]),
        }
        for item in summary["per_task"]
    ]
    if len(per_task) != expected_task_count or [
        item["task_id"] for item in per_task
    ] != list(dict.fromkeys(item["task_id"] for item in per_task)):
        raise ValueError("final assessment per-task outcomes are incomplete")
    return {
        "source": "postselection-final-n8",
        "model_id": expected_model[0],
        "model_revision": expected_model[1],
        "adapter": adapter,
        "task_count": expected_task_count,
        "candidate_count": expected_task_count * 8,
        "tasks_with_verified_candidate": summary["tasks_with_verified_candidate"],
        "pass_at_k": summary["pass_at_k"],
        "category_counts": summary["category_counts"],
        "finish_reason_counts": summary["finish_reason_counts"],
        "per_task": per_task,
        "ordered_task_ids_sha256": generation["ordered_task_ids_sha256"],
        "generation_sha256": sha256_file(root / "generations.jsonl"),
        "results_sha256": sha256_file(root / "results.jsonl"),
        "timing_seconds": run["runtime"],
        "first_complete_result_overwrite_protected": True,
    }


def _compact_reused_screening(
    value: dict[str, Any], *, model_label: str, selected_checkpoint: str
) -> dict[str, Any]:
    per_task = value.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != int(value["task_count"]):
        raise ValueError("screening evidence lacks per-task final outcomes")
    return {
        "source": "preselection-screening-n8-reused-after-freeze",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter": (
            None
            if model_label == "base"
            else {
                "checkpoint_id": selected_checkpoint,
                "adapter_model_sha256": value["adapter_model_sha256"],
            }
        ),
        "task_count": value["task_count"],
        "candidate_count": value["candidate_count"],
        "tasks_with_verified_candidate": value["tasks_with_verified_candidate"],
        "pass_at_k": value["pass_at_k"],
        "category_counts": value["category_counts"],
        "finish_reason_counts": value["finish_reason_counts"],
        "per_task": per_task,
        "ordered_task_ids_sha256": value["ordered_task_ids_sha256"],
        "generation_sha256": value["generation_sha256"],
        "results_sha256": value["results_sha256"],
        "first_complete_result_overwrite_protected": None,
    }


def _task_breakdowns(
    per_task: list[dict[str, Any]], task_metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts = {
        str(item["task_id"]): int(item["verified_candidate_count"]) for item in per_task
    }
    dimensions: dict[str, dict[str, list[int]]] = {
        "structural_class": defaultdict(list),
        "generator_family": defaultdict(list),
        "topic_tag": defaultdict(list),
    }
    for task_id, count in counts.items():
        metadata = task_metadata.get(task_id)
        if not metadata:
            continue
        for dimension in ("structural_class", "generator_family"):
            value = metadata.get(dimension)
            if value is not None:
                dimensions[dimension][str(value)].append(count)
        for value in metadata.get("topic_tags", []):
            dimensions["topic_tag"][str(value)].append(count)
    output: dict[str, Any] = {}
    for dimension, groups in dimensions.items():
        if not groups:
            continue
        output[dimension] = {
            key: {
                "task_count": len(values),
                "solved_within_8": sum(value > 0 for value in values),
                "pass_at_k": {
                    f"pass@{k}": fmean(
                        estimate_pass_at_k(8, value, k) for value in values
                    )
                    for k in (1, 4, 8)
                },
            }
            for key, values in sorted(groups.items())
        }
    return output


def compact_final_assessment_evidence(
    config: GeneralistV2Config,
    q0_evidence_path: Path,
    selection_path: Path,
    final_root: Path,
    package_root: Path,
    view_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Compact frozen n=8 Base/generalist/DeepSeek comparisons."""

    config.validate()
    q0 = _read_json(q0_evidence_path)
    selection = _read_json(selection_path)
    selected_checkpoint = str(
        selection.get("selection", {}).get("selected_checkpoint", "")
    )
    if (
        q0.get("schema_version") != "generalist-v2-q0-evidence-v1"
        or selection.get("schema_version") != "generalist-v2-checkpoint-selection-v1"
        or selection.get("status") != "frozen"
        or selected_checkpoint not in {"Q1", "Q2", "Q3", "Q4"}
    ):
        raise ValueError("final assessment requires frozen Q0 and checkpoint evidence")

    validation_workloads = {
        "minif2f-valid-clean-v2",
        "fresh-composition-valid-v2",
        "riemann-fresh-valid-v2",
    }
    workloads: dict[str, Any] = {}
    for workload_id, task_count in FINAL_ASSESSMENT_WORKLOADS.items():
        lanes: dict[str, dict[str, Any]] = {}
        if workload_id in validation_workloads:
            lanes["base"] = _compact_reused_screening(
                q0["workloads"][workload_id],
                model_label="base",
                selected_checkpoint=selected_checkpoint,
            )
            lanes["selected"] = _compact_reused_screening(
                selection["checkpoints"][selected_checkpoint]["workloads"][workload_id],
                model_label="selected",
                selected_checkpoint=selected_checkpoint,
            )
        else:
            for model_label in ("base", "selected"):
                lanes[model_label] = _compact_final_run(
                    final_root / model_label / workload_id,
                    model_label=model_label,
                    selected_checkpoint=selected_checkpoint,
                    workload_id=workload_id,
                    expected_task_count=task_count,
                )
        lanes["deepseek"] = _compact_final_run(
            final_root / "deepseek" / workload_id,
            model_label="deepseek",
            selected_checkpoint=selected_checkpoint,
            workload_id=workload_id,
            expected_task_count=task_count,
        )
        identities = {str(lane["ordered_task_ids_sha256"]) for lane in lanes.values()}
        if len(identities) != 1:
            raise ValueError(f"final task identity differs across {workload_id}")
        task_ids = [str(item["task_id"]) for item in lanes["base"]["per_task"]]
        for lane in lanes.values():
            if [str(item["task_id"]) for item in lane["per_task"]] != task_ids:
                raise ValueError(f"final task order differs across {workload_id}")
        counts = {
            label: [int(item["verified_candidate_count"]) for item in lane["per_task"]]
            for label, lane in lanes.items()
        }
        _, _, _, task_metadata = materialize_q0_workload(
            workload_id, package_root, view_dir
        )
        for lane in lanes.values():
            lane["breakdowns"] = _task_breakdowns(lane["per_task"], task_metadata)
        paired = {
            "selected_vs_base": compare_paired_verified_counts(
                counts["base"],
                counts["selected"],
                resamples=int(config.evaluation["bootstrap_resamples"]),
                seed=int(config.evaluation["bootstrap_seed"]),
            ),
            "deepseek_vs_base": compare_paired_verified_counts(
                counts["base"],
                counts["deepseek"],
                resamples=int(config.evaluation["bootstrap_resamples"]),
                seed=int(config.evaluation["bootstrap_seed"]),
            ),
            "selected_vs_deepseek": compare_paired_verified_counts(
                counts["deepseek"],
                counts["selected"],
                resamples=int(config.evaluation["bootstrap_resamples"]),
                seed=int(config.evaluation["bootstrap_seed"]),
            ),
        }
        gap_closure: dict[str, float | None] = {}
        for key in ("pass@1", "pass@4", "pass@8"):
            base = float(lanes["base"]["pass_at_k"][key])
            selected = float(lanes["selected"]["pass_at_k"][key])
            deepseek = float(lanes["deepseek"]["pass_at_k"][key])
            denominator = deepseek - base
            gap_closure[key] = (
                None if denominator == 0.0 else (selected - base) / denominator
            )
        workloads[workload_id] = {
            "task_count": task_count,
            "candidates_per_task": 8,
            "models": lanes,
            "paired_comparisons": paired,
            "deepseek_gap_closed_fraction": gap_closure,
            "per_task_solved_outcomes": [
                {
                    "task_id": task_id,
                    **{
                        label: count_values[index] > 0
                        for label, count_values in counts.items()
                    },
                }
                for index, task_id in enumerate(task_ids)
            ],
        }

    evidence = {
        "schema_version": "generalist-v2-final-assessment-v1",
        "status": "complete",
        "selected_checkpoint": selected_checkpoint,
        "selected_adapter_model_sha256": selection["selected_checkpoint"][
            "adapter_model_sha256"
        ],
        "q0_evidence_sha256": sha256_file(q0_evidence_path),
        "selection_evidence_sha256": sha256_file(selection_path),
        "prompt_format_id": "lean-sft-v2-raw-whole-proof",
        "candidates_per_task": 8,
        "models": {
            "base": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION},
            "selected": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "checkpoint_id": selected_checkpoint,
                "adapter_model_sha256": selection["selected_checkpoint"][
                    "adapter_model_sha256"
                ],
            },
            "deepseek": {
                "model_id": DEEPSEEK_MODEL_ID,
                "model_revision": DEEPSEEK_MODEL_REVISION,
            },
        },
        "workloads": workloads,
        "checkpoint_selection_frozen_before_test": True,
        "test_workloads_first_used_after_checkpoint_freeze": True,
        "test_workloads_regenerated": False,
        "riemann_validation_used_for_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def run_historical_riemann_assessment(
    config: GeneralistV2Config,
    selection_path: Path,
    riemann_config_path: Path,
    repository_root: Path,
    domain_config_path: Path,
    mathlib_root: Path,
    preflight_path: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Run the frozen 556x4 historical lane with the selected adapter."""

    from .riemann_assessment import run_assessment, validate_assessment_config

    config.validate()
    selection = _read_json(selection_path)
    selected = str(selection.get("selection", {}).get("selected_checkpoint", ""))
    if selection.get("status") != "frozen" or selected not in {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("historical Riemann assessment needs a frozen checkpoint")
    adapter = _checkpoint_adapter_spec(config, selected, adapter_dir)
    if (
        sha256_file(adapter_dir / "adapter_model.safetensors")
        != selection["selected_checkpoint"]["adapter_model_sha256"]
    ):
        raise ValueError("historical Riemann selected adapter hash differs")
    if (output_dir / "run.json").exists():
        raise FileExistsError(
            f"refusing to overwrite completed historical Riemann run: {output_dir}"
        )
    riemann_config = Phase1Config.load(riemann_config_path)
    validate_assessment_config(riemann_config)
    _, _, summary = run_assessment(
        riemann_config,
        repository_root,
        domain_config_path,
        mathlib_root,
        preflight_path,
        output_dir,
        verification_workers=workers,
        adapter=adapter,
    )
    if not summary["complete"] or summary["infrastructure_error_count"]:
        raise RuntimeError("historical Riemann selected-adapter run is incomplete")
    return summary


def _paired_solved(reference: list[bool], candidate: list[bool]) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("historical paired outcomes are incomplete")
    both = sum(left and right for left, right in zip(reference, candidate, strict=True))
    candidate_only = sum(
        not left and right for left, right in zip(reference, candidate, strict=True)
    )
    reference_only = sum(
        left and not right for left, right in zip(reference, candidate, strict=True)
    )
    neither = len(reference) - both - candidate_only - reference_only
    discordant = candidate_only + reference_only
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(candidate_only, reference_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "both_solved": both,
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "neither_solved": neither,
        "paired_wins_candidate_minus_reference": candidate_only - reference_only,
        "exact_two_sided_mcnemar_p": p_value,
    }


def _historical_breakdowns(
    counts: dict[str, int], metadata_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    dimensions = (
        "primary_domain",
        "relevance_distance",
        "inclusion_scope",
        "component_inclusion_basis",
    )
    output: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[int]] = defaultdict(list)
        for row in metadata_rows:
            groups[str(row[dimension])].append(counts[str(row["task_id"])])
        output[dimension] = {
            key: {
                "task_count": len(values),
                "solved_within_4": sum(value > 0 for value in values),
                "pass_at_k": {
                    f"pass@{k}": fmean(
                        estimate_pass_at_k(4, value, k) for value in values
                    )
                    for k in (1, 4)
                },
            }
            for key, values in sorted(groups.items())
        }
    return output


def compact_historical_riemann_evidence(
    selection_path: Path,
    artifact_dir: Path,
    base_evidence_dir: Path,
    deepseek_evidence_dir: Path,
    output: Path,
) -> dict[str, Any]:
    selection = _read_json(selection_path)
    selected = str(selection.get("selection", {}).get("selected_checkpoint", ""))
    run = _read_json(artifact_dir / "run.json")
    summary = _read_json(artifact_dir / "summary.json")
    adapter = run.get("selected_adapter_binding")
    if (
        selection.get("status") != "frozen"
        or selected not in {"Q1", "Q2", "Q3", "Q4"}
        or run.get("model_id") != MODEL_ID
        or run.get("model_revision") != MODEL_REVISION
        or run.get("adapter_enabled") is not True
        or not isinstance(adapter, dict)
        or adapter.get("adapter_id") != f"qwen-lean-generalist-v2-{selected.lower()}"
        or sha256_file(Path(str(adapter["adapter_path"])) / "adapter_model.safetensors")
        != selection["selected_checkpoint"]["adapter_model_sha256"]
        or run.get("workload_id") != "riemann-specialist-validation-v1"
        or run.get("candidates_per_task") != 4
        or summary.get("task_count") != 556
        or summary.get("candidate_count") != 2224
        or summary.get("complete") is not True
        or summary.get("infrastructure_error_count") != 0
    ):
        raise ValueError("historical Riemann selected-adapter evidence is incomplete")
    selected_rows = [
        {
            "task_id": str(item["task_id"]),
            "verified_candidate_count": int(item["verified_candidate_count"]),
            "solved": int(item["verified_candidate_count"]) > 0,
        }
        for item in summary["per_task"]
    ]
    if len(selected_rows) != 556:
        raise ValueError("historical selected per-task outcomes are incomplete")

    anchors: dict[str, Any] = {}
    anchor_rows: dict[str, list[dict[str, Any]]] = {}
    for label, root, expected_model, expected_revision in (
        ("base", base_evidence_dir, MODEL_ID, MODEL_REVISION),
        ("deepseek", deepseek_evidence_dir, DEEPSEEK_MODEL_ID, DEEPSEEK_MODEL_REVISION),
    ):
        full = _read_json(root / "full.json")
        rows = list(_iter_jsonl(root / "task-outcomes.jsonl"))
        identity = full.get("execution_identity", {})
        if (
            full.get("status") != "passed"
            or identity.get("model_id") != expected_model
            or identity.get("model_revision") != expected_revision
            or full.get("workload", {}).get("task_count") != 556
            or full.get("workload", {}).get("candidates_per_task") != 4
            or full.get("workload", {}).get("protected_holdouts_used") is not False
            or len(rows) != 556
        ):
            raise ValueError(f"accepted historical {label} anchor differs")
        anchors[label] = {
            "model_id": expected_model,
            "model_revision": expected_revision,
            "overall": full["overall"],
            "full_evidence_sha256": sha256_file(root / "full.json"),
            "task_outcomes_sha256": sha256_file(root / "task-outcomes.jsonl"),
        }
        anchor_rows[label] = rows

    task_ids = [str(item["task_id"]) for item in selected_rows]
    if any(
        [str(item["task_id"]) for item in rows] != task_ids
        for rows in anchor_rows.values()
    ):
        raise ValueError("historical paired task identities differ")
    selected_counts = {
        str(item["task_id"]): int(item["verified_candidate_count"])
        for item in selected_rows
    }
    evidence = {
        "schema_version": "generalist-v2-historical-riemann-v1",
        "status": "complete",
        "interpretation": "historical-comparability-and-learned-knowledge-only",
        "clean_unseen_generalization": False,
        "dataset_v2_training_source_knowledge_may_overlap": True,
        "workload": {
            "id": "riemann-specialist-validation-v1",
            "task_count": 556,
            "candidates_per_task": 4,
        },
        "selected_checkpoint": selected,
        "selected_adapter_model_sha256": selection["selected_checkpoint"][
            "adapter_model_sha256"
        ],
        "selected": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "pass_at_k": summary["pass_at_k"],
            "tasks_with_verified_candidate": summary["tasks_with_verified_candidate"],
            "category_counts": summary["category_counts"],
            "finish_reason_counts": summary["finish_reason_counts"],
            "per_task": selected_rows,
            "breakdowns": _historical_breakdowns(selected_counts, anchor_rows["base"]),
            "generation_sha256": sha256_file(artifact_dir / "generation.jsonl"),
            "results_sha256": sha256_file(artifact_dir / "results.jsonl"),
        },
        "accepted_anchors_regenerated": False,
        "accepted_anchors": anchors,
        "paired_solved_within_4": {
            f"selected_vs_{label}": _paired_solved(
                [bool(item["solved"]) for item in rows],
                [bool(item["solved"]) for item in selected_rows],
            )
            for label, rows in anchor_rows.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence
