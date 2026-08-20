from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import read_artifacts, write_artifacts
from .baseline import GeneratedCandidate, _generate_candidates, _local_cuda_runtime
from .metrics import pass_at_k, summarize_results
from .minif2f import Phase1Config
from .phase2_schema import MathlibProofRecord
from .phase2_verification import run_lean_source, validate_record_source_identity
from .phase3 import render_sft_prompt
from .phase3_verification import reconstruct_generated_proof
from .prompt import PROMPT_FORMAT_ID, normalize_transport
from .qwen35_4b_base_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    VLLM_SOURCE_REVISION,
    VLLM_VERSION,
    _GpuMemorySampler,
)
from .schema import (
    PHASE1_RESULT_SCHEMA_VERSION,
    CandidateResult,
    RunMetadata,
    TaskRecord,
)


WORKLOAD_ID = "riemann-specialist-validation-v1"
EXPECTED_TASKS = 556
EXPECTED_CANDIDATES = EXPECTED_TASKS * 4
DOMAIN_SCHEMA_VERSION = "riemann-domain-breakdown-v1"
EVIDENCE_SCHEMA_VERSION = "qwen35-4b-base-riemann-assessment-v1"


def validate_assessment_config(config: Phase1Config) -> None:
    expected: tuple[tuple[str, ...], Any] = (
        (("assessment", "id"), "qwen35-4b-base-riemann-casting-v1"),
        (("assessment", "vllm_source_revision"), VLLM_SOURCE_REVISION),
        (("benchmark", "repository"), "https://github.com/leanprover-community/mathlib4"),
        (("benchmark", "revision"), "81a5d257c8e410db227a6665ed08f64fea08e997"),
        (("benchmark", "source_path"), f"data/riemann/corpora/{WORKLOAD_ID}"),
        (("benchmark", "split"), "validation"),
        (("benchmark", "expected_primary_task_count"), EXPECTED_TASKS),
        (("benchmark", "lean_toolchain"), "leanprover/lean4:v4.32.0"),
        (("model", "model_id"), MODEL_ID),
        (("model", "model_revision"), MODEL_REVISION),
        (("model", "tokenizer_id"), MODEL_ID),
        (("model", "tokenizer_revision"), MODEL_REVISION),
        (("model", "chat_template"), None),
        (("sampling", "candidates_per_task"), 4),
        (("sampling", "do_sample"), True),
        (("sampling", "temperature"), 0.8),
        (("sampling", "top_p"), 0.95),
        (("sampling", "top_k"), -1),
        (("sampling", "max_new_tokens"), 1024),
        (("sampling", "stop"), "tokenizer_eos_or_token_limit"),
        (("sampling", "seed"), 0),
        (("engine", "name"), "vllm"),
        (("engine", "version"), VLLM_VERSION),
        (("engine", "dtype"), "bfloat16"),
        (("engine", "tensor_parallel_size"), 1),
        (("engine", "quantization"), None),
        (("engine", "language_model_only"), True),
        (("engine", "resolve_pinned_snapshot"), True),
        (("engine", "use_flashinfer_sampler"), False),
        (("engine", "worker_multiproc_method"), "spawn"),
        (("verifier", "timeout_seconds"), 30.0),
        (
            ("verifier", "candidate_handling"),
            "raw_continuation_no_extraction_repair_or_feedback",
        ),
        (("verifier", "regenerate_after_lean_feedback"), False),
        (("workloads", WORKLOAD_ID, "selection"), "all"),
        (("workloads", WORKLOAD_ID, "expected_task_count"), EXPECTED_TASKS),
    )
    for path, wanted in expected:
        observed: Any = config.value
        for key in path:
            observed = observed[key]
        if observed != wanted:
            raise ValueError(
                f"Riemann assessment {'.'.join(path)} differs: "
                f"{observed!r} != {wanted!r}"
            )


def load_domain_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != DOMAIN_SCHEMA_VERSION:
        raise ValueError(f"unknown Riemann domain schema: {value.get('schema_version')}")
    rules = list(value.get("ordered_primary_domain_rules", []))
    if not rules or sum(bool(rule.get("fallback")) for rule in rules) != 1:
        raise ValueError("Riemann domains require exactly one fallback rule")
    if not rules[-1].get("fallback"):
        raise ValueError("Riemann fallback domain must be the final rule")
    ids = [str(rule["id"]) for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Riemann primary-domain identifiers must be unique")
    return value


def load_validation_workload(
    config: Phase1Config,
    repository_root: Path,
    domain_config: Mapping[str, Any],
) -> tuple[list[MathlibProofRecord], list[dict[str, Any]]]:
    validate_assessment_config(config)
    corpus_root = repository_root / str(config.benchmark["source_path"])
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "corpus_id": WORKLOAD_ID,
        "record_count": EXPECTED_TASKS,
        "role": "evaluation-target",
        "phase2_source_revision": config.benchmark["revision"],
        "membership_sha256": config.benchmark["membership_sha256"],
        "record_store_sha256": config.benchmark["record_store_sha256"],
    }
    for key, wanted in expected_manifest.items():
        if manifest.get(key) != wanted:
            raise ValueError(
                f"Riemann workload manifest {key} differs: "
                f"{manifest.get(key)!r} != {wanted!r}"
            )

    membership_path = corpus_root / str(manifest["membership"])
    record_store_path = corpus_root / str(manifest["record_store"])
    if _sha256_file(membership_path) != str(manifest["membership_sha256"]):
        raise ValueError("Riemann validation membership hash differs")
    if _sha256_file(record_store_path) != str(manifest["record_store_sha256"]):
        raise ValueError("Riemann validation record-store hash differs")

    membership = _read_jsonl(membership_path)
    member_ids = [str(item["record_id"]) for item in membership]
    if len(member_ids) != EXPECTED_TASKS or len(set(member_ids)) != EXPECTED_TASKS:
        raise ValueError("Riemann validation membership is not 556 unique tasks")
    if member_ids != sorted(member_ids):
        raise ValueError("Riemann validation membership is not deterministically sorted")
    if any(item.get("phase2_split") != "validation" for item in membership):
        raise ValueError("Riemann validation membership contains another Phase 2 split")

    wanted_ids = set(member_ids)
    raw_records: dict[str, dict[str, Any]] = {}
    with gzip.open(record_store_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            record_id = str(raw["id"])
            if record_id in wanted_ids:
                if record_id in raw_records:
                    raise ValueError(f"duplicate Riemann record: {record_id}")
                raw_records[record_id] = raw
    missing = wanted_ids - raw_records.keys()
    if missing:
        raise ValueError(f"Riemann record store is missing validation IDs: {sorted(missing)}")

    records: list[MathlibProofRecord] = []
    for member in membership:
        raw = raw_records[str(member["record_id"])]
        record = MathlibProofRecord.from_dict(raw)
        if record.split != "validation":
            raise ValueError(f"Riemann task {record.id} is not Phase 2 validation")
        if record.source_revision != str(config.benchmark["revision"]):
            raise ValueError(f"Riemann task {record.id} source revision differs")
        expected_identity = {
            "phase2_component": record.component_id,
            "phase2_fingerprint": record.statement_fingerprint,
            "file_path": record.file_path,
            "declaration_name": record.declaration_name,
        }
        for key, wanted in expected_identity.items():
            if member.get(key) != wanted:
                raise ValueError(f"Riemann task {record.id} membership {key} differs")
        if member.get("relevance_class") != raw.get("riemann", {}).get(
            "relevance_class"
        ):
            raise ValueError(f"Riemann task {record.id} relevance metadata differs")
        records.append(record)

    metadata = materialize_task_metadata(records, raw_records, domain_config)
    return records, metadata


def materialize_task_metadata(
    records: Sequence[MathlibProofRecord],
    raw_records: Mapping[str, Mapping[str, Any]],
    domain_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bubble_classes = set(domain_config["riemann_bubble_classes"])
    component_records: dict[str, list[MathlibProofRecord]] = defaultdict(list)
    for record in records:
        component_records[record.component_id].append(record)

    component_basis: dict[str, dict[str, Any]] = {}
    for component_id, members in component_records.items():
        classes = sorted(
            {
                str(raw_records[member.id]["riemann"]["relevance_class"])
                for member in members
                if raw_records[member.id]["riemann"]["relevance_class"] is not None
            }
        )
        if "core" in classes:
            basis = "component-contains-core"
        elif set(classes) & bubble_classes:
            basis = "component-contains-riemann-bubble"
        elif any(member.file_path.startswith("Mathlib/NumberTheory/") for member in members):
            basis = "component-contains-number-theory-control"
        else:
            raise ValueError(
                f"Riemann validation component has no committed inclusion basis: {component_id}"
            )
        component_basis[component_id] = {
            "component_inclusion_basis": basis,
            "component_anchor_relevance_classes": classes,
            "component_task_count": len(members),
        }

    output: list[dict[str, Any]] = []
    for record in records:
        raw = raw_records[record.id]
        relevance = raw["riemann"]["relevance_class"]
        relevance_key = "component-associated" if relevance is None else str(relevance)
        distance = domain_config["relevance_distance"][relevance_key]
        primary_domain, domain_rule = classify_primary_domain(record, domain_config)
        direct_number_theory = record.file_path.startswith("Mathlib/NumberTheory/")
        output.append(
            {
                "task_id": record.id,
                "declaration_name": record.declaration_name,
                "file_path": record.file_path,
                "phase2_component": record.component_id,
                "phase2_fingerprint": record.statement_fingerprint,
                "phase2_split": record.split,
                "primary_domain": primary_domain,
                "primary_domain_rule": domain_rule,
                "direct_relevance_class": relevance,
                "relevance_distance": distance,
                "direct_riemann_bubble": relevance in bubble_classes,
                "direct_riemann_core": relevance == "core",
                "direct_number_theory_control": direct_number_theory,
                "seed_families": list(raw["riemann"].get("seed_families", [])),
                "inclusion_scope": (
                    "direct"
                    if relevance is not None or direct_number_theory
                    else "component-associated"
                ),
                **component_basis[record.component_id],
            }
        )
    return output


def classify_primary_domain(
    record: MathlibProofRecord, domain_config: Mapping[str, Any]
) -> tuple[str, str]:
    path = record.file_path.casefold()
    declaration = record.declaration_name.casefold()
    for rule in domain_config["ordered_primary_domain_rules"]:
        if rule.get("fallback"):
            return str(rule["id"]), "fallback"
        path_match = any(
            str(fragment).casefold() in path for fragment in rule.get("path_contains", [])
        )
        declaration_match = any(
            str(fragment).casefold() in declaration
            for fragment in rule.get("declaration_contains", [])
        )
        if path_match or declaration_match:
            return str(rule["id"]), "committed-source-metadata"
    raise AssertionError("unreachable Riemann domain classification")


def run_preflight(
    config: Phase1Config,
    repository_root: Path,
    domain_config_path: Path,
    mathlib_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    domain_config = load_domain_config(domain_config_path)
    records, task_metadata = load_validation_workload(
        config, repository_root, domain_config
    )
    environment = _validate_mathlib_environment(
        config, mathlib_root, repository_root
    )
    sources = {
        record.id: validate_record_source_identity(record, mathlib_root)
        for record in records
    }

    probe_record = records[0]
    known_valid = run_lean_source(
        reconstruct_generated_proof(
            sources[probe_record.id], probe_record, probe_record.completion
        ),
        mathlib_root,
        timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
        lean_environment_root=repository_root,
    )
    placeholder = run_lean_source(
        reconstruct_generated_proof(sources[probe_record.id], probe_record, "sorry"),
        mathlib_root,
        timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
        lean_environment_root=repository_root,
    )
    if known_valid.status != "accepted":
        raise RuntimeError(
            f"Riemann known-valid verifier probe failed as {known_valid.status}: "
            f"{known_valid.diagnostic}"
        )
    if placeholder.status != "rejected":
        raise RuntimeError(
            f"Riemann placeholder verifier probe was {placeholder.status}"
        )

    try:
        import huggingface_hub
        import transformers
        import vllm
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Riemann model preflight requires the #44 runtime") from error
    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError(
            f"Riemann vLLM version differs: {vllm.__version__} != {VLLM_VERSION}"
        )
    snapshot = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    ).resolve()
    if snapshot.name != MODEL_REVISION:
        raise ValueError("Riemann model snapshot did not resolve to the pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    prompt_token_counts = [
        len(
            tokenizer.encode(
                render_sft_prompt(record), add_special_tokens=False
            )
        )
        for record in records
    ]
    maximum_context = max(prompt_token_counts) + int(config.sampling["max_new_tokens"])
    if maximum_context > int(config.engine["max_model_len"]):
        raise ValueError(
            f"Riemann maximum prompt plus generation is {maximum_context}, "
            f"above max_model_len {config.engine['max_model_len']}"
        )

    runtime = _local_cuda_runtime(config)
    evidence = {
        "schema_version": "qwen35-4b-base-riemann-preflight-v1",
        "status": "passed",
        "assessment_id": config.value["assessment"]["id"],
        "config_sha256": _sha256_file(config.path),
        "domain_config_sha256": _sha256_file(domain_config_path),
        "model": config.model,
        "model_snapshot_revision": snapshot.name,
        "runtime": {
            **runtime,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "vllm": vllm.__version__,
            "vllm_source_revision": VLLM_SOURCE_REVISION,
        },
        "workload": {
            "id": WORKLOAD_ID,
            "tasks": len(records),
            "candidates_per_task": int(config.sampling["candidates_per_task"]),
            "candidate_count": len(records)
            * int(config.sampling["candidates_per_task"]),
            "task_ids_sha256": _task_ids_digest(records),
            "membership_sha256": config.benchmark["membership_sha256"],
            "record_store_sha256": config.benchmark["record_store_sha256"],
            "protected_holdouts_loaded": False,
        },
        "prompt": {
            "prompt_format_id": PROMPT_FORMAT_ID,
            "chat_template": None,
            "raw_continuation": True,
            "prompt_token_counts": _distribution(prompt_token_counts),
            "max_prompt_plus_generation_tokens": maximum_context,
            "max_model_len": int(config.engine["max_model_len"]),
        },
        "domain_task_counts": dict(
            sorted(Counter(item["primary_domain"] for item in task_metadata).items())
        ),
        "mathlib_environment": environment,
        "source_identity": {
            "matched_records": len(sources),
            "expected_records": EXPECTED_TASKS,
            "original_source_span_reconstruction": True,
        },
        "verifier_probe": {
            "record_id": probe_record.id,
            "known_valid_candidate_category": known_valid.status,
            "known_valid_candidate_exit_code": known_valid.exit_code,
            "placeholder_candidate_category": placeholder.status,
            "placeholder_candidate_exit_code": placeholder.exit_code,
        },
    }
    _write_json(output_path, evidence)
    return evidence


def run_assessment(
    config: Phase1Config,
    repository_root: Path,
    domain_config_path: Path,
    mathlib_root: Path,
    preflight_path: Path,
    output_dir: Path,
    *,
    verification_workers: int,
    report_progress: bool = True,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    if verification_workers < 1:
        raise ValueError("Riemann verification workers must be positive")
    domain_config = load_domain_config(domain_config_path)
    records, _ = load_validation_workload(config, repository_root, domain_config)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_preflight(config, domain_config_path, preflight, records)
    environment = _validate_mathlib_environment(
        config, mathlib_root, repository_root
    )
    sources = {
        record.id: validate_record_source_identity(record, mathlib_root)
        for record in records
    }
    tasks = [
        TaskRecord(
            id=record.id,
            preamble="",
            declaration=record.declaration,
            declaration_name=record.declaration_name,
        )
        for record in records
    ]
    prompts = [render_sft_prompt(record) for record in records]

    required_environment = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    for key, value in required_environment.items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(f"{key}={existing!r} conflicts with required {value!r}")
        os.environ[key] = value

    runtime = _local_cuda_runtime(config)
    generation_started = time.perf_counter()
    with _GpuMemorySampler() as memory:
        generated, engine_version = _generate_candidates(
            config,
            tasks,
            prompts=prompts,
            sampling=config.sampling,
        )
    generation_wall = time.perf_counter() - generation_started
    if not memory.samples_mib:
        raise RuntimeError(f"Riemann GPU memory sampling failed: {memory.errors}")
    if len(generated) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Riemann generation returned {len(generated)} candidates, "
            f"expected {EXPECTED_CANDIDATES}"
        )
    _write_generation_artifact(output_dir, generated)

    record_by_id = {record.id: record for record in records}
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=verification_workers) as executor:
        results: list[CandidateResult] = []
        boundaries = {
            max(1, round(EXPECTED_CANDIDATES * fraction))
            for fraction in (0.25, 0.5, 0.75, 1.0)
        }
        inputs = [
            (
                item,
                record_by_id[item.task.id],
                sources[item.task.id],
                mathlib_root,
                repository_root,
                float(config.value["verifier"]["timeout_seconds"]),
            )
            for item in generated
        ]
        for completed, result in enumerate(
            executor.map(lambda value: _verify_candidate(*value), inputs), start=1
        ):
            results.append(result)
            if report_progress and completed in boundaries:
                print(
                    json.dumps(
                        {
                            "phase": "verification",
                            "completed_candidates": completed,
                            "total_candidates": EXPECTED_CANDIDATES,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    verification_wall = time.perf_counter() - verification_started

    summary = summarize_results(
        results,
        expected_task_ids=[record.id for record in records],
        candidates_per_task=4,
        ks=(1, 4),
    )
    tokens = _token_statistics(results)
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    summary.update(
        {
            "workload_id": WORKLOAD_ID,
            "generated_tokens": tokens,
            "generation_wall_time_seconds": generation_wall,
            "verification_wall_time_seconds": verification_wall,
            "run_wall_time_seconds": generation_wall + verification_wall,
            "throughput": {
                "generated_tokens_per_second": tokens["total"] / generation_wall,
                "candidates_per_second": len(results) / generation_wall,
            },
            "generation_efficiency_per_solved_task": {
                "generated_tokens": tokens["total"] / solved if solved else None,
                "generation_wall_time_seconds": generation_wall / solved
                if solved
                else None,
                "run_wall_time_seconds": (generation_wall + verification_wall) / solved
                if solved
                else None,
            },
        }
    )

    runtime.update(
        {
            "python": platform.python_version(),
            "generation_wall_time_seconds": generation_wall,
            "verification_wall_time_seconds": verification_wall,
            "verification_workers": verification_workers,
            "peak_gpu_memory_mib": max(memory.samples_mib),
            "peak_gpu_memory_bytes": max(memory.samples_mib) * 1024 * 1024,
            "gpu_memory_sample_count": len(memory.samples_mib),
            "gpu_memory_sampling_interval_seconds": memory.interval_seconds,
            "gpu_memory_measurement": "device memory.used sampled with nvidia-smi",
            "vllm_environment": required_environment,
            "candidate_generation_retries": 0,
            "candidate_regenerations_after_lean_feedback": 0,
        }
    )
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source=(
            f"qwen-lean:{config.benchmark['source_path']}@"
            f"{config.benchmark['membership_sha256']}"
        ),
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=str(config.benchmark["lean_toolchain"]),
        mathlib_revision=str(config.benchmark["revision"]),
        verifier_timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=WORKLOAD_ID,
        benchmark_split="validation",
        benchmark_repository=str(config.benchmark["repository"]),
        benchmark_revision=str(config.benchmark["revision"]),
        verifier_environment={
            **environment,
            "original_source_span_reconstruction": True,
            "candidate_handling": config.value["verifier"]["candidate_handling"],
            "verifier_timeout_semantics": "unsuccessful_proof_outcome",
            "membership_sha256": config.benchmark["membership_sha256"],
            "record_store_sha256": config.benchmark["record_store_sha256"],
        },
        candidates_per_task=4,
        inference_engine=str(config.engine["name"]),
        inference_engine_version=engine_version,
        adapter_enabled=False,
        generation_settings={
            **config.sampling,
            "dtype": config.engine["dtype"],
            "tensor_parallel_size": config.engine["tensor_parallel_size"],
            "max_model_len": config.engine["max_model_len"],
            "max_num_seqs": config.engine["max_num_seqs"],
            "gpu_memory_utilization": config.engine["gpu_memory_utilization"],
            "enforce_eager": config.engine["enforce_eager"],
            "quantization": None,
            "language_model_only": True,
            "model_artifact_resolution": "pinned_local_snapshot",
            "chat_template": None,
            "prompt_transformation": None,
            "proof_extraction": None,
            "semantic_repair": None,
            "lean_feedback": None,
        },
        runtime=runtime,
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
    return metadata, results, summary


def write_compact_evidence(
    config: Phase1Config,
    repository_root: Path,
    domain_config_path: Path,
    preflight_path: Path,
    artifact_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    domain_config = load_domain_config(domain_config_path)
    records, task_metadata = load_validation_workload(
        config, repository_root, domain_config
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_preflight(config, domain_config_path, preflight, records)
    metadata, results = read_artifacts(artifact_dir)
    stored_summary = json.loads(
        (artifact_dir / "summary.json").read_text(encoding="utf-8")
    )
    summary = _validate_and_recompute_run(config, records, metadata, results)
    for key in (
        "complete",
        "completeness_errors",
        "task_count",
        "candidate_count",
        "candidates_per_task",
        "tasks_with_verified_candidate",
        "pass_at_k",
        "category_counts",
        "finish_reason_counts",
        "verifier_timeout_count",
        "infrastructure_error_count",
    ):
        if stored_summary.get(key) != summary.get(key):
            raise ValueError(f"stored Riemann summary differs at {key}")
    if not summary["complete"] or summary["infrastructure_error_count"] != 0:
        raise ValueError("Riemann assessment is not complete and infrastructure-clean")

    metadata_by_id = {item["task_id"]: item for item in task_metadata}
    outcome_rows = _task_outcome_rows(records, results, metadata_by_id)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = evidence_dir / "task-outcomes.jsonl"
    _write_jsonl(outcomes_path, outcome_rows)

    domains = [
        str(rule["id"])
        for rule in domain_config["ordered_primary_domain_rules"]
    ]
    direct_relevance_values = [
        "core",
        "premise-1",
        "premise-2",
        "user-1",
        "user-2",
        "source-neighborhood",
        "number-theory-control",
        "none",
    ]
    full = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "assessment_id": config.value["assessment"]["id"],
        "config_sha256": _sha256_file(config.path),
        "domain_config": {
            "schema_version": DOMAIN_SCHEMA_VERSION,
            "sha256": _sha256_file(domain_config_path),
            "classification_input": domain_config["classification_input"],
            "model_outputs_used": False,
        },
        "workload": {
            "id": WORKLOAD_ID,
            "task_count": EXPECTED_TASKS,
            "candidate_count": EXPECTED_CANDIDATES,
            "candidates_per_task": 4,
            "task_ids_sha256": _task_ids_digest(records),
            "membership_sha256": config.benchmark["membership_sha256"],
            "record_store_sha256": config.benchmark["record_store_sha256"],
            "split": "validation",
            "protected_holdouts_used": False,
        },
        "execution_identity": {
            "model_id": metadata.model_id,
            "model_revision": metadata.model_revision,
            "tokenizer_id": metadata.tokenizer_id,
            "tokenizer_revision": metadata.tokenizer_revision,
            "prompt_format_id": metadata.prompt_format_id,
            "generation_settings": metadata.generation_settings,
            "lean_toolchain": metadata.lean_toolchain,
            "mathlib_revision": metadata.mathlib_revision,
            "verifier_timeout_seconds": metadata.verifier_timeout_seconds,
            "verifier_timeout_semantics": "unsuccessful_proof_outcome",
            "inference_engine": metadata.inference_engine,
            "inference_engine_version": metadata.inference_engine_version,
            "vllm_source_revision": VLLM_SOURCE_REVISION,
        },
        "overall": {
            "verified_candidates": summary["category_counts"]["verified"],
            "solved_tasks": summary["tasks_with_verified_candidate"]["count"],
            "pass_at_k": summary["pass_at_k"],
            "category_counts": summary["category_counts"],
            "finish_reason_counts": summary["finish_reason_counts"],
            "verifier_timeout_count": summary["verifier_timeout_count"],
            "infrastructure_error_count": summary["infrastructure_error_count"],
        },
        "generated_tokens": _token_statistics(results),
        "timing_seconds": {
            "generation_wall": metadata.runtime["generation_wall_time_seconds"],
            "verification_wall": metadata.runtime["verification_wall_time_seconds"],
            "run_wall": float(metadata.runtime["generation_wall_time_seconds"])
            + float(metadata.runtime["verification_wall_time_seconds"]),
        },
        "throughput": stored_summary["throughput"],
        "generation_efficiency_per_solved_task": stored_summary[
            "generation_efficiency_per_solved_task"
        ],
        "runtime": {
            key: metadata.runtime[key]
            for key in (
                "python",
                "torch",
                "torch_cuda_version",
                "package_versions",
                "inference_execution",
                "cuda_device",
                "cuda_device_capability",
                "cuda_device_total_memory_bytes",
                "sampling_backend",
                "peak_gpu_memory_mib",
                "peak_gpu_memory_bytes",
                "gpu_memory_sample_count",
                "gpu_memory_sampling_interval_seconds",
                "gpu_memory_measurement",
                "vllm_environment",
                "candidate_generation_retries",
                "candidate_regenerations_after_lean_feedback",
            )
        },
        "domain_breakdown": _breakdown(
            outcome_rows, "primary_domain", domains
        ),
        "direct_relevance_breakdown": _breakdown(
            outcome_rows,
            "direct_relevance_class",
            direct_relevance_values,
            none_label="none",
        ),
        "relevance_distance_breakdown": _breakdown(
            outcome_rows, "relevance_distance", None
        ),
        "component_inclusion_breakdown": _breakdown(
            outcome_rows, "component_inclusion_basis", None
        ),
        "inclusion_scope_breakdown": _breakdown(
            outcome_rows, "inclusion_scope", ["direct", "component-associated"]
        ),
        "task_outcomes": {
            "path": "task-outcomes.jsonl",
            "rows": len(outcome_rows),
            "sha256": _sha256_file(outcomes_path),
            "paired_analysis_fields": [
                "task_id",
                "solved",
                "verified_candidate_count",
                "verified_candidate_indices",
            ],
        },
        "generation_identity_sha256": _generation_digest(results),
    }
    compact_preflight = {
        "schema_version": preflight["schema_version"],
        "status": preflight["status"],
        "assessment_id": preflight["assessment_id"],
        "config_sha256": preflight["config_sha256"],
        "domain_config_sha256": preflight["domain_config_sha256"],
        "model": preflight["model"],
        "model_snapshot_revision": preflight["model_snapshot_revision"],
        "workload": preflight["workload"],
        "prompt": preflight["prompt"],
        "domain_task_counts": preflight["domain_task_counts"],
        "mathlib_environment": preflight["mathlib_environment"],
        "source_identity": preflight["source_identity"],
        "verifier_probe": preflight["verifier_probe"],
        "runtime": preflight["runtime"],
    }
    _write_json(evidence_dir / "preflight.json", compact_preflight)
    _write_json(evidence_dir / "full.json", full)
    (evidence_dir / "README.md").write_text(
        _readme(full), encoding="utf-8"
    )
    return full


def _validate_mathlib_environment(
    config: Phase1Config,
    mathlib_root: Path,
    lean_environment_root: Path,
) -> dict[str, Any]:
    revision = _git_output(mathlib_root, "rev-parse", "HEAD")
    if revision != str(config.benchmark["revision"]):
        raise ValueError(
            f"Riemann mathlib revision differs: {revision} != "
            f"{config.benchmark['revision']}"
        )
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "Mathlib"],
        cwd=mathlib_root,
        check=False,
    )
    if diff.returncode != 0:
        raise ValueError("Riemann mathlib source differs from its pinned revision")
    toolchain = (mathlib_root / "lean-toolchain").read_text(encoding="utf-8").strip()
    if toolchain != str(config.benchmark["lean_toolchain"]):
        raise ValueError("Riemann Lean toolchain differs from the frozen workload")
    manifest = json.loads(
        (lean_environment_root / "lake-manifest.json").read_text(encoding="utf-8")
    )
    dependency = next(
        package for package in manifest["packages"] if package["name"] == "mathlib"
    )
    if dependency["rev"] != revision:
        raise ValueError("Riemann Lake environment mathlib revision differs")
    installed_mathlib = lean_environment_root / ".lake/packages/mathlib"
    if _git_output(installed_mathlib, "rev-parse", "HEAD") != revision:
        raise ValueError("Riemann installed Lake mathlib revision differs")
    lean_version = subprocess.run(
        ["lake", "env", "lean", "--version"],
        cwd=lean_environment_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {
        "repository": config.benchmark["repository"],
        "revision": revision,
        "lean_toolchain": toolchain,
        "lean_version": lean_version,
        "lean_environment_root_revision": revision,
        "source_worktree_clean": True,
    }


def _validate_preflight(
    config: Phase1Config,
    domain_config_path: Path,
    preflight: Mapping[str, Any],
    records: Sequence[MathlibProofRecord],
) -> None:
    expected = {
        "schema_version": "qwen35-4b-base-riemann-preflight-v1",
        "status": "passed",
        "assessment_id": config.value["assessment"]["id"],
        "config_sha256": _sha256_file(config.path),
        "domain_config_sha256": _sha256_file(domain_config_path),
        "model": config.model,
        "model_snapshot_revision": MODEL_REVISION,
    }
    for key, wanted in expected.items():
        if preflight.get(key) != wanted:
            raise ValueError(f"Riemann preflight {key} differs")
    workload = preflight.get("workload", {})
    if workload.get("id") != WORKLOAD_ID or workload.get("tasks") != EXPECTED_TASKS:
        raise ValueError("Riemann preflight workload differs")
    if workload.get("candidate_count") != EXPECTED_CANDIDATES:
        raise ValueError("Riemann preflight candidate budget differs")
    if workload.get("task_ids_sha256") != _task_ids_digest(records):
        raise ValueError("Riemann preflight task IDs differ")
    if workload.get("protected_holdouts_loaded") is not False:
        raise ValueError("Riemann preflight touched a protected holdout")
    source = preflight.get("source_identity", {})
    if source.get("matched_records") != EXPECTED_TASKS:
        raise ValueError("Riemann preflight did not match every source record")
    probe = preflight.get("verifier_probe", {})
    if probe.get("known_valid_candidate_category") != "accepted":
        raise ValueError("Riemann preflight known-valid probe did not pass")
    if probe.get("placeholder_candidate_category") != "rejected":
        raise ValueError("Riemann preflight placeholder probe was not rejected")
    runtime = preflight.get("runtime", {})
    if runtime.get("inference_execution") != "local_cuda":
        raise ValueError("Riemann preflight did not use local CUDA")
    if runtime.get("vllm") != VLLM_VERSION:
        raise ValueError("Riemann preflight vLLM differs")


def _validate_and_recompute_run(
    config: Phase1Config,
    records: Sequence[MathlibProofRecord],
    metadata: RunMetadata,
    results: Sequence[CandidateResult],
) -> dict[str, Any]:
    expected = {
        "schema_version": PHASE1_RESULT_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "workload_id": WORKLOAD_ID,
        "prompt_format_id": PROMPT_FORMAT_ID,
        "lean_toolchain": config.benchmark["lean_toolchain"],
        "mathlib_revision": config.benchmark["revision"],
        "verifier_timeout_seconds": config.value["verifier"]["timeout_seconds"],
        "candidates_per_task": 4,
        "inference_engine": "vllm",
        "inference_engine_version": VLLM_VERSION,
        "adapter_enabled": False,
        "benchmark_split": "validation",
    }
    for key, wanted in expected.items():
        if getattr(metadata, key) != wanted:
            raise ValueError(f"Riemann run metadata {key} differs")
    settings = metadata.generation_settings or {}
    required_settings = {
        **config.sampling,
        "dtype": "bfloat16",
        "quantization": None,
        "language_model_only": True,
        "chat_template": None,
        "prompt_transformation": None,
        "proof_extraction": None,
        "semantic_repair": None,
        "lean_feedback": None,
    }
    for key, wanted in required_settings.items():
        if settings.get(key) != wanted:
            raise ValueError(f"Riemann generation setting {key} differs")
    if metadata.runtime.get("inference_execution") != "local_cuda":
        raise ValueError("Riemann generation did not execute on local CUDA")
    if metadata.runtime.get("candidate_regenerations_after_lean_feedback") != 0:
        raise ValueError("Riemann run regenerated after Lean feedback")
    if metadata.runtime.get("candidate_generation_retries") != 0:
        raise ValueError("Riemann run retried candidate generation")
    return summarize_results(
        results,
        expected_task_ids=[record.id for record in records],
        candidates_per_task=4,
        ks=(1, 4),
    )


def _verify_candidate(
    generated: GeneratedCandidate,
    record: MathlibProofRecord,
    source: str,
    mathlib_root: Path,
    lean_environment_root: Path,
    timeout_seconds: float,
) -> CandidateResult:
    if generated.generation_error is not None:
        return _candidate_result(
            generated,
            record,
            category="generation_error",
            exit_code=None,
            diagnostic=generated.generation_error,
            verification_latency=None,
        )
    if not normalize_transport(generated.text):
        return _candidate_result(
            generated,
            record,
            category="empty_candidate",
            exit_code=None,
            diagnostic="empty generated continuation",
            verification_latency=0.0,
        )
    try:
        reconstructed = reconstruct_generated_proof(source, record, generated.text)
        check = run_lean_source(
            reconstructed,
            mathlib_root,
            timeout_seconds=timeout_seconds,
            lean_environment_root=lean_environment_root,
        )
    except Exception as error:  # noqa: BLE001 - isolate one verifier failure.
        return _candidate_result(
            generated,
            record,
            category="verifier_error",
            exit_code=None,
            diagnostic=f"{type(error).__name__}: {error}",
            verification_latency=None,
        )
    category = {
        "accepted": "verified",
        "rejected": "lean_rejected",
        "timeout": "verifier_timeout",
        "infrastructure_error": "verifier_error",
    }[check.status]
    return _candidate_result(
        generated,
        record,
        category=category,
        exit_code=check.exit_code,
        diagnostic=check.diagnostic,
        verification_latency=check.latency_seconds,
    )


def _candidate_result(
    generated: GeneratedCandidate,
    record: MathlibProofRecord,
    *,
    category: str,
    exit_code: int | None,
    diagnostic: str,
    verification_latency: float | None,
) -> CandidateResult:
    return CandidateResult(
        task_id=record.id,
        candidate_id=f"model-{generated.candidate_index}",
        candidate_index=generated.candidate_index,
        candidate_text=generated.text,
        category=category,  # type: ignore[arg-type]
        lean_exit_code=exit_code,
        diagnostics={"stdout": "", "stderr": diagnostic},
        generation_latency_seconds=generated.generation_latency_seconds,
        verification_latency_seconds=verification_latency,
        total_latency_seconds=(
            generated.generation_latency_seconds
            + (0.0 if verification_latency is None else verification_latency)
        ),
        generated_token_count=generated.token_count,
        finish_reason=generated.finish_reason,
    )


def _task_outcome_rows(
    records: Sequence[MathlibProofRecord],
    results: Sequence[CandidateResult],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[CandidateResult]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result)
    rows: list[dict[str, Any]] = []
    for record in records:
        task_results = sorted(
            by_task[record.id], key=lambda result: result.candidate_index
        )
        if [result.candidate_index for result in task_results] != list(range(4)):
            raise ValueError(f"Riemann task {record.id} lacks candidate indices 0..3")
        verified_indices = [
            result.candidate_index
            for result in task_results
            if result.category == "verified"
        ]
        item = metadata_by_id[record.id]
        rows.append(
            {
                **item,
                "candidate_categories": [
                    result.category for result in task_results
                ],
                "verified_candidate_count": len(verified_indices),
                "verified_candidate_indices": verified_indices,
                "solved": bool(verified_indices),
            }
        )
    return rows


def _breakdown(
    outcome_rows: Sequence[Mapping[str, Any]],
    key: str,
    expected_values: Sequence[str] | None,
    *,
    none_label: str | None = None,
) -> list[dict[str, Any]]:
    observed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcome_rows:
        raw = row[key]
        value = none_label if raw is None and none_label is not None else str(raw)
        observed[str(value)].append(row)
    values = (
        list(expected_values)
        if expected_values is not None
        else sorted(observed)
    )
    unexpected = set(observed) - set(values)
    if unexpected:
        raise ValueError(f"unexpected Riemann {key} values: {sorted(unexpected)}")
    return [_outcome_group(value, observed.get(value, [])) for value in values]


def _outcome_group(
    value: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    verified = [int(row["verified_candidate_count"]) for row in rows]
    return {
        "value": value,
        "task_count": len(rows),
        "candidate_count": len(rows) * 4,
        "verified_candidates": sum(verified),
        "solved_tasks": sum(bool(row["solved"]) for row in rows),
        "pass_at_k": (
            None
            if not rows
            else {
                "pass@1": fmean(pass_at_k(4, count, 1) for count in verified),
                "pass@4": fmean(pass_at_k(4, count, 4) for count in verified),
            }
        ),
    }


def _token_statistics(results: Sequence[CandidateResult]) -> dict[str, Any]:
    values = [result.generated_token_count for result in results]
    if any(value is None for value in values):
        raise ValueError("Riemann results have missing generated-token counts")
    tokens = [int(value) for value in values if value is not None]
    return {
        "total": sum(tokens),
        "minimum": min(tokens),
        "maximum": max(tokens),
        "mean": fmean(tokens),
        "median": median(tokens),
        "p95": _nearest_rank(tokens, 0.95),
        "p99": _nearest_rank(tokens, 0.99),
    }


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": min(ordered),
        "maximum": max(ordered),
        "mean": fmean(ordered),
        "median": median(ordered),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
    }


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def _generation_digest(results: Sequence[CandidateResult]) -> str:
    projection = [
        {
            "task_id": result.task_id,
            "candidate_index": result.candidate_index,
            "candidate_text": result.candidate_text,
            "generated_token_count": result.generated_token_count,
            "finish_reason": result.finish_reason,
        }
        for result in results
    ]
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_generation_artifact(
    output_dir: Path, generated: Sequence[GeneratedCandidate]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "generation.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in generated:
            value = asdict(item)
            value["task"] = item.task.to_dict()
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def _readme(full: Mapping[str, Any]) -> str:
    overall = full["overall"]
    domains = full["domain_breakdown"]
    domain_rows = "\n".join(
        "| {value} | {task_count} | {solved_tasks} | {pass1} | {pass4} |".format(
            value=item["value"],
            task_count=item["task_count"],
            solved_tasks=item["solved_tasks"],
            pass1=(
                "n/a"
                if item["pass_at_k"] is None
                else f"{item['pass_at_k']['pass@1']:.6f}"
            ),
            pass4=(
                "n/a"
                if item["pass_at_k"] is None
                else f"{item['pass_at_k']['pass@4']:.6f}"
            ),
        )
        for item in domains
    )
    efficiency = full["generation_efficiency_per_solved_task"]
    efficiency_text = (
        "not available because no tasks were solved"
        if efficiency["generated_tokens"] is None
        else (
            f"{efficiency['generated_tokens']:.2f} generated tokens and "
            f"{efficiency['generation_wall_time_seconds']:.2f} generation seconds"
        )
    )
    return f"""# Qwen3.5-4B-Base Riemann foundation casting

`OBSERVED`: the frozen `{WORKLOAD_ID}` assessment completed all 556 validation
tasks with four candidates per task and zero unresolved generation or verifier
infrastructure errors. It verified {overall['verified_candidates']} candidates and
solved {overall['solved_tasks']} tasks: pass@1
{overall['pass_at_k']['pass@1']:.6f}, pass@4
{overall['pass_at_k']['pass@4']:.6f}.

| Deterministic source domain | Tasks | Solved | pass@1 | pass@4 |
| --- | ---: | ---: | ---: | ---: |
{domain_rows}

`ACCEPTED`: the run reused `Qwen/Qwen3.5-4B-Base` and its tokenizer at
`{MODEL_REVISION}` exactly as accepted by issue #44: BF16, no quantization, the
text-only lane, and no chat template. The common casting contract used raw
`whole-proof-v1` continuations, temperature 0.8, top-p 0.95, no top-k, four
candidates, a 1,024-token limit, and seed 0. No extraction, semantic repair,
Lean-feedback retry, or candidate regeneration was applied.

`OBSERVED`: generation took {full['timing_seconds']['generation_wall']:.2f}
seconds, verification took {full['timing_seconds']['verification_wall']:.2f}
seconds, and the run generated {full['generated_tokens']['total']} tokens at
{full['throughput']['generated_tokens_per_second']:.2f} tokens/s. Per solved task,
generation used {efficiency_text}. Device-level peak memory was
{full['runtime']['peak_gpu_memory_mib']} MiB on
{full['runtime']['cuda_device']}.

`OBSERVED`: domain labels use committed source path/declaration metadata only.
Direct graph relevance, relevance distance, component inclusion, and
component-associated prerequisite views remain separate in `full.json` and
`task-outcomes.jsonl`; model outputs never define a category. Protected near and
far holdouts were not loaded. Raw generations, model caches, and bulky logs stay
outside Git.
"""


def _task_ids_digest(records: Sequence[MathlibProofRecord]) -> str:
    return hashlib.sha256(
        "\n".join(record.id for record in records).encode("utf-8")
    ).hexdigest()


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
