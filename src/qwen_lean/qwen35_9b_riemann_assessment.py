from __future__ import annotations

import gc
import gzip
import hashlib
import json
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import read_artifacts, write_artifacts
from .baseline import GeneratedCandidate
from .metrics import pass_at_k, summarize_results
from .phase2_corpus import substitute_span, validate_record_source_text
from .phase2_schema import MathlibProofRecord
from .phase2_verification import INVALID_PROOF, run_lean_source
from .prompt import (
    PROMPT_FORMAT_ID,
    normalize_transport,
    render_proof_request,
)
from .qwen35_9b_base_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    _DeviceMemorySampler,
    _configure_vllm_environment,
    _convert_outputs,
    _generation_attempt,
    _local_runtime,
    _validate_runtime_versions,
    validate_model_snapshot,
    vllm_engine_kwargs,
    vllm_sampling_kwargs,
)
from .schema import PHASE1_RESULT_SCHEMA_VERSION, CandidateResult, RunMetadata, TaskRecord


CONFIG_SCHEMA_VERSION = "qwen35-9b-riemann-assessment-config-v1"
PREFLIGHT_SCHEMA_VERSION = "qwen35-9b-riemann-preflight-v1"
GENERATION_SCHEMA_VERSION = "qwen35-9b-riemann-generation-v1"
EVIDENCE_SCHEMA_VERSION = "qwen35-9b-riemann-evidence-v1"
DOMAIN_SCHEMA_VERSION = "riemann-domain-breakdown-v1"
FROZEN_DOMAIN_CONFIG_SHA256 = (
    "b5c19a9c6d134c39751f391b88840c506d889cf370fc4931949c3627bc323d2a"
)
WORKLOAD_ID = "riemann-specialist-validation-v1"


@dataclass(frozen=True)
class RiemannAssessmentConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> RiemannAssessmentConfig:
        config = cls(
            path=path.resolve(),
            value=json.loads(path.read_text(encoding="utf-8")),
        )
        config.validate()
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.value["sampling"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.value["runtime"]

    @property
    def verifier(self) -> dict[str, Any]:
        return self.value["verifier"]

    @property
    def lane(self) -> dict[str, Any]:
        return self.value["lane"]

    @property
    def workload(self) -> dict[str, Any]:
        return self.value["workload"]

    def digest(self) -> str:
        return _sha256_file(self.path)

    def validate(self) -> None:
        if self.value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported Qwen3.5 9B Riemann assessment config")
        expected_model = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
        }
        if self.model != expected_model:
            raise ValueError("model/tokenizer identity differs from accepted issue #43")
        expected_sampling = {
            "candidates_per_task": 4,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": -1,
            "max_new_tokens": 1024,
            "stop": "tokenizer_eos_or_token_limit",
            "seed": 0,
        }
        if self.sampling != expected_sampling:
            raise ValueError("sampling differs from frozen issue #63")
        workload = self.workload
        if (
            workload.get("corpus_id") != WORKLOAD_ID
            or int(workload.get("expected_task_count", 0)) != 556
            or workload.get("split") != "validation"
            or workload.get("source_revision")
            != "81a5d257c8e410db227a6665ed08f64fea08e997"
            or workload.get("lean_toolchain") != "leanprover/lean4:v4.32.0"
        ):
            raise ValueError("workload differs from frozen issue #63/#37 identity")
        runtime = self.runtime
        expected_runtime = {
            "inference_engine": "vllm",
            "inference_engine_version": "0.23.0",
            "torch_version": "2.11.0+cu130",
            "transformers_version": "5.15.0",
            "bitsandbytes_version": "0.49.1",
            "cuda_toolkit_source": "isolated-python-runtime",
            "cuda_linker_layout": "python-wheel-lib64-compat-v1",
            "expected_cuda_device_name": "NVIDIA RTX 4000 Ada Generation",
            "vllm_enable_v1_multiprocessing": False,
            "vllm_worker_multiproc_method": "spawn",
        }
        if runtime != expected_runtime:
            raise ValueError("runtime differs from accepted issue #43 identity")
        lane = self.lane
        expected_lane = {
            "lane_id": "bf16-text-only-v1",
            "dtype": "bfloat16",
            "quantization": None,
            "load_format": "auto",
            "language_model_only": True,
            "cpu_offload_gb": 0,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.95,
            "max_model_len": 2048,
            "max_num_seqs": 32,
            "enforce_eager": True,
        }
        if lane != expected_lane:
            raise ValueError("precision lane differs from accepted BF16 issue #43 identity")
        if (
            float(self.verifier.get("timeout_seconds", 0)) != 30.0
            or int(self.verifier.get("verification_workers", 0)) < 1
        ):
            raise ValueError("verifier differs from frozen issue #63 semantics")
        if not self.value.get("domain_views"):
            raise ValueError("deterministic domain views are required")


@dataclass(frozen=True)
class SpecialistTask:
    record: MathlibProofRecord
    relevance_class: str | None
    riemann_metadata: dict[str, Any]

    @property
    def task(self) -> TaskRecord:
        return TaskRecord(
            id=self.record.id,
            preamble="",
            declaration=self.record.declaration,
            declaration_name=self.record.declaration_name,
        )


def load_specialist_tasks(
    config: RiemannAssessmentConfig, repository_root: Path
) -> list[SpecialistTask]:
    root = repository_root.resolve()
    workload = config.workload
    membership_path = root / str(workload["membership"])
    record_store_path = root / str(workload["record_store"])
    if _sha256_file(membership_path) != workload["membership_sha256"]:
        raise ValueError("specialist membership hash differs from issue #37")
    if _sha256_file(record_store_path) != workload["record_store_sha256"]:
        raise ValueError("specialist record-store hash differs from issue #37")
    members = _read_jsonl(membership_path)
    if len(members) != int(workload["expected_task_count"]):
        raise ValueError("specialist membership count differs from 556")
    member_by_id = {str(item["record_id"]): item for item in members}
    if len(member_by_id) != len(members):
        raise ValueError("specialist membership contains duplicate record ids")
    records: dict[str, dict[str, Any]] = {}
    with gzip.open(record_store_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            record_id = str(value["id"])
            if record_id in member_by_id:
                records[record_id] = value
    if set(records) != set(member_by_id):
        raise ValueError("specialist membership references missing records")
    tasks: list[SpecialistTask] = []
    for record_id in sorted(member_by_id):
        raw = records[record_id]
        member = member_by_id[record_id]
        if (
            raw["split"] != workload["split"]
            or raw["source_revision"] != workload["source_revision"]
            or raw["file_path"] != member["file_path"]
            or raw["declaration_name"] != member["declaration_name"]
            or raw["component_id"] != member["phase2_component"]
            or raw["statement_fingerprint"] != member["phase2_fingerprint"]
            or raw.get("riemann", {}).get("relevance_class")
            != member.get("relevance_class")
        ):
            raise ValueError(f"specialist identity mismatch for {record_id}")
        tasks.append(
            SpecialistTask(
                record=MathlibProofRecord.from_dict(raw),
                relevance_class=(
                    None
                    if member.get("relevance_class") is None
                    else str(member["relevance_class"])
                ),
                riemann_metadata=dict(raw.get("riemann", {})),
            )
        )
    return tasks


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


def classify_primary_domain(
    record: MathlibProofRecord, domain_config: Mapping[str, Any]
) -> tuple[str, str]:
    path = record.file_path.casefold()
    declaration = record.declaration_name.casefold()
    for rule in domain_config["ordered_primary_domain_rules"]:
        if rule.get("fallback"):
            return str(rule["id"]), "fallback"
        path_match = any(
            str(fragment).casefold() in path
            for fragment in rule.get("path_contains", [])
        )
        declaration_match = any(
            str(fragment).casefold() in declaration
            for fragment in rule.get("declaration_contains", [])
        )
        if path_match or declaration_match:
            return str(rule["id"]), "committed-source-metadata"
    raise AssertionError("unreachable Riemann domain classification")


def materialize_task_metadata(
    tasks: Sequence[SpecialistTask], domain_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    bubble_classes = set(domain_config["riemann_bubble_classes"])
    component_tasks: dict[str, list[SpecialistTask]] = defaultdict(list)
    for task in tasks:
        component_tasks[task.record.component_id].append(task)

    component_basis: dict[str, dict[str, Any]] = {}
    for component_id, members in component_tasks.items():
        classes = sorted(
            {
                str(member.relevance_class)
                for member in members
                if member.relevance_class is not None
            }
        )
        if "core" in classes:
            basis = "component-contains-core"
        elif set(classes) & bubble_classes:
            basis = "component-contains-riemann-bubble"
        elif any(
            member.record.file_path.startswith("Mathlib/NumberTheory/")
            for member in members
        ):
            basis = "component-contains-number-theory-control"
        else:
            raise ValueError(
                "Riemann validation component has no committed inclusion basis: "
                f"{component_id}"
            )
        component_basis[component_id] = {
            "component_inclusion_basis": basis,
            "component_anchor_relevance_classes": classes,
            "component_task_count": len(members),
        }

    output: list[dict[str, Any]] = []
    for task in tasks:
        record = task.record
        relevance = task.relevance_class
        relevance_key = "component-associated" if relevance is None else relevance
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
                "seed_families": list(task.riemann_metadata.get("seed_families", [])),
                "inclusion_scope": (
                    "direct"
                    if relevance is not None or direct_number_theory
                    else "component-associated"
                ),
                **component_basis[record.component_id],
            }
        )
    return output


def run_preflight(
    config: RiemannAssessmentConfig,
    repository_root: Path,
    mathlib_root: Path,
    lean_environment_root: Path,
    model_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    _configure_vllm_environment(config)  # type: ignore[arg-type]
    snapshot = validate_model_snapshot(config, model_snapshot)  # type: ignore[arg-type]
    tasks = load_specialist_tasks(config, repository_root)
    _validate_mathlib_checkout(config, mathlib_root)
    sources = _load_and_validate_sources(tasks, mathlib_root)
    lean_environment = _validate_lean_environment(config, lean_environment_root)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    prompts = [render_proof_request(item.record.declaration) for item in tasks]
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    if max(prompt_lengths) + int(config.sampling["max_new_tokens"]) > int(
        config.lane["max_model_len"]
    ):
        raise ValueError("specialist prompt exceeds the accepted model context")

    control = tasks[0].record
    source = sources[control.file_path]
    positive = run_lean_source(
        substitute_span(source, control.proof_span, control.proof),
        mathlib_root,
        timeout_seconds=float(config.verifier["timeout_seconds"]),
        lean_environment_root=lean_environment_root,
    )
    negative = run_lean_source(
        substitute_span(source, control.proof_span, INVALID_PROOF),
        mathlib_root,
        timeout_seconds=float(config.verifier["timeout_seconds"]),
        lean_environment_root=lean_environment_root,
    )
    if positive.status != "accepted" or negative.status != "rejected":
        raise RuntimeError("Lean verifier preflight did not accept/reject controls")

    runtime = _local_runtime(config)  # type: ignore[arg-type]
    bf16 = _generation_attempt(
        config, snapshot, config.lane, [prompts[0]]  # type: ignore[arg-type]
    )
    status = "passed" if bf16["status"] == "passed" else "failed"
    evidence = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "model": config.model,
        "model_snapshot": str(snapshot),
        "accepted_lane": config.lane["lane_id"] if status == "passed" else None,
        "bf16_attempt": bf16,
        "prompt_format_id": PROMPT_FORMAT_ID,
        "chat_template": None,
        "prompt_transformation": None,
        "workload": {
            **config.workload,
            "loaded_task_count": len(tasks),
            "prompt_token_statistics": _number_statistics(prompt_lengths),
        },
        "runtime": runtime,
        "verifier_environment": lean_environment,
        "verifier_controls": {
            "task_id": control.id,
            "original_proof_status": positive.status,
            "controlled_invalid_status": negative.status,
            "timeout_seconds": config.verifier["timeout_seconds"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def run_generation(
    config: RiemannAssessmentConfig,
    repository_root: Path,
    model_snapshot: Path,
    preflight_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _configure_vllm_environment(config)  # type: ignore[arg-type]
    snapshot = validate_model_snapshot(config, model_snapshot)  # type: ignore[arg-type]
    preflight = _load_valid_preflight(config, preflight_path, snapshot)
    tasks = load_specialist_tasks(config, repository_root)
    runtime = _local_runtime(config)  # type: ignore[arg-type]
    generated, metrics = _generate_specialist_candidates(
        config, snapshot, config.lane, tasks
    )
    runtime.update(metrics)
    expected_count = len(tasks) * int(config.sampling["candidates_per_task"])
    if len(generated) != expected_count:
        raise RuntimeError(f"expected {expected_count} candidates, got {len(generated)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "generations.jsonl"
    with candidate_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in generated:
            stream.write(json.dumps(_generated_to_dict(item), sort_keys=True) + "\n")
    manifest = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "preflight_sha256": _sha256_file(preflight_path),
        "model": config.model,
        "model_snapshot": str(snapshot),
        "accepted_lane": preflight["accepted_lane"],
        "workload_id": WORKLOAD_ID,
        "task_count": len(tasks),
        "candidate_count": len(generated),
        "generation_settings": _generation_settings(config),
        "runtime": runtime,
        "generations_sha256": _sha256_file(candidate_path),
    }
    (output_dir / "generation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_verification(
    config: RiemannAssessmentConfig,
    repository_root: Path,
    mathlib_root: Path,
    lean_environment_root: Path,
    preflight_path: Path,
    generation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tasks = load_specialist_tasks(config, repository_root)
    _validate_mathlib_checkout(config, mathlib_root)
    sources = _load_and_validate_sources(tasks, mathlib_root)
    lean_environment = _validate_lean_environment(config, lean_environment_root)
    generation = _load_valid_generation(config, preflight_path, generation_dir, tasks)
    generated = _read_generations(generation_dir / "generations.jsonl", tasks)
    task_by_id = {item.record.id: item for item in tasks}
    started = time.perf_counter()
    results: list[CandidateResult] = []
    with ThreadPoolExecutor(
        max_workers=int(config.verifier["verification_workers"])
    ) as executor:
        futures = [
            executor.submit(
                _verify_generated,
                item,
                task_by_id[item.task.id].record,
                sources[task_by_id[item.task.id].record.file_path],
                mathlib_root,
                lean_environment_root,
                float(config.verifier["timeout_seconds"]),
            )
            for item in generated
        ]
        progress_step = max(1, len(futures) // 4)
        for completed_count, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed_count % progress_step == 0 or completed_count == len(futures):
                print(
                    f"verification-progress {completed_count}/{len(futures)}",
                    flush=True,
                )
    verification_seconds = time.perf_counter() - started
    results.sort(key=lambda item: (item.task_id, item.candidate_index))
    task_ids = [item.record.id for item in tasks]
    summary = summarize_results(
        results,
        expected_task_ids=task_ids,
        candidates_per_task=int(config.sampling["candidates_per_task"]),
    )
    summary.update(
        {
            "workload_id": WORKLOAD_ID,
            "generated_tokens": _token_statistics(results),
            "engine_load_time_seconds": generation["runtime"]["engine_load_time_seconds"],
            "generation_wall_time_seconds": generation["runtime"][
                "generation_wall_time_seconds"
            ],
            "verification_wall_time_seconds": verification_seconds,
        }
    )
    summary["run_wall_time_seconds"] = (
        float(summary["engine_load_time_seconds"])
        + float(summary["generation_wall_time_seconds"])
        + verification_seconds
    )
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    tokens = int(summary["generated_tokens"]["total"])
    generation_seconds = float(summary["generation_wall_time_seconds"])
    summary["throughput"] = {
        "generated_tokens_per_second": tokens / generation_seconds,
        "candidates_per_second": len(results) / generation_seconds,
    }
    summary["compute_per_solved_task"] = {
        "generated_tokens": tokens / solved if solved else None,
        "generation_gpu_seconds": generation_seconds / solved if solved else None,
    }

    runtime = dict(generation["runtime"])
    runtime.update(
        {
            "verification_wall_time_seconds": verification_seconds,
            "verification_workers": int(config.verifier["verification_workers"]),
            "config_sha256": config.digest(),
            "inference_execution": "local_cuda",
        }
    )
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source=(
            f"{config.workload['corpus_id']}@{config.workload['membership_sha256']}"
        ),
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=str(config.workload["lean_toolchain"]),
        mathlib_revision=str(config.workload["source_revision"]),
        verifier_timeout_seconds=float(config.verifier["timeout_seconds"]),
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=WORKLOAD_ID,
        benchmark_split="validation",
        benchmark_repository=str(config.workload["source_repository"]),
        benchmark_revision=str(config.workload["source_revision"]),
        verifier_environment=lean_environment,
        candidates_per_task=int(config.sampling["candidates_per_task"]),
        inference_engine=str(config.runtime["inference_engine"]),
        inference_engine_version=str(config.runtime["inference_engine_version"]),
        generation_settings=_generation_settings(config),
        runtime=runtime,
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
    return summary


def write_compact_evidence(
    config: RiemannAssessmentConfig,
    repository_root: Path,
    domain_config_path: Path,
    preflight_path: Path,
    generation_dir: Path,
    artifact_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    tasks = load_specialist_tasks(config, repository_root)
    domain_config = load_domain_config(domain_config_path)
    if _sha256_file(domain_config_path) != FROZEN_DOMAIN_CONFIG_SHA256:
        raise ValueError("domain classification differs from accepted issue #64")
    task_metadata = materialize_task_metadata(tasks, domain_config)
    generation = _load_valid_generation(
        config, preflight_path, generation_dir, tasks
    )
    generated = _read_generations(generation_dir / "generations.jsonl", tasks)
    metadata, results = read_artifacts(artifact_dir)
    stored = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    task_ids = [item.record.id for item in tasks]
    _validate_final_artifacts(config, metadata, results, stored, task_ids)
    if len(results) != len(generated) or any(
        result.task_id != candidate.task.id
        or result.candidate_index != candidate.candidate_index
        or result.candidate_text != candidate.text
        or result.generated_token_count != candidate.token_count
        or result.finish_reason != candidate.finish_reason
        or result.generation_latency_seconds
        != candidate.generation_latency_seconds
        for result, candidate in zip(results, generated)
    ):
        raise ValueError("verified results differ from immutable generation checkpoint")
    recomputed = summarize_results(
        results,
        expected_task_ids=task_ids,
        candidates_per_task=int(config.sampling["candidates_per_task"]),
    )
    result_by_task: dict[str, list[CandidateResult]] = defaultdict(list)
    for result in results:
        result_by_task[result.task_id].append(result)
    task_outcomes: list[dict[str, Any]] = []
    for task, task_metadata_item in zip(tasks, task_metadata, strict=True):
        task_results = sorted(
            result_by_task[task.record.id], key=lambda item: item.candidate_index
        )
        if [item.candidate_index for item in task_results] != list(range(4)):
            raise ValueError(f"task {task.record.id} lacks candidate indices 0..3")
        verified_indices = [
            item.candidate_index
            for item in task_results
            if item.category == "verified"
        ]
        task_outcomes.append(
            {
                **task_metadata_item,
                "candidate_categories": [item.category for item in task_results],
                "verified_candidate_count": len(verified_indices),
                "verified_candidate_indices": verified_indices,
                "solved": bool(verified_indices),
            }
        )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_evidence_preflight(config, preflight)
    compact_preflight = {
        key: value
        for key, value in preflight.items()
        if key != "model_snapshot"
    }
    compact_preflight["model_snapshot"] = {
        "revision": MODEL_REVISION,
        "local_cache_used": True,
        "path_committed": False,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = evidence_dir / "task-outcomes.jsonl"
    with outcomes_path.open("w", encoding="utf-8", newline="\n") as stream:
        for outcome in task_outcomes:
            stream.write(json.dumps(outcome, sort_keys=True) + "\n")
    domain_values = [
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
        "workload_id": WORKLOAD_ID,
        "task_count": recomputed["task_count"],
        "candidate_count": recomputed["candidate_count"],
        "verified_candidate_count": recomputed["category_counts"]["verified"],
        "tasks_with_verified_candidate": recomputed["tasks_with_verified_candidate"],
        "pass_at_k": recomputed["pass_at_k"],
        "category_counts": recomputed["category_counts"],
        "finish_reason_counts": recomputed["finish_reason_counts"],
        "verifier_timeout_count": recomputed["verifier_timeout_count"],
        "infrastructure_error_count": recomputed["infrastructure_error_count"],
        "generated_tokens": stored["generated_tokens"],
        "engine_load_time_seconds": stored["engine_load_time_seconds"],
        "generation_wall_time_seconds": stored["generation_wall_time_seconds"],
        "verification_wall_time_seconds": stored["verification_wall_time_seconds"],
        "run_wall_time_seconds": stored["run_wall_time_seconds"],
        "throughput": stored["throughput"],
        "compute_per_solved_task": stored["compute_per_solved_task"],
        "lane_id": config.lane["lane_id"],
        "precision": config.lane["dtype"],
        "quantization": config.lane["quantization"],
        "runtime": metadata.runtime,
        "verifier_environment": metadata.verifier_environment,
        "results_jsonl_sha256": _sha256_file(artifact_dir / "results.jsonl"),
        "candidate_text_sha256": _candidate_text_digest(results),
        "generation_checkpoint": {
            "schema_version": generation["schema_version"],
            "manifest_sha256": _sha256_file(generation_dir / "generation.json"),
            "generations_sha256": generation["generations_sha256"],
            "task_count": generation["task_count"],
            "candidate_count": generation["candidate_count"],
            "results_match_generation": True,
            "raw_continuations_committed": False,
        },
        "domain_config": {
            "schema_version": DOMAIN_SCHEMA_VERSION,
            "sha256": FROZEN_DOMAIN_CONFIG_SHA256,
            "classification_input": domain_config["classification_input"],
            "first_accepted_candidate": "issue-64-pr-73",
            "model_outputs_used": False,
            "legacy_generation_config_domain_views_used": False,
        },
        "domain_breakdown": _outcome_breakdown(
            task_outcomes, "primary_domain", domain_values
        ),
        "direct_relevance_breakdown": _outcome_breakdown(
            task_outcomes,
            "direct_relevance_class",
            direct_relevance_values,
            none_label="none",
        ),
        "relevance_distance_breakdown": _outcome_breakdown(
            task_outcomes, "relevance_distance", None
        ),
        "component_inclusion_breakdown": _outcome_breakdown(
            task_outcomes, "component_inclusion_basis", None
        ),
        "inclusion_scope_breakdown": _outcome_breakdown(
            task_outcomes,
            "inclusion_scope",
            ["direct", "component-associated"],
        ),
        "task_outcomes": {
            "path": "task-outcomes.jsonl",
            "rows": len(task_outcomes),
            "sha256": _sha256_file(outcomes_path),
            "task_ids_sha256": _task_ids_digest(task_ids),
            "paired_analysis_fields": [
                "task_id",
                "solved",
                "verified_candidate_count",
                "verified_candidate_indices",
            ],
        },
    }
    if (
        full["task_count"] != 556
        or full["candidate_count"] != 2224
        or full["infrastructure_error_count"] != 0
    ):
        raise ValueError("final Riemann evidence fails coverage/infrastructure gates")
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "model": config.model,
        "preflight": compact_preflight,
        "full": full,
        "task_outcomes": task_outcomes,
        "classification": {
            "source": domain_config["classification_input"],
            "domain_config_sha256": FROZEN_DOMAIN_CONFIG_SHA256,
            "first_accepted_candidate": "issue-64-pr-73",
            "model_outputs_used": False,
        },
        "limitations": [
            "Primary domain views are deterministic, mutually exclusive, and retain zero-task views explicitly.",
            "verifier_timeout is an unsuccessful proof outcome, not an infrastructure error.",
            "Raw candidate continuations remain outside Git.",
            "No miniF2F or protected holdout inference was run for this assessment.",
        ],
    }
    (evidence_dir / "preflight.json").write_text(
        json.dumps(compact_preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence_dir / "full.json").write_text(
        json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence_dir / "README.md").write_text(
        _render_readme(payload), encoding="utf-8"
    )
    return payload


def _validate_mathlib_checkout(
    config: RiemannAssessmentConfig, mathlib_root: Path
) -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mathlib_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if revision != config.workload["source_revision"]:
        raise ValueError("mathlib checkout differs from issue #37 source revision")
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "Mathlib"], cwd=mathlib_root
    )
    if diff.returncode != 0:
        raise ValueError("mathlib source checkout has local changes")


def _generate_specialist_candidates(
    config: RiemannAssessmentConfig,
    snapshot: Path,
    lane: Mapping[str, Any],
    tasks: Sequence[SpecialistTask],
) -> tuple[list[GeneratedCandidate], dict[str, Any]]:
    import torch

    _configure_vllm_environment(config)  # type: ignore[arg-type]
    import vllm
    from vllm import LLM, SamplingParams

    _validate_runtime_versions(config, vllm)  # type: ignore[arg-type]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler = _DeviceMemorySampler(torch.cuda.current_device())
    llm: Any | None = None
    try:
        with sampler:
            load_started = time.perf_counter()
            llm = LLM(
                **vllm_engine_kwargs(config, snapshot, lane)  # type: ignore[arg-type]
            )
            load_seconds = time.perf_counter() - load_started
            prompts = [render_proof_request(item.record.declaration) for item in tasks]
            generation_started = time.perf_counter()
            outputs = llm.generate(
                prompts,
                SamplingParams(**vllm_sampling_kwargs(config.sampling)),
                use_tqdm=True,
            )
            generation_seconds = time.perf_counter() - generation_started
        generated = _convert_outputs(
            [item.task for item in tasks],
            prompts,
            outputs,
            generation_seconds,
            candidates_per_task=int(config.sampling["candidates_per_task"]),
        )
        metrics = {
            "engine_load_time_seconds": load_seconds,
            "generation_wall_time_seconds": generation_seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_device_memory_used_mib": sampler.peak_used_mib,
        }
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return generated, metrics


def _validate_lean_environment(
    config: RiemannAssessmentConfig, lean_environment_root: Path
) -> dict[str, Any]:
    version = subprocess.run(
        ["lake", "env", "lean", "--version"],
        cwd=lean_environment_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    manifest = json.loads(
        (lean_environment_root / "lake-manifest.json").read_text(encoding="utf-8")
    )
    mathlib = next(item for item in manifest["packages"] if item["name"] == "mathlib")
    if mathlib["rev"] != config.workload["source_revision"] or "4.32.0" not in version:
        raise ValueError("Lean verifier environment differs from issue #37")
    return {
        "project": "murillo128/qwen-lean",
        "lean_version": version,
        "lean_toolchain": config.workload["lean_toolchain"],
        "mathlib_revision": mathlib["rev"],
        "candidate_reconstruction": "phase2-source-proof-span-substitution-v1",
        "has_sorry_disabled": True,
    }


def _load_and_validate_sources(
    tasks: Sequence[SpecialistTask], mathlib_root: Path
) -> dict[str, str]:
    sources: dict[str, str] = {}
    for item in tasks:
        path = item.record.file_path
        source = sources.setdefault(
            path, (mathlib_root / path).read_text(encoding="utf-8")
        )
        validate_record_source_text(item.record, source)
    return sources


def _load_valid_preflight(
    config: RiemannAssessmentConfig, preflight_path: Path, snapshot: Path
) -> dict[str, Any]:
    value = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or value.get("status") != "passed"
        or value.get("config_sha256") != config.digest()
        or value.get("model") != config.model
        or value.get("accepted_lane") != "bf16-text-only-v1"
        or value.get("bf16_attempt", {}).get("status") != "passed"
        or Path(str(value.get("model_snapshot"))).resolve() != snapshot.resolve()
    ):
        raise ValueError("generation requires the exact passing BF16 preflight")
    return value


def _load_valid_generation(
    config: RiemannAssessmentConfig,
    preflight_path: Path,
    generation_dir: Path,
    tasks: Sequence[SpecialistTask],
) -> dict[str, Any]:
    value = json.loads((generation_dir / "generation.json").read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != GENERATION_SCHEMA_VERSION
        or value.get("config_sha256") != config.digest()
        or value.get("preflight_sha256") != _sha256_file(preflight_path)
        or value.get("model") != config.model
        or value.get("accepted_lane") != "bf16-text-only-v1"
        or value.get("workload_id") != WORKLOAD_ID
        or int(value.get("task_count", 0)) != len(tasks)
        or int(value.get("candidate_count", 0)) != len(tasks) * 4
        or value.get("generation_settings") != _generation_settings(config)
        or value.get("generations_sha256")
        != _sha256_file(generation_dir / "generations.jsonl")
    ):
        raise ValueError("generation artifact differs from the frozen accepted run")
    return value


def _read_generations(
    path: Path, tasks: Sequence[SpecialistTask]
) -> list[GeneratedCandidate]:
    task_by_id = {item.record.id: item.task for item in tasks}
    generated: list[GeneratedCandidate] = []
    for value in _read_jsonl(path):
        task_id = str(value["task_id"])
        if task_id not in task_by_id:
            raise ValueError(f"generation has unexpected task id: {task_id}")
        generated.append(
            GeneratedCandidate(
                task=task_by_id[task_id],
                candidate_index=int(value["candidate_index"]),
                text=str(value["text"]),
                token_count=int(value["token_count"]),
                finish_reason=str(value["finish_reason"]),
                generation_latency_seconds=float(value["generation_latency_seconds"]),
            )
        )
    expected = [(item.record.id, index) for item in tasks for index in range(4)]
    actual = [(item.task.id, item.candidate_index) for item in generated]
    if actual != expected:
        raise ValueError("generation task/candidate ordering is incomplete or changed")
    return generated


def _verify_generated(
    generated: GeneratedCandidate,
    record: MathlibProofRecord,
    source: str,
    mathlib_root: Path,
    lean_environment_root: Path,
    timeout_seconds: float,
) -> CandidateResult:
    candidate = normalize_transport(generated.text)
    if not candidate:
        return CandidateResult(
            task_id=record.id,
            candidate_id=f"model-{generated.candidate_index}",
            candidate_index=generated.candidate_index,
            candidate_text=generated.text,
            category="empty_candidate",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": "candidate is empty"},
            generation_latency_seconds=generated.generation_latency_seconds,
            verification_latency_seconds=0.0,
            total_latency_seconds=generated.generation_latency_seconds,
            generated_token_count=generated.token_count,
            finish_reason=generated.finish_reason,
        )
    replacement = f"by\n  {candidate}"
    check = run_lean_source(
        substitute_span(source, record.proof_span, replacement),
        mathlib_root,
        timeout_seconds=timeout_seconds,
        lean_environment_root=lean_environment_root,
    )
    category = {
        "accepted": "verified",
        "rejected": "lean_rejected",
        "timeout": "verifier_timeout",
        "infrastructure_error": "verifier_error",
    }[check.status]
    return CandidateResult(
        task_id=record.id,
        candidate_id=f"model-{generated.candidate_index}",
        candidate_index=generated.candidate_index,
        candidate_text=generated.text,
        category=category,  # type: ignore[arg-type]
        lean_exit_code=check.exit_code,
        diagnostics={"stdout": "", "stderr": check.diagnostic},
        generation_latency_seconds=generated.generation_latency_seconds,
        verification_latency_seconds=check.latency_seconds,
        total_latency_seconds=generated.generation_latency_seconds
        + check.latency_seconds,
        generated_token_count=generated.token_count,
        finish_reason=generated.finish_reason,
    )


def _validate_final_artifacts(
    config: RiemannAssessmentConfig,
    metadata: RunMetadata,
    results: Sequence[CandidateResult],
    stored: Mapping[str, Any],
    task_ids: list[str],
) -> None:
    if (
        metadata.workload_id != WORKLOAD_ID
        or metadata.model_id != MODEL_ID
        or metadata.model_revision != MODEL_REVISION
        or metadata.tokenizer_id != MODEL_ID
        or metadata.tokenizer_revision != MODEL_REVISION
        or metadata.prompt_format_id != PROMPT_FORMAT_ID
        or metadata.candidate_source != "model"
        or metadata.task_source
        != f"{WORKLOAD_ID}@{config.workload['membership_sha256']}"
        or metadata.lean_toolchain != config.workload["lean_toolchain"]
        or metadata.mathlib_revision != config.workload["source_revision"]
        or metadata.benchmark_split != "validation"
        or metadata.benchmark_repository != config.workload["source_repository"]
        or metadata.benchmark_revision != config.workload["source_revision"]
        or metadata.verifier_timeout_seconds != 30.0
        or metadata.candidates_per_task != 4
        or metadata.inference_engine != config.runtime["inference_engine"]
        or metadata.inference_engine_version
        != config.runtime["inference_engine_version"]
        or not metadata.verifier_environment
        or metadata.verifier_environment.get("mathlib_revision")
        != config.workload["source_revision"]
        or metadata.verifier_environment.get("lean_toolchain")
        != config.workload["lean_toolchain"]
        or metadata.verifier_environment.get("candidate_reconstruction")
        != "phase2-source-proof-span-substitution-v1"
        or metadata.verifier_environment.get("has_sorry_disabled") is not True
        or metadata.generation_settings != _generation_settings(config)
        or metadata.runtime.get("config_sha256") != config.digest()
        or metadata.runtime.get("inference_execution") != "local_cuda"
        or metadata.runtime.get("cuda_device")
        != config.runtime["expected_cuda_device_name"]
        or metadata.runtime.get("torch") != config.runtime["torch_version"]
        or metadata.runtime.get("transformers")
        != config.runtime["transformers_version"]
        or metadata.runtime.get("bitsandbytes")
        != config.runtime["bitsandbytes_version"]
    ):
        raise ValueError("final artifact identity differs from issue #65")
    recomputed = summarize_results(
        results, expected_task_ids=task_ids, candidates_per_task=4
    )
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
        "per_task",
    ):
        if recomputed[key] != stored[key]:
            raise ValueError(f"stored summary differs from raw results for {key}")
    if not recomputed["complete"] or recomputed["infrastructure_error_count"]:
        raise ValueError("final assessment is incomplete or infrastructure-failed")
    if stored["generated_tokens"] != _token_statistics(results):
        raise ValueError("stored generated-token statistics differ from raw results")
    timing = {
        "engine_load_time_seconds": metadata.runtime.get("engine_load_time_seconds"),
        "generation_wall_time_seconds": metadata.runtime.get(
            "generation_wall_time_seconds"
        ),
        "verification_wall_time_seconds": metadata.runtime.get(
            "verification_wall_time_seconds"
        ),
    }
    if any(stored.get(key) != value for key, value in timing.items()):
        raise ValueError("stored timing differs from run metadata")
    engine_seconds = float(timing["engine_load_time_seconds"])
    generation_seconds = float(timing["generation_wall_time_seconds"])
    verification_seconds = float(timing["verification_wall_time_seconds"])
    if stored.get("run_wall_time_seconds") != (
        engine_seconds + generation_seconds + verification_seconds
    ):
        raise ValueError("stored run wall time differs from run metadata")
    tokens = int(stored["generated_tokens"]["total"])
    expected_throughput = {
        "generated_tokens_per_second": tokens / generation_seconds,
        "candidates_per_second": len(results) / generation_seconds,
    }
    if stored.get("throughput") != expected_throughput:
        raise ValueError("stored throughput differs from raw results and timing")
    solved = int(recomputed["tasks_with_verified_candidate"]["count"])
    expected_compute = {
        "generated_tokens": tokens / solved if solved else None,
        "generation_gpu_seconds": generation_seconds / solved if solved else None,
    }
    if stored.get("compute_per_solved_task") != expected_compute:
        raise ValueError("stored solved-task efficiency differs from raw results")


def _validate_evidence_preflight(
    config: RiemannAssessmentConfig, preflight: Mapping[str, Any]
) -> None:
    workload = preflight.get("workload", {})
    runtime = preflight.get("runtime", {})
    bf16_attempt = preflight.get("bf16_attempt", {})
    verifier_environment = preflight.get("verifier_environment", {})
    verifier_controls = preflight.get("verifier_controls", {})
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status") != "passed"
        or preflight.get("assessment_id") != config.value["assessment_id"]
        or preflight.get("config_sha256") != config.digest()
        or preflight.get("model") != config.model
        or preflight.get("accepted_lane") != "bf16-text-only-v1"
        or preflight.get("prompt_format_id") != PROMPT_FORMAT_ID
        or preflight.get("chat_template") is not None
        or preflight.get("prompt_transformation") is not None
        or bf16_attempt.get("status") != "passed"
        or int(bf16_attempt.get("candidate_count", 0)) != 4
        or bf16_attempt.get("lane") != config.lane
        or workload.get("corpus_id") != WORKLOAD_ID
        or int(workload.get("loaded_task_count", 0)) != 556
        or workload.get("membership_sha256")
        != config.workload["membership_sha256"]
        or workload.get("record_store_sha256")
        != config.workload["record_store_sha256"]
        or workload.get("source_revision") != config.workload["source_revision"]
        or runtime.get("cuda_device") != config.runtime["expected_cuda_device_name"]
        or runtime.get("torch") != config.runtime["torch_version"]
        or runtime.get("transformers") != config.runtime["transformers_version"]
        or runtime.get("bitsandbytes") != config.runtime["bitsandbytes_version"]
        or runtime.get("vllm") != config.runtime["inference_engine_version"]
        or verifier_environment.get("mathlib_revision")
        != config.workload["source_revision"]
        or verifier_environment.get("lean_toolchain")
        != config.workload["lean_toolchain"]
        or verifier_environment.get("candidate_reconstruction")
        != "phase2-source-proof-span-substitution-v1"
        or verifier_environment.get("has_sorry_disabled") is not True
        or verifier_controls.get("original_proof_status") != "accepted"
        or verifier_controls.get("controlled_invalid_status") != "rejected"
        or float(verifier_controls.get("timeout_seconds", 0)) != 30.0
    ):
        raise ValueError("preflight evidence differs from the accepted issue #65 run")


def _outcome_breakdown(
    outcomes: Sequence[Mapping[str, Any]],
    key: str,
    expected_values: Sequence[str] | None,
    *,
    none_label: str | None = None,
) -> list[dict[str, Any]]:
    observed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        raw = outcome[key]
        value = none_label if raw is None and none_label is not None else str(raw)
        observed[str(value)].append(outcome)
    values = list(expected_values) if expected_values is not None else sorted(observed)
    unexpected = set(observed) - set(values)
    if unexpected:
        raise ValueError(f"unexpected Riemann {key} values: {sorted(unexpected)}")
    groups = []
    for value in values:
        rows = observed.get(value, [])
        verified = [int(row["verified_candidate_count"]) for row in rows]
        groups.append(
            {
                "value": value,
                "task_count": len(rows),
                "candidate_count": len(rows) * 4,
                "verified_candidates": sum(verified),
                "solved_tasks": sum(bool(row["solved"]) for row in rows),
                "pass_at_k": (
                    None
                    if not rows
                    else {
                        "pass@1": statistics.fmean(
                            pass_at_k(4, count, 1) for count in verified
                        ),
                        "pass@4": statistics.fmean(
                            pass_at_k(4, count, 4) for count in verified
                        ),
                    }
                ),
            }
        )
    return groups


def _relevance_views(tasks: Sequence[SpecialistTask]) -> dict[str, list[str]]:
    views: dict[str, list[str]] = {
        "riemann-bubble": [],
        "number-theory-control": [],
        "component-context": [],
    }
    classes = ["core", "premise-1", "premise-2", "user-1", "user-2", "source-neighborhood"]
    views.update({name: [] for name in classes})
    for item in tasks:
        relevance = item.relevance_class
        if relevance in classes:
            views["riemann-bubble"].append(item.record.id)
            views[str(relevance)].append(item.record.id)
        elif relevance == "number-theory-control":
            views["number-theory-control"].append(item.record.id)
        else:
            views["component-context"].append(item.record.id)
    return views


def _generation_settings(config: RiemannAssessmentConfig) -> dict[str, Any]:
    return {
        **config.sampling,
        **config.lane,
        "chat_template": None,
        "prompt_transformation": None,
        "proof_extraction": None,
        "semantic_repair": None,
        "lean_feedback": None,
        "candidate_regeneration": None,
    }


def _generated_to_dict(value: GeneratedCandidate) -> dict[str, Any]:
    return {
        "task_id": value.task.id,
        "candidate_index": value.candidate_index,
        "text": value.text,
        "token_count": value.token_count,
        "finish_reason": value.finish_reason,
        "generation_latency_seconds": value.generation_latency_seconds,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_statistics(results: Sequence[CandidateResult]) -> dict[str, Any]:
    raw_values = [item.generated_token_count for item in results]
    if any(value is None for value in raw_values):
        raise ValueError("Riemann results have missing generated-token counts")
    values = [int(value) for value in raw_values if value is not None]
    return _number_statistics(values)


def _number_statistics(values: Sequence[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": sum(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
    }


def _nearest_rank(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    index = max(0, int(fraction * len(values) + 0.999999999) - 1)
    return values[min(index, len(values) - 1)]


def _candidate_text_digest(results: Iterable[CandidateResult]) -> str:
    digest = hashlib.sha256()
    for result in results:
        digest.update(result.task_id.encode())
        digest.update(b"\0")
        digest.update(str(result.candidate_index).encode())
        digest.update(b"\0")
        digest.update(result.candidate_text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _task_ids_digest(task_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()


def _render_readme(payload: Mapping[str, Any]) -> str:
    full = payload["full"]
    domains = full["domain_breakdown"]
    rows = "\n".join(
        f"| {value['value']} | {value['task_count']} | {value['solved_tasks']} | "
        f"{value['pass_at_k']['pass@1']:.10f} | {value['pass_at_k']['pass@4']:.10f} |"
        if value["task_count"]
        else f"| {value['value']} | 0 | 0 | n/a | n/a |"
        for value in domains
    )
    return f"""# Qwen3.5-9B-Base Riemann specialist validation

**OBSERVED:** the immutable `{MODEL_ID}` snapshot `{MODEL_REVISION}` completed the full 556-task `riemann-specialist-validation-v1` workload locally on the project RTX 4000 Ada. The run used the accepted unquantized `bf16-text-only-v1` lane and exact `whole-proof-v1` raw continuations; no chat template, extraction, repair, Lean feedback, retry, training, miniF2F rerun, or protected-holdout access occurred.

| Tasks | Candidates | Solved tasks | Verified candidates | pass@1 | pass@4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| {full['task_count']} | {full['candidate_count']} | {full['tasks_with_verified_candidate']['count']} | {full['verified_candidate_count']} | {full['pass_at_k']['pass@1']:.10f} | {full['pass_at_k']['pass@4']:.10f} |

| Deterministic domain view | Tasks | Solved | pass@1 | pass@4 |
| --- | ---: | ---: | ---: | ---: |
{rows}

**ACCEPTED:** domain and relevance views reuse the first accepted candidate's
`riemann-domain-breakdown-v1` rules exactly (SHA-256
`{FROZEN_DOMAIN_CONFIG_SHA256}`). The rules use committed source and graph
metadata only; the legacy multi-label views stored in the generation config were
not used for final classification.

The run generated {full['generated_tokens']['total']} tokens in {full['generation_wall_time_seconds']:.2f} generation seconds and used {full['runtime']['peak_device_memory_used_mib']} MiB peak device memory. Verification recorded {full['verifier_timeout_count']} unsuccessful timeout proof outcome(s) and {full['infrastructure_error_count']} unresolved infrastructure errors. `task-outcomes.jsonl` preserves all 556 compact paired outcomes; the final results match the immutable raw checkpoint `{full['generation_checkpoint']['generations_sha256']}` exactly. Raw continuations and model/cache artifacts remain outside Git.
"""
