from __future__ import annotations

import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import read_artifacts, write_artifacts
from .baseline import validate_minif2f_environment
from .metrics import summarize_results
from .minif2f import (
    PHASE1_CONFIG_SCHEMA_VERSION,
    Phase1Config,
    materialize_benchmark_tasks,
)
from .prompt import PROMPT_FORMAT_ID, render_prompt
from .qwen35_9b_base_assessment import (
    _configure_vllm_environment,
    _generate_candidates,
    _generation_attempt,
    _local_runtime,
    _token_statistics,
    _verify_candidate,
)
from .schema import PHASE1_RESULT_SCHEMA_VERSION, CandidateResult, RunMetadata
from .verifier import LeanVerifier

CONFIG_SCHEMA_VERSION = "qwen36-27b-assessment-config-v1"
PREFLIGHT_SCHEMA_VERSION = "qwen36-27b-preflight-v1"
EVIDENCE_SCHEMA_VERSION = "qwen36-27b-assessment-evidence-v1"
BLOCKER_EVIDENCE_SCHEMA_VERSION = "qwen36-27b-hardware-blocker-v1"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LANE_ID = "bitsandbytes-fp4-online-w4a32-v1"
WORKLOADS = ("minif2f-valid-dev16-v1", "minif2f-valid-v1")
STRICT_SAMPLING = {
    "candidates_per_task": 4,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": -1,
    "max_new_tokens": 1024,
    "stop": "tokenizer_eos_or_token_limit",
    "seed": 0,
}
QUANTIZATION_METADATA = {
    "bits": 4,
    "quant_type": "fp4",
    "double_quantization": False,
    "compute_dtype": "float32",
    "quant_storage": "uint8",
    "activation_dtype": "bfloat16",
    "source_checkpoint_dtype": "bfloat16",
    "conversion": "vllm online from pinned official BF16 safetensors",
    "prequantized_checkpoint": False,
}


@dataclass(frozen=True)
class Qwen36AssessmentConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Qwen36AssessmentConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        config = cls(path=path.resolve(), value=value)
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
        return self.value["quantized_lane"]

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def phase1_config(self) -> Phase1Config:
        return Phase1Config(
            path=self.path,
            value={
                "schema_version": PHASE1_CONFIG_SCHEMA_VERSION,
                "benchmark": self.value["benchmark"],
                "workloads": self.value["workloads"],
                "model": self.model,
                "sampling": self.sampling,
                "engine": {
                    "name": self.runtime["inference_engine"],
                    "version": self.runtime["inference_engine_version"],
                    **self.lane,
                },
                "verifier": self.verifier,
            },
        )

    def validate(self) -> None:
        if self.value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("unknown Qwen3.6-27B assessment config schema")
        expected_model = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "license": "Apache-2.0",
        }
        if self.model != expected_model:
            raise ValueError("model/tokenizer identity differs from issue #35")
        if self.sampling != STRICT_SAMPLING:
            raise ValueError("sampling differs from the strict issue #35 contract")
        if set(self.value["workloads"]) != set(WORKLOADS):
            raise ValueError("assessment workloads differ from issue #35")
        if (
            self.value["benchmark"]["split"] != "validation"
            or int(self.value["benchmark"]["expected_primary_task_count"]) != 244
        ):
            raise ValueError("assessment must use all 244 miniF2F validation tasks")
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
            raise ValueError("runtime differs from the frozen local Ada stack")
        expected_lane = {
            "lane_id": LANE_ID,
            "dtype": "bfloat16",
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
            "language_model_only": True,
            "cpu_offload_gb": 0,
            "swap_space": 0,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 2048,
            "max_num_seqs": 4,
            "enforce_eager": True,
            "quantization_metadata": QUANTIZATION_METADATA,
        }
        if self.lane != expected_lane:
            raise ValueError("quantized lane differs from the frozen issue #35 lane")
        if self.verifier["timeout_seconds"] != 30.0:
            raise ValueError("verifier timeout must remain 30 seconds")
        if int(self.verifier["verification_workers"]) < 1:
            raise ValueError("verification worker count must be positive")
        expected_assessment = {
            "prompt_format_id": PROMPT_FORMAT_ID,
            "raw_continuation": True,
            "chat_template": None,
            "proof_extraction": False,
            "verifier_feedback": False,
            "repair": False,
            "reference_evidence": "evidence/phase6/minif2f-validation.json",
            "reference_logical_id": "reference-sft-v1",
        }
        if self.value["assessment"] != expected_assessment:
            raise ValueError("prompt or comparison contract differs from issue #35")


def validate_model_snapshot(config: Qwen36AssessmentConfig, snapshot: Path) -> Path:
    resolved = snapshot.resolve()
    if resolved.name != MODEL_REVISION:
        raise ValueError(
            f"model snapshot must resolve to pinned revision {MODEL_REVISION}: "
            f"{resolved}"
        )
    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (resolved / name).is_file())
    if missing:
        raise ValueError(f"model snapshot is incomplete: {missing}")
    index = json.loads(
        (resolved / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (resolved / name).is_file()]
    if missing_shards:
        raise ValueError(f"model snapshot is missing weight shards: {missing_shards}")
    return resolved


def run_preflight(
    config: Qwen36AssessmentConfig,
    benchmark_root: Path,
    model_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    _configure_vllm_environment(config)  # type: ignore[arg-type]
    _validate_bitsandbytes_contract(config)
    snapshot = validate_model_snapshot(config, model_snapshot)
    phase1 = config.phase1_config()
    environment = validate_minif2f_environment(
        phase1,
        benchmark_root,
        timeout_seconds=float(config.verifier["timeout_seconds"]),
    )
    tasks = phase1.select_workload(
        "minif2f-valid-dev16-v1",
        materialize_benchmark_tasks(phase1, benchmark_root),
    )
    runtime = _local_runtime(config)  # type: ignore[arg-type]
    attempt = _generation_attempt(  # type: ignore[arg-type]
        config, snapshot, config.lane, [render_prompt(tasks[0])]
    )
    passed = (
        attempt["status"] == "passed"
        and attempt["candidate_count"] == 4
        and config.lane["cpu_offload_gb"] == 0
        and config.lane["swap_space"] == 0
    )
    evidence = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "model": config.model,
        "model_snapshot": str(snapshot),
        "prompt_format_id": PROMPT_FORMAT_ID,
        "chat_template": None,
        "prompt_transformation": None,
        "benchmark_environment": environment,
        "runtime": runtime,
        "accepted_lane": LANE_ID if passed else None,
        "quantized_attempt": attempt,
        "fully_gpu_resident_contract": {
            "cpu_weight_offload_gb": 0,
            "kv_swap_space_gb": 0,
            "single_project_gpu": True,
            "hosted_inference": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def _validate_preflight(
    config: Qwen36AssessmentConfig,
    preflight: dict[str, Any],
    snapshot: Path,
) -> None:
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unknown or missing Qwen3.6-27B preflight")
    if preflight.get("status") != "passed":
        raise ValueError("accepted generation requires a passing preflight")
    if preflight.get("config_sha256") != config.digest():
        raise ValueError("preflight config digest differs from current config")
    if preflight.get("model") != config.model:
        raise ValueError("preflight model identity differs from current config")
    if Path(str(preflight.get("model_snapshot"))).resolve() != snapshot:
        raise ValueError("preflight model snapshot differs from requested snapshot")
    if preflight.get("accepted_lane") != LANE_ID:
        raise ValueError("preflight did not accept the frozen 4-bit lane")
    if preflight.get("quantized_attempt", {}).get("status") != "passed":
        raise ValueError("preflight has no successful 4-bit generation attempt")


def _validate_bitsandbytes_contract(config: Qwen36AssessmentConfig) -> None:
    from vllm.model_executor.layers.quantization.bitsandbytes import (
        BitsAndBytesConfig,
    )

    observed = BitsAndBytesConfig.from_config({})
    expected = config.lane["quantization_metadata"]
    fields = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": expected["quant_type"],
        "bnb_4bit_compute_dtype": expected["compute_dtype"],
        "bnb_4bit_quant_storage": expected["quant_storage"],
        "bnb_4bit_use_double_quant": expected["double_quantization"],
        "llm_int8_enable_fp32_cpu_offload": False,
    }
    for key, value in fields.items():
        if getattr(observed, key) != value:
            raise RuntimeError(
                f"vLLM BitsAndBytes default changed for {key}: "
                f"expected {value!r}, got {getattr(observed, key)!r}"
            )


def run_assessment(
    config: Qwen36AssessmentConfig,
    benchmark_root: Path,
    model_snapshot: Path,
    preflight_path: Path,
    workload_id: str,
    output_dir: Path,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    _configure_vllm_environment(config)  # type: ignore[arg-type]
    _validate_bitsandbytes_contract(config)
    if workload_id not in WORKLOADS:
        raise ValueError(f"unknown assessment workload: {workload_id}")
    snapshot = validate_model_snapshot(config, model_snapshot)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_preflight(config, preflight, snapshot)
    phase1 = config.phase1_config()
    tasks = phase1.select_workload(
        workload_id, materialize_benchmark_tasks(phase1, benchmark_root)
    )
    environment_validation = validate_minif2f_environment(
        phase1,
        benchmark_root,
        timeout_seconds=float(config.verifier["timeout_seconds"]),
    )
    runtime = _local_runtime(config)  # type: ignore[arg-type]
    generated, engine_metrics = _generate_candidates(  # type: ignore[arg-type]
        config, snapshot, config.lane, tasks
    )
    runtime.update(engine_metrics)

    verifier = LeanVerifier(
        benchmark_root, timeout_seconds=float(config.verifier["timeout_seconds"])
    )
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=int(config.verifier["verification_workers"])
    ) as executor:
        results = list(
            executor.map(lambda item: _verify_candidate(verifier, item), generated)
        )
    verification_seconds = time.perf_counter() - verification_started
    runtime["verification_wall_time_seconds"] = verification_seconds
    runtime["verification_workers"] = int(config.verifier["verification_workers"])

    summary = summarize_results(
        results,
        expected_task_ids=[task.id for task in tasks],
        candidates_per_task=4,
    )
    token_statistics = _token_statistics(results)
    generation_seconds = float(engine_metrics["generation_wall_time_seconds"])
    load_seconds = float(engine_metrics["engine_load_time_seconds"])
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    summary.update(
        {
            "workload_id": workload_id,
            "generated_tokens": token_statistics,
            "engine_load_time_seconds": load_seconds,
            "generation_wall_time_seconds": generation_seconds,
            "run_wall_time_seconds": (
                load_seconds + generation_seconds + verification_seconds
            ),
            "throughput": {
                "generated_tokens_per_second": (
                    token_statistics["total"] / generation_seconds
                    if generation_seconds
                    else None
                ),
                "candidates_per_second": (
                    len(results) / generation_seconds if generation_seconds else None
                ),
            },
            "compute_per_solved_task": {
                "generated_tokens": (
                    token_statistics["total"] / solved if solved else None
                ),
                "generation_gpu_seconds": (
                    generation_seconds / solved if solved else None
                ),
            },
        }
    )
    environment = environment_validation["verifier_environment"]
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source=(
            f"{phase1.benchmark['repository']}@{phase1.benchmark['revision']}:"
            f"{phase1.benchmark['source_path']}"
        ),
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=str(environment["lean_toolchain"]),
        mathlib_revision=str(environment["dependencies"]["mathlib"]),
        verifier_timeout_seconds=float(config.verifier["timeout_seconds"]),
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=workload_id,
        benchmark_split="validation",
        benchmark_repository=str(phase1.benchmark["repository"]),
        benchmark_revision=str(phase1.benchmark["revision"]),
        verifier_environment=environment,
        candidates_per_task=4,
        inference_engine=str(config.runtime["inference_engine"]),
        inference_engine_version=str(config.runtime["inference_engine_version"]),
        generation_settings={
            **config.sampling,
            **config.lane,
            "chat_template": None,
            "prompt_transformation": None,
            "proof_extraction": None,
            "lean_feedback": None,
            "repair": None,
        },
        runtime={
            **runtime,
            "assessment_id": config.value["assessment_id"],
            "config_sha256": config.digest(),
            "accepted_preflight": str(preflight_path.resolve()),
            "model_snapshot": str(snapshot),
            "inference_execution": "local_cuda",
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
    return metadata, results, summary


def write_compact_evidence(
    config: Qwen36AssessmentConfig,
    preflight_path: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    snapshot = Path(str(preflight.get("model_snapshot"))).resolve()
    _validate_preflight(config, preflight, snapshot)
    dev16 = _compact_run(
        config,
        dev16_dir,
        expected_workload="minif2f-valid-dev16-v1",
        expected_tasks=16,
    )
    full = _compact_run(
        config,
        full_dir,
        expected_workload="minif2f-valid-v1",
        expected_tasks=244,
    )
    reference_path = config.path.parents[1] / str(
        config.value["assessment"]["reference_evidence"]
    )
    reference_source = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = reference_source["adapter"]
    reference_metrics = reference["pass_at_k"]
    strict_metrics = full["pass_at_k"]
    deltas = {
        key: float(strict_metrics[key]) - float(reference_metrics[key])
        for key in ("pass@1", "pass@4")
    }
    if all(value > 0 for value in deltas.values()):
        result = "qwen36_27b_4bit_higher_at_pass1_and_pass4"
    elif all(value < 0 for value in deltas.values()):
        result = "reference_sft_higher_at_pass1_and_pass4"
    else:
        result = "mixed_or_tied_pass1_pass4_result"

    compact_preflight = {
        key: value for key, value in preflight.items() if key != "model_snapshot"
    }
    compact_preflight["model_snapshot"] = {
        "revision": MODEL_REVISION,
        "local_cache_used": True,
        "path_committed": False,
    }
    comparison = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "OBSERVED",
        "assessment_id": config.value["assessment_id"],
        "result": result,
        "strict_lane_label": "Qwen3.6-27B / 4-bit Ada",
        "strict_lane": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "lane_id": LANE_ID,
            "quantization": QUANTIZATION_METADATA,
            "candidates_per_task": 4,
            "pass_at_k": strict_metrics,
        },
        "reference": {
            "logical_id": config.value["assessment"]["reference_logical_id"],
            "candidates_per_task": reference["candidates_per_task"],
            "pass_at_k": {key: reference_metrics[key] for key in ("pass@1", "pass@4")},
            "source": config.value["assessment"]["reference_evidence"],
        },
        "delta_qwen36_4bit_minus_reference": deltas,
        "comparison_limitations": [
            "The Qwen3.6 result is quantized and is not precision-identical to BF16 anchors; score differences cannot be attributed solely to model generation.",
            "The strict lane uses four candidates per task while accepted reference-sft-v1 pass@k estimates use eight.",
        ],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("preflight.json", compact_preflight),
        ("dev16.json", dev16),
        ("full.json", full),
        ("comparison.json", comparison),
    ):
        (evidence_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "README.md").write_text(
        _render_readme(dev16, full, comparison), encoding="utf-8"
    )
    return comparison


def write_blocker_evidence(
    config: Qwen36AssessmentConfig,
    preflight_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unknown or missing Qwen3.6-27B preflight")
    if preflight.get("status") != "failed":
        raise ValueError("hardware-blocker evidence requires a failed preflight")
    if preflight.get("assessment_id") != config.value["assessment_id"]:
        raise ValueError("preflight assessment identity differs from current config")
    if preflight.get("config_sha256") != config.digest():
        raise ValueError("preflight config digest differs from current config")
    if preflight.get("model") != config.model:
        raise ValueError("preflight model identity differs from current config")
    snapshot = Path(str(preflight.get("model_snapshot"))).resolve()
    if snapshot.name != MODEL_REVISION:
        raise ValueError("preflight model snapshot is not the pinned official revision")
    if preflight.get("accepted_lane") is not None:
        raise ValueError("failed preflight must not identify an accepted lane")
    attempt = preflight.get("quantized_attempt", {})
    if (
        attempt.get("status") != "failed"
        or attempt.get("memory_failure") is not True
        or attempt.get("lane") != config.lane
    ):
        raise ValueError("preflight does not establish the frozen-lane memory blocker")
    residency = preflight.get("fully_gpu_resident_contract")
    if residency != {
        "cpu_weight_offload_gb": 0,
        "kv_swap_space_gb": 0,
        "single_project_gpu": True,
        "hosted_inference": False,
    }:
        raise ValueError("preflight residency contract differs from issue #35")
    load_observation = preflight.get("engine_log_observations", {}).get(
        "model_loading_report"
    )
    if (
        not isinstance(load_observation, dict)
        or not isinstance(load_observation.get("seconds"), (float, int))
        or not isinstance(load_observation.get("memory_gib"), (float, int))
        or load_observation.get("source")
        != "vllm engine log emitted during this preflight"
    ):
        raise ValueError("failed preflight is missing the observed vLLM load report")

    compact = {
        key: value for key, value in preflight.items() if key != "model_snapshot"
    }
    compact.update(
        {
            "schema_version": BLOCKER_EVIDENCE_SCHEMA_VERSION,
            "status": "BLOCKED",
            "result": "hardware_infeasible_under_frozen_fully_gpu_resident_lane",
            "failed_gate": "stage0_real_generation",
            "model_snapshot": {
                "revision": MODEL_REVISION,
                "local_cache_used": True,
                "path_committed": False,
            },
            "benchmark_generation_started": False,
            "configuration_changed_after_failure": False,
        }
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "preflight.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence_dir / "README.md").write_text(
        _render_blocker_readme(compact), encoding="utf-8"
    )
    return compact


def _compact_run(
    config: Qwen36AssessmentConfig,
    artifact_dir: Path,
    *,
    expected_workload: str,
    expected_tasks: int,
) -> dict[str, Any]:
    metadata, results = read_artifacts(artifact_dir)
    stored = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    if metadata.workload_id != expected_workload:
        raise ValueError(
            f"unexpected workload in {artifact_dir}: {metadata.workload_id}"
        )
    if (
        metadata.model_id != MODEL_ID
        or metadata.model_revision != MODEL_REVISION
        or metadata.tokenizer_id != MODEL_ID
        or metadata.tokenizer_revision != MODEL_REVISION
    ):
        raise ValueError("artifact model/tokenizer identity differs from issue #35")
    if metadata.prompt_format_id != PROMPT_FORMAT_ID:
        raise ValueError("artifact prompt format differs from whole-proof-v1")
    if metadata.verifier_timeout_seconds != 30.0:
        raise ValueError("artifact verifier timeout differs from 30 seconds")
    if metadata.candidates_per_task != 4:
        raise ValueError("artifact candidate budget differs from four")
    if metadata.runtime.get("config_sha256") != config.digest():
        raise ValueError("artifact config digest differs from current config")
    if metadata.runtime.get("inference_execution") != "local_cuda":
        raise ValueError("artifact inference did not execute on local CUDA")
    settings = metadata.generation_settings or {}
    for key, value in config.sampling.items():
        if settings.get(key) != value:
            raise ValueError(f"artifact sampling mismatch for {key}")
    for key, value in config.lane.items():
        if settings.get(key) != value:
            raise ValueError(f"artifact quantized-lane mismatch for {key}")
    if any(
        settings.get(key) is not None
        for key in (
            "chat_template",
            "prompt_transformation",
            "proof_extraction",
            "lean_feedback",
            "repair",
        )
    ):
        raise ValueError("artifact applied a forbidden prompt/proof transformation")
    expected_ids = list(
        config.value["workloads"][expected_workload].get("task_ids", [])
    )
    if not expected_ids:
        expected_ids = [str(item["task_id"]) for item in stored["per_task"]]
    recomputed = summarize_results(
        results, expected_task_ids=expected_ids, candidates_per_task=4
    )
    if len(expected_ids) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, got {len(expected_ids)}")
    for key in (
        "complete",
        "completeness_errors",
        "task_count",
        "candidate_count",
        "candidates_per_task",
        "tasks_with_verified_candidate",
        "pass_at_k",
        "category_counts",
        "category_fractions",
        "finish_reason_counts",
        "verifier_timeout_count",
        "infrastructure_error_count",
        "per_task",
    ):
        if recomputed[key] != stored[key]:
            raise ValueError(f"stored summary differs from raw results for {key}")
    if not recomputed["complete"] or recomputed["infrastructure_error_count"]:
        raise ValueError("assessment artifacts are incomplete or infrastructure-failed")
    expected_candidates = expected_tasks * 4
    if len(results) != expected_candidates:
        raise ValueError(
            f"expected {expected_candidates} candidates, got {len(results)}"
        )
    token_statistics = _token_statistics(results)
    if token_statistics != stored.get("generated_tokens"):
        raise ValueError("stored generated-token summary differs from raw results")
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "OBSERVED",
        "workload_id": expected_workload,
        "task_count": expected_tasks,
        "candidate_count": expected_candidates,
        "verified_candidate_count": recomputed["category_counts"]["verified"],
        "tasks_with_verified_candidate": recomputed["tasks_with_verified_candidate"],
        "pass_at_k": recomputed["pass_at_k"],
        "category_counts": recomputed["category_counts"],
        "category_fractions": recomputed["category_fractions"],
        "finish_reason_counts": recomputed["finish_reason_counts"],
        "verifier_timeout_count": recomputed["verifier_timeout_count"],
        "verifier_timeout_semantics": "unsuccessful_proof_outcome",
        "infrastructure_error_count": recomputed["infrastructure_error_count"],
        "generated_tokens": token_statistics,
        "latency_seconds": {
            "generation": _numeric_summary(
                item.generation_latency_seconds for item in results
            ),
            "verification": _numeric_summary(
                item.verification_latency_seconds
                for item in results
                if item.verification_latency_seconds is not None
            ),
            "total": _numeric_summary(item.total_latency_seconds for item in results),
        },
        "timing_seconds": stored["timing_seconds"],
        "engine_load_time_seconds": stored["engine_load_time_seconds"],
        "generation_wall_time_seconds": stored["generation_wall_time_seconds"],
        "run_wall_time_seconds": stored["run_wall_time_seconds"],
        "throughput": stored["throughput"],
        "compute_per_solved_task": stored["compute_per_solved_task"],
        "lane_id": settings["lane_id"],
        "precision": settings["dtype"],
        "quantization": settings["quantization"],
        "quantization_metadata": settings["quantization_metadata"],
        "runtime": {
            key: value
            for key, value in metadata.runtime.items()
            if key not in {"model_snapshot", "accepted_preflight"}
        },
        "verifier_environment": metadata.verifier_environment,
        "results_jsonl_sha256": _file_sha256(artifact_dir / "results.jsonl"),
        "candidate_text_sha256": _candidate_text_digest(results),
        "raw_candidates_retained_outside_git": True,
    }


def _numeric_summary(values: Any) -> dict[str, float | int | None]:
    materialized = sorted(float(value) for value in values if value is not None)
    if not materialized:
        return {
            "count": 0,
            "total": 0,
            "minimum": None,
            "mean": None,
            "median": None,
            "maximum": None,
        }
    return {
        "count": len(materialized),
        "total": sum(materialized),
        "minimum": materialized[0],
        "mean": statistics.fmean(materialized),
        "median": statistics.median(materialized),
        "maximum": materialized[-1],
    }


def _candidate_text_digest(results: list[CandidateResult]) -> str:
    digest = hashlib.sha256()
    for result in results:
        digest.update(result.task_id.encode())
        digest.update(b"\0")
        digest.update(str(result.candidate_index).encode())
        digest.update(b"\0")
        digest.update(result.candidate_text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_readme(
    dev16: dict[str, Any], full: dict[str, Any], comparison: dict[str, Any]
) -> str:
    reference = comparison["reference"]["pass_at_k"]
    peak = float(full["runtime"]["peak_device_memory_used_mib"]) / 1024
    return f"""# Qwen3.6-27B 4-bit Ada strict miniF2F assessment

**OBSERVED:** `{comparison["result"]}`. `Qwen3.6-27B / 4-bit Ada` strict pass@1/pass@4 were {full["pass_at_k"]["pass@1"]:.6f}/{full["pass_at_k"]["pass@4"]:.6f}; `reference-sft-v1` pass@1/pass@4 were {reference["pass@1"]:.6f}/{reference["pass@4"]:.6f}.

| Workload | Tasks | Candidates | Verified candidates | pass@1 | pass@4 | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | {dev16["task_count"]} | {dev16["candidate_count"]} | {dev16["verified_candidate_count"]} | {dev16["pass_at_k"]["pass@1"]:.6f} | {dev16["pass_at_k"]["pass@4"]:.6f} | {dev16["infrastructure_error_count"]} | {dev16["verifier_timeout_count"]} |
| full validation | {full["task_count"]} | {full["candidate_count"]} | {full["verified_candidate_count"]} | {full["pass_at_k"]["pass@1"]:.6f} | {full["pass_at_k"]["pass@4"]:.6f} | {full["infrastructure_error_count"]} | {full["verifier_timeout_count"]} |

The accepted lane used online bitsandbytes FP4 4-bit conversion from the pinned official BF16 safetensors, float32 quantized-linear compute with BF16 activations, no prequantized substitute, no CPU weight offload, and no KV swap space. The full run peaked at {peak:.2f} GiB device memory on the project RTX 4000 Ada.

The strict lane preserves the raw `whole-proof-v1` continuation prompt with four candidates per task, temperature 0.8, top-p 0.95, no top-k, 1,024 new tokens, seed 0, and no chat wrapper, proof extraction, Lean feedback, or repair. `verifier_timeout` remains an unsuccessful proof outcome rather than an infrastructure error.

This 4-bit result is not precision-identical to the BF16 anchors, so score differences cannot be attributed solely to model generation. Accepted `reference-sft-v1` metrics also use eight candidates per task rather than four. Raw candidates, weights, caches, and bulky logs remain outside Git.
"""


def _render_blocker_readme(preflight: dict[str, Any]) -> str:
    attempt = preflight["quantized_attempt"]
    load = preflight["engine_log_observations"]["model_loading_report"]
    peak_used = int(attempt["peak_device_memory_used_mib"])
    allocated = int(attempt["peak_cuda_allocated_bytes"])
    reserved = int(attempt["peak_cuda_reserved_bytes"])
    return f"""# Qwen3.6-27B 4-bit Ada feasibility

**BLOCKED:** the frozen, fully GPU-resident Stage 0 lane loaded the official `Qwen/Qwen3.6-27B` revision `{MODEL_REVISION}` but failed before a real generation. vLLM reported that model loading took {load["seconds"]:.6f} seconds and {load["memory_gib"]:.2f} GiB, then reported a CUDA out-of-memory error while requesting another 272 MiB, when only 216.38 MiB remained free.

The attempted lane used online BitsAndBytes FP4 conversion from the pinned official BF16 safetensors with float32 quantized-linear compute, BF16 activations, tensor parallelism 1, a 2,048-token context, four maximum sequences, eager mode, zero CPU weight offload, and zero KV swap space. It ran only on the project NVIDIA RTX 4000 Ada Generation; hosted inference was not used.

Peak observations were {allocated:,} CUDA-allocated bytes, {reserved:,} CUDA-reserved bytes, and {peak_used:,} MiB device memory used. The attempt produced {attempt["candidate_count"]} candidates and {attempt["generated_token_count"]} generated tokens. No dev16 or full-validation benchmark began, and the frozen configuration was not changed after the failure.

This is a hardware-feasibility result, not a model-quality result. Weights, caches, and bulky runtime logs remain outside Git; `preflight.json` retains the exact package, GPU, quantization, memory, and failure metadata while omitting the machine-local cache path.
"""
