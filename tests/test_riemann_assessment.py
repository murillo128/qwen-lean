import json
from collections import Counter
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.metrics import summarize_results
from qwen_lean.minif2f import Phase1Config
from qwen_lean.phase3 import render_sft_prompt
from qwen_lean.riemann_assessment import (
    EXPECTED_CANDIDATES,
    EXPECTED_TASKS,
    MODEL_ID,
    MODEL_REVISION,
    VLLM_VERSION,
    WORKLOAD_ID,
    _sha256_file,
    _task_ids_digest,
    load_domain_config,
    load_validation_workload,
    validate_assessment_config,
    write_compact_evidence,
)
from qwen_lean.schema import (
    PHASE1_RESULT_SCHEMA_VERSION,
    CandidateResult,
    RunMetadata,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-4b-base-riemann-assessment.json"
DOMAIN_CONFIG_PATH = ROOT / "config/riemann-domain-breakdown.json"


def test_config_freezes_issue_64_contract() -> None:
    config = Phase1Config.load(CONFIG_PATH)

    validate_assessment_config(config)
    assert config.model == {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template": None,
    }
    assert config.sampling == {
        "candidates_per_task": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_new_tokens": 1024,
        "stop": "tokenizer_eos_or_token_limit",
        "seed": 0,
    }
    assert config.engine["dtype"] == "bfloat16"
    assert config.engine["quantization"] is None
    assert config.engine["language_model_only"] is True
    assert config.value["verifier"]["regenerate_after_lean_feedback"] is False


def test_config_rejects_sampling_drift(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["sampling"]["top_p"] = 0.9
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="sampling.top_p differs"):
        validate_assessment_config(Phase1Config.load(path))


def test_frozen_workload_and_domain_views_are_deterministic() -> None:
    config = Phase1Config.load(CONFIG_PATH)
    domains = load_domain_config(DOMAIN_CONFIG_PATH)

    records, metadata = load_validation_workload(config, ROOT, domains)

    assert len(records) == EXPECTED_TASKS
    assert _task_ids_digest(records) == (
        "746cc51878ba7099a01d00f2ffd3d02965b216856766816cbe62d8af98887f4b"
    )
    assert Counter(item["primary_domain"] for item in metadata) == {
        "shared-formal-prerequisite": 409,
        "real-complex-analysis": 57,
        "broad-number-theory": 54,
        "prime-arithmetic-divisibility": 19,
        "zeta-analytic-number-theory": 17,
    }
    assert Counter(item["direct_relevance_class"] for item in metadata) == {
        None: 471,
        "number-theory-control": 61,
        "premise-1": 10,
        "core": 6,
        "premise-2": 4,
        "source-neighborhood": 3,
        "user-1": 1,
    }
    assert Counter(item["component_inclusion_basis"] for item in metadata) == {
        "component-contains-riemann-bubble": 485,
        "component-contains-number-theory-control": 61,
        "component-contains-core": 10,
    }
    assert all(record.split == "validation" for record in records)
    assert all(
        not record.file_path.startswith("data/riemann/corpora/riemann-near-holdout")
        for record in records
    )
    assert render_sft_prompt(records[0]).startswith(
        "/- Complete the proof below."
    )
    assert "import Mathlib" not in render_sft_prompt(records[0])


def test_compact_evidence_recomputes_all_task_outcomes(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    domains = load_domain_config(DOMAIN_CONFIG_PATH)
    records, _ = load_validation_workload(config, ROOT, domains)
    preflight_path = tmp_path / "preflight.json"
    _write_preflight(config, records, preflight_path)
    artifact_dir = tmp_path / "artifacts"
    results = [
        _result(record.id, index)
        for record in records
        for index in range(4)
    ]
    summary = summarize_results(
        results,
        expected_task_ids=[record.id for record in records],
        candidates_per_task=4,
        ks=(1, 4),
    )
    summary.update(
        {
            "workload_id": WORKLOAD_ID,
            "generated_tokens": {
                "total": EXPECTED_CANDIDATES * 2,
                "minimum": 2,
                "maximum": 2,
                "mean": 2.0,
                "median": 2.0,
                "p95": 2,
                "p99": 2,
            },
            "generation_wall_time_seconds": 10.0,
            "verification_wall_time_seconds": 20.0,
            "run_wall_time_seconds": 30.0,
            "throughput": {
                "generated_tokens_per_second": EXPECTED_CANDIDATES / 5,
                "candidates_per_second": EXPECTED_CANDIDATES / 10,
            },
            "generation_efficiency_per_solved_task": {
                "generated_tokens": None,
                "generation_wall_time_seconds": None,
                "run_wall_time_seconds": None,
            },
        }
    )
    write_artifacts(
        artifact_dir,
        _metadata(config),
        results,
        summary=summary,
    )

    full = write_compact_evidence(
        config,
        ROOT,
        DOMAIN_CONFIG_PATH,
        preflight_path,
        artifact_dir,
        tmp_path / "evidence",
    )

    assert full["workload"]["candidate_count"] == EXPECTED_CANDIDATES
    assert full["overall"]["pass_at_k"] == {"pass@1": 0.0, "pass@4": 0.0}
    assert full["task_outcomes"]["rows"] == EXPECTED_TASKS
    by_domain = {item["value"]: item for item in full["domain_breakdown"]}
    assert by_domain["prime-counting-pnt"]["task_count"] == 0
    assert by_domain["arithmetic-functions"]["task_count"] == 0
    assert by_domain["shared-formal-prerequisite"]["task_count"] == 409
    outcomes = (tmp_path / "evidence/task-outcomes.jsonl").read_text(
        encoding="utf-8"
    )
    assert "candidate_text" not in outcomes
    assert "lean_rejected" in outcomes


def _result(task_id: str, index: int) -> CandidateResult:
    return CandidateResult(
        task_id=task_id,
        candidate_id=f"model-{index}",
        candidate_index=index,
        candidate_text="exact h",
        category="lean_rejected",
        lean_exit_code=1,
        diagnostics={"stdout": "", "stderr": "fixture"},
        generation_latency_seconds=0.2,
        verification_latency_seconds=0.1,
        total_latency_seconds=0.3,
        generated_token_count=2,
        finish_reason="eos",
    )


def _metadata(config: Phase1Config) -> RunMetadata:
    return RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source="fixture",
        prompt_format_id="whole-proof-v1",
        lean_toolchain=config.benchmark["lean_toolchain"],
        mathlib_revision=config.benchmark["revision"],
        verifier_timeout_seconds=30.0,
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=WORKLOAD_ID,
        benchmark_split="validation",
        benchmark_repository=config.benchmark["repository"],
        benchmark_revision=config.benchmark["revision"],
        candidates_per_task=4,
        inference_engine="vllm",
        inference_engine_version=VLLM_VERSION,
        generation_settings={
            **config.sampling,
            "dtype": "bfloat16",
            "quantization": None,
            "language_model_only": True,
            "chat_template": None,
            "prompt_transformation": None,
            "proof_extraction": None,
            "semantic_repair": None,
            "lean_feedback": None,
        },
        runtime={
            "python": "3.12",
            "torch": "fixture",
            "torch_cuda_version": "fixture",
            "package_versions": {"vllm": VLLM_VERSION},
            "inference_execution": "local_cuda",
            "cuda_device": "NVIDIA RTX 4000 Ada Generation",
            "cuda_device_capability": [8, 9],
            "cuda_device_total_memory_bytes": 20_000_000_000,
            "sampling_backend": "vllm_pytorch_native",
            "peak_gpu_memory_mib": 19000,
            "peak_gpu_memory_bytes": 19000 * 1024 * 1024,
            "gpu_memory_sample_count": 10,
            "gpu_memory_sampling_interval_seconds": 1.0,
            "gpu_memory_measurement": "fixture",
            "vllm_environment": {
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
            "candidate_generation_retries": 0,
            "candidate_regenerations_after_lean_feedback": 0,
            "generation_wall_time_seconds": 10.0,
            "verification_wall_time_seconds": 20.0,
        },
    )


def _write_preflight(
    config: Phase1Config, records: list, path: Path
) -> None:
    value = {
        "schema_version": "qwen35-4b-base-riemann-preflight-v1",
        "status": "passed",
        "assessment_id": config.value["assessment"]["id"],
        "config_sha256": _sha256_file(CONFIG_PATH),
        "domain_config_sha256": _sha256_file(DOMAIN_CONFIG_PATH),
        "model": config.model,
        "model_snapshot_revision": MODEL_REVISION,
        "runtime": {
            "inference_execution": "local_cuda",
            "vllm": VLLM_VERSION,
        },
        "workload": {
            "id": WORKLOAD_ID,
            "tasks": EXPECTED_TASKS,
            "candidates_per_task": 4,
            "candidate_count": EXPECTED_CANDIDATES,
            "task_ids_sha256": _task_ids_digest(records),
            "membership_sha256": config.benchmark["membership_sha256"],
            "record_store_sha256": config.benchmark["record_store_sha256"],
            "protected_holdouts_loaded": False,
        },
        "prompt": {"prompt_format_id": "whole-proof-v1"},
        "domain_task_counts": {},
        "mathlib_environment": {},
        "source_identity": {"matched_records": EXPECTED_TASKS},
        "verifier_probe": {
            "known_valid_candidate_category": "accepted",
            "placeholder_candidate_category": "rejected",
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
