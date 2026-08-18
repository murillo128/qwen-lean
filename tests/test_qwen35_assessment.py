from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.baseline import vllm_engine_kwargs
from qwen_lean.metrics import summarize_results
from qwen_lean.minif2f import Phase1Config
from qwen_lean.qwen35_assessment import (
    EVIDENCE_SCHEMA_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    PREFLIGHT_SCHEMA_VERSION,
    generated_token_summary,
    validate_assessment_config,
    write_compact_evidence,
)
from qwen_lean.schema import CandidateResult, PHASE1_RESULT_SCHEMA_VERSION, RunMetadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-2b-base-assessment.json"


def test_config_pins_strict_four_candidate_contract() -> None:
    config = Phase1Config.load(CONFIG_PATH)

    validate_assessment_config(config)

    assert config.model == {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
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


def test_config_rejects_quality_tuning_or_chat_capable_engine() -> None:
    config = Phase1Config.load(CONFIG_PATH)
    tuned = Phase1Config(
        path=config.path,
        value={
            **config.value,
            "sampling": {**config.sampling, "temperature": 0.7},
        },
    )
    with pytest.raises(ValueError, match="sampling contract"):
        validate_assessment_config(tuned)

    multimodal = Phase1Config(
        path=config.path,
        value={
            **config.value,
            "engine": {
                **config.engine,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
            },
        },
    )
    with pytest.raises(ValueError, match="limit_mm_per_prompt"):
        validate_assessment_config(multimodal)


def test_vllm_kwargs_isolate_qwen35_text_only_compatibility() -> None:
    config = Phase1Config.load(CONFIG_PATH)

    kwargs = vllm_engine_kwargs(config, config.sampling, None)

    assert kwargs["revision"] == MODEL_REVISION
    assert kwargs["tokenizer_revision"] == MODEL_REVISION
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["quantization"] is None
    assert kwargs["trust_remote_code"] is False
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}


def test_generated_token_summary_is_complete() -> None:
    results = [
        _result("task", index, "lean_rejected", tokens)
        for index, tokens in enumerate((1, 2, 3, 4))
    ]

    assert generated_token_summary(results) == {
        "count": 4,
        "total": 10,
        "mean": 2.5,
        "median": 2.5,
        "min": 1,
        "max": 4,
    }


def test_compact_evidence_accepts_verifier_timeout_not_infrastructure_error(
    tmp_path: Path,
) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(_preflight(config)), encoding="utf-8"
    )
    dev = tmp_path / "dev16"
    full = tmp_path / "full"
    _write_run(config, dev, "minif2f-valid-dev16-v1", 16, timeout=True)
    _write_run(config, full, "minif2f-valid-v1", 244, timeout=True)

    outputs = write_compact_evidence(
        config, preflight, dev, full, tmp_path / "evidence"
    )

    assert outputs["full"]["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert outputs["full"]["results"]["candidate_count"] == 976
    assert outputs["full"]["results"]["verifier_timeout_count"] == 1
    assert outputs["full"]["results"]["infrastructure_error_count"] == 0
    assert set(outputs["full"]["results"]["pass_at_k"]) == {"pass@1", "pass@4"}
    assert outputs["full"]["timing"]["compute_per_solved_task_available"]
    readme = (tmp_path / "evidence/README.md").read_text(encoding="utf-8")
    assert "No chat template, extraction, repair" in readme
    assert "verifier_timeout" in readme


def test_compact_evidence_rejects_generation_infrastructure_error(
    tmp_path: Path,
) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(_preflight(config)), encoding="utf-8")
    dev = tmp_path / "dev16"
    full = tmp_path / "full"
    _write_run(config, dev, "minif2f-valid-dev16-v1", 16)
    _write_run(config, full, "minif2f-valid-v1", 244, infrastructure_error=True)

    with pytest.raises(ValueError, match="infrastructure errors"):
        write_compact_evidence(config, preflight, dev, full, tmp_path / "evidence")


def _preflight(config: Phase1Config) -> dict[str, object]:
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "complete": True,
        "model": config.model,
        "prompt_format_id": "whole-proof-v1",
        "chat_template": None,
        "prompt_transformation": None,
        "inference": {
            "engine": "vllm",
            "engine_version": "0.17.0",
            "dtype": "bfloat16",
            "quantization": None,
            "text_only_multimodal_limits": {"image": 0, "video": 0},
            "execution": "local_cuda",
        },
        "gpu": {
            "device": "NVIDIA RTX 4000 Ada Generation",
            "peak_gpu_memory_used_bytes": 10,
            "gpu_memory_headroom_at_peak_bytes": 5,
        },
    }


def _write_run(
    config: Phase1Config,
    path: Path,
    workload_id: str,
    task_count: int,
    *,
    timeout: bool = False,
    infrastructure_error: bool = False,
) -> None:
    results: list[CandidateResult] = []
    for task_index in range(task_count):
        for candidate_index in range(4):
            category = (
                "verified"
                if task_index == 0 and candidate_index == 0
                else "lean_rejected"
            )
            if timeout and task_index == task_count - 1 and candidate_index == 3:
                category = "verifier_timeout"
            if (
                infrastructure_error
                and task_index == task_count - 1
                and candidate_index == 3
            ):
                category = "generation_error"
            results.append(_result(f"task-{task_index}", candidate_index, category, 5))

    summary = summarize_results(
        results,
        expected_task_ids=[f"task-{index}" for index in range(task_count)],
        candidates_per_task=4,
    )
    summary["workload_id"] = workload_id
    summary["run_wall_time_seconds"] = 20.0
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source="miniF2F",
        prompt_format_id="whole-proof-v1",
        lean_toolchain="leanprover/lean4:v4.27.0",
        mathlib_revision="mathlib-revision",
        verifier_timeout_seconds=30.0,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        workload_id=workload_id,
        benchmark_split="validation",
        benchmark_repository="google-deepmind/miniF2F",
        benchmark_revision="f0a20e14c1eeccd859d51bb4c2b3ee487889c303",
        candidates_per_task=4,
        inference_engine="vllm",
        inference_engine_version="0.17.0",
        generation_settings={
            **config.sampling,
            "chat_template": None,
            "prompt_transformation": None,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 2048,
            "max_num_seqs": 32,
            "enforce_eager": True,
            "quantization": None,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
        },
        runtime={
            "python": "3.12.14",
            "torch": "2.10.0+cu128",
            "transformers": "4.57.6",
            "torch_cuda_version": "12.8",
            "cuda_device": "NVIDIA RTX 4000 Ada Generation",
            "cuda_device_capability": [8, 9],
            "gpu_driver_version": "580.173.02",
            "gpu_memory_total_bytes": 20,
            "peak_gpu_memory_used_bytes": 10,
            "gpu_memory_headroom_at_peak_bytes": 10,
            "generation_wall_time_seconds": 8.0,
            "verification_wall_time_seconds": 12.0,
            "inference_execution": "local_cuda",
        },
        verifier_environment={"project_revision": "benchmark-revision"},
    )
    write_artifacts(path, metadata, results, summary=summary)


def _result(
    task_id: str,
    candidate_index: int,
    category: str,
    generated_tokens: int,
) -> CandidateResult:
    return CandidateResult(
        task_id=task_id,
        candidate_id=f"model-{candidate_index}",
        candidate_index=candidate_index,
        candidate_text="ring",
        category=category,  # type: ignore[arg-type]
        lean_exit_code=0 if category == "verified" else 1,
        diagnostics={"stdout": "", "stderr": ""},
        generation_latency_seconds=0.2,
        verification_latency_seconds=0.1,
        total_latency_seconds=0.3,
        generated_token_count=generated_tokens,
        finish_reason="eos",
    )
