import copy
import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.baseline import vllm_engine_kwargs
from qwen_lean.deepseek_prover_assessment import (
    DEV16_WORKLOAD_ID,
    FULL_WORKLOAD_ID,
    MODEL_ID,
    MODEL_REVISION,
    load_assessment_config,
    validate_assessment_config,
    write_compact_evidence,
)
from qwen_lean.metrics import summarize_results
from qwen_lean.schema import CandidateResult, PHASE1_RESULT_SCHEMA_VERSION, RunMetadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/deepseek-prover-v2-7b-assessment.json"


def test_config_freezes_strict_specialist_contract() -> None:
    config = load_assessment_config(CONFIG_PATH)

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
    assert config.value["assessment"] == {
        "prompt_format_id": "whole-proof-v1",
        "chat_template": None,
        "proof_extraction": False,
        "verifier_feedback": False,
        "repair": False,
        "native_mode_diagnostic": False,
        "environment_probe_timeout_seconds": 120.0,
        "model_license": "MIT",
    }

    kwargs = vllm_engine_kwargs(config, config.sampling, None)
    assert kwargs["revision"] == MODEL_REVISION
    assert kwargs["tokenizer_revision"] == MODEL_REVISION
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["quantization"] is None
    assert kwargs["trust_remote_code"] is False


def test_config_rejects_sampling_prompt_and_runtime_drift() -> None:
    config = load_assessment_config(CONFIG_PATH)
    changed = copy.deepcopy(config.value)
    changed["sampling"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="sampling differs"):
        validate_assessment_config(type(config)(path=config.path, value=changed))

    changed = copy.deepcopy(config.value)
    changed["assessment"]["chat_template"] = "native"
    with pytest.raises(ValueError, match="chat_template"):
        validate_assessment_config(type(config)(path=config.path, value=changed))

    changed = copy.deepcopy(config.value)
    changed["engine"]["max_num_seqs"] = 32
    with pytest.raises(ValueError, match="max_num_seqs"):
        validate_assessment_config(type(config)(path=config.path, value=changed))


def test_compact_evidence_checks_counts_and_uses_accepted_anchors(
    tmp_path: Path,
) -> None:
    config = load_assessment_config(CONFIG_PATH)
    dev16_dir = tmp_path / "dev16"
    full_dir = tmp_path / "full"
    _write_strict_run(config, dev16_dir, DEV16_WORKLOAD_ID, 16)
    _write_strict_run(config, full_dir, FULL_WORKLOAD_ID, 244)

    comparison = write_compact_evidence(
        config,
        dev16_dir=dev16_dir,
        full_dir=full_dir,
        base_dir=ROOT / "evidence/phase1/baseline",
        reference_path=ROOT / "evidence/phase5/minif2f.json",
        qwen3_posttrained_path=ROOT / "evidence/qwen3-8b-posttrained/full.json",
        qwen35_4b_base_path=ROOT / "evidence/qwen35-4b-base/full.json",
        goedel_path=tmp_path / "no-accepted-goedel.json",
        evidence_dir=tmp_path / "evidence",
    )
    full = json.loads((tmp_path / "evidence/full.json").read_text())

    assert comparison["status"] == "passed"
    assert comparison["anchors_regenerated"] is False
    assert comparison["strict_execution_integrity"]["candidate_count"] == 976
    assert comparison["metrics"]["pass@1"]["deepseek_prover_v2_7b"] == 0.25
    assert comparison["metrics"]["pass@4"]["qwen35_4b_base"] == pytest.approx(
        0.18442622950819673
    )
    assert comparison["accepted_anchors"]["reference_sft_v1"]["id"] == (
        "reference-sft-v1"
    )
    assert comparison["goedel_comparison"]["status"] == "unavailable"
    assert "goedel_prover_v2_8b" not in comparison["metrics"]["pass@1"]
    assert "shared host" in comparison["execution_limitations"][0]
    assert "shared host" in full["execution_limitations"][0]
    assert full["generated_token_counts"]["total"] == 9760
    assert full["verifier_timeout_semantics"] == "unsuccessful_proof_outcome"


def _write_strict_run(
    config: object,
    output_dir: Path,
    workload_id: str,
    task_count: int,
) -> None:
    task_ids = [f"task-{index}" for index in range(task_count)]
    results = [
        CandidateResult(
            task_id=task_id,
            candidate_id=f"model-{candidate_index}",
            candidate_index=candidate_index,
            candidate_text="exact True.intro",
            category="verified" if candidate_index == 0 else "lean_rejected",
            lean_exit_code=0 if candidate_index == 0 else 1,
            diagnostics={"stdout": "", "stderr": ""},
            generation_latency_seconds=0.2,
            verification_latency_seconds=0.1,
            total_latency_seconds=0.3,
            generated_token_count=10,
            finish_reason="eos",
        )
        for task_id in task_ids
        for candidate_index in range(4)
    ]
    summary = summarize_results(
        results, expected_task_ids=task_ids, candidates_per_task=4
    )
    summary["workload_id"] = workload_id
    summary["run_wall_time_seconds"] = 20.0
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source="google-deepmind/miniF2F@revision:MiniF2F/Valid.lean",
        prompt_format_id="whole-proof-v1",
        lean_toolchain="leanprover/lean4:v4.27.0",
        mathlib_revision="a3a10db0e9d66acbebf76c5e6a135066525ac900",
        verifier_timeout_seconds=30.0,
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=workload_id,
        benchmark_split="validation",
        benchmark_repository="google-deepmind/miniF2F",
        benchmark_revision="f0a20e14c1eeccd859d51bb4c2b3ee487889c303",
        verifier_environment={"dependencies": {"mathlib": "revision"}},
        candidates_per_task=4,
        inference_engine="vllm",
        inference_engine_version=getattr(config, "engine")["version"],
        generation_settings={
            **getattr(config, "sampling"),
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": 2048,
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.95,
            "enforce_eager": True,
            "quantization": None,
            "chat_template": None,
            "prompt_transformation": None,
            "adapter": None,
        },
        runtime={
            "python": "3.12.14",
            "torch": "2.8.0+cu128",
            "torch_cuda_version": "12.8",
            "inference_execution": "local_cuda",
            "cuda_device_index": 0,
            "cuda_device": "NVIDIA RTX 4000 Ada Generation",
            "cuda_device_capability": [8, 9],
            "cuda_device_total_memory_bytes": 20_000_000_000,
            "generation_wall_time_seconds": 10.0,
            "verification_wall_time_seconds": 10.0,
            "verification_workers": 8,
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
