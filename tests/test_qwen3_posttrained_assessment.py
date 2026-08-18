import copy
import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.baseline import vllm_engine_kwargs
from qwen_lean.metrics import summarize_results
from qwen_lean.qwen3_posttrained_assessment import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    DEV16_WORKLOAD_ID,
    FULL_WORKLOAD_ID,
    MODEL_ID,
    MODEL_REVISION,
    REFERENCE_ADAPTER_ID,
    load_assessment_config,
    validate_assessment_config,
    write_compact_evidence,
)
from qwen_lean.schema import CandidateResult, PHASE1_RESULT_SCHEMA_VERSION, RunMetadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen3-8b-posttrained-assessment.json"


def test_config_freezes_strict_cross_model_contract() -> None:
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
        "model_license": "Apache-2.0",
    }

    kwargs = vllm_engine_kwargs(config, config.sampling, None)
    assert kwargs["revision"] == MODEL_REVISION
    assert kwargs["tokenizer_revision"] == MODEL_REVISION
    assert kwargs["quantization"] is None


def test_config_rejects_sampling_and_prompt_drift() -> None:
    config = load_assessment_config(CONFIG_PATH)
    changed = copy.deepcopy(config.value)
    changed["sampling"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="sampling differs"):
        validate_assessment_config(type(config)(path=config.path, value=changed))

    changed = copy.deepcopy(config.value)
    changed["assessment"]["chat_template"] = "native"
    with pytest.raises(ValueError, match="chat_template"):
        validate_assessment_config(type(config)(path=config.path, value=changed))


def test_compact_evidence_checks_counts_and_compares_both_anchors(
    tmp_path: Path,
) -> None:
    config = load_assessment_config(CONFIG_PATH)
    dev16_dir = tmp_path / "dev16"
    full_dir = tmp_path / "full"
    _write_strict_run(config, dev16_dir, DEV16_WORKLOAD_ID, 16)
    _write_strict_run(config, full_dir, FULL_WORKLOAD_ID, 244)
    base_dir = tmp_path / "base"
    _write_base_anchor(base_dir)
    reference_path = tmp_path / "reference.json"
    _write_reference_anchor(reference_path)

    comparison = write_compact_evidence(
        config,
        dev16_dir=dev16_dir,
        full_dir=full_dir,
        base_dir=base_dir,
        reference_path=reference_path,
        evidence_dir=tmp_path / "evidence",
    )
    full = json.loads((tmp_path / "evidence/full.json").read_text())

    assert comparison["status"] == "passed"
    assert comparison["anchors_regenerated"] is False
    assert comparison["strict_execution_integrity"]["candidate_count"] == 976
    assert comparison["metrics"]["pass@1"] == pytest.approx(
        {
            "qwen3_8b_posttrained": 0.25,
            "qwen3_8b_base": 0.01,
            "reference_sft_v1": 0.04,
            "delta_posttrained_minus_base": 0.24,
            "delta_posttrained_minus_reference": 0.21,
            "fraction_of_base": 25.0,
            "fraction_of_reference": 6.25,
        }
    )
    assert full["generated_token_counts"]["total"] == 9760
    assert full["throughput"]["generated_tokens_per_second"] == pytest.approx(976)
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
            "max_num_seqs": 32,
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


def _write_base_anchor(output_dir: Path) -> None:
    output_dir.mkdir()
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source="source",
        prompt_format_id="whole-proof-v1",
        lean_toolchain="leanprover/lean4:v4.27.0",
        mathlib_revision="revision",
        verifier_timeout_seconds=30.0,
        model_id=BASE_MODEL_ID,
        tokenizer_id=BASE_MODEL_ID,
        model_revision=BASE_MODEL_REVISION,
        tokenizer_revision=BASE_MODEL_REVISION,
        workload_id=FULL_WORKLOAD_ID,
        candidates_per_task=8,
    )
    (output_dir / "run.json").write_text(json.dumps(metadata.to_dict()))
    (output_dir / "summary.json").write_text(
        json.dumps(_anchor_summary({"pass@1": 0.01, "pass@4": 0.05}))
    )


def _write_reference_anchor(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_id": BASE_MODEL_ID,
                "model_revision": BASE_MODEL_REVISION,
                "adapter": {"artifact_id": REFERENCE_ADAPTER_ID},
                "summary": {
                    **_anchor_summary({"pass@1": 0.04, "pass@4": 0.1}),
                    "adapter_enabled": True,
                },
            }
        )
    )


def _anchor_summary(metrics: dict[str, float]) -> dict[str, object]:
    return {
        "complete": True,
        "task_count": 244,
        "candidate_count": 1952,
        "candidates_per_task": 8,
        "infrastructure_error_count": 0,
        "pass_at_k": metrics,
    }
