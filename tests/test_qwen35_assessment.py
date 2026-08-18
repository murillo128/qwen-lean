import copy
import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.baseline import vllm_engine_kwargs
from qwen_lean.metrics import summarize_results
from qwen_lean.qwen35_assessment import (
    DEV16_WORKLOAD_ID,
    FULL_WORKLOAD_ID,
    MODEL_ID,
    MODEL_REVISION,
    PREFLIGHT_WORKLOAD_ID,
    load_assessment_config,
    validate_assessment_config,
    write_compact_evidence,
)
from qwen_lean.schema import CandidateResult, PHASE1_RESULT_SCHEMA_VERSION, RunMetadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-4b-assessment.json"


def test_qwen35_config_freezes_cross_model_casting_contract() -> None:
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
    assert config.engine["dtype"] == "bfloat16"
    assert config.engine["quantization"] is None
    assert config.engine["language_model_only"] is True
    assert config.engine["use_flashinfer_sampler"] is False
    assert config.value["assessment"]["chat_template"] is None

    kwargs = vllm_engine_kwargs(config, config.sampling, None)
    assert kwargs["revision"] == MODEL_REVISION
    assert kwargs["tokenizer_revision"] == MODEL_REVISION
    assert kwargs["language_model_only"] is True


def test_qwen35_config_rejects_sampling_or_prompt_drift() -> None:
    config = load_assessment_config(CONFIG_PATH)
    changed = copy.deepcopy(config.value)
    changed["sampling"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="sampling differs"):
        validate_assessment_config(type(config)(path=config.path, value=changed))

    changed = copy.deepcopy(config.value)
    changed["assessment"]["chat_template"] = "native"
    with pytest.raises(ValueError, match="chat_template"):
        validate_assessment_config(type(config)(path=config.path, value=changed))


def test_qwen35_compact_evidence_checks_counts_and_keeps_runtime_metrics(
    tmp_path: Path,
) -> None:
    config = load_assessment_config(CONFIG_PATH)
    preflight_dir = tmp_path / "preflight"
    dev16_dir = tmp_path / "dev16"
    full_dir = tmp_path / "full"
    _write_run(config, preflight_dir, PREFLIGHT_WORKLOAD_ID, 1, 1)
    _write_run(config, dev16_dir, DEV16_WORKLOAD_ID, 16, 4)
    _write_run(config, full_dir, FULL_WORKLOAD_ID, 244, 4)
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"summary": {"pass_at_k": {"pass@1": 0.04, "pass@4": 0.1}}}),
        encoding="utf-8",
    )

    comparison = write_compact_evidence(
        config,
        preflight_dir=preflight_dir,
        dev16_dir=dev16_dir,
        full_dir=full_dir,
        reference_summary_path=reference,
        evidence_dir=tmp_path / "evidence",
    )
    full = json.loads((tmp_path / "evidence/full.json").read_text())

    assert comparison["status"] == "passed"
    assert comparison["strict_execution_integrity"]["candidate_count"] == 976
    assert full["generated_token_counts"]["total"] == 9760
    assert full["throughput"]["generated_tokens_per_second"] == pytest.approx(976)
    assert full["runtime"]["gpu_memory_peak_bytes"] == 12_000_000_000
    assert full["verifier_timeout_semantics"] == "unsuccessful_proof_outcome"


def _write_run(
    config: object,
    output_dir: Path,
    workload_id: str,
    task_count: int,
    candidates_per_task: int,
) -> None:
    task_ids = [f"task-{index}" for index in range(task_count)]
    results = [
        CandidateResult(
            task_id=task_id,
            candidate_id=f"model-{candidate_index}",
            candidate_index=candidate_index,
            candidate_text="exact True.intro",
            category=(
                "verified"
                if task_index == 0 and candidate_index == 0
                else "lean_rejected"
            ),
            lean_exit_code=0 if task_index == 0 and candidate_index == 0 else 1,
            diagnostics={"stdout": "", "stderr": ""},
            generation_latency_seconds=0.2,
            verification_latency_seconds=0.1,
            total_latency_seconds=0.3,
            generated_token_count=10,
            finish_reason="eos",
        )
        for task_index, task_id in enumerate(task_ids)
        for candidate_index in range(candidates_per_task)
    ]
    summary = summarize_results(
        results,
        expected_task_ids=task_ids,
        candidates_per_task=candidates_per_task,
    )
    summary["workload_id"] = workload_id
    summary["run_wall_time_seconds"] = 20.0
    generation_settings = {
        **getattr(config, "sampling"),
        "candidates_per_task": candidates_per_task,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "quantization": None,
        "language_model_only": True,
        "use_flashinfer_sampler": False,
        "chat_template": None,
        "prompt_transformation": None,
        "adapter": None,
    }
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
        candidates_per_task=candidates_per_task,
        inference_engine="vllm",
        inference_engine_version=getattr(config, "engine")["version"],
        generation_settings=generation_settings,
        runtime={
            "python": "3.12.14",
            "torch": "2.13.0+cu132",
            "torch_cuda_version": "13.2",
            "package_versions": {
                "flashinfer-python": "0.6.12",
                "nvidia-ml-py": "13.580.65",
                "torch": "2.13.0+cu132",
                "transformers": "5.15.0",
                "vllm": "0.23.0",
            },
            "inference_execution": "local_cuda",
            "cuda_device_index": 0,
            "cuda_device": "NVIDIA RTX 4000 Ada Generation",
            "cuda_device_capability": [8, 9],
            "cuda_device_total_memory_bytes": 20_000_000_000,
            "gpu_memory_monitoring": "nvml_device_used_bytes",
            "sampling_backend": "vllm_pytorch_native",
            "gpu_memory_before_bytes": 2_000_000,
            "gpu_memory_peak_bytes": 12_000_000_000,
            "gpu_memory_peak_delta_bytes": 11_998_000_000,
            "generation_wall_time_seconds": 10.0,
            "verification_wall_time_seconds": 10.0,
            "verification_workers": 8,
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
