from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.baseline import GeneratedCandidate, vllm_engine_kwargs
from qwen_lean.metrics import summarize_results
from qwen_lean.olmo3_assessment import (
    DEV16_WORKLOAD_ID,
    FULL_WORKLOAD_ID,
    MODEL_ID,
    MODEL_REVISION,
    VLLM_VERSION,
    load_assessment_config,
    run_preflight,
    validate_assessment_config,
    write_compact_evidence,
)
from qwen_lean.schema import (
    PHASE1_RESULT_SCHEMA_VERSION,
    CandidateResult,
    RunMetadata,
    TaskRecord,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/olmo3-7b-assessment.json"


def test_config_freezes_strict_olmo_contract() -> None:
    config = load_assessment_config(CONFIG_PATH)

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
    kwargs = vllm_engine_kwargs(config, config.sampling, None)
    assert kwargs["revision"] == MODEL_REVISION
    assert kwargs["tokenizer_revision"] == MODEL_REVISION
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["quantization"] is None
    assert kwargs["trust_remote_code"] is False
    assert "language_model_only" not in kwargs
    assert config.engine["vllm_enable_v1_multiprocessing"] is False


def test_config_rejects_model_sampling_and_runtime_drift() -> None:
    config = load_assessment_config(CONFIG_PATH)
    for section, key, changed_value, match in (
        ("model", "model_revision", "floating", "model.model_revision"),
        ("sampling", "top_p", 0.9, "sampling differs"),
        ("engine", "dtype", "float16", "engine.dtype"),
        ("assessment", "chat_template", "native", "chat_template"),
    ):
        changed = copy.deepcopy(config.value)
        changed[section][key] = changed_value
        with pytest.raises(ValueError, match=match):
            validate_assessment_config(type(config)(path=config.path, value=changed))


def test_preflight_records_local_bf16_peak_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qwen_lean.olmo3_assessment as module

    config = load_assessment_config(CONFIG_PATH)
    runtime = {
        "python": "3.12.14",
        "torch": "2.9.0",
        "torch_cuda_version": "12.8",
        "package_versions": {
            "huggingface-hub": "1.27.0",
            "nvidia-ml-py": "13.610.43",
            "transformers": "4.57.3",
            "vllm": VLLM_VERSION,
        },
        "inference_execution": "local_cuda",
        "cuda_device_index": 0,
        "cuda_device": "NVIDIA RTX 4000 Ada Generation",
        "cuda_device_capability": [8, 9],
        "cuda_device_total_memory_bytes": 20_000_000_000,
        "sampling_backend": "vllm_pytorch_native",
    }

    class Monitor:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> dict[str, object]:
            return {
                "gpu_memory_monitoring": "nvml_device_used_bytes",
                "gpu_memory_peak_bytes": 18_000_000_000,
            }

    def generate(
        _: object, tasks: list[TaskRecord], **__: object
    ) -> tuple[list[GeneratedCandidate], str]:
        return [
            GeneratedCandidate(
                task=tasks[0],
                candidate_index=0,
                text="exact rfl",
                token_count=3,
                finish_reason="eos",
                generation_latency_seconds=0.1,
            )
        ], VLLM_VERSION

    monkeypatch.setattr(module, "_local_cuda_runtime", lambda _: runtime)
    monkeypatch.setattr(module, "_GpuMemoryMonitor", Monitor)
    monkeypatch.setattr(module, "_generate_candidates", generate)
    output = tmp_path / "preflight.json"
    evidence = run_preflight(config, output)

    assert evidence["status"] == "passed"
    assert evidence["runtime"]["dtype"] == "bfloat16"
    assert evidence["runtime"]["gpu_memory_peak_bytes"] == 18_000_000_000
    assert evidence["runtime"]["gpu_memory_headroom_at_peak_bytes"] == 2_000_000_000
    assert json.loads(output.read_text()) == evidence


def test_compact_evidence_requires_complete_strict_runs(tmp_path: Path) -> None:
    config = load_assessment_config(CONFIG_PATH)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(_preflight(config.model), indent=2) + "\n", encoding="utf-8"
    )
    dev16_dir = tmp_path / "dev16"
    full_dir = tmp_path / "full"
    _write_run(config, dev16_dir, DEV16_WORKLOAD_ID, 16)
    _write_run(config, full_dir, FULL_WORKLOAD_ID, 244)

    outputs = write_compact_evidence(
        config,
        preflight_path=preflight_path,
        dev16_dir=dev16_dir,
        full_dir=full_dir,
        evidence_dir=tmp_path / "evidence",
    )

    assert outputs["full"]["candidate_count"] == 976
    assert outputs["full"]["pass_at_k"] == pytest.approx(
        {"pass@1": 0.25, "pass@4": 1.0}
    )
    assert outputs["full"]["runtime"]["gpu_memory_headroom_at_peak_bytes"] == (
        2_000_000_000
    )
    assert "No chat template" in (tmp_path / "evidence/README.md").read_text()


def _preflight(model: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "olmo3-7b-preflight-v1",
        "status": "passed",
        "model": model,
        "prompt_format_id": "whole-proof-v1",
        "raw_continuation": True,
        "chat_template": None,
        "runtime": {
            "inference_execution": "local_cuda",
            "inference_engine": "vllm",
            "inference_engine_version": VLLM_VERSION,
            "dtype": "bfloat16",
            "quantization": None,
            "vllm_enable_v1_multiprocessing": False,
            "cuda_device": "NVIDIA RTX 4000 Ada Generation",
            "gpu_memory_peak_bytes": 18_000_000_000,
            "gpu_memory_headroom_at_peak_bytes": 2_000_000_000,
            "package_versions": {
                "huggingface-hub": "1.27.0",
                "nvidia-ml-py": "13.610.43",
                "transformers": "4.57.3",
                "vllm": VLLM_VERSION,
            },
        },
        "probe": {"generated_token_count": 3, "finish_reason": "eos"},
    }


def _write_run(
    config: object, output_dir: Path, workload_id: str, task_count: int
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
        inference_engine_version=VLLM_VERSION,
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
            "torch": "2.9.0",
            "torch_cuda_version": "12.8",
            "package_versions": {
                "huggingface-hub": "1.27.0",
                "nvidia-ml-py": "13.610.43",
                "transformers": "4.57.3",
                "vllm": VLLM_VERSION,
            },
            "inference_execution": "local_cuda",
            "cuda_device_index": 0,
            "cuda_device": "NVIDIA RTX 4000 Ada Generation",
            "cuda_device_capability": [8, 9],
            "cuda_device_total_memory_bytes": 20_000_000_000,
            "gpu_memory_peak_bytes": 18_000_000_000,
            "generation_wall_time_seconds": 10.0,
            "verification_wall_time_seconds": 10.0,
            "verification_workers": 8,
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
