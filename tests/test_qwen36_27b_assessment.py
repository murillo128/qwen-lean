import copy
import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.metrics import summarize_results
from qwen_lean.prompt import PROMPT_FORMAT_ID
from qwen_lean.qwen35_9b_base_assessment import vllm_engine_kwargs
from qwen_lean.qwen36_27b_assessment import (
    BLOCKER_EVIDENCE_SCHEMA_VERSION,
    LANE_ID,
    MODEL_ID,
    MODEL_REVISION,
    PREFLIGHT_SCHEMA_VERSION,
    Qwen36AssessmentConfig,
    _token_statistics,
    _validate_preflight,
    validate_model_snapshot,
    write_blocker_evidence,
    write_compact_evidence,
)
from qwen_lean.schema import PHASE1_RESULT_SCHEMA_VERSION, CandidateResult, RunMetadata

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen36-27b-assessment.json"


def test_config_freezes_issue_35_quantized_casting_contract() -> None:
    config = Qwen36AssessmentConfig.load(CONFIG_PATH)

    assert config.model["model_id"] == MODEL_ID
    assert config.model["model_revision"] == MODEL_REVISION
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
    assert config.lane["lane_id"] == LANE_ID
    assert config.lane["quantization"] == "bitsandbytes"
    assert config.lane["quantization_metadata"]["quant_type"] == "fp4"
    assert config.lane["quantization_metadata"]["compute_dtype"] == "float32"
    assert config.lane["max_model_len"] == 2048
    assert config.value["assessment"]["chat_template"] is None


def test_config_rejects_sampling_quantization_or_prompt_drift() -> None:
    config = Qwen36AssessmentConfig.load(CONFIG_PATH)

    for section, key, value in (
        ("sampling", "temperature", 0.7),
        ("quantized_lane", "cpu_offload_gb", 1),
        ("assessment", "chat_template", "native"),
    ):
        changed = copy.deepcopy(config.value)
        changed[section][key] = value
        with pytest.raises(ValueError):
            Qwen36AssessmentConfig(path=config.path, value=changed).validate()


def test_vllm_kwargs_require_text_only_fully_gpu_resident_lane(
    tmp_path: Path,
) -> None:
    config = Qwen36AssessmentConfig.load(CONFIG_PATH)
    snapshot = tmp_path / MODEL_REVISION

    kwargs = vllm_engine_kwargs(config, snapshot, config.lane)  # type: ignore[arg-type]

    assert kwargs["model"] == kwargs["tokenizer"] == str(snapshot)
    assert kwargs["language_model_only"] is True
    assert kwargs["cpu_offload_gb"] == 0.0
    assert kwargs["swap_space"] == 0.0
    assert kwargs["quantization"] == "bitsandbytes"
    assert kwargs["load_format"] == "bitsandbytes"
    assert kwargs["max_model_len"] == 2048
    assert kwargs["max_num_seqs"] == 4


def test_snapshot_requires_pinned_official_revision_and_all_shards(
    tmp_path: Path,
) -> None:
    config = Qwen36AssessmentConfig.load(CONFIG_PATH)
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-1.safetensors"}}),
        encoding="utf-8",
    )
    (snapshot / "model-1.safetensors").write_bytes(b"fixture")

    assert validate_model_snapshot(config, snapshot) == snapshot.resolve()
    (snapshot / "model-1.safetensors").unlink()
    with pytest.raises(ValueError, match="missing weight shards"):
        validate_model_snapshot(config, snapshot)


def test_assessment_requires_exact_passing_preflight(tmp_path: Path) -> None:
    config = Qwen36AssessmentConfig.load(CONFIG_PATH)
    snapshot = tmp_path / MODEL_REVISION
    preflight = _preflight(config, snapshot)

    _validate_preflight(config, preflight, snapshot)
    preflight["quantized_attempt"]["status"] = "failed"
    with pytest.raises(ValueError, match="successful 4-bit"):
        _validate_preflight(config, preflight, snapshot)


def test_compact_evidence_recomputes_counts_and_keeps_quantized_runtime(
    tmp_path: Path,
) -> None:
    config = Qwen36AssessmentConfig.load(CONFIG_PATH)
    snapshot = tmp_path / MODEL_REVISION
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(_preflight(config, snapshot)), encoding="utf-8"
    )
    dev_ids = config.value["workloads"]["minif2f-valid-dev16-v1"]["task_ids"]
    full_ids = [
        line
        for line in (ROOT / "config/minif2f-valid-task-ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]
    _write_run(config, tmp_path / "dev16", "minif2f-valid-dev16-v1", dev_ids)
    _write_run(config, tmp_path / "full", "minif2f-valid-v1", full_ids)

    comparison = write_compact_evidence(
        config,
        preflight_path,
        tmp_path / "dev16",
        tmp_path / "full",
        tmp_path / "evidence",
    )
    full = json.loads((tmp_path / "evidence/full.json").read_text())

    assert comparison["status"] == "OBSERVED"
    assert comparison["strict_lane_label"] == "Qwen3.6-27B / 4-bit Ada"
    assert full["task_count"] == 244
    assert full["candidate_count"] == 976
    assert full["infrastructure_error_count"] == 0
    assert full["runtime"]["peak_device_memory_used_mib"] == 18_000
    assert full["quantization_metadata"]["prequantized_checkpoint"] is False
    assert full["raw_candidates_retained_outside_git"] is True


def test_blocker_evidence_requires_and_redacts_failed_frozen_lane(
    tmp_path: Path,
) -> None:
    config = Qwen36AssessmentConfig.load(CONFIG_PATH)
    snapshot = tmp_path / MODEL_REVISION
    failed = {
        **_preflight(config, snapshot),
        "status": "failed",
        "accepted_lane": None,
        "fully_gpu_resident_contract": {
            "cpu_weight_offload_gb": 0,
            "kv_swap_space_gb": 0,
            "single_project_gpu": True,
            "hosted_inference": False,
        },
        "engine_log_observations": {
            "model_loading_report": {
                "memory_gib": 17.93,
                "seconds": 63.727833,
                "source": "vllm engine log emitted during this preflight",
            }
        },
        "quantized_attempt": {
            "status": "failed",
            "memory_failure": True,
            "lane": config.lane,
            "error": "OutOfMemoryError: CUDA out of memory",
            "candidate_count": 0,
            "generated_token_count": 0,
            "peak_cuda_allocated_bytes": 20_425_753_088,
            "peak_cuda_reserved_bytes": 20_661_141_504,
            "peak_device_memory_used_mib": 19_840,
        },
    }
    preflight_path = tmp_path / "failed.json"
    preflight_path.write_text(json.dumps(failed), encoding="utf-8")

    compact = write_blocker_evidence(config, preflight_path, tmp_path / "evidence")

    assert compact["schema_version"] == BLOCKER_EVIDENCE_SCHEMA_VERSION
    assert compact["status"] == "BLOCKED"
    assert compact["failed_gate"] == "stage0_real_generation"
    assert compact["model_snapshot"] == {
        "revision": MODEL_REVISION,
        "local_cache_used": True,
        "path_committed": False,
    }
    assert str(tmp_path) not in (tmp_path / "evidence/preflight.json").read_text()
    assert "**BLOCKED:**" in (tmp_path / "evidence/README.md").read_text()

    failed["quantized_attempt"]["lane"] = {
        **config.lane,
        "cpu_offload_gb": 1,
    }
    preflight_path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen-lane memory blocker"):
        write_blocker_evidence(config, preflight_path, tmp_path / "changed")


def _preflight(config: Qwen36AssessmentConfig, snapshot: Path) -> dict[str, object]:
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "model": config.model,
        "model_snapshot": str(snapshot),
        "accepted_lane": LANE_ID,
        "quantized_attempt": {"status": "passed"},
    }


def _result(task_id: str, index: int) -> CandidateResult:
    return CandidateResult(
        task_id=task_id,
        candidate_id=f"model-{index}",
        candidate_index=index,
        candidate_text="exact h",
        category="verified" if index == 0 else "lean_rejected",
        lean_exit_code=0 if index == 0 else 1,
        diagnostics={"stdout": "", "stderr": ""},
        generation_latency_seconds=0.2,
        verification_latency_seconds=0.1,
        total_latency_seconds=0.3,
        generated_token_count=10,
        finish_reason="eos",
    )


def _write_run(
    config: Qwen36AssessmentConfig,
    output_dir: Path,
    workload_id: str,
    task_ids: list[str],
) -> None:
    results = [_result(task_id, index) for task_id in task_ids for index in range(4)]
    summary = summarize_results(
        results, expected_task_ids=task_ids, candidates_per_task=4
    )
    tokens = _token_statistics(results)
    summary.update(
        {
            "workload_id": workload_id,
            "generated_tokens": tokens,
            "engine_load_time_seconds": 10.0,
            "generation_wall_time_seconds": 20.0,
            "run_wall_time_seconds": 40.0,
            "throughput": {
                "generated_tokens_per_second": tokens["total"] / 20,
                "candidates_per_second": len(results) / 20,
            },
            "compute_per_solved_task": {
                "generated_tokens": tokens["total"] / len(task_ids),
                "generation_gpu_seconds": 20 / len(task_ids),
            },
        }
    )
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source="fixture",
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain="leanprover/lean4:v4.27.0",
        mathlib_revision="fixture",
        verifier_timeout_seconds=30.0,
        model_id=MODEL_ID,
        tokenizer_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        workload_id=workload_id,
        benchmark_split="validation",
        benchmark_repository="google-deepmind/miniF2F",
        benchmark_revision="fixture",
        verifier_environment={"fixture": True},
        candidates_per_task=4,
        inference_engine="vllm",
        inference_engine_version="0.23.0",
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
            "config_sha256": config.digest(),
            "inference_execution": "local_cuda",
            "cuda_device": "NVIDIA RTX 4000 Ada Generation",
            "cuda_device_total_memory_bytes": 20_000_000_000,
            "peak_device_memory_used_mib": 18_000,
            "engine_load_time_seconds": 10.0,
            "generation_wall_time_seconds": 20.0,
            "verification_wall_time_seconds": 10.0,
            "model_snapshot": "/cache/model",
            "accepted_preflight": "/artifacts/preflight.json",
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
