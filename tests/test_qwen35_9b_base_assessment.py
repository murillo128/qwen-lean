import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.metrics import summarize_results
from qwen_lean.prompt import PROMPT_FORMAT_ID
from qwen_lean.qwen35_9b_base_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    Qwen35BaseAssessmentConfig,
    _compact_run,
    _token_statistics,
    _validate_preflight,
    validate_model_snapshot,
    vllm_engine_kwargs,
    vllm_sampling_kwargs,
    write_compact_evidence,
)
from qwen_lean.schema import CandidateResult, PHASE1_RESULT_SCHEMA_VERSION, RunMetadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-9b-base-assessment.json"


def test_config_freezes_issue_43_contract() -> None:
    config = Qwen35BaseAssessmentConfig.load(CONFIG_PATH)

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
    assert config.value["benchmark"]["split"] == "validation"
    assert config.value["benchmark"]["expected_primary_task_count"] == 244
    assert config.verifier["timeout_seconds"] == 30.0


def test_config_rejects_quality_contract_drift(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["sampling"]["temperature"] = 0.7
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="sampling differs"):
        Qwen35BaseAssessmentConfig.load(path)


def test_engine_kwargs_keep_local_text_only_no_offload_lanes(tmp_path: Path) -> None:
    config = Qwen35BaseAssessmentConfig.load(CONFIG_PATH)
    snapshot = tmp_path / MODEL_REVISION

    bf16 = vllm_engine_kwargs(config, snapshot, config.value["bf16_lane"])
    fallback = vllm_engine_kwargs(config, snapshot, config.value["fallback_lane"])

    assert bf16["model"] == bf16["tokenizer"] == str(snapshot)
    assert bf16["language_model_only"] is True
    assert bf16["cpu_offload_gb"] == 0.0
    assert bf16["quantization"] is None
    assert fallback["language_model_only"] is True
    assert fallback["cpu_offload_gb"] == 0.0
    assert fallback["quantization"] == "bitsandbytes"
    assert fallback["load_format"] == "bitsandbytes"
    assert fallback["trust_remote_code"] is False


def test_sampling_kwargs_are_raw_four_candidate_continuations() -> None:
    config = Qwen35BaseAssessmentConfig.load(CONFIG_PATH)

    assert vllm_sampling_kwargs(config.sampling) == {
        "n": 4,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_tokens": 1024,
        "seed": 0,
        "ignore_eos": False,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
    }


def test_snapshot_requires_pinned_revision_and_all_shards(tmp_path: Path) -> None:
    config = Qwen35BaseAssessmentConfig.load(CONFIG_PATH)
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


def test_fallback_preflight_requires_bf16_memory_failure(tmp_path: Path) -> None:
    config = Qwen35BaseAssessmentConfig.load(CONFIG_PATH)
    snapshot = tmp_path / MODEL_REVISION
    preflight = {
        "schema_version": "qwen35-9b-base-preflight-v1",
        "status": "passed",
        "config_sha256": config.digest(),
        "model": config.model,
        "model_snapshot": str(snapshot),
        "accepted_lane": "bitsandbytes-nf4-online-v1",
        "bf16_attempt": {"status": "failed", "memory_failure": False},
        "fallback_attempt": {"status": "passed"},
    }

    with pytest.raises(ValueError, match="recorded BF16 memory failure"):
        _validate_preflight(config, preflight, snapshot)


def test_token_summary_uses_all_candidates() -> None:
    results = [_result("task", index, tokens=index + 1) for index in range(4)]

    assert _token_statistics(results) == {
        "total": 10,
        "minimum": 1,
        "maximum": 4,
        "mean": 2.5,
        "median": 2.5,
        "p95": 4,
        "p99": 4,
    }


def test_compact_evidence_recomputes_raw_candidate_counts(tmp_path: Path) -> None:
    config = Qwen35BaseAssessmentConfig.load(CONFIG_PATH)
    dev_ids = config.value["workloads"]["minif2f-valid-dev16-v1"]["task_ids"]
    full_ids = [
        line
        for line in (ROOT / "config/minif2f-valid-task-ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "qwen35-9b-base-preflight-v1",
                "status": "passed",
                "assessment_id": config.value["assessment_id"],
                "config_sha256": config.digest(),
                "model": config.model,
                "model_snapshot": str(tmp_path / MODEL_REVISION),
                "accepted_lane": "bitsandbytes-nf4-online-v1",
                "bf16_attempt": {"status": "failed", "memory_failure": True},
                "fallback_attempt": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    _write_run(config, tmp_path / "dev16", "minif2f-valid-dev16-v1", dev_ids)
    _write_run(config, tmp_path / "full", "minif2f-valid-v1", full_ids)

    payload = write_compact_evidence(
        config,
        preflight,
        tmp_path / "dev16",
        tmp_path / "full",
        tmp_path / "evidence",
    )

    assert payload["dev16"]["candidate_count"] == 64
    assert payload["full"]["candidate_count"] == 976
    assert payload["full"]["pass_at_k"] == {"pass@1": 0.0, "pass@4": 0.0}
    assert "model_snapshot" not in payload["full"]["runtime"]
    assert not (tmp_path / "evidence/full.json").read_text().startswith("candidate_text")

    run = json.loads((tmp_path / "full/run.json").read_text(encoding="utf-8"))
    run["verifier_timeout_seconds"] = 31.0
    (tmp_path / "full/run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="timeout differs"):
        _compact_run(
            config,
            tmp_path / "full",
            expected_workload="minif2f-valid-v1",
            expected_tasks=244,
        )


def _result(task_id: str, index: int, *, tokens: int = 2) -> CandidateResult:
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
        generated_token_count=tokens,
        finish_reason="eos",
    )


def _write_run(
    config: Qwen35BaseAssessmentConfig,
    output_dir: Path,
    workload_id: str,
    task_ids: list[str],
) -> None:
    results = [_result(task_id, index) for task_id in task_ids for index in range(4)]
    summary = summarize_results(
        results, expected_task_ids=task_ids, candidates_per_task=4
    )
    summary.update(
        {
            "workload_id": workload_id,
            "generated_tokens": _token_statistics(results),
            "engine_load_time_seconds": 1.0,
            "generation_wall_time_seconds": 2.0,
            "run_wall_time_seconds": 3.0,
            "throughput": {
                "generated_tokens_per_second": len(results),
                "candidates_per_second": len(results) / 2,
            },
            "compute_per_solved_task": {
                "generated_tokens": None,
                "generation_gpu_seconds": None,
            },
        }
    )
    lane = config.value["fallback_lane"]
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
            "lane_id": lane["lane_id"],
            "dtype": lane["dtype"],
            "quantization": lane["quantization"],
            "quantization_metadata": lane["quantization_metadata"],
            "chat_template": None,
            "prompt_transformation": None,
            "proof_extraction": None,
            "lean_feedback": None,
        },
        runtime={
            "config_sha256": config.digest(),
            "model_snapshot": "/cache/model",
            "accepted_preflight": "/artifacts/preflight.json",
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
