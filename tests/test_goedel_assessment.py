from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.goedel_assessment import (
    MODEL_REVISION,
    validate_assessment_contract,
    validate_model_snapshot,
    write_compact_evidence,
)
from qwen_lean.metrics import summarize_results
from qwen_lean.minif2f import Phase1Config
from qwen_lean.schema import CandidateResult, RunMetadata


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/goedel-prover-v2-assessment.json"


def test_frozen_assessment_contract() -> None:
    config = Phase1Config.load(CONFIG)
    validate_assessment_contract(config)
    assert config.sampling["candidates_per_task"] == 4
    assert config.value["verifier"]["verification_workers"] == 1
    assert config.value["goedel_assessment"] == {
        "id": "goedel-prover-v2-8b-strict-casting-v1",
        "prompt_format_id": "whole-proof-v1",
        "raw_continuation": True,
        "chat_template": None,
        "proof_extraction": False,
        "lean_guided_retry": False,
        "self_correction": False,
        "native_lane_run": False,
        "environment_probe_timeout_seconds": 120.0,
        "reference_sft_evidence": "evidence/phase6/minif2f-validation.json",
        "qwen_base_evidence": "evidence/phase1/baseline/summary.json",
    }


def test_contract_rejects_chat_wrapper() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["goedel_assessment"]["chat_template"] = "native"
    config = Phase1Config(path=CONFIG, value=value)
    with pytest.raises(ValueError, match="chat_template"):
        validate_assessment_contract(config)


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    for name in (
        "README.md",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "config.json").write_text(
        json.dumps(
            {"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"}
        ),
        encoding="utf-8",
    )
    shard = "model-00001-of-00001.safetensors"
    (snapshot / shard).write_bytes(b"weights")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard}}), encoding="utf-8"
    )
    return snapshot


def test_model_snapshot_is_bound_to_revision_and_complete(tmp_path: Path) -> None:
    value = validate_model_snapshot(Phase1Config.load(CONFIG), _snapshot(tmp_path))
    assert value["revision"] == MODEL_REVISION
    assert value["weight_shard_count"] == 1
    assert value["weight_bytes"] == len(b"weights")


def _result(task_id: str, index: int, category: str) -> CandidateResult:
    return CandidateResult(
        task_id=task_id,
        candidate_id=f"model-{index}",
        candidate_index=index,
        candidate_text="exact h",
        category=category,  # type: ignore[arg-type]
        lean_exit_code=(
            0
            if category == "verified"
            else None
            if category == "verifier_timeout"
            else 1
        ),
        diagnostics={"stdout": "", "stderr": ""},
        generation_latency_seconds=1.0,
        verification_latency_seconds=2.0,
        total_latency_seconds=3.0,
        generated_token_count=10 + index,
        finish_reason="eos" if index < 3 else "token_limit",
    )


def _write_run(
    config: Phase1Config, output: Path, workload_id: str, ids: list[str]
) -> None:
    results = [
        _result(task_id, index, "verified" if index == 0 else "lean_rejected")
        for task_id in ids
        for index in range(4)
    ]
    metadata = RunMetadata(
        schema_version="phase1-v1",
        candidate_source="model",
        task_source="google-deepmind/miniF2F@revision:MiniF2F/Valid.lean",
        prompt_format_id="whole-proof-v1",
        lean_toolchain="leanprover/lean4:v4.27.0",
        mathlib_revision="mathlib",
        verifier_timeout_seconds=30.0,
        model_id=config.model["model_id"],
        tokenizer_id=config.model["tokenizer_id"],
        model_revision=config.model["model_revision"],
        tokenizer_revision=config.model["tokenizer_revision"],
        workload_id=workload_id,
        benchmark_split="validation",
        benchmark_repository=config.benchmark["repository"],
        benchmark_revision=config.benchmark["revision"],
        candidates_per_task=4,
        inference_engine="vllm",
        inference_engine_version="0.10.2",
        generation_settings={
            **config.sampling,
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
            "inference_execution": "local_cuda",
            "generation_wall_time_seconds": 10.0,
            "verification_wall_time_seconds": 20.0,
        },
    )
    summary = summarize_results(
        results, expected_task_ids=ids, candidates_per_task=4
    )
    summary["workload_id"] = workload_id
    summary["run_wall_time_seconds"] = 30.0
    write_artifacts(output, metadata, results, summary=summary)


def test_compact_evidence_preserves_timeout_as_unsuccessful_proof(
    tmp_path: Path,
) -> None:
    config_value = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    dev_ids = config_value["workloads"]["minif2f-valid-dev16-v1"]["task_ids"]
    full_ids = dev_ids + [f"task-{index}" for index in range(228)]
    config_value["benchmark"]["primary_task_manifest"] = "ids.txt"
    (config_dir / "ids.txt").write_text(
        "\n".join(full_ids) + "\n", encoding="utf-8"
    )
    config_path = config_dir / CONFIG.name
    config_path.write_text(json.dumps(config_value), encoding="utf-8")
    config = Phase1Config.load(config_path)

    (tmp_path / "evidence/phase6").mkdir(parents=True)
    (tmp_path / "evidence/phase1/baseline").mkdir(parents=True)
    (tmp_path / "evidence/phase6/minif2f-validation.json").write_text(
        json.dumps(
            {
                "adapter": {
                    "candidates_per_task": 8,
                    "pass_at_k": {"pass@1": 0.04, "pass@4": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "evidence/phase1/baseline/summary.json").write_text(
        json.dumps(
            {
                "candidates_per_task": 8,
                "pass_at_k": {"pass@1": 0.01, "pass@4": 0.05},
            }
        ),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "goedel-prover-v2-preflight-v1",
                "status": "passed",
                "assessment_id": config.value["goedel_assessment"]["id"],
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "model": config.model,
                "snapshot": {"revision": MODEL_REVISION},
            }
        ),
        encoding="utf-8",
    )
    dev_dir, full_dir = tmp_path / "dev", tmp_path / "full"
    _write_run(config, dev_dir, "minif2f-valid-dev16-v1", dev_ids)
    _write_run(config, full_dir, "minif2f-valid-v1", full_ids)

    results_path = full_dir / "results.jsonl"
    values = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    values[1]["category"] = "verifier_timeout"
    values[1]["lean_exit_code"] = None
    results_path.write_text(
        "\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8"
    )
    results = [CandidateResult.from_dict(value) for value in values]
    summary = summarize_results(
        results, expected_task_ids=full_ids, candidates_per_task=4
    )
    summary["workload_id"] = "minif2f-valid-v1"
    summary["run_wall_time_seconds"] = 30.0
    (full_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    comparison = write_compact_evidence(
        config, preflight, dev_dir, full_dir, tmp_path / "compact"
    )
    full = json.loads(
        (tmp_path / "compact/full.json").read_text(encoding="utf-8")
    )
    assert comparison["status"] == "OBSERVED"
    assert full["summary"]["verifier_timeout_count"] == 1
    assert full["summary"]["infrastructure_error_count"] == 0
    assert (
        full["verifier_timeout_semantics"]
        == "unsuccessful_proof_outcome_not_infrastructure_error"
    )
    assert "one worker" in full["execution_notes"][0]
    assert "host-load-dependent" in comparison["comparison_limitations"][2]
