from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Sequence

from .baseline import (
    LoRAAdapterSpec,
    _generate_candidates,
    _local_cuda_runtime,
    run_phase1_baseline,
)
from .minif2f import Phase1Config
from .phase3 import Phase3Config, load_phase3_workload
from .prompt import normalize_transport
from .schema import PHASE3_RESULT_SCHEMA_VERSION, TaskRecord


MEMORIZATION_SCHEMA_VERSION = "phase3-vllm-memorization-v1"


def adapter_spec(config: Phase3Config, adapter_dir: Path) -> LoRAAdapterSpec:
    return LoRAAdapterSpec(
        adapter_id=str(config.lora["artifact_id"]),
        path=adapter_dir.resolve(),
        rank=int(config.lora["r"]),
        base_model_id=str(config.model["model_id"]),
        base_model_revision=str(config.model["model_revision"]),
    )


def _phase1_config(config: Phase3Config) -> Phase1Config:
    project_root = config.path.parents[1]
    relative = Path(str(config.value["minif2f_smoke"]["phase1_config"]))
    phase1 = Phase1Config.load(project_root / relative)
    expected = {
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
    }
    for key, value in expected.items():
        if phase1.model[key] != value:
            raise ValueError(f"Phase 1 {key} differs from Phase 3")
    return phase1


def _memorization_sampling(config: Phase3Config) -> dict[str, Any]:
    value = config.value["memorization_generation"]
    return {
        "candidates_per_task": int(value["candidates_per_prompt"]),
        "do_sample": bool(value["do_sample"]),
        "temperature": float(value["temperature"]),
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": int(value["max_new_tokens"]),
        "stop": "tokenizer_eos_or_token_limit",
        "seed": int(config.training["seed"]),
    }


def run_vllm_memorization(
    config: Phase3Config,
    workload_path: Path,
    adapter_dir: Path,
    output: Path,
    *,
    optimizer_step: int | None = None,
) -> dict[str, Any]:
    examples, _ = load_phase3_workload(workload_path, config)
    phase1 = _phase1_config(config)
    adapter = adapter_spec(config, adapter_dir)
    adapter.validate(phase1)
    tasks = [
        TaskRecord(
            id=example.record_id,
            preamble="",
            declaration=example.declaration_name,
            declaration_name=example.declaration_name,
        )
        for example in examples
    ]
    sampling = _memorization_sampling(config)
    runtime = _local_cuda_runtime(phase1)
    generated, engine_version = _generate_candidates(
        phase1,
        tasks,
        prompts=[example.prompt for example in examples],
        sampling=sampling,
        adapter=adapter,
    )
    if len(generated) != len(examples):
        raise RuntimeError(
            f"vLLM returned {len(generated)} candidates for {len(examples)} prompts"
        )
    results: list[dict[str, Any]] = []
    exact_matches = 0
    infrastructure_errors = 0
    for example, candidate in zip(examples, generated, strict=True):
        exact = candidate.generation_error is None and normalize_transport(
            candidate.text
        ) == normalize_transport(example.completion)
        exact_matches += int(exact)
        infrastructure_errors += int(candidate.generation_error is not None)
        results.append(
            {
                "record_id": example.record_id,
                "candidate_text": candidate.text,
                "target_completion": example.completion,
                "normalized_exact_match": exact,
                "generated_token_count": candidate.token_count,
                "finish_reason": candidate.finish_reason,
                "generation_latency_seconds": candidate.generation_latency_seconds,
                "generation_error": candidate.generation_error,
            }
        )
    minimum = int(config.value["memorization_generation"]["minimum_exact_matches"])
    passed = infrastructure_errors == 0 and exact_matches >= minimum
    value = {
        "schema_version": MEMORIZATION_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "pipeline_interpretation": "training-workload memorization check, not generalization",
        "model": config.model,
        "adapter": adapter.metadata(),
        "serialization_id": config.value["serialization"]["id"],
        "workload_id": config.workload["id"],
        "selected_record_ids": list(config.selected_record_ids),
        "optimizer_step": optimizer_step,
        "generation_settings": sampling,
        "inference_engine": phase1.engine["name"],
        "inference_engine_version": engine_version,
        "runtime": {"python": platform.python_version(), **runtime},
        "examples": len(examples),
        "exact_matches": exact_matches,
        "minimum_exact_matches": minimum,
        "exact_match_rate": exact_matches / len(examples),
        "generation_infrastructure_errors": infrastructure_errors,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            f"Phase 3 vLLM memorization failed: exact={exact_matches}/64, "
            f"infrastructure_errors={infrastructure_errors}"
        )
    return value


def _smoke_sampling(config: Phase3Config) -> dict[str, Any]:
    value = config.value["minif2f_smoke"]
    return {
        "candidates_per_task": int(value["candidates_per_task"]),
        "do_sample": bool(value["do_sample"]),
        "temperature": float(value["temperature"]),
        "top_p": float(value["top_p"]),
        "top_k": int(value["top_k"]),
        "max_new_tokens": int(value["max_new_tokens"]),
        "stop": "tokenizer_eos_or_token_limit",
        "seed": int(value["seed"]),
    }


def run_adapter_minif2f_smoke(
    config: Phase3Config,
    benchmark_root: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    timeout_seconds: float,
    verification_workers: int,
) -> tuple[Any, Sequence[Any], dict[str, Any]]:
    phase1 = _phase1_config(config)
    workload_id = str(config.value["minif2f_smoke"]["workload_id"])
    metadata, results, summary = run_phase1_baseline(
        phase1,
        benchmark_root,
        workload_id,
        output_dir,
        timeout_seconds=timeout_seconds,
        verification_workers=verification_workers,
        sampling_override=_smoke_sampling(config),
        adapter=adapter_spec(config, adapter_dir),
        result_schema_version=PHASE3_RESULT_SCHEMA_VERSION,
    )
    infrastructure_errors = sum(
        result.category in {"generation_error", "verifier_error"} for result in results
    )
    passed = len(results) == 16 and infrastructure_errors == 0
    summary.update(
        {
            "phase3_pipeline_smoke": True,
            "phase1_quality_comparable": False,
            "adapter_enabled": True,
            "adapter_id": config.lora["artifact_id"],
            "expected_candidates": 16,
            "observed_candidates": len(results),
            "infrastructure_errors": infrastructure_errors,
            "phase3_smoke_passed": passed,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            f"Phase 3 miniF2F smoke failed: candidates={len(results)}, "
            f"infrastructure_errors={infrastructure_errors}"
        )
    return metadata, results, summary
