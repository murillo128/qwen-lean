from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .artifacts import read_artifacts
from .baseline import (
    _local_cuda_runtime,
    run_phase1_baseline,
    validate_minif2f_environment,
)
from .metrics import summarize_results
from .minif2f import Phase1Config, materialize_benchmark_tasks
from .prompt import PROMPT_FORMAT_ID, render_prompt
from .schema import CandidateResult, RunMetadata


PREFLIGHT_SCHEMA_VERSION = "goedel-prover-v2-preflight-v1"
EVIDENCE_SCHEMA_VERSION = "goedel-prover-v2-assessment-v1"
MODEL_ID = "Goedel-LM/Goedel-Prover-V2-8B"
MODEL_REVISION = "dfd02e6271a58375dfbf3ece0175277cf6b6a89a"
WORKLOADS = ("minif2f-valid-dev16-v1", "minif2f-valid-v1")


def validate_assessment_contract(config: Phase1Config) -> None:
    assessment = config.value.get("goedel_assessment")
    if not isinstance(assessment, dict):
        raise ValueError("missing goedel_assessment contract")
    expected = {
        ("model", "model_id"): MODEL_ID,
        ("model", "model_revision"): MODEL_REVISION,
        ("model", "tokenizer_id"): MODEL_ID,
        ("model", "tokenizer_revision"): MODEL_REVISION,
        ("sampling", "candidates_per_task"): 4,
        ("sampling", "do_sample"): True,
        ("sampling", "temperature"): 0.8,
        ("sampling", "top_p"): 0.95,
        ("sampling", "top_k"): -1,
        ("sampling", "max_new_tokens"): 1024,
        ("sampling", "stop"): "tokenizer_eos_or_token_limit",
        ("sampling", "seed"): 0,
        ("engine", "name"): "vllm",
        ("engine", "version"): "0.10.2",
        ("engine", "dtype"): "bfloat16",
        ("engine", "quantization"): None,
        ("engine", "max_model_len"): 2048,
        ("verifier", "verification_workers"): 1,
        ("goedel_assessment", "prompt_format_id"): PROMPT_FORMAT_ID,
        ("goedel_assessment", "raw_continuation"): True,
        ("goedel_assessment", "chat_template"): None,
        ("goedel_assessment", "proof_extraction"): False,
        ("goedel_assessment", "lean_guided_retry"): False,
        ("goedel_assessment", "self_correction"): False,
        ("goedel_assessment", "native_lane_run"): False,
        ("goedel_assessment", "environment_probe_timeout_seconds"): 120.0,
    }
    for (section, key), wanted in expected.items():
        actual = config.value[section][key]
        if actual != wanted:
            raise ValueError(
                f"Goedel assessment contract changed at {section}.{key}: "
                f"expected {wanted!r}, got {actual!r}"
            )
    if set(config.value["workloads"]) != set(WORKLOADS):
        raise ValueError("Goedel assessment workloads differ from issue #30")
    if (
        config.benchmark["split"] != "validation"
        or int(config.benchmark["expected_primary_task_count"]) != 244
    ):
        raise ValueError("Goedel assessment must use 244 miniF2F validation tasks")
    if float(config.value["verifier"]["timeout_seconds"]) != 30.0:
        raise ValueError("Goedel verifier timeout must remain 30 seconds")


def validate_model_snapshot(config: Phase1Config, snapshot: Path) -> dict[str, Any]:
    validate_assessment_contract(config)
    root = snapshot.resolve()
    if root.name != MODEL_REVISION:
        raise ValueError(
            f"model snapshot must resolve to pinned revision {MODEL_REVISION}: {root}"
        )
    required = {
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ValueError(f"model snapshot is incomplete: {missing}")
    index = _read_json(root / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (root / name).is_file()]
    if missing_shards:
        raise ValueError(f"model snapshot is missing weight shards: {missing_shards}")
    model_config = _read_json(root / "config.json")
    if model_config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("pinned Goedel snapshot architecture differs")
    if model_config.get("torch_dtype") != "bfloat16":
        raise ValueError("pinned Goedel snapshot dtype differs")
    return {
        "revision": root.name,
        "architecture": model_config["architectures"][0],
        "source_dtype": model_config["torch_dtype"],
        "weight_shard_count": len(shards),
        "weight_bytes": sum((root / name).stat().st_size for name in shards),
    }


def run_preflight(
    config: Phase1Config,
    benchmark_root: Path,
    model_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    validate_assessment_contract(config)
    snapshot = validate_model_snapshot(config, model_snapshot)
    environment = validate_minif2f_environment(
        config,
        benchmark_root,
        timeout_seconds=float(
            config.value["goedel_assessment"][
                "environment_probe_timeout_seconds"
            ]
        ),
    )
    tasks = materialize_benchmark_tasks(config, benchmark_root)
    dev16 = config.select_workload("minif2f-valid-dev16-v1", tasks)
    prompts = [render_prompt(task) for task in dev16]
    if any("<|im_start|>" in prompt or "```" in prompt for prompt in prompts):
        raise ValueError("strict Goedel prompt contains a chat or fenced-code wrapper")
    evidence = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "assessment_id": config.value["goedel_assessment"]["id"],
        "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
        "model": config.model,
        "snapshot": snapshot,
        "benchmark_environment": environment,
        "prompt_contract": {
            "prompt_format_id": PROMPT_FORMAT_ID,
            "raw_continuation": True,
            "chat_template": None,
            "proof_extraction": False,
            "lean_guided_retry": False,
            "self_correction": False,
            "dev16_prompt_count": len(prompts),
            "dev16_prompts_sha256": _ordered_strings_sha256(prompts),
        },
        "runtime": {
            **_local_cuda_runtime(config),
            "packages": _package_versions(),
        },
    }
    _write_json(output, evidence)
    return evidence


def run_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    preflight_path: Path,
    workload_id: str,
    output_dir: Path,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    validate_assessment_contract(config)
    _validate_preflight(config, _read_json(preflight_path))
    if workload_id not in WORKLOADS:
        raise ValueError(f"unknown Goedel workload: {workload_id}")
    metadata, results, summary = run_phase1_baseline(
        config,
        benchmark_root,
        workload_id,
        output_dir,
        timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
        verification_workers=int(config.value["verifier"]["verification_workers"]),
        report_progress=True,
        environment_probe_timeout_seconds=float(
            config.value["goedel_assessment"][
                "environment_probe_timeout_seconds"
            ]
        ),
    )
    return metadata, results, summary


def write_compact_evidence(
    config: Phase1Config,
    preflight_path: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    validate_assessment_contract(config)
    preflight = _read_json(preflight_path)
    _validate_preflight(config, preflight)
    dev16 = _compact_run(config, dev16_dir, WORKLOADS[0], 16)
    full = _compact_run(config, full_dir, WORKLOADS[1], 244)

    project_root = config.path.parents[1]
    assessment = config.value["goedel_assessment"]
    reference_path = project_root / str(assessment["reference_sft_evidence"])
    base_path = project_root / str(assessment["qwen_base_evidence"])
    reference = _read_json(reference_path)["adapter"]
    qwen_base = _read_json(base_path)
    observed = full["summary"]["pass_at_k"]
    anchors = {
        "reference-sft-v1": {
            "candidates_per_task": reference["candidates_per_task"],
            "pass_at_k": _pass1_pass4(reference["pass_at_k"]),
            "source": str(assessment["reference_sft_evidence"]),
        },
        "Qwen/Qwen3-8B-Base": {
            "candidates_per_task": qwen_base["candidates_per_task"],
            "pass_at_k": _pass1_pass4(qwen_base["pass_at_k"]),
            "source": str(assessment["qwen_base_evidence"]),
        },
    }
    deltas = {
        name: {
            key: float(observed[key]) - float(anchor["pass_at_k"][key])
            for key in ("pass@1", "pass@4")
        }
        for name, anchor in anchors.items()
    }
    comparison = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "OBSERVED",
        "assessment_id": assessment["id"],
        "strict_lane": {
            "model_id": config.model["model_id"],
            "model_revision": config.model["model_revision"],
            "candidates_per_task": 4,
            "pass_at_k": _pass1_pass4(observed),
        },
        "anchors": anchors,
        "delta_strict_minus_anchor": deltas,
        "native_prover_diagnostic": {
            "run": False,
            "reason": "optional lane omitted; its chat prompt and verifier-guided self-correction are materially distinct from the required one-shot contract",
        },
        "comparison_limitations": [
            "The strict lane uses four candidates per task while both accepted anchors use eight; pass@1/pass@4 estimators are shown without importing published Pass@32 claims.",
            "This assessment uses qwen-lean's Lean 4.27 verifier, not the model card's Lean 4.9 environment.",
            "Verification used one worker under concurrent shared-host load; this preserves proof outcomes but makes verification and total wall-time comparisons host-load-dependent.",
            "Peak GPU memory was not measured because the optional NVML monitor package was unavailable.",
        ],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    compact_preflight = dict(preflight)
    compact_preflight["candidate_artifacts_retained_outside_git"] = True
    _write_json(evidence_dir / "preflight.json", compact_preflight)
    _write_json(evidence_dir / "dev16.json", dev16)
    _write_json(evidence_dir / "full.json", full)
    _write_json(evidence_dir / "comparison.json", comparison)
    (evidence_dir / "README.md").write_text(
        _render_readme(dev16, full, comparison), encoding="utf-8"
    )
    return comparison


def _compact_run(
    config: Phase1Config,
    artifact_dir: Path,
    workload_id: str,
    expected_tasks: int,
) -> dict[str, Any]:
    metadata, results = read_artifacts(artifact_dir)
    stored = _read_json(artifact_dir / "summary.json")
    expected_ids = _expected_task_ids(config, workload_id)
    if len(expected_ids) != expected_tasks:
        raise ValueError(f"{workload_id} expected {expected_tasks} task IDs")
    recomputed = summarize_results(
        results,
        expected_task_ids=expected_ids,
        candidates_per_task=4,
    )
    for key in (
        "complete",
        "completeness_errors",
        "candidate_count",
        "candidates_per_task",
        "pass_at_k",
        "category_counts",
        "finish_reason_counts",
        "verifier_timeout_count",
        "infrastructure_error_count",
    ):
        if stored[key] != recomputed[key]:
            raise ValueError(f"stored {workload_id} {key} differs from raw results")
    _validate_run_identity(config, metadata, workload_id)
    if not recomputed["complete"] or recomputed["infrastructure_error_count"]:
        raise ValueError(f"{workload_id} is not infrastructure-complete")
    lengths = [int(result.generated_token_count or 0) for result in results]
    solved = int(recomputed["tasks_with_verified_candidate"]["count"])
    generation_wall = float(metadata.runtime["generation_wall_time_seconds"])
    run_wall = float(stored["run_wall_time_seconds"])
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "workload_id": workload_id,
        "model": {
            "model_id": metadata.model_id,
            "model_revision": metadata.model_revision,
            "tokenizer_id": metadata.tokenizer_id,
            "tokenizer_revision": metadata.tokenizer_revision,
        },
        "benchmark": {
            "repository": metadata.benchmark_repository,
            "revision": metadata.benchmark_revision,
            "split": metadata.benchmark_split,
            "lean_toolchain": metadata.lean_toolchain,
            "mathlib_revision": metadata.mathlib_revision,
        },
        "prompt_format_id": metadata.prompt_format_id,
        "generation_settings": metadata.generation_settings,
        "inference_engine": metadata.inference_engine,
        "inference_engine_version": metadata.inference_engine_version,
        "runtime": metadata.runtime,
        "summary": {key: recomputed[key] for key in (
            "complete",
            "task_count",
            "candidate_count",
            "candidates_per_task",
            "tasks_with_verified_candidate",
            "pass_at_k",
            "category_counts",
            "category_fractions",
            "finish_reason_counts",
            "verifier_timeout_count",
            "infrastructure_error_count",
            "timing_seconds",
        )},
        "generated_token_lengths": {
            "minimum": min(lengths),
            "mean": fmean(lengths),
            "median": median(lengths),
            "maximum": max(lengths),
            "total": sum(lengths),
        },
        "wall_time_seconds": {
            "generation": generation_wall,
            "verification": float(metadata.runtime["verification_wall_time_seconds"]),
            "total": run_wall,
        },
        "compute_per_solved_task": {
            "solved_tasks": solved,
            "generation_gpu_seconds": None if solved == 0 else generation_wall / solved,
            "run_wall_seconds": None if solved == 0 else run_wall / solved,
        },
        "execution_notes": [
            "Verification used one worker to avoid timeout distortion from a concurrent shared-host Dataset v2 build; the 30-second per-candidate timeout and verifier acceptance semantics were unchanged.",
            "Verification and total wall times reflect concurrent shared-host load; generation wall time was measured separately during local GPU inference.",
            "Peak GPU memory was unavailable because the optional NVML monitor package was not installed; the GPU identity, total memory, and configured utilization are retained in runtime and generation settings.",
        ],
        "verifier_timeout_semantics": "unsuccessful_proof_outcome_not_infrastructure_error",
        "candidate_results_retained_outside_git": True,
    }


def _validate_run_identity(
    config: Phase1Config, metadata: RunMetadata, workload_id: str
) -> None:
    if (
        metadata.workload_id != workload_id
        or metadata.model_id != MODEL_ID
        or metadata.tokenizer_id != MODEL_ID
        or metadata.model_revision != MODEL_REVISION
        or metadata.tokenizer_revision != MODEL_REVISION
        or metadata.prompt_format_id != PROMPT_FORMAT_ID
        or metadata.benchmark_revision != config.benchmark["revision"]
        or metadata.candidates_per_task != 4
        or metadata.inference_engine != "vllm"
        or metadata.inference_engine_version != config.engine["version"]
    ):
        raise ValueError(f"{workload_id} run identity differs from the frozen contract")
    settings = metadata.generation_settings or {}
    for key, wanted in {
        **config.sampling,
        "chat_template": None,
        "prompt_transformation": None,
        "quantization": None,
    }.items():
        if settings.get(key) != wanted:
            raise ValueError(f"{workload_id} generation setting {key} differs")
    if metadata.runtime.get("inference_execution") != "local_cuda":
        raise ValueError(f"{workload_id} did not use local CUDA inference")


def _validate_preflight(config: Phase1Config, value: dict[str, Any]) -> None:
    if value.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unknown Goedel preflight schema")
    if value.get("status") != "passed":
        raise ValueError("Goedel assessment requires a passed preflight")
    if value.get("assessment_id") != config.value["goedel_assessment"]["id"]:
        raise ValueError("Goedel preflight assessment identity differs")
    digest = hashlib.sha256(config.path.read_bytes()).hexdigest()
    if value.get("config_sha256") != digest:
        raise ValueError("Goedel preflight config hash differs")
    if value.get("model") != config.model:
        raise ValueError("Goedel preflight model identity differs")
    if value.get("snapshot", {}).get("revision") != MODEL_REVISION:
        raise ValueError("Goedel preflight snapshot revision differs")


def _expected_task_ids(config: Phase1Config, workload_id: str) -> list[str]:
    workload = config.value["workloads"][workload_id]
    if workload["selection"] == "explicit_ids":
        return [str(value) for value in workload["task_ids"]]
    manifest = config.path.parent / str(config.benchmark["primary_task_manifest"])
    return [
        line
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def _package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "vllm", "huggingface-hub")
    }


def _pass1_pass4(value: dict[str, Any]) -> dict[str, float]:
    return {key: float(value[key]) for key in ("pass@1", "pass@4")}


def _ordered_strings_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_readme(
    dev16: dict[str, Any], full: dict[str, Any], comparison: dict[str, Any]
) -> str:
    rows = []
    for name, run in (("dev16", dev16), ("full validation", full)):
        summary = run["summary"]
        rows.append(
            f"| {name} | {summary['task_count']} | {summary['candidate_count']} | "
            f"{summary['pass_at_k']['pass@1']:.6f} | "
            f"{summary['pass_at_k']['pass@4']:.6f} | "
            f"{summary['infrastructure_error_count']} | "
            f"{summary['verifier_timeout_count']} |"
        )
    strict = comparison["strict_lane"]["pass_at_k"]
    reference = comparison["anchors"]["reference-sft-v1"]["pass_at_k"]
    base = comparison["anchors"]["Qwen/Qwen3-8B-Base"]["pass_at_k"]
    return (
        "# Goedel-Prover-V2-8B strict miniF2F assessment\n\n"
        f"**OBSERVED:** strict pass@1/pass@4 were {strict['pass@1']:.6f}/"
        f"{strict['pass@4']:.6f}; `reference-sft-v1` measured "
        f"{reference['pass@1']:.6f}/{reference['pass@4']:.6f}, and the Qwen3-8B "
        f"Base anchor measured {base['pass@1']:.6f}/{base['pass@4']:.6f}.\n\n"
        "| Workload | Tasks | Candidates | pass@1 | pass@4 | Infrastructure errors | Verifier timeouts |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\nThe strict lane uses the raw `whole-proof-v1` continuation with no chat "
        "wrapper, proof extraction, verifier-guided retry, or self-correction. "
        "Raw candidates, model weights, caches, and bulky logs remain outside Git. "
        "The optional native-prover diagnostic was not run and is not mixed into "
        "these scores. Verification used one worker to avoid timeout distortion "
        "from a concurrent shared-host Dataset v2 build; verifier semantics were "
        "unchanged, but verification and total wall times are host-load-dependent. "
        "Peak GPU memory was not measured because the optional NVML monitor package "
        "was unavailable.\n"
    )
