from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from .metrics import summarize_results
from .minif2f import Phase1Config, materialize_benchmark_tasks
from .phase5 import ordered_record_ids_sha256
from .phase6 import (
    REFERENCE_SFT_ID,
    TRAIN_WORKLOAD_ID,
    Phase6Config,
    _read_json,
    _write_json,
    differential_gap_bootstrap,
    generalization_gaps,
    paired_task_bootstrap,
    per_task_verified_counts,
    summarize_phase6_train_results,
    validate_phase6_train_workload,
)
from .schema import CandidateResult


def write_phase6_checkpoint_a_evidence(
    config: Phase6Config,
    candidate_manifest_path: Path,
    train_workload_path: Path,
    benchmark_root: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    candidate = _read_json(candidate_manifest_path)
    workload = _read_json(train_workload_path)
    examples = validate_phase6_train_workload(config, workload)
    if candidate.get("logical_id") != REFERENCE_SFT_ID:
        raise ValueError("Phase 6 checkpoint candidate is not reference-sft-v1")

    project_root = config.path.parents[1]
    test_config = config.phase1_test_config()
    test_tasks = materialize_benchmark_tasks(test_config, benchmark_root)
    test_manifest = [task.id for task in test_tasks]
    validation_phase1 = Phase1Config.load(project_root / "config/phase1-minif2f.json")
    validation_tasks = materialize_benchmark_tasks(validation_phase1, benchmark_root)
    validation_manifest = [task.id for task in validation_tasks]
    validation_config = validation_phase1.value
    if len(test_manifest) != 244 or len(validation_manifest) != 244:
        raise ValueError("miniF2F validation/test manifest denominator differs")
    if test_manifest == validation_manifest:
        raise ValueError("miniF2F validation and test manifests are identical")

    prompt_lengths = [item.prompt_tokens for item in examples]
    compact_workload = {
        key: copy.deepcopy(workload[key])
        for key in (
            "schema_version",
            "workload_id",
            "source_membership",
            "selection",
            "selected_record_ids",
            "selected_record_ids_sha256",
        )
    }
    compact_workload["prompt_token_counts"] = {
        "minimum": min(prompt_lengths),
        "maximum": max(prompt_lengths),
        "mean": fmean(prompt_lengths),
    }
    compact_workload["raw_prompts_retained_outside_git"] = True
    split_manifest = {
        "schema_version": "phase6-minif2f-manifests-v1",
        "repository": test_config.benchmark["repository"],
        "revision": test_config.benchmark["revision"],
        "test": {
            "workload_id": config.value["minif2f_test"]["workload_id"],
            "source_path": test_config.benchmark["source_path"],
            "task_count": len(test_manifest),
            "ordered_task_ids": test_manifest,
            "ordered_task_ids_sha256": ordered_record_ids_sha256(test_manifest),
        },
        "validation": {
            "workload_id": "minif2f-valid-v1",
            "source_path": validation_config["benchmark"]["source_path"],
            "task_count": len(validation_manifest),
            "ordered_task_ids_sha256": ordered_record_ids_sha256(validation_manifest),
        },
        "manifests_distinct": test_manifest != validation_manifest,
    }
    _write_json(evidence_dir / "candidate.json", candidate)
    _write_json(evidence_dir / "train-workload.json", compact_workload)
    _write_json(evidence_dir / "minif2f-manifests.json", split_manifest)
    return {
        "candidate": candidate,
        "train_workload": compact_workload,
        "minif2f_manifests": split_manifest,
    }


def _read_results(path: Path) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                results.append(CandidateResult.from_dict(json.loads(line)))
    return results


def _read_phase6_train_results(
    path: Path,
) -> tuple[list[CandidateResult], dict[tuple[str, int], bool]]:
    results: list[CandidateResult] = []
    exact: dict[tuple[str, int], bool] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            target_exact = bool(value.pop("target_exact"))
            result = CandidateResult.from_dict(value)
            results.append(result)
            exact[(result.task_id, result.candidate_index)] = target_exact
    return results, exact


def _summary_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value[key])
        for key in (
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
        )
        if key in value
    }


def _runtime_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    runtime = run["runtime"]
    return {
        key: runtime[key]
        for key in (
            "torch",
            "torch_cuda_version",
            "inference_execution",
            "cuda_device_index",
            "cuda_device",
            "cuda_device_capability",
            "cuda_device_total_memory_bytes",
        )
        if key in runtime
    }


def _validate_train_pair(
    config: Phase6Config, base_dir: Path, adapter_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runs = {
        role: _read_json(directory / "run.json")
        for role, directory in (("base", base_dir), ("adapter", adapter_dir))
    }
    stored = {
        role: _read_json(directory / "summary.json")
        for role, directory in (("base", base_dir), ("adapter", adapter_dir))
    }
    recomputed: dict[str, dict[str, Any]] = {}
    exact_counts: dict[str, list[int]] = {}
    for role, directory in (("base", base_dir), ("adapter", adapter_dir)):
        results, exact = _read_phase6_train_results(directory / "results.jsonl")
        recomputed[role] = summarize_phase6_train_results(
            results,
            expected_task_ids=runs[role]["selected_record_ids"],
            target_exact_by_candidate=exact,
        )
        for key in (
            "pass_at_k",
            "exact_target_pass_at_k",
            "category_counts",
            "finish_reason_counts",
            "exact_target_candidates",
            "verified_non_exact_candidates",
            "phase6_train_integrity_passed",
        ):
            if recomputed[role][key] != stored[role][key]:
                raise ValueError(f"Phase 6 train {role} stored {key} differs")
        exact_counts[role] = [
            int(item["exact_target_candidate_count"])
            for item in recomputed[role]["per_task"]
        ]
    for key in (
        "model",
        "reference_candidate",
        "dataset_schema_version",
        "dataset_split",
        "source_membership_workload_id",
        "workload_id",
        "selected_record_ids",
        "prompt_format_id",
        "serialization_or_prompt_transformation",
        "exact_target_normalization",
        "generation_settings",
        "inference_engine",
        "inference_engine_version",
        "source_repository",
        "source_revision",
        "lean_toolchain",
        "verification",
    ):
        if runs["base"].get(key) != runs["adapter"].get(key):
            raise ValueError(f"Phase 6 train comparison differs in {key}")
    if (
        runs["base"].get("model_role") != "base"
        or runs["base"].get("adapter") is not None
        or runs["adapter"].get("model_role") != "adapter"
        or not runs["adapter"].get("adapter", {}).get("enabled")
        or _runtime_identity(runs["base"]) != _runtime_identity(runs["adapter"])
    ):
        raise ValueError("Phase 6 train model roles or runtime identity differ")
    compact: dict[str, Any] = {
        "schema_version": "phase6-train-comparison-v1",
        "status": "passed",
        "workload_id": TRAIN_WORKLOAD_ID,
        "selected_record_ids_sha256": ordered_record_ids_sha256(
            runs["base"]["selected_record_ids"]
        ),
        "evaluation_contract": {
            key: copy.deepcopy(runs["base"][key])
            for key in (
                "model",
                "dataset_schema_version",
                "dataset_split",
                "source_membership_workload_id",
                "prompt_format_id",
                "generation_settings",
                "inference_engine",
                "inference_engine_version",
                "source_repository",
                "source_revision",
                "lean_toolchain",
                "verification",
                "exact_target_normalization",
            )
        },
        "runtime_identity": _runtime_identity(runs["base"]),
        "raw_candidate_results_retained_outside_git": True,
    }
    for role in ("base", "adapter"):
        summary = _summary_core(stored[role])
        for key in (
            "exact_target_pass_at_k",
            "exact_target_candidates",
            "tasks_with_exact_target_candidate",
            "verified_non_exact_candidates",
            "tasks_with_verified_non_exact_candidate",
            "exact_target_but_not_verified_count",
            "generated_token_counts",
            "generation_wall_time_seconds",
            "verification_wall_time_seconds",
            "run_wall_time_seconds",
        ):
            summary[key] = copy.deepcopy(stored[role][key])
        summary["per_task"] = [
            {
                "task_id": item["task_id"],
                "verified_candidate_count": item["verified_candidate_count"],
                "exact_target_candidate_count": item["exact_target_candidate_count"],
                "verified_non_exact_candidate_count": item[
                    "verified_non_exact_candidate_count"
                ],
            }
            for item in recomputed[role]["per_task"]
        ]
        compact[role] = summary
    compact["delta_adapter_minus_base"] = {
        key: float(compact["adapter"]["pass_at_k"][key])
        - float(compact["base"]["pass_at_k"][key])
        for key in ("pass@1", "pass@4")
    }
    compact["delta_exact_target_adapter_minus_base"] = {
        key: float(compact["adapter"]["exact_target_pass_at_k"][key])
        - float(compact["base"]["exact_target_pass_at_k"][key])
        for key in ("pass@1", "pass@4")
    }
    resamples = int(config.value["bootstrap"]["resamples"])
    seed = int(config.value["bootstrap"]["seed"])
    compact["lean_pass_bootstrap"] = paired_task_bootstrap(
        per_task_verified_counts(recomputed["base"]),
        per_task_verified_counts(recomputed["adapter"]),
        candidates_per_task=4,
        ks=(1, 4),
        resamples=resamples,
        seed=seed,
    )
    compact["exact_target_pass_bootstrap"] = paired_task_bootstrap(
        exact_counts["base"],
        exact_counts["adapter"],
        candidates_per_task=4,
        ks=(1, 4),
        resamples=resamples,
        seed=seed,
    )
    return compact, recomputed["base"], recomputed["adapter"]


def _validate_heldout_reuse(
    config: Phase6Config,
    comparison_path: Path,
    base_dir: Path,
    adapter_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    comparison = _read_json(comparison_path)
    if (
        comparison.get("status") != "passed"
        or comparison.get("comparison_integrity_passed") is not True
        or comparison.get("workload_id")
        != config.value["phase5_inputs"]["heldout_workload_id"]
        or ordered_record_ids_sha256(comparison["selected_record_ids"])
        != config.value["phase5_inputs"]["heldout_ordered_ids_sha256"]
    ):
        raise ValueError("accepted Phase 5 heldout comparison identity differs")
    summaries: dict[str, dict[str, Any]] = {}
    for role, directory in (("base", base_dir), ("adapter", adapter_dir)):
        results = _read_results(directory / "results.jsonl")
        summary = summarize_results(
            results,
            expected_task_ids=comparison["selected_record_ids"],
            candidates_per_task=4,
            ks=(1, 4),
        )
        for key in ("pass_at_k", "category_counts", "finish_reason_counts"):
            if summary[key] != comparison[role][key]:
                raise ValueError(f"Phase 5 heldout raw {role} {key} differs")
        summaries[role] = summary
    compact = copy.deepcopy(comparison)
    compact["schema_version"] = "phase6-heldout-reuse-v1"
    compact["selected_record_ids_sha256"] = ordered_record_ids_sha256(
        compact.pop("selected_record_ids")
    )
    compact["accepted_phase5_result_reused_without_regeneration"] = True
    for role in ("base", "adapter"):
        compact[role]["per_task"] = summaries[role]["per_task"]
    compact["bootstrap"] = paired_task_bootstrap(
        per_task_verified_counts(summaries["base"]),
        per_task_verified_counts(summaries["adapter"]),
        candidates_per_task=4,
        ks=(1, 4),
        resamples=int(config.value["bootstrap"]["resamples"]),
        seed=int(config.value["bootstrap"]["seed"]),
    )
    return compact, summaries["base"], summaries["adapter"]


def _validation_comparison(
    config: Phase6Config, base_summary_path: Path, adapter_evidence_path: Path
) -> dict[str, Any]:
    base = _read_json(base_summary_path)
    adapter_evidence = _read_json(adapter_evidence_path)
    adapter = adapter_evidence["summary"]
    if (
        adapter_evidence.get("status") != "passed"
        or adapter_evidence.get("workload_id")
        != config.value["phase5_inputs"]["minif2f_validation_workload_id"]
    ):
        raise ValueError("accepted miniF2F validation adapter evidence differs")
    for role, summary in (("base", base), ("adapter", adapter)):
        if (
            summary.get("complete") is not True
            or int(summary.get("candidate_count", -1)) != 1952
            or int(summary.get("infrastructure_error_count", -1)) != 0
            or int(summary.get("verifier_timeout_count", -1)) != 0
        ):
            raise ValueError(f"accepted miniF2F validation {role} is incomplete")
    return {
        "schema_version": "phase6-minif2f-validation-reference-v1",
        "status": "passed",
        "workload_id": "minif2f-valid-v1",
        "accepted_results_reused_without_regeneration": True,
        "checkpoint_selection_influenced_by_validation": False,
        "base": _summary_core(base),
        "adapter": _summary_core(adapter),
        "delta_adapter_minus_base": {
            key: float(adapter["pass_at_k"][key]) - float(base["pass_at_k"][key])
            for key in ("pass@1", "pass@4", "pass@8")
        },
    }


def _normalized_generation_settings(run: Mapping[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(run["generation_settings"])
    settings.pop("adapter", None)
    return settings


def _validate_test_pair(
    config: Phase6Config, base_dir: Path, adapter_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runs = {
        role: _read_json(directory / "run.json")
        for role, directory in (("base", base_dir), ("adapter", adapter_dir))
    }
    summaries = {
        role: _read_json(directory / "summary.json")
        for role, directory in (("base", base_dir), ("adapter", adapter_dir))
    }
    for role, summary in summaries.items():
        if (
            summary.get("phase6_minif2f_test_integrity_passed") is not True
            or int(summary.get("candidate_count", -1)) != 1952
            or int(summary.get("infrastructure_error_count", -1)) != 0
            or int(summary.get("verifier_timeout_count", -1)) != 0
        ):
            raise ValueError(f"Phase 6 miniF2F test {role} is incomplete")
    for key in (
        "model_id",
        "tokenizer_id",
        "model_revision",
        "tokenizer_revision",
        "workload_id",
        "benchmark_split",
        "benchmark_repository",
        "benchmark_revision",
        "verifier_timeout_seconds",
        "candidates_per_task",
        "inference_engine",
        "inference_engine_version",
        "selected_adapter_binding",
    ):
        if runs["base"].get(key) != runs["adapter"].get(key):
            raise ValueError(f"Phase 6 miniF2F test comparison differs in {key}")
    if (
        runs["base"].get("benchmark_split") != "test"
        or runs["base"].get("workload_id") != "minif2f-test-v1"
        or runs["base"].get("adapter_enabled") is not False
        or runs["adapter"].get("adapter_enabled") is not True
        or _normalized_generation_settings(runs["base"])
        != _normalized_generation_settings(runs["adapter"])
        or _runtime_identity(runs["base"]) != _runtime_identity(runs["adapter"])
    ):
        raise ValueError("Phase 6 miniF2F test roles, settings, or runtime differ")
    compact: dict[str, Any] = {
        "schema_version": "phase6-minif2f-test-comparison-v1",
        "status": "passed",
        "workload_id": "minif2f-test-v1",
        "benchmark_repository": runs["base"]["benchmark_repository"],
        "benchmark_revision": runs["base"]["benchmark_revision"],
        "benchmark_split": "test",
        "candidate_selection_predates_test": summaries["base"][
            "candidate_selection_predates_test"
        ],
        "evaluation_contract": {
            "model_id": runs["base"]["model_id"],
            "model_revision": runs["base"]["model_revision"],
            "tokenizer_id": runs["base"]["tokenizer_id"],
            "tokenizer_revision": runs["base"]["tokenizer_revision"],
            "generation_settings": _normalized_generation_settings(runs["base"]),
            "lean_toolchain": runs["base"]["lean_toolchain"],
            "verifier_timeout_seconds": runs["base"]["verifier_timeout_seconds"],
        },
        "runtime_identity": _runtime_identity(runs["base"]),
        "raw_candidate_results_retained_outside_git": True,
    }
    for role in ("base", "adapter"):
        compact[role] = _summary_core(summaries[role])
        compact[role]["generated_token_counts"] = summaries[role][
            "generated_token_counts"
        ]
        compact[role]["run_wall_time_seconds"] = summaries[role][
            "run_wall_time_seconds"
        ]
        compact[role]["per_task"] = summaries[role]["per_task"]
    compact["delta_adapter_minus_base"] = {
        key: float(compact["adapter"]["pass_at_k"][key])
        - float(compact["base"]["pass_at_k"][key])
        for key in ("pass@1", "pass@4", "pass@8")
    }
    compact["bootstrap"] = paired_task_bootstrap(
        per_task_verified_counts(summaries["base"]),
        per_task_verified_counts(summaries["adapter"]),
        candidates_per_task=8,
        ks=(1, 4, 8),
        resamples=int(config.value["bootstrap"]["resamples"]),
        seed=int(config.value["bootstrap"]["seed"]),
    )
    return compact, summaries["base"], summaries["adapter"]


def write_phase6_final_evidence(
    config: Phase6Config,
    artifact_dir: Path,
    phase5_heldout_comparison: Path,
    phase5_heldout_base_dir: Path,
    phase5_heldout_adapter_dir: Path,
    phase1_validation_base_summary: Path,
    phase5_validation_adapter_evidence: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    candidate = _read_json(artifact_dir / "candidate.json")
    workload = _read_json(artifact_dir / "train-workload.json")
    validate_phase6_train_workload(config, workload)
    if candidate.get("logical_id") != REFERENCE_SFT_ID:
        raise ValueError("Phase 6 final evidence candidate differs")
    train, train_base, train_adapter = _validate_train_pair(
        config, artifact_dir / "train/base", artifact_dir / "train/adapter"
    )
    heldout, heldout_base, heldout_adapter = _validate_heldout_reuse(
        config,
        phase5_heldout_comparison,
        phase5_heldout_base_dir,
        phase5_heldout_adapter_dir,
    )
    validation = _validation_comparison(
        config, phase1_validation_base_summary, phase5_validation_adapter_evidence
    )
    test, _, _ = _validate_test_pair(
        config,
        artifact_dir / "minif2f-test/base",
        artifact_dir / "minif2f-test/adapter",
    )

    gaps = generalization_gaps(
        train["base"]["pass_at_k"],
        train["adapter"]["pass_at_k"],
        heldout["base"]["pass_at_k"],
        heldout["adapter"]["pass_at_k"],
    )
    differential = differential_gap_bootstrap(
        per_task_verified_counts(train_base),
        per_task_verified_counts(train_adapter),
        per_task_verified_counts(heldout_base),
        per_task_verified_counts(heldout_adapter),
        resamples=int(config.value["bootstrap"]["resamples"]),
        seed=int(config.value["bootstrap"]["seed"]),
    )
    for key in ("pass@1", "pass@4"):
        differential["differential_gap"][key]["estimate"] = gaps[key][
            "differential_gap"
        ]

    comparison = {
        "schema_version": "phase6-comparison-v1",
        "status": "passed",
        "reference_candidate": {
            "logical_id": REFERENCE_SFT_ID,
            "base_model": config.model,
            "adapter": config.adapter,
            "selection_predates_phase6_outputs": True,
            "phase6_metrics_changed_identity": False,
        },
        "workloads": {
            "phase6-train512-v1": {
                "base_lean": train["base"]["pass_at_k"],
                "adapter_lean": train["adapter"]["pass_at_k"],
                "base_exact_target": train["base"]["exact_target_pass_at_k"],
                "adapter_exact_target": train["adapter"]["exact_target_pass_at_k"],
                "base_verified_non_exact_candidates": train["base"][
                    "verified_non_exact_candidates"
                ],
                "adapter_verified_non_exact_candidates": train["adapter"][
                    "verified_non_exact_candidates"
                ],
            },
            "phase5-heldout512-v1": {
                "base_lean": heldout["base"]["pass_at_k"],
                "adapter_lean": heldout["adapter"]["pass_at_k"],
            },
            "minif2f-valid-v1": {
                "base_lean": validation["base"]["pass_at_k"],
                "adapter_lean": validation["adapter"]["pass_at_k"],
            },
            "minif2f-test-v1": {
                "base_lean": test["base"]["pass_at_k"],
                "adapter_lean": test["adapter"]["pass_at_k"],
            },
        },
        "train_to_heldout_generalization_gaps": gaps,
        "differential_gap_bootstrap": differential,
        "uncertainty_references": {
            "train_lean": "train-comparison.json#lean_pass_bootstrap",
            "train_exact_target": "train-comparison.json#exact_target_pass_bootstrap",
            "heldout_lean": "heldout-analysis.json#bootstrap",
            "minif2f_test_delta": "minif2f-test.json#bootstrap",
        },
        "result_types_kept_distinct": [
            "target-proof reproduction",
            "alternative Lean-valid proof generation",
            "internal heldout generalization",
            "external validation generalization",
            "external test generalization",
            "termination and verifier operational behavior",
        ],
        "opaque_combined_quality_score": None,
    }

    _write_json(evidence_dir / "candidate.json", candidate)
    checkpoint_workload = _read_json(evidence_dir / "train-workload.json")
    if (
        checkpoint_workload["selected_record_ids_sha256"]
        != workload["selected_record_ids_sha256"]
    ):
        raise ValueError("Phase 6 final train workload differs from Checkpoint A")
    _write_json(evidence_dir / "train-comparison.json", train)
    _write_json(evidence_dir / "heldout-analysis.json", heldout)
    _write_json(evidence_dir / "minif2f-validation.json", validation)
    _write_json(evidence_dir / "minif2f-test.json", test)
    _write_json(evidence_dir / "comparison.json", comparison)

    train_base_mean = float(train["base"]["generated_token_counts"]["mean"])
    train_adapter_mean = float(train["adapter"]["generated_token_counts"]["mean"])
    test_base_mean = float(test["base"]["generated_token_counts"]["mean"])
    test_adapter_mean = float(test["adapter"]["generated_token_counts"]["mean"])
    readme = f"""# Phase 6 evidence

The fixed `reference-sft-v1` candidate is the Phase 5 validation-selected step-9962 unmerged PEFT adapter at immutable Hub revision `{config.adapter["hub_revision"]}` on the pinned Qwen base. Its identity was frozen before Phase 6 train generation and the first miniF2F test evaluation; Q1/Q2/Q3 were not eligible alternatives, and no Phase 6 metric changed the candidate.

**OBSERVED:** on `phase6-train512-v1`, base Lean pass@1/pass@4 were {train["base"]["pass_at_k"]["pass@1"]:.6f}/{train["base"]["pass_at_k"]["pass@4"]:.6f}, while SFT produced {train["adapter"]["pass_at_k"]["pass@1"]:.6f}/{train["adapter"]["pass_at_k"]["pass@4"]:.6f}. Exact-target pass@1/pass@4 were {train["base"]["exact_target_pass_at_k"]["pass@1"]:.6f}/{train["base"]["exact_target_pass_at_k"]["pass@4"]:.6f} for base and {train["adapter"]["exact_target_pass_at_k"]["pass@1"]:.6f}/{train["adapter"]["exact_target_pass_at_k"]["pass@4"]:.6f} for SFT. SFT produced {train["adapter"]["verified_non_exact_candidates"]["count"]} verified non-exact candidates, which are alternative valid proofs rather than retained-target reproductions.

**OBSERVED:** accepted internal heldout base/SFT pass@1 were {heldout["base"]["pass_at_k"]["pass@1"]:.6f}/{heldout["adapter"]["pass_at_k"]["pass@1"]:.6f}; miniF2F validation base/SFT pass@1 were {validation["base"]["pass_at_k"]["pass@1"]:.6f}/{validation["adapter"]["pass_at_k"]["pass@1"]:.6f}; first-use miniF2F test base/SFT pass@1 were {test["base"]["pass_at_k"]["pass@1"]:.6f}/{test["adapter"]["pass_at_k"]["pass@1"]:.6f}. `comparison.json` reports the fixed train/heldout gap formulas, while the workload files retain deterministic 10,000-resample seed-0 task-bootstrap intervals.

**OBSERVED:** mean generated length changed from {train_base_mean:.2f} to {train_adapter_mean:.2f} tokens on the train diagnostic and from {test_base_mean:.2f} to {test_adapter_mean:.2f} on miniF2F test. Train token-limit finishes changed from {train["base"]["finish_reason_counts"]["token_limit"]} to {train["adapter"]["finish_reason_counts"]["token_limit"]}; test token-limit finishes changed from {test["base"]["finish_reason_counts"]["token_limit"]} to {test["adapter"]["finish_reason_counts"]["token_limit"]}. These operational shifts are reported separately from verifier success.

**ACCEPTED:** all new runs completed their fixed candidate counts with zero generation/verifier infrastructure errors and zero unresolved timeouts. `reference-sft-v1` is the controlled common parent and retained SFT control for the independent post-training branches; it is not claimed to be globally optimal SFT.

Raw continuations, model caches, adapter weights, source datasets, and bulky candidate artifacts remain under ignored `artifacts/phase6/` or their accepted Phase 5 locations.
"""
    (evidence_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison
