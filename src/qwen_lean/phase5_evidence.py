from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .phase4_evidence import (
    _heldout_integrity_passed,
    _validate_selected_adapter_evidence,
)
from .phase5 import (
    load_phase5_selected_adapter_binding,
    ordered_record_ids_sha256,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _token_statistics(
    examples: list[dict[str, Any]], *, heldout: bool
) -> dict[str, Any]:
    if heldout:
        lengths = [int(item["prompt_tokens"]) for item in examples]
        return {
            "minimum_prompt_tokens": min(lengths),
            "maximum_prompt_tokens": max(lengths),
            "mean_prompt_tokens": fmean(lengths),
        }
    lengths = [len(item["input_ids"]) for item in examples]
    targets = [int(item["completion_tokens"]) + 1 for item in examples]
    return {
        "minimum_sequence_tokens": min(lengths),
        "maximum_sequence_tokens": max(lengths),
        "mean_sequence_tokens": fmean(lengths),
        "minimum_supervised_tokens_including_eos": min(targets),
        "maximum_supervised_tokens_including_eos": max(targets),
        "mean_supervised_tokens_including_eos": fmean(targets),
    }


def compact_phase5_workloads(value: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value[key]
        for key in (
            "schema_version",
            "dataset_schema_version",
            "serialization_id",
            "tokenizer_id",
            "tokenizer_revision",
            "eos_token_id",
            "trajectory",
        )
    }
    compact["workloads"] = {}
    selected_sets: dict[str, set[str]] = {}
    for name in ("train", "validation", "heldout"):
        workload = value["workloads"][name]
        ids = [str(item) for item in workload["selected_record_ids"]]
        selected_sets[name] = set(ids)
        item = {
            "id": workload["id"],
            "split": workload["split"],
            "input_examples": workload["input_examples"],
            "eligible_examples": workload["eligible_examples"],
            "selected_examples": len(workload["examples"]),
            "selected_record_ids_sha256": ordered_record_ids_sha256(ids),
            "token_statistics": _token_statistics(
                workload["examples"], heldout=name == "heldout"
            ),
        }
        if name == "heldout":
            item["selected_record_ids"] = ids
        else:
            item["overlength_examples"] = workload["overlength_examples"]
            item["overlength_records"] = workload["overlength_records"]
            item["membership"] = "every input split record not listed as over-length"
        compact["workloads"][name] = item
    compact["cross_split_record_ids_disjoint"] = not any(
        selected_sets[left] & selected_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "heldout"),
            ("validation", "heldout"),
        )
    )
    return compact


def _compact_training(value: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(value)
    for name in ("train", "validation", "heldout"):
        compact["workloads"][name].pop("selected_record_ids", None)
        compact["workloads"][name]["membership_reference"] = (
            f"workloads.json#workloads.{name}"
        )
    return compact


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(summary)
    compact.pop("per_task", None)
    return compact


def _compact_heldout(value: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(value)
    compact["base"] = _compact_summary(compact["base"])
    compact["adapter"] = _compact_summary(compact["adapter"])
    return compact


def _compact_minif2f(run: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    generation_settings = dict(run["generation_settings"])
    adapter = generation_settings.pop("adapter")
    binding = run["selected_adapter_binding"]
    return {
        "schema_version": "phase5-minif2f-evidence-v1",
        "status": "passed" if summary["phase5_minif2f_passed"] else "failed",
        "model_id": run["model_id"],
        "model_revision": run["model_revision"],
        "tokenizer_id": run["tokenizer_id"],
        "tokenizer_revision": run["tokenizer_revision"],
        "adapter": {
            "artifact_id": binding["artifact_id"],
            "selected_optimizer_step": binding["selected_optimizer_step"],
            "training_relative_path": binding["training_relative_path"],
            "training_artifact_sha256": binding["training_artifact_sha256"],
            "ignored_local_path": (
                f"artifacts/phase5/training/{binding['training_relative_path']}"
            ),
            "rank": adapter["adapter_rank"],
            "format": binding["format"],
            "merged": binding["merged"],
        },
        "workload_id": run["workload_id"],
        "benchmark_repository": run["benchmark_repository"],
        "benchmark_revision": run["benchmark_revision"],
        "lean_toolchain": run["lean_toolchain"],
        "mathlib_revision": run["mathlib_revision"],
        "generation_settings": generation_settings,
        "inference_engine": run["inference_engine"],
        "inference_engine_version": run["inference_engine_version"],
        "runtime": run["runtime"],
        "summary": _compact_summary(summary),
        "candidate_results_retained_outside_git": True,
    }


def write_phase5_evidence(artifact_dir: Path, evidence_dir: Path) -> None:
    workloads = _read(artifact_dir / "workloads.json")
    preflight = _read(artifact_dir / "preflight.json")
    training_path = artifact_dir / "training/run.json"
    training = _read(training_path)
    adapter_reload = _read(artifact_dir / "adapter-reload.json")
    heldout = _read(artifact_dir / "heldout-comparison.json")
    minif2f_run = _read(artifact_dir / "minif2f/run.json")
    minif2f_summary = _read(artifact_dir / "minif2f/summary.json")
    _, binding = load_phase5_selected_adapter_binding(training_path)
    _validate_selected_adapter_evidence(
        binding,
        adapter_reload,
        heldout,
        minif2f_run,
        minif2f_summary,
    )
    accounting = training.get("trajectory", {}).get("one_pass_data_accounting", {})
    process_legs = training.get("trajectory", {}).get("process_legs", [])
    resume = training.get("trajectory", {}).get("resume_state") or {}
    training_logs = training.get("runtime", {}).get("training_log_summary", {})
    required_passes = {
        "preflight": bool(preflight.get("passed")),
        "training": training.get("status") == "passed",
        "memory": bool(training.get("memory_ceiling_passed")),
        "validation_improved": bool(
            training.get("selected_beats_pre_training_validation")
        ),
        "all_train_examples_once": bool(
            accounting.get("all_eligible_examples_consumed_exactly_once")
            and not accounting.get("duplicate_final_batch_fill", True)
        ),
        "two_process_resume": bool(
            len(process_legs) == 2
            and training.get("trajectory", {}).get("same_trajectory_resume")
            and all(
                resume.get(key)
                for key in (
                    "optimizer_state_preserved",
                    "scheduler_state_preserved",
                    "rng_state_preserved",
                    "data_position_preserved",
                )
            )
        ),
        "finite_training_logs": bool(
            training_logs.get("covers_every_optimizer_step_exactly_once")
            and training_logs.get("all_losses_finite")
            and training_logs.get("all_gradient_norms_finite")
            and int(training_logs.get("logged_optimizer_steps", -1))
            == int(training.get("optimizer_steps_completed", -2))
        ),
        "adapter_reload": bool(adapter_reload.get("passed")),
        "heldout_integrity": _heldout_integrity_passed(heldout),
        "minif2f_integrity": bool(
            minif2f_summary.get("phase5_minif2f_passed")
            and not minif2f_summary.get("miniF2F_test_evaluated", True)
        ),
    }
    if not all(required_passes.values()):
        raise ValueError(f"Phase 5 evidence gates are incomplete: {required_passes}")
    compact_workloads = compact_phase5_workloads(workloads)
    if not compact_workloads["cross_split_record_ids_disjoint"]:
        raise ValueError("Phase 5 evidence contains workload leakage")

    compact_training = _compact_training(training)
    compact_training["selected_adapter_binding"] = binding.to_dict()
    compact_heldout = _compact_heldout(heldout)
    compact_minif2f = _compact_minif2f(minif2f_run, minif2f_summary)
    _write(evidence_dir / "workloads.json", compact_workloads)
    _write(evidence_dir / "preflight.json", preflight)
    _write(evidence_dir / "training.json", compact_training)
    _write(evidence_dir / "adapter-reload.json", adapter_reload)
    _write(evidence_dir / "heldout-comparison.json", compact_heldout)
    _write(evidence_dir / "minif2f.json", compact_minif2f)

    probes = {
        int(item["optimizer_step"]): float(item["mean_target_token_cross_entropy"])
        for item in training["validation_probes"]
    }
    candidates = [int(item) for item in training["training"]["checkpoint_candidates"]]
    selected_step = binding.selected_optimizer_step
    base_metrics = heldout["base"]["pass_at_k"]
    adapter_metrics = heldout["adapter"]["pass_at_k"]
    mini_metrics = minif2f_summary["pass_at_k"]
    train_count = int(accounting["eligible_training_examples"])
    validation_count = int(training["workloads"]["validation"]["examples"])
    readme = f"""# Phase 5 evidence

`workloads.json` retains full-corpus counts, all over-length exclusions, hashes of the ordered full train/validation memberships, and the explicit ordered 512 heldout IDs. The remaining files retain compact production preflight, two-process training, selected-adapter reload, heldout comparison, and full miniF2F validation evidence. Checkpoints, adapter weights, tokenized workload rows, raw generations, model caches, and bulky logs remain under ignored `artifacts/phase5/`.

**OBSERVED:** `{train_count}` eligible training records were consumed exactly once in {accounting["planned_optimizer_steps"]} optimizer updates, with {accounting["final_optimizer_update_examples"]} record(s) in the final partial update and no duplicate fill. The production process stopped at Q2 step {candidates[1]} and resumed in a fresh process with optimizer, scheduler, RNG, and data position preserved. Full `{validation_count}`-record validation target-token cross-entropy moved from {training["pre_training_validation"]["mean_target_token_cross_entropy"]:.6f} at step 0 through {", ".join(f"{probes[step]:.6f}" for step in candidates)}; validation-only selection chose step {selected_step}. Peak CUDA reserved memory was {training["runtime"]["peak_cuda_reserved_bytes"] / 1024**3:.2f} GiB, below the 24 GiB design ceiling.

**OBSERVED:** per-step logging covers all {training_logs["logged_optimizer_steps"]} optimizer updates exactly once. Every recorded loss and pre-clipping gradient norm is finite; loss ranged from {training_logs["loss"]["minimum"]:.6f} to {training_logs["loss"]["maximum"]:.6f}, and gradient norm ranged from {training_logs["gradient_norm_before_clipping"]["minimum"]:.6f} to {training_logs["gradient_norm_before_clipping"]["maximum"]:.6f}. End-to-end training throughput including boundary validation was {training["runtime"]["cumulative_examples_per_second"]:.3f} examples/s.

**OBSERVED:** on `phase5-heldout512-v1`, unchanged base pass@1/pass@4 were {base_metrics["pass@1"]:.6f}/{base_metrics["pass@4"]:.6f}; the selected adapter produced {adapter_metrics["pass@1"]:.6f}/{adapter_metrics["pass@4"]:.6f}. Both runs completed 512 tasks and 2,048 candidates with zero infrastructure errors and zero unresolved verifier timeouts. The selected adapter's full miniF2F validation pass@1/pass@4/pass@8 were {mini_metrics["pass@1"]:.6f}/{mini_metrics["pass@4"]:.6f}/{mini_metrics["pass@8"]:.6f} over all 244 tasks and 1,952 candidates beside the accepted unchanged-base Phase 1 evidence.

**ACCEPTED:** Phase 5's full-corpus execution and comparison-integrity gates are satisfied. Semantic improvement or regression remains an observed result for Phase 6 analysis; it did not influence checkpoint selection.
"""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "README.md").write_text(readme, encoding="utf-8")
