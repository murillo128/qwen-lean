from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .phase5 import ordered_record_ids_sha256
from .phase6 import (
    differential_gap_bootstrap,
    paired_task_bootstrap,
)
from .sft2 import (
    SFT2_ENDPOINT_STEP,
    SFT2Config,
    load_sft2_endpoint_binding,
    load_sft2_workloads,
    sha256_file,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sanitized_parent(value: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(value)
    compact.pop("validated_local_path", None)
    return compact


def write_sft2_checkpoint_a_evidence(
    config: SFT2Config,
    workload_path: Path,
    preflight_path: Path,
    output: Path,
) -> dict[str, Any]:
    workloads = load_sft2_workloads(workload_path, config)
    resolved = config.resolve_for_training_examples(len(workloads.train))
    preflight = _read(preflight_path)
    if (
        preflight.get("schema_version") != "sft2-preflight-v1"
        or preflight.get("passed") is not True
        or preflight.get("existing_adapter_continued_without_stacking") is not True
        or preflight.get("only_intended_lora_parameters_trainable") is not True
        or preflight.get("adapter_parameter_changed") is not True
        or preflight.get("frozen_parameter_unchanged") is not True
        or preflight.get("memory_ceiling_passed") is not True
        or preflight.get("continuation_parent", {}).get("unchanged_after_preflight")
        is not True
    ):
        raise ValueError("SFT-2 Checkpoint A preflight is incomplete")
    value = {
        "schema_version": "sft2-checkpoint-a-v1",
        "status": "ready",
        "reference_parent": _sanitized_parent(preflight["continuation_parent"]),
        "model": resolved.model,
        "dataset": resolved.value["dataset"],
        "serialization": resolved.value["serialization"],
        "quantization": resolved.quantization,
        "lora": resolved.lora,
        "training": resolved.training,
        "exact_phase5_workload": {
            "artifact_sha256": sha256_file(workload_path),
            "train_workload_id": resolved.workloads["train"]["id"],
            "train_examples": len(workloads.train),
            "train_ordered_ids_sha256": ordered_record_ids_sha256(
                [item.record_id for item in workloads.train]
            ),
            "validation_workload_id": resolved.workloads["validation"]["id"],
            "validation_examples": len(workloads.validation),
            "validation_ordered_ids_sha256": ordered_record_ids_sha256(
                [item.record_id for item in workloads.validation]
            ),
            "heldout_optimizer_batches": False,
            "miniF2F_optimizer_batches": False,
        },
        "endpoint": {
            "fixed_optimizer_step": SFT2_ENDPOINT_STEP,
            "intermediate_checkpoints_diagnostic_only": [2491, 4981, 7472],
            "validation_selects_endpoint": False,
        },
        "preflight": {
            key: preflight[key]
            for key in (
                "loss",
                "all_gradients_finite",
                "only_intended_lora_parameters_trainable",
                "adapter_parameter_changed",
                "frozen_parameter_unchanged",
                "trainable_parameter_count",
                "total_parameter_count",
                "memory_ceiling_bytes",
                "memory_ceiling_passed",
                "runtime",
            )
        },
        "d015_or_reference_mutation": False,
    }
    _write(output, value)
    return value


def _task_counts(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], key: str
) -> tuple[list[int], list[int]]:
    reference_items = list(reference["per_task"])
    candidate_items = list(candidate["per_task"])
    reference_ids = [str(item["task_id"]) for item in reference_items]
    candidate_ids = [str(item["task_id"]) for item in candidate_items]
    if reference_ids != candidate_ids:
        raise ValueError("SFT-2 paired comparison task identities/order differ")
    return (
        [int(item[key]) for item in reference_items],
        [int(item[key]) for item in candidate_items],
    )


def _delta(
    candidate: Mapping[str, float], reference: Mapping[str, float], keys: Sequence[str]
) -> dict[str, float]:
    return {key: float(candidate[key]) - float(reference[key]) for key in keys}


def train_to_heldout_gap(
    reference_train: Mapping[str, float],
    sft2_train: Mapping[str, float],
    reference_heldout: Mapping[str, float],
    sft2_heldout: Mapping[str, float],
    *,
    keys: Sequence[str] = ("pass@1", "pass@4"),
) -> dict[str, dict[str, float]]:
    value: dict[str, dict[str, float]] = {}
    for key in keys:
        reference_gap = float(reference_train[key]) - float(reference_heldout[key])
        sft2_gap = float(sft2_train[key]) - float(sft2_heldout[key])
        value[key] = {
            "reference_sft_v1_train_minus_heldout": reference_gap,
            "sft2_train_minus_heldout": sft2_gap,
            "change_sft2_minus_reference": sft2_gap - reference_gap,
        }
    return value


def _compact_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(dict(value))
    compact.pop("per_task", None)
    return compact


def _compact_training(value: Mapping[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(dict(value))
    for name in ("train", "validation", "heldout"):
        ids = compact["workloads"][name].pop("selected_record_ids")
        compact["workloads"][name]["selected_record_ids_sha256"] = (
            ordered_record_ids_sha256(ids)
        )
        compact["workloads"][name]["membership_reference"] = (
            f"exact Phase 5 {compact['workloads'][name]['id']}"
        )
    compact["continuation_parent"] = _sanitized_parent(compact["continuation_parent"])
    return compact


def _interval_spans_zero(value: Mapping[str, Any]) -> bool:
    low, high = value["ci95"]
    return float(low) <= 0.0 <= float(high)


def write_sft2_final_evidence(
    config: SFT2Config,
    artifact_dir: Path,
    reference_train_dir: Path,
    reference_heldout_dir: Path,
    reference_minif2f_dir: Path,
    phase6_comparison_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    preflight = _read(artifact_dir / "preflight.json")
    training_path = artifact_dir / "training/run.json"
    training = _read(training_path)
    reload = _read(artifact_dir / "adapter-reload.json")
    checkpoint_path = artifact_dir / "checkpoint-a.json"
    if not checkpoint_path.is_file():
        checkpoint_path = evidence_dir / "checkpoint-a.json"
    checkpoint_a = _read(checkpoint_path)
    train = _read(artifact_dir / "train512/summary.json")
    heldout = _read(artifact_dir / "heldout512/summary.json")
    minif2f = _read(artifact_dir / "minif2f-validation/summary.json")
    reference_train = _read(reference_train_dir / "summary.json")
    reference_heldout = _read(reference_heldout_dir / "summary.json")
    reference_minif2f = _read(reference_minif2f_dir / "summary.json")
    phase6_comparison = _read(phase6_comparison_path)
    _, binding = load_sft2_endpoint_binding(
        training_path,
        expected_artifact_id=str(config.lora["artifact_id"]),
        adapter_dir=artifact_dir
        / "training/trainer-state"
        / f"checkpoint-{SFT2_ENDPOINT_STEP}",
    )

    accounting = training.get("trajectory", {}).get("one_pass_data_accounting", {})
    logs = training.get("runtime", {}).get("training_log_summary", {})
    gates = {
        "checkpoint_a_ready": checkpoint_a.get("status") == "ready",
        "preflight_passed": preflight.get("passed") is True,
        "immutable_parent_preserved": training.get("continuation_parent", {}).get(
            "unchanged_after_process_leg"
        )
        is True,
        "training_passed": training.get("status") == "passed",
        "fixed_q4_endpoint": binding.selected_optimizer_step == SFT2_ENDPOINT_STEP,
        "all_examples_once": bool(
            accounting.get("all_eligible_examples_consumed_exactly_once")
            and not accounting.get("duplicate_final_batch_fill", True)
            and int(accounting.get("eligible_training_examples", -1)) == 79696
        ),
        "finite_training": bool(
            logs.get("covers_every_optimizer_step_exactly_once")
            and logs.get("all_losses_finite")
            and logs.get("all_gradient_norms_finite")
            and int(logs.get("logged_optimizer_steps", -1)) == SFT2_ENDPOINT_STEP
        ),
        "memory": training.get("memory_ceiling_passed") is True,
        "adapter_reload": reload.get("passed") is True,
        "train512": train.get("sft2_train_integrity_passed") is True,
        "heldout512": heldout.get("sft2_heldout_integrity_passed") is True,
        "minif2f_validation": bool(
            minif2f.get("sft2_minif2f_validation_integrity_passed")
            and minif2f.get("miniF2F_test_evaluated") is False
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"SFT-2 final evidence gates are incomplete: {gates}")

    expected_reference = {
        "train_lean": {"pass@1": 0.03271484375, "pass@4": 0.087890625},
        "train_exact": {"pass@1": 0.01611328125, "pass@4": 0.04296875},
        "heldout_lean": {"pass@1": 0.0166015625, "pass@4": 0.048828125},
        "minif2f_lean": {
            "pass@1": 0.03944672131147541,
            "pass@4": 0.1031615925058548,
            "pass@8": 0.14344262295081966,
        },
    }
    observed_reference = {
        "train_lean": reference_train["pass_at_k"],
        "train_exact": reference_train["exact_target_pass_at_k"],
        "heldout_lean": reference_heldout["pass_at_k"],
        "minif2f_lean": reference_minif2f["pass_at_k"],
    }
    if observed_reference != expected_reference:
        raise ValueError(
            "retained reference-sft-v1 metrics differ from accepted evidence"
        )

    train_reference_counts, train_sft2_counts = _task_counts(
        reference_train, train, "verified_candidate_count"
    )
    exact_reference_counts, exact_sft2_counts = _task_counts(
        reference_train, train, "exact_target_candidate_count"
    )
    heldout_reference_counts, heldout_sft2_counts = _task_counts(
        reference_heldout, heldout, "verified_candidate_count"
    )
    mini_reference_counts, mini_sft2_counts = _task_counts(
        reference_minif2f, minif2f, "verified_candidate_count"
    )
    bootstrap = {
        "train_lean": paired_task_bootstrap(
            train_reference_counts,
            train_sft2_counts,
            candidates_per_task=4,
            ks=(1, 4),
            resamples=10000,
            seed=0,
        ),
        "train_exact_target": paired_task_bootstrap(
            exact_reference_counts,
            exact_sft2_counts,
            candidates_per_task=4,
            ks=(1, 4),
            resamples=10000,
            seed=0,
        ),
        "heldout_lean": paired_task_bootstrap(
            heldout_reference_counts,
            heldout_sft2_counts,
            candidates_per_task=4,
            ks=(1, 4),
            resamples=10000,
            seed=0,
        ),
        "minif2f_validation_lean": paired_task_bootstrap(
            mini_reference_counts,
            mini_sft2_counts,
            candidates_per_task=8,
            ks=(1, 4, 8),
            resamples=10000,
            seed=0,
        ),
        "train_minus_heldout_increment": differential_gap_bootstrap(
            train_reference_counts,
            train_sft2_counts,
            heldout_reference_counts,
            heldout_sft2_counts,
            candidates_per_task=4,
            ks=(1, 4),
            resamples=10000,
            seed=0,
        ),
    }

    deltas = {
        "train_lean": _delta(
            train["pass_at_k"], reference_train["pass_at_k"], ("pass@1", "pass@4")
        ),
        "train_exact_target": _delta(
            train["exact_target_pass_at_k"],
            reference_train["exact_target_pass_at_k"],
            ("pass@1", "pass@4"),
        ),
        "heldout_lean": _delta(
            heldout["pass_at_k"], reference_heldout["pass_at_k"], ("pass@1", "pass@4")
        ),
        "minif2f_validation_lean": _delta(
            minif2f["pass_at_k"],
            reference_minif2f["pass_at_k"],
            ("pass@1", "pass@4", "pass@8"),
        ),
    }
    gap = train_to_heldout_gap(
        reference_train["pass_at_k"],
        train["pass_at_k"],
        reference_heldout["pass_at_k"],
        heldout["pass_at_k"],
    )
    retained_phase6_gap = phase6_comparison["train_to_heldout_generalization_gaps"]
    for key in ("pass@1", "pass@4"):
        if gap[key]["reference_sft_v1_train_minus_heldout"] != float(
            retained_phase6_gap[key]["sft_train_gap"]
        ):
            raise ValueError("retained Phase 6 train-to-heldout gap differs")
        gap[key]["bootstrap_ci95"] = bootstrap["train_minus_heldout_increment"][
            "differential_gap"
        ][key]["ci95"]

    primary_intervals = (
        bootstrap["train_lean"]["metrics"]["pass@4"]["delta_adapter_minus_base"],
        bootstrap["heldout_lean"]["metrics"]["pass@4"]["delta_adapter_minus_base"],
        bootstrap["minif2f_validation_lean"]["metrics"]["pass@4"][
            "delta_adapter_minus_base"
        ],
    )
    train_delta = deltas["train_lean"]["pass@4"]
    heldout_delta = deltas["heldout_lean"]["pass@4"]
    mini_delta = deltas["minif2f_validation_lean"]["pass@4"]
    exact_delta = deltas["train_exact_target"]["pass@4"]
    outcome_signals = {
        "additional_learning_or_generalization": bool(
            heldout_delta > 0 or mini_delta > 0
        ),
        "memorization_heavy_gain": bool(
            train_delta > 0
            and exact_delta > 0
            and train_delta > heldout_delta
            and train_delta > mini_delta
        ),
        "saturation": all(_interval_spans_zero(item) for item in primary_intervals),
        "overfitting_or_regression": bool(
            train_delta > 0
            and (
                heldout_delta < 0
                or mini_delta < 0
                or deltas["minif2f_validation_lean"]["pass@8"] < 0
            )
        ),
        "signals_are_nonexclusive": True,
        "opaque_combined_score": False,
    }
    comparison = {
        "schema_version": "sft2-comparison-v1",
        "status": "passed",
        "quality_improvement_required": False,
        "reference_control": "reference-sft-v1",
        "candidate": "sft2-q4-v1-lora",
        "fixed_endpoint_step": SFT2_ENDPOINT_STEP,
        "workloads": {
            "phase6-train512-v1": {
                "reference": {
                    "lean": reference_train["pass_at_k"],
                    "exact_target": reference_train["exact_target_pass_at_k"],
                },
                "sft2": {
                    "lean": train["pass_at_k"],
                    "exact_target": train["exact_target_pass_at_k"],
                    "exact_target_candidates": train["exact_target_candidates"],
                    "tasks_with_exact_target_candidate": train[
                        "tasks_with_exact_target_candidate"
                    ],
                    "verified_non_exact_candidates": train[
                        "verified_non_exact_candidates"
                    ],
                    "tasks_with_verified_non_exact_candidate": train[
                        "tasks_with_verified_non_exact_candidate"
                    ],
                    "finish_reason_counts": train["finish_reason_counts"],
                    "generated_token_counts": train["generated_token_counts"],
                },
                "delta_sft2_minus_reference": {
                    "lean": deltas["train_lean"],
                    "exact_target": deltas["train_exact_target"],
                },
            },
            "phase5-heldout512-v1": {
                "reference": reference_heldout["pass_at_k"],
                "sft2": heldout["pass_at_k"],
                "delta_sft2_minus_reference": deltas["heldout_lean"],
                "sft2_finish_reason_counts": heldout["finish_reason_counts"],
                "sft2_generated_token_counts": heldout["generated_token_counts"],
            },
            "minif2f-valid-v1": {
                "reference": reference_minif2f["pass_at_k"],
                "sft2": minif2f["pass_at_k"],
                "delta_sft2_minus_reference": deltas["minif2f_validation_lean"],
                "sft2_finish_reason_counts": minif2f["finish_reason_counts"],
                "sft2_generated_token_counts": minif2f["generated_token_counts"],
                "miniF2F_test_evaluated": False,
            },
        },
        "train_to_heldout_gap": gap,
        "bootstrap": bootstrap,
        "outcome_signals": outcome_signals,
        "result_types_kept_distinct": [
            "Lean-verified proof success",
            "exact retained-target reproduction",
            "verified non-exact alternatives",
            "teacher-forced validation metrics",
        ],
        "raw_candidate_results_retained_outside_git": True,
    }

    compact_preflight = copy.deepcopy(preflight)
    compact_preflight["continuation_parent"] = _sanitized_parent(
        compact_preflight["continuation_parent"]
    )
    _write(evidence_dir / "checkpoint-a.json", checkpoint_a)
    _write(evidence_dir / "preflight.json", compact_preflight)
    _write(evidence_dir / "training.json", _compact_training(training))
    _write(evidence_dir / "adapter-reload.json", reload)
    _write(evidence_dir / "comparison.json", comparison)

    probes = [training["pre_training_validation"], *training["validation_probes"]]
    trajectory = ", ".join(
        f"{int(item.get('optimizer_step', 0))}: "
        f"CE {float(item['mean_target_token_cross_entropy']):.6f}, "
        f"accuracy {float(item['target_token_next_token_accuracy']):.6f}"
        for item in probes
    )
    readme = f"""# SFT-2 ablation evidence

The immutable `reference-sft-v1` adapter was continued without merging or stacking a second LoRA. A fresh optimizer and fresh 312-step cosine warmup were started for this stage. All 79,696 exact Phase 5 training members were consumed once in 9,962 staged updates; Q1/Q2/Q3 remained diagnostic and the primary endpoint was fixed at Q4 before any SFT-2 result was observed.

**OBSERVED:** full Phase 5 validation trajectory (staged step: target-token metrics) was {trajectory}. Peak CUDA reserved memory was {training["runtime"]["peak_cuda_reserved_bytes"] / 1024**3:.2f} GiB. All logged losses and gradients were finite, the reference parent hashes remained unchanged, and the Q4 adapter reloaded in a fresh process.

**OBSERVED:** SFT-2 minus `reference-sft-v1` Lean pass@1/pass@4 deltas were {deltas["train_lean"]["pass@1"]:.6f}/{deltas["train_lean"]["pass@4"]:.6f} on train512 and {deltas["heldout_lean"]["pass@1"]:.6f}/{deltas["heldout_lean"]["pass@4"]:.6f} on heldout512. Exact-target train512 deltas were {deltas["train_exact_target"]["pass@1"]:.6f}/{deltas["train_exact_target"]["pass@4"]:.6f}. miniF2F validation pass@1/pass@4/pass@8 deltas were {deltas["minif2f_validation_lean"]["pass@1"]:.6f}/{deltas["minif2f_validation_lean"]["pass@4"]:.6f}/{deltas["minif2f_validation_lean"]["pass@8"]:.6f}. `comparison.json` retains deterministic paired bootstrap intervals and separate learning, memorization, saturation, and regression signals; no opaque combined score is used.

**ACCEPTED:** the bounded ablation and all integrity gates completed. Quality was not an execution gate. D015 and `reference-sft-v1` remain unchanged, miniF2F test was not evaluated, and SFT-2 is not automatically promoted or published as the reference parent.
"""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison
