from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .phase4 import SelectedAdapterBinding, load_selected_adapter_binding


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _compact_workloads(value: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value[key]
        for key in (
            "schema_version",
            "dataset_schema_version",
            "serialization_id",
            "tokenizer_id",
            "tokenizer_revision",
            "eos_token_id",
        )
    }
    compact["workloads"] = {}
    for name, workload in value["workloads"].items():
        examples = workload["examples"]
        if name in {"train", "validation"}:
            lengths = [len(item["input_ids"]) for item in examples]
            target_lengths = [int(item["completion_tokens"]) + 1 for item in examples]
            token_statistics = {
                "minimum_sequence_tokens": min(lengths),
                "maximum_sequence_tokens": max(lengths),
                "mean_sequence_tokens": fmean(lengths),
                "minimum_supervised_tokens_including_eos": min(target_lengths),
                "maximum_supervised_tokens_including_eos": max(target_lengths),
                "mean_supervised_tokens_including_eos": fmean(target_lengths),
            }
        else:
            lengths = [int(item["prompt_tokens"]) for item in examples]
            token_statistics = {
                "minimum_prompt_tokens": min(lengths),
                "maximum_prompt_tokens": max(lengths),
                "mean_prompt_tokens": fmean(lengths),
            }
        compact["workloads"][name] = {
            "id": workload["id"],
            "split": workload["split"],
            "eligible_examples": workload["eligible_examples"],
            "selected_examples": len(examples),
            "selected_record_ids": workload["selected_record_ids"],
            "token_statistics": token_statistics,
        }
    compact["cross_split_record_ids_disjoint"] = not any(
        set(compact["workloads"][left]["selected_record_ids"])
        & set(compact["workloads"][right]["selected_record_ids"])
        for left, right in (
            ("train", "validation"),
            ("train", "heldout"),
            ("validation", "heldout"),
        )
    )
    return compact


def _compact_training(value: dict[str, Any]) -> dict[str, Any]:
    compact = deepcopy(value)
    for name in ("train", "validation", "heldout"):
        workload = compact["workloads"][name]
        workload.pop("selected_record_ids", None)
        workload["selected_record_ids_reference"] = (
            f"workloads.json#workloads.{name}.selected_record_ids"
        )
    return compact


def _compact_minif2f(run: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    generation_settings = dict(run["generation_settings"])
    adapter = generation_settings.pop("adapter")
    compact_summary = deepcopy(summary)
    selected_binding = deepcopy(run["selected_adapter_binding"])
    return {
        "schema_version": "phase4-minif2f-evidence-v1",
        "status": "passed" if summary["phase4_minif2f_passed"] else "failed",
        "model_id": run["model_id"],
        "model_revision": run["model_revision"],
        "tokenizer_id": run["tokenizer_id"],
        "tokenizer_revision": run["tokenizer_revision"],
        "adapter": {
            "artifact_id": selected_binding["artifact_id"],
            "selected_optimizer_step": selected_binding["selected_optimizer_step"],
            "training_relative_path": selected_binding["training_relative_path"],
            "training_artifact_sha256": selected_binding["training_artifact_sha256"],
            "ignored_local_path": (
                "artifacts/phase4/training/"
                f"{selected_binding['training_relative_path']}"
            ),
            "rank": adapter["adapter_rank"],
            "format": selected_binding["format"],
            "merged": selected_binding["merged"],
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
        "summary": compact_summary,
        "candidate_results_retained_outside_git": True,
    }


def _validate_selected_adapter_evidence(
    binding: SelectedAdapterBinding,
    adapter_reload: dict[str, Any],
    heldout: dict[str, Any],
    minif2f_run: dict[str, Any],
    minif2f_summary: dict[str, Any],
) -> None:
    expected_step = binding.selected_optimizer_step
    expected_id = binding.artifact_id
    expected_relative_path = binding.training_relative_path
    if (
        adapter_reload.get("selected_adapter_binding") != binding.to_dict()
        or int(adapter_reload.get("selected_optimizer_step", -1)) != expected_step
        or adapter_reload.get("adapter_artifact_id") != expected_id
        or adapter_reload.get("adapter_training_relative_path")
        != expected_relative_path
        or adapter_reload.get("training_artifact_sha256")
        != binding.training_artifact_sha256
        or adapter_reload.get("adapter_format") != binding.format
        or adapter_reload.get("adapter_merged") is not binding.merged
    ):
        raise ValueError(
            "Phase 4 adapter reload does not match the training-selected checkpoint"
        )

    heldout_binding = heldout.get("selected_adapter", {})
    heldout_adapter = heldout.get("runs", {}).get("adapter", {}).get("adapter", {})
    if (
        int(heldout.get("selected_optimizer_step", -1)) != expected_step
        or heldout_binding != binding.to_dict()
        or heldout_adapter.get("adapter_id") != expected_id
        or heldout_adapter.get("training_relative_path") != expected_relative_path
        or heldout_adapter.get("training_artifact_sha256")
        != binding.training_artifact_sha256
        or heldout_adapter.get("merged") is not binding.merged
    ):
        raise ValueError(
            "Phase 4 heldout evidence does not match the training-selected checkpoint"
        )

    mini_adapter = minif2f_run.get("generation_settings", {}).get("adapter", {})
    mini_adapter_path = Path(str(mini_adapter.get("adapter_path", ""))).resolve()
    if (
        minif2f_run.get("selected_adapter_binding") != binding.to_dict()
        or minif2f_summary.get("selected_adapter_binding") != binding.to_dict()
        or int(minif2f_summary.get("selected_optimizer_step", -1)) != expected_step
        or minif2f_summary.get("adapter_id") != expected_id
        or minif2f_summary.get("adapter_enabled") is not True
        or mini_adapter.get("adapter_id") != expected_id
        or mini_adapter_path != binding.checkpoint_path
        or mini_adapter.get("merged") is not binding.merged
    ):
        raise ValueError(
            "Phase 4 miniF2F evidence does not match the training-selected checkpoint"
        )


def _heldout_integrity_passed(value: dict[str, Any]) -> bool:
    contract = value.get("evaluation_contract", {})
    model = contract.get("model", {})
    engine = contract.get("inference_engine", {})
    verification = contract.get("verification", {})
    runs = value.get("runs", {})
    base_run = runs.get("base", {})
    adapter_run = runs.get("adapter", {})
    return bool(
        value.get("status") == "passed"
        and value.get("comparison_integrity_passed")
        and all(
            summary.get("complete")
            and int(summary.get("infrastructure_error_count", -1)) == 0
            and int(summary.get("verifier_timeout_count", -1)) == 0
            for summary in (value.get("base", {}), value.get("adapter", {}))
        )
        and model.get("model_revision")
        and model.get("tokenizer_revision")
        and engine.get("name") == "vllm"
        and engine.get("version")
        and contract.get("prompt_format_id") == "whole-proof-v1"
        and contract.get("source_revision")
        and contract.get("lean_toolchain")
        and verification.get("original_source_span_reconstruction") is True
        and verification.get("raw_continuation_no_repair") is True
        and base_run.get("adapter") is None
        and base_run.get("runtime", {}).get("inference_execution") == "local_cuda"
        and adapter_run.get("adapter", {}).get("enabled") is True
        and adapter_run.get("runtime", {}).get("inference_execution") == "local_cuda"
    )


def write_phase4_evidence(artifact_dir: Path, evidence_dir: Path) -> None:
    workloads = _read(artifact_dir / "workloads.json")
    preflight = _read(artifact_dir / "preflight.json")
    training = _read(artifact_dir / "training/run.json")
    adapter_reload = _read(artifact_dir / "adapter-reload.json")
    heldout = _read(artifact_dir / "heldout-comparison.json")
    minif2f_run = _read(artifact_dir / "minif2f/run.json")
    minif2f_summary = _read(artifact_dir / "minif2f/summary.json")
    _, binding = load_selected_adapter_binding(artifact_dir / "training/run.json")
    _validate_selected_adapter_evidence(
        binding,
        adapter_reload,
        heldout,
        minif2f_run,
        minif2f_summary,
    )
    required_passes = {
        "preflight": bool(preflight.get("passed")),
        "training": training.get("status") == "passed",
        "memory": bool(training.get("memory_ceiling_passed")),
        "validation_improved": bool(
            training.get("selected_beats_pre_training_validation")
        ),
        "adapter_reload": bool(adapter_reload.get("passed")),
        "heldout_integrity": _heldout_integrity_passed(heldout),
        "minif2f_integrity": bool(minif2f_summary.get("phase4_minif2f_passed")),
    }
    if not all(required_passes.values()):
        raise ValueError(f"Phase 4 evidence gates are incomplete: {required_passes}")
    compact_workloads = _compact_workloads(workloads)
    if not compact_workloads["cross_split_record_ids_disjoint"]:
        raise ValueError("Phase 4 evidence contains workload leakage")
    selected_step = binding.selected_optimizer_step
    compact_minif2f = _compact_minif2f(minif2f_run, minif2f_summary)

    _write(evidence_dir / "workloads.json", compact_workloads)
    _write(evidence_dir / "preflight.json", preflight)
    compact_training = _compact_training(training)
    compact_training["selected_adapter_binding"] = binding.to_dict()
    _write(evidence_dir / "training.json", compact_training)
    _write(evidence_dir / "adapter-reload.json", adapter_reload)
    _write(evidence_dir / "heldout-comparison.json", heldout)
    _write(evidence_dir / "minif2f.json", compact_minif2f)

    probes = {
        int(item["optimizer_step"]): float(item["mean_target_token_cross_entropy"])
        for item in training["validation_probes"]
    }
    base_metrics = heldout["base"]["pass_at_k"]
    adapter_metrics = heldout["adapter"]["pass_at_k"]
    mini_metrics = minif2f_summary["pass_at_k"]
    readme = f"""# Phase 4 evidence

`workloads.json` records every ordered record ID and eligibility count without committing tokenized dataset rows. The remaining files retain the compact production preflight, full-state two-process training trajectory, selected-adapter reload, heldout comparison, and Phase 1-comparable miniF2F result. Every post-selection artifact is bound to the training-selected adapter by optimizer step, logical identity, canonical training-relative path, and the SHA-256 of the raw training artifact. Checkpoints, adapter weights, raw generations, and detailed candidates remain under ignored `artifacts/phase4/`.

**OBSERVED:** the fixed 4,096-example QLoRA trajectory stopped at optimizer step 256 and resumed in a fresh process to step 512 with optimizer, scheduler, RNG, and derived data position preserved. Validation target-token cross-entropy moved from {training["pre_training_validation"]["mean_target_token_cross_entropy"]:.6f} at step 0 through {probes[128]:.6f}, {probes[256]:.6f}, {probes[384]:.6f}, and {probes[512]:.6f}; validation-only selection chose step {selected_step}. Peak CUDA reserved memory was {training["runtime"]["peak_cuda_reserved_bytes"] / 1024**3:.2f} GiB, below the 24 GiB design ceiling.

**OBSERVED:** on `phase4-heldout64-v1`, unchanged base pass@1/pass@4 were {base_metrics["pass@1"]:.6f}/{base_metrics["pass@4"]:.6f}; the selected adapter produced {adapter_metrics["pass@1"]:.6f}/{adapter_metrics["pass@4"]:.6f}. Both runs completed all 64 tasks and 256 candidates with zero infrastructure errors and zero unresolved verifier timeouts. Adapter miniF2F dev16 pass@1/pass@4/pass@8 were {mini_metrics["pass@1"]:.6f}/{mini_metrics["pass@4"]:.6f}/{mini_metrics["pass@8"]:.6f} over all 128 candidates under the exact Phase 1 contract.

**ACCEPTED:** Phase 4's smoke/data/comparison integrity gates are satisfied. This is evidence that the fixed infrastructure and configuration are safe to consider for Phase 5 design; it is not a full-corpus SFT result or a requirement that smoke quality improve over the base model.
"""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "README.md").write_text(readme, encoding="utf-8")
