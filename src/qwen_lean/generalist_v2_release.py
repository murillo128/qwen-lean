from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .generalist_v2 import GeneralistV2Config
from .generalist_v2_dataset import sha256_file


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"generalist-v2 evidence is not an object: {path}")
    return value


def _headline_final_workloads(final: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for workload_id, workload in final["workloads"].items():
        output[workload_id] = {
            "task_count": workload["task_count"],
            "models": {
                label: {
                    "pass_at_k": lane["pass_at_k"],
                    "tasks_with_verified_candidate": lane[
                        "tasks_with_verified_candidate"
                    ],
                    "verified_candidate_count": lane["category_counts"]["verified"],
                }
                for label, lane in workload["models"].items()
            },
            "deepseek_gap_closed_fraction": workload["deepseek_gap_closed_fraction"],
        }
    return output


def compact_generalist_v2_release_evidence(
    config: GeneralistV2Config,
    binding_path: Path,
    training_path: Path,
    selection_path: Path,
    extended_path: Path,
    final_path: Path,
    historical_path: Path,
    deepseek_preflight_path: Path,
    lora_parity_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind the selected adapter identity to all complete experiment evidence."""

    config.validate()
    binding = _read(binding_path)
    training = _read(training_path)
    selection = _read(selection_path)
    extended = _read(extended_path)
    final = _read(final_path)
    historical = _read(historical_path)
    deepseek_preflight = _read(deepseek_preflight_path)
    from .generalist_v2_parity import validate_lora_parity_gate

    lora_parity_gate = validate_lora_parity_gate(config, lora_parity_path)
    selected = str(selection.get("selection", {}).get("selected_checkpoint", ""))
    adapter_hash = str(
        selection.get("selected_checkpoint", {}).get("adapter_model_sha256", "")
    )
    trained_checkpoint = training.get("checkpoints", {}).get(selected, {})
    if (
        binding.get("schema_version") != "generalist-v2-dataset-binding-v1"
        or binding.get("artifact_id") != "qwen-lean-generalist-v2"
        or training.get("schema_version") != "generalist-v2-full-training-v1"
        or training.get("status") != "passed"
        or selection.get("schema_version") != "generalist-v2-checkpoint-selection-v1"
        or selection.get("status") != "frozen"
        or selected not in {"Q1", "Q2", "Q3", "Q4"}
        or not adapter_hash
        or trained_checkpoint.get("adapter_model_sha256") != adapter_hash
        or extended.get("schema_version") != "generalist-v2-extended-validation-v1"
        or extended.get("status") != "selected-checkpoint-extended-validation-complete"
        or extended.get("screening_selected_checkpoint") != selected
        or extended.get("evaluated_checkpoint", {}).get("adapter_model_sha256")
        != adapter_hash
        or final.get("schema_version") != "generalist-v2-final-assessment-v1"
        or final.get("status") != "complete"
        or final.get("selected_checkpoint") != selected
        or final.get("selected_adapter_model_sha256") != adapter_hash
        or historical.get("schema_version") != "generalist-v2-historical-riemann-v1"
        or historical.get("status") != "complete"
        or historical.get("selected_checkpoint") != selected
        or historical.get("selected_adapter_model_sha256") != adapter_hash
        or deepseek_preflight.get("schema_version")
        != "generalist-v2-deepseek-final-preflight-v1"
        or deepseek_preflight.get("status") != "passed"
        or deepseek_preflight.get("selected_checkpoint_frozen") != selected
    ):
        raise ValueError("generalist-v2 release evidence is incomplete or inconsistent")
    if (
        binding["dataset"]["manifest_sha256"]
        != training["dataset"]["binding_manifest_sha256"]
        or binding["dataset"]["manifest_sha256"]
        != config.dataset["binding"]["manifest_sha256"]
        or binding["serialization"]["lengths"]["selected_context_tokens"]
        != config.training["resolved_context_tokens"]
        or binding["trajectory"]["optimizer_visible_variants"]
        != training["training"]["trajectory"]["optimizer_visible_variants"]
    ):
        raise ValueError("generalist-v2 release binding or trajectory differs")

    selected_screening = selection["checkpoints"][selected]["workloads"]
    evidence = {
        "schema_version": "generalist-v2-release-evidence-v1",
        "status": "ready-for-review",
        "artifact_id": "qwen-lean-generalist-v2",
        "parent": training["model"],
        "adapter": {
            "format": training["adapter"]["format"],
            "merged": training["adapter"]["merged"],
            "base_model_shards_saved": training["adapter"]["base_model_shards_saved"],
            "selected_checkpoint": selected,
            "optimizer_step": trained_checkpoint["optimizer_step"],
            "adapter_config_sha256": trained_checkpoint["adapter_config_sha256"],
            "adapter_model_sha256": adapter_hash,
            "trainable_parameter_count": training["adapter"][
                "trainable_parameter_count"
            ],
            "weights_committed_to_git": False,
            "weights_published_to_artifact_store": False,
        },
        "dataset": {
            "package_id": config.dataset["package_id"],
            "manifest_sha256": binding["dataset"]["manifest_sha256"],
            "canonical_records_sha256": binding["dataset"]["files"]["records.jsonl.gz"][
                "sha256"
            ],
            "general_train_sha256": binding["dataset"]["general_train"]["sha256"],
            "training_statements": binding["resolved"]["general_train"]["statements"],
            "training_proof_variants": binding["resolved"]["general_train"][
                "proof_variants"
            ],
        },
        "serialization": {
            "id": binding["serialization"]["id"],
            "context_tokens": binding["serialization"]["lengths"][
                "selected_context_tokens"
            ],
            "completion_and_one_eos_supervised": binding["serialization"][
                "completion_and_one_eos_supervised"
            ],
            "prompt_supervised": binding["serialization"]["prompt_supervised"],
            "packing": binding["serialization"]["packing"],
            "truncation": binding["serialization"]["truncation"],
        },
        "weighting": binding["weights"],
        "training": {
            "lane": training["selected_lane"],
            "lora": config.lora,
            "optimizer_and_schedule": config.training,
            "completed_optimizer_steps": training["training"][
                "completed_optimizer_steps"
            ],
            "exactly_one_complete_pass": training["training"][
                "exactly_one_complete_pass"
            ],
            "source_run_sha256": training["source_run_sha256"],
        },
        "screening": {
            "selection_rule": selection["selection"]["rule"],
            "selected_checkpoint": selected,
            "screening_ranking": selection["selection"]["screening_ranking"],
            "selected_workloads": {
                workload_id: {
                    "pass_at_k": value["pass_at_k"],
                    "tasks_with_verified_candidate": value[
                        "tasks_with_verified_candidate"
                    ],
                }
                for workload_id, value in selected_screening.items()
            },
            "test_or_riemann_used_for_selection": False,
        },
        "extended_validation": extended["evaluated_checkpoint"]["workloads"],
        "final_assessment": _headline_final_workloads(final),
        "historical_riemann": {
            "interpretation": historical["interpretation"],
            "clean_unseen_generalization": historical["clean_unseen_generalization"],
            "selected": {
                "pass_at_k": historical["selected"]["pass_at_k"],
                "tasks_with_verified_candidate": historical["selected"][
                    "tasks_with_verified_candidate"
                ],
                "category_counts": historical["selected"]["category_counts"],
            },
            "accepted_anchors": historical["accepted_anchors"],
            "paired_solved_within_4": historical["paired_solved_within_4"],
        },
        "deepseek_final_preflight": deepseek_preflight,
        "lora_inference_parity_gate": lora_parity_gate,
        "evidence_sha256": {
            "dataset_binding": sha256_file(binding_path),
            "full_training": sha256_file(training_path),
            "checkpoint_selection": sha256_file(selection_path),
            "extended_validation": sha256_file(extended_path),
            "final_assessment": sha256_file(final_path),
            "historical_riemann": sha256_file(historical_path),
            "deepseek_final_preflight": sha256_file(deepseek_preflight_path),
            "lora_inference_parity": sha256_file(lora_parity_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence
