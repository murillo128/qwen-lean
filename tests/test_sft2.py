import copy
import json
from pathlib import Path

import pytest

from qwen_lean.phase5 import derive_phase5_trajectory
from qwen_lean.sft2 import (
    SFT2_ARTIFACT_ID,
    SFT2_ENDPOINT_STEP,
    SFT2Config,
    load_sft2_endpoint_binding,
    validate_step0_reference,
)
from qwen_lean.sft2_evidence import _delta, _task_counts, train_to_heldout_gap


ROOT = Path(__file__).resolve().parents[1]


def test_sft2_config_pins_parent_phase5_contract_and_fixed_trajectory() -> None:
    config = SFT2Config.load(ROOT / "config/sft2-ablation.json")
    resolved = config.resolve_for_training_examples(79696)
    trajectory = derive_phase5_trajectory(79696, effective_batch_size=8)

    assert config.parent["logical_id"] == "reference-sft-v1"
    assert config.parent["immutable"] is True
    assert config.parent_adapter["hub_revision"] == (
        "5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5"
    )
    assert config.lora["artifact_id"] == SFT2_ARTIFACT_ID
    assert config.training["staged_optimizer_restart"] is True
    assert config.training["staged_scheduler_restart"] is True
    assert config.training["uninterrupted_two_epoch_equivalence_claimed"] is False
    assert config.training["endpoint_policy"] == "fixed_complete_q4"
    assert trajectory.maximum_optimizer_steps == SFT2_ENDPOINT_STEP
    assert trajectory.checkpoint_candidates == (2491, 4981, 7472, 9962)
    assert trajectory.warmup_steps == 312
    assert trajectory.final_optimizer_update_examples == 8
    assert resolved.training["checkpoint_candidates"] == [2491, 4981, 7472, 9962]

    changed = copy.deepcopy(config.value)
    changed["training"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="differs from Phase 5"):
        SFT2Config(path=config.path, value=changed).validate()


def test_sft2_config_forbids_test_and_pins_exact_membership_hashes() -> None:
    config = SFT2Config.load(ROOT / "config/sft2-ablation.json")

    assert config.value["minif2f"] == {
        "phase1_config": "config/phase1-minif2f.json",
        "workload_id": "minif2f-valid-v1",
        "expected_tasks": 244,
        "candidates_per_task": 8,
        "test_evaluation_forbidden": True,
    }
    assert config.value["phase5_inputs"]["train_eligible_examples"] == 79696
    assert config.value["phase5_inputs"]["train_ordered_ids_sha256"] == (
        "0ec5ef7e969774924384d80d04b5ea9ea6e0eabac9f38cf3deff9924c714d816"
    )
    assert config.value["phase5_inputs"]["validation_ordered_ids_sha256"] == (
        "c4b7189e6b6b61a8bf22b807554cd534180fe73e9fb2ffca1a2cf0e8c603401c"
    )


def _write_training_fixture(path: Path, selected_step: int = 9962) -> Path:
    checkpoint = path / "trainer-state" / f"checkpoint-{selected_step}"
    checkpoint.mkdir(parents=True)
    training = {
        "schema_version": "sft2-training-run-v1",
        "status": "passed",
        "checkpoint_selection": {
            "selected_optimizer_step": selected_step,
            "heldout_or_minif2f_consulted": False,
            "validation_influenced_endpoint": False,
            "diagnostic_only_steps": [2491, 4981, 7472],
        },
        "adapter": {
            "artifact_id": SFT2_ARTIFACT_ID,
            "relative_path": f"trainer-state/checkpoint-{selected_step}",
            "format": "peft-lora",
            "merged": False,
        },
        "trajectory": {
            "one_pass_data_accounting": {
                "all_eligible_examples_consumed_exactly_once": True
            }
        },
    }
    training_path = path / "run.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    return training_path


def test_sft2_binding_accepts_only_fixed_complete_q4(tmp_path: Path) -> None:
    training_path = _write_training_fixture(tmp_path)
    adapter_dir = tmp_path / "trainer-state/checkpoint-9962"

    _, binding = load_sft2_endpoint_binding(
        training_path,
        expected_artifact_id=SFT2_ARTIFACT_ID,
        adapter_dir=adapter_dir,
    )

    assert binding.selected_optimizer_step == 9962
    assert binding.checkpoint_path == adapter_dir.resolve()

    bad_path = _write_training_fixture(tmp_path / "bad", selected_step=7472)
    with pytest.raises(ValueError, match="fixed complete Q4"):
        load_sft2_endpoint_binding(bad_path)


def test_sft2_step0_must_reproduce_reference_validation() -> None:
    config = SFT2Config.load(ROOT / "config/sft2-ablation.json")
    expected = config.value["step0_reference_validation"]
    metrics = {
        "examples": expected["examples"],
        "target_tokens": expected["target_tokens"],
        "mean_target_token_cross_entropy": expected["mean_target_token_cross_entropy"],
        "target_token_next_token_accuracy": expected[
            "target_token_next_token_accuracy"
        ],
    }

    validate_step0_reference(config, metrics)

    metrics["mean_target_token_cross_entropy"] += 0.01
    with pytest.raises(RuntimeError, match="cross-entropy mismatch"):
        validate_step0_reference(config, metrics)


def test_sft2_comparison_keeps_paired_tasks_and_gap_math_explicit() -> None:
    reference = {
        "per_task": [
            {"task_id": "a", "verified_candidate_count": 0},
            {"task_id": "b", "verified_candidate_count": 1},
        ]
    }
    candidate = {
        "per_task": [
            {"task_id": "a", "verified_candidate_count": 2},
            {"task_id": "b", "verified_candidate_count": 1},
        ]
    }

    assert _task_counts(reference, candidate, "verified_candidate_count") == (
        [0, 1],
        [2, 1],
    )
    assert _delta(
        {"pass@1": 0.2, "pass@4": 0.5},
        {"pass@1": 0.1, "pass@4": 0.25},
        ("pass@1", "pass@4"),
    ) == pytest.approx({"pass@1": 0.1, "pass@4": 0.25})
    gaps = train_to_heldout_gap(
        {"pass@1": 0.1, "pass@4": 0.2},
        {"pass@1": 0.2, "pass@4": 0.5},
        {"pass@1": 0.05, "pass@4": 0.1},
        {"pass@1": 0.1, "pass@4": 0.2},
    )
    assert gaps["pass@4"] == pytest.approx(
        {
            "reference_sft_v1_train_minus_heldout": 0.1,
            "sft2_train_minus_heldout": 0.3,
            "change_sft2_minus_reference": 0.2,
        }
    )

    reversed_candidate = {"per_task": list(reversed(candidate["per_task"]))}
    with pytest.raises(ValueError, match="identities/order"):
        _task_counts(reference, reversed_candidate, "verified_candidate_count")
