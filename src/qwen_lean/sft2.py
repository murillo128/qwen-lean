from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .phase4 import SelectedAdapterBinding, load_selected_adapter_binding
from .phase5 import (
    Phase5Config,
    Phase5Workloads,
    derive_phase5_trajectory,
    load_phase5_workloads,
    ordered_record_ids_sha256,
)
from .phase6 import Phase6Config, load_reference_candidate


SFT2_CONFIG_SCHEMA_VERSION = "sft2-config-v1"
SFT2_TRAINING_SCHEMA_VERSION = "sft2-training-run-v1"
SFT2_ARTIFACT_ID = "sft2-q4-v1-lora"
SFT2_ENDPOINT_STEP = 9962


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SFT2Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> SFT2Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        config = cls(path=path.resolve(), value=value)
        config.validate()
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def workloads(self) -> dict[str, Any]:
        return self.value["workloads"]

    @property
    def quantization(self) -> dict[str, Any]:
        return self.value["quantization"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.value["lora"]

    @property
    def training(self) -> dict[str, Any]:
        return self.value["training"]

    @property
    def parent(self) -> dict[str, Any]:
        return self.value["reference_parent"]

    @property
    def parent_adapter(self) -> dict[str, Any]:
        return self.parent["adapter"]

    def phase1_validation_config(self) -> Any:
        from .minif2f import Phase1Config

        return Phase1Config.load(
            self.path.parents[1] / str(self.value["minif2f"]["phase1_config"])
        )

    def phase6_config(self) -> Phase6Config:
        return Phase6Config.load(self.path.parent / "phase6-eval.json")

    def resolve_for_training_examples(self, count: int) -> SFT2Config:
        value = copy.deepcopy(self.value)
        effective_batch = int(value["training"]["per_device_micro_batch_size"]) * int(
            value["training"]["gradient_accumulation_steps"]
        )
        trajectory = derive_phase5_trajectory(
            count, effective_batch_size=effective_batch
        )
        value["training"].update(
            {
                "maximum_optimizer_steps": trajectory.maximum_optimizer_steps,
                "warmup_steps": trajectory.warmup_steps,
                "checkpoint_candidates": list(trajectory.checkpoint_candidates),
                "mandatory_process_stop_step": trajectory.mandatory_process_stop_step,
                "checkpoint_interval_steps": trajectory.maximum_optimizer_steps,
            }
        )
        resolved = SFT2Config(path=self.path, value=value)
        resolved.validate()
        return resolved

    def validate(self) -> None:
        if self.value.get("schema_version") != SFT2_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unknown SFT-2 config schema: {self.value.get('schema_version')}"
            )
        project_root = self.path.parents[1]
        phase5 = Phase5Config.load(project_root / "config/phase5-full.json")
        for section in (
            "model",
            "dataset",
            "serialization",
            "workloads",
            "quantization",
            "heldout_generation",
        ):
            if self.value.get(section) != phase5.value[section]:
                raise ValueError(f"SFT-2 {section} differs from Phase 5")

        lora_shape = {
            key: value for key, value in self.lora.items() if key != "artifact_id"
        }
        phase5_lora_shape = {
            key: value for key, value in phase5.lora.items() if key != "artifact_id"
        }
        if self.lora.get("artifact_id") != SFT2_ARTIFACT_ID:
            raise ValueError("SFT-2 output adapter identity differs")
        if lora_shape != phase5_lora_shape:
            raise ValueError("SFT-2 LoRA shape differs from Phase 5")

        phase5_training_keys = (
            "trainer",
            "per_device_micro_batch_size",
            "gradient_accumulation_steps",
            "maximum_sequence_tokens",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "maximum_gradient_norm",
            "lr_schedule",
            "warmup_fraction_denominator",
            "seed",
            "packing",
            "truncation",
            "gradient_checkpointing",
            "epochs",
            "duplicate_final_batch_fill",
            "manual_checkpoint_boundaries",
            "memory_ceiling_bytes",
        )
        for key in phase5_training_keys:
            if self.training.get(key) != phase5.training[key]:
                raise ValueError(f"SFT-2 training.{key} differs from Phase 5")
        required_stage = {
            "staged_optimizer_restart": True,
            "staged_scheduler_restart": True,
            "uninterrupted_two_epoch_equivalence_claimed": False,
            "endpoint_policy": "fixed_complete_q4",
        }
        for key, expected in required_stage.items():
            if self.training.get(key) != expected:
                raise ValueError(f"SFT-2 training.{key} must be {expected!r}")

        phase6 = self.phase6_config()
        expected_parent = {
            "logical_id": "reference-sft-v1",
            "immutable": True,
            "adapter": phase6.adapter,
        }
        if self.parent != expected_parent:
            raise ValueError("SFT-2 parent differs from immutable D015 identity")

        inputs = self.value["phase5_inputs"]
        expected_inputs = {
            "workload_artifact_sha256": "c25fa88ac5a8553e179e651a31ac4403e53cd42cd4e5cb2164755a24e4d85636",
            "train_ordered_ids_sha256": phase6.value["phase5_inputs"][
                "train_ordered_ids_sha256"
            ],
            "validation_ordered_ids_sha256": "c4b7189e6b6b61a8bf22b807554cd534180fe73e9fb2ffca1a2cf0e8c603401c",
            "heldout_ordered_ids_sha256": phase6.value["phase5_inputs"][
                "heldout_ordered_ids_sha256"
            ],
            "train_eligible_examples": 79696,
            "validation_eligible_examples": 4426,
        }
        if inputs != expected_inputs:
            raise ValueError("SFT-2 exact Phase 5 input binding differs")

        expected_step0 = {
            "mean_target_token_cross_entropy": 1.108313809418986,
            "target_token_next_token_accuracy": 0.7075208913649025,
            "target_tokens": 308381,
            "examples": 4426,
            "maximum_cross_entropy_absolute_delta": 0.005,
            "maximum_accuracy_absolute_delta": 0.002,
        }
        if self.value.get("step0_reference_validation") != expected_step0:
            raise ValueError("SFT-2 step-0 reference validation contract differs")
        if self.value.get("train_evaluation") != {
            "workload_id": "phase6-train512-v1",
            "expected_examples": 512,
            "workload_ordered_ids_sha256": "ad0c5f4e9cf158f324bac101f6d9bb492483a6c2f70399bea254b9d944adc2c8",
            "candidates_per_task": 4,
        }:
            raise ValueError("SFT-2 train512 evaluation contract differs")
        if self.value.get("minif2f") != {
            "phase1_config": "config/phase1-minif2f.json",
            "workload_id": "minif2f-valid-v1",
            "expected_tasks": 244,
            "candidates_per_task": 8,
            "test_evaluation_forbidden": True,
        }:
            raise ValueError("SFT-2 miniF2F validation contract differs")
        if self.value.get("bootstrap") != {
            "resamples": 10000,
            "seed": 0,
            "interval_percentiles": [2.5, 97.5],
        }:
            raise ValueError("SFT-2 bootstrap contract differs")

        derived_keys = {
            "maximum_optimizer_steps",
            "warmup_steps",
            "checkpoint_candidates",
            "mandatory_process_stop_step",
            "checkpoint_interval_steps",
        }
        present = derived_keys & self.training.keys()
        if present and present != derived_keys:
            raise ValueError("SFT-2 resolved training trajectory is incomplete")
        if present:
            expected = {
                "maximum_optimizer_steps": 9962,
                "warmup_steps": 312,
                "checkpoint_candidates": [2491, 4981, 7472, 9962],
                "mandatory_process_stop_step": 4981,
                "checkpoint_interval_steps": 9962,
            }
            for key, wanted in expected.items():
                if self.training[key] != wanted:
                    raise ValueError(f"SFT-2 resolved training.{key} differs")


def load_sft2_workloads(path: Path, config: SFT2Config) -> Phase5Workloads:
    expected_hash = str(config.value["phase5_inputs"]["workload_artifact_sha256"])
    if sha256_file(path) != expected_hash:
        raise ValueError("SFT-2 workload artifact differs from exact Phase 5 artifact")
    workloads = load_phase5_workloads(path, config)  # type: ignore[arg-type]
    inputs = config.value["phase5_inputs"]
    ids = {
        "train": [item.record_id for item in workloads.train],
        "validation": [item.record_id for item in workloads.validation],
        "heldout": [item.record_id for item in workloads.heldout],
    }
    for name in ("train", "validation", "heldout"):
        if ordered_record_ids_sha256(ids[name]) != inputs[f"{name}_ordered_ids_sha256"]:
            raise ValueError(f"SFT-2 {name} membership differs from Phase 5")
    if len(workloads.train) != int(inputs["train_eligible_examples"]):
        raise ValueError("SFT-2 train count differs from Phase 5")
    if len(workloads.validation) != int(inputs["validation_eligible_examples"]):
        raise ValueError("SFT-2 validation count differs from Phase 5")
    return workloads


def validate_sft2_parent(
    config: SFT2Config, adapter_dir: Path, candidate_manifest_path: Path
) -> dict[str, Any]:
    manifest = load_reference_candidate(
        config.phase6_config(), candidate_manifest_path, adapter_dir
    )
    return {
        "logical_id": manifest["logical_id"],
        "immutable": True,
        "model": config.model,
        "adapter": dict(config.parent_adapter),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "validated_local_path": str(adapter_dir.resolve()),
        "source_kind": manifest["adapter"]["local_source_kind"],
    }


def validate_step0_reference(config: SFT2Config, metrics: dict[str, Any]) -> None:
    expected = config.value["step0_reference_validation"]
    if int(metrics.get("examples", -1)) != int(expected["examples"]):
        raise RuntimeError("SFT-2 step-0 validation example count differs")
    if int(metrics.get("target_tokens", -1)) != int(expected["target_tokens"]):
        raise RuntimeError("SFT-2 step-0 validation target-token count differs")
    ce_delta = abs(
        float(metrics["mean_target_token_cross_entropy"])
        - float(expected["mean_target_token_cross_entropy"])
    )
    accuracy_delta = abs(
        float(metrics["target_token_next_token_accuracy"])
        - float(expected["target_token_next_token_accuracy"])
    )
    if ce_delta > float(expected["maximum_cross_entropy_absolute_delta"]):
        raise RuntimeError("SFT-2 step-0 validation cross-entropy mismatch")
    if accuracy_delta > float(expected["maximum_accuracy_absolute_delta"]):
        raise RuntimeError("SFT-2 step-0 validation accuracy mismatch")


def load_sft2_endpoint_binding(
    training_path: Path,
    *,
    expected_artifact_id: str | None = None,
    adapter_dir: Path | None = None,
) -> tuple[dict[str, Any], SelectedAdapterBinding]:
    training, binding = load_selected_adapter_binding(
        training_path,
        expected_artifact_id=expected_artifact_id,
        adapter_dir=adapter_dir,
    )
    selection = training.get("checkpoint_selection") or {}
    if training.get("schema_version") != SFT2_TRAINING_SCHEMA_VERSION:
        raise ValueError("adapter is not from an SFT-2 training run")
    if (
        binding.selected_optimizer_step != SFT2_ENDPOINT_STEP
        or selection.get("validation_influenced_endpoint") is not False
        or selection.get("diagnostic_only_steps") != [2491, 4981, 7472]
        or training.get("trajectory", {})
        .get("one_pass_data_accounting", {})
        .get("all_eligible_examples_consumed_exactly_once")
        is not True
    ):
        raise ValueError("SFT-2 adapter is not the fixed complete Q4 endpoint")
    return training, binding
