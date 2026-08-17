from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .phase2_corpus import load_phase2_dataset
from .phase2_schema import MathlibProofRecord
from .phase3 import (
    BASE_MODEL_ID,
    BASE_REVISION,
    SFT_SERIALIZATION_ID,
    TokenizedSFTExample,
    load_pinned_tokenizer,
    render_sft_prompt,
    tokenize_sft_record,
)
from .phase4 import (
    HeldoutPromptExample,
    SelectedAdapterBinding,
    load_selected_adapter_binding,
)


PHASE5_CONFIG_SCHEMA_VERSION = "phase5-config-v1"
PHASE5_WORKLOAD_SCHEMA_VERSION = "phase5-workloads-v1"
TRAIN_WORKLOAD_ID = "phase5-train-full-v1"
VALIDATION_WORKLOAD_ID = "phase5-validation-full-v1"
HELDOUT_WORKLOAD_ID = "phase5-heldout512-v1"


@dataclass(frozen=True)
class Phase5Trajectory:
    eligible_training_examples: int
    effective_batch_size: int
    maximum_optimizer_steps: int
    warmup_steps: int
    checkpoint_candidates: tuple[int, int, int, int]
    mandatory_process_stop_step: int
    final_optimizer_update_examples: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checkpoint_candidates"] = list(self.checkpoint_candidates)
        return value


def derive_phase5_trajectory(
    eligible_training_examples: int, *, effective_batch_size: int = 8
) -> Phase5Trajectory:
    if eligible_training_examples < 1 or effective_batch_size < 1:
        raise ValueError("Phase 5 trajectory counts must be positive")
    steps = math.ceil(eligible_training_examples / effective_batch_size)
    candidates = (
        math.ceil(steps / 4),
        math.ceil(steps / 2),
        math.ceil(3 * steps / 4),
        steps,
    )
    if len(set(candidates)) != 4:
        raise ValueError("Phase 5 full trajectory must have four distinct quarters")
    final_examples = eligible_training_examples - effective_batch_size * (steps - 1)
    return Phase5Trajectory(
        eligible_training_examples=eligible_training_examples,
        effective_batch_size=effective_batch_size,
        maximum_optimizer_steps=steps,
        warmup_steps=math.ceil(steps / 32),
        checkpoint_candidates=candidates,
        mandatory_process_stop_step=candidates[1],
        final_optimizer_update_examples=final_examples,
    )


@dataclass(frozen=True)
class Phase5Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Phase5Config:
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

    def resolve_for_training_examples(self, count: int) -> Phase5Config:
        value = copy.deepcopy(self.value)
        effective_batch_size = int(
            value["training"]["per_device_micro_batch_size"]
        ) * int(value["training"]["gradient_accumulation_steps"])
        trajectory = derive_phase5_trajectory(
            count, effective_batch_size=effective_batch_size
        )
        value["training"].update(
            {
                "maximum_optimizer_steps": trajectory.maximum_optimizer_steps,
                "warmup_steps": trajectory.warmup_steps,
                "checkpoint_candidates": list(trajectory.checkpoint_candidates),
                "mandatory_process_stop_step": (trajectory.mandatory_process_stop_step),
                # Required by the shared trainer's save configuration; Phase 5
                # itself requests saves only at the explicit uneven boundaries.
                "checkpoint_interval_steps": trajectory.maximum_optimizer_steps,
            }
        )
        resolved = Phase5Config(path=self.path, value=value)
        resolved.validate()
        return resolved

    def validate(self) -> None:
        if self.value.get("schema_version") != PHASE5_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unknown Phase 5 config schema: {self.value.get('schema_version')}"
            )
        expected: tuple[tuple[str, ...], Any] = (
            (("model", "model_id"), BASE_MODEL_ID),
            (("model", "model_revision"), BASE_REVISION),
            (("model", "tokenizer_id"), BASE_MODEL_ID),
            (("model", "tokenizer_revision"), BASE_REVISION),
            (("model", "add_special_tokens"), False),
            (("model", "chat_template"), None),
            (("dataset", "schema_version"), "mathlib-whole-proof-v1"),
            (("serialization", "id"), SFT_SERIALIZATION_ID),
            (("serialization", "supervise_prompt"), False),
            (("serialization", "supervise_eos"), True),
            (("workloads", "train", "id"), TRAIN_WORKLOAD_ID),
            (("workloads", "train", "split"), "train"),
            (("workloads", "train", "selection"), "all_eligible"),
            (("workloads", "train", "expected_input_examples"), 80062),
            (("workloads", "train", "maximum_sequence_tokens"), 1024),
            (("workloads", "validation", "id"), VALIDATION_WORKLOAD_ID),
            (("workloads", "validation", "split"), "validation"),
            (("workloads", "validation", "selection"), "all_eligible"),
            (("workloads", "validation", "expected_input_examples"), 4448),
            (("workloads", "validation", "maximum_sequence_tokens"), 1024),
            (("workloads", "heldout", "id"), HELDOUT_WORKLOAD_ID),
            (("workloads", "heldout", "split"), "heldout"),
            (
                ("workloads", "heldout", "selection_hash_prefix"),
                f"{HELDOUT_WORKLOAD_ID}\0",
            ),
            (("workloads", "heldout", "expected_input_examples"), 4448),
            (("workloads", "heldout", "expected_examples"), 512),
            (
                ("workloads", "heldout", "maximum_prompt_and_generation_tokens"),
                2048,
            ),
            (("quantization", "load_in_4bit"), True),
            (("quantization", "quantization_type"), "nf4"),
            (("quantization", "double_quantization"), True),
            (("quantization", "compute_dtype"), "bfloat16"),
            (("lora", "task_type"), "CAUSAL_LM"),
            (("lora", "r"), 16),
            (("lora", "lora_alpha"), 32),
            (("lora", "lora_dropout"), 0.0),
            (("lora", "bias"), "none"),
            (("lora", "modules_to_save"), None),
            (("training", "trainer"), "trl-sft"),
            (("training", "per_device_micro_batch_size"), 1),
            (("training", "gradient_accumulation_steps"), 8),
            (("training", "maximum_sequence_tokens"), 1024),
            (("training", "optimizer"), "paged_adamw_8bit"),
            (("training", "learning_rate"), 0.0001),
            (("training", "weight_decay"), 0.0),
            (("training", "maximum_gradient_norm"), 1.0),
            (("training", "lr_schedule"), "cosine"),
            (("training", "warmup_fraction_denominator"), 32),
            (("training", "seed"), 0),
            (("training", "packing"), False),
            (("training", "truncation"), False),
            (("training", "gradient_checkpointing"), True),
            (("training", "epochs"), 1),
            (("training", "duplicate_final_batch_fill"), False),
            (("training", "manual_checkpoint_boundaries"), True),
            (("training", "memory_ceiling_bytes"), 24 * 1024**3),
            (("heldout_generation", "candidates_per_task"), 4),
            (("heldout_generation", "do_sample"), True),
            (("heldout_generation", "temperature"), 0.8),
            (("heldout_generation", "top_p"), 0.95),
            (("heldout_generation", "top_k"), -1),
            (("heldout_generation", "max_new_tokens"), 1024),
            (
                ("heldout_generation", "stop"),
                "tokenizer_eos_or_token_limit",
            ),
            (("heldout_generation", "seed"), 0),
            (("minif2f", "phase1_config"), "config/phase1-minif2f.json"),
            (("minif2f", "workload_id"), "minif2f-valid-v1"),
        )
        for path, wanted in expected:
            observed: Any = self.value
            for key in path:
                observed = observed[key]
            if observed != wanted:
                joined = ".".join(path)
                raise ValueError(
                    f"Phase 5 {joined} must be {wanted!r}, got {observed!r}"
                )

        required_modules = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        if tuple(self.lora["target_modules"]) != required_modules:
            raise ValueError(
                "Phase 5 LoRA target modules differ from the accepted order"
            )
        if int(self.value["verification"]["workers"]) < 1:
            raise ValueError("Phase 5 verification workers must be positive")
        for key in ("heldout_timeout_seconds", "minif2f_timeout_seconds"):
            if float(self.value["verification"][key]) <= 0:
                raise ValueError(f"Phase 5 verification {key} must be positive")

        derived_keys = {
            "maximum_optimizer_steps",
            "warmup_steps",
            "checkpoint_candidates",
            "mandatory_process_stop_step",
            "checkpoint_interval_steps",
        }
        present = derived_keys & self.training.keys()
        if present and present != derived_keys:
            raise ValueError("Phase 5 resolved training trajectory is incomplete")


@dataclass(frozen=True)
class OverlengthSFTRecord:
    record_id: str
    serialized_tokens: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OverlengthSFTRecord:
        return cls(
            record_id=str(value["record_id"]),
            serialized_tokens=int(value["serialized_tokens"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase5Workloads:
    train: tuple[TokenizedSFTExample, ...]
    validation: tuple[TokenizedSFTExample, ...]
    heldout: tuple[HeldoutPromptExample, ...]
    input_counts: dict[str, int]
    eligible_counts: dict[str, int]
    overlength: dict[str, tuple[OverlengthSFTRecord, ...]]
    eos_token_id: int
    trajectory: Phase5Trajectory


def _digest(prefix: str, record_id: str) -> bytes:
    return hashlib.sha256(prefix.encode("utf-8") + record_id.encode("utf-8")).digest()


def ordered_record_ids_sha256(record_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record_id in record_ids:
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def select_full_sft_workload(
    records: Iterable[MathlibProofRecord],
    tokenizer: Any,
    config: Phase5Config,
    workload_name: str,
) -> tuple[list[TokenizedSFTExample], list[OverlengthSFTRecord], int]:
    if workload_name not in {"train", "validation"}:
        raise ValueError(f"unknown Phase 5 SFT workload: {workload_name}")
    workload = config.workloads[workload_name]
    split = str(workload["split"])
    maximum = int(workload["maximum_sequence_tokens"])
    selected: list[TokenizedSFTExample] = []
    excluded: list[OverlengthSFTRecord] = []
    seen: set[str] = set()
    input_count = 0
    for record in records:
        input_count += 1
        if record.split != split:
            raise ValueError(
                f"Phase 5 {workload_name} input contains {record.split} record "
                f"{record.id}"
            )
        if record.id in seen:
            raise ValueError(
                f"duplicate Phase 5 {workload_name} candidate record ID: {record.id}"
            )
        seen.add(record.id)
        example = tokenize_sft_record(record, tokenizer, expected_split=split)
        length = len(example.input_ids)
        if length > maximum:
            excluded.append(
                OverlengthSFTRecord(
                    record_id=record.id,
                    serialized_tokens=length,
                )
            )
        else:
            example.validate(int(tokenizer.eos_token_id), maximum)
            selected.append(example)
    return selected, excluded, input_count


def select_phase5_heldout_workload(
    records: Iterable[MathlibProofRecord],
    tokenizer: Any,
    config: Phase5Config,
) -> tuple[list[HeldoutPromptExample], int, int]:
    workload = config.workloads["heldout"]
    generation_tokens = int(config.value["heldout_generation"]["max_new_tokens"])
    maximum = int(workload["maximum_prompt_and_generation_tokens"])
    eligible: list[tuple[bytes, HeldoutPromptExample]] = []
    seen: set[str] = set()
    input_count = 0
    for record in records:
        input_count += 1
        if record.split != "heldout":
            raise ValueError(
                f"Phase 5 heldout input contains {record.split} record {record.id}"
            )
        if record.id in seen:
            raise ValueError(
                f"duplicate Phase 5 heldout candidate record ID: {record.id}"
            )
        seen.add(record.id)
        prompt = render_sft_prompt(record)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens + generation_tokens > maximum:
            continue
        eligible.append(
            (
                _digest(str(workload["selection_hash_prefix"]), record.id),
                HeldoutPromptExample(
                    record_id=record.id,
                    declaration_name=record.declaration_name,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                ),
            )
        )
    eligible.sort(key=lambda item: item[0])
    expected_count = int(workload["expected_examples"])
    if len(eligible) < expected_count:
        raise ValueError(
            f"{HELDOUT_WORKLOAD_ID} requires {expected_count} examples; "
            f"only {len(eligible)} are eligible"
        )
    return (
        [example for _, example in eligible[:expected_count]],
        len(eligible),
        input_count,
    )


def _validate_input_count(config: Phase5Config, name: str, observed: int) -> None:
    expected = int(config.workloads[name]["expected_input_examples"])
    if observed != expected:
        raise ValueError(
            f"Phase 5 {name} input count is {observed}, expected {expected}"
        )


def materialize_phase5_workloads(
    artifact_dir: Path, config: Phase5Config, tokenizer: Any | None = None
) -> Phase5Workloads:
    tokenizer = load_pinned_tokenizer(config) if tokenizer is None else tokenizer
    dataset = load_phase2_dataset(artifact_dir)
    train, train_excluded, train_input = select_full_sft_workload(
        (MathlibProofRecord.from_dict(value) for value in dataset["train"]),
        tokenizer,
        config,
        "train",
    )
    validation, validation_excluded, validation_input = select_full_sft_workload(
        (MathlibProofRecord.from_dict(value) for value in dataset["validation"]),
        tokenizer,
        config,
        "validation",
    )
    heldout, heldout_eligible, heldout_input = select_phase5_heldout_workload(
        (MathlibProofRecord.from_dict(value) for value in dataset["heldout"]),
        tokenizer,
        config,
    )
    for name, count in (
        ("train", train_input),
        ("validation", validation_input),
        ("heldout", heldout_input),
    ):
        _validate_input_count(config, name, count)
    effective_batch_size = int(config.training["per_device_micro_batch_size"]) * int(
        config.training["gradient_accumulation_steps"]
    )
    workloads = Phase5Workloads(
        train=tuple(train),
        validation=tuple(validation),
        heldout=tuple(heldout),
        input_counts={
            "train": train_input,
            "validation": validation_input,
            "heldout": heldout_input,
        },
        eligible_counts={
            "train": len(train),
            "validation": len(validation),
            "heldout": heldout_eligible,
        },
        overlength={
            "train": tuple(train_excluded),
            "validation": tuple(validation_excluded),
        },
        eos_token_id=int(tokenizer.eos_token_id),
        trajectory=derive_phase5_trajectory(
            len(train), effective_batch_size=effective_batch_size
        ),
    )
    _validate_phase5_workload_integrity(workloads)
    return workloads


def _validate_phase5_workload_integrity(workloads: Phase5Workloads) -> None:
    id_sets = {
        "train": {example.record_id for example in workloads.train},
        "validation": {example.record_id for example in workloads.validation},
        "heldout": {example.record_id for example in workloads.heldout},
    }
    if any(
        id_sets[left] & id_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "heldout"),
            ("validation", "heldout"),
        )
    ):
        raise ValueError("Phase 5 workloads contain cross-split record leakage")
    for name in ("train", "validation"):
        excluded_ids = {item.record_id for item in workloads.overlength[name]}
        if excluded_ids & id_sets[name]:
            raise ValueError(f"Phase 5 {name} includes an over-length record")
        if len(excluded_ids) != len(workloads.overlength[name]):
            raise ValueError(f"Phase 5 {name} over-length IDs are duplicated")
        if (
            len(id_sets[name]) + len(excluded_ids) != workloads.input_counts[name]
            or len(id_sets[name]) != workloads.eligible_counts[name]
        ):
            raise ValueError(f"Phase 5 {name} full-corpus accounting differs")


def write_phase5_workloads(
    output: Path, config: Phase5Config, workloads: Phase5Workloads
) -> dict[str, Any]:
    _validate_phase5_workload_integrity(workloads)
    value = {
        "schema_version": PHASE5_WORKLOAD_SCHEMA_VERSION,
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "serialization_id": config.value["serialization"]["id"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "eos_token_id": workloads.eos_token_id,
        "trajectory": workloads.trajectory.to_dict(),
        "workloads": {
            "train": _sft_workload_value(config, workloads, "train", workloads.train),
            "validation": _sft_workload_value(
                config, workloads, "validation", workloads.validation
            ),
            "heldout": {
                "id": config.workloads["heldout"]["id"],
                "split": "heldout",
                "input_examples": workloads.input_counts["heldout"],
                "eligible_examples": workloads.eligible_counts["heldout"],
                "selected_record_ids": [item.record_id for item in workloads.heldout],
                "selected_record_ids_sha256": ordered_record_ids_sha256(
                    [item.record_id for item in workloads.heldout]
                ),
                "examples": [item.to_dict() for item in workloads.heldout],
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def _sft_workload_value(
    config: Phase5Config,
    workloads: Phase5Workloads,
    name: str,
    examples: Sequence[TokenizedSFTExample],
) -> dict[str, Any]:
    record_ids = [item.record_id for item in examples]
    return {
        "id": config.workloads[name]["id"],
        "split": name,
        "input_examples": workloads.input_counts[name],
        "eligible_examples": workloads.eligible_counts[name],
        "overlength_examples": len(workloads.overlength[name]),
        "overlength_records": [item.to_dict() for item in workloads.overlength[name]],
        "selected_record_ids": record_ids,
        "selected_record_ids_sha256": ordered_record_ids_sha256(record_ids),
        "examples": [item.to_dict() for item in examples],
    }


def load_phase5_workloads(path: Path, config: Phase5Config) -> Phase5Workloads:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PHASE5_WORKLOAD_SCHEMA_VERSION:
        raise ValueError(
            f"unknown Phase 5 workload schema: {value.get('schema_version')}"
        )
    expected_metadata = {
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "serialization_id": config.value["serialization"]["id"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
    }
    for key, expected in expected_metadata.items():
        if value.get(key) != expected:
            raise ValueError(f"Phase 5 workload {key} differs from the configuration")

    eos_token_id = int(value["eos_token_id"])
    loaded: dict[str, tuple[TokenizedSFTExample, ...]] = {}
    input_counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    overlength: dict[str, tuple[OverlengthSFTRecord, ...]] = {}
    for name in ("train", "validation"):
        item = value["workloads"][name]
        expected = config.workloads[name]
        if item.get("id") != expected["id"] or item.get("split") != name:
            raise ValueError(f"Phase 5 {name} workload identity differs")
        examples = tuple(TokenizedSFTExample.from_dict(row) for row in item["examples"])
        ids = [example.record_id for example in examples]
        if ids != item.get("selected_record_ids"):
            raise ValueError(f"Phase 5 {name} selected record order differs")
        if ordered_record_ids_sha256(ids) != item.get("selected_record_ids_sha256"):
            raise ValueError(f"Phase 5 {name} selected record hash differs")
        for example in examples:
            example.validate(eos_token_id, int(expected["maximum_sequence_tokens"]))
        exclusions = tuple(
            OverlengthSFTRecord.from_dict(row)
            for row in item.get("overlength_records", [])
        )
        if any(
            excluded.serialized_tokens <= int(expected["maximum_sequence_tokens"])
            for excluded in exclusions
        ):
            raise ValueError(f"Phase 5 {name} excluded an eligible record")
        loaded[name] = examples
        input_counts[name] = int(item["input_examples"])
        eligible_counts[name] = int(item["eligible_examples"])
        overlength[name] = exclusions
        _validate_input_count(config, name, input_counts[name])

    heldout_item = value["workloads"]["heldout"]
    if (
        heldout_item.get("id") != config.workloads["heldout"]["id"]
        or heldout_item.get("split") != "heldout"
    ):
        raise ValueError("Phase 5 heldout workload identity differs")
    heldout = tuple(
        HeldoutPromptExample.from_dict(row) for row in heldout_item["examples"]
    )
    heldout_ids = [item.record_id for item in heldout]
    if heldout_ids != heldout_item.get("selected_record_ids"):
        raise ValueError("Phase 5 heldout selected record order differs")
    if ordered_record_ids_sha256(heldout_ids) != heldout_item.get(
        "selected_record_ids_sha256"
    ):
        raise ValueError("Phase 5 heldout selected record hash differs")
    if len(heldout) != int(config.workloads["heldout"]["expected_examples"]):
        raise ValueError("Phase 5 heldout workload count differs")
    maximum = int(config.workloads["heldout"]["maximum_prompt_and_generation_tokens"])
    generation_tokens = int(config.value["heldout_generation"]["max_new_tokens"])
    if any(item.prompt_tokens + generation_tokens > maximum for item in heldout):
        raise ValueError("Phase 5 heldout workload exceeds the model-length boundary")
    input_counts["heldout"] = int(heldout_item["input_examples"])
    eligible_counts["heldout"] = int(heldout_item["eligible_examples"])
    _validate_input_count(config, "heldout", input_counts["heldout"])

    effective_batch_size = int(config.training["per_device_micro_batch_size"]) * int(
        config.training["gradient_accumulation_steps"]
    )
    trajectory = derive_phase5_trajectory(
        len(loaded["train"]), effective_batch_size=effective_batch_size
    )
    if value.get("trajectory") != trajectory.to_dict():
        raise ValueError("Phase 5 derived trajectory differs from workload membership")
    if "maximum_optimizer_steps" in config.training:
        resolved = config.resolve_for_training_examples(len(loaded["train"]))
        if resolved.training != config.training:
            raise ValueError("Phase 5 resolved config differs from workload trajectory")
    workloads = Phase5Workloads(
        train=loaded["train"],
        validation=loaded["validation"],
        heldout=heldout,
        input_counts=input_counts,
        eligible_counts=eligible_counts,
        overlength=overlength,
        eos_token_id=eos_token_id,
        trajectory=trajectory,
    )
    _validate_phase5_workload_integrity(workloads)
    return workloads


def load_phase5_selected_adapter_binding(
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
    if training.get("schema_version") != "phase5-training-run-v1":
        raise ValueError("selected adapter is not from a Phase 5 training run")
    return training, binding
