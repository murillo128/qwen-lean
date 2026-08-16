from __future__ import annotations

import hashlib
import json
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


PHASE4_CONFIG_SCHEMA_VERSION = "phase4-config-v1"
PHASE4_WORKLOAD_SCHEMA_VERSION = "phase4-workloads-v1"
TRAIN_WORKLOAD_ID = "phase4-train4096-v1"
VALIDATION_WORKLOAD_ID = "phase4-validation512-v1"
HELDOUT_WORKLOAD_ID = "phase4-heldout64-v1"


@dataclass(frozen=True)
class Phase4Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Phase4Config:
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

    def validate(self) -> None:
        if self.value.get("schema_version") != PHASE4_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unknown Phase 4 config schema: {self.value.get('schema_version')}"
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
            (
                ("workloads", "train", "selection_hash_prefix"),
                f"{TRAIN_WORKLOAD_ID}\0",
            ),
            (("workloads", "train", "expected_examples"), 4096),
            (("workloads", "train", "maximum_sequence_tokens"), 1024),
            (("workloads", "validation", "id"), VALIDATION_WORKLOAD_ID),
            (("workloads", "validation", "split"), "validation"),
            (
                ("workloads", "validation", "selection_hash_prefix"),
                f"{VALIDATION_WORKLOAD_ID}\0",
            ),
            (("workloads", "validation", "expected_examples"), 512),
            (("workloads", "validation", "maximum_sequence_tokens"), 1024),
            (("workloads", "heldout", "id"), HELDOUT_WORKLOAD_ID),
            (("workloads", "heldout", "split"), "heldout"),
            (
                ("workloads", "heldout", "selection_hash_prefix"),
                f"{HELDOUT_WORKLOAD_ID}\0",
            ),
            (("workloads", "heldout", "expected_examples"), 64),
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
            (("training", "warmup_steps"), 16),
            (("training", "seed"), 0),
            (("training", "packing"), False),
            (("training", "gradient_checkpointing"), True),
            (("training", "maximum_optimizer_steps"), 512),
            (("training", "checkpoint_interval_steps"), 128),
            (("training", "mandatory_process_stop_step"), 256),
            (("training", "checkpoint_candidates"), [128, 256, 384, 512]),
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
            (("minif2f", "workload_id"), "minif2f-valid-dev16-v1"),
        )
        for path, wanted in expected:
            observed: Any = self.value
            for key in path:
                observed = observed[key]
            if observed != wanted:
                joined = ".".join(path)
                raise ValueError(
                    f"Phase 4 {joined} must be {wanted!r}, got {observed!r}"
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
                "Phase 4 LoRA target modules differ from the accepted order"
            )
        if int(self.value["verification"]["workers"]) < 1:
            raise ValueError("Phase 4 verification workers must be positive")
        for key in ("heldout_timeout_seconds", "minif2f_timeout_seconds"):
            if float(self.value["verification"][key]) <= 0:
                raise ValueError(f"Phase 4 verification {key} must be positive")


@dataclass(frozen=True)
class HeldoutPromptExample:
    record_id: str
    declaration_name: str
    prompt: str
    prompt_tokens: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HeldoutPromptExample:
        return cls(
            record_id=str(value["record_id"]),
            declaration_name=str(value["declaration_name"]),
            prompt=str(value["prompt"]),
            prompt_tokens=int(value["prompt_tokens"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase4Workloads:
    train: tuple[TokenizedSFTExample, ...]
    validation: tuple[TokenizedSFTExample, ...]
    heldout: tuple[HeldoutPromptExample, ...]
    eligible_counts: dict[str, int]
    eos_token_id: int


@dataclass(frozen=True)
class SelectedAdapterBinding:
    selected_optimizer_step: int
    artifact_id: str
    training_relative_path: str
    checkpoint_path: Path
    training_artifact_sha256: str
    format: str
    merged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_optimizer_step": self.selected_optimizer_step,
            "artifact_id": self.artifact_id,
            "training_relative_path": self.training_relative_path,
            "training_artifact_sha256": self.training_artifact_sha256,
            "format": self.format,
            "merged": self.merged,
        }


def load_selected_adapter_binding(
    training_path: Path,
    *,
    expected_artifact_id: str | None = None,
    adapter_dir: Path | None = None,
) -> tuple[dict[str, Any], SelectedAdapterBinding]:
    training_bytes = training_path.read_bytes()
    training = json.loads(training_bytes)
    if training.get("status") != "passed":
        raise ValueError("Phase 4 selected adapter requires passed training")
    selection = training.get("checkpoint_selection") or {}
    selected_step = int(selection.get("selected_optimizer_step", -1))
    if selected_step < 0:
        raise ValueError("Phase 4 training has no validation-selected checkpoint")
    if bool(selection.get("heldout_or_minif2f_consulted", True)):
        raise ValueError("Phase 4 checkpoint selection used post-selection evidence")

    adapter = training.get("adapter") or {}
    artifact_id = str(adapter.get("artifact_id", ""))
    expected_relative_path = f"trainer-state/checkpoint-{selected_step}"
    if adapter.get("relative_path") != expected_relative_path:
        raise ValueError(
            "Phase 4 training adapter path does not match its selected checkpoint"
        )
    if expected_artifact_id is not None and artifact_id != expected_artifact_id:
        raise ValueError("Phase 4 training selected a different adapter identity")
    if adapter.get("format") != "peft-lora" or adapter.get("merged") is not False:
        raise ValueError("Phase 4 training selected adapter is not unmerged PEFT LoRA")

    checkpoint_path = (
        training_path.resolve().parent / expected_relative_path
    ).resolve()
    if adapter_dir is not None and adapter_dir.resolve() != checkpoint_path:
        raise ValueError(
            "adapter directory does not match the training-selected checkpoint path"
        )
    binding = SelectedAdapterBinding(
        selected_optimizer_step=selected_step,
        artifact_id=artifact_id,
        training_relative_path=expected_relative_path,
        checkpoint_path=checkpoint_path,
        training_artifact_sha256=hashlib.sha256(training_bytes).hexdigest(),
        format="peft-lora",
        merged=False,
    )
    return training, binding


def _digest(prefix: str, record_id: str) -> bytes:
    return hashlib.sha256(prefix.encode("utf-8") + record_id.encode("utf-8")).digest()


def select_sft_workload(
    records: Iterable[MathlibProofRecord],
    tokenizer: Any,
    config: Phase4Config,
    workload_name: str,
) -> tuple[list[TokenizedSFTExample], int]:
    if workload_name not in {"train", "validation"}:
        raise ValueError(f"unknown Phase 4 SFT workload: {workload_name}")
    workload = config.workloads[workload_name]
    split = str(workload["split"])
    eligible: list[tuple[bytes, TokenizedSFTExample]] = []
    seen: set[str] = set()
    for record in records:
        if record.split != split:
            raise ValueError(
                f"Phase 4 {workload_name} input contains {record.split} record "
                f"{record.id}"
            )
        if record.id in seen:
            raise ValueError(
                f"duplicate Phase 4 {workload_name} candidate record ID: {record.id}"
            )
        seen.add(record.id)
        example = tokenize_sft_record(record, tokenizer, expected_split=split)
        if len(example.input_ids) > int(workload["maximum_sequence_tokens"]):
            continue
        eligible.append(
            (
                _digest(str(workload["selection_hash_prefix"]), record.id),
                example,
            )
        )

    eligible.sort(key=lambda item: item[0])
    expected_count = int(workload["expected_examples"])
    if len(eligible) < expected_count:
        raise ValueError(
            f"{workload['id']} requires {expected_count} examples; "
            f"only {len(eligible)} are eligible"
        )
    selected = [example for _, example in eligible[:expected_count]]
    eos_token_id = int(tokenizer.eos_token_id)
    for example in selected:
        example.validate(eos_token_id, int(workload["maximum_sequence_tokens"]))
    return selected, len(eligible)


def select_heldout_workload(
    records: Iterable[MathlibProofRecord],
    tokenizer: Any,
    config: Phase4Config,
) -> tuple[list[HeldoutPromptExample], int]:
    workload = config.workloads["heldout"]
    generation_tokens = int(config.value["heldout_generation"]["max_new_tokens"])
    maximum = int(workload["maximum_prompt_and_generation_tokens"])
    eligible: list[tuple[bytes, HeldoutPromptExample]] = []
    seen: set[str] = set()
    for record in records:
        if record.split != "heldout":
            raise ValueError(
                f"Phase 4 heldout input contains {record.split} record {record.id}"
            )
        if record.id in seen:
            raise ValueError(
                f"duplicate Phase 4 heldout candidate record ID: {record.id}"
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
    return [example for _, example in eligible[:expected_count]], len(eligible)


def materialize_phase4_workloads(
    artifact_dir: Path, config: Phase4Config, tokenizer: Any | None = None
) -> Phase4Workloads:
    tokenizer = load_pinned_tokenizer(config) if tokenizer is None else tokenizer
    dataset = load_phase2_dataset(artifact_dir)
    train, train_eligible = select_sft_workload(
        (MathlibProofRecord.from_dict(value) for value in dataset["train"]),
        tokenizer,
        config,
        "train",
    )
    validation, validation_eligible = select_sft_workload(
        (MathlibProofRecord.from_dict(value) for value in dataset["validation"]),
        tokenizer,
        config,
        "validation",
    )
    heldout, heldout_eligible = select_heldout_workload(
        (MathlibProofRecord.from_dict(value) for value in dataset["heldout"]),
        tokenizer,
        config,
    )
    workloads = Phase4Workloads(
        train=tuple(train),
        validation=tuple(validation),
        heldout=tuple(heldout),
        eligible_counts={
            "train": train_eligible,
            "validation": validation_eligible,
            "heldout": heldout_eligible,
        },
        eos_token_id=int(tokenizer.eos_token_id),
    )
    _validate_workload_disjointness(workloads)
    return workloads


def _validate_workload_disjointness(workloads: Phase4Workloads) -> None:
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
        raise ValueError("Phase 4 workloads contain cross-split record leakage")


def write_phase4_workloads(
    output: Path, config: Phase4Config, workloads: Phase4Workloads
) -> dict[str, Any]:
    _validate_workload_disjointness(workloads)
    value = {
        "schema_version": PHASE4_WORKLOAD_SCHEMA_VERSION,
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "serialization_id": config.value["serialization"]["id"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "eos_token_id": workloads.eos_token_id,
        "workloads": {
            "train": {
                "id": config.workloads["train"]["id"],
                "split": "train",
                "eligible_examples": workloads.eligible_counts["train"],
                "selected_record_ids": [
                    example.record_id for example in workloads.train
                ],
                "examples": [example.to_dict() for example in workloads.train],
            },
            "validation": {
                "id": config.workloads["validation"]["id"],
                "split": "validation",
                "eligible_examples": workloads.eligible_counts["validation"],
                "selected_record_ids": [
                    example.record_id for example in workloads.validation
                ],
                "examples": [example.to_dict() for example in workloads.validation],
            },
            "heldout": {
                "id": config.workloads["heldout"]["id"],
                "split": "heldout",
                "eligible_examples": workloads.eligible_counts["heldout"],
                "selected_record_ids": [
                    example.record_id for example in workloads.heldout
                ],
                "examples": [example.to_dict() for example in workloads.heldout],
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def load_phase4_workloads(path: Path, config: Phase4Config) -> Phase4Workloads:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PHASE4_WORKLOAD_SCHEMA_VERSION:
        raise ValueError(
            f"unknown Phase 4 workload schema: {value.get('schema_version')}"
        )
    expected_metadata = {
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "serialization_id": config.value["serialization"]["id"],
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
    }
    for key, expected in expected_metadata.items():
        if value.get(key) != expected:
            raise ValueError(f"Phase 4 workload {key} differs from the configuration")

    loaded: dict[str, Any] = {}
    eligible_counts: dict[str, int] = {}
    eos_token_id = int(value["eos_token_id"])
    for name in ("train", "validation"):
        item = value["workloads"][name]
        expected = config.workloads[name]
        if item.get("id") != expected["id"] or item.get("split") != expected["split"]:
            raise ValueError(f"Phase 4 {name} workload identity differs")
        examples = tuple(TokenizedSFTExample.from_dict(row) for row in item["examples"])
        ids = [example.record_id for example in examples]
        if ids != item.get("selected_record_ids"):
            raise ValueError(f"Phase 4 {name} selected record order differs")
        if len(examples) != int(expected["expected_examples"]):
            raise ValueError(f"Phase 4 {name} workload count differs")
        for example in examples:
            example.validate(eos_token_id, int(expected["maximum_sequence_tokens"]))
        loaded[name] = examples
        eligible_counts[name] = int(item["eligible_examples"])

    heldout_item = value["workloads"]["heldout"]
    heldout_expected = config.workloads["heldout"]
    if (
        heldout_item.get("id") != heldout_expected["id"]
        or heldout_item.get("split") != "heldout"
    ):
        raise ValueError("Phase 4 heldout workload identity differs")
    heldout = tuple(
        HeldoutPromptExample.from_dict(row) for row in heldout_item["examples"]
    )
    heldout_ids = [example.record_id for example in heldout]
    if heldout_ids != heldout_item.get("selected_record_ids"):
        raise ValueError("Phase 4 heldout selected record order differs")
    if len(heldout) != int(heldout_expected["expected_examples"]):
        raise ValueError("Phase 4 heldout workload count differs")
    maximum = int(heldout_expected["maximum_prompt_and_generation_tokens"])
    generation_tokens = int(config.value["heldout_generation"]["max_new_tokens"])
    if any(item.prompt_tokens + generation_tokens > maximum for item in heldout):
        raise ValueError("Phase 4 heldout workload exceeds the model-length boundary")
    eligible_counts["heldout"] = int(heldout_item["eligible_examples"])

    workloads = Phase4Workloads(
        train=loaded["train"],
        validation=loaded["validation"],
        heldout=heldout,
        eligible_counts=eligible_counts,
        eos_token_id=eos_token_id,
    )
    _validate_workload_disjointness(workloads)
    return workloads


def ordered_record_ids(examples: Sequence[Any]) -> list[str]:
    return [str(example.record_id) for example in examples]
