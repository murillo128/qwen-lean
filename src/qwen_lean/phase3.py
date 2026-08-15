from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .phase2_corpus import load_phase2_dataset
from .phase2_schema import MathlibProofRecord
from .prompt import render_proof_request


PHASE3_CONFIG_SCHEMA_VERSION = "phase3-config-v1"
SFT_SERIALIZATION_ID = "mathlib-sft-v1"
OVERFIT_WORKLOAD_ID = "phase3-overfit64-v1"
PHASE3_WORKLOAD_SCHEMA_VERSION = "phase3-workload-v1"
BASE_MODEL_ID = "Qwen/Qwen3-8B-Base"
BASE_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
IGNORE_INDEX = -100


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


@dataclass(frozen=True)
class Phase3Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Phase3Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        config = cls(path=path.resolve(), value=value)
        config.validate()
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def workload(self) -> dict[str, Any]:
        return self.value["workload"]

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
    def selected_record_ids(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.workload["selected_record_ids"])

    def validate(self) -> None:
        if self.value.get("schema_version") != PHASE3_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unknown Phase 3 config schema: {self.value.get('schema_version')}"
            )
        expected: tuple[tuple[str, ...], Any] = (
            (("model", "model_id"), BASE_MODEL_ID),
            (("model", "model_revision"), BASE_REVISION),
            (("model", "tokenizer_id"), BASE_MODEL_ID),
            (("model", "tokenizer_revision"), BASE_REVISION),
            (("model", "add_special_tokens"), False),
            (("model", "chat_template"), None),
            (("dataset", "schema_version"), "mathlib-whole-proof-v1"),
            (("dataset", "split"), "train"),
            (("serialization", "id"), SFT_SERIALIZATION_ID),
            (("serialization", "supervise_prompt"), False),
            (("serialization", "supervise_eos"), True),
            (("workload", "id"), OVERFIT_WORKLOAD_ID),
            (("workload", "selection_hash_prefix"), f"{OVERFIT_WORKLOAD_ID}\0"),
            (("workload", "expected_examples"), 64),
            (("workload", "maximum_sequence_tokens"), 512),
            (("workload", "minimum_completion_tokens"), 8),
            (("workload", "maximum_completion_tokens"), 128),
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
            (("training", "gradient_accumulation_steps"), 4),
            (("training", "maximum_sequence_tokens"), 512),
            (("training", "optimizer"), "paged_adamw_8bit"),
            (("training", "learning_rate"), 0.0002),
            (("training", "weight_decay"), 0.0),
            (("training", "maximum_gradient_norm"), 1.0),
            (("training", "lr_schedule"), "constant"),
            (("training", "warmup_steps"), 0),
            (("training", "seed"), 0),
            (("training", "packing"), False),
            (("training", "gradient_checkpointing"), True),
            (("training", "maximum_optimizer_steps"), 600),
            (("training", "memorization_probe_interval_steps"), 100),
            (("training", "target_cross_entropy_threshold"), 0.2),
            (("training", "target_accuracy_threshold"), 0.97),
            (("memorization_generation", "minimum_exact_matches"), 48),
            (("semantic_verification", "minimum_lean_accepted"), 48),
            (("semantic_verification", "maximum_target_cross_entropy"), 0.05),
            (("semantic_verification", "minimum_target_accuracy"), 0.995),
            (("semantic_verification", "workers"), 8),
            (("semantic_verification", "timeout_seconds"), 300.0),
        )
        for path, wanted in expected:
            observed: Any = self.value
            for key in path:
                observed = observed[key]
            if observed != wanted:
                joined = ".".join(path)
                raise ValueError(
                    f"Phase 3 {joined} must be {wanted!r}, got {observed!r}"
                )

        target_modules = tuple(str(item) for item in self.lora["target_modules"])
        required_modules = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        if target_modules != required_modules:
            raise ValueError(
                "Phase 3 LoRA target modules differ from the accepted order"
            )

        selected = self.selected_record_ids
        if len(selected) != 64 or len(set(selected)) != 64:
            raise ValueError("Phase 3 configuration must pin 64 unique record IDs")
        if any(len(record_id) != 64 for record_id in selected):
            raise ValueError("Phase 3 selected record IDs must be SHA-256 hex digests")


@dataclass(frozen=True)
class TokenizedSFTExample:
    record_id: str
    declaration_name: str
    prompt: str
    completion: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TokenizedSFTExample:
        return cls(
            record_id=str(value["record_id"]),
            declaration_name=str(value["declaration_name"]),
            prompt=str(value["prompt"]),
            completion=str(value["completion"]),
            input_ids=tuple(int(item) for item in value["input_ids"]),
            labels=tuple(int(item) for item in value["labels"]),
            attention_mask=tuple(int(item) for item in value["attention_mask"]),
            prompt_tokens=int(value["prompt_tokens"]),
            completion_tokens=int(value["completion_tokens"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("input_ids", "labels", "attention_mask"):
            value[key] = list(value[key])
        return value

    def validate(self, eos_token_id: int, maximum_sequence_tokens: int) -> None:
        length = len(self.input_ids)
        if not (
            length == len(self.labels) == len(self.attention_mask)
            and length == self.prompt_tokens + self.completion_tokens + 1
        ):
            raise ValueError(f"invalid tokenized lengths for {self.record_id}")
        if length > maximum_sequence_tokens:
            raise ValueError(
                f"record {self.record_id} has {length} tokens; truncation is forbidden"
            )
        if self.input_ids[-1] != eos_token_id or self.labels[-1] != eos_token_id:
            raise ValueError(f"record {self.record_id} has no supervised terminal EOS")
        if self.labels[: self.prompt_tokens] != (IGNORE_INDEX,) * self.prompt_tokens:
            raise ValueError(f"record {self.record_id} supervises prompt tokens")
        target = self.labels[self.prompt_tokens :]
        if any(label == IGNORE_INDEX for label in target):
            raise ValueError(f"record {self.record_id} masks target tokens")
        if self.attention_mask != (1,) * length:
            raise ValueError(f"record {self.record_id} has an invalid unpadded mask")


def render_sft_prompt(record: MathlibProofRecord) -> str:
    """Render the exact mathlib-sft-v1 prefix without source-file context."""
    return render_proof_request(record.declaration)


def tokenize_sft_record(
    record: MathlibProofRecord,
    tokenizer: Tokenizer,
    *,
    maximum_sequence_tokens: int | None = None,
    expected_split: str = "train",
) -> TokenizedSFTExample:
    if record.split != expected_split:
        raise ValueError(
            f"cannot tokenize {record.split} record {record.id} as {expected_split}"
        )
    if tokenizer.eos_token_id is None:
        raise ValueError("the pinned tokenizer has no EOS token")
    prompt = render_sft_prompt(record)
    prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
    completion_ids = tuple(
        tokenizer.encode(record.completion, add_special_tokens=False)
    )
    input_ids = prompt_ids + completion_ids + (int(tokenizer.eos_token_id),)
    labels = (
        (IGNORE_INDEX,) * len(prompt_ids)
        + completion_ids
        + (int(tokenizer.eos_token_id),)
    )
    example = TokenizedSFTExample(
        record_id=record.id,
        declaration_name=record.declaration_name,
        prompt=prompt,
        completion=record.completion,
        input_ids=input_ids,
        labels=labels,
        attention_mask=(1,) * len(input_ids),
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(completion_ids),
    )
    if maximum_sequence_tokens is not None:
        example.validate(int(tokenizer.eos_token_id), maximum_sequence_tokens)
    return example


def select_overfit_workload(
    records: Iterable[MathlibProofRecord],
    tokenizer: Tokenizer,
    config: Phase3Config,
) -> tuple[list[TokenizedSFTExample], int]:
    workload = config.workload
    eligible: list[tuple[bytes, TokenizedSFTExample]] = []
    seen: set[str] = set()
    for record in records:
        if record.split != "train":
            raise ValueError(f"Phase 3 input contains non-train record {record.id}")
        if record.id in seen:
            raise ValueError(f"duplicate Phase 3 candidate record ID: {record.id}")
        seen.add(record.id)
        example = tokenize_sft_record(record, tokenizer)
        if len(example.input_ids) > int(workload["maximum_sequence_tokens"]):
            continue
        if not (
            int(workload["minimum_completion_tokens"])
            <= example.completion_tokens
            <= int(workload["maximum_completion_tokens"])
        ):
            continue
        digest = hashlib.sha256(
            str(workload["selection_hash_prefix"]).encode("utf-8")
            + record.id.encode("utf-8")
        ).digest()
        eligible.append((digest, example))

    eligible.sort(key=lambda item: item[0])
    expected_count = int(workload["expected_examples"])
    if len(eligible) < expected_count:
        raise ValueError(
            f"{OVERFIT_WORKLOAD_ID} requires {expected_count} examples; "
            f"only {len(eligible)} are eligible"
        )
    selected = [example for _, example in eligible[:expected_count]]
    actual_ids = tuple(example.record_id for example in selected)
    if actual_ids != config.selected_record_ids:
        raise ValueError(
            "deterministic Phase 3 record IDs differ from the configuration"
        )
    eos_token_id = int(tokenizer.eos_token_id)  # checked by tokenize_sft_record
    for example in selected:
        example.validate(eos_token_id, int(workload["maximum_sequence_tokens"]))
    return selected, len(eligible)


def load_phase2_train_records(artifact_dir: Path) -> Iterable[MathlibProofRecord]:
    dataset = load_phase2_dataset(artifact_dir)
    for value in dataset["train"]:
        yield MathlibProofRecord.from_dict(value)


def load_pinned_tokenizer(config: Phase3Config) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Phase 3 requires the training optional dependencies"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model["tokenizer_id"]),
        revision=str(config.model["tokenizer_revision"]),
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("the pinned tokenizer has no EOS token")
    return tokenizer


def write_phase3_workload(
    output: Path,
    config: Phase3Config,
    examples: Sequence[TokenizedSFTExample],
    *,
    eligible_examples: int,
    eos_token_id: int,
) -> dict[str, Any]:
    value = {
        "schema_version": PHASE3_WORKLOAD_SCHEMA_VERSION,
        "workload_id": OVERFIT_WORKLOAD_ID,
        "serialization_id": SFT_SERIALIZATION_ID,
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "dataset_split": "train",
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
        "eos_token_id": eos_token_id,
        "eligible_examples": eligible_examples,
        "selected_record_ids": [example.record_id for example in examples],
        "examples": [example.to_dict() for example in examples],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def load_phase3_workload(
    path: Path, config: Phase3Config
) -> tuple[list[TokenizedSFTExample], int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PHASE3_WORKLOAD_SCHEMA_VERSION:
        raise ValueError(
            f"unknown Phase 3 workload schema: {value.get('schema_version')}"
        )
    expected_metadata = {
        "workload_id": OVERFIT_WORKLOAD_ID,
        "serialization_id": SFT_SERIALIZATION_ID,
        "dataset_schema_version": config.value["dataset"]["schema_version"],
        "dataset_split": "train",
        "tokenizer_id": config.model["tokenizer_id"],
        "tokenizer_revision": config.model["tokenizer_revision"],
    }
    for key, expected in expected_metadata.items():
        if value.get(key) != expected:
            raise ValueError(f"Phase 3 workload {key} differs from the configuration")
    examples = [TokenizedSFTExample.from_dict(item) for item in value["examples"]]
    ids = tuple(example.record_id for example in examples)
    if ids != config.selected_record_ids or tuple(value["selected_record_ids"]) != ids:
        raise ValueError("Phase 3 workload record IDs differ from the configuration")
    eos_token_id = int(value["eos_token_id"])
    for example in examples:
        example.validate(eos_token_id, int(config.workload["maximum_sequence_tokens"]))
    return examples, eos_token_id


def pad_target_only_batch(
    features: Sequence[Mapping[str, Sequence[int]]], *, pad_token_id: int
) -> dict[str, list[list[int]]]:
    if not features:
        raise ValueError("cannot collate an empty Phase 3 batch")
    maximum = max(len(feature["input_ids"]) for feature in features)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for feature in features:
        input_ids = [int(item) for item in feature["input_ids"]]
        labels = [int(item) for item in feature["labels"]]
        attention_mask = [int(item) for item in feature["attention_mask"]]
        if not (len(input_ids) == len(labels) == len(attention_mask)):
            raise ValueError("Phase 3 batch feature lengths differ")
        padding = maximum - len(input_ids)
        batch["input_ids"].append(input_ids + [pad_token_id] * padding)
        batch["labels"].append(labels + [IGNORE_INDEX] * padding)
        batch["attention_mask"].append(attention_mask + [0] * padding)
    return batch


class TargetOnlyDataCollator:
    """Pad pre-tokenized examples without regenerating their target-only labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self, features: Sequence[Mapping[str, Sequence[int]]]
    ) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Phase 3 training requires PyTorch") from error
        batch = pad_target_only_batch(features, pad_token_id=self.pad_token_id)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


def trainer_dataset_rows(
    examples: Sequence[TokenizedSFTExample],
) -> list[dict[str, list[int]]]:
    return [
        {
            "input_ids": list(example.input_ids),
            "labels": list(example.labels),
            "attention_mask": list(example.attention_mask),
        }
        for example in examples
    ]
