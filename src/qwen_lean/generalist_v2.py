from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from .metrics import pass_at_k
from .phase3 import IGNORE_INDEX
from .phase6 import paired_task_bootstrap
from .prompt import normalize_transport, render_proof_request

GENERALIST_CONFIG_SCHEMA_VERSION = "generalist-v2-config-v1"
GENERALIST_SERIALIZATION_ID = "lean-sft-v2"
GENERALIST_ARTIFACT_ID = "qwen-lean-generalist-v2"
MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
CONTEXT_CHOICES = (4096, 8192, 16384, 32768)
CHECKPOINT_IDS = ("Q0", "Q1", "Q2", "Q3", "Q4")
TRAINED_CHECKPOINT_IDS = CHECKPOINT_IDS[1:]
COMPOSITION_CLASSES = ("direct", "branching", "deep")
LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LORA_TARGET_REGEX = (
    r"^model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
EXPECTED_LORA_MODULE_COUNTS = {
    "q_proj": 8,
    "k_proj": 8,
    "v_proj": 8,
    "o_proj": 8,
    "in_proj_qkv": 24,
    "in_proj_z": 24,
    "in_proj_a": 24,
    "in_proj_b": 24,
    "out_proj": 24,
    "gate_proj": 32,
    "up_proj": 32,
    "down_proj": 32,
}
RIEMANN_DOMAIN_TAGS = frozenset(
    {
        "zeta",
        "analytic-number-theory",
        "riemann-core",
        "riemann-bubble",
        "prime-counting",
        "pnt",
        "pnt-plus",
        "arithmetic-functions",
        "prime-arithmetic",
        "divisibility",
    }
)


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


@dataclass(frozen=True)
class GeneralistV2Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> GeneralistV2Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        config = cls(path=path.resolve(), value=value)
        config.validate()
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.value["dataset"]

    @property
    def serialization(self) -> dict[str, Any]:
        return self.value["serialization"]

    @property
    def weighting(self) -> dict[str, Any]:
        return self.value["weighting"]

    @property
    def precision(self) -> dict[str, Any]:
        return self.value["precision"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.value["lora"]

    @property
    def training(self) -> dict[str, Any]:
        return self.value["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.value["evaluation"]

    def validate(self) -> None:
        if self.value.get("schema_version") != GENERALIST_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "unknown generalist-v2 config schema: "
                f"{self.value.get('schema_version')}"
            )
        expected: tuple[tuple[str, ...], Any] = (
            (("artifact_id",), GENERALIST_ARTIFACT_ID),
            (("model", "model_id"), MODEL_ID),
            (("model", "model_revision"), MODEL_REVISION),
            (("model", "tokenizer_id"), MODEL_ID),
            (("model", "tokenizer_revision"), MODEL_REVISION),
            (("model", "text_only"), True),
            (("model", "architecture_class"), "Qwen3_5ForCausalLM"),
            (("model", "add_special_tokens"), False),
            (("model", "chat_template"), None),
            (("dataset", "package_id"), "lean-whole-proof-v2"),
            (("dataset", "training_membership"), "general-train-v2"),
            (
                ("dataset", "validation_memberships", "fresh_composition"),
                "fresh-composition-valid-v2",
            ),
            (
                ("dataset", "validation_memberships", "minif2f"),
                "minif2f-valid-clean-v2",
            ),
            (("dataset", "training_probe"), "dataset-v2-train-probe"),
            (("serialization", "id"), GENERALIST_SERIALIZATION_ID),
            (("serialization", "supervise_prompt"), False),
            (("serialization", "supervise_eos"), True),
            (("serialization", "terminal_eos_count"), 1),
            (("serialization", "packing"), False),
            (("serialization", "truncation"), False),
            (("weighting", "statement_normalized_variants"), True),
            (("weighting", "synthetic_target_mass_fraction"), 0.1),
            (("weighting", "synthetic_max_statement_multiplier"), 4.0),
            (("weighting", "resolved_synthetic_statement_multiplier"), 4.0),
            (
                ("weighting", "resolved_synthetic_mass_fraction"),
                0.06845878885428207,
            ),
            (("weighting", "domain_multipliers"), {}),
            (("precision", "preferred", "lane"), "bf16-lora"),
            (("precision", "preferred", "minimum_vram_bytes"), 48 * 1024**3),
            (("precision", "preferred", "base_dtype"), "bfloat16"),
            (("precision", "fallback", "lane"), "nf4-qlora"),
            (("precision", "fallback", "load_in_4bit"), True),
            (("precision", "fallback", "quantization_type"), "nf4"),
            (("precision", "fallback", "double_quantization"), True),
            (("precision", "fallback", "compute_dtype"), "bfloat16"),
            (("lora", "target_regex"), LORA_TARGET_REGEX),
            (("lora", "r"), 16),
            (("lora", "lora_alpha"), 32),
            (("lora", "lora_dropout"), 0.0),
            (("lora", "bias"), "none"),
            (("lora", "modules_to_save"), None),
            (("training", "trainer"), "trl-sft-weighted"),
            (("training", "per_device_micro_batch_size"), 1),
            (("training", "gradient_accumulation_steps"), 8),
            (("training", "optimizer"), "paged_adamw_8bit"),
            (("training", "learning_rate"), 0.0001),
            (("training", "weight_decay"), 0.0),
            (("training", "maximum_gradient_norm"), 1.0),
            (("training", "lr_schedule"), "cosine"),
            (("training", "warmup_fraction"), 1 / 32),
            (("training", "seed"), 0),
            (("training", "data_seed"), 0),
            (("training", "gradient_checkpointing"), True),
            (("training", "packing"), False),
            (("training", "truncation"), False),
            (("training", "complete_passes"), 1),
            (("training", "resolved_context_tokens"), 32768),
            (("evaluation", "candidates_per_task"), 8),
            (("evaluation", "primary_metric"), "pass@8"),
            (("evaluation", "bootstrap_resamples"), 10000),
            (("evaluation", "bootstrap_seed"), 0),
            (("evaluation", "sampling", "do_sample"), True),
            (("evaluation", "sampling", "temperature"), 0.8),
            (("evaluation", "sampling", "top_p"), 0.95),
            (("evaluation", "sampling", "top_k"), -1),
            (("evaluation", "sampling", "max_new_tokens"), 1024),
            (
                ("evaluation", "sampling", "stop"),
                "tokenizer_eos_or_token_limit",
            ),
            (("evaluation", "sampling", "seed"), 0),
            (("evaluation", "verifier_timeout_seconds"), 30.0),
        )
        for path, wanted in expected:
            observed: Any = self.value
            for key in path:
                observed = observed[key]
            if observed != wanted:
                joined = ".".join(path)
                raise ValueError(
                    f"generalist-v2 {joined} must be {wanted!r}, got {observed!r}"
                )

        observed_suffixes = tuple(self.lora["target_suffixes"])
        if observed_suffixes != LORA_TARGET_SUFFIXES:
            raise ValueError("generalist-v2 LoRA target suffixes differ")
        if dict(self.lora["expected_module_counts"]) != EXPECTED_LORA_MODULE_COUNTS:
            raise ValueError("generalist-v2 expected LoRA module counts differ")
        if tuple(self.training["context_choices"]) != CONTEXT_CHOICES:
            raise ValueError("generalist-v2 context choices differ")
        binding = self.dataset.get("binding", {})
        expected_binding = {
            "manifest_sha256": (
                "c4e5586470f41fe403fc04557548bcea4498dca88e9ca00e544a64d52414ea5e"
            ),
            "canonical_records_sha256": (
                "a66855d8fa9e5132ea895fa206481e9a38cb8cc1baa7494a2a1f8f910030442c"
            ),
            "general_train_sha256": (
                "c0dbbf5f6e7e95e4acdc39219140e2f3c418c2ed131044b526a40e096b081367"
            ),
            "training_statements": 181531,
            "training_proof_variants": 182812,
            "fresh_composition_valid_statements": 406,
            "fresh_composition_test_statements": 415,
        }
        if binding != expected_binding:
            raise ValueError("generalist-v2 Dataset-v2 binding differs")
        if (
            self.value.get("riemann", {}).get("checkpoint_selection_role")
            != "diagnostic-only"
        ):
            raise ValueError("Riemann validation must remain diagnostic-only")
        if self.value["riemann"].get("historical_candidates_per_task") != 4:
            raise ValueError(
                "historical Riemann comparability must retain 4 candidates"
            )
        if self.value["riemann"].get("fresh_candidates_per_task") != 8:
            raise ValueError("fresh Riemann evaluation must use 8 candidates")


@dataclass(frozen=True)
class GeneralistProofVariant:
    statement_id: str
    proof_variant_id: str
    declaration_name: str
    declaration: str
    completion: str
    preamble: str
    split: str
    optimizer_eligible: bool
    source_kind: str
    generator_family: str | None = None
    composition_class: str | None = None
    derivation_family_id: str | None = None
    domain_tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GeneralistProofVariant:
        return cls(
            statement_id=str(value["statement_id"]),
            proof_variant_id=str(value["proof_variant_id"]),
            declaration_name=str(value["declaration_name"]),
            declaration=str(value["declaration"]),
            completion=str(value["completion"]),
            preamble=str(value["preamble"]),
            split=str(value["split"]),
            optimizer_eligible=bool(value["optimizer_eligible"]),
            source_kind=str(value["source_kind"]),
            generator_family=(
                None
                if value.get("generator_family") is None
                else str(value["generator_family"])
            ),
            composition_class=(
                None
                if value.get("composition_class") is None
                else str(value["composition_class"])
            ),
            derivation_family_id=(
                None
                if value.get("derivation_family_id") is None
                else str(value["derivation_family_id"])
            ),
            domain_tags=tuple(str(item) for item in value.get("domain_tags", ())),
        )

    def validate(self) -> None:
        required_text = {
            "statement_id": self.statement_id,
            "proof_variant_id": self.proof_variant_id,
            "declaration_name": self.declaration_name,
            "declaration": self.declaration,
            "completion": self.completion,
            "preamble": self.preamble,
        }
        empty = [name for name, value in required_text.items() if not value.strip()]
        if empty:
            raise ValueError(f"generalist-v2 record has empty fields: {empty}")
        if self.source_kind not in {"real", "synthetic"}:
            raise ValueError(f"unknown source kind: {self.source_kind}")
        if self.source_kind == "synthetic":
            if not self.generator_family or not self.derivation_family_id:
                raise ValueError(
                    "synthetic records need generator and derivation families"
                )
            if self.composition_class not in COMPOSITION_CLASSES:
                raise ValueError("synthetic record has an unknown composition class")
        active_code = _lean_code_without_comments_and_strings(
            normalize_transport(self.completion)
        )
        if re.search(r"\b(?:sorry|admit)\b", active_code):
            raise ValueError("generalist-v2 completion contains a placeholder")


def _lean_code_without_comments_and_strings(value: str) -> str:
    """Retain active Lean code while masking comments and string contents."""

    output: list[str] = []
    index = 0
    block_depth = 0
    line_comment = False
    in_string = False
    escaped = False
    while index < len(value):
        pair = value[index : index + 2]
        character = value[index]
        if line_comment:
            if character == "\n":
                line_comment = False
                output.append(character)
            else:
                output.append(" ")
            index += 1
            continue
        if block_depth:
            if pair == "/-":
                block_depth += 1
                output.extend("  ")
                index += 2
            elif pair == "-/":
                block_depth -= 1
                output.extend("  ")
                index += 2
            else:
                output.append("\n" if character == "\n" else " ")
                index += 1
            continue
        if in_string:
            output.append("\n" if character == "\n" else " ")
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if pair == "--":
            line_comment = True
            output.extend("  ")
            index += 2
        elif pair == "/-":
            block_depth = 1
            output.extend("  ")
            index += 2
        elif character == '"':
            in_string = True
            output.append(" ")
            index += 1
        else:
            output.append(character)
            index += 1
    if block_depth or in_string:
        raise ValueError("generalist-v2 completion has unterminated Lean syntax")
    return "".join(output)


@dataclass(frozen=True)
class TrainingWeights:
    statement_weights: dict[str, float]
    variant_weights: dict[str, float]
    variants_per_statement: dict[str, int]
    real_statement_count: int
    synthetic_statement_count: int
    synthetic_base_multiplier: float
    real_mass: float
    synthetic_mass: float
    synthetic_mass_fraction: float
    maximum_statement_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeightedTokenizedExample:
    statement_id: str
    proof_variant_id: str
    declaration_name: str
    prompt: str
    completion: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    example_weight: float

    def validate(
        self, eos_token_id: int, maximum_sequence_tokens: int | None = None
    ) -> None:
        length = len(self.input_ids)
        if not (
            length == len(self.labels) == len(self.attention_mask)
            and length == self.prompt_tokens + self.completion_tokens + 1
        ):
            raise ValueError(f"invalid tokenized lengths for {self.proof_variant_id}")
        if maximum_sequence_tokens is not None and length > maximum_sequence_tokens:
            raise ValueError(
                f"proof variant {self.proof_variant_id} has {length} tokens; "
                "truncation is forbidden"
            )
        if self.input_ids[-1] != eos_token_id or self.labels[-1] != eos_token_id:
            raise ValueError("generalist-v2 example lacks its supervised terminal EOS")
        if eos_token_id in self.input_ids[:-1]:
            raise ValueError("generalist-v2 example contains a non-terminal EOS")
        if self.labels[: self.prompt_tokens] != (IGNORE_INDEX,) * self.prompt_tokens:
            raise ValueError("generalist-v2 example supervises prompt tokens")
        if any(label == IGNORE_INDEX for label in self.labels[self.prompt_tokens :]):
            raise ValueError("generalist-v2 example masks completion or EOS tokens")
        if self.attention_mask != (1,) * length:
            raise ValueError("generalist-v2 unpadded attention mask is invalid")
        if not math.isfinite(self.example_weight) or self.example_weight <= 0:
            raise ValueError("generalist-v2 example weight must be finite and positive")

    def to_trainer_row(self) -> dict[str, Any]:
        return {
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "attention_mask": list(self.attention_mask),
            "example_weight": self.example_weight,
        }


@dataclass(frozen=True)
class CheckpointValidation:
    fresh_composition_verified_counts: tuple[int, ...]
    minif2f_verified_counts: tuple[int, ...]


def _training_records(
    records: Iterable[GeneralistProofVariant],
    *,
    forbidden_proof_variant_ids: Iterable[str] = (),
) -> list[GeneralistProofVariant]:
    materialized = list(records)
    if not materialized:
        raise ValueError("generalist-v2 training membership is empty")
    forbidden = set(forbidden_proof_variant_ids)
    seen_variants: set[str] = set()
    statements: dict[str, GeneralistProofVariant] = {}
    for record in materialized:
        record.validate()
        if record.split != "train" or not record.optimizer_eligible:
            raise ValueError(
                f"non-training proof variant reached optimizer membership: "
                f"{record.proof_variant_id}"
            )
        if record.proof_variant_id in forbidden:
            raise ValueError(
                f"validation/test proof variant is optimizer-visible: "
                f"{record.proof_variant_id}"
            )
        if record.proof_variant_id in seen_variants:
            raise ValueError(f"duplicate proof variant: {record.proof_variant_id}")
        seen_variants.add(record.proof_variant_id)
        previous = statements.get(record.statement_id)
        if previous is not None and (
            previous.source_kind != record.source_kind
            or previous.declaration != record.declaration
            or previous.preamble != record.preamble
            or previous.generator_family != record.generator_family
            or previous.composition_class != record.composition_class
            or previous.derivation_family_id != record.derivation_family_id
        ):
            raise ValueError(
                f"inconsistent metadata across variants for {record.statement_id}"
            )
        statements[record.statement_id] = record
    return materialized


def _capped_proportional_weights(
    raw: Mapping[str, float], *, total: float, cap: float
) -> dict[str, float]:
    if not raw or total <= 0 or cap <= 0:
        raise ValueError("balanced synthetic weights require positive inputs")
    remaining = set(raw)
    result: dict[str, float] = {}
    remaining_total = total
    while remaining:
        raw_total = sum(raw[key] for key in remaining)
        if raw_total <= 0:
            raise ValueError("synthetic balance contains no positive mass")
        proposed = {key: remaining_total * raw[key] / raw_total for key in remaining}
        capped = {key for key, value in proposed.items() if value > cap}
        if not capped:
            result.update(proposed)
            break
        for key in sorted(capped):
            result[key] = cap
            remaining.remove(key)
            remaining_total -= cap
        if remaining_total < -1e-12:
            raise ValueError("synthetic balance cap cannot satisfy target mass")
    if not math.isclose(sum(result.values()), total, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("synthetic statement weights lost target mass")
    return result


def compute_training_weights(
    records: Iterable[GeneralistProofVariant],
    *,
    target_synthetic_fraction: float = 0.1,
    maximum_synthetic_statement_weight: float = 4.0,
) -> TrainingWeights:
    materialized = _training_records(records)
    if target_synthetic_fraction != 0.1:
        raise ValueError("generalist-v2 synthetic target fraction must remain 0.1")
    if maximum_synthetic_statement_weight != 4.0:
        raise ValueError("generalist-v2 synthetic statement cap must remain 4.0")

    by_statement: dict[str, list[GeneralistProofVariant]] = defaultdict(list)
    for record in materialized:
        by_statement[record.statement_id].append(record)
    real_ids = sorted(
        statement_id
        for statement_id, variants in by_statement.items()
        if variants[0].source_kind == "real"
    )
    synthetic_ids = sorted(set(by_statement) - set(real_ids))
    if not real_ids or not synthetic_ids:
        raise ValueError(
            "final generalist-v2 weighting needs both real and synthetic statements"
        )

    real_count = len(real_ids)
    synthetic_count = len(synthetic_ids)
    target_odds = target_synthetic_fraction / (1 - target_synthetic_fraction)
    uncapped_multiplier = real_count * target_odds / synthetic_count
    base_multiplier = min(maximum_synthetic_statement_weight, uncapped_multiplier)
    synthetic_target_mass = synthetic_count * base_multiplier

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    families: dict[str, set[str]] = defaultdict(set)
    for statement_id in synthetic_ids:
        record = by_statement[statement_id][0]
        assert record.generator_family is not None
        assert record.composition_class is not None
        key = (record.generator_family, record.composition_class)
        strata[key].append(statement_id)
        families[record.generator_family].add(record.composition_class)

    raw: dict[str, float] = {}
    family_count = len(families)
    for (family, composition_class), statement_ids in sorted(strata.items()):
        stratum_mass = 1 / family_count / len(families[family])
        per_statement = stratum_mass / len(statement_ids)
        for statement_id in statement_ids:
            raw[statement_id] = per_statement
    synthetic_weights = _capped_proportional_weights(
        raw,
        total=synthetic_target_mass,
        cap=maximum_synthetic_statement_weight,
    )
    statement_weights = {statement_id: 1.0 for statement_id in real_ids}
    statement_weights.update(synthetic_weights)

    variants_per_statement = {
        statement_id: len(variants)
        for statement_id, variants in sorted(by_statement.items())
    }
    variant_weights = {
        record.proof_variant_id: (
            statement_weights[record.statement_id]
            / variants_per_statement[record.statement_id]
        )
        for record in materialized
    }
    for statement_id, variants in by_statement.items():
        aggregate = sum(variant_weights[item.proof_variant_id] for item in variants)
        if not math.isclose(
            aggregate,
            statement_weights[statement_id],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("proof-variant weights do not normalize by statement")
    real_mass = float(real_count)
    synthetic_mass = sum(synthetic_weights.values())
    return TrainingWeights(
        statement_weights=statement_weights,
        variant_weights=variant_weights,
        variants_per_statement=variants_per_statement,
        real_statement_count=real_count,
        synthetic_statement_count=synthetic_count,
        synthetic_base_multiplier=base_multiplier,
        real_mass=real_mass,
        synthetic_mass=synthetic_mass,
        synthetic_mass_fraction=synthetic_mass / (real_mass + synthetic_mass),
        maximum_statement_weight=max(statement_weights.values()),
    )


def render_generalist_prompt(record: GeneralistProofVariant) -> str:
    record.validate()
    return f"{record.preamble.rstrip()}\n\n{render_proof_request(record.declaration.rstrip())}"


def tokenize_generalist_variant(
    record: GeneralistProofVariant,
    tokenizer: Tokenizer,
    *,
    example_weight: float,
    maximum_sequence_tokens: int | None = None,
) -> WeightedTokenizedExample:
    record.validate()
    if tokenizer.eos_token_id is None:
        raise ValueError("the pinned Qwen3.5 tokenizer has no EOS token")
    prompt = render_generalist_prompt(record)
    completion = normalize_transport(record.completion)
    prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
    completion_ids = tuple(tokenizer.encode(completion, add_special_tokens=False))
    eos_token_id = int(tokenizer.eos_token_id)
    if eos_token_id in (*prompt_ids, *completion_ids):
        raise ValueError("serialized prompt/completion already contains EOS")
    input_ids = prompt_ids + completion_ids + (eos_token_id,)
    labels = (IGNORE_INDEX,) * len(prompt_ids) + completion_ids + (eos_token_id,)
    example = WeightedTokenizedExample(
        statement_id=record.statement_id,
        proof_variant_id=record.proof_variant_id,
        declaration_name=record.declaration_name,
        prompt=prompt,
        completion=completion,
        input_ids=input_ids,
        labels=labels,
        attention_mask=(1,) * len(input_ids),
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(completion_ids),
        example_weight=float(example_weight),
    )
    example.validate(eos_token_id, maximum_sequence_tokens)
    return example


def build_weighted_training_examples(
    records: Iterable[GeneralistProofVariant],
    tokenizer: Tokenizer,
    *,
    forbidden_proof_variant_ids: Iterable[str] = (),
) -> tuple[list[WeightedTokenizedExample], TrainingWeights]:
    materialized = _training_records(
        records,
        forbidden_proof_variant_ids=forbidden_proof_variant_ids,
    )
    weights = compute_training_weights(materialized)
    examples = [
        tokenize_generalist_variant(
            record,
            tokenizer,
            example_weight=weights.variant_weights[record.proof_variant_id],
        )
        for record in materialized
    ]
    if Counter(item.proof_variant_id for item in examples) != Counter(
        item.proof_variant_id for item in materialized
    ):
        raise RuntimeError("generalist-v2 membership changed during serialization")
    return examples, weights


def normalized_example_loss_scales(weights: Sequence[float]) -> tuple[float, ...]:
    if not weights or any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("loss scales require finite positive example weights")
    normalizer = fmean(weights)
    scales = tuple(value / normalizer for value in weights)
    if not math.isclose(sum(scales), len(scales), rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("normalized example scales changed aggregate loss mass")
    return scales


def _percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def length_distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values or any(value < 1 for value in values):
        raise ValueError("length distribution needs positive token counts")
    return {
        "count": len(values),
        "minimum": min(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "maximum": max(values),
    }


def select_context_length(
    full_sequence_lengths: Sequence[int],
    *,
    choices: Sequence[int] = CONTEXT_CHOICES,
) -> dict[str, Any]:
    distribution = length_distribution(full_sequence_lengths)
    maximum = int(distribution["maximum"])
    selected = next((value for value in choices if maximum <= value), None)
    if selected is None:
        raise ValueError(
            f"optimizer-visible variant has {maximum} tokens; maximum supported is "
            f"{max(choices)}"
        )
    return {
        "rule": "smallest supported context that consumes every optimizer-visible variant",
        "choices": list(choices),
        "selected_context_tokens": selected,
        "full_sequence_tokens": distribution,
        "truncated_or_dropped_variants": 0,
    }


def serialization_length_evidence(
    examples: Sequence[WeightedTokenizedExample],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot summarize an empty generalist-v2 workload")
    context = select_context_length([len(item.input_ids) for item in examples])
    context["completion_tokens_excluding_eos"] = length_distribution(
        [item.completion_tokens for item in examples]
    )
    return context


def deterministic_training_order(
    records: Iterable[GeneralistProofVariant], *, seed: int = 0
) -> list[GeneralistProofVariant]:
    materialized = _training_records(records)
    if seed != 0:
        raise ValueError("generalist-v2 data seed must remain 0")
    return sorted(
        materialized,
        key=lambda record: hashlib.sha256(
            (
                f"generalist-v2-one-pass-v1\0{seed}\0{record.statement_id}\0"
                f"{record.proof_variant_id}"
            ).encode()
        ).digest(),
    )


def one_pass_trajectory(
    records: Iterable[GeneralistProofVariant], *, effective_batch_size: int = 8
) -> dict[str, Any]:
    ordered = deterministic_training_order(records)
    return one_pass_membership_trajectory(
        [(record.statement_id, record.proof_variant_id) for record in ordered],
        effective_batch_size=effective_batch_size,
        membership_is_ordered=True,
    )


def one_pass_membership_trajectory(
    membership: Sequence[tuple[str, str]],
    *,
    effective_batch_size: int = 8,
    membership_is_ordered: bool = False,
) -> dict[str, Any]:
    if effective_batch_size != 8:
        raise ValueError("generalist-v2 effective batch size must remain 8")
    if not membership:
        raise ValueError("generalist-v2 one-pass membership is empty")
    if any(
        not statement_id or not proof_variant_id
        for statement_id, proof_variant_id in membership
    ):
        raise ValueError("generalist-v2 one-pass membership has an empty identity")
    proof_variant_ids = [proof_variant_id for _, proof_variant_id in membership]
    if len(set(proof_variant_ids)) != len(proof_variant_ids):
        raise ValueError("generalist-v2 one-pass membership duplicates a proof variant")
    ordered_membership = list(membership)
    if not membership_is_ordered:
        ordered_membership.sort(
            key=lambda item: hashlib.sha256(
                (f"generalist-v2-one-pass-v1\0{0}\0{item[0]}\0{item[1]}").encode()
            ).digest()
        )
    row_count = len(ordered_membership)
    optimizer_steps = math.ceil(row_count / effective_batch_size)
    if optimizer_steps < 4:
        raise ValueError("Q1-Q4 checkpointing needs at least four optimizer updates")
    checkpoints = {
        f"Q{quarter}": math.ceil(optimizer_steps * quarter / 4)
        for quarter in range(1, 5)
    }
    if len(set(checkpoints.values())) != 4:
        raise ValueError("Q1-Q4 optimizer boundaries are not distinct")
    serialized_membership = [list(item) for item in ordered_membership]
    digest = hashlib.sha256(
        json.dumps(serialized_membership, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "rule": "one complete deterministic pass over every optimizer-visible proof variant",
        "seed": 0,
        "optimizer_visible_variants": row_count,
        "unique_optimizer_visible_variants": len(
            {proof_variant_id for _, proof_variant_id in ordered_membership}
        ),
        "effective_batch_size": effective_batch_size,
        "optimizer_steps": optimizer_steps,
        "final_optimizer_update_rows": row_count
        - (optimizer_steps - 1) * effective_batch_size,
        "duplicate_final_batch_fill": False,
        "checkpoint_optimizer_steps": checkpoints,
        "ordered_membership_sha256": digest,
    }


def validation_evaluation_plan(config: GeneralistV2Config) -> dict[str, Any]:
    config.validate()
    fresh = config.dataset["validation_memberships"]["fresh_composition"]
    minif2f = config.dataset["validation_memberships"]["minif2f"]
    train_probe = config.dataset["training_probe"]
    riemann = config.value["riemann"]["validation_view"]
    checkpoints: dict[str, Any] = {}
    for checkpoint_id in CHECKPOINT_IDS:
        workloads = [fresh, minif2f, riemann]
        if checkpoint_id in {"Q0", "Q2", "Q4"}:
            workloads.append(train_probe)
        checkpoints[checkpoint_id] = {
            "workloads": workloads,
            "candidates_per_task": config.evaluation["candidates_per_task"],
            "checkpoint_selection_inputs": [fresh, minif2f],
            "diagnostic_only": [
                item for item in workloads if item not in {fresh, minif2f}
            ],
        }
    return {
        "stage": "preselection-validation",
        "checkpoints": checkpoints,
        "test_workloads_consulted": False,
        "riemann_used_for_selection": False,
    }


def final_evaluation_plan(
    config: GeneralistV2Config, *, selected_checkpoint: str
) -> dict[str, Any]:
    config.validate()
    if selected_checkpoint not in TRAINED_CHECKPOINT_IDS:
        raise ValueError("final evaluation needs a frozen Q1-Q4 checkpoint")
    generic = [
        config.dataset["validation_memberships"]["minif2f"],
        config.dataset["test_memberships"]["minif2f"],
        config.dataset["validation_memberships"]["fresh_composition"],
        config.dataset["test_memberships"]["fresh_composition"],
    ]
    models = ["Q0", selected_checkpoint, "deepseek-ai/DeepSeek-Prover-V2-7B"]
    return {
        "stage": "postselection-final",
        "selected_checkpoint_frozen": selected_checkpoint,
        "models": models,
        "generic_workloads": generic,
        "fresh_riemann_workloads": [
            config.value["riemann"]["validation_view"],
            config.value["riemann"]["test_view"],
        ],
        "historical_riemann": {
            "workload": config.value["riemann"]["historical_workload"],
            "clean_generalization": False,
            "evaluate_checkpoint": selected_checkpoint,
            "candidates_per_task": config.value["riemann"][
                "historical_candidates_per_task"
            ],
        },
        "generic_candidates_per_task": config.evaluation["candidates_per_task"],
        "fresh_riemann_candidates_per_task": config.value["riemann"][
            "fresh_candidates_per_task"
        ],
        "test_workloads_consulted_after_checkpoint_freeze": True,
    }


def _validate_verified_counts(counts: Sequence[int], *, name: str) -> None:
    if not counts:
        raise ValueError(f"{name} has no task outcomes")
    if any(count < 0 or count > 8 for count in counts):
        raise ValueError(f"{name} verified counts must be within 0..8")


def compare_paired_verified_counts(
    reference: Sequence[int],
    candidate: Sequence[int],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    _validate_verified_counts(reference, name="reference")
    _validate_verified_counts(candidate, name="candidate")
    if len(reference) != len(candidate):
        raise ValueError("paired model outcomes use different task memberships")
    both = sum(a > 0 and b > 0 for a, b in zip(reference, candidate, strict=True))
    candidate_only = sum(
        a == 0 and b > 0 for a, b in zip(reference, candidate, strict=True)
    )
    reference_only = sum(
        a > 0 and b == 0 for a, b in zip(reference, candidate, strict=True)
    )
    neither = len(reference) - both - candidate_only - reference_only
    discordant = candidate_only + reference_only
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(candidate_only, reference_only) + 1)
        )
        mcnemar = min(1.0, 2.0 * tail / (2**discordant))
    else:
        mcnemar = 1.0
    return {
        "task_count": len(reference),
        "paired_outcomes": {
            "both_solved": both,
            "candidate_only": candidate_only,
            "reference_only": reference_only,
            "neither_solved": neither,
            "paired_wins_candidate_minus_reference": candidate_only - reference_only,
        },
        "exact_two_sided_mcnemar_p": mcnemar,
        "bootstrap": paired_task_bootstrap(
            reference,
            candidate,
            candidates_per_task=8,
            ks=(1, 4, 8),
            resamples=resamples,
            seed=seed,
        ),
    }


def _pass_estimate(counts: Sequence[int], k: int) -> float:
    return fmean(pass_at_k(8, count, k) for count in counts)


def select_generalist_checkpoint(
    evaluations: Mapping[str, CheckpointValidation],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    if set(evaluations) != set(CHECKPOINT_IDS):
        raise ValueError("checkpoint selection requires complete Q0-Q4 validation")
    baseline = evaluations["Q0"]
    _validate_verified_counts(
        baseline.fresh_composition_verified_counts, name="Q0 fresh composition"
    )
    _validate_verified_counts(baseline.minif2f_verified_counts, name="Q0 miniF2F")
    results: dict[str, Any] = {}
    eligible: list[str] = []
    for checkpoint_id in TRAINED_CHECKPOINT_IDS:
        evaluation = evaluations[checkpoint_id]
        if len(evaluation.fresh_composition_verified_counts) != len(
            baseline.fresh_composition_verified_counts
        ):
            raise ValueError(
                "fresh-composition task membership changed across checkpoints"
            )
        comparison = compare_paired_verified_counts(
            baseline.minif2f_verified_counts,
            evaluation.minif2f_verified_counts,
            resamples=resamples,
            seed=seed,
        )
        interval = comparison["bootstrap"]["metrics"]["pass@8"][
            "delta_adapter_minus_base"
        ]["ci95"]
        confidently_negative = interval[1] < 0
        if not confidently_negative:
            eligible.append(checkpoint_id)
        results[checkpoint_id] = {
            "fresh_composition": {
                "pass@1": _pass_estimate(
                    evaluation.fresh_composition_verified_counts, 1
                ),
                "pass@4": _pass_estimate(
                    evaluation.fresh_composition_verified_counts, 4
                ),
                "pass@8": _pass_estimate(
                    evaluation.fresh_composition_verified_counts, 8
                ),
            },
            "minif2f_clean": {
                "pass@1": _pass_estimate(evaluation.minif2f_verified_counts, 1),
                "pass@4": _pass_estimate(evaluation.minif2f_verified_counts, 4),
                "pass@8": _pass_estimate(evaluation.minif2f_verified_counts, 8),
                "paired_against_q0": comparison,
                "confidently_negative_pass8_delta": confidently_negative,
            },
        }
    if not eligible:
        raise ValueError(
            "all Q1-Q4 checkpoints have confidently negative miniF2F pass@8"
        )
    selected = max(
        eligible,
        key=lambda checkpoint_id: (
            results[checkpoint_id]["fresh_composition"]["pass@8"],
            results[checkpoint_id]["minif2f_clean"]["pass@8"],
            results[checkpoint_id]["fresh_composition"]["pass@1"],
            -int(checkpoint_id[1:]),
        ),
    )
    return {
        "rule": [
            "reject only confidently negative miniF2F-clean pass@8 delta vs Q0",
            "highest fresh-composition pass@8",
            "higher miniF2F-clean pass@8",
            "higher fresh-composition pass@1",
            "earlier checkpoint",
        ],
        "selection_inputs": [
            "fresh-composition-valid-v2",
            "minif2f-valid-clean-v2",
        ],
        "test_or_riemann_outcomes_consulted": False,
        "eligible_checkpoints": eligible,
        "selected_checkpoint": selected,
        "checkpoints": results,
    }


def _normalized_tag(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace("+", "-plus")


def normalized_riemann_domain_tags(values: Iterable[str]) -> frozenset[str]:
    """Map Dataset-v2 namespaced topic tags onto the frozen Riemann view tags."""

    normalized = {_normalized_tag(value) for value in values}
    expanded: set[str] = set()
    family_expansions = {
        "zeta-analytic-number-theory": {"zeta", "analytic-number-theory"},
        "riemann-core-bubble": {"riemann-core", "riemann-bubble"},
        "prime-counting-pnt": {"prime-counting", "pnt"},
        "pnt-plus": {"pnt-plus"},
        "arithmetic-functions": {"arithmetic-functions"},
        "prime-arithmetic-divisibility": {"prime-arithmetic", "divisibility"},
    }
    for tag in normalized:
        if tag in RIEMANN_DOMAIN_TAGS:
            expanded.add(tag)
        namespace, separator, value = tag.partition(":")
        if separator and namespace == "prime-family":
            expanded.update(family_expansions.get(value, ()))
        if tag == "riemann-relevance:core":
            expanded.add("riemann-core")
    return frozenset(expanded)


def materialize_fresh_riemann_views(
    records: Iterable[GeneralistProofVariant],
    *,
    training_statement_ids: Iterable[str],
    training_derivation_family_ids: Iterable[str],
) -> dict[str, Any]:
    training_statements = set(training_statement_ids)
    training_families = set(training_derivation_family_ids)
    selected: dict[str, list[GeneralistProofVariant]] = {
        "riemann-fresh-valid-v2": [],
        "riemann-fresh-test-v2": [],
    }
    seen_statements: set[str] = set()
    family_splits: dict[str, str] = {}
    for record in records:
        record.validate()
        tags = normalized_riemann_domain_tags(record.domain_tags)
        if not tags & RIEMANN_DOMAIN_TAGS:
            continue
        if record.source_kind != "synthetic" or record.optimizer_eligible:
            raise ValueError(
                "fresh Riemann views must contain optimizer-invisible synthetic rows"
            )
        if record.split not in {"validation", "test"}:
            raise ValueError("fresh Riemann view row has an invalid split")
        if record.statement_id in training_statements:
            raise ValueError("fresh Riemann statement leaks from training")
        assert record.derivation_family_id is not None
        if record.derivation_family_id in training_families:
            raise ValueError("fresh Riemann derivation family leaks from training")
        if record.statement_id in seen_statements:
            raise ValueError("fresh Riemann statement appears more than once")
        previous_split = family_splits.get(record.derivation_family_id)
        if previous_split is not None and previous_split != record.split:
            raise ValueError("fresh Riemann derivation family crosses validation/test")
        seen_statements.add(record.statement_id)
        family_splits[record.derivation_family_id] = record.split
        view = (
            "riemann-fresh-valid-v2"
            if record.split == "validation"
            else "riemann-fresh-test-v2"
        )
        selected[view].append(record)
    if any(not rows for rows in selected.values()):
        raise ValueError(
            "fresh Riemann validation and test views must both be non-empty"
        )
    views: dict[str, Any] = {}
    for view_id, rows in selected.items():
        ordered = sorted(
            rows, key=lambda item: (item.statement_id, item.proof_variant_id)
        )
        views[view_id] = {
            "statement_ids": [item.statement_id for item in ordered],
            "derivation_family_ids": sorted(
                {str(item.derivation_family_id) for item in ordered}
            ),
            "task_count": len(ordered),
            "optimizer_visible_rows": 0,
            "training_statement_overlap": 0,
            "training_derivation_family_overlap": 0,
            "domain_breakdown": dict(
                sorted(
                    Counter(
                        tag
                        for item in ordered
                        for tag in normalized_riemann_domain_tags(item.domain_tags)
                    ).items()
                )
            ),
        }
    return {
        "views": views,
        "cross_view_statement_overlap": 0,
        "cross_view_derivation_family_overlap": 0,
    }
