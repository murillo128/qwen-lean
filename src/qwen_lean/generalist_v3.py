from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from .dataset_v2 import sha256_file
from .dataset_v3 import (
    first_proof_construct,
    materialize_example,
    normalized_proof_structure,
    read_records,
    read_view,
    representative_variants,
)
from .dataset_v3_schema import DatasetV3Record, DerivedExampleRef
from .generalist_v2 import (
    EXPECTED_LORA_MODULE_COUNTS,
    LORA_TARGET_REGEX,
    LORA_TARGET_SUFFIXES,
    MODEL_ID,
    MODEL_REVISION,
    WeightedTokenizedExample,
)
from .metrics import pass_at_k
from .phase3 import IGNORE_INDEX
from .prompt import normalize_transport


GENERALIST_V3_CONFIG_SCHEMA_VERSION = "generalist-v3-config-v1"
GENERALIST_V3_ARTIFACT_ID = "qwen-lean-generalist-v3"
DATASET_ID = "lean-proof-continuation-v3"
CONTEXT_CHOICES = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
CONFIGURATION_IDS = ("C0", "C1", "C2", "C3")
PRIMARY_CONFIGURATION_IDS = CONFIGURATION_IDS[:3]
RETAINED_STEPS = (100, 250, 500, 1000, 2000, 4000, 8000)
STRUCTURAL_BUCKETS = ("direct", "multi-step", "branching", "deep")
EXECUTION_CONTEXT_TOKENS = 65536


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class GeneralistV3Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> GeneralistV3Config:
        config = cls(path=path.resolve(), value=_read_json(path))
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
    def precision(self) -> dict[str, Any]:
        return self.value["precision"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.value["lora"]

    @property
    def training(self) -> dict[str, Any]:
        return self.value["training"]

    @property
    def preservation(self) -> dict[str, Any]:
        return self.value["preservation"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.value["evaluation"]

    @property
    def collapse_gates(self) -> dict[str, Any]:
        return self.value["collapse_gates"]

    def validate(self) -> None:
        expected = {
            "schema_version": GENERALIST_V3_CONFIG_SCHEMA_VERSION,
            "artifact_id": GENERALIST_V3_ARTIFACT_ID,
        }
        for key, wanted in expected.items():
            if self.value.get(key) != wanted:
                raise ValueError(f"generalist-v3 {key} differs")
        model = self.model
        if (
            model.get("model_id") != MODEL_ID
            or model.get("model_revision") != MODEL_REVISION
            or model.get("tokenizer_id") != MODEL_ID
            or model.get("tokenizer_revision") != MODEL_REVISION
            or model.get("text_only") is not True
            or model.get("chat_template") is not None
            or model.get("add_special_tokens") is not False
        ):
            raise ValueError("generalist-v3 pinned model/tokenizer contract differs")
        binding = self.dataset.get("binding", {})
        expected_binding = {
            "manifest_sha256": "ff73265f882e975e339fd4238fc465e4dd733856c68a6032ad8e435ae73b8dc7",
            "records_sha256": "10f548fd824fd61edf894cf275d4035f9ef4f0f1860628cf1168e498e3d7dc96",
            "optimizer_view_sha256": "0ea4043628dda878ac574e1cde06dfe6bdf69bc602f12f7f83dcfd8e6d07f96d",
            "validation_membership_sha256": "0ec547ff16ce0091bc6f48f673585e4253158aef52cd9dc850d299394c6876fb",
            "test_membership_sha256": "6bb99ed9c476f8ad2dfe7111ecc7b4ec92fbdd6ef21d66ecfe9d71155fdffc19",
            "training_theorems": 178448,
            "validation_theorems": 48,
            "test_theorems": 48,
            "derived_optimizer_examples": 317554,
        }
        if self.dataset.get("package_id") != DATASET_ID or binding != expected_binding:
            raise ValueError("generalist-v3 Dataset-v3 binding differs")
        if self.dataset.get("consume_as_frozen") is not True:
            raise ValueError("generalist-v3 must consume Dataset v3 as frozen")
        if tuple(self.lora.get("target_suffixes", ())) != LORA_TARGET_SUFFIXES:
            raise ValueError("generalist-v3 LoRA target suffixes differ")
        if self.lora.get("target_regex") != LORA_TARGET_REGEX:
            raise ValueError("generalist-v3 LoRA target regex differs")
        if self.lora.get("expected_module_counts") != EXPECTED_LORA_MODULE_COUNTS:
            raise ValueError("generalist-v3 LoRA target counts differ")
        for key, wanted in (
            ("r", 16),
            ("lora_alpha", 32),
            ("lora_dropout", 0.0),
            ("bias", "none"),
            ("modules_to_save", None),
        ):
            if self.lora.get(key) != wanted:
                raise ValueError(f"generalist-v3 lora.{key} differs")
        training = self.training
        if tuple(training.get("context_choices", ())) != CONTEXT_CHOICES:
            raise ValueError("generalist-v3 context choices differ")
        if (
            int(training.get("canonical_context_tokens", 0)) != 262144
            or int(training.get("canonical_maximum_observed_sequence_tokens", 0))
            != 157034
            or int(training.get("resolved_context_tokens", 0))
            != EXECUTION_CONTEXT_TOKENS
        ):
            raise ValueError("generalist-v3 canonical/execution context contract differs")
        execution_view = training.get("execution_view", {})
        if (
            execution_view.get("maximum_sequence_tokens") != EXECUTION_CONTEXT_TOKENS
            or execution_view.get("expected_quarantined_examples") != 6
            or execution_view.get("renormalize_remaining_example_mass_within_theorem")
            is not True
            or execution_view.get("exclude_theorem_when_no_example_remains") is not True
        ):
            raise ValueError("generalist-v3 execution-view contract differs")
        structural = training.get("structural_reweighting", {})
        if (
            tuple(structural.get("bucket_order", ())) != STRUCTURAL_BUCKETS
            or structural.get("raw_multiplier") != "sqrt((1/K)/f_b)"
            or structural.get("lower_bound") != 0.75
            or structural.get("upper_bound") != 1.5
            or structural.get("normalize_training_set_mean_to") != 1.0
        ):
            raise ValueError("generalist-v3 structural reweighting contract differs")
        if tuple(training.get("retained_steps", ())) != RETAINED_STEPS:
            raise ValueError("generalist-v3 retained steps differ")
        if (
            training.get("stream_seed") != 0
            or training.get("sample_theorem_uniformly") is not False
            or training.get("sample_theorem_by_structural_multiplier") is not True
            or training.get("sample_example_by_exact_mass") is not True
            or training.get("gradient_accumulation_steps") != 8
            or training.get("warmup_steps") != 100
            or training.get("packing") is not False
            or training.get("truncation") is not False
        ):
            raise ValueError("generalist-v3 stream/optimizer envelope differs")
        configurations = training.get("configurations", {})
        expected_configs = {
            "C0": (3e-5, 0.0, False),
            "C1": (3e-5, 0.1, True),
            "C2": (1e-5, 0.1, True),
            "C3": (1e-5, 0.3, True),
        }
        for identifier, (lr, coefficient, eligible) in expected_configs.items():
            observed = configurations.get(identifier, {})
            if (
                observed.get("learning_rate") != lr
                or observed.get("base_kl_lambda") != coefficient
                or observed.get("eligible") is not eligible
            ):
                raise ValueError(f"generalist-v3 configuration {identifier} differs")
        preservation = self.preservation
        if (
            preservation.get("anchor_count") != 512
            or preservation.get("whole_anchors") != 256
            or preservation.get("incremental_anchors") != 256
            or preservation.get("reference_logits") != "full-vocabulary"
            or preservation.get("loss_direction") != "KL(p_base||p_current)"
            or preservation.get("maximum_anchor_fraction_per_structural_bucket")
            != 0.4
        ):
            raise ValueError("generalist-v3 preservation contract differs")
        evaluation = self.evaluation
        if (
            evaluation.get("candidates_per_task") != 8
            or evaluation.get("interfaces") != ["whole", "incremental"]
            or evaluation.get("sampling", {}).get("temperature") != 0.8
            or evaluation.get("sampling", {}).get("top_p") != 0.95
            or evaluation.get("sampling", {}).get("top_k") != -1
            or evaluation.get("sampling", {}).get("max_new_tokens") != 1024
            or evaluation.get("sampling", {}).get("seed") != 0
        ):
            raise ValueError("generalist-v3 evaluation contract differs")


@dataclass(frozen=True)
class DatasetBinding:
    package_root: Path
    manifest_sha256: str
    records_sha256: str
    optimizer_view_sha256: str
    validation_membership_sha256: str
    test_membership_sha256: str
    training_theorems: int
    validation_theorems: int
    test_theorems: int
    derived_optimizer_examples: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["package_root"] = str(self.package_root)
        return value


def validate_dataset_binding(
    config: GeneralistV3Config,
    repository_root: Path,
    package_root: Path | None = None,
) -> DatasetBinding:
    config.validate()
    repository_root = repository_root.resolve()
    package_root = (
        repository_root / str(config.dataset["local_artifact_root"])
        if package_root is None
        else package_root.resolve()
    )
    committed_manifest = repository_root / str(config.dataset["committed_manifest"])
    local_manifest = package_root / "manifest.json"
    if not committed_manifest.is_file() or not local_manifest.is_file():
        raise FileNotFoundError("accepted Dataset-v3 manifest/package is unavailable")
    committed = _read_json(committed_manifest)
    local = _read_json(local_manifest)
    if committed != local:
        raise ValueError("local Dataset-v3 manifest differs from merged snapshot")
    binding = config.dataset["binding"]
    observed = {
        "manifest_sha256": sha256_file(local_manifest),
        "records_sha256": sha256_file(package_root / "records.jsonl.gz"),
        "optimizer_view_sha256": sha256_file(package_root / "optimizer-view.jsonl.gz"),
        "validation_membership_sha256": sha256_file(package_root / "validation-membership.jsonl"),
        "test_membership_sha256": sha256_file(package_root / "test-membership.jsonl"),
    }
    for key, value in observed.items():
        if value != binding[key]:
            raise ValueError(f"Dataset-v3 {key} differs from the frozen binding")
    summary = local.get("summary", {})
    roles = summary.get("roles", {})
    counts = {
        "training_theorems": int(roles.get("training", -1)),
        "validation_theorems": int(roles.get("validation", -1)),
        "test_theorems": int(roles.get("test", -1)),
        "derived_optimizer_examples": int(summary.get("derived_optimizer_examples", -1)),
    }
    for key, value in counts.items():
        if value != binding[key]:
            raise ValueError(f"Dataset-v3 {key} differs from the frozen binding")
    return DatasetBinding(package_root=package_root, **observed, **counts)


def tokenize_materialized_example(
    materialized: Mapping[str, Any],
    tokenizer: Tokenizer,
    *,
    maximum_sequence_tokens: int | None = None,
) -> WeightedTokenizedExample:
    if tokenizer.eos_token_id is None:
        raise ValueError("the pinned Qwen3.5 tokenizer has no EOS token")
    model_input = str(materialized["model_input"])
    target = normalize_transport(str(materialized["target"]))
    prompt_ids = tuple(tokenizer.encode(model_input, add_special_tokens=False))
    target_ids = tuple(tokenizer.encode(target, add_special_tokens=False))
    eos = int(tokenizer.eos_token_id)
    if eos in (*prompt_ids, *target_ids):
        raise ValueError("Dataset-v3 serialized text already contains EOS")
    input_ids = prompt_ids + target_ids + (eos,)
    example = WeightedTokenizedExample(
        statement_id=str(materialized["statement_id"]),
        proof_variant_id=str(materialized["example_id"]),
        declaration_name=str(materialized["task_kind"]),
        prompt=model_input,
        completion=target,
        input_ids=input_ids,
        labels=(IGNORE_INDEX,) * len(prompt_ids) + target_ids + (eos,),
        attention_mask=(1,) * len(input_ids),
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(target_ids),
        example_weight=1.0,
    )
    example.validate(eos, maximum_sequence_tokens)
    return example


def _training_records(path: Path) -> Iterator[DatasetV3Record]:
    """Yield the leading frozen training partition without traversing eval rows."""

    count = 0
    for record in read_records(path):
        if record.role != "training":
            break
        count += 1
        yield record
    if count == 0:
        raise ValueError("Dataset-v3 records do not begin with training membership")


def context_for_maximum(maximum_tokens: int, choices: Sequence[int] = CONTEXT_CHOICES) -> int:
    try:
        return next(value for value in choices if maximum_tokens <= value)
    except StopIteration as error:
        raise ValueError(f"no supported context contains {maximum_tokens} tokens") from error


def tokenizer_length_census(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    tokenizer: Tokenizer,
    *,
    progress_every_records: int = 0,
) -> dict[str, Any]:
    bucket_counts = Counter({str(value): 0 for value in CONTEXT_CHOICES})
    maximum: dict[str, Any] | None = None
    example_count = 0
    records = {
        record.statement_id: record
        for record in _training_records(binding.package_root / "records.jsonl.gz")
    }
    training_records = len(records)
    for reference in read_view(binding.package_root / "optimizer-view.jsonl.gz"):
        materialized = materialize_example(records[reference.statement_id], reference)
        prompt_tokens = len(tokenizer.encode(materialized["model_input"], add_special_tokens=False))
        target_tokens = len(tokenizer.encode(materialized["target"], add_special_tokens=False))
        sequence_tokens = prompt_tokens + target_tokens + 1
        selected = context_for_maximum(sequence_tokens)
        bucket_counts[str(selected)] += 1
        example_count += 1
        if maximum is None or sequence_tokens > int(maximum["sequence_tokens"]):
            maximum = {
                "statement_id": reference.statement_id,
                "example_id": reference.example_id,
                "task_kind": reference.kind,
                "prompt_tokens": prompt_tokens,
                "target_tokens_excluding_eos": target_tokens,
                "sequence_tokens": sequence_tokens,
            }
    if training_records != config.dataset["binding"]["training_theorems"]:
        raise RuntimeError("Dataset-v3 tokenizer census changed training membership")
    if example_count != config.dataset["binding"]["derived_optimizer_examples"]:
        raise RuntimeError("Dataset-v3 tokenizer census changed optimizer membership")
    if maximum is None:
        raise RuntimeError("Dataset-v3 tokenizer census is empty")
    selected_context = context_for_maximum(int(maximum["sequence_tokens"]))
    if (
        selected_context != int(config.training["canonical_context_tokens"])
        or int(maximum["sequence_tokens"])
        != int(config.training["canonical_maximum_observed_sequence_tokens"])
    ):
        raise ValueError("Dataset-v3 tokenizer census differs from the frozen config")
    return {
        "schema_version": "generalist-v3-tokenizer-census-v1",
        "model": config.model,
        "dataset_binding": binding.to_dict(),
        "training_theorems": training_records,
        "optimizer_examples": example_count,
        "context_bucket_counts": dict(bucket_counts),
        "maximum_example": maximum,
        "selected_context_tokens": selected_context,
        "truncated_or_dropped": 0,
    }


def load_training_index(
    binding: DatasetBinding,
) -> tuple[dict[str, DatasetV3Record], dict[str, tuple[DerivedExampleRef, ...]]]:
    records = {
        record.statement_id: record
        for record in _training_records(binding.package_root / "records.jsonl.gz")
    }
    by_statement: dict[str, list[DerivedExampleRef]] = defaultdict(list)
    for reference in read_view(binding.package_root / "optimizer-view.jsonl.gz"):
        by_statement[reference.statement_id].append(reference)
    if set(records) != set(by_statement):
        raise ValueError("Dataset-v3 records and optimizer view theorem memberships differ")
    normalized: dict[str, tuple[DerivedExampleRef, ...]] = {}
    for statement_id, references in by_statement.items():
        ordered = tuple(sorted(references, key=lambda item: item.example_id))
        total = sum(
            (Fraction(item.mass_numerator, item.mass_denominator) for item in ordered),
            Fraction(0, 1),
        )
        if total != Fraction(1, 1):
            raise ValueError(f"Dataset-v3 theorem mass differs: {statement_id}={total}")
        normalized[statement_id] = ordered
    return records, normalized


def structural_bucket(record: DatasetV3Record) -> str:
    """Assign the frozen #83 proof-execution structural bucket."""

    variants = representative_variants(record)
    maximum_boundaries = max((len(item.boundaries) for item in variants), default=0)
    maximum_branches = max(
        (
            sum(boundary.structural_kind == "branch" for boundary in item.boundaries)
            for item in variants
        ),
        default=0,
    )
    if maximum_boundaries >= 4 or maximum_branches >= 2:
        inferred = "deep"
    elif maximum_branches >= 1:
        inferred = "branching"
    elif maximum_boundaries >= 1:
        inferred = "multi-step"
    else:
        inferred = "direct"
    explicit = record.structural_class if record.provenance == "synthetic" else None
    if explicit is None:
        return inferred
    if explicit not in STRUCTURAL_BUCKETS:
        raise ValueError(f"unsupported Dataset-v3 structural class: {explicit}")
    return max((inferred, explicit), key=STRUCTURAL_BUCKETS.index)


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _execution_view_binding_matches(
    binding: DatasetBinding, execution_view: Mapping[str, Any]
) -> bool:
    observed = execution_view.get("dataset_binding", {})
    return all(
        observed.get(key) == getattr(binding, key)
        for key in (
            "manifest_sha256",
            "records_sha256",
            "optimizer_view_sha256",
            "validation_membership_sha256",
            "test_membership_sha256",
            "training_theorems",
            "validation_theorems",
            "test_theorems",
            "derived_optimizer_examples",
        )
    )


def freeze_training_execution_view(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    tokenizer: Tokenizer,
    census_path: Path,
    output_path: Path,
    *,
    progress_every_examples: int = 0,
) -> dict[str, Any]:
    """Freeze the explicit <=65,536-token optimizer view without mutating Dataset v3."""

    config.validate()
    census = _read_json(census_path)
    if (
        census.get("schema_version") != "generalist-v3-tokenizer-census-v1"
        or int(census.get("selected_context_tokens", 0))
        != int(config.training["canonical_context_tokens"])
        or int(census.get("maximum_example", {}).get("sequence_tokens", 0))
        != int(config.training["canonical_maximum_observed_sequence_tokens"])
    ):
        raise ValueError("training execution view needs the frozen canonical census")
    cap = int(config.training["execution_view"]["maximum_sequence_tokens"])
    records, by_statement = load_training_index(binding)
    quarantined: list[dict[str, Any]] = []
    maximum_eligible: dict[str, Any] | None = None
    token_totals: Counter[tuple[str, str]] = Counter()
    token_counts: Counter[tuple[str, str]] = Counter()
    example_count = 0
    for statement_id in sorted(by_statement):
        record = records[statement_id]
        bucket = structural_bucket(record)
        for reference in by_statement[statement_id]:
            materialized = materialize_example(record, reference)
            prompt_tokens = len(
                tokenizer.encode(materialized["model_input"], add_special_tokens=False)
            )
            target_tokens = len(
                tokenizer.encode(materialized["target"], add_special_tokens=False)
            )
            sequence_tokens = prompt_tokens + target_tokens + 1
            detail = {
                "statement_id": statement_id,
                "example_id": reference.example_id,
                "task_kind": reference.kind,
                "structural_bucket": bucket,
                "prompt_tokens": prompt_tokens,
                "target_tokens_excluding_eos": target_tokens,
                "sequence_tokens": sequence_tokens,
            }
            if sequence_tokens > cap:
                quarantined.append(
                    {
                        **detail,
                        "status": config.training["execution_view"]["quarantine_label"],
                    }
                )
            else:
                if maximum_eligible is None or sequence_tokens > int(
                    maximum_eligible["sequence_tokens"]
                ):
                    maximum_eligible = detail
                key = (bucket, reference.kind)
                token_totals[key] += target_tokens
                token_counts[key] += 1
            example_count += 1
            if progress_every_examples and example_count % progress_every_examples == 0:
                print(
                    json.dumps(
                        {
                            "execution_view_examples": example_count,
                            "total": binding.derived_optimizer_examples,
                        }
                    ),
                    flush=True,
                )
    quarantined.sort(key=lambda item: (item["statement_id"], item["example_id"]))
    if example_count != binding.derived_optimizer_examples or maximum_eligible is None:
        raise RuntimeError("training execution view did not enumerate the frozen optimizer view")
    expected_quarantined = int(
        config.training["execution_view"]["expected_quarantined_examples"]
    )
    if len(quarantined) != expected_quarantined:
        raise RuntimeError(
            f"training execution quarantine changed: {len(quarantined)} != "
            f"{expected_quarantined}"
        )
    quarantined_ids = {str(item["example_id"]) for item in quarantined}
    affected = sorted({str(item["statement_id"]) for item in quarantined})
    excluded_theorems: list[str] = []
    mass_renormalization: list[dict[str, Any]] = []
    eligible_ids: list[str] = []
    for statement_id in sorted(by_statement):
        remaining = [
            item
            for item in by_statement[statement_id]
            if item.example_id not in quarantined_ids
        ]
        eligible_ids.extend(item.example_id for item in remaining)
        if not remaining:
            excluded_theorems.append(statement_id)
        if statement_id in affected:
            eligible_mass = sum(
                (
                    Fraction(item.mass_numerator, item.mass_denominator)
                    for item in remaining
                ),
                Fraction(0, 1),
            )
            mass_renormalization.append(
                {
                    "statement_id": statement_id,
                    "frozen_example_count": len(by_statement[statement_id]),
                    "eligible_example_count": len(remaining),
                    "eligible_frozen_mass": _fraction_dict(eligible_mass),
                    "renormalization_factor": (
                        None
                        if not remaining
                        else _fraction_dict(Fraction(1, 1) / eligible_mass)
                    ),
                }
            )
    retained_examples = example_count - len(quarantined)
    target_length_by_bucket_kind = []
    for key in sorted(token_counts):
        bucket, kind = key
        target_length_by_bucket_kind.append(
            {
                "structural_bucket": bucket,
                "task_kind": kind,
                "examples": token_counts[key],
                "mean_target_tokens_excluding_eos": token_totals[key] / token_counts[key],
            }
        )
    value = {
        "schema_version": "generalist-v3-training-execution-view-v1",
        "dataset_binding": binding.to_dict(),
        "canonical_tokenizer_census_sha256": sha256_file(census_path),
        "maximum_sequence_tokens": cap,
        "canonical_optimizer_examples": example_count,
        "quarantined_example_count": len(quarantined),
        "quarantined_unique_theorems": len(affected),
        "retained_example_count": retained_examples,
        "retained_theorem_count": len(by_statement) - len(excluded_theorems),
        "retained_example_fraction": _fraction_dict(Fraction(retained_examples, example_count)),
        "quarantined_examples_sha256": _sha256_json(quarantined),
        "quarantined_examples": quarantined,
        "affected_structural_bucket_counts": dict(
            sorted(Counter(item["structural_bucket"] for item in quarantined).items())
        ),
        "affected_task_kind_counts": dict(
            sorted(Counter(item["task_kind"] for item in quarantined).items())
        ),
        "mass_renormalization_sha256": _sha256_json(mass_renormalization),
        "mass_renormalization": mass_renormalization,
        "excluded_theorem_count": len(excluded_theorems),
        "excluded_theorem_ids_sha256": _sha256_json(excluded_theorems),
        "excluded_theorem_ids": excluded_theorems,
        "eligible_example_ids_sha256": _sha256_json(eligible_ids),
        "maximum_eligible_example": maximum_eligible,
        "target_length_by_structural_bucket_and_kind": target_length_by_bucket_kind,
        "dataset_v3_mutated": False,
        "truncated_examples": 0,
        "silent_drops": 0,
        "semantic_test_accessed": False,
    }
    value["execution_view_sha256"] = _sha256_json(value)
    _write_json(output_path, value)
    return value


def load_execution_training_index(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    execution_view: Mapping[str, Any],
) -> tuple[dict[str, DatasetV3Record], dict[str, tuple[DerivedExampleRef, ...]]]:
    """Load the frozen #83 eligibility view and renormalize exact within-theorem mass."""

    if (
        execution_view.get("schema_version")
        != "generalist-v3-training-execution-view-v1"
        or not _execution_view_binding_matches(binding, execution_view)
        or int(execution_view.get("maximum_sequence_tokens", 0))
        != int(config.training["resolved_context_tokens"])
        or int(execution_view.get("quarantined_example_count", -1))
        != int(config.training["execution_view"]["expected_quarantined_examples"])
    ):
        raise ValueError("generalist-v3 training execution view differs")
    records, frozen = load_training_index(binding)
    quarantined = {
        str(item["example_id"]) for item in execution_view["quarantined_examples"]
    }
    normalized: dict[str, tuple[DerivedExampleRef, ...]] = {}
    eligible_ids: list[str] = []
    for statement_id in sorted(frozen):
        remaining = [item for item in frozen[statement_id] if item.example_id not in quarantined]
        if not remaining:
            continue
        total = sum(
            (Fraction(item.mass_numerator, item.mass_denominator) for item in remaining),
            Fraction(0, 1),
        )
        references = tuple(
            DerivedExampleRef(
                schema_version=item.schema_version,
                example_id=item.example_id,
                statement_id=item.statement_id,
                proof_variant_id=item.proof_variant_id,
                kind=item.kind,
                boundary_id=item.boundary_id,
                mass_numerator=(
                    Fraction(item.mass_numerator, item.mass_denominator) / total
                ).numerator,
                mass_denominator=(
                    Fraction(item.mass_numerator, item.mass_denominator) / total
                ).denominator,
            )
            for item in remaining
        )
        if sum(
            (Fraction(item.mass_numerator, item.mass_denominator) for item in references),
            Fraction(0, 1),
        ) != Fraction(1, 1):
            raise RuntimeError(f"execution-view theorem mass differs: {statement_id}")
        normalized[statement_id] = references
        eligible_ids.extend(item.example_id for item in references)
    if len(eligible_ids) != int(execution_view["retained_example_count"]):
        raise RuntimeError("execution-view retained example count differs")
    if _sha256_json(eligible_ids) != execution_view["eligible_example_ids_sha256"]:
        raise RuntimeError("execution-view eligible membership hash differs")
    return {key: records[key] for key in normalized}, normalized


def freeze_structural_sampling_manifest(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    execution_view: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    all_records, _ = load_training_index(binding)
    eligible_records, eligible_examples = load_execution_training_index(
        config, binding, execution_view
    )
    pre_mapping = {key: structural_bucket(value) for key, value in all_records.items()}
    mapping = {key: structural_bucket(value) for key, value in eligible_records.items()}
    pre_counts = Counter(pre_mapping.values())
    counts = Counter(mapping.values())
    nonempty = [bucket for bucket in STRUCTURAL_BUCKETS if counts[bucket]]
    theorem_count = len(mapping)
    bounded: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for bucket in nonempty:
        frequency = counts[bucket] / theorem_count
        raw = math.sqrt((1.0 / len(nonempty)) / frequency)
        bounded[bucket] = min(1.5, max(0.75, raw))
    bounded_mean = math.fsum(counts[b] * bounded[b] for b in nonempty) / theorem_count
    normalized = {bucket: bounded[bucket] / bounded_mean for bucket in nonempty}
    total_weight = math.fsum(counts[b] * normalized[b] for b in nonempty)
    for bucket in nonempty:
        frequency = counts[bucket] / theorem_count
        raw = math.sqrt((1.0 / len(nonempty)) / frequency)
        rows.append(
            {
                "structural_bucket": bucket,
                "pre_quarantine_theorems": pre_counts[bucket],
                "eligible_theorems": counts[bucket],
                "raw_theorem_fraction": frequency,
                "raw_multiplier": raw,
                "bounded_multiplier": bounded[bucket],
                "normalized_multiplier": normalized[bucket],
                "expected_optimizer_mass_fraction": (
                    counts[bucket] * normalized[bucket] / total_weight
                ),
                "eligible_whole_examples": sum(
                    item.kind == "whole"
                    for statement_id, refs in eligible_examples.items()
                    if mapping[statement_id] == bucket
                    for item in refs
                ),
                "eligible_incremental_examples": sum(
                    item.kind == "continuation"
                    for statement_id, refs in eligible_examples.items()
                    if mapping[statement_id] == bucket
                    for item in refs
                ),
            }
        )
    canonical_mapping = [
        {"statement_id": statement_id, "structural_bucket": mapping[statement_id]}
        for statement_id in sorted(mapping)
    ]
    value = {
        "schema_version": "generalist-v3-structural-sampling-v1",
        "dataset_binding": binding.to_dict(),
        "execution_view_sha256": str(execution_view["execution_view_sha256"]),
        "bucket_order": list(STRUCTURAL_BUCKETS),
        "inference_rule": {
            "deep": "max_boundaries>=4 or max_branch_boundaries>=2",
            "branching": "otherwise max_branch_boundaries>=1",
            "multi-step": "otherwise max_boundaries>=1",
            "direct": "otherwise",
            "synthetic_override": "take more complex explicit structural_class",
        },
        "eligible_theorems": theorem_count,
        "nonempty_bucket_count": len(nonempty),
        "pre_quarantine_bucket_counts": dict(
            (bucket, pre_counts[bucket]) for bucket in STRUCTURAL_BUCKETS
        ),
        "post_quarantine_bucket_counts": dict(
            (bucket, counts[bucket]) for bucket in STRUCTURAL_BUCKETS
        ),
        "multipliers": rows,
        "normalized_theorem_multiplier_mean": total_weight / theorem_count,
        "theorem_to_bucket_mapping_sha256": _sha256_json(canonical_mapping),
        "semantic_validation_or_test_accessed": False,
    }
    value["structural_sampling_sha256"] = _sha256_json(value)
    _write_json(output_path, value)
    return value


def choose_exact_mass_reference(
    references: Sequence[DerivedExampleRef], threshold: Fraction
) -> DerivedExampleRef:
    if not Fraction(0, 1) <= threshold < Fraction(1, 1):
        raise ValueError("exact mass threshold must be in [0,1)")
    cumulative = Fraction(0, 1)
    for reference in references:
        cumulative += Fraction(reference.mass_numerator, reference.mass_denominator)
        if threshold < cumulative:
            return reference
    raise RuntimeError("Dataset-v3 exact mass choice did not resolve")


def deterministic_stream_references(
    examples_by_statement: Mapping[str, Sequence[DerivedExampleRef]],
    *,
    microbatches: int,
    seed: int = 0,
    theorem_weights: Mapping[str, float] | None = None,
) -> Iterator[DerivedExampleRef]:
    if microbatches < 1 or seed != 0:
        raise ValueError("generalist-v3 stream requires positive rows and seed=0")
    statement_ids = sorted(examples_by_statement)
    if not statement_ids:
        raise ValueError("generalist-v3 stream has no training theorems")
    if theorem_weights is None:
        weights = [1.0] * len(statement_ids)
    else:
        if set(theorem_weights) != set(statement_ids):
            raise ValueError("structural theorem weights differ from execution membership")
        weights = [float(theorem_weights[item]) for item in statement_ids]
        if any(not math.isfinite(item) or item <= 0 for item in weights):
            raise ValueError("structural theorem weights must be finite and positive")
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        cumulative.append(running)
    rng = random.Random(seed)
    denominator = 1 << 256
    for _ in range(microbatches):
        statement_threshold = rng.random() * cumulative[-1]
        statement_id = statement_ids[bisect_right(cumulative, statement_threshold)]
        threshold = Fraction(rng.getrandbits(256), denominator)
        yield choose_exact_mass_reference(examples_by_statement[statement_id], threshold)


def write_training_stream(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    execution_view: Mapping[str, Any],
    structural_manifest: Mapping[str, Any],
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    records, by_statement = load_execution_training_index(config, binding, execution_view)
    if (
        structural_manifest.get("schema_version")
        != "generalist-v3-structural-sampling-v1"
        or structural_manifest.get("execution_view_sha256")
        != execution_view.get("execution_view_sha256")
    ):
        raise ValueError("training stream needs the frozen structural manifest")
    bucket_multipliers = {
        str(item["structural_bucket"]): float(item["normalized_multiplier"])
        for item in structural_manifest["multipliers"]
    }
    statement_buckets = {
        statement_id: structural_bucket(record) for statement_id, record in records.items()
    }
    theorem_weights = {
        statement_id: bucket_multipliers[statement_buckets[statement_id]]
        for statement_id in by_statement
    }
    microbatches = int(config.training["maximum_optimizer_steps"]) * int(
        config.training["gradient_accumulation_steps"]
    )
    digest = hashlib.sha256()
    kind_counts = Counter()
    bucket_counts = Counter()
    bucket_kind_counts = Counter()
    statement_counts = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="\n") as handle:
        for stream_index, reference in enumerate(
            deterministic_stream_references(
                by_statement,
                microbatches=microbatches,
                theorem_weights=theorem_weights,
            ),
        ):
            materialized = materialize_example(records[reference.statement_id], reference)
            bucket = statement_buckets[reference.statement_id]
            row = {
                "schema_version": "generalist-v3-stream-row-v1",
                "stream_index": stream_index,
                "optimizer_step": stream_index // 8 + 1,
                "accumulation_index": stream_index % 8,
                "statement_id": reference.statement_id,
                "example_id": reference.example_id,
                "task_kind": reference.kind,
                "structural_bucket": bucket,
                "model_input": materialized["model_input"],
                "target": materialized["target"],
                "model_input_sha256": _sha256_text(materialized["model_input"]),
                "target_sha256": _sha256_text(materialized["target"]),
            }
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(encoded + "\n")
            digest.update(encoded.encode("utf-8") + b"\n")
            kind_counts[reference.kind] += 1
            bucket_counts[bucket] += 1
            bucket_kind_counts[(bucket, reference.kind)] += 1
            statement_counts[reference.statement_id] += 1
    expected_by_bucket = {
        str(item["structural_bucket"]): float(item["expected_optimizer_mass_fraction"])
        for item in structural_manifest["multipliers"]
    }
    agreement = {}
    for bucket, expected in expected_by_bucket.items():
        observed = bucket_counts[bucket] / microbatches
        ordinary_limit = max(
            0.005,
            6.0 * math.sqrt(expected * (1.0 - expected) / microbatches),
        )
        agreement[bucket] = {
            "expected_fraction": expected,
            "observed_fraction": observed,
            "absolute_difference": abs(observed - expected),
            "ordinary_finite_sample_limit": ordinary_limit,
            "passed": abs(observed - expected) <= ordinary_limit,
        }
    if not all(item["passed"] for item in agreement.values()):
        raise RuntimeError("structurally reweighted stream differs from expected distribution")
    manifest = {
        "schema_version": "generalist-v3-training-stream-v1",
        "stream_id": config.training["stream_id"],
        "seed": 0,
        "sampling": "structural-theorem-weight-then-renormalized-exact-example-mass-with-replacement",
        "dataset_binding": binding.to_dict(),
        "execution_view_sha256": execution_view["execution_view_sha256"],
        "structural_sampling_sha256": structural_manifest["structural_sampling_sha256"],
        "microbatches": microbatches,
        "optimizer_steps": int(config.training["maximum_optimizer_steps"]),
        "gradient_accumulation_steps": 8,
        "kind_counts": dict(sorted(kind_counts.items())),
        "structural_bucket_counts": dict(sorted(bucket_counts.items())),
        "structural_bucket_kind_counts": {
            f"{bucket}:{kind}": count
            for (bucket, kind), count in sorted(bucket_kind_counts.items())
        },
        "structural_distribution_agreement": agreement,
        "unique_theorems_sampled": len(statement_counts),
        "maximum_theorem_draws": max(statement_counts.values()),
        "minimum_sampled_theorem_draws": min(statement_counts.values()),
        "canonical_rows_sha256": digest.hexdigest(),
        "gzip_file_sha256": sha256_file(output_path),
        "first_rows": list(iter_training_stream(output_path, limit=8)),
    }
    _write_json(manifest_path, manifest)
    return manifest


def iter_training_stream(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            if line.strip():
                yield json.loads(line)


def _length_bucket(tokens: int, boundaries: Sequence[int]) -> str:
    for boundary in boundaries:
        if tokens <= boundary:
            return f"le-{boundary}"
    return f"gt-{boundaries[-1]}"


def _balanced_hash_selection(
    candidates: Sequence[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    by_bucket_stratum: dict[
        str, dict[tuple[str, str, str], list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        bucket = str(candidate["structural_bucket"])
        key = (
            str(candidate["proof_form"]),
            str(candidate["first_construct"]),
            str(candidate["length_bucket"]),
        )
        by_bucket_stratum[bucket][key].append(candidate)
    bucket_orders: dict[str, list[dict[str, Any]]] = {}
    for bucket, by_stratum in by_bucket_stratum.items():
        for values in by_stratum.values():
            values.sort(
                key=lambda item: (
                    not bool(item["preferred_length"]),
                    hashlib.sha256(
                        f"generalist-v3-anchor-v2\0{seed}\0{item['example_id']}".encode()
                    ).digest(),
                )
            )
        order: list[dict[str, Any]] = []
        cursor = 0
        keys = sorted(by_stratum)
        while True:
            progressed = False
            for key in keys:
                values = by_stratum[key]
                if cursor < len(values):
                    order.append(values[cursor])
                    progressed = True
            if not progressed:
                break
            cursor += 1
        bucket_orders[bucket] = order
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        progressed = False
        for bucket in STRUCTURAL_BUCKETS:
            values = bucket_orders.get(bucket, [])
            if cursor < len(values):
                selected.append(values[cursor])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"anchor candidates contain fewer than {count} examples")
        cursor += 1
    return selected


def freeze_anchor_manifest(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    tokenizer: Tokenizer,
    execution_view: Mapping[str, Any],
    structural_manifest: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    records, by_statement = load_execution_training_index(config, binding, execution_view)
    if (
        structural_manifest.get("schema_version")
        != "generalist-v3-structural-sampling-v1"
        or structural_manifest.get("execution_view_sha256")
        != execution_view.get("execution_view_sha256")
    ):
        raise ValueError("anchor selection needs the frozen structural manifest")
    boundaries = tuple(int(value) for value in config.preservation["length_buckets"])
    candidates: dict[str, list[dict[str, Any]]] = {"whole": [], "continuation": []}
    for statement_id in sorted(by_statement):
        record = records[statement_id]
        bucket = structural_bucket(record)
        variants = {item.proof_variant_id: item for item in record.proof_variants}
        for reference in by_statement[statement_id]:
            materialized = materialize_example(record, reference)
            tokens = len(tokenizer.encode(materialized["model_input"], add_special_tokens=False))
            target_construct = first_proof_construct(materialized["target"])
            candidates[reference.kind].append(
                {
                    "statement_id": statement_id,
                    "example_id": reference.example_id,
                    "proof_variant_id": reference.proof_variant_id,
                    "task_kind": reference.kind,
                    "boundary_id": reference.boundary_id,
                    "proof_form": variants[reference.proof_variant_id].proof_form,
                    "structural_bucket": bucket,
                    "first_construct": target_construct,
                    "input_tokens": tokens,
                    "length_bucket": _length_bucket(tokens, boundaries),
                    "preferred_length": tokens <= 4096,
                    "model_input_sha256": _sha256_text(materialized["model_input"]),
                }
            )
    selected = [
        *_balanced_hash_selection(candidates["whole"], count=256, seed=0),
        *_balanced_hash_selection(candidates["continuation"], count=256, seed=0),
    ]
    if len(selected) != 512 or len({item["example_id"] for item in selected}) != 512:
        raise RuntimeError("generalist-v3 anchor selection is not 512 unique examples")
    bucket_counts = Counter(item["structural_bucket"] for item in selected)
    maximum_fraction = float(
        config.preservation["maximum_anchor_fraction_per_structural_bucket"]
    )
    if max(bucket_counts.values()) / len(selected) > maximum_fraction:
        raise RuntimeError("generalist-v3 anchors exceed the structural-bucket cap")
    selected.sort(key=lambda item: (item["task_kind"], item["example_id"]))
    value = {
        "schema_version": "generalist-v3-anchor-manifest-v1",
        "dataset_binding": binding.to_dict(),
        "execution_view_sha256": execution_view["execution_view_sha256"],
        "structural_sampling_sha256": structural_manifest["structural_sampling_sha256"],
        "selection_seed": 0,
        "selection_rule": config.preservation["selection_rule"],
        "anchor_count": len(selected),
        "kind_counts": dict(Counter(item["task_kind"] for item in selected)),
        "proof_form_counts": dict(Counter(item["proof_form"] for item in selected)),
        "first_construct_counts": dict(Counter(item["first_construct"] for item in selected)),
        "length_bucket_counts": dict(Counter(item["length_bucket"] for item in selected)),
        "structural_bucket_counts": dict(sorted(bucket_counts.items())),
        "maximum_structural_bucket_fraction": max(bucket_counts.values()) / len(selected),
        "preferred_length_count": sum(bool(item["preferred_length"]) for item in selected),
        "anchors_sha256": _sha256_json(selected),
        "anchors": selected,
        "validation_or_test_anchors": 0,
    }
    _write_json(output_path, value)
    return value


def _imports_preamble(record: DatasetV3Record) -> str:
    return "\n".join(f"import {name}" for name in record.environment.imports)


def freeze_canary_manifest(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    output_path: Path,
    *,
    role: str = "validation",
) -> dict[str, Any]:
    if role not in {"validation", "test"}:
        raise ValueError("generalist-v3 canary role must be validation or test")
    records = []
    for record in read_records(binding.package_root / "records.jsonl.gz"):
        if role == "validation" and record.role == "test":
            break
        if record.role == role:
            records.append(record)
    expected = int(config.dataset["binding"][f"{role}_theorems"])
    if len(records) != expected:
        raise ValueError(f"Dataset-v3 {role} canary membership differs")
    tasks: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.statement_id):
        variant = sorted(record.proof_variants, key=lambda item: item.proof_variant_id)[0]
        boundaries = sorted(
            variant.boundaries,
            key=lambda item: (item.segment_index, item.prefix_end, item.boundary_id),
        )
        if not boundaries:
            raise ValueError(f"{role} theorem lacks an incremental boundary")
        boundary = boundaries[len(boundaries) // 2]
        prefix = variant.proof_text[: boundary.prefix_end]
        continuation = variant.proof_text[boundary.prefix_end :]
        shared = {
            "statement_id": record.statement_id,
            "declaration": record.canonical_declaration,
            "declaration_name": variant.source_declaration_name,
            "preamble": _imports_preamble(record),
            "structural_class": record.structural_class,
            "logic_shape": record.logic_shape,
            "generator_family": record.generator_family,
            "derivation_family_fingerprint": record.derivation_family_fingerprint,
            "target_proof_sha256": _sha256_text(variant.proof_text),
        }
        tasks.append(
            {
                **shared,
                "task_id": f"{record.statement_id}:whole",
                "interface": "whole",
                "boundary_id": None,
                "model_input": f"{record.canonical_declaration} := ",
                "proof_prefix": "",
                "target": variant.proof_text,
            }
        )
        tasks.append(
            {
                **shared,
                "task_id": f"{record.statement_id}:incremental",
                "interface": "incremental",
                "boundary_id": boundary.boundary_id,
                "model_input": f"{record.canonical_declaration} := {prefix}",
                "proof_prefix": prefix,
                "target": continuation,
            }
        )
    for task in tasks:
        task["model_input_sha256"] = _sha256_text(task["model_input"])
        task["target_sha256"] = _sha256_text(task["target"])
    value = {
        "schema_version": "generalist-v3-canary-manifest-v1",
        "role": role,
        "sealed": role == "test",
        "dataset_binding": binding.to_dict(),
        "boundary_rule": config.evaluation["incremental_boundary_rule"],
        "theorem_count": len(records),
        "interface_task_count": len(tasks),
        "interface_counts": dict(Counter(item["interface"] for item in tasks)),
        "ordered_tasks_sha256": _sha256_json(tasks),
        "sampling": {
            "candidates_per_task": config.evaluation["candidates_per_task"],
            **config.evaluation["sampling"],
        },
        "tasks": tasks,
    }
    _write_json(output_path, value)
    return value


def anchor_schedule(anchor_count: int, steps: int, *, seed: int = 0) -> tuple[int, ...]:
    if anchor_count != 512 or steps < 1 or seed != 0:
        raise ValueError("generalist-v3 anchor schedule contract differs")
    rng = random.Random(seed)
    result: list[int] = []
    while len(result) < steps:
        cycle = list(range(anchor_count))
        rng.shuffle(cycle)
        result.extend(cycle)
    return tuple(result[:steps])


def normalized_template_hash(candidate: str) -> str:
    return _sha256_text(normalized_proof_structure(normalize_transport(candidate)))


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_canary_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    expected_task_ids: Sequence[str],
) -> dict[str, Any]:
    expected = set(expected_task_ids)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["task_id"])].append(candidate)
    if set(grouped) != expected or any(len(values) != 8 for values in grouped.values()):
        raise ValueError("generalist-v3 canary candidate membership is incomplete")
    interface_groups: dict[str, list[str]] = defaultdict(list)
    for task_id in expected_task_ids:
        interface_groups[task_id.rsplit(":", 1)[1]].append(task_id)

    def lane(task_ids: Sequence[str]) -> dict[str, Any]:
        lane_candidates = [item for task_id in task_ids for item in grouped[task_id]]
        verified_counts = [
            sum(item["category"] == "verified" for item in grouped[task_id])
            for task_id in task_ids
        ]
        token_counts = [int(item["generated_token_count"]) for item in lane_candidates]
        template_tasks: dict[str, set[str]] = defaultdict(set)
        template_occurrences = Counter()
        template_verified = Counter()
        for item in lane_candidates:
            template = normalized_template_hash(str(item["candidate_text"]))
            template_tasks[template].add(str(item["task_id"]).rsplit(":", 1)[0])
            template_occurrences[template] += 1
            template_verified[template] += item["category"] == "verified"
        dominant = max(template_occurrences, key=template_occurrences.get)
        return {
            "task_count": len(task_ids),
            "candidate_count": len(lane_candidates),
            "solved_at_8": sum(value > 0 for value in verified_counts),
            "pass_at_1": fmean(pass_at_k(8, value, 1) for value in verified_counts),
            "pass_at_4": fmean(pass_at_k(8, value, 4) for value in verified_counts),
            "pass_at_8": fmean(pass_at_k(8, value, 8) for value in verified_counts),
            "verified_candidates": sum(verified_counts),
            "verified_density": sum(verified_counts) / len(lane_candidates),
            "finish_reason_counts": dict(Counter(str(item["finish_reason"]) for item in lane_candidates)),
            "generated_tokens": {
                "minimum": min(token_counts),
                "p25": _percentile(token_counts, 0.25),
                "median": _percentile(token_counts, 0.5),
                "p75": _percentile(token_counts, 0.75),
                "p95": _percentile(token_counts, 0.95),
                "maximum": max(token_counts),
                "le_64_fraction": sum(value <= 64 for value in token_counts) / len(token_counts),
            },
            "first_construct_counts": dict(Counter(first_proof_construct(str(item["candidate_text"])) for item in lane_candidates)),
            "unique_normalized_templates": len(template_occurrences),
            "normalized_template_diversity": len(template_occurrences) / len(lane_candidates),
            "dominant_template": {
                "sha256": dominant,
                "theorem_count": len(template_tasks[dominant]),
                "occurrences": template_occurrences[dominant],
                "verified_occurrences": template_verified[dominant],
            },
            "verified_counts": verified_counts,
        }

    whole = lane(sorted(interface_groups["whole"]))
    incremental = lane(sorted(interface_groups["incremental"]))
    combined_candidates = whole["candidate_count"] + incremental["candidate_count"]
    combined_verified = whole["verified_candidates"] + incremental["verified_candidates"]
    return {
        "whole": whole,
        "incremental": incremental,
        "combined": {
            "interface_task_count": whole["task_count"] + incremental["task_count"],
            "candidate_count": combined_candidates,
            "solved_at_8": whole["solved_at_8"] + incremental["solved_at_8"],
            "verified_candidates": combined_verified,
            "verified_density": combined_verified / combined_candidates,
            "normalized_template_diversity": (
                whole["unique_normalized_templates"] + incremental["unique_normalized_templates"]
            ) / combined_candidates,
        },
    }


def evaluate_collapse_gates(
    config: GeneralistV3Config,
    summary: Mapping[str, Any],
    base_summary: Mapping[str, Any],
    *,
    retained_base_solved: int,
) -> dict[str, Any]:
    gates = config.collapse_gates
    repeated_lanes = []
    for interface in ("whole", "incremental"):
        lane = summary[interface]
        dominant = lane["dominant_template"]
        theorem_fraction = dominant["theorem_count"] / lane["task_count"]
        verified_fraction = (
            dominant["verified_occurrences"] / dominant["occurrences"]
            if dominant["occurrences"]
            else 0.0
        )
        if (
            theorem_fraction >= gates["repeated_template"]["theorem_fraction_gte"]
            and verified_fraction < gates["repeated_template"]["verified_occurrence_fraction_lt"]
        ):
            repeated_lanes.append(interface)
    whole = summary["whole"]
    whole_candidates = whole["candidate_count"]
    eos_fraction = whole["finish_reason_counts"].get("eos", 0) / whole_candidates
    short_eos = (
        eos_fraction >= gates["short_eos"]["whole_eos_fraction_gte"]
        and whole["generated_tokens"]["le_64_fraction"]
        >= gates["short_eos"]["whole_le_64_token_fraction_gte"]
        and whole["solved_at_8"] <= base_summary["whole"]["solved_at_8"]
    )
    base_solved = int(base_summary["combined"]["solved_at_8"])
    catastrophic_coverage = (
        base_solved >= gates["base_coverage"]["minimum_base_solved_interface_tasks"]
        and retained_base_solved / base_solved < gates["base_coverage"]["retention_fraction_lt"]
        and summary["combined"]["solved_at_8"] < base_solved
    )
    return {
        "repeated_template_collapse": bool(repeated_lanes),
        "repeated_template_interfaces": repeated_lanes,
        "short_eos_collapse": short_eos,
        "catastrophic_base_coverage_loss": catastrophic_coverage,
        "eligible": not repeated_lanes and not short_eos and not catastrophic_coverage,
    }


def positive_500_step_gate(
    config: GeneralistV3Config,
    summary: Mapping[str, Any],
    base_summary: Mapping[str, Any],
    collapse: Mapping[str, Any],
) -> bool:
    if collapse.get("eligible") is not True:
        return False
    thresholds = config.collapse_gates["positive_500_step"]
    solved = int(summary["combined"]["solved_at_8"])
    base_solved = int(base_summary["combined"]["solved_at_8"])
    density = float(summary["combined"]["verified_density"])
    base_density = float(base_summary["combined"]["verified_density"])
    solved_gain = solved >= base_solved + int(thresholds["combined_solved_gain"])
    density_gain = (
        (density >= base_density * (1 + float(thresholds["verified_density_relative_gain"])))
        if base_density > 0
        else density > 0
    ) and solved >= base_solved
    return solved_gain or density_gain


def selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    summary = candidate["summary"]
    return (
        -int(summary["combined"]["solved_at_8"]),
        -int(summary["whole"]["solved_at_8"]),
        -int(candidate["retained_base_solved"]),
        -float(summary["combined"]["verified_density"]),
        -float(summary["combined"]["normalized_template_diversity"]),
        float(candidate["mean_anchor_kl"]),
        int(candidate["optimizer_step"]),
        float(candidate["learning_rate"]),
    )


def select_checkpoint(candidates: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        item
        for item in candidates
        if item.get("configuration_id") != "C0"
        and item.get("collapse_gates", {}).get("eligible") is True
    ]
    if not eligible:
        raise ValueError("no eligible preservation checkpoint exists")
    return min(eligible, key=selection_key)
