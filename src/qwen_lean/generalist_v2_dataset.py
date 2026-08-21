from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .dataset_v2 import sha256_file
from .dataset_v2_schema import (
    DATASET_V2_MANIFEST_SCHEMA_VERSION,
    DatasetV2Record,
)
from .generalist_v2 import (
    GENERALIST_SERIALIZATION_ID,
    MODEL_ID,
    MODEL_REVISION,
    GeneralistProofVariant,
    GeneralistV2Config,
    compute_training_weights,
    length_distribution,
    materialize_fresh_riemann_views,
    one_pass_trajectory,
    render_generalist_prompt,
    select_context_length,
)
from .prompt import normalize_transport

DATASET_BINDING_SCHEMA_VERSION = "generalist-v2-dataset-binding-v1"
EXPECTED_CANONICAL_COUNTS = {
    "statements": 182_352,
    "proof_variants": 183_633,
    "roles": {"training": 181_531, "validation": 406, "test": 415},
    "provenance": {
        "real-mathlib": 178_234,
        "external-lean": 22,
        "synthetic": 4_096,
    },
}
EXPECTED_TRAINING_COUNTS = {
    "statements": 181_531,
    "proof_variants": 182_812,
    "provenance": {
        "real-mathlib": 178_234,
        "external-lean": 22,
        "synthetic": 3_275,
    },
}
EXPECTED_SYNTHETIC_MULTIPLIER = 4.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSON row at {path}:{line_number}")
            yield value


def validate_canonical_package(package_root: Path) -> dict[str, Any]:
    root = package_root.resolve()
    manifest_path = root / "manifest.json"
    training_views_path = root / "training-views.json"
    manifest = _read_json(manifest_path)
    training_views = _read_json(training_views_path)
    if manifest.get("schema_version") != DATASET_V2_MANIFEST_SCHEMA_VERSION:
        raise ValueError("generalist-v2 received an unknown Dataset-v2 manifest")
    if manifest.get("dataset_id") != "lean-whole-proof-v2":
        raise ValueError("generalist-v2 received the wrong Dataset-v2 package")
    summary = manifest.get("summary", {})
    observed_summary = {
        key: summary.get(key) for key in EXPECTED_CANONICAL_COUNTS
    }
    if observed_summary != EXPECTED_CANONICAL_COUNTS:
        raise ValueError("canonical Dataset-v2 counts differ from issue #78")
    if training_views.get("dataset_id") != "lean-whole-proof-v2":
        raise ValueError("Dataset-v2 training views name a different package")

    files: dict[str, Any] = {}
    for name, expected in sorted(manifest["files"].items()):
        path = root / name
        if not path.is_file():
            raise ValueError(f"canonical Dataset-v2 file is missing: {name}")
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed != expected:
            raise ValueError(
                f"canonical Dataset-v2 file identity differs for {name}: {observed}"
            )
        files[name] = observed

    general = training_views["views"]["general-train-v2"]
    expected_general = {
        **EXPECTED_TRAINING_COUNTS,
        "file": "general-train-v2.jsonl.gz",
        "bytes": files["general-train-v2.jsonl.gz"]["bytes"],
        "sha256": files["general-train-v2.jsonl.gz"]["sha256"],
    }
    if general != expected_general:
        raise ValueError("general-train-v2 identity/counts differ from issue #78")
    validation = manifest.get("validation", {})
    if validation.get("role_isolation") != {
        "cross_role_derivation_families": 0,
        "cross_role_proofs": 0,
        "cross_role_statements": 0,
    }:
        raise ValueError("canonical Dataset-v2 role-isolation evidence is not clean")
    return {
        "repository_snapshot": f"data/{root.name}",
        "manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "target_environment": manifest["target_environment"],
        "canonical_counts": manifest["summary"],
        "general_train": general,
        "accepted_validation": validation,
    }


def read_training_membership(path: Path) -> dict[str, tuple[str, ...]]:
    membership: dict[str, tuple[str, ...]] = {}
    seen_variants: set[str] = set()
    for value in _iter_jsonl(path):
        statement_id = str(value["statement_id"])
        variants = tuple(str(item) for item in value["proof_variant_ids"])
        if statement_id in membership:
            raise ValueError(f"general-train-v2 repeats statement {statement_id}")
        if not variants or len(set(variants)) != len(variants):
            raise ValueError(f"general-train-v2 has invalid variants for {statement_id}")
        duplicate_variants = seen_variants.intersection(variants)
        if duplicate_variants:
            raise ValueError("general-train-v2 repeats proof variants across statements")
        membership[statement_id] = variants
        seen_variants.update(variants)
    return membership


def dataset_record_preamble(record: DatasetV2Record) -> str:
    imports = tuple(dict.fromkeys(record.environment.imports))
    if not imports:
        raise ValueError(f"Dataset-v2 record has no persisted imports: {record.statement_id}")
    return "\n".join(f"import {module}" for module in imports)


def generalist_variants(record: DatasetV2Record) -> tuple[GeneralistProofVariant, ...]:
    split = "train" if record.role == "training" else record.role
    source_kind = "synthetic" if record.provenance == "synthetic" else "real"
    preamble = dataset_record_preamble(record)
    variants = tuple(
        GeneralistProofVariant(
            statement_id=record.statement_id,
            proof_variant_id=variant.proof_variant_id,
            declaration_name=variant.source_declaration_name,
            declaration=record.canonical_declaration,
            completion=variant.completion,
            preamble=preamble,
            split=split,
            optimizer_eligible=record.role == "training",
            source_kind=source_kind,
            generator_family=record.generator_family,
            composition_class=record.structural_class,
            derivation_family_id=record.derivation_family_fingerprint,
            domain_tags=record.topic_tags,
        )
        for variant in record.proof_variants
    )
    for variant in variants:
        variant.validate()
    return variants


def load_bound_training_variants(package_root: Path) -> list[GeneralistProofVariant]:
    """Load the exact canonical ``general-train-v2`` proof-variant membership."""
    validate_canonical_package(package_root)
    root = package_root.resolve()
    membership = read_training_membership(root / "general-train-v2.jsonl.gz")
    remaining = dict(membership)
    training: list[GeneralistProofVariant] = []
    statement_ids: set[str] = set()
    variant_ids: set[str] = set()
    provenance: Counter[str] = Counter()

    for value in _iter_jsonl(root / "records.jsonl.gz"):
        record = DatasetV2Record.from_dict(value)
        if record.role != "training":
            if record.statement_id in membership:
                raise ValueError("validation/test statement is optimizer-visible")
            continue
        variants = generalist_variants(record)
        observed_ids = tuple(item.proof_variant_id for item in variants)
        expected_ids = remaining.pop(record.statement_id, None)
        if expected_ids is None:
            raise ValueError(
                "training statement is absent from general-train-v2: "
                f"{record.statement_id}"
            )
        if observed_ids != expected_ids:
            raise ValueError(f"proof variants do not resolve for {record.statement_id}")
        if variant_ids.intersection(observed_ids):
            raise ValueError("general-train-v2 repeats a proof variant")
        training.extend(variants)
        statement_ids.add(record.statement_id)
        variant_ids.update(observed_ids)
        provenance[record.provenance] += 1

    if remaining:
        raise ValueError(f"general-train-v2 has {len(remaining)} unresolved statements")
    observed = {
        "statements": len(statement_ids),
        "proof_variants": len(training),
        "provenance": dict(sorted(provenance.items())),
    }
    if observed != EXPECTED_TRAINING_COUNTS:
        raise ValueError("resolved general-train-v2 counts differ from issue #78")
    return training


def _write_id_view(path: Path, statement_ids: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(statement_ids)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError(f"ID-only view is empty or duplicated: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for statement_id in ordered:
            handle.write(json.dumps({"statement_id": statement_id}, sort_keys=True))
            handle.write("\n")
    return {
        "file": path.name,
        "task_count": len(ordered),
        "sha256": sha256_file(path),
    }


def _read_train_probe(path: Path, training_statement_ids: set[str]) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("id") != "dataset-v2-train-probe":
        raise ValueError("Dataset-v2 train probe has the wrong identity")
    strata = value.get("strata")
    if not isinstance(strata, dict):
        raise TypeError("Dataset-v2 train probe has no strata")
    flattened = [str(item) for rows in strata.values() for item in rows]
    if len(flattened) != 256 or len(set(flattened)) != 256:
        raise ValueError("Dataset-v2 train probe must contain 256 unique statements")
    missing = set(flattened) - training_statement_ids
    if missing:
        raise ValueError("Dataset-v2 train probe references non-training statements")
    counts = {str(name): len(rows) for name, rows in sorted(strata.items())}
    if set(counts.values()) != {64}:
        raise ValueError("Dataset-v2 train probe strata must each contain 64 statements")
    return {
        "id": value["id"],
        "sha256": sha256_file(path),
        "statement_count": len(flattened),
        "strata": counts,
    }


def _load_tokenizer(model_snapshot: Path | None) -> tuple[Any, dict[str, Any]]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 tokenizer census needs its isolated runtime"
        ) from error
    if model_snapshot is None:
        source = MODEL_ID
        source_kwargs: dict[str, Any] = {
            "revision": MODEL_REVISION,
            "local_files_only": True,
        }
    else:
        snapshot = model_snapshot.resolve()
        if snapshot.name != MODEL_REVISION:
            raise ValueError("tokenizer snapshot is not the pinned Qwen3.5 revision")
        if not (snapshot / "tokenizer.json").is_file():
            raise ValueError("pinned Qwen3.5 tokenizer snapshot is incomplete")
        source = str(snapshot)
        source_kwargs = {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=False, **source_kwargs
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("pinned Qwen3.5 tokenizer has no EOS token")
    return tokenizer, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": (
            None if tokenizer.pad_token_id is None else int(tokenizer.pad_token_id)
        ),
        "add_special_tokens": False,
        "chat_template_applied": False,
        "local_files_only": True,
    }


def tokenizer_length_census(
    records: Sequence[GeneralistProofVariant],
    *,
    model_snapshot: Path | None,
    batch_size: int = 512,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("tokenizer census batch size must be positive")
    tokenizer, identity = _load_tokenizer(model_snapshot)
    eos_token_id = int(tokenizer.eos_token_id)
    full_lengths: list[int] = []
    completion_lengths: list[int] = []
    maximum_variant: dict[str, Any] | None = None
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        prompts = [render_generalist_prompt(item) for item in batch]
        completions = [normalize_transport(item.completion) for item in batch]
        prompt_ids = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        completion_ids = tokenizer(
            completions,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        for record, prompt, completion in zip(
            batch, prompt_ids, completion_ids, strict=True
        ):
            if eos_token_id in prompt or eos_token_id in completion:
                raise ValueError(
                    f"serialized variant contains in-band EOS: {record.proof_variant_id}"
                )
            full_length = len(prompt) + len(completion) + 1
            full_lengths.append(full_length)
            completion_lengths.append(len(completion))
            if maximum_variant is None or full_length > maximum_variant["tokens"]:
                maximum_variant = {
                    "statement_id": record.statement_id,
                    "proof_variant_id": record.proof_variant_id,
                    "tokens": full_length,
                }
    context = select_context_length(full_lengths)
    context["completion_tokens_excluding_eos"] = length_distribution(
        completion_lengths
    )
    context["maximum_variant"] = maximum_variant
    context["serialization_id"] = GENERALIST_SERIALIZATION_ID
    return context, identity


def _digest_weight_map(values: Mapping[str, float]) -> str:
    digest = hashlib.sha256()
    for identity, value in sorted(values.items()):
        digest.update(f"{identity}\0{value:.17g}\n".encode())
    return digest.hexdigest()


def bind_dataset_v2(
    config: GeneralistV2Config,
    package_root: Path,
    view_dir: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    package = validate_canonical_package(package_root)
    root = package_root.resolve()
    membership = read_training_membership(root / "general-train-v2.jsonl.gz")
    membership_remaining = dict(membership)
    training: list[GeneralistProofVariant] = []
    fresh: list[GeneralistProofVariant] = []
    training_statement_ids: set[str] = set()
    training_variant_ids: set[str] = set()
    evaluation_variant_ids: set[str] = set()
    training_derivation_families: set[str] = set()
    role_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    training_provenance: Counter[str] = Counter()
    structural_counts: Counter[str] = Counter()
    generator_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    canonical_variant_count = 0

    for value in _iter_jsonl(root / "records.jsonl.gz"):
        record = DatasetV2Record.from_dict(value)
        variants = generalist_variants(record)
        role_counts[record.role] += 1
        provenance_counts[record.provenance] += 1
        topic_counts.update(record.topic_tags)
        canonical_variant_count += len(variants)
        variant_ids = tuple(item.proof_variant_id for item in variants)
        if record.role == "training":
            expected = membership_remaining.pop(record.statement_id, None)
            if expected is None:
                raise ValueError(
                    f"training statement is absent from general-train-v2: {record.statement_id}"
                )
            if variant_ids != expected:
                raise ValueError(
                    f"proof variants do not resolve for {record.statement_id}"
                )
            training.extend(variants)
            training_statement_ids.add(record.statement_id)
            training_variant_ids.update(variant_ids)
            training_provenance[record.provenance] += 1
            if record.derivation_family_fingerprint is not None:
                training_derivation_families.add(
                    record.derivation_family_fingerprint
                )
        else:
            if record.statement_id in membership:
                raise ValueError("validation/test statement is optimizer-visible")
            evaluation_variant_ids.update(variant_ids)
            if record.provenance == "synthetic":
                fresh.append(variants[0])
        if record.provenance == "synthetic":
            structural_counts[str(record.structural_class)] += 1
            generator_counts[str(record.generator_family)] += 1

    if membership_remaining:
        raise ValueError(
            f"general-train-v2 has {len(membership_remaining)} unresolved statements"
        )
    if training_variant_ids.intersection(evaluation_variant_ids):
        raise ValueError("validation/test proof variants leak into optimizer membership")
    observed_canonical = {
        "statements": sum(role_counts.values()),
        "proof_variants": canonical_variant_count,
        "roles": dict(sorted(role_counts.items())),
        "provenance": dict(sorted(provenance_counts.items())),
    }
    if observed_canonical != EXPECTED_CANONICAL_COUNTS:
        raise ValueError("resolved canonical Dataset-v2 counts differ from issue #78")
    observed_training = {
        "statements": len(training_statement_ids),
        "proof_variants": len(training),
        "provenance": dict(sorted(training_provenance.items())),
    }
    if observed_training != EXPECTED_TRAINING_COUNTS:
        raise ValueError("resolved general-train-v2 counts differ from issue #78")

    weights = compute_training_weights(
        training,
        target_synthetic_fraction=float(
            config.weighting["synthetic_target_mass_fraction"]
        ),
        maximum_synthetic_statement_weight=float(
            config.weighting["synthetic_max_statement_multiplier"]
        ),
    )
    if weights.synthetic_base_multiplier != EXPECTED_SYNTHETIC_MULTIPLIER:
        raise ValueError("final synthetic multiplier differs from the frozen 4x cap")
    expected_mass_fraction = (3_275 * 4) / (178_256 + 3_275 * 4)
    if not math.isclose(
        weights.synthetic_mass_fraction,
        expected_mass_fraction,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("final synthetic mass differs from the issue #78 amendment")

    fresh_valid = sorted(
        item.statement_id for item in fresh if item.split == "validation"
    )
    fresh_test = sorted(item.statement_id for item in fresh if item.split == "test")
    if len(fresh_valid) != 406 or len(fresh_test) != 415:
        raise ValueError("fresh-composition views differ from canonical role counts")
    riemann = materialize_fresh_riemann_views(
        fresh,
        training_statement_ids=training_statement_ids,
        training_derivation_family_ids=training_derivation_families,
    )
    views = {
        "fresh-composition-valid-v2": _write_id_view(
            view_dir / "fresh-composition-valid-v2.jsonl", fresh_valid
        ),
        "fresh-composition-test-v2": _write_id_view(
            view_dir / "fresh-composition-test-v2.jsonl", fresh_test
        ),
    }
    for view_id, view in riemann["views"].items():
        views[view_id] = _write_id_view(
            view_dir / f"{view_id}.jsonl", view["statement_ids"]
        )

    train_probe = _read_train_probe(
        root / "train-probe.json", training_statement_ids
    )
    context, tokenizer = tokenizer_length_census(
        training, model_snapshot=model_snapshot
    )
    trajectory = one_pass_trajectory(training)
    return {
        "schema_version": DATASET_BINDING_SCHEMA_VERSION,
        "artifact_id": config.value["artifact_id"],
        "dataset": package,
        "resolved": {
            "canonical": observed_canonical,
            "general_train": observed_training,
            "optimizer_membership_exactly_once": len(training_variant_ids)
            == len(training),
            "unresolved_training_statements": 0,
            "validation_test_optimizer_variant_overlap": 0,
            "fresh_composition": {
                "validation_statements": len(fresh_valid),
                "test_statements": len(fresh_test),
            },
            "synthetic_structural_classes": dict(sorted(structural_counts.items())),
            "synthetic_generator_families": dict(sorted(generator_counts.items())),
            "topic_tags": dict(sorted(topic_counts.items())),
        },
        "serialization": {
            "id": GENERALIST_SERIALIZATION_ID,
            "tokenizer": tokenizer,
            "lengths": context,
            "prompt_supervised": False,
            "completion_and_one_eos_supervised": True,
            "packing": False,
            "truncation": False,
        },
        "weights": {
            "real_statement_count": weights.real_statement_count,
            "synthetic_statement_count": weights.synthetic_statement_count,
            "synthetic_base_multiplier": weights.synthetic_base_multiplier,
            "real_statement_mass": weights.real_mass,
            "synthetic_statement_mass": weights.synthetic_mass,
            "synthetic_mass_fraction": weights.synthetic_mass_fraction,
            "maximum_statement_weight": weights.maximum_statement_weight,
            "variant_weight_count": len(weights.variant_weights),
            "statement_weight_sha256": _digest_weight_map(weights.statement_weights),
            "variant_weight_sha256": _digest_weight_map(weights.variant_weights),
            "statement_normalized_variants": True,
            "domain_multipliers": {},
        },
        "trajectory": trajectory,
        "views": views,
        "riemann_fresh_views": riemann,
        "train_probe": train_probe,
    }


def write_dataset_binding_evidence(
    config: GeneralistV2Config,
    package_root: Path,
    view_dir: Path,
    output: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    evidence = bind_dataset_v2(
        config,
        package_root,
        view_dir,
        model_snapshot=model_snapshot,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence
