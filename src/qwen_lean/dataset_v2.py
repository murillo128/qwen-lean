from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .dataset_v2_contract import proof_variant_id, statement_fingerprint_v2, statement_id
from .dataset_v2_schema import (
    DATASET_V2_SCHEMA_VERSION,
    DatasetV2Record,
    LengthMetadata,
    ProofVariant,
)


ROLES = ("training", "validation", "test")
PRIME_INCLUDED = "included-training"
PRIME_DISPOSITIONS = frozenset(
    {
        PRIME_INCLUDED,
        "not-verified-lean",
        "unproved/knowledge-only",
        "license/use-blocked",
        "placeholder/axiom-policy-blocked",
        "source-integrity-blocked",
        "target-compatibility-blocked",
    }
)


def merge_statement_records(records: Iterable[DatasetV2Record]) -> list[DatasetV2Record]:
    """Collapse equal statements and exact proof text without changing sampling weight."""

    grouped: dict[str, list[DatasetV2Record]] = defaultdict(list)
    for record in records:
        record.validate()
        grouped[record.statement_id].append(record)

    merged: list[DatasetV2Record] = []
    for identity, group in sorted(grouped.items()):
        roles = {item.role for item in group}
        if len(roles) != 1:
            raise ValueError(f"statement {identity} crosses roles: {sorted(roles)}")
        environments = {item.environment.environment_id for item in group}
        if len(environments) != 1:
            raise ValueError(f"statement {identity} crosses target environments")
        provenances = {item.provenance for item in group}
        if "synthetic" in provenances and len(provenances) != 1:
            raise ValueError(f"statement {identity} crosses real and synthetic lanes")

        variants: dict[str, ProofVariant] = {}
        for item in group:
            for variant in item.proof_variants:
                variants.setdefault(variant.proof_fingerprint, variant)
        first = group[0]
        ordered_variants = tuple(
            sorted(variants.values(), key=lambda item: item.proof_variant_id)
        )
        longest_variant = max(
            ordered_variants,
            key=lambda item: len(
                f"{first.canonical_declaration} := {item.canonical_proof}".encode("utf-8")
            ),
        )
        value = replace(
            first,
            provenance=(
                first.provenance if len(provenances) == 1 else "mixed-real"
            ),
            proof_variants=ordered_variants,
            topic_tags=tuple(sorted({tag for item in group for tag in item.topic_tags})),
            memberships=tuple(
                sorted({tag for item in group for tag in item.memberships})
            ),
            length=LengthMetadata(
                declaration_chars=len(first.canonical_declaration),
                proof_chars=len(longest_variant.canonical_proof),
                completion_chars=len(longest_variant.completion),
                declaration_lines=first.canonical_declaration.count("\n") + 1,
                proof_lines=longest_variant.canonical_proof.count("\n") + 1,
                utf8_bytes=len(
                    f"{first.canonical_declaration} := {longest_variant.canonical_proof}".encode(
                        "utf-8"
                    )
                ),
            ),
        )
        value.validate()
        merged.append(value)
    return merged


def validate_record_identity(record: DatasetV2Record) -> None:
    expected_fingerprint = statement_fingerprint_v2(record.canonical_declaration)
    if record.normalized_statement_fingerprint != expected_fingerprint:
        raise ValueError(f"statement fingerprint mismatch for {record.statement_id}")
    expected_id = statement_id(record.canonical_declaration)
    if record.statement_id != expected_id or record.sampling_group_id != expected_id:
        raise ValueError("statement/sampling identity mismatch")
    for variant in record.proof_variants:
        expected_variant = proof_variant_id(record.statement_id, variant.canonical_proof)
        if variant.proof_variant_id != expected_variant:
            raise ValueError(f"proof variant identity mismatch: {variant.proof_variant_id}")


def assign_synthetic_roles(records: Sequence[DatasetV2Record], *, seed: str) -> list[DatasetV2Record]:
    """Assign deterministic 80/10/10 roles at statement/derivation-family group level."""

    groups: dict[tuple[str, str], list[DatasetV2Record]] = defaultdict(list)
    for record in records:
        if record.provenance != "synthetic" or not record.derivation_family_fingerprint:
            raise ValueError("synthetic role assignment received a non-synthetic record")
        groups[(record.statement_id, record.derivation_family_fingerprint)].append(record)

    strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, group in groups.items():
        domains = sorted(
            tag for item in group for tag in item.topic_tags if tag.startswith("domain:")
        )
        domain = domains[0] if domains else "domain:generic"
        structural = str(group[0].structural_class)
        strata[(domain, structural)].append(key)

    assignments: dict[tuple[str, str], str] = {}
    for stratum, keys in sorted(strata.items()):
        reserved = [
            key
            for key in keys
            if str(groups[key][0].generator_family).startswith("final-only:")
        ]
        ordinary = [key for key in keys if key not in set(reserved)]
        for key in reserved:
            assignments[key] = "test"
        ordered = sorted(
            ordinary,
            key=lambda key: hashlib.sha256(
                f"{seed}\0{stratum[0]}\0{stratum[1]}\0{key[0]}\0{key[1]}".encode()
            ).hexdigest(),
        )
        total_size = len(keys)
        validation_count = total_size // 10
        test_count = max(0, total_size // 10 - len(reserved))
        if total_size >= 10:
            validation_count = max(1, validation_count)
            if not reserved:
                test_count = max(1, test_count)
        for index, key in enumerate(ordered):
            if index < validation_count:
                assignments[key] = "validation"
            elif index < validation_count + test_count:
                assignments[key] = "test"
            else:
                assignments[key] = "training"

    assigned = [
        replace(record, role=assignments[(record.statement_id, str(record.derivation_family_fingerprint))])
        for record in records
    ]
    validate_role_isolation(assigned)
    return assigned


def validate_role_isolation(records: Sequence[DatasetV2Record]) -> dict[str, int]:
    statement_roles: dict[str, set[str]] = defaultdict(set)
    family_roles: dict[str, set[str]] = defaultdict(set)
    proof_roles: dict[str, set[str]] = defaultdict(set)
    for record in records:
        statement_roles[record.statement_id].add(record.role)
        if record.derivation_family_fingerprint:
            family_roles[record.derivation_family_fingerprint].add(record.role)
        for variant in record.proof_variants:
            proof_roles[variant.proof_variant_id].add(record.role)
    crossing_statements = [key for key, roles in statement_roles.items() if len(roles) > 1]
    crossing_families = [key for key, roles in family_roles.items() if len(roles) > 1]
    crossing_proofs = [key for key, roles in proof_roles.items() if len(roles) > 1]
    if crossing_statements or crossing_families or crossing_proofs:
        raise ValueError(
            "Dataset-v2 role leakage: "
            f"statements={len(crossing_statements)}, "
            f"families={len(crossing_families)}, proofs={len(crossing_proofs)}"
        )
    return {
        "cross_role_statements": 0,
        "cross_role_derivation_families": 0,
        "cross_role_proofs": 0,
    }


def validate_synthetic_source_resolvability(
    records: Sequence[DatasetV2Record],
) -> dict[str, int]:
    """Require every synthetic source lemma to be canonical optimizer knowledge."""

    training_ids = {
        record.statement_id for record in records if record.role == "training"
    }
    synthetic = [record for record in records if record.provenance == "synthetic"]
    missing_references = [
        (record.statement_id, source_id)
        for record in synthetic
        for source_id in record.source_lemma_ids
        if source_id not in training_ids
    ]
    if missing_references:
        affected = {statement_id for statement_id, _ in missing_references}
        missing = {source_id for _, source_id in missing_references}
        raise ValueError(
            "synthetic source lemma ids do not resolve to canonical training: "
            f"records={len(affected)}, references={len(missing_references)}, "
            f"source_ids={len(missing)}"
        )
    return {
        "synthetic_records": len(synthetic),
        "source_lemma_references": sum(
            len(record.source_lemma_ids) for record in synthetic
        ),
        "missing_source_lemma_references": 0,
        "missing_source_statement_ids": 0,
    }


def plan_training_examples(
    records: Sequence[DatasetV2Record],
    *,
    max_utf8_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Return every training statement once and make context incompatibility explicit."""

    planned: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.statement_id):
        if record.role != "training":
            continue
        status = (
            "fits"
            if max_utf8_bytes is None or record.length.utf8_bytes <= max_utf8_bytes
            else "long-context"
        )
        planned.append(
            {
                "statement_id": record.statement_id,
                "sampling_group_id": record.sampling_group_id,
                "proof_variant_ids": [
                    item.proof_variant_id for item in record.proof_variants
                ],
                "context_status": status,
                "utf8_bytes": record.length.utf8_bytes,
            }
        )
    if len({item["statement_id"] for item in planned}) != len(planned):
        raise ValueError("training planner multiplied a statement by proof variants")
    return planned


def iter_optimizer_examples(
    records: Sequence[DatasetV2Record],
    *,
    variant_seed: str,
    max_utf8_bytes: int | None = None,
) -> Iterator[tuple[DatasetV2Record, ProofVariant]]:
    plan = plan_training_examples(records, max_utf8_bytes=max_utf8_bytes)
    long = [item["statement_id"] for item in plan if item["context_status"] != "fits"]
    if long:
        raise ValueError(
            f"{len(long)} included training statements exceed the configured context; "
            "refusing to truncate or omit them"
        )
    by_id = {record.statement_id: record for record in records}
    for item in plan:
        record = by_id[str(item["statement_id"])]
        variant = min(
            record.proof_variants,
            key=lambda candidate: hashlib.sha256(
                f"{variant_seed}\0{record.statement_id}\0{candidate.proof_variant_id}".encode()
            ).hexdigest(),
        )
        yield record, variant


def validate_prime_coverage(
    entries: Sequence[Mapping[str, Any]],
    records: Sequence[DatasetV2Record],
) -> dict[str, Any]:
    record_ids = {record.statement_id for record in records if record.role == "training"}
    seen: set[str] = set()
    omitted: list[str] = []
    dispositions: Counter[str] = Counter()
    for entry in entries:
        identity = str(entry["coverage_id"])
        if identity in seen:
            raise ValueError(f"duplicate prime coverage identity: {identity}")
        seen.add(identity)
        disposition = str(entry["disposition"])
        if disposition not in PRIME_DISPOSITIONS:
            raise ValueError(f"unknown prime disposition: {disposition}")
        dispositions[disposition] += 1
        eligible = bool(entry.get("verified_lean")) and bool(entry.get("legally_usable")) and bool(
            entry.get("target_compatible")
        )
        if disposition == PRIME_INCLUDED:
            statement_ids = {str(item) for item in entry.get("statement_ids", [])}
            if not statement_ids or not statement_ids <= record_ids:
                raise ValueError(f"prime coverage entry {identity} has invalid training ids")
        elif eligible:
            omitted.append(identity)
    if omitted:
        raise ValueError(
            f"verified/legal/target-compatible prime omissions: {len(omitted)}"
        )
    return {
        "entries": len(entries),
        "dispositions": dict(sorted(dispositions.items())),
        "verified_legal_target_compatible_omissions": 0,
    }


def filter_clean_benchmark(
    benchmark_records: Sequence[Mapping[str, Any]],
    training_records: Sequence[DatasetV2Record],
) -> tuple[list[Mapping[str, Any]], list[dict[str, str]]]:
    training_fingerprints = {
        record.normalized_statement_fingerprint
        for record in training_records
        if record.role == "training"
    }
    retained: list[Mapping[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for task in benchmark_records:
        fingerprint = statement_fingerprint_v2(str(task["declaration"]))
        if fingerprint in training_fingerprints:
            excluded.append(
                {
                    "task_id": str(task["task_id"]),
                    "reason": "normalized-statement-overlap-with-dataset-v2-training",
                    "normalized_statement_fingerprint": fingerprint,
                }
            )
        else:
            retained.append(task)
    return retained, excluded


def write_records(path: Path, records: Sequence[DatasetV2Record]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(handle: Any) -> None:
        for record in sorted(
            records, key=lambda item: (ROLES.index(item.role), item.statement_id)
        ):
            validate_record_identity(record)
            record.validate()
            handle.write(
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
            )
            handle.write("\n")

    if path.suffix == ".gz":
        with path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as binary_handle:
                with io.TextIOWrapper(
                    binary_handle, encoding="utf-8", newline="\n"
                ) as handle:
                    emit(handle)
    else:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            emit(handle)
    return sha256_file(path)


def read_records(path: Path) -> list[DatasetV2Record]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[DatasetV2Record] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = DatasetV2Record.from_dict(json.loads(line))
                validate_record_identity(record)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid Dataset-v2 record at {path}:{line_number}: {error}") from error
            records.append(record)
    validate_role_isolation(records)
    return records


def write_membership_view(
    path: Path,
    records: Sequence[DatasetV2Record],
    statement_ids: Iterable[str],
) -> str:
    """Write a deterministic ID-only view over canonical training records."""

    by_id = {record.statement_id: record for record in records}
    selected_ids = sorted(set(statement_ids))
    missing = [identity for identity in selected_ids if identity not in by_id]
    if missing:
        raise ValueError(f"membership view has {len(missing)} unresolved statement ids")
    non_training = [
        identity for identity in selected_ids if by_id[identity].role != "training"
    ]
    if non_training:
        raise ValueError(
            f"membership view has {len(non_training)} non-training statement ids"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as binary_handle:
            with io.TextIOWrapper(
                binary_handle, encoding="utf-8", newline="\n"
            ) as handle:
                for identity in selected_ids:
                    record = by_id[identity]
                    handle.write(
                        json.dumps(
                            {
                                "proof_variant_ids": [
                                    variant.proof_variant_id
                                    for variant in record.proof_variants
                                ],
                                "statement_id": identity,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    handle.write("\n")
    return sha256_file(path)


def read_membership_view(
    path: Path, records: Sequence[DatasetV2Record]
) -> list[DatasetV2Record]:
    """Resolve and validate an ID-only membership view against its canonical corpus."""

    by_id = {record.statement_id: record for record in records}
    opener = gzip.open if path.suffix == ".gz" else open
    resolved: list[DatasetV2Record] = []
    seen: set[str] = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                identity = str(value["statement_id"])
                if identity in seen:
                    raise ValueError(f"duplicate statement id: {identity}")
                record = by_id[identity]
                if record.role != "training":
                    raise ValueError(f"non-training statement id: {identity}")
                expected_variants = [
                    variant.proof_variant_id for variant in record.proof_variants
                ]
                observed_variants = [
                    str(item) for item in value["proof_variant_ids"]
                ]
                if observed_variants != expected_variants:
                    raise ValueError(f"proof variant ids do not resolve: {identity}")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid Dataset-v2 membership at {path}:{line_number}: {error}"
                ) from error
            seen.add(identity)
            resolved.append(record)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
