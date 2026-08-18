from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .dataset_v2 import validate_prime_coverage
from .dataset_v2_composition import CompositionSource
from .dataset_v2_contract import statement_id
from .dataset_v2_extraction import SourceCandidate
from .dataset_v2_schema import (
    DATASET_V2_MANIFEST_SCHEMA_VERSION,
    PRIME_COVERAGE_SCHEMA_VERSION,
    DatasetV2Record,
)


PRIME_FAMILIES = (
    "prime-arithmetic-divisibility",
    "arithmetic-functions",
    "prime-counting-pnt",
    "zeta-analytic-number-theory",
    "riemann-core-bubble",
    "pnt-plus",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_riemann_metadata(root: Path) -> dict[tuple[str, str], dict[str, tuple[str, ...]]]:
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    topics: dict[tuple[str, str], set[str]] = defaultdict(set)
    for membership_path in sorted((root / "corpora").glob("*/membership.jsonl")):
        corpus_id = membership_path.parent.name
        for value in read_jsonl(membership_path):
            if "file_path" not in value:
                continue
            key = (str(value["file_path"]), str(value["declaration_name"]))
            memberships[key].add(corpus_id)
            relevance = value.get("relevance_class")
            if relevance:
                topics[key].add(f"riemann-relevance:{relevance}")
    for value in read_jsonl(root / "corpora/records.jsonl.gz"):
        key = (str(value["file_path"]), str(value["declaration_name"]))
        riemann = value.get("riemann", {})
        relevance = riemann.get("relevance_class")
        if relevance:
            topics[key].add(f"riemann-relevance:{relevance}")
        for family in riemann.get("seed_families", []):
            topics[key].add(f"riemann-seed:{family}")
    return {
        key: {
            "memberships": tuple(sorted(values)),
            "topic_tags": tuple(sorted(topics.get(key, set()))),
        }
        for key, values in memberships.items()
    }


def prime_families_for(
    *,
    file_path: str,
    declaration_name: str,
    memberships: Iterable[str] = (),
    topic_tags: Iterable[str] = (),
    pnt_plus: bool = False,
) -> tuple[str, ...]:
    if pnt_plus:
        return ("pnt-plus",)
    haystack = " ".join(
        [file_path, declaration_name, *memberships, *topic_tags]
    ).lower()
    families: set[str] = set()
    if any(word in haystack for word in ("divisib", "prime", "factorization")):
        families.add("prime-arithmetic-divisibility")
    if any(word in haystack for word in ("arithmeticfunction", "arithmetic_function", "moebius", "mangoldt")):
        families.add("arithmetic-functions")
    if any(word in haystack for word in ("primecount", "prime_count", "chebyshev", "prime-number-theorem", "pnt")):
        families.add("prime-counting-pnt")
    if any(word in haystack for word in ("zeta", "lseries", "l-function", "dirichletseries", "analytic-number")):
        families.add("zeta-analytic-number-theory")
    if any(
        membership in {"riemann-core-v1", "riemann-bubble-v1"}
        for membership in memberships
    ):
        families.add("riemann-core-bubble")
    return tuple(sorted(families))


def annotate_candidate(candidate: SourceCandidate) -> SourceCandidate:
    families = prime_families_for(
        file_path=candidate.file_path,
        declaration_name=candidate.declaration_name,
        memberships=candidate.memberships,
        topic_tags=candidate.topic_tags,
        pnt_plus="PrimeNumberTheoremAnd" in candidate.source_repository,
    )
    tags = set(candidate.topic_tags)
    tags.add("domain:generic")
    for family in families:
        tags.add("domain:prime-number-theory")
        tags.add(f"prime-family:{family}")
    return replace(candidate, topic_tags=tuple(sorted(tags)))


def composition_pools(
    candidates: Sequence[SourceCandidate],
) -> dict[str, list[CompositionSource]]:
    pools: dict[str, dict[str, CompositionSource]] = defaultdict(dict)
    for candidate in candidates:
        source = CompositionSource(
            statement_id=statement_id(candidate.declaration),
            declaration_name=candidate.declaration_name,
            source_module=candidate.module,
            topic_tags=candidate.topic_tags,
            domain_family="generic",
        )
        if not any(tag == "prime-family:pnt-plus" for tag in candidate.topic_tags):
            pools["generic"].setdefault(candidate.declaration_name, source)
        for tag in candidate.topic_tags:
            if not tag.startswith("prime-family:"):
                continue
            family = tag.removeprefix("prime-family:")
            pools[family].setdefault(
                candidate.declaration_name, replace(source, domain_family=family)
            )
    return {
        family: sorted(values.values(), key=lambda item: item.declaration_name)
        for family, values in pools.items()
    }


def distribute_prime_counts(total: int) -> dict[str, int]:
    quotient, remainder = divmod(total, len(PRIME_FAMILIES))
    return {
        family: quotient + (1 if index < remainder else 0)
        for index, family in enumerate(PRIME_FAMILIES)
    }


def build_prime_coverage_manifest(
    records: Sequence[DatasetV2Record],
    *,
    config: Mapping[str, Any],
    pnt_source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.statement_id):
        if record.role != "training" or record.provenance == "synthetic":
            continue
        families = sorted(
            tag.removeprefix("prime-family:")
            for tag in record.topic_tags
            if tag.startswith("prime-family:")
        )
        if not families:
            continue
        entries.append(
            {
                "coverage_id": f"statement:{record.statement_id}",
                "source": record.environment.repository,
                "source_revision": record.environment.revision,
                "declaration_names": sorted(
                    {item.source_declaration_name for item in record.proof_variants}
                ),
                "prime_families": families,
                "verified_lean": True,
                "legally_usable": True,
                "target_compatible": True,
                "disposition": "included-training",
                "statement_ids": [record.statement_id],
            }
        )
    for project in pnt_source_manifest.get("projects", []):
        repository = str(project["repository"])
        revision = str(project["revision"])
        for module in project.get("module_inventory", []):
            status = str(module["status"])
            if status == "accepted-records":
                continue
            disposition = (
                "placeholder/axiom-policy-blocked"
                if "sorry" in status or "no-accepted" in status
                else "source-integrity-blocked"
            )
            entries.append(
                {
                    "coverage_id": f"pnt-module:{module['path']}",
                    "source": repository,
                    "source_revision": revision,
                    "verified_lean": False,
                    "legally_usable": True,
                    "target_compatible": True,
                    "disposition": disposition,
                    "reason": status,
                    "statement_ids": [],
                }
            )
    for candidate in config["external_refresh"]["candidates"]:
        entries.append(
            {
                "coverage_id": "external-refresh:"
                + hashlib.sha256(
                    f"{candidate['repository']}\0{candidate['revision']}".encode()
                ).hexdigest(),
                "source": candidate["repository"],
                "source_revision": candidate["revision"],
                "verified_lean": False,
                "legally_usable": bool(candidate.get("license")),
                "target_compatible": False,
                "disposition": candidate["disposition"],
                "reason": candidate["reason"],
                "statement_ids": [],
            }
        )
    summary = validate_prime_coverage(entries, records)
    return {
        "schema_version": PRIME_COVERAGE_SCHEMA_VERSION,
        "dataset_id": config["dataset"]["id"],
        "entries": entries,
        "summary": summary,
    }


def select_train_probe(
    records: Sequence[DatasetV2Record], *, per_stratum: int, seed: str
) -> dict[str, list[str]]:
    strata: dict[str, list[DatasetV2Record]] = defaultdict(list)
    for record in records:
        if record.role != "training":
            continue
        prime = any(tag.startswith("prime-family:") for tag in record.topic_tags)
        if record.provenance == "synthetic":
            key = "synthetic-prime-composition" if prime else "synthetic-generic-composition"
        else:
            key = "real-prime-number-theory" if prime else "real-generic"
        strata[key].append(record)
    selected: dict[str, list[str]] = {}
    for key in (
        "real-generic",
        "real-prime-number-theory",
        "synthetic-generic-composition",
        "synthetic-prime-composition",
    ):
        values = strata.get(key, [])
        if len(values) < per_stratum:
            raise ValueError(f"train probe stratum {key} needs {per_stratum}, found {len(values)}")
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(
                f"{seed}\0{key}\0{item.statement_id}".encode()
            ).hexdigest(),
        )
        selected[key] = [item.statement_id for item in ordered[:per_stratum]]
    return selected


def summarize_records(records: Sequence[DatasetV2Record]) -> dict[str, Any]:
    return {
        "records": len(records),
        "statements": len({item.statement_id for item in records}),
        "proof_variants": sum(len(item.proof_variants) for item in records),
        "roles": dict(sorted(Counter(item.role for item in records).items())),
        "provenance": dict(sorted(Counter(item.provenance for item in records).items())),
        "transformations": dict(
            sorted(
                Counter(
                    variant.transformation_kind
                    for item in records
                    for variant in item.proof_variants
                ).items()
            )
        ),
    }


def dataset_manifest(
    records: Sequence[DatasetV2Record],
    *,
    config: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_V2_MANIFEST_SCHEMA_VERSION,
        "dataset_id": config["dataset"]["id"],
        "target_environment": config["target_environment"],
        "summary": summarize_records(records),
        "identity_contract": {
            "sampling_unit": "normalized-statement",
            "proof_variant_policy": "retain variants; one deterministic variant per optimizer encounter",
            "alpha_and_declaration_name_normalization": True,
        },
        "retention_contract": {
            "silent_truncation": False,
            "silent_omission": False,
            "long_context_status_recorded": True,
        },
        "validation": dict(validation),
        "files": {name: dict(value) for name, value in sorted(files.items())},
    }


def json_ready(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value
