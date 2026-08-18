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
from .phase2_corpus import _lex_lean
from .dataset_v2_extraction import ExtractionDiagnostics, SourceCandidate
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
            canonical_declaration=candidate.declaration,
            resolved_dependencies=candidate.resolved_dependencies,
            type_head=_statement_type_head(candidate.declaration),
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


def _statement_type_head(declaration: str) -> str:
    tokens = _lex_lean(declaration)
    depth = 0
    main_colon = None
    for index, (kind, token) in enumerate(tokens):
        if kind == "symbol" and token in "([{⦃":
            depth += 1
        elif kind == "symbol" and token in ")]}⦄":
            depth -= 1
        elif depth == 0 and kind == "symbol" and token == ":":
            main_colon = index
            break
    if main_colon is None:
        return "other"
    depth = 0
    for kind, token in tokens[main_colon + 1 :]:
        if kind == "symbol" and token in "([{⦃":
            depth += 1
        elif kind == "symbol" and token in ")]}⦄":
            depth -= 1
        elif depth == 0 and kind == "symbol":
            if token == "↔":
                return "iff"
            if token == "∧":
                return "and"
            if token == "∨":
                return "or"
            if token == "→":
                return "implication"
            if token == "=":
                return "equality"
    return "other"


def distribute_prime_counts(total: int) -> dict[str, int]:
    quotient, remainder = divmod(total, len(PRIME_FAMILIES))
    return {
        family: quotient + (1 if index < remainder else 0)
        for index, family in enumerate(PRIME_FAMILIES)
    }


def _source_disposition_id(
    repository: str, revision: str, file_path: str, declaration_name: str
) -> str:
    payload = "\0".join(
        (
            "dataset-v2-source-disposition-v1",
            repository,
            revision,
            file_path,
            declaration_name,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_source_dispositions(
    candidates: Sequence[SourceCandidate],
    *,
    diagnostics: ExtractionDiagnostics,
    config: Mapping[str, Any],
    topic_metadata: Mapping[tuple[str, str], Mapping[str, Sequence[str]]],
) -> list[dict[str, Any]]:
    """Account for every extracted or explicitly rejected real-source declaration."""

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in candidates:
        key = (
            candidate.source_repository,
            candidate.source_revision,
            candidate.file_path,
            candidate.declaration_name,
        )
        if key in seen:
            raise ValueError(f"duplicate source disposition identity: {key}")
        seen.add(key)
        accepted = candidate.verification_status == "accepted"
        entries.append(
            {
                "source_disposition_id": _source_disposition_id(*key),
                "source_repository": candidate.source_repository,
                "source_revision": candidate.source_revision,
                "file_path": candidate.file_path,
                "declaration_name": candidate.declaration_name,
                "provenance": candidate.provenance,
                "memberships": list(candidate.memberships),
                "topic_tags": list(candidate.topic_tags),
                "prime_families": list(
                    prime_families_for(
                        file_path=candidate.file_path,
                        declaration_name=candidate.declaration_name,
                        memberships=candidate.memberships,
                        topic_tags=candidate.topic_tags,
                        pnt_plus=candidate.provenance == "external-lean",
                    )
                ),
                "transformation_kind": candidate.transformation_kind,
                "verification_status": candidate.verification_status,
                "verification_method": candidate.verification_method,
                "verification_diagnostic": candidate.verification_diagnostic,
                "disposition": (
                    "included-training" if accepted else "source-integrity-blocked"
                ),
                "reason": (
                    "accepted-canonical-proof"
                    if accepted
                    else "canonical-proof-verification-failed"
                ),
                "statement_ids": [statement_id(candidate.declaration)] if accepted else [],
            }
        )

    environment = config["target_environment"]
    repository = str(environment["mathlib_repository"])
    revision = str(environment["mathlib_revision"])
    for exclusion in diagnostics.exclusions:
        file_path = str(exclusion["file_path"])
        declaration_name = str(exclusion.get("declaration_name", ""))
        key = (repository, revision, file_path, declaration_name)
        if key in seen:
            raise ValueError(f"source exclusion duplicates an extracted candidate: {key}")
        seen.add(key)
        metadata = topic_metadata.get((file_path, declaration_name), {})
        memberships = tuple(str(item) for item in metadata.get("memberships", ()))
        topic_tags = tuple(str(item) for item in metadata.get("topic_tags", ()))
        reason = str(exclusion["reason"])
        disposition = (
            "placeholder/axiom-policy-blocked"
            if reason.startswith("proof-placeholder")
            else "source-integrity-blocked"
        )
        entries.append(
            {
                "source_disposition_id": _source_disposition_id(*key),
                "source_repository": repository,
                "source_revision": revision,
                "file_path": file_path,
                "declaration_name": declaration_name,
                "provenance": "real-mathlib",
                "memberships": list(memberships),
                "topic_tags": list(topic_tags),
                "prime_families": list(
                    prime_families_for(
                        file_path=file_path,
                        declaration_name=declaration_name,
                        memberships=memberships,
                        topic_tags=topic_tags,
                    )
                ),
                "transformation_kind": None,
                "verification_status": "excluded-before-canonical-verification",
                "verification_method": "classified-extraction-exclusion",
                "verification_diagnostic": str(exclusion.get("detail", "")),
                "disposition": disposition,
                "reason": reason,
                "statement_ids": [],
            }
        )
    return sorted(
        entries,
        key=lambda item: (
            str(item["source_repository"]),
            str(item["file_path"]),
            str(item["declaration_name"]),
        ),
    )


def historical_source_crosswalk(
    source_dispositions: Sequence[Mapping[str, Any]],
    *,
    historical_records: Sequence[Mapping[str, Any]],
    membership_inventories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Map historical v1 and Riemann identities to Dataset-v2 dispositions."""

    by_source = {
        (str(item["file_path"]), str(item["declaration_name"])): item
        for item in source_dispositions
        if item.get("declaration_name")
    }

    def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        missing: list[dict[str, str]] = []
        missing_count = 0
        for value in values:
            key = (str(value["file_path"]), str(value["declaration_name"]))
            disposition = by_source.get(key)
            if disposition is None:
                counts["source-integrity-blocked"] += 1
                missing_count += 1
                if len(missing) < 100:
                    missing.append(
                        {
                            "file_path": key[0],
                            "declaration_name": key[1],
                            "reason": "historical-source-identity-not-recovered",
                        }
                    )
            else:
                counts[str(disposition["disposition"])] += 1
        return {
            "records": len(values),
            "dispositions": dict(sorted(counts.items())),
            "missing_source_identities": missing_count,
            "missing_examples": missing,
        }

    return {
        "schema_version": "dataset-v2-historical-crosswalk-v1",
        "mathlib_v1": summarize(historical_records),
        "riemann_inventories": {
            name: summarize(values)
            for name, values in sorted(membership_inventories.items())
        },
    }


def build_prime_coverage_manifest(
    records: Sequence[DatasetV2Record],
    *,
    config: Mapping[str, Any],
    pnt_source_manifest: Mapping[str, Any],
    source_dispositions: Sequence[Mapping[str, Any]] = (),
    atlas_entries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    disposition_by_source = {
        (str(item["file_path"]), str(item["declaration_name"])): item
        for item in source_dispositions
        if item.get("declaration_name")
    }
    for source in source_dispositions:
        families = [str(item) for item in source.get("prime_families", [])]
        if not families or not source.get("declaration_name"):
            continue
        included = source["disposition"] == "included-training"
        entries.append(
            {
                "coverage_id": f"source:{source['source_disposition_id']}",
                "source": source["source_repository"],
                "source_revision": source["source_revision"],
                "declaration_names": [source["declaration_name"]],
                "prime_families": families,
                "verified_lean": included,
                "legally_usable": True,
                "target_compatible": included,
                "disposition": source["disposition"],
                "reason": source["reason"],
                "statement_ids": list(source.get("statement_ids", [])),
            }
        )
    for atlas in atlas_entries:
        formalization = atlas.get("formalization") or {}
        file_path = str(formalization.get("file_path", ""))
        declaration_name = str(formalization.get("declaration_name", ""))
        mapped = disposition_by_source.get((file_path, declaration_name))
        if mapped is not None:
            included = mapped["disposition"] == "included-training"
            disposition = str(mapped["disposition"])
            reason = "mapped-exact-lean-identity"
            statement_ids = list(mapped.get("statement_ids", []))
        else:
            included = False
            statement_ids = []
            if declaration_name:
                disposition = "source-integrity-blocked"
                reason = "atlas-lean-identity-not-recovered"
            elif atlas.get("formalization_status") == "other-prover":
                disposition = "not-verified-lean"
                reason = "formalized-outside-lean"
            else:
                disposition = "unproved/knowledge-only"
                reason = "atlas-entry-has-no-exact-verified-lean-identity"
        entries.append(
            {
                "coverage_id": f"atlas:{atlas['id']}",
                "source": str(formalization.get("repository", "riemann-atlas")),
                "source_revision": str(formalization.get("revision", "")),
                "declaration_names": [declaration_name] if declaration_name else [],
                "prime_families": [],
                "verified_lean": included,
                "legally_usable": True,
                "target_compatible": included,
                "disposition": disposition,
                "reason": reason,
                "statement_ids": statement_ids,
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
