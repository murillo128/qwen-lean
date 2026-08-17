from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .phase2_corpus import statement_fingerprint, strip_lean_comments
from .phase2_schema import MathlibProofRecord, SPLIT_NAMES

RIEMANN_CONFIG_SCHEMA_VERSION = "riemann-data-config-v1"
RIEMANN_MANIFEST_SCHEMA_VERSION = "riemann-data-manifest-v1"
RIEMANN_ATLAS_CONFIG_SCHEMA_VERSION = "riemann-atlas-config-v1"
RIEMANN_ATLAS_SCHEMA_VERSION = "riemann-theorem-atlas-v1"
RIEMANN_GRAPH_SCHEMA_VERSION = "riemann-internal-graph-v1"
RIEMANN_EXTERNAL_SCHEMA_VERSION = "riemann-external-lean-v1"
PHASE2_SNAPSHOT_SCHEMA_VERSION = "mathlib-whole-proof-repository-snapshot-v1"

RELEVANCE_CLASSES = (
    "core",
    "premise-1",
    "premise-2",
    "user-1",
    "user-2",
    "source-neighborhood",
    "number-theory-control",
)
RELEVANCE_PRIORITY = {value: index for index, value in enumerate(RELEVANCE_CLASSES)}
RELATIONSHIP_CLASSES = {
    "definition/foundation",
    "prerequisite",
    "equivalent-to-RH",
    "RH-implies",
    "implies-RH",
    "conditional-on-RH",
    "partial-progress",
    "analogue/generalization",
    "heuristic/conjectural-neighborhood",
}
FORMALIZATION_STATUSES = {
    "mathlib",
    "external-lean",
    "other-prover",
    "literature-only",
}
ATLAS_EDGE_TYPES = {
    "equivalent-to",
    "implies",
    "consequence-of",
    "prerequisite-for",
    "strengthens",
    "weakens",
    "analogue-generalization-of",
    "formalization-of",
    "Lean-counterpart-of",
    "Lean-component-of",
}
ALLOWED_EXTERNAL_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


@dataclass(frozen=True)
class RiemannDataConfig:
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> RiemannDataConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != RIEMANN_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported Riemann data configuration schema")
        return cls(value=value)


@dataclass(frozen=True)
class RiemannAtlasConfig:
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> RiemannAtlasConfig:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != RIEMANN_ATLAS_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported Riemann atlas configuration schema")
        return cls(value=value)


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: str) -> str:
    return _sha256_bytes("\0".join(parts).encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, pretty=True), encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(_canonical_json(value))
            handle.write("\n")


def _write_gzip_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for value in values:
                zipped.write(_canonical_json(value).encode("utf-8"))
                zipped.write(b"\n")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(percentile * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "p95": _nearest_rank(ordered, 0.95),
        "max": ordered[-1],
    }


def load_phase2_records(
    artifact_dir: Path, config: RiemannDataConfig
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = config.value["phase2_source"]
    if manifest.get("dataset_schema_version") != expected["dataset_schema_version"]:
        raise ValueError("Phase 2 dataset schema differs from the Riemann source contract")
    source = manifest.get("source", {})
    for key in ("repository", "revision", "lean_toolchain"):
        if source.get(key) != expected[key]:
            raise ValueError(f"Phase 2 source {key} differs from the Riemann source contract")

    snapshot = manifest.get("canonical_snapshot")
    if snapshot and snapshot.get("schema_version") != PHASE2_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported canonical Phase 2 snapshot schema")

    records: list[dict[str, Any]] = []
    actual_split_counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        if snapshot:
            identity = snapshot.get("splits", {}).get(split, {})
            path = artifact_dir / str(identity.get("path", ""))
            if not path.is_file():
                raise ValueError(f"canonical Phase 2 split is missing: {split}")
            if path.stat().st_size != int(identity.get("bytes", -1)):
                raise ValueError(f"canonical Phase 2 split size differs: {split}")
            if _sha256_file(path) != identity.get("sha256"):
                raise ValueError(f"canonical Phase 2 split hash differs: {split}")
        else:
            candidates = (
                artifact_dir / f"{split}.jsonl.gz",
                artifact_dir / f"{split}.jsonl",
            )
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
        split_count = 0
        for value in _read_jsonl(path):
            record = MathlibProofRecord.from_dict(value)
            if record.split != split:
                raise ValueError(f"Phase 2 physical split differs for {record.id}")
            if record.source_revision != expected["revision"]:
                raise ValueError(f"Phase 2 record has a mixed source revision: {record.id}")
            records.append(value)
            split_count += 1
        actual_split_counts[split] = split_count
        declared_split_count = manifest.get("splits", {}).get(split, {}).get("records")
        if declared_split_count is not None and int(declared_split_count) != split_count:
            raise ValueError(f"Phase 2 split count differs from its accepted manifest: {split}")
        if snapshot and int(snapshot["splits"][split].get("records", -1)) != split_count:
            raise ValueError(f"canonical Phase 2 split record count differs: {split}")

    declared_count = int(manifest.get("counts", {}).get("final_records", len(records)))
    if declared_count != len(records):
        raise ValueError("Phase 2 record count differs from its accepted manifest")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Phase 2 record identifiers are not unique")
    expected_records = expected.get("expected_records")
    if expected_records is not None and len(records) != int(expected_records):
        raise ValueError("Phase 2 record count differs from the frozen Riemann contract")
    expected_split_counts = expected.get("expected_split_counts", {})
    for split, count in expected_split_counts.items():
        if actual_split_counts.get(split) != int(count):
            raise ValueError(f"Phase 2 split count differs from the frozen contract: {split}")
    for key, value in expected.get("expected_split_hygiene", {}).items():
        if manifest.get("split_hygiene", {}).get(key) != value:
            raise ValueError(f"Phase 2 split hygiene differs from the frozen contract: {key}")
    expected_overlap = expected.get("expected_remaining_exact_statement_matches")
    if (
        expected_overlap is not None
        and manifest.get("contamination", {}).get("remaining_exact_statement_matches")
        != expected_overlap
    ):
        raise ValueError("Phase 2 miniF2F contamination result differs from the frozen contract")
    return manifest, records


def materialize_phase2_snapshot(
    source_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_identities: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        for stale_suffix in (".jsonl", ".jsonl.gz"):
            (output_dir / f"{split}{stale_suffix}").unlink(missing_ok=True)
        path = output_dir / f"{split}.jsonl.gz"
        split_records = [record for record in records if record["split"] == split]
        _write_gzip_jsonl(path, split_records)
        split_identities[split] = {
            "path": path.name,
            "records": len(split_records),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    manifest = json.loads(json.dumps(source_manifest))
    manifest["publication"] = {
        **manifest.get("publication", {}),
        "distribution_license_review_performed": True,
        "repository_snapshot_committed": True,
        "distribution_license": manifest["source"]["license"],
    }
    manifest["canonical_snapshot"] = {
        "schema_version": PHASE2_SNAPSHOT_SCHEMA_VERSION,
        "compression": "gzip-mtime-0-canonical-jsonl",
        "records": len(records),
        "splits": split_identities,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _matching_seed_families(
    record: Mapping[str, Any], families: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    file_path = str(record["file_path"])
    declaration_name = str(record["declaration_name"])
    for family in families:
        if file_path not in set(family["paths"]):
            continue
        patterns = family.get("declaration_patterns", [".*"])
        if any(re.search(str(pattern), declaration_name) for pattern in patterns):
            matches.append(family)
    return matches


def _bounded_seed_traversal(
    seed_ids: Sequence[str],
    adjacency: Mapping[str, set[str]],
    *,
    relation: str,
    max_depth: int,
    max_neighbors: int,
    max_ring_nodes: int,
) -> tuple[dict[str, int], dict[str, set[str]], dict[str, Any]]:
    distances: dict[str, int] = {}
    provenance: dict[str, set[str]] = defaultdict(set)
    truncated_frontiers = 0
    seed_set = set(seed_ids)

    for seed_id in sorted(seed_ids):
        visited = {seed_id}
        frontier = {seed_id}
        for distance in range(1, max_depth + 1):
            next_frontier: set[str] = set()
            for source_id in sorted(frontier):
                neighbors = sorted(adjacency.get(source_id, set()))
                if len(neighbors) > max_neighbors:
                    truncated_frontiers += 1
                for target_id in neighbors[:max_neighbors]:
                    if target_id in visited:
                        continue
                    visited.add(target_id)
                    next_frontier.add(target_id)
                    if target_id not in seed_set:
                        previous = distances.get(target_id)
                        if previous is None or distance < previous:
                            distances[target_id] = distance
                            provenance[target_id] = {seed_id}
                        elif distance == previous:
                            provenance[target_id].add(seed_id)
            frontier = next_frontier
            if not frontier:
                break

    ranked = sorted(distances, key=lambda item: (distances[item], item))
    retained = set(ranked[:max_ring_nodes])
    dropped = set(ranked[max_ring_nodes:])
    for record_id in dropped:
        distances.pop(record_id, None)
        provenance.pop(record_id, None)
    diagnostics = {
        "relation": relation,
        "candidate_nodes_before_global_bound": len(ranked),
        "retained_nodes": len(retained),
        "dropped_by_global_bound": len(dropped),
        "frontiers_truncated_by_degree_bound": truncated_frontiers,
        "max_depth": max_depth,
        "max_neighbors_per_frontier": max_neighbors,
        "max_ring_nodes": max_ring_nodes,
    }
    return distances, provenance, diagnostics


def build_internal_graph(
    records: Sequence[Mapping[str, Any]], config: RiemannDataConfig
) -> dict[str, Any]:
    graph_config = config.value["graph"]
    by_id = {str(record["id"]): record for record in records}
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        name_to_ids[str(record["declaration_name"])].append(str(record["id"]))
    unique_name_to_id = {
        name: ids[0] for name, ids in name_to_ids.items() if len(ids) == 1
    }

    premise_adjacency: dict[str, set[str]] = defaultdict(set)
    user_adjacency: dict[str, set[str]] = defaultdict(set)
    premise_edges: set[tuple[str, str]] = set()
    total_premises = 0
    for record in records:
        source_id = str(record["id"])
        for premise in record["premises"]:
            total_premises += 1
            target_id = unique_name_to_id.get(str(premise))
            if target_id is None or target_id == source_id:
                continue
            premise_edges.add((source_id, target_id))
            premise_adjacency[source_id].add(target_id)
            user_adjacency[target_id].add(source_id)

    source_window = int(graph_config["source_neighbor_window"])
    source_adjacency: dict[str, set[str]] = defaultdict(set)
    source_edges: set[tuple[str, str]] = set()
    by_file: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_file[str(record["file_path"])].append(record)
    for file_records in by_file.values():
        ordered = sorted(
            file_records,
            key=lambda record: (
                int(record["source_span"]["start"]["line"]),
                int(record["source_span"]["start"]["column"]),
                str(record["id"]),
            ),
        )
        for index, record in enumerate(ordered):
            source_id = str(record["id"])
            for target in ordered[index + 1 : index + 1 + source_window]:
                target_id = str(target["id"])
                source_adjacency[source_id].add(target_id)
                source_adjacency[target_id].add(source_id)
                source_edges.add((min(source_id, target_id), max(source_id, target_id)))

    families = graph_config["seed_families"]
    seed_manifest: list[dict[str, Any]] = []
    seed_ids: set[str] = set()
    seed_families: dict[str, list[str]] = defaultdict(list)
    for record in records:
        for family in _matching_seed_families(record, families):
            record_id = str(record["id"])
            seed_ids.add(record_id)
            seed_families[record_id].append(str(family["id"]))
            seed_manifest.append(
                {
                    "record_id": record_id,
                    "declaration_name": record["declaration_name"],
                    "file_path": record["file_path"],
                    "split": record["split"],
                    "family": family["id"],
                    "reason": family["reason"],
                    "rule": {
                        "paths": family["paths"],
                        "declaration_patterns": family["declaration_patterns"],
                    },
                    "atlas_target": family["atlas_target"],
                }
            )
    missing_families = sorted(
        str(family["id"])
        for family in families
        if not any(item["family"] == family["id"] for item in seed_manifest)
    )
    if missing_families:
        raise ValueError(f"seed families retained no declarations: {missing_families}")

    traversal_kwargs = {
        "max_depth": int(graph_config["max_dependency_depth"]),
        "max_neighbors": int(graph_config["max_neighbors_per_frontier"]),
        "max_ring_nodes": int(graph_config["max_ring_nodes_per_direction"]),
    }
    premise_distances, premise_provenance, premise_diagnostics = _bounded_seed_traversal(
        sorted(seed_ids), premise_adjacency, relation="premise", **traversal_kwargs
    )
    user_distances, user_provenance, user_diagnostics = _bounded_seed_traversal(
        sorted(seed_ids), user_adjacency, relation="user", **traversal_kwargs
    )

    source_candidates: dict[str, set[str]] = defaultdict(set)
    for seed_id in sorted(seed_ids):
        for target_id in sorted(source_adjacency.get(seed_id, set())):
            if target_id not in seed_ids:
                source_candidates[target_id].add(seed_id)
    source_ranked = sorted(source_candidates)
    source_limit = int(graph_config["max_source_neighborhood_nodes"])
    source_selected = set(source_ranked[:source_limit])

    classes: dict[str, str] = {}
    for record_id in seed_ids:
        classes[record_id] = "core"
    for record_id, distance in premise_distances.items():
        classes.setdefault(record_id, f"premise-{distance}")
    for record_id, distance in user_distances.items():
        classes.setdefault(record_id, f"user-{distance}")
    for record_id in source_selected:
        classes.setdefault(record_id, "source-neighborhood")
    control_prefix = str(graph_config["number_theory_control_prefix"])
    for record in records:
        if str(record["file_path"]).startswith(control_prefix):
            classes.setdefault(str(record["id"]), "number-theory-control")

    provenance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target_id, seeds in premise_provenance.items():
        for seed_id in sorted(seeds):
            provenance[target_id].append(
                {
                    "seed_record_id": seed_id,
                    "seed_declaration": by_id[seed_id]["declaration_name"],
                    "relation": "premise",
                    "distance": premise_distances[target_id],
                }
            )
    for target_id, seeds in user_provenance.items():
        for seed_id in sorted(seeds):
            provenance[target_id].append(
                {
                    "seed_record_id": seed_id,
                    "seed_declaration": by_id[seed_id]["declaration_name"],
                    "relation": "user",
                    "distance": user_distances[target_id],
                }
            )
    for target_id in sorted(source_selected):
        for seed_id in sorted(source_candidates[target_id]):
            provenance[target_id].append(
                {
                    "seed_record_id": seed_id,
                    "seed_declaration": by_id[seed_id]["declaration_name"],
                    "relation": "same-source",
                    "distance": 1,
                }
            )

    nodes: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["id"])):
        record_id = str(record["id"])
        nodes.append(
            {
                "schema_version": RIEMANN_GRAPH_SCHEMA_VERSION,
                "record_id": record_id,
                "declaration_name": record["declaration_name"],
                "file_path": record["file_path"],
                "source_revision": record["source_revision"],
                "phase2_split": record["split"],
                "phase2_fingerprint": record["statement_fingerprint"],
                "phase2_component": record["component_id"],
                "source_span": record["source_span"],
                "token_lengths": record["token_lengths"],
                "relevance_class": classes.get(record_id),
                "seed_families": sorted(seed_families.get(record_id, [])),
                "provenance": sorted(
                    provenance.get(record_id, []),
                    key=lambda item: (
                        item["relation"],
                        item["distance"],
                        item["seed_record_id"],
                    ),
                ),
                "degrees": {
                    "premise": len(premise_adjacency.get(record_id, set())),
                    "user": len(user_adjacency.get(record_id, set())),
                    "same-source": len(source_adjacency.get(record_id, set())),
                },
            }
        )

    edges: list[tuple[str, str, str]] = []
    for source_id, target_id in sorted(premise_edges):
        edges.append(("premise", source_id, target_id))
        edges.append(("user", target_id, source_id))
    for source_id, target_id in sorted(source_edges):
        edges.append(("same-source", source_id, target_id))
        edges.append(("same-source", target_id, source_id))
    edges.sort()

    class_counts = Counter(classes.values())
    selected_proportion = len(classes) / len(records) if records else 0.0
    max_selected_proportion = float(graph_config.get("max_selected_proportion", 0.5))
    diagnostics = {
        "nodes": len(records),
        "seed_entries": len(seed_manifest),
        "unique_seed_records": len(seed_ids),
        "premises_total": total_premises,
        "premises_resolved_to_retained_nodes": len(premise_edges),
        "premise_coverage": (len(premise_edges) / total_premises if total_premises else 0.0),
        "edge_counts": {
            "premise": len(premise_edges),
            "user": len(premise_edges),
            "same-source": len(source_edges) * 2,
        },
        "class_counts": {value: class_counts.get(value, 0) for value in RELEVANCE_CLASSES},
        "degree_distributions": {
            "premise": _distribution([len(premise_adjacency.get(item, set())) for item in by_id]),
            "user": _distribution([len(user_adjacency.get(item, set())) for item in by_id]),
            "same-source": _distribution(
                [len(source_adjacency.get(item, set())) for item in by_id]
            ),
        },
        "bounds": {
            "premise": premise_diagnostics,
            "user": user_diagnostics,
            "source": {
                "candidate_nodes_before_global_bound": len(source_ranked),
                "retained_nodes": len(source_selected),
                "dropped_by_global_bound": len(source_ranked) - len(source_selected),
                "window": source_window,
                "max_nodes": source_limit,
            },
        },
        "selected_proportion": selected_proportion,
        "max_selected_proportion": max_selected_proportion,
        "most_of_mathlib_guard_passed": selected_proportion <= max_selected_proportion,
    }
    if not diagnostics["most_of_mathlib_guard_passed"]:
        raise ValueError("bounded relevance graph selected at least half of Phase 2")

    return {
        "nodes": nodes,
        "edges": edges,
        "classes": classes,
        "seed_manifest": sorted(
            seed_manifest, key=lambda item: (item["family"], item["declaration_name"])
        ),
        "diagnostics": diagnostics,
        "premise_edges": premise_edges,
    }


def build_specialist_partitions(
    records: Sequence[Mapping[str, Any]], graph: Mapping[str, Any], config: RiemannDataConfig
) -> dict[str, Any]:
    corpus_config = config.value["corpora"]
    classes: Mapping[str, str] = graph["classes"]
    bubble_classes = {
        "core",
        "premise-1",
        "premise-2",
        "user-1",
        "user-2",
        "source-neighborhood",
    }
    control_prefix = str(config.value["graph"]["number_theory_control_prefix"])

    by_component: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_component[(str(record["split"]), str(record["component_id"]))].append(record)

    memberships: dict[str, set[str]] = {
        corpus_config["number_theory_wide"]: set(),
        corpus_config["riemann_bubble"]: set(),
        corpus_config["riemann_core"]: set(),
        corpus_config["specialist_validation"]: set(),
        corpus_config["riemann_near_holdout"]: set(),
        corpus_config["number_theory_far_holdout"]: set(),
    }
    for record in records:
        if record["split"] != "train":
            continue
        record_id = str(record["id"])
        relevance = classes.get(record_id)
        if relevance == "core":
            memberships[corpus_config["riemann_core"]].add(record_id)
        if relevance in bubble_classes:
            memberships[corpus_config["riemann_bubble"]].add(record_id)
        if relevance in bubble_classes or str(record["file_path"]).startswith(control_prefix):
            memberships[corpus_config["number_theory_wide"]].add(record_id)

    for (split, _component), component_records in sorted(by_component.items()):
        component_ids = {str(record["id"]) for record in component_records}
        has_bubble = any(classes.get(record_id) in bubble_classes for record_id in component_ids)
        has_number_theory = any(
            str(record["file_path"]).startswith(control_prefix) for record in component_records
        )
        if split == "validation" and (has_bubble or has_number_theory):
            memberships[corpus_config["specialist_validation"]].update(component_ids)
        elif split == "heldout" and has_bubble:
            memberships[corpus_config["riemann_near_holdout"]].update(component_ids)
        elif split == "heldout" and has_number_theory:
            memberships[corpus_config["number_theory_far_holdout"]].update(component_ids)

    required_nonempty = config.value.get("validation", {}).get("require_nonempty_corpora", True)
    if required_nonempty:
        empty = sorted(name for name, members in memberships.items() if not members)
        if empty:
            raise ValueError(f"specialist corpora are unexpectedly empty: {empty}")

    by_id = {str(record["id"]): record for record in records}
    training_names = {
        corpus_config["number_theory_wide"],
        corpus_config["riemann_bubble"],
        corpus_config["riemann_core"],
    }
    evaluation_names = set(memberships) - training_names
    for name in training_names:
        if any(by_id[record_id]["split"] != "train" for record_id in memberships[name]):
            raise ValueError(f"Phase 2 evaluation record leaked into training corpus {name}")
    for name in evaluation_names:
        if any(by_id[record_id]["split"] == "train" for record_id in memberships[name]):
            raise ValueError(f"Phase 2 train record leaked into evaluation corpus {name}")

    training_union = set().union(*(memberships[name] for name in training_names))
    validation_ids = memberships[corpus_config["specialist_validation"]]
    holdout_ids = memberships[corpus_config["riemann_near_holdout"]] | memberships[
        corpus_config["number_theory_far_holdout"]
    ]

    def values(record_ids: Iterable[str], key: str) -> set[str]:
        return {str(by_id[record_id][key]) for record_id in record_ids}

    forbidden_pairs = {
        "train-validation": (training_union, validation_ids),
        "train-holdout": (training_union, holdout_ids),
        "validation-holdout": (validation_ids, holdout_ids),
        "near-far-holdout": (
            memberships[corpus_config["riemann_near_holdout"]],
            memberships[corpus_config["number_theory_far_holdout"]],
        ),
    }
    leakage: dict[str, dict[str, int]] = {}
    for label, (left, right) in forbidden_pairs.items():
        leakage[label] = {
            "fingerprints": len(values(left, "statement_fingerprint") & values(right, "statement_fingerprint")),
            "components": len(values(left, "component_id") & values(right, "component_id")),
        }
    if any(count for pair in leakage.values() for count in pair.values()):
        raise ValueError(f"forbidden specialist leakage detected: {leakage}")

    crossing = Counter()
    for source_id, target_id in graph["premise_edges"]:
        source_split = str(by_id[source_id]["split"])
        target_split = str(by_id[target_id]["split"])
        if source_split != target_split:
            crossing[f"{source_split}->{target_split}"] += 1

    overlaps: dict[str, int] = {}
    ordered_names = sorted(memberships)
    for index, left in enumerate(ordered_names):
        for right in ordered_names[index + 1 :]:
            count = len(memberships[left] & memberships[right])
            if count:
                overlaps[f"{left}|{right}"] = count

    return {
        "memberships": memberships,
        "training_names": training_names,
        "evaluation_names": evaluation_names,
        "diagnostics": {
            "counts": {name: len(members) for name, members in sorted(memberships.items())},
            "overlaps": overlaps,
            "forbidden_leakage": leakage,
            "dependency_edges_crossing_phase2_splits": dict(sorted(crossing.items())),
            "membership_frozen_without_model_outputs": True,
            "phase2_validation_and_heldout_evaluation_only": True,
            "minif2f_hygiene_inherited_unchanged": True,
        },
    }


_TOP_LEVEL_START = re.compile(
    r"^(?:@\[|blueprint_comment\b|/[-*]|theorem\b|lemma\b|def\b|"
    r"noncomputable\b|private\b|protected\b|namespace\b|section\b|end\b|"
    r"open\b|variable\b|set_option\b|attribute\b|instance\b|#)"
)


def extract_lean_declaration(source: str, local_name: str) -> dict[str, Any]:
    lines = source.splitlines(keepends=True)
    declaration = re.compile(
        rf"^(theorem|lemma)\s+{re.escape(local_name)}(?=\s|:|\(|\{{|\[)"
    )
    starts = [index for index, line in enumerate(lines) if declaration.match(line)]
    if len(starts) != 1:
        raise ValueError(f"expected one source declaration for {local_name}, found {len(starts)}")
    start = starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _TOP_LEVEL_START.match(lines[index]):
            end = index
            break
    block = "".join(lines[start:end]).rstrip()
    proof_match = re.search(r":=\s*(by\b[\s\S]*)$", block)
    if proof_match is None:
        raise ValueError(f"external declaration is not a tactic-style by proof: {local_name}")
    statement = block[: proof_match.start()].rstrip()
    proof = proof_match.group(1).rstrip()
    proof_without_comments = strip_lean_comments(proof)
    if re.search(r"\b(?:sorry|admit)\b", proof_without_comments):
        raise ValueError(f"external declaration contains a proof placeholder: {local_name}")
    return {
        "source_text": block,
        "statement": statement,
        "proof": proof,
        "source_span": {"start_line": start + 1, "end_line": end},
    }


def extract_external_lean_records(
    external_root: Path | None,
    phase2_records: Sequence[Mapping[str, Any]],
    config: RiemannDataConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_projects = config.value["external_discovery"]["projects"]
    accepted_projects = [project for project in source_projects if project["status"] == "accepted"]
    records: list[dict[str, Any]] = []
    phase2_fingerprints = {str(record["statement_fingerprint"]) for record in phase2_records}

    for project in accepted_projects:
        if external_root is None:
            raise ValueError("an external Lean checkout is required for accepted extraction")
        if project.get("require_git_revision", True):
            observed = subprocess.run(
                ["git", "-C", str(external_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if observed != project["revision"]:
                raise ValueError("external Lean checkout differs from its pinned revision")
        if project["license"] not in {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"}:
            raise ValueError("accepted external Lean source lacks a distributable source license")
        native = project["native_verification"]
        if native["status"] != "passed" or native["project_revision"] != project["revision"]:
            raise ValueError("external Lean source lacks exact native-project verification")

        for selected in project["selected_declarations"]:
            path = external_root / selected["file_path"]
            source = path.read_text(encoding="utf-8")
            extracted = extract_lean_declaration(source, selected["local_name"])
            axioms = set(selected["axioms"])
            if not axioms <= ALLOWED_EXTERNAL_AXIOMS:
                raise ValueError(f"external declaration has nonstandard axioms: {selected['full_name']}")
            fingerprint = statement_fingerprint(extracted["statement"])
            record_id = _stable_id(
                RIEMANN_EXTERNAL_SCHEMA_VERSION,
                project["repository"],
                project["revision"],
                selected["file_path"],
                selected["full_name"],
            )
            records.append(
                {
                    "schema_version": RIEMANN_EXTERNAL_SCHEMA_VERSION,
                    "id": record_id,
                    "repository": project["repository"],
                    "revision": project["revision"],
                    "license": project["license"],
                    "lean_toolchain": project["lean_toolchain"],
                    "mathlib_revision": project["mathlib_revision"],
                    "file_path": selected["file_path"],
                    "declaration_name": selected["full_name"],
                    "declaration_kind": extracted["statement"].split(None, 1)[0],
                    "source_span": extracted["source_span"],
                    "statement": extracted["statement"],
                    "proof": extracted["proof"],
                    "source_text": extracted["source_text"],
                    "source_sha256": _sha256_bytes(extracted["source_text"].encode("utf-8")),
                    "statement_fingerprint": fingerprint,
                    "dependencies": [],
                    "dependencies_extracted": False,
                    "native_verification": {
                        "status": "verified",
                        "build_command": native["build_command"],
                        "build_jobs": native["build_jobs"],
                        "checked_on": native["checked_on"],
                        "axiom_audit_command": native["axiom_audit_command"],
                        "axioms": sorted(axioms),
                    },
                    "duplicate_analogue": (
                        "exact-statement-match"
                        if fingerprint in phase2_fingerprints
                        else "source-specific-analogue"
                    ),
                    "topic_tags": sorted(selected["topic_tags"]),
                    "atlas_target": selected["atlas_target"],
                }
            )

    source_manifest = {
        "schema_version": "riemann-source-manifest-v1",
        "snapshot_date": config.value["snapshot_date"],
        "projects": source_projects,
        "accepted_external_records": len(records),
        "accepted_projects": [project["repository"] for project in accepted_projects],
        "discovery_scope": config.value["external_discovery"]["scope_note"],
    }
    return sorted(records, key=lambda item: item["id"]), source_manifest


def _alias_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def build_atlas(
    atlas_config: RiemannAtlasConfig,
    graph: Mapping[str, Any],
    phase2_records: Sequence[Mapping[str, Any]],
    external_records: Sequence[Mapping[str, Any]],
    config: RiemannDataConfig,
) -> dict[str, Any]:
    source_references = {
        str(source["id"]): source for source in atlas_config.value["sources"]
    }
    entries: list[dict[str, Any]] = []
    for raw in atlas_config.value["entries"]:
        entry = dict(raw)
        if entry["relationship_to_rh"]["class"] not in RELATIONSHIP_CLASSES:
            raise ValueError(f"unknown atlas relationship class for {entry['id']}")
        if entry["formalization_status"] not in FORMALIZATION_STATUSES:
            raise ValueError(f"unknown formalization status for {entry['id']}")
        relation_class = entry["relationship_to_rh"]["class"]
        if relation_class in {"equivalent-to-RH", "RH-implies", "implies-RH"}:
            source_ids = entry["relationship_to_rh"].get("source_ids", [])
            if not source_ids:
                raise ValueError(f"directed RH relationship lacks attribution: {entry['id']}")
        for source_id in entry["source_ids"]:
            if source_id not in source_references:
                raise ValueError(f"atlas entry references an unknown source: {source_id}")
        entries.append(entry)

    formalization_status_by_id = {
        str(entry["id"]): str(entry["formalization_status"]) for entry in entries
    }

    def lean_relationship_type(target: str) -> str:
        if formalization_status_by_id.get(target) == "literature-only":
            return "Lean-component-of"
        return "Lean-counterpart-of"

    by_record_id = {str(record["id"]): record for record in phase2_records}
    family_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in graph["seed_manifest"]:
        family_by_record[str(seed["record_id"])].append(seed)
    formal_edges: list[dict[str, Any]] = []
    for record_id, seeds in sorted(family_by_record.items()):
        record = by_record_id[record_id]
        targets = sorted({str(seed["atlas_target"]) for seed in seeds})
        entry_id = f"mathlib-{record_id[:24]}"
        entries.append(
            {
                "id": entry_id,
                "canonical_name": record["declaration_name"],
                "display_name": record["declaration_name"],
                "aliases": [],
                "statement": record["declaration"],
                "authors": [],
                "year": None,
                "source_ids": ["mathlib-phase2"],
                "relationship_to_rh": {
                    "class": "prerequisite",
                    "direction": "entry-to-RH-prerequisite",
                    "source_ids": ["mathlib-phase2"],
                },
                "prerequisites": [],
                "topic_tags": sorted({seed["family"] for seed in seeds}),
                "formalization_status": "mathlib",
                "formalization": {
                    "repository": record["source_repository"],
                    "revision": record["source_revision"],
                    "file_path": record["file_path"],
                    "declaration_name": record["declaration_name"],
                    "phase2_record_id": record_id,
                    "phase2_split": record["split"],
                },
                "verification_status": "verified-in-pinned-mathlib",
                "candidate_role": (
                    "training-source" if record["split"] == "train" else "evaluation-target"
                ),
                "kind": "theorem",
                "notes": "Source-specific pinned mathlib identity; not deduplicated away.",
            }
        )
        for target in targets:
            formal_edges.append(
                {
                    "source": entry_id,
                    "target": target,
                    "type": lean_relationship_type(target),
                    "source_ids": ["mathlib-phase2"],
                }
            )

    for record in external_records:
        entry_id = f"external-lean-{record['id'][:24]}"
        entries.append(
            {
                "id": entry_id,
                "canonical_name": record["declaration_name"],
                "display_name": record["declaration_name"],
                "aliases": [],
                "statement": record["statement"],
                "authors": [],
                "year": 2026,
                "source_ids": ["pnt-plus"],
                "relationship_to_rh": {
                    "class": "prerequisite",
                    "direction": "entry-to-RH-prerequisite",
                    "source_ids": ["pnt-plus"],
                },
                "prerequisites": [],
                "topic_tags": record["topic_tags"],
                "formalization_status": "external-lean",
                "formalization": {
                    "repository": record["repository"],
                    "revision": record["revision"],
                    "file_path": record["file_path"],
                    "declaration_name": record["declaration_name"],
                    "external_record_id": record["id"],
                },
                "verification_status": "verified-in-native-project",
                "candidate_role": "training-source",
                "kind": "theorem",
                "notes": "Separate provenance lane; not merged with Phase 2-derived training data.",
            }
        )
        formal_edges.append(
            {
                "source": entry_id,
                "target": record["atlas_target"],
                "type": lean_relationship_type(str(record["atlas_target"])),
                "source_ids": ["pnt-plus"],
            }
        )

    ids = [str(entry["id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(f"atlas identifiers are not unique: {duplicates}")
    entry_ids = set(ids)

    relationships = [dict(value) for value in atlas_config.value["relationships"]]
    relationships.extend(formal_edges)
    for relationship in relationships:
        if relationship["type"] not in ATLAS_EDGE_TYPES:
            raise ValueError(f"unknown atlas edge type: {relationship['type']}")
        if relationship["source"] not in entry_ids or relationship["target"] not in entry_ids:
            raise ValueError(f"atlas edge has an unknown endpoint: {relationship}")
        if (
            relationship["type"] == "Lean-counterpart-of"
            and formalization_status_by_id[relationship["target"]] == "literature-only"
        ):
            raise ValueError("literature-only atlas entry has a Lean counterpart edge")
        if relationship["type"] in {"equivalent-to", "implies", "consequence-of"}:
            if not relationship.get("source_ids"):
                raise ValueError(f"directed mathematical edge lacks attribution: {relationship}")

    alias_owners: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for alias in [entry["canonical_name"], entry["display_name"], *entry["aliases"]]:
            key = _alias_key(str(alias))
            if key:
                alias_owners[key].add(str(entry["id"]))
    alias_collisions = [
        {"normalized_alias": alias, "entry_ids": sorted(owners)}
        for alias, owners in sorted(alias_owners.items())
        if len(owners) > 1
    ]

    status_counts = Counter(str(entry["formalization_status"]) for entry in entries)
    relation_counts = Counter(str(entry["relationship_to_rh"]["class"]) for entry in entries)
    topic_counts = Counter(
        str(topic) for entry in entries for topic in entry.get("topic_tags", [])
    )
    return {
        "entries": sorted(entries, key=lambda item: str(item["id"])),
        "relationships": sorted(
            relationships,
            key=lambda item: (item["type"], item["source"], item["target"]),
        ),
        "sources": sorted(source_references.values(), key=lambda item: str(item["id"])),
        "alias_collisions": alias_collisions,
        "diagnostics": {
            "entries": len(entries),
            "relationships": len(relationships),
            "by_formalization_status": dict(sorted(status_counts.items())),
            "by_relationship_class": dict(sorted(relation_counts.items())),
            "by_topic": dict(sorted(topic_counts.items())),
            "alias_collisions": len(alias_collisions),
            "census_scope": atlas_config.value["census_scope"],
            "globally_complete": False,
        },
    }


def _membership_rows(
    record_ids: Iterable[str],
    by_id: Mapping[str, Mapping[str, Any]],
    classes: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    for record_id in sorted(record_ids):
        record = by_id[record_id]
        yield {
            "record_id": record_id,
            "phase2_split": record["split"],
            "phase2_component": record["component_id"],
            "phase2_fingerprint": record["statement_fingerprint"],
            "file_path": record["file_path"],
            "declaration_name": record["declaration_name"],
            "relevance_class": classes.get(record_id),
        }


def _token_statistics(record_ids: Iterable[str], by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = [int(by_id[record_id]["token_lengths"]["declaration_and_completion"]) for record_id in record_ids]
    return _distribution(values)


def _rank_opportunities(
    atlas: Mapping[str, Any],
    external_records: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    config: RiemannDataConfig,
) -> dict[str, Any]:
    by_id = {str(record["id"]): record for record in records}
    corpus_config = config.value["corpora"]
    missing = [
        {
            "rank": index + 1,
            "atlas_id": entry["id"],
            "name": entry["display_name"],
            "reason": "Important attributed RH relation without verified Lean source in the census.",
        }
        for index, entry in enumerate(
            sorted(
                (
                    entry
                    for entry in atlas["entries"]
                    if entry["candidate_role"] == "formalize-in-lean"
                    and entry["formalization_status"] != "mathlib"
                ),
                key=lambda item: (
                    item["relationship_to_rh"]["class"] != "equivalent-to-RH",
                    item["id"],
                ),
            )[:20]
        )
    ]
    near_ids = sorted(partitions["memberships"][corpus_config["riemann_near_holdout"]])
    evaluation_targets = [
        {
            "rank": index + 1,
            "record_id": record_id,
            "declaration_name": by_id[record_id]["declaration_name"],
            "file_path": by_id[record_id]["file_path"],
            "reason": "Pinned verified high-relevance Phase 2 heldout theorem.",
        }
        for index, record_id in enumerate(near_ids[:20])
    ]
    knowledge = [
        {
            "rank": index + 1,
            "atlas_id": entry["id"],
            "name": entry["display_name"],
            "reason": "Conjectural, heuristic, or unproved anchor; never a Lean SFT target.",
        }
        for index, entry in enumerate(
            sorted(
                (
                    entry
                    for entry in atlas["entries"]
                    if entry["candidate_role"] == "knowledge-only"
                ),
                key=lambda item: item["id"],
            )
        )
    ]
    external = [
        {
            "rank": index + 1,
            "record_id": record["id"],
            "declaration_name": record["declaration_name"],
            "repository": record["repository"],
            "revision": record["revision"],
            "reason": "Native-verified, licensed, source-complete external Lean declaration.",
        }
        for index, record in enumerate(
            sorted(external_records, key=lambda item: (item["atlas_target"], item["declaration_name"]))
        )
    ]
    return {
        "schema_version": "riemann-opportunity-ranking-v1",
        "non_binding": True,
        "external_lean_ingestion": external,
        "missing_lean_formalizations": missing,
        "future_clean_evaluation_targets": evaluation_targets,
        "knowledge_only_anchors": knowledge,
    }


def _report_markdown(
    manifest: Mapping[str, Any],
    graph: Mapping[str, Any],
    partitions: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    atlas: Mapping[str, Any],
) -> str:
    lines = [
        "# Riemann data/discovery stage evidence",
        "",
        "This is a source-bounded census, not a claim of global completeness or progress on proving RH.",
        "",
        "## Pinned inputs",
        "",
        f"- Phase 2 mathlib revision: `{manifest['phase2_source']['revision']}`",
        f"- Phase 2 records indexed: {graph['diagnostics']['nodes']}",
        f"- External projects inspected: {len(source_manifest['projects'])}",
        f"- Native-verified external Lean records accepted: {source_manifest['accepted_external_records']}",
        "",
        "## Internal graph and corpora",
        "",
    ]
    for relevance, count in graph["diagnostics"]["class_counts"].items():
        lines.append(f"- `{relevance}`: {count}")
    lines.extend(["", "Corpus counts:", ""])
    for name, count in partitions["diagnostics"]["counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## Atlas",
            "",
            f"- Entries: {atlas['diagnostics']['entries']}",
            f"- Typed mathematical relationships: {atlas['diagnostics']['relationships']}",
            "- Internal Lean premise/user/source edges are stored separately from mathematical relations.",
            "",
            "## Material limitations",
            "",
            "- RH itself, GRH, and heuristic criteria are knowledge anchors, not ordinary proof targets.",
            "- PNT+ compiles at the pinned native revision, but modules/declarations containing or depending on `sorry` were excluded from the external corpus.",
            "- Other-prover entries establish formalization presence only; they are not Lean SFT records.",
            "- Literature statements are attributed knowledge records and never treated as verified Lean proofs.",
            "- The census is bounded by the source list in `sources/source-manifest.json` and records unresolved gaps explicitly.",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_riemann_data(
    phase2_artifact_dir: Path,
    output_dir: Path,
    config: RiemannDataConfig,
    atlas_config: RiemannAtlasConfig,
    *,
    external_root: Path | None,
    phase2_snapshot_dir: Path,
) -> dict[str, Any]:
    source_manifest, source_records = load_phase2_records(phase2_artifact_dir, config)
    materialize_phase2_snapshot(source_manifest, source_records, phase2_snapshot_dir)
    phase2_manifest, records = load_phase2_records(phase2_snapshot_dir, config)
    graph = build_internal_graph(records, config)
    partitions = build_specialist_partitions(records, graph, config)
    external_records, source_manifest = extract_external_lean_records(
        external_root, records, config
    )
    atlas = build_atlas(atlas_config, graph, records, external_records, config)
    opportunities = _rank_opportunities(atlas, external_records, partitions, records, config)

    by_id = {str(record["id"]): record for record in records}
    node_by_id = {str(node["record_id"]): node for node in graph["nodes"]}
    classes: Mapping[str, str] = graph["classes"]
    all_internal_ids = set().union(*partitions["memberships"].values())

    _write_json(output_dir / "sources/source-manifest.json", source_manifest)
    _write_jsonl(output_dir / "internal/seed-manifest.jsonl", graph["seed_manifest"])
    _write_gzip_jsonl(output_dir / "internal/nodes.jsonl.gz", graph["nodes"])
    _write_gzip_jsonl(
        output_dir / "internal/edges.jsonl.gz",
        (
            {
                "schema_version": RIEMANN_GRAPH_SCHEMA_VERSION,
                "type": edge_type,
                "source_record_id": source_id,
                "target_record_id": target_id,
            }
            for edge_type, source_id, target_id in graph["edges"]
        ),
    )
    _write_json(output_dir / "internal/diagnostics.json", graph["diagnostics"])

    _write_gzip_jsonl(
        output_dir / "corpora/records.jsonl.gz",
        (
            {
                **by_id[record_id],
                "riemann": {
                    "relevance_class": classes.get(record_id),
                    "seed_families": node_by_id[record_id]["seed_families"],
                },
            }
            for record_id in sorted(all_internal_ids)
        ),
    )
    record_store_hash = _sha256_file(output_dir / "corpora/records.jsonl.gz")
    corpus_roles: dict[str, str] = {}
    for name, record_ids in sorted(partitions["memberships"].items()):
        membership_path = output_dir / f"corpora/{name}/membership.jsonl"
        _write_jsonl(membership_path, _membership_rows(record_ids, by_id, classes))
        role = "training-source" if name in partitions["training_names"] else "evaluation-target"
        corpus_roles[name] = role
        _write_json(
            output_dir / f"corpora/{name}/manifest.json",
            {
                "schema_version": "riemann-corpus-manifest-v1",
                "corpus_id": name,
                "record_count": len(record_ids),
                "role": role,
                "record_store": "../records.jsonl.gz",
                "record_store_sha256": record_store_hash,
                "membership": "membership.jsonl",
                "membership_sha256": _sha256_file(membership_path),
                "phase2_source_revision": config.value["phase2_source"]["revision"],
                "membership_frozen_on": config.value["snapshot_date"],
                "token_lengths_declaration_and_completion": _token_statistics(record_ids, by_id),
            },
        )

    external_path = output_dir / "external/riemann-external-lean-v1.jsonl"
    _write_jsonl(external_path, external_records)
    external_name = config.value["corpora"]["external_lean"]
    external_membership_path = output_dir / f"corpora/{external_name}/membership.jsonl"
    _write_jsonl(
        external_membership_path,
        ({"record_id": record["id"], "declaration_name": record["declaration_name"]} for record in external_records),
    )
    corpus_roles[external_name] = "separate-training-source"
    _write_json(
        output_dir / f"corpora/{external_name}/manifest.json",
        {
            "schema_version": "riemann-corpus-manifest-v1",
            "corpus_id": external_name,
            "record_count": len(external_records),
            "role": "separate-training-source",
            "record_store": f"../../external/{external_path.name}",
            "record_store_sha256": _sha256_file(external_path),
            "membership": "membership.jsonl",
            "membership_sha256": _sha256_file(external_membership_path),
            "provenance_lane": "external-lean",
            "merged_into_phase2_pool": False,
        },
    )

    _write_jsonl(output_dir / "atlas/entries.jsonl", atlas["entries"])
    _write_jsonl(output_dir / "atlas/relationships.jsonl", atlas["relationships"])
    _write_json(output_dir / "atlas/sources.json", atlas["sources"])
    _write_json(output_dir / "atlas/diagnostics.json", atlas["diagnostics"])
    _write_json(output_dir / "evidence/duplicate-aliases.json", atlas["alias_collisions"])
    _write_json(output_dir / "evidence/opportunities.json", opportunities)

    evidence = {
        "schema_version": "riemann-evidence-v1",
        "snapshot_date": config.value["snapshot_date"],
        "phase2": {
            "records": len(records),
            "source": phase2_manifest["source"],
            "split_hygiene": phase2_manifest.get("split_hygiene", {}),
            "contamination": phase2_manifest.get("contamination", {}),
        },
        "internal_graph": graph["diagnostics"],
        "specialist_partitions": partitions["diagnostics"],
        "external_lean": {
            "sources_inspected": len(source_manifest["projects"]),
            "accepted_records": len(external_records),
            "accepted_corpus_materialized": bool(external_records),
        },
        "atlas": atlas["diagnostics"],
        "model_outputs_used_for_membership": False,
        "gpu_required": False,
    }
    _write_json(output_dir / "evidence/summary.json", evidence)

    preliminary_manifest = {
        "schema_version": RIEMANN_MANIFEST_SCHEMA_VERSION,
        "snapshot_date": config.value["snapshot_date"],
        "phase2_source": config.value["phase2_source"],
        "phase2_snapshot": {
            "relative_path": Path(os.path.relpath(phase2_snapshot_dir, output_dir)).as_posix(),
            "manifest_sha256": _sha256_file(phase2_snapshot_dir / "manifest.json"),
            "record_count": len(records),
            "split_counts": {split: sum(record["split"] == split for record in records) for split in SPLIT_NAMES},
        },
        "corpus_counts": {
            **partitions["diagnostics"]["counts"],
            external_name: len(external_records),
        },
        "corpus_roles": corpus_roles,
        "atlas_schema_version": RIEMANN_ATLAS_SCHEMA_VERSION,
        "internal_graph_schema_version": RIEMANN_GRAPH_SCHEMA_VERSION,
        "authoritative_location": "repository",
        "inference_performed": False,
    }
    (output_dir / "REPORT.md").write_text(
        _report_markdown(preliminary_manifest, graph, partitions, source_manifest, atlas),
        encoding="utf-8",
    )

    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != output_dir / "manifest.json":
            relative = path.relative_to(output_dir).as_posix()
            files[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    manifest = {**preliminary_manifest, "files": files}
    _write_json(output_dir / "manifest.json", manifest)
    return validate_materialized_riemann_data(output_dir)


def validate_materialized_riemann_data(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RIEMANN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported materialized Riemann manifest")
    for relative, identity in manifest["files"].items():
        path = output_dir / relative
        if not path.is_file():
            raise ValueError(f"materialized Riemann file is missing: {relative}")
        if _sha256_file(path) != identity["sha256"]:
            raise ValueError(f"materialized Riemann file hash differs: {relative}")

    snapshot_identity = manifest.get("phase2_snapshot")
    if not snapshot_identity:
        raise ValueError("materialized Riemann data lacks a canonical Phase 2 snapshot")
    phase2_snapshot_dir = (output_dir / snapshot_identity["relative_path"]).resolve()
    phase2_manifest_path = phase2_snapshot_dir / "manifest.json"
    if _sha256_file(phase2_manifest_path) != snapshot_identity["manifest_sha256"]:
        raise ValueError("canonical Phase 2 manifest hash differs from the Riemann manifest")
    phase2_config = RiemannDataConfig({"phase2_source": manifest["phase2_source"]})
    _, phase2_records = load_phase2_records(phase2_snapshot_dir, phase2_config)
    if len(phase2_records) != int(snapshot_identity["record_count"]):
        raise ValueError("canonical Phase 2 record count differs from the Riemann manifest")
    phase2_by_id = {str(record["id"]): record for record in phase2_records}

    internal_records = {
        str(record["id"]): record for record in _read_jsonl(output_dir / "corpora/records.jsonl.gz")
    }
    for record_id, enriched_record in internal_records.items():
        canonical_record = phase2_by_id.get(record_id)
        base_record = {key: value for key, value in enriched_record.items() if key != "riemann"}
        if canonical_record is None or base_record != canonical_record:
            raise ValueError(f"specialist record differs from canonical Phase 2 identity: {record_id}")
    corpus_ids: dict[str, set[str]] = {}
    for name, expected_count in manifest["corpus_counts"].items():
        membership = output_dir / f"corpora/{name}/membership.jsonl"
        members = {str(row["record_id"]) for row in _read_jsonl(membership)}
        if len(members) != int(expected_count):
            raise ValueError(f"corpus count differs for {name}")
        corpus_ids[name] = members
        role = manifest["corpus_roles"][name]
        if role != "separate-training-source":
            if not members <= set(internal_records):
                raise ValueError(f"corpus has records absent from shared store: {name}")
            required_split = "train" if role == "training-source" else None
            if required_split and any(internal_records[item]["split"] != required_split for item in members):
                raise ValueError(f"evaluation data leaked into {name}")

    atlas_entry_statuses = {
        str(entry["id"]): str(entry["formalization_status"])
        for entry in _read_jsonl(output_dir / "atlas/entries.jsonl")
    }
    atlas_entries = set(atlas_entry_statuses)
    for relationship in _read_jsonl(output_dir / "atlas/relationships.jsonl"):
        if relationship["type"] not in ATLAS_EDGE_TYPES:
            raise ValueError("mathematical relationship graph has an invalid edge type")
        if relationship["source"] not in atlas_entries or relationship["target"] not in atlas_entries:
            raise ValueError("mathematical relationship graph has an unknown endpoint")
        if (
            relationship["type"] == "Lean-counterpart-of"
            and atlas_entry_statuses[relationship["target"]] == "literature-only"
        ):
            raise ValueError("literature-only atlas entry has a Lean counterpart edge")

    internal_edge_types = {
        str(edge["type"]) for edge in _read_jsonl(output_dir / "internal/edges.jsonl.gz")
    }
    if internal_edge_types - {"premise", "user", "same-source"}:
        raise ValueError("internal Lean graph contains mathematical relation edge types")

    return {
        "manifest": str(output_dir / "manifest.json"),
        "files_verified": len(manifest["files"]),
        "corpus_counts": manifest["corpus_counts"],
        "internal_record_store_count": len(internal_records),
        "phase2_snapshot_records": len(phase2_records),
        "atlas_entries": len(atlas_entries),
        "complete": True,
    }
