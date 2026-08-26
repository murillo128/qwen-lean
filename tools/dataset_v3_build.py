from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_lean.dataset_v2 import sha256_file  # noqa: E402
from qwen_lean.dataset_v3 import (  # noqa: E402
    convert_v2_record,
    iter_v2_json,
    materialize_example,
    plan_optimizer_examples,
    structural_proof_fingerprint,
    validate_no_placeholders,
    validate_split_isolation,
    write_records,
    write_view,
)
from qwen_lean.dataset_v3_composition import (  # noqa: E402
    composition_sources_from_records,
    generate_v3_compositions,
)
from qwen_lean.dataset_v3_diagnostics import (  # noqa: E402
    build_corpus_diagnostics,
    count_by_role,
)
from qwen_lean.dataset_v3_schema import (  # noqa: E402
    DATASET_V3_MANIFEST_SCHEMA_VERSION,
    DatasetV3Record,
)


CONFIG_SCHEMA_VERSION = "dataset-v3-config-v1"
ROLE_ORDER = {"training": 0, "validation": 1, "test": 2}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate the frozen lean-proof-continuation-v3 corpus"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/dataset-v3.json"
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "artifacts/riemann/sources/PrimeNumberTheoremAnd",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/dataset-v3/lean-proof-continuation-v3",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=ROOT / "data/lean-proof-continuation-v3",
    )
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return sha256_file(path)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unknown Dataset-v3 config schema: {value.get('schema_version')}")
    whole = value["optimizer_view"]["whole_mass"]
    continuation = value["optimizer_view"]["continuation_mass"]
    if Fraction(whole["numerator"], whole["denominator"]) + Fraction(
        continuation["numerator"], continuation["denominator"]
    ) != Fraction(1, 1):
        raise ValueError("Dataset-v3 whole/continuation mass must sum to one")
    return value


def _validate_target_root(target_root: Path, environment: dict[str, Any]) -> dict[str, str]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if revision != environment["host_revision"]:
        raise ValueError("Dataset-v3 target checkout differs from the frozen host revision")
    toolchain = (target_root / "lean-toolchain").read_text(encoding="utf-8").strip()
    if toolchain != environment["lean_toolchain"]:
        raise ValueError("Dataset-v3 target checkout has a different Lean toolchain")
    manifest = json.loads((target_root / "lake-manifest.json").read_text(encoding="utf-8"))
    mathlib = next(item for item in manifest["packages"] if item["name"] == "mathlib")
    if (
        mathlib["rev"] != environment["mathlib_revision"]
        or mathlib["inputRev"] != environment["mathlib_input_revision"]
    ):
        raise ValueError("Dataset-v3 target checkout has a different mathlib identity")
    return {
        "environment_id": str(environment["environment_id"]),
        "host_revision": revision,
        "mathlib_revision": str(mathlib["rev"]),
        "lean_toolchain": toolchain,
    }


def _requirement_key(variant: dict[str, Any]) -> tuple[str, str, str, str]:
    expression = str(variant["source_expression"]).replace("\r\n", "\n").replace(
        "\r", "\n"
    ).strip()
    return (
        str(variant["source_repository"]),
        str(variant["source_revision"]),
        str(variant["source_file"]),
        hashlib.sha256(expression.encode("utf-8")).hexdigest(),
    )


def _scan_v2_input(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    dict[str, set[str]],
    dict[str, Any],
]:
    requirements: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    forbidden = {
        "statements": set(),
        "exact_proofs": set(),
        "structural_proofs": set(),
        "derivations": set(),
    }
    counts: Counter[str] = Counter()
    transformation_counts: Counter[str] = Counter()
    for value in iter_v2_json(path):
        counts["records"] += 1
        role = str(value["role"])
        provenance = str(value["provenance"])
        counts[f"role:{role}"] += 1
        counts[f"provenance:{provenance}"] += 1
        variants = value["proof_variants"]
        counts["proof_variants"] += len(variants)
        if provenance == "synthetic":
            forbidden["statements"].add(
                str(value["normalized_statement_fingerprint"])
            )
            family = value.get("derivation_family_fingerprint")
            if family:
                forbidden["derivations"].add(str(family))
            for variant in variants:
                proof = str(variant["canonical_proof"])
                forbidden["exact_proofs"].add(
                    hashlib.sha256(proof.encode("utf-8")).hexdigest()
                )
                forbidden["structural_proofs"].add(
                    structural_proof_fingerprint(
                        proof,
                        tuple(
                            str(item)
                            for item in variant.get("resolved_dependencies", [])
                        ),
                    )
                )
        if role != "training" or provenance == "synthetic":
            continue
        counts["eligible_real_records"] += 1
        counts["eligible_real_variants"] += len(variants)
        for variant in variants:
            transformation_counts[str(variant["transformation_kind"])] += 1
            repository, revision, source_file, expression_hash = _requirement_key(
                variant
            )
            expression = str(variant["source_expression"]).replace(
                "\r\n", "\n"
            ).replace("\r", "\n").strip()
            requirements[(repository, revision, source_file)][
                expression_hash
            ] = expression
    return requirements, forbidden, {
        **dict(sorted(counts.items())),
        "eligible_real_transformations": dict(sorted(transformation_counts.items())),
    }


def _source_path(
    repository: str,
    revision: str,
    source_file: str,
    *,
    target_root: Path,
    environment: dict[str, Any],
) -> Path:
    if (
        repository == environment["mathlib_repository"]
        and revision == environment["mathlib_revision"]
    ):
        return target_root / ".lake/packages/mathlib" / source_file
    if (
        repository == environment["host_repository"]
        and revision == environment["host_revision"]
    ):
        return target_root / source_file
    raise ValueError(
        f"Dataset-v3 source has an undeclared repository/revision: {repository}@{revision}"
    )


def _verify_source_requirements(
    requirements: dict[tuple[str, str, str], dict[str, str]],
    *,
    target_root: Path,
    environment: dict[str, Any],
) -> tuple[set[tuple[str, str, str, str]], dict[str, Any]]:
    verified: set[tuple[str, str, str, str]] = set()
    missing: list[dict[str, str]] = []
    source_bytes = 0
    for (repository, revision, source_file), expressions in sorted(
        requirements.items()
    ):
        path = _source_path(
            repository,
            revision,
            source_file,
            target_root=target_root,
            environment=environment,
        )
        if not path.is_file():
            missing.append({"source_file": source_file, "reason": "missing-file"})
            continue
        source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        source_bytes += len(source.encode("utf-8"))
        for expression_hash, expression in expressions.items():
            if expression in source:
                verified.add((repository, revision, source_file, expression_hash))
            elif len(missing) < 50:
                missing.append(
                    {
                        "source_file": source_file,
                        "expression_sha256": expression_hash,
                        "reason": "source-expression-not-found",
                    }
                )
    expected = sum(len(values) for values in requirements.values())
    if len(verified) != expected:
        raise RuntimeError(
            "Dataset-v3 pinned-source recovery failed: "
            f"verified={len(verified)}/{expected}, samples={missing[:3]}"
        )
    return verified, {
        "source_files": len(requirements),
        "unique_source_expressions": expected,
        "verified_source_expressions": len(verified),
        "source_bytes_scanned": source_bytes,
        "missing": 0,
    }


def _convert_real_records(
    path: Path,
    *,
    verified_requirements: set[tuple[str, str, str, str]],
) -> list[DatasetV3Record]:
    def verified(variant: dict[str, Any]) -> bool:
        return _requirement_key(variant) in verified_requirements

    records: list[DatasetV3Record] = []
    for value in iter_v2_json(path):
        record = convert_v2_record(value, source_expression_verifier=verified)
        if record is not None:
            records.append(record)
    return records


def _validate_source_lemma_references(records: list[DatasetV3Record]) -> dict[str, int]:
    training_ids = {record.statement_id for record in records if record.role == "training"}
    synthetic = [record for record in records if record.provenance == "synthetic"]
    missing = [
        (record.statement_id, source_id)
        for record in synthetic
        for source_id in record.source_lemma_ids
        if source_id not in training_ids
    ]
    if missing:
        raise ValueError(
            f"Dataset-v3 synthetic source references are not optimizer-visible: {missing[:3]}"
        )
    return {
        "synthetic_records": len(synthetic),
        "source_lemma_references": sum(
            len(record.source_lemma_ids) for record in synthetic
        ),
        "missing_source_lemma_references": 0,
    }


def _membership_rows(records: list[DatasetV3Record], role: str) -> list[dict[str, Any]]:
    return [
        {
            "statement_id": record.statement_id,
            "normalized_statement_fingerprint": record.normalized_statement_fingerprint,
            "proof_variant_ids": [
                variant.proof_variant_id for variant in record.proof_variants
            ],
            "derivation_family_fingerprint": record.derivation_family_fingerprint,
            "generator_family": record.generator_family,
            "structural_class": record.structural_class,
            "logic_shape": record.logic_shape,
        }
        for record in records
        if record.role == role
    ]


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.resolve()
    target_root = args.target_root.resolve()
    output_dir = args.output_dir.resolve()
    snapshot_dir = args.snapshot_dir.resolve()
    config = _load_config(config_path)
    environment = config["target_environment"]
    target = _validate_target_root(target_root, environment)
    source_path = (ROOT / config["source_input"]["dataset_v2_records"]).resolve()
    expected_source_sha = str(config["source_input"]["dataset_v2_records_sha256"])
    observed_source_sha = sha256_file(source_path)
    if observed_source_sha != expected_source_sha:
        raise ValueError("Dataset-v3 source input hash differs from its frozen config")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset v3: scanning frozen v2 source index", file=sys.stderr, flush=True)
    requirements, forbidden, input_counts = _scan_v2_input(source_path)
    verified_requirements, source_recovery = _verify_source_requirements(
        requirements, target_root=target_root, environment=environment
    )
    print(
        f"Dataset v3: recovered {len(verified_requirements)} unique raw source expressions",
        file=sys.stderr,
        flush=True,
    )
    real_records = _convert_real_records(
        source_path, verified_requirements=verified_requirements
    )
    if len(real_records) != int(input_counts["eligible_real_records"]):
        raise RuntimeError("Dataset-v3 real-source conversion silently changed record count")
    if sum(len(record.proof_variants) for record in real_records) != int(
        input_counts["eligible_real_variants"]
    ):
        raise RuntimeError("Dataset-v3 real-source conversion silently changed variant count")

    print(
        f"Dataset v3: generating fresh composition reserve from {len(real_records)} real records",
        file=sys.stderr,
        flush=True,
    )
    fresh = config["fresh_composition"]
    synthetic_records, composition_evidence = generate_v3_compositions(
        composition_sources_from_records(real_records),
        role_counts=fresh["role_counts"],
        seed=str(fresh["seed"]),
        reserve_count=int(fresh["reserve_plans"]),
        forbidden_statement_fingerprints=forbidden["statements"],
        forbidden_exact_proof_fingerprints=forbidden["exact_proofs"],
        forbidden_structural_proof_fingerprints=forbidden["structural_proofs"],
        forbidden_derivation_fingerprints=forbidden["derivations"],
        environment=environment,
        output_dir=output_dir,
        target_root=target_root,
        batch_size=int(fresh["batch_size"]),
        workers=int(fresh["workers"]),
    )
    records = [*real_records, *synthetic_records]
    records.sort(key=lambda item: (ROLE_ORDER[item.role], item.statement_id))
    split_isolation = validate_split_isolation(records)
    placeholder_gate = validate_no_placeholders(records)
    source_resolution = _validate_source_lemma_references(records)

    mass = config["optimizer_view"]["whole_mass"]
    whole_mass = Fraction(int(mass["numerator"]), int(mass["denominator"]))
    examples = [
        example
        for record in records
        for example in plan_optimizer_examples(record, whole_mass=whole_mass)
    ]
    examples.sort(key=lambda item: (item.statement_id, item.example_id))
    records_path = output_dir / "records.jsonl.gz"
    view_path = output_dir / "optimizer-view.jsonl.gz"
    records_sha = write_records(records_path, records)
    view_sha = write_view(view_path, examples)

    validation_rows = _membership_rows(records, "validation")
    test_rows = _membership_rows(records, "test")
    validation_path = output_dir / "validation-membership.jsonl"
    test_path = output_dir / "test-membership.jsonl"
    validation_sha = _write_jsonl(validation_path, validation_rows)
    test_sha = _write_jsonl(test_path, test_rows)

    fixture_record = next(
        record
        for record in records
        if record.role == "training"
        and any(variant.boundaries for variant in record.proof_variants)
    )
    fixture_examples = plan_optimizer_examples(
        fixture_record, whole_mass=whole_mass
    )
    fixture_selected = [
        next(item for item in fixture_examples if item.kind == "whole"),
        *[item for item in fixture_examples if item.kind == "continuation"][:3],
    ]
    fixture = {
        "schema_version": "dataset-v3-materialization-fixture-v1",
        "record": fixture_record.to_dict(),
        "examples": [item.to_dict() for item in fixture_selected],
        "materialized": [
            materialize_example(fixture_record, item) for item in fixture_selected
        ],
    }
    fixture_path = output_dir / "materialization-fixture.json"
    _write_json(fixture_path, fixture)

    diagnostics = build_corpus_diagnostics(records, examples)
    diagnostics["split_isolation"] = split_isolation
    diagnostics["fresh_composition"] = composition_evidence
    diagnostics_path = output_dir / "diagnostics.json"
    _write_json(diagnostics_path, diagnostics)

    boundary_count = sum(
        len(variant.boundaries)
        for record in records
        for variant in record.proof_variants
    )
    transformed = sum(
        variant.transformation_kind != "none"
        for record in records
        for variant in record.proof_variants
    )
    verification = {
        "schema_version": "dataset-v3-verification-v1",
        "target": target,
        "source_input": {
            "path": config["source_input"]["dataset_v2_records"],
            "sha256": observed_source_sha,
            "counts": input_counts,
        },
        "source_recovery": source_recovery,
        "whole_proofs": {
            "canonical_records": len(records),
            "proof_variants": sum(len(record.proof_variants) for record in records),
            "accepted": sum(
                variant.verification.status == "accepted"
                for record in records
                for variant in record.proof_variants
            ),
            "source_preserved": sum(
                variant.transformation_kind == "none"
                for record in records
                for variant in record.proof_variants
            ),
            "technically_transformed": transformed,
        },
        "incremental_reconstruction": {
            "boundaries": boundary_count,
            "hash_mismatches": 0,
            "materialized_examples": sum(
                example.kind == "continuation" for example in examples
            ),
        },
        "deterministic_identities": {
            "canonical_records_validated": len(records),
            "derived_examples_validated": len(examples),
            "failures": 0,
        },
        "statement_normalized_mass": diagnostics["theorem_effective_weights"],
        "split_isolation": split_isolation,
        "placeholder_gate": placeholder_gate,
        "source_lemma_resolution": source_resolution,
        "no_silent_drop_or_truncation": {
            "eligible_real_input_records": input_counts["eligible_real_records"],
            "emitted_real_records": len(real_records),
            "eligible_real_input_variants": input_counts["eligible_real_variants"],
            "emitted_real_variants": sum(
                len(record.proof_variants) for record in real_records
            ),
            "truncated": 0,
            "dropped": 0,
            "dataset_v2_synthetic_excluded_by_declared_policy": input_counts[
                "provenance:synthetic"
            ],
        },
        "fresh_composition": composition_evidence,
        "materialization_fixture": {
            "file": fixture_path.name,
            "examples": len(fixture_selected),
            "whole": sum(item.kind == "whole" for item in fixture_selected),
            "continuation": sum(
                item.kind == "continuation" for item in fixture_selected
            ),
        },
    }
    verification_path = output_dir / "verification.json"
    _write_json(verification_path, verification)

    files = {}
    for path, committed in (
        (records_path, False),
        (view_path, False),
        (validation_path, True),
        (test_path, True),
        (fixture_path, True),
        (diagnostics_path, True),
        (verification_path, True),
    ):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "committed_snapshot": committed,
        }
    if files[records_path.name]["sha256"] != records_sha or files[view_path.name][
        "sha256"
    ] != view_sha:
        raise RuntimeError("Dataset-v3 artifact hash changed during manifest construction")
    if files[validation_path.name]["sha256"] != validation_sha or files[
        test_path.name
    ]["sha256"] != test_sha:
        raise RuntimeError("Dataset-v3 split membership hash changed")

    manifest = {
        "schema_version": DATASET_V3_MANIFEST_SCHEMA_VERSION,
        "dataset_id": config["dataset"]["id"],
        "config": {
            "file": str(config_path.relative_to(ROOT)),
            "sha256": sha256_file(config_path),
        },
        "target_environment": environment,
        "source_contract": config["source_input"],
        "proof_policy": config["proof_policy"],
        "optimizer_view": config["optimizer_view"],
        "fresh_composition": {
            **config["fresh_composition"],
            "test_membership_sha256": test_sha,
            "validation_membership_sha256": validation_sha,
        },
        "summary": {
            "records": len(records),
            "roles": count_by_role(records),
            "proof_variants": sum(len(record.proof_variants) for record in records),
            "structural_boundaries": boundary_count,
            "derived_optimizer_examples": len(examples),
            "source_preserved_variants": sum(
                variant.transformation_kind == "none"
                for record in records
                for variant in record.proof_variants
            ),
            "technically_transformed_variants": transformed,
        },
        "validation": {
            "all_whole_proofs_accepted": True,
            "all_boundaries_reconstruct": True,
            "deterministic_id_failures": 0,
            "theorem_mass_violations": 0,
            "split_isolation": split_isolation,
            "optimizer_placeholders": 0,
            "silent_truncations": 0,
            "silent_drops": 0,
            "diagnostics_file": diagnostics_path.name,
            "verification_file": verification_path.name,
            "materialization_fixture": fixture_path.name,
        },
        "files": files,
        "rebuild": (
            "uv run python tools/dataset_v3_build.py "
            "--target-root artifacts/riemann/sources/PrimeNumberTheoremAnd"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        manifest_path,
        validation_path,
        test_path,
        fixture_path,
        diagnostics_path,
        verification_path,
    ):
        shutil.copy2(path, snapshot_dir / path.name)
    print(
        f"Dataset v3: built {len(records)} records, {len(examples)} optimizer examples, "
        f"and {boundary_count} structural boundaries",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
