from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any, Iterable, Sequence

from .dataset_v3 import (
    first_proof_construct,
    materialize_example,
    representative_variants,
)
from .dataset_v3_schema import DatasetV3Record, DerivedExampleRef


getcontext().prec = 40


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _decimal_string(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(numerator: int, denominator: int) -> int:
        index = ((len(ordered) - 1) * numerator) // denominator
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(1, 4),
        "p50": percentile(1, 2),
        "p75": percentile(3, 4),
        "p90": percentile(9, 10),
        "p95": percentile(19, 20),
        "p99": percentile(99, 100),
        "max": ordered[-1],
    }


def _count_report(counter: Counter[str], total: int) -> dict[str, Any]:
    return {
        "denominator": total,
        "values": {
            key: {
                "count": value,
                "fraction": "0" if not total else f"{value}/{total}",
            }
            for key, value in sorted(counter.items())
        },
    }


def _mass_report(counter: dict[str, Decimal], total: Decimal) -> dict[str, Any]:
    return {
        "denominator_mass": _decimal_string(total),
        "values": {
            key: {
                "mass": _decimal_string(value),
                "fraction": (
                    "0"
                    if not total
                    else _decimal_string(value / total)
                ),
            }
            for key, value in sorted(counter.items())
        },
    }


def build_corpus_diagnostics(
    records: Sequence[DatasetV3Record],
    examples: Sequence[DerivedExampleRef],
) -> dict[str, Any]:
    by_statement = {record.statement_id: record for record in records}
    raw_variants = 0
    structurally_unique_variants = 0
    raw_proof_form: Counter[str] = Counter()
    raw_first_construct: Counter[str] = Counter()
    raw_transformation: Counter[str] = Counter()
    raw_provenance: Counter[str] = Counter()
    raw_composition: Counter[str] = Counter()
    raw_logic: Counter[str] = Counter()
    proof_lengths: list[int] = []
    boundaries_per_theorem: list[int] = []
    variants_per_theorem: list[int] = []
    unique_variants_per_theorem: list[int] = []

    for record in records:
        raw_provenance[record.provenance] += 1
        if record.structural_class:
            raw_composition[record.structural_class] += 1
        if record.logic_shape:
            raw_logic[record.logic_shape] += 1
        variants_per_theorem.append(len(record.proof_variants))
        unique = representative_variants(record)
        unique_variants_per_theorem.append(len(unique))
        raw_variants += len(record.proof_variants)
        structurally_unique_variants += len(unique)
        boundaries_per_theorem.append(sum(len(item.boundaries) for item in unique))
        for variant in record.proof_variants:
            raw_proof_form[variant.proof_form] += 1
            raw_first_construct[first_proof_construct(variant.proof_text)] += 1
            raw_transformation[
                "original-source"
                if variant.transformation_kind == "none"
                else "transformed"
            ] += 1
            proof_lengths.append(len(variant.proof_text))

    mass_total = Decimal(0)
    optimizer_task_mass: dict[str, Decimal] = defaultdict(Decimal)
    optimizer_transformation_mass: dict[str, Decimal] = defaultdict(Decimal)
    optimizer_first_construct_mass: dict[str, Decimal] = defaultdict(Decimal)
    optimizer_proof_form_mass: dict[str, Decimal] = defaultdict(Decimal)
    optimizer_provenance_mass: dict[str, Decimal] = defaultdict(Decimal)
    optimizer_composition_mass: dict[str, Decimal] = defaultdict(Decimal)
    optimizer_logic_mass: dict[str, Decimal] = defaultdict(Decimal)
    theorem_mass: dict[str, Fraction] = defaultdict(Fraction)
    examples_per_theorem: Counter[str] = Counter()
    continuation_lengths: list[int] = []
    whole_lengths: list[int] = []

    for example in examples:
        record = by_statement[example.statement_id]
        materialized = materialize_example(record, example)
        variant = next(
            item
            for item in record.proof_variants
            if item.proof_variant_id == example.proof_variant_id
        )
        fraction = Fraction(example.mass_numerator, example.mass_denominator)
        mass = _decimal(fraction)
        mass_total += mass
        theorem_mass[example.statement_id] += fraction
        examples_per_theorem[example.statement_id] += 1
        optimizer_task_mass[example.kind] += mass
        optimizer_transformation_mass[
            "original-source"
            if variant.transformation_kind == "none"
            else "transformed"
        ] += mass
        optimizer_first_construct_mass[
            first_proof_construct(str(materialized["target"]))
        ] += mass
        optimizer_proof_form_mass[variant.proof_form] += mass
        optimizer_provenance_mass[record.provenance] += mass
        if record.structural_class:
            optimizer_composition_mass[record.structural_class] += mass
        if record.logic_shape:
            optimizer_logic_mass[record.logic_shape] += mass
        target_length = len(str(materialized["target"]))
        if example.kind == "continuation":
            continuation_lengths.append(target_length)
        else:
            whole_lengths.append(target_length)

    training_records = [record for record in records if record.role == "training"]
    expected_mass = {
        record.statement_id: Fraction(
            record.theorem_mass_numerator, record.theorem_mass_denominator
        )
        for record in training_records
    }
    violations = {
        statement_id: (theorem_mass.get(statement_id, Fraction()), expected)
        for statement_id, expected in expected_mass.items()
        if theorem_mass.get(statement_id, Fraction()) != expected
    }
    if violations:
        first = next(iter(violations.items()))
        raise ValueError(f"Dataset-v3 theorem mass invariant failed: {first}")
    mass_total = _decimal(
        sum(expected_mass.values(), Fraction(0, 1))
    )
    mass_values = sorted(set(theorem_mass.values()))
    example_counts = list(examples_per_theorem.values())

    return {
        "schema_version": "dataset-v3-diagnostics-v1",
        "raw": {
            "records": len(records),
            "training_records": len(training_records),
            "proof_variants": raw_variants,
            "structurally_unique_proof_variants": structurally_unique_variants,
            "proof_form": _count_report(raw_proof_form, raw_variants),
            "first_proof_construct": _count_report(
                raw_first_construct, raw_variants
            ),
            "source_vs_transformed": _count_report(
                raw_transformation, raw_variants
            ),
            "provenance": _count_report(raw_provenance, len(records)),
            "composition_class": _count_report(
                raw_composition, sum(raw_composition.values())
            ),
            "logic_shape": _count_report(raw_logic, sum(raw_logic.values())),
        },
        "optimizer_mass": {
            "examples": len(examples),
            "total": _decimal_string(mass_total),
            "whole_vs_incremental": _mass_report(optimizer_task_mass, mass_total),
            "source_vs_transformed": _mass_report(
                optimizer_transformation_mass, mass_total
            ),
            "first_proof_construct": _mass_report(
                optimizer_first_construct_mass, mass_total
            ),
            "proof_form": _mass_report(optimizer_proof_form_mass, mass_total),
            "provenance": _mass_report(optimizer_provenance_mass, mass_total),
            "composition_class": _mass_report(
                optimizer_composition_mass,
                sum(optimizer_composition_mass.values(), Decimal(0)),
            ),
            "logic_shape": _mass_report(
                optimizer_logic_mass,
                sum(optimizer_logic_mass.values(), Decimal(0)),
            ),
        },
        "lengths_chars": {
            "whole_proof": _distribution(proof_lengths),
            "whole_optimizer_target": _distribution(whole_lengths),
            "continuation_target": _distribution(continuation_lengths),
        },
        "boundaries_per_theorem": _distribution(boundaries_per_theorem),
        "variants_per_theorem": _distribution(variants_per_theorem),
        "structurally_unique_variants_per_theorem": _distribution(
            unique_variants_per_theorem
        ),
        "theorem_effective_weights": {
            "statements": len(expected_mass),
            "violations": 0,
            "unique_exact_values": [
                f"{value.numerator}/{value.denominator}" for value in mass_values
            ],
            "minimum": (
                None
                if not theorem_mass
                else f"{min(theorem_mass.values()).numerator}/"
                f"{min(theorem_mass.values()).denominator}"
            ),
            "maximum": (
                None
                if not theorem_mass
                else f"{max(theorem_mass.values()).numerator}/"
                f"{max(theorem_mass.values()).denominator}"
            ),
        },
        "theorem_contribution": {
            "examples_min": min(example_counts, default=0),
            "examples_max": max(example_counts, default=0),
            "mass_min": (
                None
                if not theorem_mass
                else f"{min(theorem_mass.values()).numerator}/"
                f"{min(theorem_mass.values()).denominator}"
            ),
            "mass_max": (
                None
                if not theorem_mass
                else f"{max(theorem_mass.values()).numerator}/"
                f"{max(theorem_mass.values()).denominator}"
            ),
        },
        "duplicates": {
            "raw_variants": raw_variants,
            "structurally_unique_variants": structurally_unique_variants,
            "cosmetic_duplicates": raw_variants - structurally_unique_variants,
            "cosmetic_duplicate_rate": (
                "0"
                if not raw_variants
                else _decimal_string(
                    Decimal(raw_variants - structurally_unique_variants)
                    / Decimal(raw_variants)
                )
            ),
        },
    }


def count_by_role(records: Iterable[DatasetV3Record]) -> dict[str, int]:
    return dict(sorted(Counter(record.role for record in records).items()))
