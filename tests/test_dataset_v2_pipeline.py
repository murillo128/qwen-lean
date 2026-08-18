from __future__ import annotations

from qwen_lean.dataset_v2_pipeline import (
    PRIME_FAMILIES,
    distribute_prime_counts,
    prime_families_for,
)


def test_prime_family_tagging_is_inclusive() -> None:
    families = prime_families_for(
        file_path="Mathlib/NumberTheory/ArithmeticFunction/Mangoldt.lean",
        declaration_name="ArithmeticFunction.vonMangoldt_apply_prime",
        memberships=("riemann-core-v1",),
    )
    assert "arithmetic-functions" in families
    assert "prime-arithmetic-divisibility" in families
    assert "riemann-core-bubble" in families


def test_prime_counts_are_deterministic_and_exhaustive() -> None:
    counts = distribute_prime_counts(32)
    assert tuple(counts) == PRIME_FAMILIES
    assert sum(counts.values()) == 32
    assert max(counts.values()) - min(counts.values()) <= 1
