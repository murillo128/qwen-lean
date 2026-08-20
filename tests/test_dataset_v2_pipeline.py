from __future__ import annotations

from qwen_lean.dataset_v2_pipeline import (
    PRIME_FAMILIES,
    build_source_dispositions,
    distribute_prime_counts,
    historical_source_crosswalk,
    prime_families_for,
)
from qwen_lean.dataset_v2_extraction import ExtractionDiagnostics, SourceCandidate
from qwen_lean.dataset_v2_schema import SourcePosition, SourceSpan


def _candidate(*, status: str = "accepted") -> SourceCandidate:
    span = SourceSpan(SourcePosition(1, 1), SourcePosition(1, 41))
    return SourceCandidate(
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="a" * 40,
        file_path="Mathlib/NumberTheory/PrimeCounting.lean",
        module="Mathlib.NumberTheory.PrimeCounting",
        declaration_name="PrimeCounting.fixture",
        declaration_kind="theorem",
        source_span=span,
        declaration_span=span,
        proof_span=span,
        declaration="theorem PrimeCounting.fixture : True",
        source_expression="by trivial",
        canonical_proof="by trivial",
        completion="trivial",
        transformation_kind="none",
        resolved_dependencies=(),
        imports=(),
        provenance="real-mathlib",
        topic_tags=("prime-family:prime-counting-pnt",),
        memberships=("riemann-core-v1",),
        verification_status=status,
        verification_method="fixture",
        verification_evidence_id="fixture",
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


def test_source_dispositions_account_for_acceptance_and_explicit_exclusion() -> None:
    diagnostics = ExtractionDiagnostics(
        source_files=1,
        traced_declarations=2,
        candidates=1,
        transformation_counts={"none": 1},
        exclusion_counts={"proof-placeholder": 1},
        exclusions=(
            {
                "file_path": "Mathlib/NumberTheory/PrimeCounting.lean",
                "declaration_name": "PrimeCounting.blocked",
                "reason": "proof-placeholder",
            },
        ),
    )
    config = {
        "target_environment": {
            "mathlib_repository": "https://github.com/leanprover-community/mathlib4",
            "mathlib_revision": "a" * 40,
        }
    }
    dispositions = build_source_dispositions(
        [_candidate()],
        diagnostics=diagnostics,
        config=config,
        topic_metadata={},
    )

    assert len(dispositions) == 2
    assert {item["disposition"] for item in dispositions} == {
        "included-training",
        "placeholder/axiom-policy-blocked",
    }
    assert all(item["prime_families"] for item in dispositions)


def test_historical_crosswalk_reports_mapped_and_missing_identities() -> None:
    dispositions = build_source_dispositions(
        [_candidate()],
        diagnostics=ExtractionDiagnostics(1, 1, 1, {"none": 1}, {}, ()),
        config={
            "target_environment": {
                "mathlib_repository": "https://github.com/leanprover-community/mathlib4",
                "mathlib_revision": "a" * 40,
            }
        },
        topic_metadata={},
    )
    records = [
        {
            "file_path": "Mathlib/NumberTheory/PrimeCounting.lean",
            "declaration_name": "PrimeCounting.fixture",
        },
        {
            "file_path": "Mathlib/Missing.lean",
            "declaration_name": "Missing.fixture",
        },
    ]
    crosswalk = historical_source_crosswalk(
        dispositions,
        historical_records=records,
        membership_inventories={"riemann-core-v1": records[:1]},
    )

    assert crosswalk["mathlib_v1"]["records"] == 2
    assert crosswalk["mathlib_v1"]["missing_source_identities"] == 1
    assert crosswalk["riemann_inventories"]["riemann-core-v1"][
        "missing_source_identities"
    ] == 0


def test_historical_crosswalk_maps_unique_name_only_inventory_rows() -> None:
    dispositions = build_source_dispositions(
        [_candidate()],
        diagnostics=ExtractionDiagnostics(1, 1, 1, {"none": 1}, {}, ()),
        config={
            "target_environment": {
                "mathlib_repository": "https://github.com/leanprover-community/mathlib4",
                "mathlib_revision": "a" * 40,
            }
        },
        topic_metadata={},
    )

    crosswalk = historical_source_crosswalk(
        dispositions,
        historical_records=[],
        membership_inventories={
            "name-only": [{"declaration_name": "PrimeCounting.fixture"}]
        },
    )

    assert crosswalk["riemann_inventories"]["name-only"]["dispositions"] == {
        "included-training": 1
    }
