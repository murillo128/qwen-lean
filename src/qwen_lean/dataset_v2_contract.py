from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from typing import Iterable

from .phase2_corpus import normalized_lean_code


DATASET_V2_PROOF_CANONICALIZATION = "dataset-v2-proof-canonicalization-v1"
DERIVATION_FAMILY_FINGERPRINT = "dataset-v2-derivation-family-v1"


@dataclass(frozen=True)
class CanonicalProof:
    """A source proof expression normalized to the whole-proof continuation contract."""

    source_expression: str
    canonical_proof: str
    completion: str
    transformation: str
    requires_lean_verification: bool = True


def _normalize_transport(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _is_by_proof(value: str) -> bool:
    return value.startswith("by") and (len(value) == 2 or value[2].isspace())


def canonicalize_proof_expression(proof_expression: str) -> CanonicalProof:
    """Normalize a Lean proof expression without claiming semantic validity.

    Existing ``by`` proofs keep their source form. A non-``by`` term proof is
    wrapped as ``by exact (...)`` so Dataset v2 can expose it through the same
    whole-proof continuation interface used by generated proofs. Every returned
    value still requires reconstruction and Lean verification in its declared
    source/environment context before it is optimizer-eligible.
    """

    source = _normalize_transport(proof_expression)
    if not source:
        raise ValueError("proof expression is empty")

    if _is_by_proof(source):
        completion = source[2:].lstrip()
        if not completion:
            raise ValueError("by proof has an empty continuation")
        return CanonicalProof(
            source_expression=source,
            canonical_proof=source,
            completion=completion,
            transformation="none",
        )

    indented = textwrap.indent(source, "    ")
    canonical = f"by\n  exact (\n{indented}\n  )"
    return CanonicalProof(
        source_expression=source,
        canonical_proof=canonical,
        completion=canonical[2:].lstrip(),
        transformation="term-to-exact",
    )


def proof_fingerprint(proof: str) -> str:
    if not proof.strip():
        raise ValueError("proof is empty")
    normalized = normalized_lean_code(proof)
    payload = f"{DATASET_V2_PROOF_CANONICALIZATION}\0{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derivation_family_fingerprint(
    source_lemma_ids: Iterable[str],
    *,
    normalized_proof_dag: str,
    generator_family: str,
) -> str:
    """Fingerprint the synthetic derivation family used as a split boundary.

    Individual source lemmas may appear in multiple splits, but the same
    combination of actually-used lemmas, normalized proof-DAG structure, and
    generator family must not cross train/validation/test.
    """

    lemmas = tuple(
        sorted(
            {
                str(item).strip()
                for item in source_lemma_ids
                if str(item).strip()
            }
        )
    )
    dag = normalized_proof_dag.strip()
    family = generator_family.strip()
    if not lemmas:
        raise ValueError("derivation family needs at least one source lemma")
    if not dag:
        raise ValueError("normalized proof DAG is empty")
    if not family:
        raise ValueError("generator family is empty")

    payload = "\0".join(
        (
            DERIVATION_FAMILY_FINGERPRINT,
            family,
            dag,
            *lemmas,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
