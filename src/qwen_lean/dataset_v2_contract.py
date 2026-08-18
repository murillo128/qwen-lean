from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from typing import Iterable

from .phase2_corpus import (
    _declaration_name_index,
    _lex_lean,
    canonical_declaration,
    normalized_lean_code,
)


DATASET_V2_PROOF_CANONICALIZATION = "dataset-v2-proof-canonicalization-v1"
DERIVATION_FAMILY_FINGERPRINT = "dataset-v2-derivation-family-v1"
STATEMENT_FINGERPRINT = "dataset-v2-alpha-statement-v1"
STATEMENT_ID = "dataset-v2-statement-id-v1"
PROOF_VARIANT_ID = "dataset-v2-proof-variant-id-v1"


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


def normalized_statement_v2(value: str) -> str:
    """Normalize a declaration name and common explicit/quantified binders.

    Lean's elaborator remains the authority for semantic equality. This lexical
    identity removes theorem/lemma spelling, declaration names, comments,
    whitespace, and alpha-renaming of explicit declaration, ``∀``, and
    ``fun``/``λ`` binders. It is deliberately conservative outside those common
    forms.
    """

    tokens = _lex_lean(canonical_declaration(value))
    name_index = _declaration_name_index(tokens)
    tokens[name_index - 1] = ("identifier", "theorem_or_lemma")
    tokens[name_index] = ("identifier", "__DECLARATION_NAME__")

    binder_names: dict[str, str] = {}
    index = name_index + 1
    while index < len(tokens):
        kind, token = tokens[index]
        if kind == "symbol" and token == ":":
            break
        if kind == "symbol" and token in "([{⦃":
            opening = token
            closing = {"(": ")", "[": "]", "{": "}", "⦃": "⦄"}[opening]
            depth = 1
            end = index + 1
            colon: int | None = None
            while end < len(tokens) and depth:
                end_kind, end_token = tokens[end]
                if end_kind == "symbol" and end_token == opening:
                    depth += 1
                elif end_kind == "symbol" and end_token == closing:
                    depth -= 1
                    if depth == 0:
                        break
                elif depth == 1 and end_kind == "symbol" and end_token == ":":
                    colon = end
                end += 1
            if depth:
                raise ValueError("unterminated declaration binder")
            if colon is not None:
                for binder_index in range(index + 1, colon):
                    binder_kind, binder_token = tokens[binder_index]
                    if binder_kind == "identifier" and binder_token != "_":
                        binder_names.setdefault(
                            binder_token, f"__BOUND_{len(binder_names)}__"
                        )
            index = end
        index += 1

    for index, (kind, token) in enumerate(tokens):
        if not (
            (kind == "symbol" and token in {"∀", "λ"})
            or (kind == "identifier" and token == "fun")
        ):
            continue
        depth = 0
        end = index + 1
        while end < len(tokens):
            end_kind, end_token = tokens[end]
            if end_kind == "symbol" and end_token in "([{⦃":
                depth += 1
            elif end_kind == "symbol" and end_token in ")]}\u2984":
                depth -= 1
            elif depth == 0 and end_kind == "symbol":
                if token == "∀" and end_token == ",":
                    break
                if (
                    token != "∀"
                    and end_token == "="
                    and end + 1 < len(tokens)
                    and tokens[end + 1] == ("symbol", ">")
                ):
                    break
            end += 1
        if end == len(tokens):
            continue
        segment = tokens[index + 1 : end]
        segment_has_colon = any(
            segment_kind == "symbol" and segment_token == ":"
            for segment_kind, segment_token in segment
        )
        if not segment_has_colon:
            for segment_kind, segment_token in segment:
                if segment_kind == "identifier" and segment_token != "_":
                    binder_names.setdefault(
                        segment_token, f"__BOUND_{len(binder_names)}__"
                    )
            continue
        local_depth = 0
        colon_seen: dict[int, bool] = {0: False}
        for segment_kind, segment_token in segment:
            if segment_kind == "symbol" and segment_token in "([{⦃":
                local_depth += 1
                colon_seen[local_depth] = False
            elif segment_kind == "symbol" and segment_token in ")]}\u2984":
                colon_seen.pop(local_depth, None)
                local_depth -= 1
            elif segment_kind == "symbol" and segment_token == ":":
                colon_seen[local_depth] = True
            elif (
                segment_kind == "identifier"
                and segment_token != "_"
                and not colon_seen.get(local_depth, False)
            ):
                binder_names.setdefault(
                    segment_token, f"__BOUND_{len(binder_names)}__"
                )

    normalized = [
        (kind, binder_names.get(token, token)) for kind, token in tokens
    ]
    return "\x1f".join(f"{kind}:{token}" for kind, token in normalized)


def statement_fingerprint_v2(value: str) -> str:
    payload = f"{STATEMENT_FINGERPRINT}\0{normalized_statement_v2(value)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def statement_id(value: str) -> str:
    fingerprint = statement_fingerprint_v2(value)
    return hashlib.sha256(f"{STATEMENT_ID}\0{fingerprint}".encode("utf-8")).hexdigest()


def proof_variant_id(statement_identity: str, proof: str) -> str:
    if not statement_identity.strip():
        raise ValueError("statement identity is empty")
    fingerprint = proof_fingerprint(proof)
    payload = f"{PROOF_VARIANT_ID}\0{statement_identity}\0{fingerprint}"
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
