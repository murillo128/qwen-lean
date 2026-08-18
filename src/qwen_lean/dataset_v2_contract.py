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
STATEMENT_FINGERPRINT = "dataset-v2-alpha-statement-v2"
STATEMENT_ID = "dataset-v2-statement-id-v2"
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


def canonicalize_equation_clauses(proof_expression: str) -> CanonicalProof:
    """Turn a declaration equation block into an explicit function proof term."""

    source = _normalize_transport(proof_expression)
    if not source.startswith("|"):
        raise ValueError("equation proof does not begin with an equation clause")
    # ``@fun`` disables declaration-level implicit-lambda insertion so equation
    # patterns bind the full declared function, including implicit arguments.
    function_term = "@fun\n" + textwrap.indent(source, "  ")
    canonical = "by\n  exact (\n" + textwrap.indent(function_term, "    ") + "\n  )"
    return CanonicalProof(
        source_expression=source,
        canonical_proof=canonical,
        completion=canonical[2:].lstrip(),
        transformation="equations-to-fun-exact",
    )


def proof_fingerprint(proof: str) -> str:
    if not proof.strip():
        raise ValueError("proof is empty")
    normalized = normalized_lean_code(proof)
    payload = f"{DATASET_V2_PROOF_CANONICALIZATION}\0{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_statement_v2_legacy(value: str) -> str:
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


_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "⦃": "⦄"}


def _matching_token(
    tokens: list[tuple[str, str]], opening_index: int, end: int
) -> int:
    opening = tokens[opening_index][1]
    closing = _OPEN_TO_CLOSE[opening]
    depth = 1
    for index in range(opening_index + 1, end):
        kind, token = tokens[index]
        if kind != "symbol":
            continue
        if token == opening:
            depth += 1
        elif token == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated Lean binder/group")


def _binder_name_indices(
    tokens: list[tuple[str, str]], start: int, end: int
) -> list[int]:
    """Find declared names in a forall/lambda binder segment."""

    names: list[int] = []
    saw_bracket = False
    index = start
    while index < end:
        kind, token = tokens[index]
        if kind == "symbol" and token in _OPEN_TO_CLOSE:
            saw_bracket = True
            closing = _matching_token(tokens, index, end)
            colon = next(
                (
                    candidate
                    for candidate in range(index + 1, closing)
                    if tokens[candidate] == ("symbol", ":")
                ),
                None,
            )
            boundary = closing if colon is None else colon
            names.extend(
                candidate
                for candidate in range(index + 1, boundary)
                if tokens[candidate][0] == "identifier"
                and tokens[candidate][1] != "_"
            )
            index = closing + 1
            continue
        index += 1
    if saw_bracket:
        return names
    colon = next(
        (
            candidate
            for candidate in range(start, end)
            if tokens[candidate] == ("symbol", ":")
        ),
        None,
    )
    boundary = end if colon is None else colon
    return [
        index
        for index in range(start, boundary)
        if tokens[index][0] == "identifier" and tokens[index][1] != "_"
    ]


def _normalize_expression_tokens(
    tokens: list[tuple[str, str]],
    start: int,
    end: int,
    environment: dict[str, str],
    counter: list[int],
) -> None:
    index = start
    while index < end:
        kind, token = tokens[index]
        if kind == "symbol" and token in _OPEN_TO_CLOSE:
            closing = _matching_token(tokens, index, end)
            _normalize_expression_tokens(
                tokens, index + 1, closing, dict(environment), counter
            )
            index = closing + 1
            continue
        is_binder = (kind == "symbol" and token in {"∀", "λ"}) or (
            kind == "identifier" and token == "fun"
        )
        if is_binder:
            depth = 0
            delimiter = index + 1
            delimiter_width = 1
            while delimiter < end:
                delimiter_kind, delimiter_token = tokens[delimiter]
                if delimiter_kind == "symbol" and delimiter_token in _OPEN_TO_CLOSE:
                    depth += 1
                elif (
                    delimiter_kind == "symbol"
                    and delimiter_token in _OPEN_TO_CLOSE.values()
                ):
                    depth -= 1
                elif depth == 0 and delimiter_kind == "symbol":
                    if token == "∀" and delimiter_token == ",":
                        break
                    if (
                        token != "∀"
                        and delimiter_token == "="
                        and delimiter + 1 < end
                        and tokens[delimiter + 1] == ("symbol", ">")
                    ):
                        delimiter_width = 2
                        break
                delimiter += 1
            if delimiter < end:
                name_indices = _binder_name_indices(tokens, index + 1, delimiter)
                replacements: list[tuple[int, str, str]] = []
                inner_environment = dict(environment)
                for name_index in name_indices:
                    old_name = tokens[name_index][1]
                    sentinel = f"__DATASET_V2_BINDER_SENTINEL_{counter[0]}__"
                    canonical = f"__BOUND_{counter[0]}__"
                    counter[0] += 1
                    tokens[name_index] = ("identifier", sentinel)
                    replacements.append((name_index, sentinel, canonical))
                    inner_environment[old_name] = canonical
                _normalize_expression_tokens(
                    tokens, index + 1, delimiter, environment, counter
                )
                for name_index, _, canonical in replacements:
                    tokens[name_index] = ("identifier", canonical)
                _normalize_expression_tokens(
                    tokens,
                    delimiter + delimiter_width,
                    end,
                    inner_environment,
                    counter,
                )
                return
        if kind == "identifier" and token in environment:
            tokens[index] = (kind, environment[token])
        index += 1


def normalized_statement_v2(value: str) -> str:
    """Normalize declaration, universe, and lexical binder names scope-safely."""

    tokens = _lex_lean(canonical_declaration(value))
    name_index = _declaration_name_index(tokens)
    tokens[name_index - 1] = ("identifier", "theorem_or_lemma")
    tokens[name_index] = ("identifier", "__DECLARATION_NAME__")
    environment: dict[str, str] = {}
    counter = [0]

    index = name_index + 1
    if (
        index + 2 < len(tokens)
        and tokens[index] == ("symbol", ".")
        and tokens[index + 1] == ("symbol", "{")
    ):
        universe_end = _matching_token(tokens, index + 1, len(tokens))
        universe_number = 0
        for universe_index in range(index + 2, universe_end):
            kind, name = tokens[universe_index]
            if kind == "identifier" and name != "_":
                canonical = f"__UNIVERSE_{universe_number}__"
                universe_number += 1
                environment[name] = canonical
                tokens[universe_index] = (kind, canonical)
        index = universe_end + 1

    while index < len(tokens):
        kind, token = tokens[index]
        if kind == "symbol" and token == ":":
            _normalize_expression_tokens(
                tokens, index + 1, len(tokens), environment, counter
            )
            break
        if kind == "symbol" and token in _OPEN_TO_CLOSE:
            closing = _matching_token(tokens, index, len(tokens))
            colon = next(
                (
                    candidate
                    for candidate in range(index + 1, closing)
                    if tokens[candidate] == ("symbol", ":")
                ),
                None,
            )
            if colon is not None:
                name_indices = [
                    candidate
                    for candidate in range(index + 1, colon)
                    if tokens[candidate][0] == "identifier"
                    and tokens[candidate][1] != "_"
                ]
                old_names = [tokens[candidate][1] for candidate in name_indices]
                for name_index_in_group in name_indices:
                    tokens[name_index_in_group] = (
                        "identifier",
                        f"__DATASET_V2_DECL_SENTINEL_{name_index_in_group}__",
                    )
                _normalize_expression_tokens(
                    tokens, index + 1, closing, environment, counter
                )
                for old_name, name_index_in_group in zip(
                    old_names, name_indices, strict=True
                ):
                    canonical = f"__BOUND_{counter[0]}__"
                    counter[0] += 1
                    tokens[name_index_in_group] = ("identifier", canonical)
                    environment[old_name] = canonical
            else:
                _normalize_expression_tokens(
                    tokens, index + 1, closing, environment, counter
                )
            index = closing + 1
            continue
        if kind == "identifier" and token in environment:
            tokens[index] = (kind, environment[token])
        index += 1

    return "\x1f".join(f"{kind}:{token}" for kind, token in tokens)


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
