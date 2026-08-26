from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .dataset_v2 import sha256_file
from .dataset_v2_contract import statement_fingerprint_v2
from .dataset_v2_schema import EnvironmentContext, ProofVerification
from .dataset_v3_schema import (
    DATASET_V3_SCHEMA_VERSION,
    DATASET_V3_VIEW_SCHEMA_VERSION,
    DatasetV3ProofVariant,
    DatasetV3Record,
    DerivedExampleRef,
    StructuralBoundary,
)
from .phase2_corpus import _lex_lean


BOUNDARY_EXTRACTOR_VERSION = "lean-layout-boundaries-v1"
STRUCTURAL_PROOF_FINGERPRINT_VERSION = "dataset-v3-structural-proof-v1"

_TACTIC_HEADS = frozenset(
    {
        "aesop",
        "all_goals",
        "any_goals",
        "apply",
        "assumption",
        "by_cases",
        "by_contra",
        "calc",
        "case",
        "cases",
        "change",
        "classical",
        "constructor",
        "contradiction",
        "convert",
        "convert!",
        "decide",
        "dsimp",
        "exact",
        "exact_mod_cast",
        "ext",
        "first",
        "focus",
        "fun_prop",
        "grind",
        "have",
        "induction",
        "intro",
        "intros",
        "left",
        "let",
        "linarith",
        "native_decide",
        "next",
        "norm_num",
        "obtain",
        "omega",
        "positivity",
        "rcases",
        "refine",
        "rename_i",
        "repeat",
        "rfl",
        "right",
        "ring",
        "ring_nf",
        "rintro",
        "rw",
        "show",
        "simpa",
        "simp",
        "simp_all",
        "simp_rw",
        "specialize",
        "subst",
        "suffices",
        "trivial",
        "unfold",
    }
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dataset_v3_statement_id(declaration: str) -> str:
    fingerprint = statement_fingerprint_v2(declaration)
    payload = f"dataset-v3-statement-id-v1\0{fingerprint}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_v3_proof_variant_id(
    statement_id: str,
    proof_text: str,
    *,
    source_repository: str,
    source_revision: str,
    source_file: str,
) -> str:
    payload = (
        f"dataset-v3-proof-variant-v1\0{statement_id}\0{_sha256_text(proof_text)}"
        f"\0{source_repository}\0{source_revision}\0{source_file}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_v3_example_id(
    statement_id: str,
    proof_variant_id: str,
    kind: str,
    boundary_id: str | None,
) -> str:
    payload = "\0".join(
        (
            "dataset-v3-example-v1",
            statement_id,
            proof_variant_id,
            kind,
            boundary_id or "whole",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strip_redundant_outer_parentheses(
    tokens: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    while len(tokens) >= 2 and tokens[0] == ("symbol", "("):
        depth = 0
        matching: int | None = None
        for index, (kind, token) in enumerate(tokens):
            if kind != "symbol":
                continue
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
                if depth == 0:
                    matching = index
                    break
        if matching != len(tokens) - 1:
            break
        tokens = tokens[1:-1]
    return tokens


def normalized_proof_structure(
    proof_text: str, resolved_dependencies: Sequence[str] = ()
) -> str:
    """Normalize cosmetic proof transport while retaining proof-token order.

    The normalizer ignores comments/whitespace, collapses a source term and its
    mechanically wrapped ``by exact (...)`` transport to one representation,
    removes grouping parentheses, and maps unique qualified/unqualified source
    dependency aliases to stable slots. It is intentionally a diversity
    fingerprint, not a claim of Lean definitional equality.
    """

    tokens = _lex_lean(proof_text)
    if len(tokens) >= 2 and tokens[:2] == [
        ("identifier", "by"),
        ("identifier", "exact"),
    ]:
        tokens = _strip_redundant_outer_parentheses(tokens[2:])
    else:
        tokens = _strip_redundant_outer_parentheses(tokens)

    basenames: dict[str, list[str]] = defaultdict(list)
    for dependency in sorted(set(resolved_dependencies)):
        basenames[dependency.rsplit(".", 1)[-1]].append(dependency)
    aliases: dict[str, str] = {}
    for dependency in sorted(set(resolved_dependencies)):
        slot = "__DEPENDENCY_" + hashlib.sha256(
            dependency.encode("utf-8")
        ).hexdigest()[:24] + "__"
        aliases[dependency] = slot
        basename = dependency.rsplit(".", 1)[-1]
        if len(basenames[basename]) == 1:
            aliases[basename] = slot

    normalized: list[tuple[str, str]] = []
    for kind, token in tokens:
        if kind == "symbol" and token in {"(", ")"}:
            continue
        normalized.append((kind, aliases.get(token, token)))
    return "\x1f".join(f"{kind}:{token}" for kind, token in normalized)


def structural_proof_fingerprint(
    proof_text: str, resolved_dependencies: Sequence[str] = ()
) -> str:
    normalized = normalized_proof_structure(proof_text, resolved_dependencies)
    payload = f"{STRUCTURAL_PROOF_FINGERPRINT_VERSION}\0{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contains_placeholder(value: str) -> bool:
    return any(
        kind == "identifier" and token in {"sorry", "admit"}
        for kind, token in _lex_lean(value)
    )


def proof_form(source_expression: str, *, generated: bool = False) -> str:
    stripped = source_expression.lstrip()
    if generated:
        return "generated-tactic"
    if stripped == "by" or stripped.startswith("by\n") or stripped.startswith("by "):
        return "tactic"
    if stripped.startswith("|"):
        return "equations"
    if stripped == "where" or stripped.startswith("where\n") or stripped.startswith("where "):
        return "where"
    return "term"


def first_proof_construct(proof_text: str) -> str:
    tokens = _lex_lean(proof_text)
    if not tokens:
        return "empty"
    if tokens[0] == ("identifier", "by"):
        tokens = tokens[1:]
    if not tokens:
        return "empty"
    kind, token = tokens[0]
    if token == "·":
        return "branch"
    if kind == "identifier":
        return token if token in _TACTIC_HEADS else "term"
    if token in {"⟨", "{", "["}:
        return "constructor-term"
    return "term"


def _line_start_states(value: str) -> dict[int, bool]:
    """Return whether each line starts outside comments, strings and delimiters."""

    states: dict[int, bool] = {0: True}
    opening = {"(": ")", "[": "]", "{": "}", "⦃": "⦄"}
    closing: list[str] = []
    block_depth = 0
    in_string = False
    escaped = False
    in_line_comment = False
    index = 0
    while index < len(value):
        pair = value[index : index + 2]
        char = value[index]
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                states[index + 1] = not block_depth and not in_string and not closing
            index += 1
            continue
        if block_depth:
            if pair == "/-":
                block_depth += 1
                index += 2
            elif pair == "-/":
                block_depth -= 1
                index += 2
            else:
                if char == "\n":
                    states[index + 1] = False
                index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            if char == "\n":
                states[index + 1] = False
            index += 1
            continue
        if pair == "--":
            in_line_comment = True
            index += 2
            continue
        if pair == "/-":
            block_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True
        elif char in opening:
            closing.append(opening[char])
        elif closing and char == closing[-1]:
            closing.pop()
        if char == "\n":
            states[index + 1] = not block_depth and not in_string and not closing
        index += 1
    return states


def _line_head(stripped_line: str) -> str:
    if stripped_line.startswith("·"):
        return "·"
    match = re.match(r"([^\s()\[\]{}]+)", stripped_line)
    return "" if match is None else match.group(1)


def structural_boundary_offsets(proof_text: str) -> list[tuple[int, str]]:
    """Find conservative Lean-layout boundaries between complete tactic units.

    Cuts are only emitted for multi-line ``by`` proofs at the indentation of the
    first tactic, when the line begins outside comments, strings and bracketed
    syntax and starts with a recognized tactic or branch command. Nested tactic
    bodies remain attached to their owning top-level unit.
    """

    normalized = proof_text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized != proof_text:
        raise ValueError("proof transport must be newline-normalized before segmentation")
    if "\t" in proof_text:
        return []
    stripped = proof_text.lstrip()
    if not (stripped == "by" or stripped.startswith("by\n") or stripped.startswith("by ")):
        return []
    leading = len(proof_text) - len(stripped)
    by_end = leading + 2
    if by_end >= len(proof_text) or "\n" not in proof_text[by_end:]:
        return []

    states = _line_start_states(proof_text)
    lines = proof_text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    commands: list[tuple[int, str]] = []
    base_indent: int | None = None
    for offset, line in zip(offsets, lines, strict=True):
        if offset <= by_end or not states.get(offset, False):
            continue
        content = line.rstrip("\r\n")
        stripped_line = content.lstrip(" ")
        if not stripped_line or stripped_line.startswith(("--", "/-", "-/")):
            continue
        indent = len(content) - len(stripped_line)
        head = _line_head(stripped_line)
        recognized = head == "·" or head in _TACTIC_HEADS
        if not recognized:
            continue
        if base_indent is None:
            base_indent = indent
        if indent != base_indent:
            continue
        kind = "branch" if head in {"·", "case", "next"} else "top-level-tactic"
        commands.append((offset, kind))

    if len(commands) < 2:
        return []
    return commands[1:]


def build_boundaries(
    proof_text: str, proof_variant_id: str
) -> tuple[StructuralBoundary, ...]:
    boundaries: list[StructuralBoundary] = []
    reconstruction_hash = _sha256_text(proof_text)
    for segment_index, (offset, kind) in enumerate(
        structural_boundary_offsets(proof_text), start=1
    ):
        payload = (
            f"dataset-v3-boundary-v1\0{proof_variant_id}\0{offset}\0{kind}"
        )
        boundaries.append(
            StructuralBoundary(
                boundary_id=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                prefix_end=offset,
                prefix_sha256=_sha256_text(proof_text[:offset]),
                continuation_sha256=_sha256_text(proof_text[offset:]),
                reconstruction_sha256=reconstruction_hash,
                structural_kind=kind,
                segment_index=segment_index,
                extractor_version=BOUNDARY_EXTRACTOR_VERSION,
            )
        )
    return tuple(boundaries)


def source_expression_in_pinned_file(source_expression: str, source_path: Path) -> bool:
    if not source_path.is_file():
        return False
    source = source_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    expression = source_expression.replace("\r\n", "\n").replace("\r", "\n").strip()
    return bool(expression) and expression in source


def convert_v2_record(
    value: Mapping[str, Any],
    *,
    source_expression_verifier: Callable[[Mapping[str, Any]], bool],
) -> DatasetV3Record | None:
    """Recover a real v3 record from v2's pinned raw-source index.

    V2 synthetic/evaluation rows are deliberately not inherited. For real rows,
    raw ``source_expression`` is the optimizer-visible proof whenever it is a
    standalone expression. Only declaration equation/``where`` forms retain the
    already Lean-verified technical standalone conversion.
    """

    if value.get("role") != "training" or value.get("provenance") == "synthetic":
        return None
    declaration = str(value["canonical_declaration"])
    statement_id = dataset_v3_statement_id(declaration)
    environment = EnvironmentContext.from_dict(dict(value["environment"]))
    variants: list[DatasetV3ProofVariant] = []
    for source_variant in value["proof_variants"]:
        source_expression = str(source_variant["source_expression"]).replace(
            "\r\n", "\n"
        ).replace("\r", "\n").strip()
        old_transformation = str(source_variant["transformation_kind"])
        form = proof_form(source_expression)
        if old_transformation in {"equations-to-fun-exact", "where-to-structure-exact"}:
            proof_text = str(source_variant["canonical_proof"])
            transformation_kind = old_transformation
            transformation_reason = (
                "Lean declaration equation/where syntax requires a standalone proof "
                "expression for deterministic theorem-plus-proof materialization"
            )
        else:
            proof_text = source_expression
            transformation_kind = "none"
            transformation_reason = None
        if contains_placeholder(proof_text):
            raise ValueError(
                f"placeholder reached Dataset-v3 source variant: {statement_id}"
            )
        source_verified = source_expression_verifier(source_variant)
        if not source_verified:
            raise ValueError(
                "raw proof expression was not recovered from its pinned source: "
                f"{source_variant['source_file']}:{source_variant['source_declaration_name']}"
            )
        proof_variant_id = dataset_v3_proof_variant_id(
            statement_id,
            proof_text,
            source_repository=str(source_variant["source_repository"]),
            source_revision=str(source_variant["source_revision"]),
            source_file=str(source_variant["source_file"]),
        )
        dependencies = tuple(
            sorted(set(str(item) for item in source_variant.get("resolved_dependencies", [])))
        )
        verification = ProofVerification.from_dict(dict(source_variant["verification"]))
        variant = DatasetV3ProofVariant(
            proof_variant_id=proof_variant_id,
            source_expression=source_expression,
            proof_text=proof_text,
            proof_form=form,  # type: ignore[arg-type]
            transformation_kind=transformation_kind,
            transformation_reason=transformation_reason,
            exact_text_fingerprint=_sha256_text(proof_text),
            structural_fingerprint=structural_proof_fingerprint(
                proof_text, dependencies
            ),
            resolved_dependencies=dependencies,
            boundaries=build_boundaries(proof_text, proof_variant_id),
            verification=ProofVerification(
                status=verification.status,
                environment_id=verification.environment_id,
                method="pinned-source-expression-recovery+" + verification.method,
                evidence_id=verification.evidence_id,
                diagnostic=verification.diagnostic,
            ),
            source_declaration_name=str(source_variant["source_declaration_name"]),
            source_repository=str(source_variant["source_repository"]),
            source_revision=str(source_variant["source_revision"]),
            source_file=str(source_variant["source_file"]),
            source_expression_verified=True,
        )
        variants.append(variant)
    record = DatasetV3Record(
        schema_version=DATASET_V3_SCHEMA_VERSION,
        statement_id=statement_id,
        canonical_declaration=declaration,
        normalized_statement_fingerprint=statement_fingerprint_v2(declaration),
        role="training",
        theorem_mass_numerator=1,
        theorem_mass_denominator=1,
        provenance=str(value["provenance"]),
        environment=environment,
        proof_variants=tuple(sorted(variants, key=lambda item: item.proof_variant_id)),
        topic_tags=tuple(str(item) for item in value.get("topic_tags", [])),
        memberships=tuple(str(item) for item in value.get("memberships", [])),
    )
    record.validate()
    return record


def representative_variants(
    record: DatasetV3Record,
) -> tuple[DatasetV3ProofVariant, ...]:
    by_structure: dict[str, DatasetV3ProofVariant] = {}
    for variant in record.proof_variants:
        current = by_structure.get(variant.structural_fingerprint)
        if current is None or variant.proof_variant_id < current.proof_variant_id:
            by_structure[variant.structural_fingerprint] = variant
    return tuple(sorted(by_structure.values(), key=lambda item: item.proof_variant_id))


def plan_optimizer_examples(
    record: DatasetV3Record,
    *,
    whole_mass: Fraction,
) -> tuple[DerivedExampleRef, ...]:
    if record.role != "training":
        return ()
    if not Fraction(0, 1) <= whole_mass <= Fraction(1, 1):
        raise ValueError("whole-proof mass must be between zero and one")
    theorem_mass = Fraction(
        record.theorem_mass_numerator, record.theorem_mass_denominator
    )
    variants = representative_variants(record)
    incremental_variants = tuple(item for item in variants if item.boundaries)
    whole_budget = theorem_mass if not incremental_variants else theorem_mass * whole_mass
    continuation_budget = theorem_mass - whole_budget
    examples: list[DerivedExampleRef] = []

    if whole_budget:
        per_variant = whole_budget / len(variants)
        for variant in variants:
            examples.append(
                DerivedExampleRef(
                    schema_version=DATASET_V3_VIEW_SCHEMA_VERSION,
                    example_id=dataset_v3_example_id(
                        record.statement_id, variant.proof_variant_id, "whole", None
                    ),
                    statement_id=record.statement_id,
                    proof_variant_id=variant.proof_variant_id,
                    kind="whole",
                    boundary_id=None,
                    mass_numerator=per_variant.numerator,
                    mass_denominator=per_variant.denominator,
                )
            )
    if continuation_budget:
        per_variant = continuation_budget / len(incremental_variants)
        for variant in incremental_variants:
            per_boundary = per_variant / len(variant.boundaries)
            for boundary in variant.boundaries:
                examples.append(
                    DerivedExampleRef(
                        schema_version=DATASET_V3_VIEW_SCHEMA_VERSION,
                        example_id=dataset_v3_example_id(
                            record.statement_id,
                            variant.proof_variant_id,
                            "continuation",
                            boundary.boundary_id,
                        ),
                        statement_id=record.statement_id,
                        proof_variant_id=variant.proof_variant_id,
                        kind="continuation",
                        boundary_id=boundary.boundary_id,
                        mass_numerator=per_boundary.numerator,
                        mass_denominator=per_boundary.denominator,
                    )
                )
    observed = sum(
        (Fraction(item.mass_numerator, item.mass_denominator) for item in examples),
        Fraction(0, 1),
    )
    if observed != theorem_mass:
        raise ValueError(
            f"statement-normalized mass mismatch for {record.statement_id}: "
            f"{observed} != {theorem_mass}"
        )
    for example in examples:
        example.validate()
    return tuple(sorted(examples, key=lambda item: item.example_id))


def materialize_example(
    record: DatasetV3Record,
    example: DerivedExampleRef,
    *,
    max_utf8_bytes: int | None = None,
) -> dict[str, Any]:
    if record.statement_id != example.statement_id:
        raise ValueError("example resolves to a different statement")
    variants = {
        variant.proof_variant_id: variant for variant in record.proof_variants
    }
    try:
        variant = variants[example.proof_variant_id]
    except KeyError as error:
        raise ValueError("example proof variant does not resolve") from error
    if example.kind == "whole":
        prefix = ""
        target = variant.proof_text
    else:
        boundaries = {
            boundary.boundary_id: boundary for boundary in variant.boundaries
        }
        try:
            boundary = boundaries[str(example.boundary_id)]
        except KeyError as error:
            raise ValueError("example boundary does not resolve") from error
        prefix = variant.proof_text[: boundary.prefix_end]
        target = variant.proof_text[boundary.prefix_end :]
        if prefix + target != variant.proof_text:
            raise ValueError("materialized continuation does not reconstruct its proof")
    model_input = f"{record.canonical_declaration} := {prefix}"
    byte_length = len((model_input + target).encode("utf-8"))
    if max_utf8_bytes is not None and byte_length > max_utf8_bytes:
        raise ValueError(
            "included Dataset-v3 example exceeds the configured context; "
            "refusing to truncate or omit it"
        )
    return {
        "schema_version": "lean-proof-continuation-v3-materialized-v1",
        "example_id": example.example_id,
        "statement_id": record.statement_id,
        "proof_variant_id": variant.proof_variant_id,
        "task_kind": example.kind,
        "declaration": record.canonical_declaration,
        "proof_prefix": prefix,
        "model_input": model_input,
        "target": target,
        "mass_numerator": example.mass_numerator,
        "mass_denominator": example.mass_denominator,
        "utf8_bytes": byte_length,
    }


def validate_split_isolation(records: Iterable[DatasetV3Record]) -> dict[str, int]:
    statement_roles: dict[str, set[str]] = defaultdict(set)
    exact_proof_roles: dict[str, set[str]] = defaultdict(set)
    structural_proof_roles: dict[str, set[str]] = defaultdict(set)
    derivation_roles: dict[str, set[str]] = defaultdict(set)
    for record in records:
        statement_roles[record.normalized_statement_fingerprint].add(record.role)
        for variant in record.proof_variants:
            exact_proof_roles[variant.exact_text_fingerprint].add(record.role)
            structural_proof_roles[variant.structural_fingerprint].add(record.role)
        if record.derivation_family_fingerprint:
            derivation_roles[record.derivation_family_fingerprint].add(record.role)
    crossings = {
        "cross_role_statements": sum(
            len(roles) > 1 for roles in statement_roles.values()
        ),
        "cross_role_exact_proofs": sum(
            len(roles) > 1 for roles in exact_proof_roles.values()
        ),
        "cross_role_structural_proofs": sum(
            len(roles) > 1 for roles in structural_proof_roles.values()
        ),
        "cross_role_derivation_families": sum(
            len(roles) > 1 for roles in derivation_roles.values()
        ),
    }
    if any(crossings.values()):
        raise ValueError(f"Dataset-v3 split leakage: {crossings}")
    return crossings


def validate_no_placeholders(records: Iterable[DatasetV3Record]) -> dict[str, int]:
    checked = 0
    for record in records:
        if record.role != "training":
            continue
        for variant in record.proof_variants:
            checked += 1
            if contains_placeholder(variant.proof_text):
                raise ValueError(
                    f"placeholder reached optimizer membership: {variant.proof_variant_id}"
                )
    return {"optimizer_proof_variants_checked": checked, "placeholders": 0}


def write_records(path: Path, records: Iterable[DatasetV3Record]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="\n") as handle:
                for record in records:
                    record.validate()
                    handle.write(
                        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    )
                    handle.write("\n")
    return sha256_file(path)


def read_records(path: Path) -> Iterator[DatasetV3Record]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield DatasetV3Record.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid Dataset-v3 record at {path}:{line_number}: {error}"
                ) from error


def write_view(path: Path, examples: Iterable[DerivedExampleRef]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="\n") as handle:
                for example in examples:
                    example.validate()
                    handle.write(
                        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True)
                    )
                    handle.write("\n")
    return sha256_file(path)


def read_view(path: Path) -> Iterator[DerivedExampleRef]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield DerivedExampleRef.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid Dataset-v3 view at {path}:{line_number}: {error}"
                ) from error


def iter_v2_json(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
