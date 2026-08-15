from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .phase2_schema import (
    PHASE2_DATASET_SCHEMA_VERSION,
    SPLIT_NAMES,
    DatasetSplit,
    MathlibProofRecord,
    SourcePosition,
    SourceSpan,
    TokenLengths,
)

FINGERPRINT_ALGORITHM = "lean-lexical-statement-v1-sha256"
COMPONENT_ALGORITHM = "source-file-statement-components-v1-sha256"
TOKEN_THRESHOLDS = (1024, 2048, 4096, 8192)
REQUIRED_RECORD_FIELDS = frozenset(MathlibProofRecord.__dataclass_fields__)


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


@dataclass(frozen=True)
class ExtractedRecord:
    id: str
    source_repository: str
    source_revision: str
    file_path: str
    declaration_name: str
    declaration_kind: str
    source_span: SourceSpan
    declaration_span: SourceSpan
    proof_span: SourceSpan
    declaration: str
    proof: str
    completion: str
    premises: tuple[str, ...]
    statement_fingerprint: str


@dataclass(frozen=True)
class RawTheorem:
    file_path: str
    declaration_name: str
    declaration_kind: str
    source_span: SourceSpan | None
    declaration_span: SourceSpan | None
    proof_span: SourceSpan | None
    declaration: str | None
    proof: str | None
    premises: tuple[str, ...] = ()
    is_private: bool = False


@dataclass(frozen=True)
class EligibilityResult:
    record: ExtractedRecord | None
    filter_reason: str | None
    detail: str | None = None


def strip_lean_comments(source: str) -> str:
    """Remove nested Lean comments while preserving strings and line structure."""

    result: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    in_quoted_identifier = False
    while index < len(source):
        char = source[index]
        following = source[index : index + 2]
        if block_depth:
            if following == "/-":
                block_depth += 1
                result.extend("  ")
                index += 2
            elif following == "-/":
                block_depth -= 1
                result.extend("  ")
                index += 2
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if in_string:
            result.append(char)
            if char == "\\" and index + 1 < len(source):
                result.append(source[index + 1])
                index += 2
            else:
                if char == '"':
                    in_string = False
                index += 1
            continue
        if in_quoted_identifier:
            result.append(char)
            if char == "»":
                in_quoted_identifier = False
            index += 1
            continue
        if following == "--":
            newline = source.find("\n", index + 2)
            if newline == -1:
                result.extend(" " * (len(source) - index))
                break
            result.extend(" " * (newline - index))
            index = newline
            continue
        if following == "/-":
            block_depth = 1
            result.extend("  ")
            index += 2
            continue
        result.append(char)
        if char == '"':
            in_string = True
        elif char == "«":
            in_quoted_identifier = True
        index += 1
    if block_depth:
        raise ValueError("unterminated Lean block comment")
    return "".join(result)


def _lex_lean(source: str) -> list[tuple[str, str]]:
    source = strip_lean_comments(source)
    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if char == '"':
            end = index + 1
            while end < len(source):
                if source[end] == "\\":
                    end += 2
                    continue
                end += 1
                if source[end - 1] == '"':
                    break
            if end > len(source) or source[end - 1] != '"':
                raise ValueError("unterminated Lean string literal")
            tokens.append(("string", source[index:end]))
            index = end
            continue
        if char == "«":
            end = source.find("»", index + 1)
            if end == -1:
                raise ValueError("unterminated quoted Lean identifier")
            tokens.append(("identifier", source[index : end + 1]))
            index = end + 1
            continue
        if char.isalpha() or char == "_":
            end = index + 1
            while end < len(source):
                candidate = source[end]
                if candidate == ".":
                    if end + 1 < len(source) and (
                        source[end + 1].isalpha() or source[end + 1] == "_"
                    ):
                        end += 1
                        continue
                    break
                if not (
                    candidate.isalnum()
                    or candidate in "_'?!"
                    or unicodedata.category(candidate) in {"Mn", "Mc"}
                ):
                    break
                end += 1
            tokens.append(("identifier", source[index:end]))
            index = end
            continue
        if char.isdigit():
            end = index + 1
            while end < len(source) and (source[end].isalnum() or source[end] in "_."):
                end += 1
            tokens.append(("number", source[index:end]))
            index = end
            continue
        tokens.append(("symbol", char))
        index += 1
    return tokens


def canonical_declaration(value: str) -> str:
    declaration = value.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    declaration = re.sub(r"\s*:=\s*$", "", declaration).rstrip()
    if not declaration:
        raise ValueError("empty declaration")
    return declaration


def _declaration_name_index(tokens: Sequence[tuple[str, str]]) -> int:
    bracket_depth = 0
    declaration_index: int | None = None
    for index, (kind, token) in enumerate(tokens):
        if kind == "symbol" and token in "([{⦃":
            bracket_depth += 1
        elif kind == "symbol" and token in ")] }⦄".replace(" ", ""):
            bracket_depth = max(0, bracket_depth - 1)
        elif (
            bracket_depth == 0
            and kind == "identifier"
            and token in {"theorem", "lemma"}
        ):
            declaration_index = index
            break
    if declaration_index is None or declaration_index + 1 >= len(tokens):
        raise ValueError("could not locate theorem/lemma declaration identifier")
    name_kind, _ = tokens[declaration_index + 1]
    if name_kind != "identifier":
        raise ValueError("ambiguous theorem/lemma declaration identifier")
    return declaration_index + 1


def declaration_identifier(value: str) -> str:
    tokens = _lex_lean(canonical_declaration(value))
    return tokens[_declaration_name_index(tokens)][1]


def normalized_statement(value: str) -> str:
    tokens = _lex_lean(canonical_declaration(value))
    name_index = _declaration_name_index(tokens)
    tokens[name_index - 1] = ("identifier", "theorem_or_lemma")
    tokens[name_index] = ("identifier", "__DECLARATION_NAME__")
    return "\x1f".join(f"{kind}:{token}" for kind, token in tokens)


def normalized_lean_code(value: str) -> str:
    return "\x1f".join(f"{kind}:{token}" for kind, token in _lex_lean(value))


def statement_fingerprint(value: str) -> str:
    normalized = normalized_statement(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def position_offset(source: str, position: SourcePosition) -> int:
    lines = source.splitlines(keepends=True)
    if (
        position.line == len(lines) + 1
        and position.column == 1
        and source.endswith("\n")
    ):
        return len(source)
    if position.line < 1 or position.line > len(lines):
        raise ValueError(f"source line {position.line} is outside the file")
    line = lines[position.line - 1]
    content = line.removesuffix("\n")
    if position.column < 1 or position.column > len(content) + 1:
        raise ValueError(
            f"source column {position.column} is outside line {position.line}"
        )
    return sum(len(item) for item in lines[: position.line - 1]) + position.column - 1


def source_slice(source: str, span: SourceSpan) -> str:
    start = position_offset(source, span.start)
    end = position_offset(source, span.end)
    if start >= end:
        raise ValueError("source span is empty or reversed")
    return source[start:end]


def substitute_span(source: str, span: SourceSpan, replacement: str) -> str:
    start = position_offset(source, span.start)
    end = position_offset(source, span.end)
    if start >= end:
        raise ValueError("replacement span is empty or reversed")
    return source[:start] + replacement + source[end:]


def validate_record_source_text(
    record: ExtractedRecord | MathlibProofRecord, source: str
) -> None:
    if record.declaration_span.start != record.source_span.start:
        raise ValueError(
            "declaration span does not begin at the declaration source span"
        )
    if record.declaration_span.end != record.proof_span.start:
        raise ValueError("declaration span does not end at the proof span")
    if not (
        record.source_span.start <= record.proof_span.start
        and record.proof_span.end <= record.source_span.end
    ):
        raise ValueError("proof span is outside the declaration source span")
    source_declaration = source_slice(source, record.declaration_span)
    source_fingerprint = statement_fingerprint(
        canonical_declaration(source_declaration)
    )
    if source_fingerprint != record.statement_fingerprint:
        raise ValueError("source declaration fingerprint differs from the record")
    source_identifier = (
        declaration_identifier(source_declaration).strip("«»").removeprefix("_root_.")
    )
    full_name = record.declaration_name.removeprefix("_root_.")
    if not (
        full_name == source_identifier or full_name.endswith(f".{source_identifier}")
    ):
        raise ValueError(
            f"source identifier {source_identifier!r} does not match "
            f"{record.declaration_name!r}"
        )
    source_proof = source_slice(source, record.proof_span)
    if normalized_lean_code(source_proof) != normalized_lean_code(record.proof):
        raise ValueError("source proof span differs from the retained raw proof")


def derive_completion(proof: str) -> str:
    if not proof.startswith("by") or (len(proof) > 2 and not proof[2].isspace()):
        raise ValueError("proof does not begin with the `by` delimiter")
    completion = proof[2:].lstrip()
    if not completion:
        raise ValueError("proof continuation is empty")
    return completion


def _contains_placeholder(proof: str) -> bool:
    return any(
        kind == "identifier" and token in {"sorry", "admit"}
        for kind, token in _lex_lean(proof)
    )


def _valid_span(span: SourceSpan | None) -> bool:
    return bool(
        span is not None
        and span.start.line > 0
        and span.start.column > 0
        and span.end.line > 0
        and span.end.column > 0
        and span.start < span.end
    )


def evaluate_raw_theorem(
    raw: RawTheorem,
    *,
    source_repository: str,
    source_revision: str,
) -> EligibilityResult:
    if raw.is_private:
        return EligibilityResult(None, "private")
    if not raw.declaration_name or raw.declaration_name.startswith("_private."):
        return EligibilityResult(None, "unstable_name")
    if raw.declaration_kind not in {"theorem", "lemma"}:
        return EligibilityResult(None, "unsupported_kind", raw.declaration_kind)
    if not raw.file_path.startswith("Mathlib/") or not raw.file_path.endswith(".lean"):
        return EligibilityResult(None, "outside_source_scope", raw.file_path)
    if not all(
        _valid_span(span)
        for span in (raw.source_span, raw.declaration_span, raw.proof_span)
    ):
        return EligibilityResult(None, "ambiguous_source_span")
    if raw.declaration is None:
        return EligibilityResult(None, "missing_declaration")
    try:
        declaration_tokens = _lex_lean(raw.declaration)
        name_index = _declaration_name_index(declaration_tokens)
    except ValueError as error:
        return EligibilityResult(None, "ambiguous_declaration", str(error))
    if any(
        kind == "identifier" and token == "private"
        for kind, token in declaration_tokens[: name_index - 1]
    ):
        return EligibilityResult(None, "private")
    if raw.proof is None:
        return EligibilityResult(None, "non_tactic_proof")
    proof = raw.proof.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    try:
        completion = derive_completion(proof)
    except ValueError as error:
        reason = "empty_completion" if "empty" in str(error) else "non_by_proof"
        return EligibilityResult(None, reason, str(error))
    if _contains_placeholder(proof):
        return EligibilityResult(None, "proof_placeholder")
    if not raw.declaration.rstrip().endswith(":="):
        return EligibilityResult(None, "unsupported_proof_delimiter")
    try:
        declaration = canonical_declaration(raw.declaration)
        fingerprint = statement_fingerprint(declaration)
    except ValueError as error:
        return EligibilityResult(None, "ambiguous_declaration", str(error))
    identity = (
        f"{PHASE2_DATASET_SCHEMA_VERSION}\0{source_revision}\0"
        f"{raw.file_path}\0{raw.declaration_name}"
    )
    record = ExtractedRecord(
        id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        source_repository=source_repository,
        source_revision=source_revision,
        file_path=raw.file_path,
        declaration_name=raw.declaration_name,
        declaration_kind=raw.declaration_kind,
        source_span=raw.source_span,  # type: ignore[arg-type]
        declaration_span=raw.declaration_span,  # type: ignore[arg-type]
        proof_span=raw.proof_span,  # type: ignore[arg-type]
        declaration=declaration,
        proof=proof,
        completion=completion,
        premises=tuple(sorted({name for name in raw.premises if name})),
        statement_fingerprint=fingerprint,
    )
    try:
        json.dumps(record.__dict__, default=lambda item: item.to_dict())
    except (TypeError, ValueError) as error:
        return EligibilityResult(None, "serialization_error", str(error))
    return EligibilityResult(record, None)


_MINIF2F_THEOREM_PATTERN = re.compile(
    r"(?ms)^theorem\s+(?P<name>[^\s(:]+)(?P<tail>.*?)\s*:=\s*by\n\s+sorry\s*(?=\n|\Z)"
)


def minif2f_statement_fingerprints(sources: Iterable[str]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for source in sources:
        source_without_comments = strip_lean_comments(source)
        for match in _MINIF2F_THEOREM_PATTERN.finditer(source_without_comments):
            name = match.group("name")
            if ".variants." in name:
                continue
            declaration = canonical_declaration(f"theorem {name}{match.group('tail')}")
            fingerprints[name] = statement_fingerprint(declaration)
    if not fingerprints:
        raise ValueError("no primary miniF2F theorem declarations were found")
    return fingerprints


def exclude_contamination(
    records: Sequence[ExtractedRecord], excluded_fingerprints: set[str]
) -> tuple[list[ExtractedRecord], list[ExtractedRecord]]:
    retained: list[ExtractedRecord] = []
    excluded: list[ExtractedRecord] = []
    for record in records:
        (
            excluded
            if record.statement_fingerprint in excluded_fingerprints
            else retained
        ).append(record)
    return retained, excluded


def exclude_ambiguous_record_identities(
    records: Sequence[ExtractedRecord],
) -> tuple[list[ExtractedRecord], list[ExtractedRecord], set[str]]:
    identity_counts = Counter(record.id for record in records)
    ambiguous_ids = {
        identity for identity, count in identity_counts.items() if count > 1
    }
    retained = [record for record in records if record.id not in ambiguous_ids]
    excluded = [record for record in records if record.id in ambiguous_ids]
    return retained, excluded, ambiguous_ids


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def build_file_components(
    records: Sequence[ExtractedRecord],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    files = sorted({record.file_path for record in records})
    disjoint_set = _DisjointSet(files)
    fingerprint_files: dict[str, set[str]] = defaultdict(set)
    for record in records:
        fingerprint_files[record.statement_fingerprint].add(record.file_path)
    for fingerprint in sorted(fingerprint_files):
        members = sorted(fingerprint_files[fingerprint])
        for member in members[1:]:
            disjoint_set.union(members[0], member)
    root_files: dict[str, list[str]] = defaultdict(list)
    for file_path in files:
        root_files[disjoint_set.find(file_path)].append(file_path)
    components: dict[str, tuple[str, ...]] = {}
    file_components: dict[str, str] = {}
    for members in sorted(tuple(sorted(group)) for group in root_files.values()):
        digest = hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()
        component_id = digest
        components[component_id] = members
        for file_path in members:
            file_components[file_path] = component_id
    return file_components, components


def assign_component_splits(
    component_sizes: Mapping[str, int],
    proportions: Mapping[str, float],
    *,
    seed: str,
) -> dict[str, DatasetSplit]:
    if set(proportions) != set(SPLIT_NAMES):
        raise ValueError(f"split proportions must define {SPLIT_NAMES}")
    if not math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split proportions must sum to one")
    if any(value <= 0 for value in proportions.values()):
        raise ValueError("split proportions must be positive")
    total = sum(component_sizes.values())
    if total <= 0:
        raise ValueError("cannot split an empty corpus")
    targets = {name: total * proportions[name] for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    ordered_components = sorted(
        component_sizes,
        key=lambda component: (
            -component_sizes[component],
            hashlib.sha256(f"{seed}\0{component}".encode()).hexdigest(),
        ),
    )
    assignments: dict[str, DatasetSplit] = {}
    for component in ordered_components:
        split = min(
            SPLIT_NAMES,
            key=lambda name: (counts[name] / targets[name], SPLIT_NAMES.index(name)),
        )
        assignments[component] = split
        counts[split] += component_sizes[component]
    return assignments


def token_lengths(record: ExtractedRecord, tokenizer: Tokenizer) -> TokenLengths:
    def count(value: str) -> int:
        return len(tokenizer.encode(value, add_special_tokens=False))

    declaration_and_proof = f"{record.declaration} := {record.proof}"
    declaration_and_completion = f"{record.declaration} := by\n{record.completion}"
    return TokenLengths(
        declaration=count(record.declaration),
        proof=count(record.proof),
        completion=count(record.completion),
        declaration_and_proof=count(declaration_and_proof),
        declaration_and_completion=count(declaration_and_completion),
    )


def finalize_records(
    records: Sequence[ExtractedRecord],
    tokenizer: Tokenizer,
    proportions: Mapping[str, float],
    *,
    seed: str,
) -> list[MathlibProofRecord]:
    file_components, _ = build_file_components(records)
    component_sizes = Counter(file_components[record.file_path] for record in records)
    assignments = assign_component_splits(component_sizes, proportions, seed=seed)
    finalized = [
        MathlibProofRecord(
            schema_version=PHASE2_DATASET_SCHEMA_VERSION,
            id=record.id,
            source_repository=record.source_repository,
            source_revision=record.source_revision,
            file_path=record.file_path,
            declaration_name=record.declaration_name,
            declaration_kind=record.declaration_kind,  # type: ignore[arg-type]
            source_span=record.source_span,
            declaration_span=record.declaration_span,
            proof_span=record.proof_span,
            declaration=record.declaration,
            proof=record.proof,
            completion=record.completion,
            premises=record.premises,
            file_group=record.file_path,
            component_id=file_components[record.file_path],
            split=assignments[file_components[record.file_path]],
            statement_fingerprint=record.statement_fingerprint,
            token_lengths=token_lengths(record, tokenizer),
        )
        for record in records
    ]
    return sorted(
        finalized,
        key=lambda record: (
            SPLIT_NAMES.index(record.split),
            record.file_path,
            record.source_span.start,
            record.declaration_name,
        ),
    )


def validate_split_hygiene(
    records: Sequence[MathlibProofRecord],
) -> dict[str, Any]:
    file_splits: dict[str, set[str]] = defaultdict(set)
    file_components: dict[str, set[str]] = defaultdict(set)
    component_splits: dict[str, set[str]] = defaultdict(set)
    fingerprint_splits: dict[str, set[str]] = defaultdict(set)
    fingerprint_components: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.file_group != record.file_path:
            raise ValueError(f"record {record.id} has a non-file structural group")
        file_splits[record.file_path].add(record.split)
        file_components[record.file_path].add(record.component_id)
        component_splits[record.component_id].add(record.split)
        fingerprint_splits[record.statement_fingerprint].add(record.split)
        fingerprint_components[record.statement_fingerprint].add(record.component_id)
    crossing_files = sorted(
        key for key, splits in file_splits.items() if len(splits) > 1
    )
    files_in_multiple_components = sorted(
        key for key, components in file_components.items() if len(components) > 1
    )
    crossing_components = sorted(
        key for key, splits in component_splits.items() if len(splits) > 1
    )
    crossing_fingerprints = sorted(
        key for key, splits in fingerprint_splits.items() if len(splits) > 1
    )
    fingerprints_in_multiple_components = sorted(
        key for key, components in fingerprint_components.items() if len(components) > 1
    )
    if (
        crossing_files
        or files_in_multiple_components
        or crossing_components
        or crossing_fingerprints
        or fingerprints_in_multiple_components
    ):
        raise ValueError(
            "split hygiene failed: "
            f"files={len(crossing_files)}, file_components={len(files_in_multiple_components)}, "
            f"components={len(crossing_components)}, fingerprints={len(crossing_fingerprints)}, "
            f"fingerprint_components={len(fingerprints_in_multiple_components)}"
        )
    return {
        "cross_split_files": 0,
        "files_in_multiple_components": 0,
        "cross_split_components": 0,
        "cross_split_statement_fingerprints": 0,
        "statement_fingerprints_in_multiple_components": 0,
    }


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot summarize empty values")
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return sorted(values)[index]


def summarize_token_lengths(
    records: Sequence[MathlibProofRecord],
) -> dict[str, dict[str, Any]]:
    if not records:
        raise ValueError("cannot summarize an empty corpus")
    result: dict[str, dict[str, Any]] = {}
    for field_name in TokenLengths.__dataclass_fields__:
        values = [getattr(record.token_lengths, field_name) for record in records]
        sorted_values = sorted(values)
        middle = len(values) // 2
        median: float | int = (
            sorted_values[middle]
            if len(values) % 2
            else (sorted_values[middle - 1] + sorted_values[middle]) / 2
        )
        result[field_name] = {
            "min": sorted_values[0],
            "median": median,
            "p90": _nearest_rank(values, 0.90),
            "p95": _nearest_rank(values, 0.95),
            "p99": _nearest_rank(values, 0.99),
            "max": sorted_values[-1],
            "at_or_above": {
                str(threshold): sum(value >= threshold for value in values)
                for threshold in TOKEN_THRESHOLDS
            },
        }
    return result


def split_statistics(
    records: Sequence[MathlibProofRecord],
    proportions: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    total = len(records)
    result: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        split_records = [record for record in records if record.split == split]
        result[split] = {
            "records": len(split_records),
            "proportion": len(split_records) / total,
            "target_proportion": proportions[split],
            "files": len({record.file_path for record in split_records}),
            "components": len({record.component_id for record in split_records}),
        }
    return result


def write_jsonl_splits(
    records: Sequence[MathlibProofRecord], output_dir: Path
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        selected = [record for record in records if record.split == split]
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                )
                handle.write("\n")
        counts[split] = len(selected)
    return counts


def read_jsonl_records(path: Path) -> list[MathlibProofRecord]:
    records: list[MathlibProofRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(MathlibProofRecord.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid record at {path}:{line_number}: {error}"
                    ) from error
    return records


def _datasets_features() -> Any:
    from datasets import Features, List, Value

    position = {"line": Value("int32"), "column": Value("int32")}
    span = {"start": position, "end": position}
    return Features(
        {
            "schema_version": Value("string"),
            "id": Value("string"),
            "source_repository": Value("string"),
            "source_revision": Value("string"),
            "file_path": Value("string"),
            "declaration_name": Value("string"),
            "declaration_kind": Value("string"),
            "source_span": span,
            "declaration_span": span,
            "proof_span": span,
            "declaration": Value("string"),
            "proof": Value("string"),
            "completion": Value("string"),
            "premises": List(Value("string")),
            "file_group": Value("string"),
            "component_id": Value("string"),
            "split": Value("string"),
            "statement_fingerprint": Value("string"),
            "token_lengths": {
                field_name: Value("int64")
                for field_name in TokenLengths.__dataclass_fields__
            },
        }
    )


def load_phase2_dataset(artifact_dir: Path) -> Any:
    from datasets import load_dataset

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_schema_version") != PHASE2_DATASET_SCHEMA_VERSION:
        raise ValueError("manifest declares an unsupported Phase 2 dataset schema")
    data_files = {
        split: str((artifact_dir / f"{split}.jsonl").resolve()) for split in SPLIT_NAMES
    }
    dataset = load_dataset("json", data_files=data_files, features=_datasets_features())
    expected = manifest["splits"]
    seen_ids: set[str] = set()
    for split in SPLIT_NAMES:
        if len(dataset[split]) != int(expected[split]["records"]):
            raise ValueError(
                f"{split} record count mismatch: manifest={expected[split]['records']} "
                f"loaded={len(dataset[split])}"
            )
        missing = REQUIRED_RECORD_FIELDS - set(dataset[split].column_names)
        if missing:
            raise ValueError(f"{split} is missing required fields: {sorted(missing)}")
        if any(value != split for value in dataset[split]["split"]):
            raise ValueError(f"{split} contains records with a different split label")
        if any(
            value != PHASE2_DATASET_SCHEMA_VERSION
            for value in dataset[split]["schema_version"]
        ):
            raise ValueError(
                f"{split} contains records with a different schema version"
            )
        split_ids = set(dataset[split]["id"])
        if len(split_ids) != len(dataset[split]) or seen_ids.intersection(split_ids):
            raise ValueError(f"{split} contains duplicate record identities")
        seen_ids.update(split_ids)
        if "source" in manifest:
            source = manifest["source"]
            if any(
                value != source["repository"]
                for value in dataset[split]["source_repository"]
            ) or any(
                value != source["revision"]
                for value in dataset[split]["source_revision"]
            ):
                raise ValueError(f"{split} contains records from a different source")
    return dataset
