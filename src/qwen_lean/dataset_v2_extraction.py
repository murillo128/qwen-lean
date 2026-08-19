from __future__ import annotations

import hashlib
import json
import pickle
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .dataset_v2_contract import (
    canonicalize_equation_clauses,
    canonicalize_proof_expression,
    canonicalize_where_fields,
    proof_fingerprint,
    proof_variant_id,
    statement_fingerprint_v2,
    statement_id,
)
from .dataset_v2_schema import (
    DATASET_V2_SCHEMA_VERSION,
    DatasetV2Record,
    EnvironmentContext,
    LengthMetadata,
    ProofVariant,
    ProofVerification,
    SourcePosition,
    SourceSpan,
)
from .phase2_corpus import (
    _lex_lean,
    canonical_declaration,
    position_offset,
    source_slice,
)


DATASET_V2_CONFIG_SCHEMA_VERSION = "dataset-v2-config-v1"
DATASET_V2_GROUP_VERIFICATION_CACHE_VERSION = (
    "dataset-v2-group-verification-cache-v2"
)


@dataclass(frozen=True)
class DatasetV2Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> DatasetV2Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != DATASET_V2_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unknown Dataset-v2 config schema: {value.get('schema_version')}")
        if value["dataset"]["record_schema_version"] != DATASET_V2_SCHEMA_VERSION:
            raise ValueError("Dataset-v2 config declares an unsupported record schema")
        return cls(path=path.resolve(), value=value)

    @property
    def environment(self) -> dict[str, Any]:
        return self.value["target_environment"]

    @property
    def preflight(self) -> dict[str, Any]:
        return self.value["preflight"]

    def validate_target_root(self, root: Path) -> dict[str, str]:
        expected = self.environment
        observed_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        if observed_revision != expected["host_revision"]:
            raise ValueError("Dataset-v2 target checkout differs from the frozen host revision")
        toolchain = (root / "lean-toolchain").read_text(encoding="utf-8").strip()
        if toolchain != expected["lean_toolchain"]:
            raise ValueError("Dataset-v2 target checkout has a different Lean toolchain")
        manifest = json.loads((root / "lake-manifest.json").read_text(encoding="utf-8"))
        mathlib = next(item for item in manifest["packages"] if item["name"] == "mathlib")
        if (
            mathlib["rev"] != expected["mathlib_revision"]
            or mathlib["inputRev"] != expected["mathlib_input_revision"]
        ):
            raise ValueError("Dataset-v2 target checkout has a different mathlib identity")
        return {
            "environment_id": str(expected["environment_id"]),
            "host_revision": observed_revision,
            "mathlib_revision": str(mathlib["rev"]),
            "lean_toolchain": toolchain,
        }


@dataclass(frozen=True)
class SourceCandidate:
    source_repository: str
    source_revision: str
    file_path: str
    module: str
    declaration_name: str
    declaration_kind: str
    source_span: SourceSpan
    declaration_span: SourceSpan
    proof_span: SourceSpan
    declaration: str
    source_expression: str
    canonical_proof: str
    completion: str
    transformation_kind: str
    resolved_dependencies: tuple[str, ...]
    imports: tuple[str, ...]
    provenance: str
    topic_tags: tuple[str, ...]
    memberships: tuple[str, ...]
    verification_status: str
    verification_method: str
    verification_evidence_id: str
    verification_diagnostic: str = ""


@dataclass(frozen=True)
class ExtractionDiagnostics:
    source_files: int
    traced_declarations: int
    candidates: int
    transformation_counts: dict[str, int]
    exclusion_counts: dict[str, int]
    exclusions: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class FileVerification:
    file_path: str
    candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    status: str
    exit_code: int | None
    latency_seconds: float
    diagnostic: str


def _position(value: Any) -> SourcePosition:
    return SourcePosition(line=int(value.line_nb), column=int(value.column_nb))


def _span(start: Any, end: Any) -> SourceSpan:
    return SourceSpan(start=_position(start), end=_position(end))


def _position_from_offset(source: str, offset: int) -> SourcePosition:
    if offset < 0 or offset > len(source):
        raise ValueError("source offset is outside the file")
    prefix = source[:offset]
    return SourcePosition(
        line=prefix.count("\n") + 1,
        column=len(prefix.rsplit("\n", 1)[-1]) + 1,
    )


def _module(file_path: str) -> str:
    return file_path.removesuffix(".lean").replace("/", ".")


def _imports(source: str) -> tuple[str, ...]:
    imports: list[str] = []
    for match in re.finditer(r"(?m)^\s*(?:public\s+)?import\s+([^\n]+)$", source):
        imports.extend(item for item in match.group(1).split() if not item.startswith("--"))
    return tuple(dict.fromkeys(imports))


def _contains_placeholder(value: str) -> bool:
    return any(
        kind == "identifier" and token in {"sorry", "admit"}
        for kind, token in _lex_lean(value)
    )


def _top_level_token_indices(value: str, token: str) -> list[int]:
    """Locate a token outside comments, strings, and brackets."""

    indices: list[int] = []
    closing_stack: list[str] = []
    block_comment_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False
    opening = {"(": ")", "[": "]", "{": "}", "⦃": "⦄"}
    index = 0
    while index < len(value):
        pair = value[index : index + 2]
        char = value[index]
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue
        if block_comment_depth:
            if pair == "/-":
                block_comment_depth += 1
                index += 2
            elif pair == "-/":
                block_comment_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if pair == "--":
            in_line_comment = True
            index += 2
            continue
        if pair == "/-":
            block_comment_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        word_token = token[0].isalnum()
        token_boundary = not word_token or (
            (index == 0 or not (value[index - 1].isalnum() or value[index - 1] in "_'†"))
            and (
                index + len(token) >= len(value)
                or not (
                    value[index + len(token)].isalnum()
                    or value[index + len(token)] in "_'†"
                )
            )
        )
        if (
            not closing_stack
            and token_boundary
            and value.startswith(token, index)
        ):
            indices.append(index)
            index += len(token)
            continue
        if char in opening:
            closing_stack.append(opening[char])
        elif closing_stack and char == closing_stack[-1]:
            closing_stack.pop()
        index += 1
    return indices


def _proof_dependencies(proof_node: Any) -> tuple[str, ...]:
    dependencies: set[str] = set()

    def collect(node: Any, _: Any) -> None:
        full_name = getattr(node, "full_name", None)
        if full_name:
            dependencies.add(str(full_name))

    proof_node.traverse_preorder(collect, node_cls=None)
    return tuple(sorted(dependencies))


def _declaration_kind(traced_theorem: Any) -> str:
    class_name = type(traced_theorem.ast).__name__
    if class_name == "CommandTheoremNode":
        return "theorem"
    if class_name in {"LemmaNode", "MathlibTacticLemmaNode"}:
        return "lemma"
    return class_name


def _candidate_id(candidate: SourceCandidate) -> str:
    payload = "\0".join(
        (
            "dataset-v2-source-candidate-v1",
            candidate.source_revision,
            candidate.file_path,
            candidate.declaration_name,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_whole_declaration(source_text: str) -> tuple[str, str]:
    """Split a complete theorem/lemma source block at its first proof assignment."""

    text = source_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text.startswith(("theorem ", "lemma ")):
        raise ValueError("external source block is not a theorem or lemma")
    block_comment_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False
    index = 0
    while index + 1 < len(text):
        pair = text[index : index + 2]
        char = text[index]
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue
        if block_comment_depth:
            if pair == "/-":
                block_comment_depth += 1
                index += 2
            elif pair == "-/":
                block_comment_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if pair == "--":
            in_line_comment = True
            index += 2
            continue
        if pair == "/-":
            block_comment_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if pair == ":=":
            declaration = canonical_declaration(text[:index].rstrip())
            proof = text[index + 2 :].strip()
            if not proof:
                raise ValueError("external source block has an empty proof")
            return declaration, proof
        index += 1
    raise ValueError("external source block has no proof assignment")


def candidate_from_external_record(
    value: Mapping[str, Any],
    *,
    source_root: Path,
    resolved_dependencies: Sequence[str] = (),
    verification_evidence_id: str,
) -> SourceCandidate:
    """Recover a whole proof from a pinned, previously accepted external record."""

    file_path = str(value["file_path"])
    source_path = source_root / file_path
    source = source_path.read_text(encoding="utf-8")
    source_text = str(value["source_text"]).replace("\r\n", "\n").replace("\r", "\n")
    observed_sha = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if observed_sha != str(value["source_sha256"]):
        raise ValueError(f"external source-block hash mismatch: {file_path}")
    start_line = int(value["source_span"]["start_line"])
    source_offset = sum(len(line) for line in source.splitlines(keepends=True)[: start_line - 1])
    if source[source_offset : source_offset + len(source_text)] != source_text:
        raise ValueError(f"external source span mismatch: {file_path}")
    declaration, proof_expression = split_whole_declaration(source_text)
    proof = canonicalize_proof_expression(proof_expression)
    assignment_index = source_text.find(":=", len(declaration))
    if assignment_index < 0:
        raise ValueError(f"external proof assignment mismatch: {file_path}")
    proof_index = assignment_index + 2
    while proof_index < len(source_text) and source_text[proof_index].isspace():
        proof_index += 1
    proof_prefix = source_text[:proof_index]
    proof_start_line = start_line + proof_prefix.count("\n")
    proof_start_column = len(proof_prefix.rsplit("\n", 1)[-1]) + 1
    end_line = start_line + source_text.count("\n")
    end_column = len(source_text.rsplit("\n", 1)[-1]) + 1
    return SourceCandidate(
        source_repository=str(value["repository"]),
        source_revision=str(value["revision"]),
        file_path=file_path,
        module=_module(file_path),
        declaration_name=str(value["declaration_name"]),
        declaration_kind=str(value["declaration_kind"]),
        source_span=SourceSpan(
            start=SourcePosition(start_line, 1),
            end=SourcePosition(end_line, end_column),
        ),
        declaration_span=SourceSpan(
            start=SourcePosition(start_line, 1),
            end=SourcePosition(
                start_line + source_text[:assignment_index].count("\n"),
                len(source_text[:assignment_index].rsplit("\n", 1)[-1]) + 1,
            ),
        ),
        proof_span=SourceSpan(
            start=SourcePosition(proof_start_line, proof_start_column),
            end=SourcePosition(end_line, end_column),
        ),
        declaration=declaration,
        source_expression=proof.source_expression,
        canonical_proof=proof.canonical_proof,
        completion=proof.completion,
        transformation_kind=proof.transformation,
        resolved_dependencies=tuple(sorted(set(resolved_dependencies))),
        imports=_imports(source),
        provenance="external-lean",
        topic_tags=tuple(sorted(set(str(item) for item in value.get("topic_tags", [])))),
        memberships=("riemann-external-lean-v1",),
        verification_status="accepted" if proof.transformation == "none" else "pending",
        verification_method=(
            "pinned-source-build-and-axiom-audit"
            if proof.transformation == "none"
            else "pending-reconstruction"
        ),
        verification_evidence_id=(
            verification_evidence_id if proof.transformation == "none" else "pending"
        ),
    )


def candidate_from_traced_theorem(
    traced_theorem: Any,
    *,
    source: str,
    file_path: str,
    source_repository: str,
    source_revision: str,
    provenance: str,
    topic_tags: Iterable[str] = (),
    memberships: Iterable[str] = (),
) -> SourceCandidate:
    if bool(traced_theorem.is_private):
        raise ValueError("private")
    declaration_name = str(traced_theorem.theorem.full_name or "")
    if not declaration_name or declaration_name.startswith("_private."):
        raise ValueError("unstable-name")
    declaration_kind = _declaration_kind(traced_theorem)
    if declaration_kind not in {"theorem", "lemma"}:
        raise ValueError(f"unsupported-kind:{declaration_kind}")

    proof_node = traced_theorem.get_proof_node()
    proof_start, proof_end = proof_node.get_closure()
    source_span = _span(traced_theorem.start, traced_theorem.end)
    declaration_span = _span(traced_theorem.start, proof_start)
    proof_span = _span(proof_start, proof_end)
    raw_declaration = source_slice(source, declaration_span)
    if any(
        kind == "identifier" and token == "private"
        for kind, token in _lex_lean(raw_declaration)
    ):
        raise ValueError("private")
    recovered_assignment = False
    where_indices = _top_level_token_indices(raw_declaration, "where")
    if where_indices:
        where_offset = where_indices[-1]
        absolute_where_offset = (
            position_offset(source, source_span.start) + where_offset
        )
        where_position = _position_from_offset(source, absolute_where_offset)
        declaration_span = SourceSpan(source_span.start, where_position)
        proof_span = SourceSpan(where_position, source_span.end)
        raw_declaration = source_slice(source, declaration_span)
    elif not raw_declaration.rstrip().endswith(":="):
        assignments = _top_level_token_indices(raw_declaration, ":=")
        if assignments:
            recovered_assignment = True
            assignment_offset = assignments[-1]
            absolute_assignment_offset = (
                position_offset(source, source_span.start) + assignment_offset
            )
            proof_offset = absolute_assignment_offset + 2
            while proof_offset < len(source) and source[proof_offset].isspace():
                proof_offset += 1
            declaration_span = SourceSpan(
                source_span.start,
                _position_from_offset(source, absolute_assignment_offset),
            )
            proof_span = SourceSpan(
                _position_from_offset(source, proof_offset), source_span.end
            )
            raw_declaration = source_slice(source, declaration_span)
    source_expression = source_slice(source, proof_span)
    if _contains_placeholder(source_expression):
        raise ValueError("proof-placeholder")
    if raw_declaration.rstrip().endswith(":=") or recovered_assignment:
        proof = canonicalize_proof_expression(source_expression)
    elif source_expression.lstrip().startswith("where"):
        proof = canonicalize_where_fields(source_expression)
    elif source_expression.lstrip().startswith("|"):
        proof = canonicalize_equation_clauses(source_expression)
    else:
        raise ValueError("unsupported-proof-delimiter")
    declaration = canonical_declaration(raw_declaration)
    verification_status = "accepted" if proof.transformation == "none" else "pending"
    verification_method = (
        "pinned-source-build" if proof.transformation == "none" else "pending-reconstruction"
    )
    evidence_id = (
        f"source-build:{source_revision}"
        if proof.transformation == "none"
        else "pending"
    )
    return SourceCandidate(
        source_repository=source_repository,
        source_revision=source_revision,
        file_path=file_path,
        module=_module(file_path),
        declaration_name=declaration_name,
        declaration_kind=declaration_kind,
        source_span=source_span,
        declaration_span=declaration_span,
        proof_span=proof_span,
        declaration=declaration,
        source_expression=proof.source_expression,
        canonical_proof=proof.canonical_proof,
        completion=proof.completion,
        transformation_kind=proof.transformation,
        resolved_dependencies=_proof_dependencies(proof_node),
        imports=_imports(source),
        provenance=provenance,
        topic_tags=tuple(sorted(set(topic_tags))),
        memberships=tuple(sorted(set(memberships))),
        verification_status=verification_status,
        verification_method=verification_method,
        verification_evidence_id=evidence_id,
    )


def extract_traced_files(
    traced_files: Iterable[Any],
    *,
    source_root: Path,
    source_repository: str,
    source_revision: str,
    provenance: str,
    selected_declarations: set[str] | None = None,
    topic_metadata: Mapping[tuple[str, str], Mapping[str, Sequence[str]]] | None = None,
) -> tuple[list[SourceCandidate], ExtractionDiagnostics]:
    candidates: list[SourceCandidate] = []
    counts: Counter[str] = Counter()
    exclusions: list[dict[str, str]] = []
    source_files = 0
    traced_declarations = 0
    for traced_file in traced_files:
        file_path = str(traced_file.path)
        source_path = source_root / file_path
        if not source_path.is_file():
            counts["source-read-error"] += 1
            exclusions.append({"file_path": file_path, "reason": "source-read-error"})
            continue
        source_files += 1
        source = source_path.read_text(encoding="utf-8")
        try:
            theorems = traced_file.get_traced_theorems()
        except Exception as error:  # noqa: BLE001 - retain every traced-file loss.
            counts["trace-read-error"] += 1
            exclusions.append(
                {"file_path": file_path, "reason": "trace-read-error", "detail": repr(error)}
            )
            continue
        for theorem in theorems:
            traced_declarations += 1
            name = str(getattr(theorem.theorem, "full_name", ""))
            if selected_declarations is not None and name not in selected_declarations:
                continue
            metadata = (topic_metadata or {}).get((file_path, name), {})
            try:
                candidate = candidate_from_traced_theorem(
                    theorem,
                    source=source,
                    file_path=file_path,
                    source_repository=source_repository,
                    source_revision=source_revision,
                    provenance=provenance,
                    topic_tags=metadata.get("topic_tags", ()),
                    memberships=metadata.get("memberships", ()),
                )
            except Exception as error:  # noqa: BLE001 - every source loss is classified.
                reason = str(error) or type(error).__name__
                counts[reason.split(":", 1)[0]] += 1
                exclusions.append(
                    {
                        "file_path": file_path,
                        "declaration_name": name,
                        "reason": reason,
                    }
                )
                continue
            candidates.append(candidate)
            counts[f"transformation:{candidate.transformation_kind}"] += 1
    transformations = {
        key.removeprefix("transformation:"): value
        for key, value in counts.items()
        if key.startswith("transformation:")
    }
    exclusions_count = {
        key: value for key, value in counts.items() if not key.startswith("transformation:")
    }
    return candidates, ExtractionDiagnostics(
        source_files=source_files,
        traced_declarations=traced_declarations,
        candidates=len(candidates),
        transformation_counts=dict(sorted(transformations.items())),
        exclusion_counts=dict(sorted(exclusions_count.items())),
        exclusions=tuple(exclusions),
    )


def select_candidates(
    candidates: Sequence[SourceCandidate],
    *,
    count: int,
    seed: str,
    transformation_kind: str | None = None,
) -> list[SourceCandidate]:
    eligible = [
        item
        for item in candidates
        if transformation_kind is None or item.transformation_kind == transformation_kind
    ]
    if len(eligible) < count:
        raise ValueError(f"candidate selection requires {count}, found {len(eligible)}")
    ordered = sorted(
        eligible,
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item.source_revision}\0{item.file_path}\0{item.declaration_name}".encode()
        ).hexdigest(),
    )
    distinct: list[SourceCandidate] = []
    repeated: list[SourceCandidate] = []
    seen_files: set[str] = set()
    for item in ordered:
        if item.file_path in seen_files:
            repeated.append(item)
        else:
            seen_files.add(item.file_path)
            distinct.append(item)
    return (distinct + repeated)[:count]


def substitute_proofs(source: str, candidates: Sequence[SourceCandidate]) -> str:
    replacements: list[tuple[int, int, str]] = []
    for candidate in candidates:
        start = position_offset(source, candidate.proof_span.start)
        end = position_offset(source, candidate.proof_span.end)
        if start >= end:
            raise ValueError(f"empty proof span for {candidate.declaration_name}")
        if source[start:end].replace("\r\n", "\n").replace("\r", "\n").strip() != candidate.source_expression:
            raise ValueError(f"source proof identity mismatch for {candidate.declaration_name}")
        replacement = (
            ":= " + candidate.canonical_proof
            if candidate.transformation_kind
            in {"equations-to-fun-exact", "where-to-structure-exact"}
            else candidate.canonical_proof
        )
        replacements.append((start, end, replacement))
    result = source
    previous_start = len(source) + 1
    for start, end, proof in sorted(replacements, reverse=True):
        if end > previous_start:
            raise ValueError("overlapping proof spans in one source file")
        result = result[:start] + proof + result[end:]
        previous_start = start
    return result


def _run_reconstructed_file(
    source: str,
    *,
    target_root: Path,
    timeout_seconds: float,
) -> tuple[str, int | None, float, str]:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="qwen-lean-dataset-v2-") as temp_dir:
            source_path = Path(temp_dir) / "Reconstructed.lean"
            source_path.write_text(source, encoding="utf-8", newline="\n")
            completed = subprocess.run(
                ["lake", "env", "lean", "-E", "hasSorry", str(source_path)],
                cwd=target_root,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            diagnostic = (completed.stderr + "\n" + completed.stdout).replace(
                str(source_path), "Reconstructed.lean"
            )
    except subprocess.TimeoutExpired as error:
        return "timeout", None, time.perf_counter() - started, str(error)[-4000:]
    except OSError as error:
        return "infrastructure-error", None, time.perf_counter() - started, str(error)
    status = "accepted" if completed.returncode == 0 else "rejected"
    return status, completed.returncode, time.perf_counter() - started, (
        "" if status == "accepted" else diagnostic[-4000:]
    )


def verify_transformed_candidates(
    candidates: Sequence[SourceCandidate],
    *,
    source_roots: Mapping[tuple[str, str], Path],
    target_root: Path,
    environment_id: str,
    evidence_id: str,
    workers: int = 8,
    timeout_seconds: float = 300.0,
    group_cache_dir: Path | None = None,
) -> tuple[list[SourceCandidate], list[FileVerification]]:
    pending = [item for item in candidates if item.transformation_kind != "none"]
    grouped: dict[tuple[str, str, str], list[SourceCandidate]] = defaultdict(list)
    for item in pending:
        grouped[(item.source_repository, item.source_revision, item.file_path)].append(item)

    if group_cache_dir is not None:
        group_cache_dir.mkdir(parents=True, exist_ok=True)

    def verify_group_cache_path(
        key: tuple[str, str, str], group: list[SourceCandidate], source: str
    ) -> Path | None:
        if group_cache_dir is None:
            return None
        repository, revision, file_path = key
        cache_identity = {
            "version": DATASET_V2_GROUP_VERIFICATION_CACHE_VERSION,
            "repository": repository,
            "revision": revision,
            "file_path": file_path,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "environment_id": environment_id,
            "evidence_id": evidence_id,
            "timeout_seconds": timeout_seconds,
            "candidates": [
                {
                    "id": _candidate_id(item),
                    "source_expression": item.source_expression,
                    "canonical_proof": item.canonical_proof,
                    "transformation_kind": item.transformation_kind,
                    "proof_span": {
                        "start": [item.proof_span.start.line, item.proof_span.start.column],
                        "end": [item.proof_span.end.line, item.proof_span.end.column],
                    },
                }
                for item in group
            ],
        }
        digest = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return group_cache_dir / f"{digest}.pkl"

    def verify_group(
        key_group: tuple[tuple[str, str, str], list[SourceCandidate]],
    ) -> tuple[list[SourceCandidate], FileVerification]:
        (repository, revision, file_path), group = key_group
        source_root = source_roots[(repository, revision)]
        source = (source_root / file_path).read_text(encoding="utf-8")
        cache_path = verify_group_cache_path(
            (repository, revision, file_path), group, source
        )
        if cache_path is not None and cache_path.is_file():
            with cache_path.open("rb") as handle:
                cache_version, cached_result = pickle.load(handle)
            if cache_version == DATASET_V2_GROUP_VERIFICATION_CACHE_VERSION:
                return cached_result
        transformed = substitute_proofs(source, group)
        status, exit_code, latency, diagnostic = _run_reconstructed_file(
            transformed, target_root=target_root, timeout_seconds=timeout_seconds
        )
        if status == "accepted":
            accepted = [
                replace(
                    item,
                    verification_status="accepted",
                    verification_method="source-file-term-canonicalization",
                    verification_evidence_id=evidence_id,
                    verification_diagnostic="",
                )
                for item in group
            ]
            result = accepted, FileVerification(
                file_path=file_path,
                candidate_ids=tuple(_candidate_id(item) for item in group),
                rejected_candidate_ids=(),
                status=status,
                exit_code=exit_code,
                latency_seconds=latency,
                diagnostic="",
            )
            if cache_path is not None:
                temporary_cache = cache_path.with_suffix(".tmp")
                with temporary_cache.open("wb") as handle:
                    pickle.dump(
                        (DATASET_V2_GROUP_VERIFICATION_CACHE_VERSION, result),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                temporary_cache.replace(cache_path)
            return result

        baseline_status, baseline_exit_code, baseline_latency, baseline_diagnostic = (
            _run_reconstructed_file(
                source, target_root=target_root, timeout_seconds=timeout_seconds
            )
        )
        if baseline_status != "accepted":
            baseline_rejected = [
                replace(
                    item,
                    verification_status=f"baseline-{baseline_status}",
                    verification_method="source-file-baseline-verification",
                    verification_evidence_id=evidence_id,
                    verification_diagnostic=baseline_diagnostic,
                )
                for item in group
            ]
            result = baseline_rejected, FileVerification(
                file_path=file_path,
                candidate_ids=tuple(_candidate_id(item) for item in group),
                rejected_candidate_ids=tuple(_candidate_id(item) for item in group),
                status=f"baseline-{baseline_status}",
                exit_code=baseline_exit_code,
                latency_seconds=latency + baseline_latency,
                diagnostic=(
                    diagnostic + "\nbaseline=" + baseline_diagnostic
                )[-4000:],
            )
            if cache_path is not None:
                temporary_cache = cache_path.with_suffix(".tmp")
                with temporary_cache.open("wb") as handle:
                    pickle.dump(
                        (DATASET_V2_GROUP_VERIFICATION_CACHE_VERSION, result),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                temporary_cache.replace(cache_path)
            return result

        attempts: list[str] = []

        def classify(subgroup: list[SourceCandidate]) -> list[SourceCandidate]:
            if not subgroup:
                return []
            subgroup_source = substitute_proofs(source, subgroup)
            subgroup_status, _, _, subgroup_diagnostic = _run_reconstructed_file(
                subgroup_source,
                target_root=target_root,
                timeout_seconds=timeout_seconds,
            )
            attempts.append(f"{len(subgroup)}:{subgroup_status}")
            if subgroup_status == "accepted":
                return [
                    replace(
                        item,
                        verification_status="accepted",
                        verification_method="source-file-term-canonicalization-bisect",
                        verification_evidence_id=evidence_id,
                        verification_diagnostic="",
                    )
                    for item in subgroup
                ]
            if len(subgroup) == 1:
                return [
                    replace(
                        subgroup[0],
                        verification_status=subgroup_status,
                        verification_method="source-position-term-canonicalization",
                        verification_evidence_id=evidence_id,
                        verification_diagnostic=subgroup_diagnostic,
                    )
                ]
            midpoint = len(subgroup) // 2
            return classify(subgroup[:midpoint]) + classify(subgroup[midpoint:])

        if len(group) == 1:
            attempts.append(f"1:{status}")
            classified = [
                replace(
                    group[0],
                    verification_status=status,
                    verification_method="source-position-term-canonicalization",
                    verification_evidence_id=evidence_id,
                    verification_diagnostic=diagnostic,
                )
            ]
        else:
            midpoint = len(group) // 2
            classified = classify(group[:midpoint]) + classify(group[midpoint:])
        accepted_count = sum(item.verification_status == "accepted" for item in classified)
        rejected_candidate_ids = tuple(
            _candidate_id(item)
            for item in classified
            if item.verification_status != "accepted"
        )
        result = classified, FileVerification(
            file_path=file_path,
            candidate_ids=tuple(_candidate_id(item) for item in group),
            rejected_candidate_ids=rejected_candidate_ids,
            status=(
                "partial"
                if accepted_count and accepted_count != len(classified)
                else status
            ),
            exit_code=exit_code,
            latency_seconds=latency,
            diagnostic=(diagnostic + "\nbisection=" + ", ".join(attempts))[-4000:],
        )
        if cache_path is not None:
            temporary_cache = cache_path.with_suffix(".tmp")
            with temporary_cache.open("wb") as handle:
                pickle.dump(
                    (DATASET_V2_GROUP_VERIFICATION_CACHE_VERSION, result),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary_cache.replace(cache_path)
        return result

    ordered_groups = sorted(grouped.items(), key=lambda item: item[0])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified_groups = list(executor.map(verify_group, ordered_groups))
    by_candidate_id = {
        _candidate_id(item): item
        for group, _ in verified_groups
        for item in group
    }
    verified = [
        by_candidate_id.get(_candidate_id(item), item) for item in candidates
    ]
    files = [result for _, result in verified_groups]
    return verified, files


def candidate_to_record(
    candidate: SourceCandidate,
    *,
    config: DatasetV2Config,
) -> DatasetV2Record:
    if candidate.verification_status != "accepted":
        raise ValueError(f"candidate {candidate.declaration_name} is not Lean-accepted")
    environment = config.environment
    identity = statement_id(candidate.declaration)
    proof_hash = proof_fingerprint(candidate.canonical_proof)
    variant = ProofVariant(
        proof_variant_id=proof_variant_id(identity, candidate.canonical_proof),
        source_expression=candidate.source_expression,
        canonical_proof=candidate.canonical_proof,
        completion=candidate.completion,
        transformation_kind=candidate.transformation_kind,  # type: ignore[arg-type]
        proof_fingerprint=proof_hash,
        resolved_dependencies=candidate.resolved_dependencies,
        verification=ProofVerification(
            status="accepted",
            environment_id=str(environment["environment_id"]),
            method=candidate.verification_method,
            evidence_id=candidate.verification_evidence_id,
            diagnostic=candidate.verification_diagnostic,
        ),
        source_declaration_name=candidate.declaration_name,
        source_repository=candidate.source_repository,
        source_revision=candidate.source_revision,
        source_file=candidate.file_path,
    )
    combined = f"{candidate.declaration} := {candidate.canonical_proof}"
    record = DatasetV2Record(
        schema_version=DATASET_V2_SCHEMA_VERSION,
        statement_id=identity,
        canonical_declaration=candidate.declaration,
        normalized_statement_fingerprint=statement_fingerprint_v2(candidate.declaration),
        role="training",
        sampling_group_id=identity,
        provenance=candidate.provenance,  # type: ignore[arg-type]
        environment=EnvironmentContext(
            environment_id=str(environment["environment_id"]),
            lean_toolchain=str(environment["lean_toolchain"]),
            repository=candidate.source_repository,
            revision=candidate.source_revision,
            mathlib_revision=str(environment["mathlib_revision"]),
            file_path=candidate.file_path,
            module=candidate.module,
            imports=candidate.imports,
            source_span=candidate.source_span,
            context_kind="source-position",
            target_compatibility="verified-target-environment",
        ),
        proof_variants=(variant,),
        topic_tags=candidate.topic_tags,
        memberships=candidate.memberships,
        length=LengthMetadata(
            declaration_chars=len(candidate.declaration),
            proof_chars=len(candidate.canonical_proof),
            completion_chars=len(candidate.completion),
            declaration_lines=candidate.declaration.count("\n") + 1,
            proof_lines=candidate.canonical_proof.count("\n") + 1,
            utf8_bytes=len(combined.encode("utf-8")),
        ),
    )
    record.validate()
    return record
