from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

PHASE2_CONFIG_SCHEMA_VERSION = "phase2-config-v1"
PHASE2_DATASET_SCHEMA_VERSION = "mathlib-whole-proof-v1"
PHASE2_MANIFEST_SCHEMA_VERSION = "phase2-manifest-v1"
PHASE2_VERIFICATION_SCHEMA_VERSION = "phase2-verification-v1"

DeclarationKind = Literal["theorem", "lemma"]
DatasetSplit = Literal["train", "validation", "heldout"]
SPLIT_NAMES: tuple[DatasetSplit, ...] = ("train", "validation", "heldout")


@dataclass(frozen=True, order=True)
class SourcePosition:
    line: int
    column: int

    @classmethod
    def from_value(cls, value: Any) -> SourcePosition:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(line=int(value["line"]), column=int(value["column"]))
        line, column = value
        return cls(line=int(line), column=int(column))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSpan:
    start: SourcePosition
    end: SourcePosition

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceSpan:
        return cls(
            start=SourcePosition.from_value(value["start"]),
            end=SourcePosition.from_value(value["end"]),
        )

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}


@dataclass(frozen=True)
class TokenLengths:
    declaration: int
    proof: int
    completion: int
    declaration_and_proof: int
    declaration_and_completion: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TokenLengths:
        return cls(**{key: int(item) for key, item in value.items()})

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MathlibProofRecord:
    schema_version: str
    id: str
    source_repository: str
    source_revision: str
    file_path: str
    declaration_name: str
    declaration_kind: DeclarationKind
    source_span: SourceSpan
    declaration_span: SourceSpan
    proof_span: SourceSpan
    declaration: str
    proof: str
    completion: str
    premises: tuple[str, ...]
    file_group: str
    component_id: str
    split: DatasetSplit
    statement_fingerprint: str
    token_lengths: TokenLengths

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MathlibProofRecord:
        schema_version = str(value["schema_version"])
        if schema_version != PHASE2_DATASET_SCHEMA_VERSION:
            raise ValueError(f"unknown Phase 2 record schema: {schema_version}")
        declaration_kind = str(value["declaration_kind"])
        if declaration_kind not in {"theorem", "lemma"}:
            raise ValueError(f"unsupported declaration kind: {declaration_kind}")
        split = str(value["split"])
        if split not in SPLIT_NAMES:
            raise ValueError(f"unsupported dataset split: {split}")
        return cls(
            schema_version=schema_version,
            id=str(value["id"]),
            source_repository=str(value["source_repository"]),
            source_revision=str(value["source_revision"]),
            file_path=str(value["file_path"]),
            declaration_name=str(value["declaration_name"]),
            declaration_kind=declaration_kind,  # type: ignore[arg-type]
            source_span=SourceSpan.from_dict(value["source_span"]),
            declaration_span=SourceSpan.from_dict(value["declaration_span"]),
            proof_span=SourceSpan.from_dict(value["proof_span"]),
            declaration=str(value["declaration"]),
            proof=str(value["proof"]),
            completion=str(value["completion"]),
            premises=tuple(str(item) for item in value["premises"]),
            file_group=str(value["file_group"]),
            component_id=str(value["component_id"]),
            split=split,  # type: ignore[arg-type]
            statement_fingerprint=str(value["statement_fingerprint"]),
            token_lengths=TokenLengths.from_dict(value["token_lengths"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["premises"] = list(self.premises)
        return value
