from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


DATASET_V2_SCHEMA_VERSION = "lean-whole-proof-v2"
DATASET_V2_MANIFEST_SCHEMA_VERSION = "lean-whole-proof-v2-manifest-v1"
PRIME_COVERAGE_SCHEMA_VERSION = "dataset-v2-prime-coverage-v1"

TrainingRole = Literal["training", "validation", "test"]
ProvenanceKind = Literal[
    "real-mathlib", "external-lean", "mixed-real", "synthetic"
]
TransformationKind = Literal[
    "none",
    "term-to-exact",
    "equations-to-fun-exact",
    "where-to-structure-exact",
]
CompatibilityStatus = Literal[
    "verified-target-environment",
    "verified-native",
    "target-compatibility-blocked",
]


@dataclass(frozen=True)
class SourcePosition:
    line: int
    column: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourcePosition:
        return cls(line=int(value["line"]), column=int(value["column"]))


@dataclass(frozen=True)
class SourceSpan:
    start: SourcePosition
    end: SourcePosition

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceSpan:
        return cls(
            start=SourcePosition.from_dict(value["start"]),
            end=SourcePosition.from_dict(value["end"]),
        )


@dataclass(frozen=True)
class EnvironmentContext:
    environment_id: str
    lean_toolchain: str
    repository: str
    revision: str
    mathlib_revision: str
    file_path: str
    module: str
    imports: tuple[str, ...]
    source_span: SourceSpan | None
    context_kind: str
    target_compatibility: CompatibilityStatus

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EnvironmentContext:
        source_span = value.get("source_span")
        return cls(
            environment_id=str(value["environment_id"]),
            lean_toolchain=str(value["lean_toolchain"]),
            repository=str(value["repository"]),
            revision=str(value["revision"]),
            mathlib_revision=str(value["mathlib_revision"]),
            file_path=str(value["file_path"]),
            module=str(value["module"]),
            imports=tuple(str(item) for item in value.get("imports", [])),
            source_span=(
                None if source_span is None else SourceSpan.from_dict(source_span)
            ),
            context_kind=str(value["context_kind"]),
            target_compatibility=str(value["target_compatibility"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LengthMetadata:
    declaration_chars: int
    proof_chars: int
    completion_chars: int
    declaration_lines: int
    proof_lines: int
    utf8_bytes: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LengthMetadata:
        return cls(**{key: int(item) for key, item in value.items()})


@dataclass(frozen=True)
class ProofVerification:
    status: str
    environment_id: str
    method: str
    evidence_id: str
    diagnostic: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProofVerification:
        return cls(
            status=str(value["status"]),
            environment_id=str(value["environment_id"]),
            method=str(value["method"]),
            evidence_id=str(value["evidence_id"]),
            diagnostic=str(value.get("diagnostic", "")),
        )


@dataclass(frozen=True)
class ProofVariant:
    proof_variant_id: str
    source_expression: str
    canonical_proof: str
    completion: str
    transformation_kind: TransformationKind
    proof_fingerprint: str
    resolved_dependencies: tuple[str, ...]
    verification: ProofVerification
    source_declaration_name: str
    source_repository: str
    source_revision: str
    source_file: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProofVariant:
        return cls(
            proof_variant_id=str(value["proof_variant_id"]),
            source_expression=str(value["source_expression"]),
            canonical_proof=str(value["canonical_proof"]),
            completion=str(value["completion"]),
            transformation_kind=str(value["transformation_kind"]),  # type: ignore[arg-type]
            proof_fingerprint=str(value["proof_fingerprint"]),
            resolved_dependencies=tuple(
                str(item) for item in value.get("resolved_dependencies", [])
            ),
            verification=ProofVerification.from_dict(value["verification"]),
            source_declaration_name=str(value["source_declaration_name"]),
            source_repository=str(value["source_repository"]),
            source_revision=str(value["source_revision"]),
            source_file=str(value["source_file"]),
        )


@dataclass(frozen=True)
class DatasetV2Record:
    schema_version: str
    statement_id: str
    canonical_declaration: str
    normalized_statement_fingerprint: str
    role: TrainingRole
    sampling_group_id: str
    provenance: ProvenanceKind
    environment: EnvironmentContext
    proof_variants: tuple[ProofVariant, ...]
    topic_tags: tuple[str, ...]
    memberships: tuple[str, ...]
    length: LengthMetadata
    derivation_family_fingerprint: str | None = None
    generator_family: str | None = None
    structural_class: str | None = None
    normalized_proof_dag: str | None = None
    source_lemma_ids: tuple[str, ...] = ()
    source_relation_edges: tuple[tuple[str, str, str], ...] = ()
    shortcut_retrieval_ids: tuple[str, ...] = ()
    shortcut_retrieval_index: tuple[tuple[str, str], ...] = ()
    shortcut_checks: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetV2Record:
        schema_version = str(value["schema_version"])
        if schema_version != DATASET_V2_SCHEMA_VERSION:
            raise ValueError(f"unknown Dataset-v2 record schema: {schema_version}")
        record = cls(
            schema_version=schema_version,
            statement_id=str(value["statement_id"]),
            canonical_declaration=str(value["canonical_declaration"]),
            normalized_statement_fingerprint=str(
                value["normalized_statement_fingerprint"]
            ),
            role=str(value["role"]),  # type: ignore[arg-type]
            sampling_group_id=str(value["sampling_group_id"]),
            provenance=str(value["provenance"]),  # type: ignore[arg-type]
            environment=EnvironmentContext.from_dict(value["environment"]),
            proof_variants=tuple(
                ProofVariant.from_dict(item) for item in value["proof_variants"]
            ),
            topic_tags=tuple(str(item) for item in value.get("topic_tags", [])),
            memberships=tuple(str(item) for item in value.get("memberships", [])),
            length=LengthMetadata.from_dict(value["length"]),
            derivation_family_fingerprint=(
                None
                if value.get("derivation_family_fingerprint") is None
                else str(value["derivation_family_fingerprint"])
            ),
            generator_family=(
                None
                if value.get("generator_family") is None
                else str(value["generator_family"])
            ),
            structural_class=(
                None
                if value.get("structural_class") is None
                else str(value["structural_class"])
            ),
            normalized_proof_dag=(
                None
                if value.get("normalized_proof_dag") is None
                else str(value["normalized_proof_dag"])
            ),
            source_lemma_ids=tuple(
                str(item) for item in value.get("source_lemma_ids", [])
            ),
            source_relation_edges=tuple(
                tuple(str(part) for part in item)  # type: ignore[misc]
                for item in value.get("source_relation_edges", [])
            ),
            shortcut_retrieval_ids=tuple(
                str(item) for item in value.get("shortcut_retrieval_ids", [])
            ),
            shortcut_retrieval_index=tuple(
                tuple(str(part) for part in item)  # type: ignore[misc]
                for item in value.get("shortcut_retrieval_index", [])
            ),
            shortcut_checks=tuple(
                str(item) for item in value.get("shortcut_checks", [])
            ),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.role not in {"training", "validation", "test"}:
            raise ValueError(f"unsupported Dataset-v2 role: {self.role}")
        if self.provenance not in {
            "real-mathlib",
            "external-lean",
            "mixed-real",
            "synthetic",
        }:
            raise ValueError(f"unsupported Dataset-v2 provenance: {self.provenance}")
        if not self.proof_variants:
            raise ValueError(f"statement {self.statement_id} has no proof variant")
        variant_ids = {item.proof_variant_id for item in self.proof_variants}
        proof_fingerprints = {item.proof_fingerprint for item in self.proof_variants}
        if len(variant_ids) != len(self.proof_variants):
            raise ValueError(f"statement {self.statement_id} repeats a proof variant id")
        if len(proof_fingerprints) != len(self.proof_variants):
            raise ValueError(f"statement {self.statement_id} repeats exact proof text")
        if self.sampling_group_id != self.statement_id:
            raise ValueError("Dataset-v2 sampling groups must be statement identities")
        if any(
            item.verification.status != "accepted"
            or item.verification.environment_id != self.environment.environment_id
            for item in self.proof_variants
        ):
            raise ValueError("every retained proof variant must be target-environment accepted")
        if self.provenance == "synthetic":
            required = (
                self.derivation_family_fingerprint,
                self.generator_family,
                self.structural_class,
                self.normalized_proof_dag,
            )
            if any(not item for item in required) or not self.source_lemma_ids:
                raise ValueError("synthetic record lacks derivation metadata")
            if len(self.source_relation_edges) < len(self.source_lemma_ids) - 1:
                raise ValueError("synthetic record lacks a connected source relation graph")
            if any(len(edge) != 3 for edge in self.source_relation_edges):
                raise ValueError("synthetic source relation edge is malformed")
            if any(len(entry) != 2 for entry in self.shortcut_retrieval_index):
                raise ValueError("synthetic shortcut retrieval index is malformed")
        elif any(
            item is not None
            for item in (
                self.derivation_family_fingerprint,
                self.generator_family,
                self.structural_class,
                self.normalized_proof_dag,
            )
        ):
            raise ValueError("real-source record contains synthetic derivation metadata")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value
