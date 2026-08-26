from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .dataset_v2_schema import EnvironmentContext, ProofVerification


DATASET_V3_SCHEMA_VERSION = "lean-proof-continuation-v3"
DATASET_V3_MANIFEST_SCHEMA_VERSION = "lean-proof-continuation-v3-manifest-v1"
DATASET_V3_VIEW_SCHEMA_VERSION = "lean-proof-continuation-v3-view-v1"

DatasetRole = Literal["training", "validation", "test"]
ExampleKind = Literal["whole", "continuation"]
ProofForm = Literal["tactic", "term", "equations", "where", "generated-tactic"]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructuralBoundary:
    boundary_id: str
    prefix_end: int
    prefix_sha256: str
    continuation_sha256: str
    reconstruction_sha256: str
    structural_kind: str
    segment_index: int
    extractor_version: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StructuralBoundary:
        return cls(
            boundary_id=str(value["boundary_id"]),
            prefix_end=int(value["prefix_end"]),
            prefix_sha256=str(value["prefix_sha256"]),
            continuation_sha256=str(value["continuation_sha256"]),
            reconstruction_sha256=str(value["reconstruction_sha256"]),
            structural_kind=str(value["structural_kind"]),
            segment_index=int(value["segment_index"]),
            extractor_version=str(value["extractor_version"]),
        )

    def validate(self, proof_text: str, proof_variant_id: str) -> None:
        if self.prefix_end <= 0 or self.prefix_end >= len(proof_text):
            raise ValueError(f"boundary {self.boundary_id} is outside its proof")
        prefix = proof_text[: self.prefix_end]
        continuation = proof_text[self.prefix_end :]
        if not prefix.strip() or not continuation.strip():
            raise ValueError(f"boundary {self.boundary_id} has an empty side")
        if self.prefix_sha256 != _sha256_text(prefix):
            raise ValueError(f"boundary {self.boundary_id} prefix hash mismatch")
        if self.continuation_sha256 != _sha256_text(continuation):
            raise ValueError(f"boundary {self.boundary_id} continuation hash mismatch")
        if self.reconstruction_sha256 != _sha256_text(prefix + continuation):
            raise ValueError(f"boundary {self.boundary_id} reconstruction hash mismatch")
        payload = (
            f"dataset-v3-boundary-v1\0{proof_variant_id}\0{self.prefix_end}"
            f"\0{self.structural_kind}"
        )
        expected_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.boundary_id != expected_id:
            raise ValueError(f"boundary {self.boundary_id} identity mismatch")


@dataclass(frozen=True)
class DatasetV3ProofVariant:
    proof_variant_id: str
    source_expression: str
    proof_text: str
    proof_form: ProofForm
    transformation_kind: str
    transformation_reason: str | None
    exact_text_fingerprint: str
    structural_fingerprint: str
    resolved_dependencies: tuple[str, ...]
    boundaries: tuple[StructuralBoundary, ...]
    verification: ProofVerification
    source_declaration_name: str
    source_repository: str
    source_revision: str
    source_file: str
    source_expression_verified: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetV3ProofVariant:
        return cls(
            proof_variant_id=str(value["proof_variant_id"]),
            source_expression=str(value["source_expression"]),
            proof_text=str(value["proof_text"]),
            proof_form=str(value["proof_form"]),  # type: ignore[arg-type]
            transformation_kind=str(value["transformation_kind"]),
            transformation_reason=(
                None
                if value.get("transformation_reason") is None
                else str(value["transformation_reason"])
            ),
            exact_text_fingerprint=str(value["exact_text_fingerprint"]),
            structural_fingerprint=str(value["structural_fingerprint"]),
            resolved_dependencies=tuple(
                str(item) for item in value.get("resolved_dependencies", [])
            ),
            boundaries=tuple(
                StructuralBoundary.from_dict(item)
                for item in value.get("boundaries", [])
            ),
            verification=ProofVerification.from_dict(value["verification"]),
            source_declaration_name=str(value["source_declaration_name"]),
            source_repository=str(value["source_repository"]),
            source_revision=str(value["source_revision"]),
            source_file=str(value["source_file"]),
            source_expression_verified=bool(value["source_expression_verified"]),
        )

    def validate(self, *, statement_id: str, environment_id: str) -> None:
        if not self.source_expression.strip() or not self.proof_text.strip():
            raise ValueError(f"proof variant {self.proof_variant_id} is empty")
        if self.proof_form not in {
            "tactic",
            "term",
            "equations",
            "where",
            "generated-tactic",
        }:
            raise ValueError(f"unsupported proof form: {self.proof_form}")
        if self.transformation_kind == "none" and self.transformation_reason is not None:
            raise ValueError("an untransformed proof cannot declare a transformation reason")
        if self.transformation_kind != "none" and not self.transformation_reason:
            raise ValueError("a transformed proof must declare its technical reason")
        if self.verification.status != "accepted":
            raise ValueError(f"proof variant {self.proof_variant_id} is not verified")
        if self.verification.environment_id != environment_id:
            raise ValueError("proof verification environment does not match the record")
        if self.exact_text_fingerprint != _sha256_text(self.proof_text):
            raise ValueError(f"proof variant {self.proof_variant_id} text hash mismatch")
        payload = (
            f"dataset-v3-proof-variant-v1\0{statement_id}\0"
            f"{self.exact_text_fingerprint}\0{self.source_repository}\0"
            f"{self.source_revision}\0{self.source_file}"
        )
        expected_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.proof_variant_id != expected_id:
            raise ValueError(f"proof variant {self.proof_variant_id} identity mismatch")
        offsets = [boundary.prefix_end for boundary in self.boundaries]
        if offsets != sorted(set(offsets)):
            raise ValueError(f"proof variant {self.proof_variant_id} repeats a boundary")
        for boundary in self.boundaries:
            boundary.validate(self.proof_text, self.proof_variant_id)


@dataclass(frozen=True)
class DatasetV3Record:
    schema_version: str
    statement_id: str
    canonical_declaration: str
    normalized_statement_fingerprint: str
    role: DatasetRole
    theorem_mass_numerator: int
    theorem_mass_denominator: int
    provenance: str
    environment: EnvironmentContext
    proof_variants: tuple[DatasetV3ProofVariant, ...]
    topic_tags: tuple[str, ...]
    memberships: tuple[str, ...]
    derivation_family_fingerprint: str | None = None
    generator_family: str | None = None
    structural_class: str | None = None
    logic_shape: str | None = None
    normalized_proof_dag: str | None = None
    source_lemma_ids: tuple[str, ...] = ()
    source_relation_edges: tuple[tuple[str, str, str], ...] = ()
    shortcut_checks: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetV3Record:
        schema_version = str(value["schema_version"])
        if schema_version != DATASET_V3_SCHEMA_VERSION:
            raise ValueError(f"unknown Dataset-v3 record schema: {schema_version}")
        record = cls(
            schema_version=schema_version,
            statement_id=str(value["statement_id"]),
            canonical_declaration=str(value["canonical_declaration"]),
            normalized_statement_fingerprint=str(
                value["normalized_statement_fingerprint"]
            ),
            role=str(value["role"]),  # type: ignore[arg-type]
            theorem_mass_numerator=int(value["theorem_mass_numerator"]),
            theorem_mass_denominator=int(value["theorem_mass_denominator"]),
            provenance=str(value["provenance"]),
            environment=EnvironmentContext.from_dict(value["environment"]),
            proof_variants=tuple(
                DatasetV3ProofVariant.from_dict(item)
                for item in value["proof_variants"]
            ),
            topic_tags=tuple(str(item) for item in value.get("topic_tags", [])),
            memberships=tuple(str(item) for item in value.get("memberships", [])),
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
            logic_shape=(
                None if value.get("logic_shape") is None else str(value["logic_shape"])
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
            shortcut_checks=tuple(
                str(item) for item in value.get("shortcut_checks", [])
            ),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.role not in {"training", "validation", "test"}:
            raise ValueError(f"unsupported Dataset-v3 role: {self.role}")
        if self.theorem_mass_denominator <= 0 or self.theorem_mass_numerator <= 0:
            raise ValueError("theorem mass must be positive")
        if not self.proof_variants:
            raise ValueError(f"statement {self.statement_id} has no proof variant")
        variant_ids = [item.proof_variant_id for item in self.proof_variants]
        if variant_ids != sorted(set(variant_ids)):
            raise ValueError(f"statement {self.statement_id} repeats a proof variant")
        for variant in self.proof_variants:
            variant.validate(
                statement_id=self.statement_id,
                environment_id=self.environment.environment_id,
            )
        if self.provenance == "synthetic":
            required = (
                self.derivation_family_fingerprint,
                self.generator_family,
                self.structural_class,
                self.logic_shape,
                self.normalized_proof_dag,
            )
            if any(not item for item in required) or not self.source_lemma_ids:
                raise ValueError("synthetic record lacks derivation metadata")
        elif any(
            item is not None
            for item in (
                self.derivation_family_fingerprint,
                self.generator_family,
                self.structural_class,
                self.logic_shape,
                self.normalized_proof_dag,
            )
        ):
            raise ValueError("real-source record contains synthetic derivation metadata")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DerivedExampleRef:
    schema_version: str
    example_id: str
    statement_id: str
    proof_variant_id: str
    kind: ExampleKind
    boundary_id: str | None
    mass_numerator: int
    mass_denominator: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DerivedExampleRef:
        example = cls(
            schema_version=str(value["schema_version"]),
            example_id=str(value["example_id"]),
            statement_id=str(value["statement_id"]),
            proof_variant_id=str(value["proof_variant_id"]),
            kind=str(value["kind"]),  # type: ignore[arg-type]
            boundary_id=(
                None if value.get("boundary_id") is None else str(value["boundary_id"])
            ),
            mass_numerator=int(value["mass_numerator"]),
            mass_denominator=int(value["mass_denominator"]),
        )
        example.validate()
        return example

    def validate(self) -> None:
        if self.schema_version != DATASET_V3_VIEW_SCHEMA_VERSION:
            raise ValueError(f"unknown Dataset-v3 view schema: {self.schema_version}")
        if self.kind not in {"whole", "continuation"}:
            raise ValueError(f"unsupported Dataset-v3 example kind: {self.kind}")
        if (self.kind == "whole") != (self.boundary_id is None):
            raise ValueError("whole/continuation boundary reference mismatch")
        if self.mass_numerator <= 0 or self.mass_denominator <= 0:
            raise ValueError("example mass must be positive")
        payload = "\0".join(
            (
                "dataset-v3-example-v1",
                self.statement_id,
                self.proof_variant_id,
                self.kind,
                self.boundary_id or "whole",
            )
        )
        expected_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.example_id != expected_id:
            raise ValueError(f"example {self.example_id} identity mismatch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
