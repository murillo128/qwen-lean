from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ResultCategory = Literal[
    "verified",
    "lean_rejected",
    "empty_candidate",
    "verifier_timeout",
    "verifier_error",
    "generation_error",
]
CandidateSource = Literal["fixture", "model"]

RESULT_SCHEMA_VERSION = "phase0-v1"
PHASE1_RESULT_SCHEMA_VERSION = "phase1-v1"
SUPPORTED_RESULT_SCHEMA_VERSIONS = {
    RESULT_SCHEMA_VERSION,
    PHASE1_RESULT_SCHEMA_VERSION,
}
RESULT_CATEGORIES = {
    "verified",
    "lean_rejected",
    "empty_candidate",
    "verifier_timeout",
    "verifier_error",
    "generation_error",
}


@dataclass(frozen=True)
class TaskRecord:
    id: str
    preamble: str
    declaration: str
    declaration_name: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskRecord:
        return cls(
            id=str(value["id"]),
            preamble=str(value["preamble"]),
            declaration=str(value["declaration"]),
            declaration_name=str(value["declaration_name"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunMetadata:
    candidate_source: CandidateSource
    task_source: str
    prompt_format_id: str
    lean_toolchain: str
    mathlib_revision: str
    verifier_timeout_seconds: float
    schema_version: str = RESULT_SCHEMA_VERSION
    model_id: str | None = None
    tokenizer_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    workload_id: str | None = None
    benchmark_split: str | None = None
    benchmark_repository: str | None = None
    benchmark_revision: str | None = None
    verifier_environment: dict[str, Any] | None = None
    candidates_per_task: int | None = None
    inference_engine: str | None = None
    inference_engine_version: str | None = None
    generation_settings: dict[str, Any] | None = None
    runtime: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunMetadata:
        candidate_source = value["candidate_source"]
        if candidate_source not in {"fixture", "model"}:
            raise ValueError(f"unknown candidate source: {candidate_source}")
        if value.get("schema_version") not in SUPPORTED_RESULT_SCHEMA_VERSIONS:
            raise ValueError(f"unknown result schema: {value.get('schema_version')}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    task_id: str
    candidate_id: str
    candidate_index: int
    candidate_text: str
    category: ResultCategory
    lean_exit_code: int | None
    diagnostics: dict[str, str]
    generation_latency_seconds: float | None
    verification_latency_seconds: float | None
    total_latency_seconds: float
    generated_token_count: int | None = None
    finish_reason: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CandidateResult:
        category = value["category"]
        if category not in RESULT_CATEGORIES:
            raise ValueError(f"unknown result category: {category}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
