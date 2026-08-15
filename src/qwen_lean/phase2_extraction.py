from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .phase2_corpus import (
    COMPONENT_ALGORITHM,
    FINGERPRINT_ALGORITHM,
    RawTheorem,
    evaluate_raw_theorem,
    exclude_ambiguous_record_identities,
    exclude_contamination,
    finalize_records,
    minif2f_statement_fingerprints,
    split_statistics,
    summarize_token_lengths,
    validate_record_source_text,
    validate_split_hygiene,
    write_jsonl_splits,
)
from .phase2_schema import (
    PHASE2_CONFIG_SCHEMA_VERSION,
    PHASE2_DATASET_SCHEMA_VERSION,
    PHASE2_MANIFEST_SCHEMA_VERSION,
    MathlibProofRecord,
    SourcePosition,
    SourceSpan,
)


@dataclass(frozen=True)
class Phase2Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Phase2Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != PHASE2_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unknown Phase 2 config schema: {value.get('schema_version')}"
            )
        if value["dataset"]["schema_version"] != PHASE2_DATASET_SCHEMA_VERSION:
            raise ValueError("Phase 2 config has an unsupported record schema")
        return cls(path=path.resolve(), value=value)

    @property
    def source(self) -> dict[str, Any]:
        return self.value["source"]

    @property
    def extractor(self) -> dict[str, Any]:
        return self.value["extractor"]

    @property
    def contamination_reference(self) -> dict[str, Any]:
        return self.value["benchmark_contamination_reference"]

    @property
    def tokenizer(self) -> dict[str, Any]:
        return self.value["tokenizer"]

    @property
    def split(self) -> dict[str, Any]:
        return self.value["split"]

    def validate_project_pins(self) -> None:
        project_root = self.path.parents[1]
        manifest = json.loads(
            (project_root / "lake-manifest.json").read_text(encoding="utf-8")
        )
        try:
            mathlib = next(
                package
                for package in manifest["packages"]
                if package["name"] == "mathlib"
            )
        except StopIteration as error:
            raise ValueError(
                "project Lake manifest has no mathlib dependency"
            ) from error
        expected = self.source
        if (
            mathlib["rev"] != expected["revision"]
            or mathlib["inputRev"] != expected["input_revision"]
            or str(mathlib["url"]).removesuffix(".git")
            != str(expected["repository"]).removesuffix(".git")
        ):
            raise ValueError(
                "Phase 2 source config differs from the project Lake manifest"
            )
        toolchain = (
            (project_root / "lean-toolchain").read_text(encoding="utf-8").strip()
        )
        if toolchain != expected["lean_toolchain"]:
            raise ValueError("Phase 2 Lean toolchain differs from the project pin")
        phase1 = json.loads(
            (project_root / "config/phase1-minif2f.json").read_text(encoding="utf-8")
        )["model"]
        if (
            phase1["tokenizer_id"] != self.tokenizer["model_id"]
            or phase1["tokenizer_revision"] != self.tokenizer["revision"]
        ):
            raise ValueError("Phase 2 tokenizer differs from the accepted Phase 1 pin")


@dataclass(frozen=True)
class ExtractionDiagnostics:
    filter_counts: dict[str, int]
    details: tuple[dict[str, str], ...]
    traced_files: int
    traced_theorems: int
    validated_source_identities: int


def _position(value: Any) -> SourcePosition:
    return SourcePosition(line=int(value.line_nb), column=int(value.column_nb))


def _span(start: Any, end: Any) -> SourceSpan:
    return SourceSpan(start=_position(start), end=_position(end))


def _declaration_kind(traced_theorem: Any) -> str:
    class_name = type(traced_theorem.ast).__name__
    if class_name == "CommandTheoremNode":
        return "theorem"
    if class_name in {"LemmaNode", "MathlibTacticLemmaNode"}:
        return "lemma"
    return class_name


def traced_theorem_to_raw(traced_theorem: Any, file_path: str) -> RawTheorem:
    proof_node = traced_theorem.get_proof_node()
    proof_start, proof_end = proof_node.get_closure()
    return RawTheorem(
        file_path=file_path,
        declaration_name=str(traced_theorem.theorem.full_name or ""),
        declaration_kind=_declaration_kind(traced_theorem),
        source_span=_span(traced_theorem.start, traced_theorem.end),
        declaration_span=_span(traced_theorem.start, proof_start),
        proof_span=_span(proof_start, proof_end),
        declaration=traced_theorem.get_theorem_statement(),
        proof=traced_theorem.get_tactic_proof(),
        premises=tuple(traced_theorem.get_premise_full_names()),
        is_private=bool(traced_theorem.is_private),
    )


def extract_traced_records(
    traced_repo: Any,
    config: Phase2Config,
    *,
    selected_files: set[str] | None = None,
) -> tuple[list[Any], ExtractionDiagnostics]:
    source_repository = str(config.source["repository"])
    source_revision = str(config.source["revision"])
    records = []
    counts: Counter[str] = Counter()
    details: list[dict[str, str]] = []
    traced_file_count = 0
    traced_theorem_count = 0
    validated_source_identities = 0
    if hasattr(traced_repo, "iter_traced_files"):
        traced_files = traced_repo.iter_traced_files()
    else:
        traced_files = iter(
            sorted(traced_repo.traced_files, key=lambda item: str(item.path))
        )
    observed_paths: set[str] = set()
    for traced_file in traced_files:
        file_path = str(traced_file.path)
        observed_paths.add(file_path)
        if not file_path.startswith("Mathlib/") or not file_path.endswith(".lean"):
            continue
        if selected_files is not None and file_path not in selected_files:
            continue
        traced_file_count += 1
        if traced_file_count % 250 == 0:
            print(
                f"Phase 2 extraction: processed {traced_file_count} traced source files",
                file=sys.stderr,
                flush=True,
            )
        try:
            source = (Path(traced_repo.root_dir) / file_path).read_text(
                encoding="utf-8"
            )
        except OSError as error:
            counts["source_read_error"] += 1
            details.append({"file_path": file_path, "error": repr(error)})
            continue
        try:
            theorems = traced_file.get_traced_theorems()
        except Exception as error:  # noqa: BLE001 - retain every LeanDojo file loss.
            counts["extraction_error"] += 1
            details.append({"file_path": file_path, "error": repr(error)})
            continue
        for traced_theorem in theorems:
            traced_theorem_count += 1
            name = str(getattr(traced_theorem.theorem, "full_name", ""))
            try:
                raw = traced_theorem_to_raw(traced_theorem, file_path)
                result = evaluate_raw_theorem(
                    raw,
                    source_repository=source_repository,
                    source_revision=source_revision,
                )
            except Exception as error:  # noqa: BLE001 - retain every theorem loss.
                counts["extraction_error"] += 1
                details.append(
                    {
                        "file_path": file_path,
                        "declaration_name": name,
                        "error": repr(error),
                    }
                )
                continue
            if result.record is None:
                reason = result.filter_reason or "unknown_filter"
                counts[reason] += 1
                if result.detail is not None:
                    details.append(
                        {
                            "file_path": file_path,
                            "declaration_name": name,
                            "filter_reason": reason,
                            "detail": result.detail,
                        }
                    )
            else:
                try:
                    validate_record_source_text(result.record, source)
                except ValueError as error:
                    counts["source_identity_mismatch"] += 1
                    details.append(
                        {
                            "file_path": file_path,
                            "declaration_name": name,
                            "filter_reason": "source_identity_mismatch",
                            "detail": str(error),
                        }
                    )
                    continue
                validated_source_identities += 1
                records.append(result.record)
    if selected_files is not None:
        missing = sorted(selected_files - observed_paths)
        if missing:
            raise ValueError(
                f"pilot files are absent from the LeanDojo trace: {missing}"
            )
    trace_read_errors = tuple(getattr(traced_repo, "trace_read_errors", ()))
    for detail in trace_read_errors:
        counts["trace_parse_error"] += 1
        details.append(
            {
                "file_path": str(detail["file_path"]),
                "filter_reason": "trace_parse_error",
                "detail": str(detail["error"]),
            }
        )
    traced_file_count += len(trace_read_errors)
    return records, ExtractionDiagnostics(
        filter_counts=dict(sorted(counts.items())),
        details=tuple(details),
        traced_files=traced_file_count,
        traced_theorems=traced_theorem_count,
        validated_source_identities=validated_source_identities,
    )


def validate_traced_repo(traced_repo: Any, config: Phase2Config) -> dict[str, Any]:
    expected_repository = (
        str(config.source["repository"]).removesuffix(".git").rstrip("/")
    )
    traced_repository = str(traced_repo.repo.url).removesuffix(".git").rstrip("/")
    if traced_repository != expected_repository:
        raise ValueError(
            f"LeanDojo traced repository mismatch: expected {expected_repository}, "
            f"got {traced_repository}"
        )
    expected_revision = str(config.source["revision"])
    traced_revision = str(traced_repo.repo.commit)
    if traced_revision != expected_revision:
        raise ValueError(
            f"LeanDojo traced revision mismatch: expected {expected_revision}, got {traced_revision}"
        )
    root = Path(traced_repo.root_dir).resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != expected_revision:
        raise ValueError(
            "LeanDojo trace root is not the configured pristine source revision"
        )
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "Mathlib"],
        cwd=root,
        check=False,
    )
    if source_diff.returncode != 0:
        raise ValueError("LeanDojo trace root has modified Mathlib source files")
    toolchain = (root / "lean-toolchain").read_text(encoding="utf-8").strip()
    if toolchain != str(config.source["lean_toolchain"]):
        raise ValueError(
            f"Lean toolchain mismatch: expected {config.source['lean_toolchain']}, got {toolchain}"
        )
    traced_files_total = int(
        getattr(
            traced_repo,
            "traced_file_count",
            len(getattr(traced_repo, "traced_files", ())),
        )
    )
    return {
        "root": str(root),
        "revision": traced_revision,
        "lean_toolchain": toolchain,
        "traced_files": traced_files_total,
    }


def validate_checkout_revision(path: Path, expected_revision: str, label: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"{label} is not a readable Git checkout: {path}")
    revision = completed.stdout.strip()
    if revision != expected_revision:
        raise ValueError(
            f"{label} revision mismatch: expected {expected_revision}, got {revision}"
        )
    return revision


def load_tokenizer(config: Phase2Config, *, cache_dir: Path | None = None) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(config.tokenizer["model_id"]),
        revision=str(config.tokenizer["revision"]),
        cache_dir=None if cache_dir is None else str(cache_dir),
        trust_remote_code=False,
    )


def _contamination_fingerprints(
    config: Phase2Config, mini_root: Path
) -> tuple[dict[str, str], dict[str, int]]:
    reference = config.contamination_reference
    validate_checkout_revision(
        mini_root,
        str(reference["revision"]),
        "miniF2F contamination reference",
    )
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *reference["source_paths"]],
        cwd=mini_root,
        check=False,
    )
    if source_diff.returncode != 0:
        raise ValueError(
            "miniF2F contamination source files differ from the pinned revision"
        )
    sources = []
    source_counts: dict[str, int] = {}
    for relative_path in reference["source_paths"]:
        source = (mini_root / str(relative_path)).read_text(encoding="utf-8")
        parsed = minif2f_statement_fingerprints([source])
        source_counts[str(relative_path)] = len(parsed)
        expected = int(
            reference["expected_primary_statement_counts"][str(relative_path)]
        )
        if len(parsed) != expected:
            raise ValueError(
                f"miniF2F primary statement count changed for {relative_path}: "
                f"expected {expected}, got {len(parsed)}"
            )
        sources.append(source)
    fingerprints = minif2f_statement_fingerprints(sources)
    return fingerprints, source_counts


def build_phase2_corpus(
    traced_repo: Any,
    config: Phase2Config,
    *,
    mini_root: Path,
    output_dir: Path,
    tokenizer_cache: Path | None = None,
    pilot: bool = False,
) -> tuple[list[MathlibProofRecord], dict[str, Any]]:
    config.validate_project_pins()
    trace_metadata = validate_traced_repo(traced_repo, config)
    print("Phase 2 extraction: reading LeanDojo theorem traces", file=sys.stderr)
    selected_files = set(config.value["pilot_files"]) if pilot else None
    extracted, diagnostics = extract_traced_records(
        traced_repo, config, selected_files=selected_files
    )
    extracted, ambiguous_records, ambiguous_ids = exclude_ambiguous_record_identities(
        extracted
    )
    filter_counts = Counter(diagnostics.filter_counts)
    diagnostic_details = list(diagnostics.details)
    if ambiguous_records:
        filter_counts["ambiguous_record_identity"] += len(ambiguous_records)
        diagnostic_details.extend(
            {
                "file_path": record.file_path,
                "declaration_name": record.declaration_name,
                "filter_reason": "ambiguous_record_identity",
                "detail": (
                    "reported source revision, file path, and fully qualified name "
                    "identify multiple traced declarations"
                ),
            }
            for record in ambiguous_records
        )
    reference, reference_source_counts = _contamination_fingerprints(config, mini_root)
    retained, excluded = exclude_contamination(extracted, set(reference.values()))
    if not retained:
        raise ValueError("Phase 2 extraction retained no records")
    tokenizer = load_tokenizer(config, cache_dir=tokenizer_cache)
    print(
        f"Phase 2 extraction: tokenizing {len(retained)} retained records",
        file=sys.stderr,
    )
    proportions: Mapping[str, float] = config.split["proportions"]
    records = finalize_records(
        retained,
        tokenizer,
        proportions,
        seed=str(config.split["seed"]),
    )
    hygiene = validate_split_hygiene(records)
    split_stats = split_statistics(records, proportions)
    if not pilot and len(records) < 50_000:
        raise ValueError(
            f"full eligible corpus has {len(records)} records, below the 50,000 discrepancy floor"
        )
    if not pilot and any(split_stats[name]["records"] == 0 for name in split_stats):
        raise ValueError("full corpus has an empty split")
    component_counts = Counter(record.component_id for record in records)
    largest_component = max(component_counts.values(), default=0)
    deviations = {
        name: abs(split_stats[name]["proportion"] - float(proportions[name]))
        for name in split_stats
    }
    largest_component_proportion = largest_component / len(records)
    normally_within_tolerance = all(value <= 0.02 for value in deviations.values())
    if (
        not pilot
        and not normally_within_tolerance
        and largest_component_proportion <= 0.02
    ):
        raise ValueError(
            "split proportions exceed the two-percentage-point tolerance without "
            "an unusually large duplicate-connected component"
        )
    manifest = {
        "schema_version": PHASE2_MANIFEST_SCHEMA_VERSION,
        "dataset_schema_version": PHASE2_DATASET_SCHEMA_VERSION,
        "mode": "pilot" if pilot else "full",
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            **config.source,
            "observed_revision": trace_metadata["revision"],
        },
        "extractor": {
            **config.extractor,
            "package": "lean-dojo-v2",
            "entrypoint": (
                "uv run --frozen --project tools/phase2-extractor "
                "python tools/phase2_extract.py"
            ),
            "trace_build_dependencies": bool(config.extractor["build_dependencies"]),
            "traced_files_total": trace_metadata["traced_files"],
            "compatibility_note": (
                "the pinned LeanDojo-v2 static version check warns above Lean 4.30.1; "
                "Phase 2 relies on the required pinned pilot and full extraction/verification "
                "evidence for Lean 4.32.0 compatibility"
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "datasets": version("datasets"),
                "tokenizers": version("tokenizers"),
                "transformers": version("transformers"),
            },
        },
        "record_identity": "sha256(schema_version, source_revision, file_path, fully_qualified_name)",
        "source_positions": "one-indexed, end-exclusive LeanDojo-v2 source positions",
        "fingerprint": {
            "algorithm": FINGERPRINT_ALGORITHM,
            "normalization": (
                "comment-free Lean lexical tokens with insignificant whitespace removed "
                "and only the declared theorem/lemma identifier replaced; the "
                "semantically equivalent theorem/lemma introducers share one token"
            ),
        },
        "split_policy": {
            "algorithm": COMPONENT_ALGORITHM,
            "seed": config.split["seed"],
            "proportions": proportions,
            "primary_group": "source file",
            "component_edge": "shared normalized statement fingerprint",
            "balance_unit": "retained record count",
            "semantics": {
                "train": "may be used for SFT fitting",
                "validation": "may be used for training diagnostics and model selection",
                "heldout": "must not be used for fitting or model/hyperparameter selection",
            },
        },
        "tokenizer": {
            **config.tokenizer,
            "counting_convention": {
                "declaration_and_proof": "tokenize `<declaration> := <full by proof>`",
                "declaration_and_completion": (
                    "tokenize `<declaration> := by\\n<completion>`"
                ),
                "chat_template_applied": False,
                "special_tokens_added": False,
            },
        },
        "counts": {
            "traced_source_files": diagnostics.traced_files,
            "traced_theorem_or_lemma_nodes": diagnostics.traced_theorems,
            "retained_before_contamination": len(extracted),
            "source_identities_validated": diagnostics.validated_source_identities,
            "ambiguous_record_identities": len(ambiguous_ids),
            "excluded_ambiguous_record_identity": len(ambiguous_records),
            "excluded_exact_minif2f_overlap": len(excluded),
            "final_records": len(records),
            "filter_reasons": dict(sorted(filter_counts.items())),
            "extraction_diagnostic_details": len(diagnostic_details),
        },
        "contamination": {
            "reference": config.contamination_reference,
            "primary_statements_by_file": reference_source_counts,
            "unique_primary_names": len(reference),
            "excluded_records": [
                {
                    "id": record.id,
                    "file_path": record.file_path,
                    "declaration_name": record.declaration_name,
                    "statement_fingerprint": record.statement_fingerprint,
                }
                for record in excluded
            ],
            "remaining_exact_statement_matches": 0,
        },
        "splits": split_stats,
        "split_hygiene": hygiene,
        "split_deviation": {
            "absolute": deviations,
            "normally_within_two_percentage_points": normally_within_tolerance,
            "largest_component_records": largest_component,
            "largest_component_proportion": largest_component_proportion,
        },
        "token_length_statistics": summarize_token_lengths(records),
        "long_examples_filtered": 0,
        "publication": {
            "full_dataset_uploaded": False,
            "distribution_license_review_performed": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written_counts = write_jsonl_splits(records, output_dir)
    if any(
        written_counts[name] != split_stats[name]["records"] for name in split_stats
    ):
        raise AssertionError("written split counts differ from computed counts")
    with (output_dir / "extraction-diagnostics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for detail in diagnostic_details:
            handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records, manifest


def write_compact_evidence(
    artifact_dir: Path, evidence_dir: Path, verification_path: Path | None = None
) -> None:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    compact = {
        key: manifest[key]
        for key in (
            "schema_version",
            "dataset_schema_version",
            "mode",
            "created_at",
            "source",
            "extractor",
            "runtime",
            "fingerprint",
            "split_policy",
            "tokenizer",
            "counts",
            "contamination",
            "splits",
            "split_hygiene",
            "split_deviation",
            "token_length_statistics",
            "long_examples_filtered",
            "publication",
        )
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "manifest.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if verification_path is not None:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        (evidence_dir / "verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
