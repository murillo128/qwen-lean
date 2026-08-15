from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .phase2_corpus import (
    read_jsonl_records,
    substitute_span,
    validate_record_source_text,
)
from .phase2_extraction import Phase2Config, validate_checkout_revision
from .phase2_schema import (
    PHASE2_VERIFICATION_SCHEMA_VERSION,
    SPLIT_NAMES,
    MathlibProofRecord,
)

INVALID_PROOF = "by\n  exact __qwen_lean_phase2_controlled_invalid_proof__"


@dataclass(frozen=True)
class LeanCheck:
    status: str
    exit_code: int | None
    latency_seconds: float
    diagnostic: str


def validate_record_source_identity(
    record: MathlibProofRecord, mathlib_root: Path
) -> str:
    source_path = mathlib_root / record.file_path
    source = source_path.read_text(encoding="utf-8")
    validate_record_source_text(record, source)
    return source


def _run_lean_source(
    source: str,
    mathlib_root: Path,
    *,
    timeout_seconds: float,
) -> LeanCheck:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="qwen-lean-phase2-") as temporary_dir:
            source_path = Path(temporary_dir) / "Reconstructed.lean"
            source_path.write_text(source, encoding="utf-8", newline="\n")
            completed = subprocess.run(
                ["lake", "env", "lean", "-E", "hasSorry", str(source_path)],
                cwd=mathlib_root,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        diagnostic = str(error.stderr or error.stdout or "Lean verification timed out")
        return LeanCheck(
            "timeout", None, time.perf_counter() - started, diagnostic[-2000:]
        )
    except (OSError, ValueError) as error:
        return LeanCheck(
            "infrastructure_error",
            None,
            time.perf_counter() - started,
            str(error),
        )
    diagnostic = (completed.stderr + "\n" + completed.stdout).replace(
        str(source_path), "Reconstructed.lean"
    )
    has_error = any(": error:" in line for line in diagnostic.splitlines())
    status = "accepted" if completed.returncode == 0 and not has_error else "rejected"
    return LeanCheck(
        status,
        completed.returncode,
        time.perf_counter() - started,
        "" if status == "accepted" else diagnostic[-2000:],
    )


def select_verification_sample(
    records: Sequence[MathlibProofRecord],
    sample_counts: Mapping[str, int],
    *,
    seed: str,
) -> list[MathlibProofRecord]:
    selected: list[MathlibProofRecord] = []
    for split in SPLIT_NAMES:
        count = int(sample_counts[split])
        candidates = sorted(
            (record for record in records if record.split == split),
            key=lambda record: hashlib.sha256(
                f"{seed}\0{record.id}".encode()
            ).hexdigest(),
        )
        if len(candidates) < count:
            raise ValueError(
                f"{split} has {len(candidates)} records but verification needs {count}"
            )
        distinct: list[MathlibProofRecord] = []
        repeated: list[MathlibProofRecord] = []
        seen_files: set[str] = set()
        for record in candidates:
            if record.file_path in seen_files:
                repeated.append(record)
            else:
                seen_files.add(record.file_path)
                distinct.append(record)
        selected.extend((distinct + repeated)[:count])
    return selected


def verify_phase2_sample(
    config: Phase2Config,
    artifact_dir: Path,
    mathlib_root: Path,
    output_path: Path,
    *,
    sample_counts: Mapping[str, int] | None = None,
    negative_substitutions: int | None = None,
    workers: int | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    validate_checkout_revision(
        mathlib_root, str(config.source["revision"]), "mathlib verification checkout"
    )
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "Mathlib"],
        cwd=mathlib_root,
        check=False,
    )
    if source_diff.returncode != 0:
        raise ValueError("mathlib verification source differs from the pinned revision")
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        record
        for split in SPLIT_NAMES
        for record in read_jsonl_records(artifact_dir / f"{split}.jsonl")
    ]
    verification_config = config.value["verification"]
    requested_counts = (
        dict(sample_counts)
        if sample_counts is not None
        else dict(verification_config["sample_counts"])
    )
    selected = select_verification_sample(
        records,
        requested_counts,
        seed=str(config.split["seed"]) + "-verification",
    )
    del records
    negative_count = int(
        verification_config["negative_substitutions"]
        if negative_substitutions is None
        else negative_substitutions
    )
    worker_count = int(verification_config["workers"] if workers is None else workers)
    timeout = float(
        verification_config["timeout_seconds"]
        if timeout_seconds is None
        else timeout_seconds
    )
    if negative_count > len(selected):
        raise ValueError("negative substitution count exceeds the verification sample")
    negative_ids = {record.id for record in selected[:negative_count]}

    def check_record(record: MathlibProofRecord) -> dict[str, Any]:
        try:
            source = validate_record_source_identity(record, mathlib_root)
        except (OSError, ValueError) as error:
            return {
                "id": record.id,
                "split": record.split,
                "file_path": record.file_path,
                "declaration_name": record.declaration_name,
                "source_identity": "failed",
                "original_proof": {"status": "not_run", "diagnostic": str(error)},
                "negative_substitution": None,
            }
        reconstructed = substitute_span(source, record.proof_span, record.proof)
        original = _run_lean_source(
            reconstructed, mathlib_root, timeout_seconds=timeout
        )
        negative: LeanCheck | None = None
        if record.id in negative_ids:
            controlled = substitute_span(source, record.proof_span, INVALID_PROOF)
            negative = _run_lean_source(
                controlled, mathlib_root, timeout_seconds=timeout
            )
        return {
            "id": record.id,
            "split": record.split,
            "file_path": record.file_path,
            "declaration_name": record.declaration_name,
            "source_identity": "matched",
            "original_proof": original.__dict__,
            "negative_substitution": None if negative is None else negative.__dict__,
        }

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(check_record, selected))
    accepted = sum(
        result["original_proof"]["status"] == "accepted" for result in results
    )
    source_matched = sum(result["source_identity"] == "matched" for result in results)
    negative_rejected = sum(
        result["negative_substitution"] is not None
        and result["negative_substitution"]["status"] == "rejected"
        for result in results
    )
    evidence = {
        "schema_version": PHASE2_VERIFICATION_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_schema_version": manifest["dataset_schema_version"],
        "source_repository": config.source["repository"],
        "source_revision": config.source["revision"],
        "lean_toolchain": config.source["lean_toolchain"],
        "sample_policy": {
            "seed": str(config.split["seed"]) + "-verification",
            "requested_by_split": requested_counts,
            "distinct_source_files_where_practical": True,
        },
        "negative_substitution": {
            "requested": negative_count,
            "proof": INVALID_PROOF,
        },
        "summary": {
            "sampled": len(results),
            "source_identity_matched": source_matched,
            "original_proofs_accepted": accepted,
            "negative_substitutions_rejected": negative_rejected,
            "infrastructure_failures": sum(
                result["original_proof"]["status"] == "infrastructure_error"
                for result in results
            ),
            "timeouts": sum(
                result["original_proof"]["status"] == "timeout" for result in results
            ),
        },
        "records": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if source_matched != len(results) or accepted != len(results):
        raise ValueError("not every sampled retained proof reconstructed and verified")
    if negative_rejected != negative_count:
        raise ValueError("a controlled invalid proof was not rejected")
    return evidence
