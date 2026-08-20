from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FULL_CACHE_VERSION = "dataset-v2-full-cache-v3"
FULL_SYNTHETIC_CACHE_VERSION = "dataset-v2-full-synthetic-cache-v4"
FULL_VERIFICATION_CACHE_VERSION = "dataset-v2-full-verification-cache-v5"
LEGACY_FULL_VERIFICATION_CACHE_VERSIONS = {
    "dataset-v2-full-verification-cache-v4"
}
COMPOSITION_BATCH_CACHE_VERSION = "dataset-v2-composition-batch-cache-v1"
FULL_SYNTHETIC_ACCEPTANCE_YIELD_FLOORS = {
    "generic": (1, 2),
    "prime-arithmetic-divisibility": (1, 2),
    "arithmetic-functions": (1, 2),
    "prime-counting-pnt": (1, 2),
    "zeta-analytic-number-theory": (1, 2),
    "riemann-core-bubble": (1, 2),
    "pnt-plus": (1, 2),
}

from qwen_lean.dataset_v2 import (  # noqa: E402
    assign_synthetic_roles,
    filter_clean_benchmark,
    iter_optimizer_examples,
    merge_statement_records,
    read_records,
    sha256_file,
    validate_role_isolation,
    write_records,
)
from qwen_lean.dataset_v2_composition import (  # noqa: E402
    build_composition_plans,
    find_missing_constants,
    lean_name_key,
    records_from_compositions,
    render_composition_source,
    render_constant_audit_source,
    render_shortcut_gate_source,
    run_composition_source,
    run_shortcut_gate_source,
    summarize_compositions,
    validate_composition_audits,
)
from qwen_lean.dataset_v2_contract import statement_id  # noqa: E402
from qwen_lean.dataset_v2_extraction import (  # noqa: E402
    DatasetV2Config,
    candidate_from_external_record,
    candidate_to_record,
    collapse_duplicate_candidates,
    collapse_duplicate_exclusions,
    extract_traced_files,
    repair_candidate_declaration_names,
    select_candidates,
    source_candidate_id,
    verify_transformed_candidates,
)
from qwen_lean.dataset_v2_pipeline import (  # noqa: E402
    PRIME_FAMILIES,
    annotate_candidate,
    build_prime_coverage_manifest,
    build_source_dispositions,
    composition_pools,
    dataset_manifest,
    distribute_prime_counts,
    historical_source_crosswalk,
    load_riemann_metadata,
    read_jsonl,
    select_train_probe,
)
from qwen_lean.minif2f import materialize_benchmark_source  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate the frozen lean-whole-proof-v2 corpus"
    )
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/dataset-v2.json")
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "artifacts/riemann/sources/PrimeNumberTheoremAnd",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=ROOT / "artifacts/phase2/leandojo-trace/mathlib4",
    )
    parser.add_argument(
        "--mini-root",
        type=Path,
        default=ROOT / "artifacts/phase2/tooling/miniF2F",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--composition-workers",
        type=int,
        default=2,
        help="maximum concurrent memory-heavy Lean composition/shortcut batches",
    )
    parser.add_argument("--verification-timeout", type=float, default=600.0)
    parser.add_argument("--composition-batch-size", type=int, default=256)
    parser.add_argument(
        "--resume-preflight",
        action="store_true",
        help="reuse local preflight extraction/verification caches from this output directory",
    )
    parser.add_argument(
        "--resume-full",
        action="store_true",
        help="reuse completed full-build stages from this output directory",
    )
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def _write_jsonl_gz(path: Path, values: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for value in values:
                    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def _source_equivalence(
    *, trace_root: Path, target_mathlib: Path, config: DatasetV2Config
) -> dict[str, Any]:
    old_revision = str(config.value["historical_inputs"]["mathlib_v1_revision"])
    observed_old = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=trace_root, text=True, capture_output=True, check=True
    ).stdout.strip()
    if observed_old != old_revision:
        raise RuntimeError("historical LeanDojo trace checkout revision mismatch")
    changed = subprocess.run(
        ["git", "diff", "--name-only", old_revision, config.environment["mathlib_revision"], "--", "Mathlib"],
        cwd=target_mathlib,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    if changed:
        raise RuntimeError(
            f"historical trace positions are unsafe: {len(changed)} target Mathlib files changed"
        )
    return {
        "historical_trace_revision": old_revision,
        "target_mathlib_revision": config.environment["mathlib_revision"],
        "changed_mathlib_source_files": 0,
        "position_reuse": "accepted-byte-identical-source",
    }


def _candidate_source_span_key(candidate: Any) -> tuple[Any, ...]:
    return (
        candidate.source_repository,
        candidate.source_revision,
        candidate.file_path,
        candidate.declaration_span.start.line,
        candidate.declaration_span.start.column,
        candidate.declaration_span.end.line,
        candidate.declaration_span.end.column,
        candidate.source_expression,
    )


def _transfer_cached_verification(
    candidates: list[Any], cached: list[Any]
) -> tuple[list[Any], list[int]]:
    cached_by_span = {
        _candidate_source_span_key(item): item
        for item in cached
    }
    transferred: list[Any] = []
    missing_indexes: list[int] = []
    for candidate in candidates:
        previous = cached_by_span.get(_candidate_source_span_key(candidate))
        if previous is None and candidate.transformation_kind != "none":
            missing_indexes.append(len(transferred))
            transferred.append(candidate)
            continue
        if previous is None:
            transferred.append(candidate)
            continue
        transferred.append(
            replace(
                candidate,
                verification_status=previous.verification_status,
                verification_method=previous.verification_method,
                verification_evidence_id=previous.verification_evidence_id,
                verification_diagnostic=previous.verification_diagnostic,
            )
        )
    return transferred, missing_indexes


def _rebuild_file_verification(
    candidates: list[Any], previous: list[Any], supplemental: list[Any]
) -> list[Any]:
    previous_by_file = {item.file_path: item for item in previous}
    supplemental_by_file = {item.file_path: item for item in supplemental}
    pending_by_file: dict[str, list[Any]] = defaultdict(list)
    for candidate in candidates:
        if candidate.transformation_kind != "none":
            pending_by_file[candidate.file_path].append(candidate)
    rebuilt: list[Any] = []
    for file_path, group in sorted(pending_by_file.items()):
        prior = previous_by_file.get(file_path)
        extra = supplemental_by_file.get(file_path)
        rejected = [item for item in group if item.verification_status != "accepted"]
        status = (
            "accepted"
            if not rejected
            else "rejected"
            if len(rejected) == len(group)
            else "partial"
        )
        diagnostics = [
            item
            for item in (
                getattr(prior, "diagnostic", ""),
                getattr(extra, "diagnostic", ""),
                *(item.verification_diagnostic for item in rejected),
            )
            if item
        ]
        rebuilt.append(
            type(prior or extra)(
                file_path=file_path,
                candidate_ids=tuple(source_candidate_id(item) for item in group),
                rejected_candidate_ids=tuple(
                    source_candidate_id(item) for item in rejected
                ),
                status=status,
                exit_code=(
                    0
                    if status == "accepted"
                    else getattr(extra or prior, "exit_code", None)
                ),
                latency_seconds=(
                    float(getattr(prior, "latency_seconds", 0.0))
                    + float(getattr(extra, "latency_seconds", 0.0))
                ),
                diagnostic="\n".join(diagnostics)[-4000:],
            )
        )
    return rebuilt


def _iter_mathlib_traces(trace_root: Path, selected_files: list[str] | None = None):
    os.environ.setdefault("GITHUB_ACCESS_TOKEN", "")
    from lean_dojo_v2.lean_dojo import LeanGitRepo, TracedFile

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=trace_root, text=True, capture_output=True, check=True
    ).stdout.strip()
    repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", revision)

    class StreamedRepo:
        def __init__(self) -> None:
            self.repo = repo
            self.root_dir = trace_root
            self.dependencies: dict[str, object] = {}

    streamed = StreamedRepo()

    class FailedTracedFile:
        def __init__(self, path: Path, error: Exception) -> None:
            self.path = path
            self.error = error

        def get_traced_theorems(self) -> list[Any]:
            raise RuntimeError(
                f"trace-deserialization-error:{type(self.error).__name__}:{self.error}"
            ) from self.error

    source_paths = (
        [trace_root / item for item in sorted(selected_files)]
        if selected_files is not None
        else sorted(trace_root.glob("Mathlib/**/*.lean"))
    )
    for index, source_path in enumerate(source_paths, start=1):
        relative = source_path.relative_to(trace_root)
        ast_path = trace_root / ".lake/build/ir" / relative.with_suffix(".ast.json")
        dep_path = trace_root / ".lake/build/ir" / relative.with_suffix(".dep_paths")
        if not ast_path.is_file() or not dep_path.is_file():
            raise RuntimeError(f"incomplete historical trace for {relative}")
        try:
            traced = TracedFile.from_traced_file(trace_root, ast_path, repo)
            traced.traced_repo = streamed
        except Exception as error:  # noqa: BLE001 - preserve an explicit file disposition.
            print(
                f"Dataset v2: classified trace deserialization failure for {relative}: "
                f"{type(error).__name__}",
                file=sys.stderr,
                flush=True,
            )
            traced = FailedTracedFile(relative, error)
        if index % 500 == 0:
            print(f"Dataset v2: read {index} traced Mathlib files", file=sys.stderr, flush=True)
        yield traced


def _load_pnt_candidates(
    *, target_root: Path, output_dir: Path, config: DatasetV2Config
) -> tuple[list[Any], dict[str, Any]]:
    external_path = ROOT / "data/riemann/external/riemann-external-lean-v1.jsonl"
    values = read_jsonl(external_path)
    names = [str(item["declaration_name"]) for item in values]
    audit_path = output_dir / "lean/PntConstantAudit.lean"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(render_constant_audit_source(names), encoding="utf-8")
    run = run_composition_source(audit_path, target_root=target_root)
    if run.status != "accepted" or len(run.audits) != len(names):
        raise RuntimeError(f"PNT constant audit failed: {run.diagnostic}")
    audits = {audit.name: audit for audit in run.audits}
    evidence_id = "pnt-constant-audit:" + sha256_file(audit_path)
    candidates = [
        annotate_candidate(
            candidate_from_external_record(
                value,
                source_root=target_root,
                resolved_dependencies=audits[str(value["declaration_name"])].actual_dependencies,
                verification_evidence_id=evidence_id,
            )
        )
        for value in values
    ]
    expected = int(config.value["pnt_plus"]["expected_accepted_declarations"])
    if len(candidates) != expected or set(audits) != set(names):
        raise RuntimeError("PNT accepted declaration inventory mismatch")
    return candidates, {
        "status": run.status,
        "accepted_declarations": len(candidates),
        "dependency_audits": len(audits),
        "actual_dependency_empty": sum(not audit.actual_dependencies for audit in audits.values()),
        "evidence_id": evidence_id,
        "latency_seconds": run.latency_seconds,
    }


def _verify_unchanged_sources(
    candidates: list[Any], *, source_root: Path, target_root: Path, timeout: float
) -> dict[str, Any]:
    paths = sorted({item.file_path for item in candidates})
    results: list[dict[str, Any]] = []
    for relative in paths:
        source_path = source_root / relative
        try:
            completed = subprocess.run(
                ["lake", "env", "lean", "-E", "hasSorry", str(source_path.resolve())],
                cwd=target_root,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            status = "accepted" if completed.returncode == 0 else "rejected"
            diagnostic = "" if status == "accepted" else (completed.stdout + completed.stderr)[-4000:]
            exit_code: int | None = completed.returncode
        except subprocess.TimeoutExpired as error:
            status, diagnostic, exit_code = "timeout", str(error)[-4000:], None
        results.append(
            {"file_path": relative, "status": status, "exit_code": exit_code, "diagnostic": diagnostic}
        )
    failures = [item for item in results if item["status"] != "accepted"]
    if failures:
        raise RuntimeError(f"unchanged source verification failed: {failures[:2]}")
    return {"files": len(results), "accepted": len(results), "results": results}


def _synthetic_records(
    *,
    candidates: list[Any],
    requested_counts: dict[str, int],
    seed: str,
    output_dir: Path,
    target_root: Path,
    environment: dict[str, Any],
    batch_size: int,
    workers: int,
    forbidden_statement_ids: set[str],
    full_scale_reserve: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    pools = composition_pools(candidates)
    missing = {name: len(pools.get(name, [])) for name in requested_counts if len(pools.get(name, [])) < 4}
    if missing:
        raise RuntimeError(f"composition source pools are too small: {missing}")
    reserve_counts = {
        name: count + max(16, (3 * count + 3) // 4)
        for name, count in requested_counts.items()
    }
    plans = build_composition_plans(pools, reserve_counts, seed=seed)
    supplemental_reserve_counts: dict[str, int] = {}
    if full_scale_reserve:
        for name, count in requested_counts.items():
            numerator, denominator = FULL_SYNTHETIC_ACCEPTANCE_YIELD_FLOORS[name]
            target = (count * denominator + numerator - 1) // numerator
            supplemental_reserve_counts[name] = max(0, target - reserve_counts[name])
        plans.extend(
            build_composition_plans(
                pools,
                supplemental_reserve_counts,
                seed=f"{seed}\0supplemental-reserve-v1",
                name_prefix="dataset_v2_synthetic_reserve",
            )
        )
    generated_plan_count = len(plans)
    all_audits: dict[str, Any] = {}
    rejected_names: set[str] = set()
    composition_runs: list[dict[str, Any]] = []
    shortcut_runs: list[dict[str, Any]] = []
    lean_dir = output_dir / "lean"
    lean_dir.mkdir(parents=True, exist_ok=True)
    planned_sources = {
        (source.declaration_name, source.source_module): source
        for plan in plans
        for source in plan.source_lemmas + plan.retrieval_lemmas
    }
    missing_constants: set[str] = set()
    universe_arguments: dict[str, tuple[str, ...]] = {}
    for pnt_plus in (False, True):
        sources = [
            source
            for source in planned_sources.values()
            if source.source_module.startswith("PrimeNumberTheoremAnd") == pnt_plus
        ]
        if not sources:
            continue
        missing, grounding = find_missing_constants(
            sources,
            source_path=lean_dir / f"ConstantPresence-{'pnt' if pnt_plus else 'mathlib'}.lean",
            target_root=target_root,
        )
        missing_constants.update(lean_name_key(item) for item in missing)
        universe_arguments.update(
            (lean_name_key(name), arguments)
            for name, arguments in grounding.items()
        )

    def ground_unconstrained_universes(source: Any) -> Any:
        return replace(
            source,
            universe_arguments=universe_arguments.get(
                lean_name_key(source.declaration_name), ()
            ),
        )

    plans = [
        replace(
            plan,
            source_lemmas=tuple(
                ground_unconstrained_universes(source)
                for source in plan.source_lemmas
            ),
            retrieval_lemmas=tuple(
                ground_unconstrained_universes(source)
                for source in plan.retrieval_lemmas
                if lean_name_key(source.declaration_name) not in missing_constants
            ),
            retrieval_index=tuple(
                entry
                for entry in plan.retrieval_index
                if lean_name_key(entry[0]) not in missing_constants
            ),
        )
        for plan in plans
        if not any(
            lean_name_key(source.declaration_name) in missing_constants
            for source in plan.source_lemmas
        )
    ]
    batches: list[list[Any]] = []
    for pnt_plus in (False, True):
        group = [plan for plan in plans if (plan.domain_family == "pnt-plus") == pnt_plus]
        batches.extend(
            group[start : start + batch_size]
            for start in range(0, len(group), batch_size)
        )
    def verify_batch(indexed_batch: tuple[int, list[Any]]) -> tuple[int, Any, Any, dict[str, Any]]:
        batch_index, batch = indexed_batch
        composition_path = lean_dir / f"Composition-{batch_index:04d}.lean"
        composition_source = render_composition_source(batch)
        composition_path.write_text(composition_source, encoding="utf-8")
        batch_digest = hashlib.sha256(
            f"{COMPOSITION_BATCH_CACHE_VERSION}\0{composition_source}".encode()
        ).hexdigest()
        cache_dir = output_dir / ".composition-batch-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{batch_index:04d}-{batch_digest}.pkl"
        if cache_path.is_file():
            with cache_path.open("rb") as handle:
                run, shortcut_run, audit_summary = pickle.load(handle)
            shortcut_source, _ = render_shortcut_gate_source(
                batch, {audit.name: audit for audit in run.audits}
            )
            (lean_dir / f"ShortcutGate-{batch_index:04d}.lean").write_text(
                shortcut_source, encoding="utf-8"
            )
            return batch_index, run, shortcut_run, audit_summary
        run = run_composition_source(composition_path, target_root=target_root)
        if run.status != "accepted":
            raise RuntimeError(f"composition batch {batch_index} failed: {run.diagnostic}")
        audit_summary = validate_composition_audits(batch, run.audits)
        shortcut_source, line_map = render_shortcut_gate_source(
            batch, {audit.name: audit for audit in run.audits}
        )
        shortcut_path = lean_dir / f"ShortcutGate-{batch_index:04d}.lean"
        shortcut_path.write_text(shortcut_source, encoding="utf-8")
        shortcut_run = run_shortcut_gate_source(
            shortcut_path, target_root=target_root, line_to_name=line_map
        )
        if shortcut_run.status not in {"accepted", "rejected-shortcuts"}:
            raise RuntimeError(
                f"shortcut batch {batch_index} infrastructure failure: {shortcut_run.diagnostic}"
            )
        temporary_cache = cache_path.with_suffix(".tmp")
        with temporary_cache.open("wb") as handle:
            pickle.dump(
                (run, shortcut_run, audit_summary),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_cache.replace(cache_path)
        return batch_index, run, shortcut_run, audit_summary

    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as executor:
        batch_results = list(executor.map(verify_batch, enumerate(batches)))
    for batch_index, run, shortcut_run, audit_summary in batch_results:
        all_audits.update({audit.name: audit for audit in run.audits})
        composition_runs.append(
            {"batch": batch_index, "status": run.status, **audit_summary}
        )
        rejected_names.update(shortcut_run.rejected_names)
        shortcut_runs.append(
            {
                "batch": batch_index,
                "status": shortcut_run.status,
                "rejected": len(shortcut_run.rejected_names),
            }
        )
    selected: list[Any] = []
    selected_statement_ids: set[str] = set()
    selected_source_sets: set[tuple[str, ...]] = set()
    selected_derivation_families: set[str] = set()
    shortfalls: dict[str, tuple[int, int, int, int]] = {}
    for family, requested in requested_counts.items():
        eligible = [
            plan
            for plan in plans
            if plan.domain_family == family and plan.synthetic_name not in rejected_names
        ]
        family_selected: list[Any] = []
        for plan in eligible:
            identity = statement_id(
                f"theorem {plan.synthetic_name} : {all_audits[plan.synthetic_name].statement_type}"
            )
            source_set = tuple(sorted(item.statement_id for item in plan.source_lemmas))
            if (
                identity in forbidden_statement_ids
                or identity in selected_statement_ids
                or source_set in selected_source_sets
                or plan.derivation_family_fingerprint in selected_derivation_families
            ):
                continue
            family_selected.append(plan)
            selected_statement_ids.add(identity)
            selected_source_sets.add(source_set)
            selected_derivation_families.add(plan.derivation_family_fingerprint)
            if len(family_selected) == requested:
                break
        if len(family_selected) < requested:
            resolved = sum(plan.domain_family == family for plan in plans)
            shortfalls[family] = (
                len(family_selected),
                requested,
                len(eligible),
                resolved,
            )
        selected.extend(family_selected)
    if shortfalls:
        rendered = ", ".join(
            (
                f"{family}=accepted:{accepted}/requested:{requested}"
                f"/shortcut-eligible:{eligible}/presence-resolved:{resolved}"
            )
            for family, (accepted, requested, eligible, resolved) in shortfalls.items()
        )
        raise RuntimeError(
            "synthetic regeneration reserve exhausted after shortcut/dedup gates: "
            + rendered
        )
    shortcut_status = {
        plan.synthetic_name: (
            "assumption:no-closure",
            "rfl:no-closure",
            "simp:no-closure",
            "exact-indexed-single-theorem:no-closure",
            "simpa-using-indexed-single-theorem:no-closure",
            "retrieval:indexed-type-head-dependency-neighborhood",
        )
        for plan in selected
    }
    source_digest = hashlib.sha256(
        "".join(plan.synthetic_name + plan.derivation_family_fingerprint for plan in selected).encode()
    ).hexdigest()
    records = records_from_compositions(
        selected,
        all_audits,
        environment=environment,
        verification_evidence_id=f"composition-audit:{source_digest}",
        shortcut_status=shortcut_status,
    )
    assigned = assign_synthetic_roles(records, seed=str(environment["split_seed"]))
    summary = summarize_compositions(assigned)
    if any(
        summary[key] != len(assigned)
        for key in (
            "unique_statements",
            "unique_source_lemma_sets",
            "unique_derivation_families",
        )
    ):
        raise RuntimeError("synthetic uniqueness gate failed after record construction")
    return assigned, {
        "requested": sum(requested_counts.values()),
        "generated_with_reserve": generated_plan_count,
        "initial_reserve_counts": reserve_counts,
        "supplemental_reserve_counts": supplemental_reserve_counts,
        "resolved_after_presence_audit": len(plans),
        "shortcut_rejected": len(rejected_names),
        "unresolved_source_constants": len(missing_constants),
        "composition_runs": composition_runs,
        "shortcut_runs": shortcut_runs,
        "summary": summary,
        "pool_sizes": {name: len(pools.get(name, [])) for name in requested_counts},
        "source_module_coverage": {
            "unique_modules": len(
                {source.source_module for plan in selected for source in plan.source_lemmas}
            ),
            "module_uses": dict(
                sorted(
                    Counter(
                        source.source_module
                        for plan in selected
                        for source in plan.source_lemmas
                    ).items()
                )
            ),
            "package_uses": dict(
                sorted(
                    Counter(
                        "pnt-plus"
                        if source.source_module.startswith("PrimeNumberTheoremAnd")
                        else "mathlib"
                        for plan in selected
                        for source in plan.source_lemmas
                    ).items()
                )
            ),
        },
        "source_relation_coverage": dict(
            sorted(
                Counter(
                    relation
                    for plan in selected
                    for _, _, relation in plan.relation_edges
                ).items()
            )
        ),
        "shortcut_retrieval_candidates": sum(
            len(plan.retrieval_lemmas) for plan in selected
        ),
        "shortcut_retrieval_index_coverage": dict(
            sorted(
                Counter(
                    origin
                    for plan in selected
                    for _, origins in plan.retrieval_index
                    for origin in origins.split(",")
                ).items()
            )
        ),
    }


def _benchmark_outputs(
    *, mini_root: Path, config: DatasetV2Config, records: list[Any], output_dir: Path
) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=mini_root, text=True, capture_output=True, check=True
    ).stdout.strip()
    expected_revision = str(config.value["benchmark"]["revision"])
    if revision != expected_revision:
        raise RuntimeError("miniF2F revision mismatch")
    result: dict[str, Any] = {"revision": revision, "splits": {}}
    for split in ("valid", "test"):
        source_key = f"{split}_source"
        expected = int(config.value["benchmark"]["expected_primary_counts"][split])
        tasks = materialize_benchmark_source(
            (mini_root / config.value["benchmark"][source_key]).read_text(encoding="utf-8"),
            expected_primary_task_count=expected,
        )
        raw = [
            {
                "task_id": task.id,
                "declaration": task.declaration,
                "declaration_name": task.declaration_name,
                "preamble": task.preamble,
                "source_split": split,
            }
            for task in tasks
        ]
        retained, excluded = filter_clean_benchmark(raw, records)
        path = output_dir / f"minif2f-{split}-clean-v2.jsonl"
        _write_jsonl(path, list(retained))
        result["splits"][split] = {
            "source": len(raw),
            "retained": len(retained),
            "excluded": len(excluded),
            "exclusions": excluded,
            "file": path.name,
            "sha256": sha256_file(path),
        }
    return result


def _preflight_real_candidates(
    *,
    config: DatasetV2Config,
    mathlib_candidates: list[Any],
    pnt_candidates: list[Any],
    target_mathlib: Path,
    target_root: Path,
    workers: int,
    timeout: float,
) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    seed = str(config.preflight["seed"])
    term_reserve = select_candidates(
        mathlib_candidates,
        count=int(config.preflight["term_proof_sample"]) + 16,
        seed=seed + ":term",
        transformation_kind="term-to-exact",
    )
    selected_by = select_candidates(
        mathlib_candidates,
        count=int(config.preflight["by_proof_sample"]),
        seed=seed + ":by",
        transformation_kind="none",
    )
    selected_equations = select_candidates(
        mathlib_candidates,
        count=int(config.preflight["equation_clause_sample"]),
        seed=seed + ":equations",
        transformation_kind="equations-to-fun-exact",
    )
    pnt_reserve = select_candidates(
        pnt_candidates, count=len(pnt_candidates), seed=seed + ":pnt"
    )
    verified, file_verification = verify_transformed_candidates(
        term_reserve + selected_equations + pnt_reserve,
        source_roots={
            (
                str(config.environment["mathlib_repository"]),
                str(config.environment["mathlib_revision"]),
            ): target_mathlib,
            (
                str(config.environment["host_repository"]),
                str(config.environment["host_revision"]),
            ): target_root,
        },
        target_root=target_root,
        environment_id=str(config.environment["environment_id"]),
        evidence_id="dataset-v2-preflight-term-reconstruction-v1",
        workers=workers,
        timeout_seconds=timeout,
    )
    by_id = {
        (item.source_repository, item.file_path, item.declaration_name): item
        for item in verified
    }
    verified_term_reserve = [
        by_id[(item.source_repository, item.file_path, item.declaration_name)]
        for item in term_reserve
    ]
    verified_pnt_reserve = [
        by_id.get((item.source_repository, item.file_path, item.declaration_name), item)
        for item in pnt_reserve
    ]
    verified_equations = [
        by_id[(item.source_repository, item.file_path, item.declaration_name)]
        for item in selected_equations
    ]
    accepted_terms = [
        item for item in verified_term_reserve if item.verification_status == "accepted"
    ]
    accepted_pnt = [
        item for item in verified_pnt_reserve if item.verification_status == "accepted"
    ]
    accepted_equations = [
        item for item in verified_equations if item.verification_status == "accepted"
    ]
    term_count = int(config.preflight["term_proof_sample"])
    pnt_count = int(config.preflight["pnt_plus_sample"])
    equation_count = int(config.preflight["equation_clause_sample"])
    if (
        len(accepted_terms) < term_count
        or len(accepted_equations) < equation_count
        or len(accepted_pnt) < pnt_count
    ):
        raise RuntimeError(
            "preflight regeneration reserve exhausted: "
            f"terms={len(accepted_terms)}/{term_count}, "
            f"equations={len(accepted_equations)}/{equation_count}, "
            f"PNT={len(accepted_pnt)}/{pnt_count}"
        )
    selected_term = accepted_terms[:term_count]
    selected_pnt = accepted_pnt[:pnt_count]
    selected_equations = accepted_equations[:equation_count]
    unchanged = _verify_unchanged_sources(
        selected_by,
        source_root=target_mathlib,
        target_root=target_root,
        timeout=timeout,
    )
    pnt_unchanged = _verify_unchanged_sources(
        [item for item in selected_pnt if item.transformation_kind == "none"],
        source_root=target_root,
        target_root=target_root,
        timeout=timeout,
    )
    requested_counts = {
        "generic": int(config.preflight["synthetic_statements"])
        - int(config.preflight["prime_synthetic_statements"]),
        "prime-arithmetic-divisibility": 8,
        "arithmetic-functions": 7,
        "prime-counting-pnt": 6,
        "zeta-analytic-number-theory": 1,
        "riemann-core-bubble": 6,
        "pnt-plus": 4,
    }
    if sum(value for key, value in requested_counts.items() if key != "generic") != int(
        config.preflight["prime_synthetic_statements"]
    ):
        raise RuntimeError("preflight prime-family allocation does not sum to its contract")
    return (
        selected_term + selected_equations + selected_by + selected_pnt,
        file_verification,
        unchanged,
        pnt_unchanged,
        requested_counts,
    )


def main() -> int:
    args = _parser().parse_args()
    if (
        args.workers < 1
        or args.composition_workers < 1
        or args.composition_batch_size < 1
    ):
        raise SystemExit("worker and batch counts must be positive")
    if args.resume_preflight and args.resume_full:
        raise SystemExit("select only one resume mode")
    if args.resume_preflight and args.mode != "preflight":
        raise SystemExit("--resume-preflight requires --mode preflight")
    if args.resume_full and args.mode != "full":
        raise SystemExit("--resume-full requires --mode full")
    config = DatasetV2Config.load(args.config.resolve())
    target_root = args.target_root.resolve()
    trace_root = args.trace_root.resolve()
    mini_root = args.mini_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            ROOT / "artifacts/dataset-v2/preflight"
            if args.mode == "preflight"
            else ROOT / "data/lean-whole-proof-v2"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = config.validate_target_root(target_root)
    target_mathlib = target_root / ".lake/packages/mathlib"
    source_equivalence = _source_equivalence(
        trace_root=trace_root, target_mathlib=target_mathlib, config=config
    )
    metadata = load_riemann_metadata(ROOT / "data/riemann")
    extraction_cache = output_dir / ".extraction-cache.pkl"
    if args.resume_preflight:
        if not extraction_cache.is_file():
            raise RuntimeError("--resume-preflight requires an existing preflight cache")
        with extraction_cache.open("rb") as handle:
            mathlib_candidates, diagnostics, pnt_candidates, pnt_audit = pickle.load(handle)
    elif args.resume_full:
        if not extraction_cache.is_file():
            raise RuntimeError("--resume-full requires an existing extraction cache")
        with extraction_cache.open("rb") as handle:
            (
                cache_version,
                mathlib_candidates,
                diagnostics,
                pnt_candidates,
                pnt_audit,
            ) = pickle.load(handle)
        if cache_version != FULL_CACHE_VERSION:
            raise RuntimeError("full extraction cache version mismatch")
    else:
        mathlib_candidates, diagnostics = extract_traced_files(
            _iter_mathlib_traces(
                trace_root,
                list(config.preflight["trace_files"])
                if args.mode == "preflight"
                else None,
            ),
            source_root=target_mathlib,
            source_repository=str(config.environment["mathlib_repository"]),
            source_revision=str(config.environment["mathlib_revision"]),
            provenance="real-mathlib",
            topic_metadata=metadata,
        )
        mathlib_candidates = [annotate_candidate(item) for item in mathlib_candidates]
        pnt_candidates, pnt_audit = _load_pnt_candidates(
            target_root=target_root, output_dir=output_dir, config=config
        )
        if args.mode == "preflight":
            with extraction_cache.open("wb") as handle:
                pickle.dump(
                    (mathlib_candidates, diagnostics, pnt_candidates, pnt_audit),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        else:
            with extraction_cache.open("wb") as handle:
                pickle.dump(
                    (
                        FULL_CACHE_VERSION,
                        mathlib_candidates,
                        diagnostics,
                        pnt_candidates,
                        pnt_audit,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
    mathlib_candidates, repaired_mathlib_names = repair_candidate_declaration_names(
        mathlib_candidates, source_root=target_mathlib
    )
    mathlib_candidates, duplicate_mathlib_candidates = collapse_duplicate_candidates(
        mathlib_candidates
    )
    unique_exclusions, duplicate_exclusions = collapse_duplicate_exclusions(
        diagnostics.exclusions
    )
    diagnostics = replace(
        diagnostics,
        candidates=len(mathlib_candidates),
        transformation_counts=dict(
            sorted(Counter(item.transformation_kind for item in mathlib_candidates).items())
        ),
        duplicate_candidates=(
            getattr(diagnostics, "duplicate_candidates", 0)
            + duplicate_mathlib_candidates
        ),
        repaired_declaration_names=(
            getattr(diagnostics, "repaired_declaration_names", 0)
            + repaired_mathlib_names
        ),
        exclusions=tuple(unique_exclusions),
        exclusion_counts=dict(
            sorted(
                Counter(
                    item["reason"].split(":", 1)[0]
                    for item in unique_exclusions
                ).items()
            )
        ),
        duplicate_exclusions=(
            getattr(diagnostics, "duplicate_exclusions", 0)
            + duplicate_exclusions
        ),
    )
    pnt_candidates, duplicate_pnt_candidates = collapse_duplicate_candidates(pnt_candidates)
    pnt_audit = {
        **pnt_audit,
        "duplicate_candidates_collapsed": (
            int(pnt_audit.get("duplicate_candidates_collapsed", 0))
            + duplicate_pnt_candidates
        ),
    }
    print(
        (
            f"Dataset v2: extracted {len(mathlib_candidates)} unique Mathlib and "
            f"{len(pnt_candidates)} unique PNT+ candidates "
            f"({diagnostics.duplicate_candidates + duplicate_pnt_candidates} duplicate "
            f"trace occurrences collapsed; {diagnostics.repaired_declaration_names} "
            f"declaration names repaired; {diagnostics.duplicate_exclusions} duplicate "
            "exclusions collapsed)"
        ),
        file=sys.stderr,
        flush=True,
    )

    seed = str(config.preflight["seed"])
    if args.mode == "preflight":
        verification_cache = output_dir / ".verification-cache.pkl"
        if args.resume_preflight:
            if not verification_cache.is_file():
                raise RuntimeError("--resume-preflight requires a verification cache")
            with verification_cache.open("rb") as handle:
                (
                    real_candidates,
                    file_verification,
                    unchanged,
                    pnt_unchanged,
                    requested_counts,
                ) = pickle.load(handle)
        else:
            (
                real_candidates,
                file_verification,
                unchanged,
                pnt_unchanged,
                requested_counts,
            ) = _preflight_real_candidates(
                config=config,
                mathlib_candidates=mathlib_candidates,
                pnt_candidates=pnt_candidates,
                target_mathlib=target_mathlib,
                target_root=target_root,
                workers=args.workers,
                timeout=args.verification_timeout,
            )
            with verification_cache.open("wb") as handle:
                pickle.dump(
                    (
                        real_candidates,
                        file_verification,
                        unchanged,
                        pnt_unchanged,
                        requested_counts,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
    else:
        verification_cache = output_dir / ".verification-cache.pkl"
        all_candidates = mathlib_candidates + pnt_candidates
        if args.resume_full and verification_cache.is_file():
            with verification_cache.open("rb") as handle:
                cache_version, cached_candidates, cached_file_verification = pickle.load(
                    handle
                )
            if cache_version not in {
                FULL_VERIFICATION_CACHE_VERSION,
                *LEGACY_FULL_VERIFICATION_CACHE_VERSIONS,
            }:
                raise RuntimeError("full verification cache version mismatch")
            classified_candidates, missing_indexes = _transfer_cached_verification(
                all_candidates, cached_candidates
            )
            supplemental_files: list[Any] = []
            if missing_indexes:
                missing = [classified_candidates[index] for index in missing_indexes]
                verified_missing, supplemental_files = verify_transformed_candidates(
                    missing,
                    source_roots={
                        (str(config.environment["mathlib_repository"]), str(config.environment["mathlib_revision"])): target_mathlib,
                        (str(config.environment["host_repository"]), str(config.environment["host_revision"])): target_root,
                    },
                    target_root=target_root,
                    environment_id=str(config.environment["environment_id"]),
                    evidence_id="dataset-v2-full-term-reconstruction-v1",
                    workers=args.workers,
                    timeout_seconds=args.verification_timeout,
                    group_cache_dir=output_dir / ".verification-group-cache",
                )
                for index, verified in zip(missing_indexes, verified_missing, strict=True):
                    classified_candidates[index] = verified
            file_verification = _rebuild_file_verification(
                classified_candidates,
                cached_file_verification,
                supplemental_files,
            )
            if cache_version != FULL_VERIFICATION_CACHE_VERSION or missing_indexes:
                with verification_cache.open("wb") as handle:
                    pickle.dump(
                        (
                            FULL_VERIFICATION_CACHE_VERSION,
                            classified_candidates,
                            file_verification,
                        ),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
        else:
            classified_candidates, file_verification = verify_transformed_candidates(
                all_candidates,
                source_roots={
                    (str(config.environment["mathlib_repository"]), str(config.environment["mathlib_revision"])): target_mathlib,
                    (str(config.environment["host_repository"]), str(config.environment["host_revision"])): target_root,
                },
                target_root=target_root,
                environment_id=str(config.environment["environment_id"]),
                evidence_id="dataset-v2-full-term-reconstruction-v1",
                workers=args.workers,
                timeout_seconds=args.verification_timeout,
                group_cache_dir=output_dir / ".verification-group-cache",
            )
            with verification_cache.open("wb") as handle:
                pickle.dump(
                    (
                        FULL_VERIFICATION_CACHE_VERSION,
                        classified_candidates,
                        file_verification,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        classified_candidates, _ = collapse_duplicate_candidates(classified_candidates)
        real_candidates = [
            item
            for item in classified_candidates
            if item.verification_status == "accepted"
        ]
        unchanged = {"files": 0, "accepted": 0, "results": [], "evidence": "frozen target build"}
        pnt_unchanged = unchanged
        requested_counts = {
            "generic": int(config.value["synthetic"]["unique_statements"])
            - int(config.value["synthetic"]["prime_domain_statements"])
        }
        requested_counts.update(
            distribute_prime_counts(int(config.value["synthetic"]["prime_domain_statements"]))
        )

    real_candidates, _ = collapse_duplicate_candidates(real_candidates)
    environment = dict(config.environment)
    environment["split_seed"] = config.value["synthetic"]["split_seed"]
    synthetic_cache = output_dir / ".synthetic-v4-cache.pkl"
    if args.resume_full and synthetic_cache.is_file():
        with synthetic_cache.open("rb") as handle:
            cache_version, synthetic, synthetic_evidence = pickle.load(handle)
        if cache_version != FULL_SYNTHETIC_CACHE_VERSION:
            raise RuntimeError("full synthetic cache version mismatch")
    else:
        synthetic, synthetic_evidence = _synthetic_records(
            candidates=mathlib_candidates + pnt_candidates,
            requested_counts=requested_counts,
            seed=(seed if args.mode == "preflight" else str(config.value["synthetic"]["split_seed"])),
            output_dir=output_dir,
            target_root=target_root,
            environment=environment,
            batch_size=args.composition_batch_size,
            workers=args.composition_workers,
            forbidden_statement_ids={statement_id(item.declaration) for item in real_candidates},
            full_scale_reserve=args.mode == "full",
        )
        if args.mode == "full":
            with synthetic_cache.open("wb") as handle:
                pickle.dump(
                    (FULL_SYNTHETIC_CACHE_VERSION, synthetic, synthetic_evidence),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
    real_records = [candidate_to_record(item, config=config) for item in real_candidates]
    records = merge_statement_records([*real_records, *synthetic])
    role_validation = validate_role_isolation(records)
    records_path = output_dir / "records.jsonl.gz"
    records_sha = write_records(records_path, records)
    loaded = read_records(records_path)
    optimizer_ids = {
        record.statement_id
        for record, _ in iter_optimizer_examples(loaded, variant_seed="dataset-v2-loader-smoke-v1")
    }
    eval_ids = {item.statement_id for item in loaded if item.role != "training"}
    if optimizer_ids & eval_ids:
        raise RuntimeError("Dataset-v2 optimizer loader exposed evaluation records")
    benchmark = _benchmark_outputs(
        mini_root=mini_root, config=config, records=records, output_dir=output_dir
    )
    if args.mode == "full":
        disposition_candidates = classified_candidates
    else:
        verified_by_source = {
            (item.source_repository, item.source_revision, item.file_path, item.declaration_name): item
            for item in real_candidates
        }
        disposition_candidates = [
            verified_by_source.get(
                (
                    item.source_repository,
                    item.source_revision,
                    item.file_path,
                    item.declaration_name,
                ),
                item,
            )
            for item in mathlib_candidates + pnt_candidates
        ]
    source_dispositions = build_source_dispositions(
        disposition_candidates,
        diagnostics=diagnostics,
        config=config.value,
        topic_metadata=metadata,
    )
    source_dispositions_path = output_dir / "source-dispositions.jsonl.gz"
    source_dispositions_sha = _write_jsonl_gz(
        source_dispositions_path, source_dispositions
    )
    crosswalk_path: Path | None = None
    crosswalk: dict[str, Any] | None = None
    if args.mode == "full":
        historical_records = [
            value
            for split in ("train", "validation", "heldout")
            for value in read_jsonl(
                ROOT / f"data/mathlib-whole-proof-v1/{split}.jsonl.gz"
            )
        ]
        membership_inventories = {
            path.parent.name: read_jsonl(path)
            for path in sorted(
                (ROOT / "data/riemann/corpora").glob("*/membership.jsonl")
            )
        }
        crosswalk = historical_source_crosswalk(
            source_dispositions,
            historical_records=historical_records,
            membership_inventories=membership_inventories,
        )
        crosswalk_path = output_dir / "historical-crosswalk.json"
        _write_json(crosswalk_path, crosswalk)
    source_manifest = json.loads(
        (ROOT / config.value["pnt_plus"]["accepted_declarations_manifest"]).read_text(encoding="utf-8")
    )
    coverage = build_prime_coverage_manifest(
        records,
        config=config.value,
        pnt_source_manifest=source_manifest,
        source_dispositions=source_dispositions,
        atlas_entries=read_jsonl(ROOT / "data/riemann/atlas/entries.jsonl"),
    )
    coverage_path = output_dir / "prime-coverage.json"
    _write_json(coverage_path, coverage)
    verification_path = output_dir / "verification.json"
    verified_candidate_ids = {
        candidate_id
        for item in file_verification
        for candidate_id in item.candidate_ids
    }
    rejected_candidate_ids = {
        candidate_id
        for item in file_verification
        for candidate_id in item.rejected_candidate_ids
    }
    verification = {
        "mode": args.mode,
        "target": target,
        "source_equivalence": source_equivalence,
        "extraction": asdict(diagnostics),
        "term_reconstruction": {
            "files": [
                {
                    key: value
                    for key, value in asdict(item).items()
                    if key != "latency_seconds"
                }
                for item in file_verification
            ],
            "candidate_attempts": len(verified_candidate_ids),
            "accepted_attempts": len(verified_candidate_ids - rejected_candidate_ids),
            "rejected_attempts": len(rejected_candidate_ids),
            "retained_transformed_candidates": sum(
                item.transformation_kind != "none" for item in real_candidates
            ),
            "retained_transformations": dict(
                sorted(
                    Counter(
                        item.transformation_kind
                        for item in real_candidates
                        if item.transformation_kind != "none"
                    ).items()
                )
            ),
        },
        "ordinary_by_sample": unchanged,
        "pnt_sample": pnt_unchanged,
        "pnt_dependency_audit": {
            key: value for key, value in pnt_audit.items() if key != "latency_seconds"
        },
        "source_dispositions": {
            "entries": len(source_dispositions),
            "dispositions": dict(
                sorted(Counter(item["disposition"] for item in source_dispositions).items())
            ),
            "reasons": dict(
                sorted(Counter(item["reason"] for item in source_dispositions).items())
            ),
        },
        "synthetic": synthetic_evidence,
        "role_isolation": role_validation,
        "loader": {
            "loaded_records": len(loaded),
            "optimizer_statements": len(optimizer_ids),
            "evaluation_statements_exposed": 0,
        },
        "benchmark": benchmark,
    }
    if crosswalk is not None:
        verification["historical_crosswalk"] = crosswalk
    if args.mode == "full":
        probe = select_train_probe(
            records,
            per_stratum=int(config.value["train_probe"]["per_stratum"]),
            seed=str(config.value["train_probe"]["seed"]),
        )
        probe_path = output_dir / "train-probe.json"
        _write_json(
            probe_path,
            {"id": config.value["train_probe"]["id"], "strata": probe},
        )
        verification["train_probe"] = {
            "file": probe_path.name,
            "sha256": sha256_file(probe_path),
            "counts": {key: len(value) for key, value in probe.items()},
        }
    _write_json(verification_path, verification)
    files = {
        records_path.name: {"sha256": records_sha, "bytes": records_path.stat().st_size},
        source_dispositions_path.name: {
            "sha256": source_dispositions_sha,
            "bytes": source_dispositions_path.stat().st_size,
        },
        coverage_path.name: {"sha256": sha256_file(coverage_path), "bytes": coverage_path.stat().st_size},
        verification_path.name: {"sha256": sha256_file(verification_path), "bytes": verification_path.stat().st_size},
    }
    if crosswalk_path is not None:
        files[crosswalk_path.name] = {
            "sha256": sha256_file(crosswalk_path),
            "bytes": crosswalk_path.stat().st_size,
        }
    for split in ("valid", "test"):
        path = output_dir / f"minif2f-{split}-clean-v2.jsonl"
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = dataset_manifest(
        records,
        config=config.value,
        files=files,
        validation={
            "role_isolation": role_validation,
            "loader_evaluation_records_exposed": 0,
            "prime_coverage_eligible_omissions": coverage["summary"]["verified_legal_target_compatible_omissions"],
            "synthetic_shortcut_rejections": synthetic_evidence["shortcut_rejected"],
        },
    )
    _write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output_dir": str(output_dir),
                "records": len(records),
                "roles": Counter(item.role for item in records),
                "prime_coverage_entries": coverage["summary"]["entries"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    del records, loaded
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
