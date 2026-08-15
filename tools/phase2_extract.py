from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace pinned mathlib with isolated LeanDojo-v2 and build Phase 2"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/phase2-mathlib.json"
    )
    parser.add_argument("--mini-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/phase2/mathlib-whole-proof-v1",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "artifacts/phase2/leandojo-cache"
    )
    parser.add_argument(
        "--temp-dir", type=Path, default=ROOT / "artifacts/phase2/leandojo-tmp"
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=ROOT / "artifacts/phase2/leandojo-trace/mathlib4",
    )
    parser.add_argument(
        "--tokenizer-cache",
        type=Path,
        default=ROOT / "artifacts/phase2/tokenizer-cache",
    )
    parser.add_argument("--trace-workers", type=int, default=8)
    parser.add_argument("--verification-workers", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--loader-smoke", action="store_true")
    parser.add_argument("--evidence-dir", type=Path)
    return parser


def _assert_pinned_leandojo(expected_revision: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution("lean-dojo-v2")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("LeanDojo-v2 installation has no direct Git provenance")
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info", {})
    observed_revision = vcs_info.get("commit_id") or vcs_info.get("requested_revision")
    if observed_revision != expected_revision:
        raise RuntimeError(
            f"LeanDojo-v2 revision mismatch: expected {expected_revision}, got {observed_revision}"
        )
    return {
        "package_version": distribution.version,
        "source_url": str(direct_url.get("url")),
        "revision": str(observed_revision),
    }


def _validate_trace_files(
    trace_root: Path,
    expected_revision: str,
    selected_files: list[str] | None = None,
) -> int:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=trace_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != expected_revision:
        raise RuntimeError("trace root is not the configured mathlib revision")
    source_paths = (
        [trace_root / relative_path for relative_path in sorted(selected_files)]
        if selected_files is not None
        else sorted(trace_root.glob("Mathlib/**/*.lean"))
    )
    missing: list[str] = []
    for source_path in source_paths:
        relative_path = source_path.relative_to(trace_root)
        trace_base = trace_root / ".lake/build/ir" / relative_path
        if not trace_base.with_suffix(".ast.json").is_file():
            missing.append(str(relative_path.with_suffix(".ast.json")))
        if not trace_base.with_suffix(".dep_paths").is_file():
            missing.append(str(relative_path.with_suffix(".dep_paths")))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"LeanDojo trace is incomplete ({len(missing)} missing outputs; first: {preview})"
        )
    if not source_paths:
        raise RuntimeError("trace root contains no Mathlib source files")
    return len(source_paths)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_trace_provenance(
    trace_root: Path,
    *,
    source_revision: str,
    extractor_revision: str,
    installed_extractor_path: Path,
    traced_now: bool,
) -> dict[str, object]:
    marker_path = trace_root / ".qwen-lean-phase2-trace.json"
    extractor_sha256 = _sha256_file(installed_extractor_path)
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("source_revision") != source_revision
            or marker.get("extractor_revision") != extractor_revision
            or marker.get("extract_data_sha256") != extractor_sha256
        ):
            raise RuntimeError("trace provenance marker differs from the pinned inputs")
        return marker
    retained_extractor = trace_root / "ExtractData.lean"
    if not traced_now and (
        not retained_extractor.is_file()
        or _sha256_file(retained_extractor) != extractor_sha256
    ):
        raise RuntimeError(
            "existing trace has no verifiable pinned LeanDojo extractor provenance"
        )
    marker = {
        "schema_version": "phase2-trace-provenance-v1",
        "source_revision": source_revision,
        "extractor_revision": extractor_revision,
        "extract_data_sha256": extractor_sha256,
        "build_dependencies": False,
    }
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return marker


def main() -> int:
    from qwen_lean.phase2_corpus import load_phase2_dataset
    from qwen_lean.phase2_extraction import (
        Phase2Config,
        build_phase2_corpus,
        write_compact_evidence,
    )
    from qwen_lean.phase2_verification import verify_phase2_sample

    args = _parser().parse_args()
    if args.trace_workers < 1:
        raise SystemExit("--trace-workers must be positive")
    config = Phase2Config.load(args.config)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    args.tokenizer_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GITHUB_ACCESS_TOKEN", "")
    os.environ["CACHE_DIR"] = str(args.cache_dir.resolve())
    os.environ["TMP_DIR"] = str(args.temp_dir.resolve())
    os.environ["NUM_PROCS"] = str(args.trace_workers)
    os.environ["DISABLE_REMOTE_CACHE"] = "1"
    tool = _assert_pinned_leandojo(str(config.extractor["revision"]))

    from lean_dojo_v2.lean_dojo import LeanGitRepo, TracedFile
    from lean_dojo_v2.lean_dojo.data_extraction.trace import (
        LEAN4_DATA_EXTRACTOR_PATH,
        _trace,
    )
    from lean_dojo_v2.utils.filesystem import working_directory

    repo = LeanGitRepo(str(config.source["repository"]), str(config.source["revision"]))
    trace_root = args.trace_root.resolve()
    if trace_root.name != repo.name:
        raise RuntimeError(
            f"--trace-root must end in the repository name {repo.name!r}"
        )
    selected_trace_files = list(config.value["pilot_files"]) if args.pilot else None
    traced_now = False
    try:
        traced_file_count = _validate_trace_files(
            trace_root,
            str(config.source["revision"]),
            selected_files=selected_trace_files,
        )
    except (OSError, RuntimeError):
        if trace_root.exists():
            current_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=trace_root,
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
            if current_revision != str(config.source["revision"]):
                raise RuntimeError(
                    "existing trace root has a different revision; refusing to overwrite it"
                )
        with working_directory(trace_root.parent):
            _trace(repo, build_deps=bool(config.extractor["build_dependencies"]))
        traced_now = True
        traced_file_count = _validate_trace_files(
            trace_root,
            str(config.source["revision"]),
            selected_files=selected_trace_files,
        )
    trace_provenance = _ensure_trace_provenance(
        trace_root,
        source_revision=str(config.source["revision"]),
        extractor_revision=str(config.extractor["revision"]),
        installed_extractor_path=Path(LEAN4_DATA_EXTRACTOR_PATH),
        traced_now=traced_now,
    )

    class StreamedTracedRepo:
        def __init__(self) -> None:
            self.repo = repo
            self.root_dir = trace_root
            self.dependencies: dict[str, object] = {}
            self.traced_file_count = traced_file_count
            self.trace_read_errors: list[dict[str, str]] = []

        def iter_traced_files(self):
            source_paths = (
                [trace_root / path for path in sorted(selected_trace_files)]
                if selected_trace_files is not None
                else sorted(trace_root.glob("Mathlib/**/*.lean"))
            )
            for source_path in source_paths:
                relative_path = source_path.relative_to(trace_root)
                ast_path = (
                    trace_root
                    / ".lake/build/ir"
                    / relative_path.with_suffix(".ast.json")
                )
                try:
                    traced_file = TracedFile.from_traced_file(
                        trace_root, ast_path, self.repo
                    )
                except Exception as error:  # noqa: BLE001 - audit every file loss.
                    detail = {
                        "file_path": str(relative_path),
                        "error": f"{type(error).__name__}: {error!r}",
                    }
                    self.trace_read_errors.append(detail)
                    print(
                        "Phase 2 extraction: LeanDojo could not parse "
                        f"{relative_path}: {detail['error']}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                traced_file.traced_repo = self
                yield traced_file

    traced_repo = StreamedTracedRepo()
    records, manifest = build_phase2_corpus(
        traced_repo,
        config,
        mini_root=args.mini_root.resolve(),
        output_dir=args.output_dir.resolve(),
        tokenizer_cache=args.tokenizer_cache.resolve(),
        pilot=args.pilot,
    )
    manifest["extractor"]["installed_provenance"] = tool
    manifest["extractor"]["reader"] = (
        "streamed LeanDojo-v2 TracedFile/TracedTheorem objects"
    )
    manifest["extractor"]["trace_provenance"] = trace_provenance
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record_count = len(records)
    del records
    gc.collect()
    verification_path: Path | None = None
    if args.verify:
        verification_path = args.output_dir / "verification.json"
        verify_phase2_sample(
            config,
            args.output_dir,
            trace_root,
            verification_path,
            sample_counts=(
                {"train": 1, "validation": 0, "heldout": 0} if args.pilot else None
            ),
            negative_substitutions=1 if args.pilot else None,
            workers=args.verification_workers,
        )
    loaded_counts = None
    if args.loader_smoke:
        dataset = load_phase2_dataset(args.output_dir)
        loaded_counts = {name: len(dataset[name]) for name in dataset}
    if args.evidence_dir is not None:
        if args.pilot:
            raise SystemExit("committed evidence is only generated for the full corpus")
        write_compact_evidence(
            args.output_dir,
            args.evidence_dir,
            verification_path=verification_path,
        )
    print(
        json.dumps(
            {
                "mode": manifest["mode"],
                "records": record_count,
                "splits": {
                    name: value["records"] for name, value in manifest["splits"].items()
                },
                "verification": None
                if verification_path is None
                else str(verification_path),
                "loader_counts": loaded_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
