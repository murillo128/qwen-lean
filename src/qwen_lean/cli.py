from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import (
    reverify_phase1_artifacts,
    run_phase1_baseline,
    validate_minif2f_environment,
    write_environment_validation,
)
from .evaluator import load_fixture_set, run_fixture_evaluation
from .generation import run_model_smoke
from .minif2f import Phase1Config


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    root = _project_root()
    parser = argparse.ArgumentParser(prog="qwen-lean")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture", help="evaluate fixed candidates")
    fixture.add_argument("--fixtures", type=Path, default=root / "fixtures/phase0.json")
    fixture.add_argument("--output-dir", type=Path, default=root / "artifacts/fixture")
    fixture.add_argument("--project-root", type=Path, default=root)
    fixture.add_argument("--timeout", type=float, default=30.0)

    smoke = subparsers.add_parser("model-smoke", help="run one real Qwen generation")
    smoke.add_argument("--fixtures", type=Path, default=root / "fixtures/phase0.json")
    smoke.add_argument("--task-id", default="core-identity")
    smoke.add_argument("--output-dir", type=Path, default=root / "artifacts/model-smoke")
    smoke.add_argument("--project-root", type=Path, default=root)
    smoke.add_argument("--timeout", type=float, default=30.0)
    smoke.add_argument("--max-new-tokens", type=int, default=128)

    minif2f_validate = subparsers.add_parser(
        "minif2f-validate", help="validate the pinned miniF2F environment"
    )
    minif2f_validate.add_argument("--benchmark-root", type=Path, required=True)
    minif2f_validate.add_argument(
        "--config", type=Path, default=root / "config/phase1-minif2f.json"
    )
    minif2f_validate.add_argument("--timeout", type=float)
    minif2f_validate.add_argument("--output", type=Path)

    baseline = subparsers.add_parser(
        "phase1-baseline", help="run the local-vLLM miniF2F baseline"
    )
    baseline.add_argument("--benchmark-root", type=Path, required=True)
    baseline.add_argument(
        "--config", type=Path, default=root / "config/phase1-minif2f.json"
    )
    baseline.add_argument(
        "--workload",
        default="minif2f-valid-dev16-v1",
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--timeout", type=float)
    baseline.add_argument("--verification-workers", type=int, default=8)

    reverify = subparsers.add_parser(
        "phase1-reverify", help="reverify stored Phase 1 candidate continuations"
    )
    reverify.add_argument("--benchmark-root", type=Path, required=True)
    reverify.add_argument(
        "--config", type=Path, default=root / "config/phase1-minif2f.json"
    )
    reverify.add_argument("--input-dir", type=Path, required=True)
    reverify.add_argument("--output-dir", type=Path, required=True)
    reverify.add_argument("--timeout", type=float)
    reverify.add_argument("--verification-workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fixture":
        _, results, mismatches = run_fixture_evaluation(
            args.fixtures,
            args.output_dir,
            args.project_root,
            timeout_seconds=args.timeout,
        )
        print(json.dumps([result.to_dict() for result in results], indent=2))
        if mismatches:
            for mismatch in mismatches:
                print(mismatch)
            return 1
        return 0

    if args.command == "minif2f-validate":
        config = Phase1Config.load(args.config)
        timeout = (
            float(config.value["verifier"]["timeout_seconds"])
            if args.timeout is None
            else args.timeout
        )
        evidence = validate_minif2f_environment(
            config,
            args.benchmark_root,
            timeout_seconds=timeout,
        )
        if args.output is not None:
            write_environment_validation(args.output, evidence)
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "phase1-baseline":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        config = Phase1Config.load(args.config)
        timeout = (
            float(config.value["verifier"]["timeout_seconds"])
            if args.timeout is None
            else args.timeout
        )
        _, _, summary = run_phase1_baseline(
            config,
            args.benchmark_root,
            args.workload,
            args.output_dir,
            timeout_seconds=timeout,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "phase1-reverify":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        config = Phase1Config.load(args.config)
        timeout = (
            float(config.value["verifier"]["timeout_seconds"])
            if args.timeout is None
            else args.timeout
        )
        _, _, summary = reverify_phase1_artifacts(
            config,
            args.benchmark_root,
            args.input_dir,
            args.output_dir,
            timeout_seconds=timeout,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    fixture_id, tasks, _ = load_fixture_set(args.fixtures)
    try:
        task = next(task for task in tasks if task.id == args.task_id)
    except StopIteration:
        print(f"unknown task id: {args.task_id}")
        return 2
    _, result = run_model_smoke(
        task,
        args.output_dir,
        args.project_root,
        task_source=fixture_id,
        timeout_seconds=args.timeout,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.category in {"generation_error", "verifier_error"} else 0
