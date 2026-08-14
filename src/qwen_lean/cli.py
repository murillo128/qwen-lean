from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import load_fixture_set, run_fixture_evaluation
from .generation import run_model_smoke


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

    _, tasks, _ = load_fixture_set(args.fixtures)
    try:
        task = next(task for task in tasks if task.id == args.task_id)
    except StopIteration:
        print(f"unknown task id: {args.task_id}")
        return 2
    _, result = run_model_smoke(
        task,
        args.output_dir,
        args.project_root,
        timeout_seconds=args.timeout,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.category in {"generation_error", "verifier_error"} else 0
