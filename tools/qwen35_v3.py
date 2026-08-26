from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_lean.dataset_v2 import sha256_file
from qwen_lean.generalist_v3 import (
    GeneralistV3Config,
    _read_json,
    _write_json,
    freeze_anchor_manifest,
    freeze_canary_manifest,
    freeze_structural_sampling_manifest,
    freeze_training_execution_view,
    tokenizer_length_census,
    validate_dataset_binding,
    write_training_stream,
)
from qwen_lean.generalist_v3_training import (
    cache_base_reference_logits,
    run_no_update_near_max_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/qwen35-4b-generalist-v3.json"
DEFAULT_PACKAGE = ROOT / "artifacts/dataset-v3/lean-proof-continuation-v3"
DEFAULT_ARTIFACTS = ROOT / "artifacts/qwen-lean-generalist-v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute qwen-lean generalist v3")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--model-snapshot", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bind")
    subparsers.add_parser("census")
    subparsers.add_parser("freeze-execution-view")
    subparsers.add_parser("freeze-stage0")
    subparsers.add_parser("preflight")
    subparsers.add_parser("cache-base-logits")
    return parser


def _tokenizer(config: GeneralistV3Config, model_snapshot: Path | None):
    from transformers import AutoTokenizer

    source = str(model_snapshot) if model_snapshot is not None else config.model["model_id"]
    kwargs = (
        {"local_files_only": True}
        if model_snapshot is not None
        else {
            "revision": config.model["model_revision"],
            "local_files_only": True,
        }
    )
    return AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)


def main() -> int:
    args = _parser().parse_args()
    config = GeneralistV3Config.load(args.config)
    binding = validate_dataset_binding(config, ROOT, args.package_root)
    artifact_root: Path = args.artifact_root
    stage0 = artifact_root / "stage0"
    if args.command == "bind":
        print(json.dumps(binding.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "census":
        tokenizer = _tokenizer(config, args.model_snapshot)
        value = tokenizer_length_census(config, binding, tokenizer)
        _write_json(stage0 / "tokenizer-census.json", value)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-execution-view":
        tokenizer = _tokenizer(config, args.model_snapshot)
        value = freeze_training_execution_view(
            config,
            binding,
            tokenizer,
            stage0 / "tokenizer-census.json",
            stage0 / "training-execution-view.json",
            progress_every_examples=5000,
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-stage0":
        tokenizer = _tokenizer(config, args.model_snapshot)
        execution_view = _read_json(stage0 / "training-execution-view.json")
        preflight = _read_json(stage0 / "near-max-preflight.json")
        if (
            preflight.get("schema_version") != "generalist-v3-near-max-preflight-v3"
            or preflight.get("status") != "passed"
            or preflight.get("execution_view_identity_sha256")
            != execution_view.get("execution_view_sha256")
            or preflight.get("forward_backward", {}).get("optimizer_update_run") is not False
        ):
            raise RuntimeError("Stage 0 freeze requires the passed 32k no-update preflight")
        structural = freeze_structural_sampling_manifest(
            config,
            binding,
            execution_view,
            stage0 / "structural-sampling.json",
        )
        anchor = freeze_anchor_manifest(
            config,
            binding,
            tokenizer,
            execution_view,
            structural,
            stage0 / "anchor-manifest.json",
        )
        canary = freeze_canary_manifest(
            config, binding, stage0 / "validation-canary.json", role="validation"
        )
        stream = write_training_stream(
            config,
            binding,
            execution_view,
            structural,
            artifact_root / "training-stream.jsonl.gz",
            artifact_root / "training-stream-manifest.json",
        )
        value = {
            "schema_version": "generalist-v3-stage0-freeze-v1",
            "config_sha256": sha256_file(args.config),
            "dataset_binding": binding.to_dict(),
            "training_execution_view_sha256": sha256_file(
                stage0 / "training-execution-view.json"
            ),
            "near_max_preflight_sha256": sha256_file(
                stage0 / "near-max-preflight.json"
            ),
            "structural_sampling_sha256": sha256_file(
                stage0 / "structural-sampling.json"
            ),
            "anchor_manifest_sha256": sha256_file(stage0 / "anchor-manifest.json"),
            "validation_canary_sha256": sha256_file(stage0 / "validation-canary.json"),
            "training_stream_manifest_sha256": sha256_file(
                artifact_root / "training-stream-manifest.json"
            ),
            "anchor_count": anchor["anchor_count"],
            "validation_interface_tasks": canary["interface_task_count"],
            "training_stream_microbatches": stream["microbatches"],
            "test_metadata_access_disclosed": True,
            "semantic_test_accessed": False,
            "optimizer_updates": 0,
        }
        _write_json(stage0 / "freeze.json", value)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        value = run_no_update_near_max_preflight(
            config,
            binding,
            stage0 / "training-execution-view.json",
            stage0 / "near-max-preflight.json",
            model_snapshot=args.model_snapshot,
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "cache-base-logits":
        value = cache_base_reference_logits(
            config,
            binding,
            stage0 / "anchor-manifest.json",
            stage0 / "base-reference-logits.safetensors",
            stage0 / "base-reference-logits.json",
            model_snapshot=args.model_snapshot,
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
