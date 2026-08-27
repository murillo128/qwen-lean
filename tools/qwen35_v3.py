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
    compact_stage0_evidence,
    measure_checkpoint_anchor_drift,
    run_bounded_configuration_training,
    run_no_update_near_max_preflight,
)
from qwen_lean.generalist_v3_evaluation import (
    compact_base_canary_evidence,
    compact_checkpoint_canary_evidence,
    finalize_existing_base_canary,
    run_base_validation_canary,
)
from qwen_lean.generalist_v3_parity import (
    build_and_run_hf_parity_sentinel,
    compact_lora_parity_evidence,
    run_vllm_parity_sentinel,
)
from qwen_lean.generalist_v3_reporting import compact_bounded_trajectory_evidence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/qwen35-4b-generalist-v3.json"
DEFAULT_PACKAGE = ROOT / "artifacts/dataset-v3/lean-proof-continuation-v3"
DEFAULT_ARTIFACTS = ROOT / "artifacts/qwen-lean-generalist-v3"
DEFAULT_VERIFIER_ROOT = ROOT / "artifacts/riemann/sources/PrimeNumberTheoremAnd"
DEFAULT_EVIDENCE = ROOT / "evidence/qwen-lean-generalist-v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute qwen-lean generalist v3")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--verifier-root", type=Path, default=DEFAULT_VERIFIER_ROOT)
    parser.add_argument("--configuration", choices=("C0", "C1", "C2", "C3"))
    parser.add_argument("--optimizer-step", type=int)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bind")
    subparsers.add_parser("census")
    subparsers.add_parser("freeze-execution-view")
    subparsers.add_parser("freeze-stage0")
    subparsers.add_parser("preflight")
    subparsers.add_parser("cache-base-logits")
    subparsers.add_parser("compact-stage0")
    subparsers.add_parser("base-canary")
    subparsers.add_parser("compact-base-canary")
    subparsers.add_parser("finalize-base-canary")
    subparsers.add_parser("parity-hf")
    subparsers.add_parser("parity-vllm")
    subparsers.add_parser("compact-parity")
    subparsers.add_parser("train-bounded")
    subparsers.add_parser("anchor-drift")
    subparsers.add_parser("checkpoint-canary")
    subparsers.add_parser("compact-checkpoint-canary")
    subparsers.add_parser("compact-trajectory")
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
        clean_gpu_baseline = _read_json(stage0 / "clean-gpu-baseline.json")
        preflight = _read_json(stage0 / "near-max-preflight.json")
        if (
            preflight.get("schema_version") != "generalist-v3-near-max-preflight-v4"
            or preflight.get("status") != "passed"
            or clean_gpu_baseline.get("clean_gpu_gate_passed") is not True
            or preflight.get("clean_gpu_baseline_sha256")
            != sha256_file(stage0 / "clean-gpu-baseline.json")
            or preflight.get("execution_view_identity_sha256")
            != execution_view.get("execution_view_sha256")
            or preflight.get("forward_backward", {}).get("optimizer_update_run") is not False
        ):
            raise RuntimeError("Stage 0 freeze requires the passed clean 16k preflight")
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
            "clean_gpu_baseline_sha256": sha256_file(
                stage0 / "clean-gpu-baseline.json"
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
            stage0 / "clean-gpu-baseline.json",
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
    if args.command == "compact-stage0":
        value = compact_stage0_evidence(
            config,
            artifact_root,
            DEFAULT_EVIDENCE / "stage0-16k.json",
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "base-canary":
        freeze = _read_json(stage0 / "freeze.json")
        if (
            freeze.get("validation_canary_sha256")
            != sha256_file(stage0 / "validation-canary.json")
            or freeze.get("optimizer_updates") != 0
        ):
            raise RuntimeError("Base canary requires the frozen Stage 0 manifest")
        value = run_base_validation_canary(
            config,
            stage0 / "validation-canary.json",
            binding.package_root / "manifest.json",
            args.verifier_root,
            artifact_root / "base-validation-canary",
            model_snapshot=args.model_snapshot,
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "compact-base-canary":
        value = compact_base_canary_evidence(
            config,
            stage0 / "validation-canary.json",
            artifact_root / "base-validation-canary",
            DEFAULT_EVIDENCE / "base-validation-canary.json",
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "finalize-base-canary":
        value = finalize_existing_base_canary(
            config,
            stage0 / "validation-canary.json",
            binding.package_root / "manifest.json",
            args.verifier_root,
            artifact_root / "base-validation-canary",
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "parity-hf":
        value = build_and_run_hf_parity_sentinel(
            config,
            binding,
            stage0 / "anchor-manifest.json",
            artifact_root / "parity/sentinel-adapter",
            artifact_root / "parity/hf-runtime.json",
            model_snapshot=args.model_snapshot,
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "parity-vllm":
        value = run_vllm_parity_sentinel(
            config,
            artifact_root / "parity/hf-runtime.json",
            artifact_root / "parity/sentinel-adapter",
            artifact_root / "parity/vllm-runtime.json",
            model_snapshot=args.model_snapshot,
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "compact-parity":
        value = compact_lora_parity_evidence(
            config,
            artifact_root / "parity/hf-runtime.json",
            artifact_root / "parity/vllm-runtime.json",
            DEFAULT_EVIDENCE / "lora-parity.json",
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "train-bounded":
        if args.configuration is None:
            raise ValueError("train-bounded requires --configuration C0/C1/C2/C3")
        value = run_bounded_configuration_training(
            config,
            binding,
            stage0 / "freeze.json",
            DEFAULT_EVIDENCE / "base-validation-canary.json",
            DEFAULT_EVIDENCE / "lora-parity.json",
            artifact_root / "training-stream.jsonl.gz",
            artifact_root / "training-stream-manifest.json",
            stage0 / "anchor-manifest.json",
            stage0 / "base-reference-logits.safetensors",
            stage0 / "base-reference-logits.json",
            artifact_root / "training" / args.configuration,
            configuration_id=args.configuration,
            model_snapshot=args.model_snapshot,
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command in {
        "anchor-drift",
        "checkpoint-canary",
        "compact-checkpoint-canary",
    }:
        if args.configuration is None or args.optimizer_step not in {100, 250, 500}:
            raise ValueError(
                f"{args.command} requires --configuration and "
                "--optimizer-step 100/250/500"
            )
        configuration = args.configuration
        step = int(args.optimizer_step)
        checkpoint = artifact_root / "training" / configuration / f"checkpoint-{step}"
        validation_run = artifact_root / "validation" / configuration / str(step)
        if args.command == "anchor-drift":
            value = measure_checkpoint_anchor_drift(
                config,
                binding,
                stage0 / "anchor-manifest.json",
                stage0 / "base-reference-logits.safetensors",
                checkpoint,
                validation_run / "anchor-drift.json",
                configuration_id=configuration,
                optimizer_step=step,
                model_snapshot=args.model_snapshot,
            )
        elif args.command == "checkpoint-canary":
            value = run_base_validation_canary(
                config,
                stage0 / "validation-canary.json",
                binding.package_root / "manifest.json",
                args.verifier_root,
                validation_run / "canary",
                model_snapshot=args.model_snapshot,
                adapter_dir=checkpoint,
                checkpoint_id=f"{configuration}-{step}",
                parity_evidence_path=DEFAULT_EVIDENCE / "lora-parity.json",
            )
        else:
            value = compact_checkpoint_canary_evidence(
                config,
                stage0 / "validation-canary.json",
                validation_run / "canary",
                DEFAULT_EVIDENCE / "base-validation-canary.json",
                validation_run / "anchor-drift.json",
                DEFAULT_EVIDENCE
                / "validation"
                / configuration
                / f"step-{step}.json",
                configuration_id=configuration,
                optimizer_step=step,
            )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "compact-trajectory":
        value = compact_bounded_trajectory_evidence(
            config,
            DEFAULT_EVIDENCE / "base-validation-canary.json",
            DEFAULT_EVIDENCE / "validation",
            artifact_root / "training",
            DEFAULT_EVIDENCE / "bounded-trajectory.json",
            DEFAULT_EVIDENCE / "bounded-validation-trajectory.svg",
            DEFAULT_EVIDENCE / "bounded-training-trajectory.svg",
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
