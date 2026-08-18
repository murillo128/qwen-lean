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
from .gpt53_assessment import (
    GPT53Config,
    run_assessment,
    run_preflight,
)
from .gpt53_assessment import (
    write_compact_evidence as write_gpt53_evidence,
)
from .minif2f import Phase1Config
from .phase2_corpus import load_phase2_dataset
from .phase2_extraction import Phase2Config, write_compact_evidence
from .phase2_verification import verify_phase2_sample
from .phase3 import (
    Phase3Config,
    load_phase2_train_records,
    load_pinned_tokenizer,
    select_overfit_workload,
    write_phase3_workload,
)
from .phase3_evidence import write_phase3_evidence
from .phase3_inference import (
    run_adapter_minif2f_smoke,
    run_vllm_memorization,
)
from .phase3_training import (
    run_adapter_reload_check,
    run_overfit_training,
    run_training_preflight,
)
from .phase3_verification import run_phase3_semantic_verification
from .phase4 import (
    Phase4Config,
    materialize_phase4_workloads,
    write_phase4_workloads,
)
from .phase4_evidence import write_phase4_evidence
from .phase4_inference import (
    compare_phase4_heldout_runs,
    run_phase4_heldout,
    run_phase4_minif2f,
)
from .phase4_training import (
    run_phase4_adapter_reload,
    run_phase4_preflight,
    run_phase4_training,
)
from .phase5 import (
    Phase5Config,
    materialize_phase5_workloads,
    write_phase5_workloads,
)
from .phase5_evidence import write_phase5_evidence
from .phase5_inference import (
    compare_phase5_heldout_runs,
    run_phase5_heldout,
    run_phase5_minif2f,
)
from .phase5_training import (
    run_phase5_adapter_reload,
    run_phase5_preflight,
    run_phase5_training,
)
from .phase6 import (
    Phase6Config,
    freeze_reference_candidate,
    materialize_phase6_train_workload,
    write_phase6_train_workload,
)
from .phase6_evidence import (
    write_phase6_checkpoint_a_evidence,
    write_phase6_final_evidence,
)
from .phase6_inference import run_phase6_minif2f_test, run_phase6_train
from .riemann_data import (
    RiemannAtlasConfig,
    RiemannDataConfig,
    materialize_riemann_data,
    validate_materialized_riemann_data,
)
from .sft2 import SFT2Config
from .sft2_evidence import (
    write_sft2_checkpoint_a_evidence,
    write_sft2_final_evidence,
)
from .sft2_inference import (
    reverify_sft2_minif2f_validation,
    reverify_sft2_train512,
    run_sft2_heldout512,
    run_sft2_minif2f_validation,
    run_sft2_train512,
)
from .sft2_training import (
    run_sft2_adapter_reload,
    run_sft2_preflight,
    run_sft2_training,
)


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
    smoke.add_argument(
        "--output-dir", type=Path, default=root / "artifacts/model-smoke"
    )
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

    gpt53_preflight = subparsers.add_parser(
        "gpt53-spark-preflight",
        help="prove the pinned GPT-5.3-Codex Spark/xhigh nested CLI contract",
    )
    gpt53_preflight.add_argument(
        "--config", type=Path, default=root / "config/gpt53-assessment.json"
    )
    gpt53_preflight.add_argument(
        "--output-dir", type=Path, default=root / "artifacts/gpt53-spark/preflight"
    )

    gpt53_assess = subparsers.add_parser(
        "gpt53-spark-assess",
        help="run isolated one-shot GPT-5.3-Codex Spark candidates through Lean",
    )
    gpt53_assess.add_argument("--benchmark-root", type=Path, required=True)
    gpt53_assess.add_argument(
        "--config", type=Path, default=root / "config/gpt53-assessment.json"
    )
    gpt53_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    gpt53_assess.add_argument(
        "--preflight-dir", type=Path, default=root / "artifacts/gpt53-spark/preflight"
    )
    gpt53_assess.add_argument("--output-dir", type=Path, required=True)
    gpt53_assess.add_argument(
        "--resume",
        action="store_true",
        help="reuse only exact, hash-validated accepted candidate artifacts",
    )

    gpt53_evidence = subparsers.add_parser(
        "gpt53-spark-evidence",
        help="write compact GPT-5.3-Codex Spark comparison evidence",
    )
    gpt53_evidence.add_argument(
        "--config", type=Path, default=root / "config/gpt53-assessment.json"
    )
    gpt53_evidence.add_argument(
        "--preflight-dir", type=Path, default=root / "artifacts/gpt53-spark/preflight"
    )
    gpt53_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/gpt53-spark/dev16"
    )
    gpt53_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/gpt53-spark/full"
    )
    gpt53_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/gpt53-spark"
    )

    phase2_loader = subparsers.add_parser(
        "phase2-loader-smoke", help="load and validate local Phase 2 JSONL splits"
    )
    phase2_loader.add_argument("--artifact-dir", type=Path, required=True)

    phase2_verify = subparsers.add_parser(
        "phase2-verify", help="reconstruct and verify the Phase 2 stratified sample"
    )
    phase2_verify.add_argument("--artifact-dir", type=Path, required=True)
    phase2_verify.add_argument("--mathlib-root", type=Path, required=True)
    phase2_verify.add_argument(
        "--config", type=Path, default=root / "config/phase2-mathlib.json"
    )
    phase2_verify.add_argument("--output", type=Path, required=True)
    phase2_verify.add_argument("--workers", type=int)
    phase2_verify.add_argument("--timeout", type=float)

    phase2_evidence = subparsers.add_parser(
        "phase2-evidence",
        help="write compact review evidence from local Phase 2 artifacts",
    )
    phase2_evidence.add_argument("--artifact-dir", type=Path, required=True)
    phase2_evidence.add_argument("--verification", type=Path)
    phase2_evidence.add_argument("--evidence-dir", type=Path, required=True)

    riemann_materialize = subparsers.add_parser(
        "riemann-materialize",
        help="materialize the pinned Riemann graph, corpora, holdouts, and atlas",
    )
    riemann_materialize.add_argument("--phase2-artifact-dir", type=Path, required=True)
    riemann_materialize.add_argument(
        "--phase2-snapshot-dir",
        type=Path,
        default=root / "data/mathlib-whole-proof-v1",
    )
    riemann_materialize.add_argument("--external-root", type=Path, required=True)
    riemann_materialize.add_argument(
        "--config", type=Path, default=root / "config/riemann-data.json"
    )
    riemann_materialize.add_argument(
        "--atlas-config", type=Path, default=root / "config/riemann-atlas.json"
    )
    riemann_materialize.add_argument(
        "--output-dir", type=Path, default=root / "data/riemann"
    )

    riemann_validate = subparsers.add_parser(
        "riemann-validate", help="validate committed Riemann data and file hashes"
    )
    riemann_validate.add_argument(
        "--data-dir", type=Path, default=root / "data/riemann"
    )

    phase3_materialize = subparsers.add_parser(
        "phase3-materialize",
        help="materialize the deterministic Phase 3 overfit workload",
    )
    phase3_materialize.add_argument("--artifact-dir", type=Path, required=True)
    phase3_materialize.add_argument(
        "--config", type=Path, default=root / "config/phase3-overfit.json"
    )
    phase3_materialize.add_argument("--output", type=Path, required=True)

    phase3_preflight = subparsers.add_parser(
        "phase3-preflight", help="run the one-step QLoRA GPU preflight"
    )
    phase3_preflight.add_argument("--workload", type=Path, required=True)
    phase3_preflight.add_argument(
        "--config", type=Path, default=root / "config/phase3-overfit.json"
    )
    phase3_preflight.add_argument("--output", type=Path, required=True)

    phase3_train = subparsers.add_parser(
        "phase3-train", help="overfit the fixed 64-example Phase 3 workload"
    )
    phase3_train.add_argument("--workload", type=Path, required=True)
    phase3_train.add_argument(
        "--config", type=Path, default=root / "config/phase3-overfit.json"
    )
    phase3_train.add_argument("--output-dir", type=Path, required=True)
    phase3_train.add_argument("--target-step", type=int, required=True)
    phase3_train.add_argument("--resume-from-checkpoint", type=Path)

    phase3_reload = subparsers.add_parser(
        "phase3-adapter-reload", help="reload the saved PEFT adapter on the pinned base"
    )
    phase3_reload.add_argument("--workload", type=Path, required=True)
    phase3_reload.add_argument("--adapter-dir", type=Path, required=True)
    phase3_reload.add_argument(
        "--config", type=Path, default=root / "config/phase3-overfit.json"
    )
    phase3_reload.add_argument("--output", type=Path, required=True)

    phase3_memorization = subparsers.add_parser(
        "phase3-memorization", help="run the adapter-backed vLLM overfit probe"
    )
    phase3_memorization.add_argument("--workload", type=Path, required=True)
    phase3_memorization.add_argument("--adapter-dir", type=Path, required=True)
    phase3_memorization.add_argument(
        "--config", type=Path, default=root / "config/phase3-overfit.json"
    )
    phase3_memorization.add_argument("--output", type=Path, required=True)
    phase3_memorization.add_argument("--optimizer-step", type=int)

    phase3_smoke = subparsers.add_parser(
        "phase3-adapter-smoke", help="run the adapter-backed miniF2F dev16 Lean smoke"
    )
    phase3_smoke.add_argument("--benchmark-root", type=Path, required=True)
    phase3_smoke.add_argument("--adapter-dir", type=Path, required=True)
    phase3_smoke.add_argument(
        "--config", type=Path, default=root / "config/phase3-overfit.json"
    )
    phase3_smoke.add_argument("--output-dir", type=Path, required=True)
    phase3_smoke.add_argument("--timeout", type=float, default=30.0)
    phase3_smoke.add_argument("--verification-workers", type=int, default=8)

    phase3_semantic = subparsers.add_parser(
        "phase3-semantic-verify",
        help="verify the raw step-600 continuations in original mathlib contexts",
    )
    phase3_semantic.add_argument("--dataset-dir", type=Path, required=True)
    phase3_semantic.add_argument("--mathlib-root", type=Path, required=True)
    phase3_semantic.add_argument("--memorization", type=Path, required=True)
    phase3_semantic.add_argument("--training", type=Path, required=True)
    phase3_semantic.add_argument("--output", type=Path, required=True)
    phase3_semantic.add_argument("--optimizer-step", type=int, default=600)
    phase3_semantic.add_argument("--workers", type=int)
    phase3_semantic.add_argument("--timeout", type=float)
    phase3_semantic.add_argument(
        "--config", type=Path, default=root / "config/phase3-overfit.json"
    )
    phase3_semantic.add_argument(
        "--phase2-config", type=Path, default=root / "config/phase2-mathlib.json"
    )

    phase3_evidence = subparsers.add_parser(
        "phase3-evidence", help="write compact evidence from local Phase 3 artifacts"
    )
    phase3_evidence.add_argument("--artifact-dir", type=Path, required=True)
    phase3_evidence.add_argument("--evidence-dir", type=Path, required=True)

    phase4_materialize = subparsers.add_parser(
        "phase4-materialize",
        help="materialize deterministic Phase 4 train/validation/heldout workloads",
    )
    phase4_materialize.add_argument("--artifact-dir", type=Path, required=True)
    phase4_materialize.add_argument(
        "--config", type=Path, default=root / "config/phase4-smoke.json"
    )
    phase4_materialize.add_argument("--output", type=Path, required=True)

    phase4_preflight = subparsers.add_parser(
        "phase4-preflight", help="run the near-maximum 1024-token QLoRA preflight"
    )
    phase4_preflight.add_argument("--workload", type=Path, required=True)
    phase4_preflight.add_argument(
        "--config", type=Path, default=root / "config/phase4-smoke.json"
    )
    phase4_preflight.add_argument("--output", type=Path, required=True)

    phase4_train = subparsers.add_parser(
        "phase4-train", help="run or resume the fixed Phase 4 smoke trajectory"
    )
    phase4_train.add_argument("--workload", type=Path, required=True)
    phase4_train.add_argument(
        "--config", type=Path, default=root / "config/phase4-smoke.json"
    )
    phase4_train.add_argument("--output-dir", type=Path, required=True)
    phase4_train.add_argument("--resume-from-checkpoint", type=Path)

    phase4_reload = subparsers.add_parser(
        "phase4-adapter-reload",
        help="reload the validation-selected Phase 4 PEFT adapter",
    )
    phase4_reload.add_argument("--workload", type=Path, required=True)
    phase4_reload.add_argument("--training", type=Path, required=True)
    phase4_reload.add_argument("--adapter-dir", type=Path, required=True)
    phase4_reload.add_argument(
        "--config", type=Path, default=root / "config/phase4-smoke.json"
    )
    phase4_reload.add_argument("--output", type=Path, required=True)

    phase4_heldout = subparsers.add_parser(
        "phase4-heldout",
        help="run a base or selected-adapter Phase 2 heldout evaluation",
    )
    phase4_heldout.add_argument("--dataset-dir", type=Path, required=True)
    phase4_heldout.add_argument("--mathlib-root", type=Path, required=True)
    phase4_heldout.add_argument("--workload", type=Path, required=True)
    phase4_heldout.add_argument("--training", type=Path, required=True)
    phase4_heldout.add_argument("--mode", choices=("base", "adapter"), required=True)
    phase4_heldout.add_argument("--adapter-dir", type=Path)
    phase4_heldout.add_argument("--output-dir", type=Path, required=True)
    phase4_heldout.add_argument("--workers", type=int)
    phase4_heldout.add_argument("--timeout", type=float)
    phase4_heldout.add_argument(
        "--config", type=Path, default=root / "config/phase4-smoke.json"
    )
    phase4_heldout.add_argument(
        "--phase2-config", type=Path, default=root / "config/phase2-mathlib.json"
    )

    phase4_compare = subparsers.add_parser(
        "phase4-heldout-compare", help="validate and summarize heldout comparability"
    )
    phase4_compare.add_argument("--training", type=Path, required=True)
    phase4_compare.add_argument("--base-dir", type=Path, required=True)
    phase4_compare.add_argument("--adapter-dir", type=Path, required=True)
    phase4_compare.add_argument("--output", type=Path, required=True)

    phase4_minif2f = subparsers.add_parser(
        "phase4-minif2f", help="evaluate the selected adapter on miniF2F dev16"
    )
    phase4_minif2f.add_argument("--benchmark-root", type=Path, required=True)
    phase4_minif2f.add_argument("--training", type=Path, required=True)
    phase4_minif2f.add_argument("--adapter-dir", type=Path, required=True)
    phase4_minif2f.add_argument("--output-dir", type=Path, required=True)
    phase4_minif2f.add_argument("--workers", type=int)
    phase4_minif2f.add_argument("--timeout", type=float)
    phase4_minif2f.add_argument(
        "--config", type=Path, default=root / "config/phase4-smoke.json"
    )

    phase4_evidence = subparsers.add_parser(
        "phase4-evidence", help="write compact Phase 4 review evidence"
    )
    phase4_evidence.add_argument("--artifact-dir", type=Path, required=True)
    phase4_evidence.add_argument("--evidence-dir", type=Path, required=True)

    phase5_materialize = subparsers.add_parser(
        "phase5-materialize",
        help="materialize full eligible Phase 5 train/validation workloads and heldout512",
    )
    phase5_materialize.add_argument("--artifact-dir", type=Path, required=True)
    phase5_materialize.add_argument(
        "--config", type=Path, default=root / "config/phase5-full.json"
    )
    phase5_materialize.add_argument("--output", type=Path, required=True)

    phase5_preflight = subparsers.add_parser(
        "phase5-preflight", help="run the full-corpus Phase 5 production preflight"
    )
    phase5_preflight.add_argument("--workload", type=Path, required=True)
    phase5_preflight.add_argument(
        "--config", type=Path, default=root / "config/phase5-full.json"
    )
    phase5_preflight.add_argument("--output", type=Path, required=True)

    phase5_train = subparsers.add_parser(
        "phase5-train", help="run or resume the one-pass Phase 5 trajectory"
    )
    phase5_train.add_argument("--workload", type=Path, required=True)
    phase5_train.add_argument(
        "--config", type=Path, default=root / "config/phase5-full.json"
    )
    phase5_train.add_argument("--output-dir", type=Path, required=True)
    phase5_train.add_argument("--resume-from-checkpoint", type=Path)

    phase5_reload = subparsers.add_parser(
        "phase5-adapter-reload",
        help="reload the validation-selected Phase 5 PEFT adapter",
    )
    phase5_reload.add_argument("--workload", type=Path, required=True)
    phase5_reload.add_argument("--training", type=Path, required=True)
    phase5_reload.add_argument("--adapter-dir", type=Path, required=True)
    phase5_reload.add_argument(
        "--config", type=Path, default=root / "config/phase5-full.json"
    )
    phase5_reload.add_argument("--output", type=Path, required=True)

    phase5_heldout = subparsers.add_parser(
        "phase5-heldout",
        help="run a base or selected-adapter Phase 5 heldout512 evaluation",
    )
    phase5_heldout.add_argument("--dataset-dir", type=Path, required=True)
    phase5_heldout.add_argument("--mathlib-root", type=Path, required=True)
    phase5_heldout.add_argument("--workload", type=Path, required=True)
    phase5_heldout.add_argument("--training", type=Path, required=True)
    phase5_heldout.add_argument("--mode", choices=("base", "adapter"), required=True)
    phase5_heldout.add_argument("--adapter-dir", type=Path)
    phase5_heldout.add_argument("--output-dir", type=Path, required=True)
    phase5_heldout.add_argument("--workers", type=int)
    phase5_heldout.add_argument("--timeout", type=float)
    phase5_heldout.add_argument(
        "--config", type=Path, default=root / "config/phase5-full.json"
    )
    phase5_heldout.add_argument(
        "--phase2-config", type=Path, default=root / "config/phase2-mathlib.json"
    )

    phase5_compare = subparsers.add_parser(
        "phase5-heldout-compare", help="validate and summarize heldout512 comparability"
    )
    phase5_compare.add_argument("--training", type=Path, required=True)
    phase5_compare.add_argument("--base-dir", type=Path, required=True)
    phase5_compare.add_argument("--adapter-dir", type=Path, required=True)
    phase5_compare.add_argument("--output", type=Path, required=True)

    phase5_minif2f = subparsers.add_parser(
        "phase5-minif2f",
        help="evaluate the selected adapter on full miniF2F validation",
    )
    phase5_minif2f.add_argument("--benchmark-root", type=Path, required=True)
    phase5_minif2f.add_argument("--training", type=Path, required=True)
    phase5_minif2f.add_argument("--adapter-dir", type=Path, required=True)
    phase5_minif2f.add_argument("--output-dir", type=Path, required=True)
    phase5_minif2f.add_argument("--workers", type=int)
    phase5_minif2f.add_argument("--timeout", type=float)
    phase5_minif2f.add_argument(
        "--config", type=Path, default=root / "config/phase5-full.json"
    )

    phase5_evidence = subparsers.add_parser(
        "phase5-evidence", help="write compact Phase 5 review evidence"
    )
    phase5_evidence.add_argument("--artifact-dir", type=Path, required=True)
    phase5_evidence.add_argument("--evidence-dir", type=Path, required=True)

    phase6_freeze = subparsers.add_parser(
        "phase6-freeze", help="freeze and validate the Phase 6 reference SFT candidate"
    )
    phase6_freeze.add_argument("--adapter-dir", type=Path, required=True)
    phase6_freeze.add_argument(
        "--phase5-training-evidence",
        type=Path,
        default=root / "evidence/phase5/training.json",
    )
    phase6_freeze.add_argument(
        "--config", type=Path, default=root / "config/phase6-eval.json"
    )
    phase6_freeze.add_argument("--output", type=Path, required=True)

    phase6_materialize = subparsers.add_parser(
        "phase6-materialize", help="materialize deterministic phase6-train512-v1"
    )
    phase6_materialize.add_argument("--dataset-dir", type=Path, required=True)
    phase6_materialize.add_argument(
        "--phase5-workload-evidence",
        type=Path,
        default=root / "evidence/phase5/workloads.json",
    )
    phase6_materialize.add_argument(
        "--config", type=Path, default=root / "config/phase6-eval.json"
    )
    phase6_materialize.add_argument("--output", type=Path, required=True)

    phase6_checkpoint = subparsers.add_parser(
        "phase6-checkpoint-a-evidence",
        help="write compact pre-generation Phase 6 integrity evidence",
    )
    phase6_checkpoint.add_argument("--candidate", type=Path, required=True)
    phase6_checkpoint.add_argument("--train-workload", type=Path, required=True)
    phase6_checkpoint.add_argument("--benchmark-root", type=Path, required=True)
    phase6_checkpoint.add_argument("--evidence-dir", type=Path, required=True)
    phase6_checkpoint.add_argument(
        "--config", type=Path, default=root / "config/phase6-eval.json"
    )

    phase6_train = subparsers.add_parser(
        "phase6-train", help="evaluate base or reference SFT on phase6-train512-v1"
    )
    phase6_train.add_argument("--dataset-dir", type=Path, required=True)
    phase6_train.add_argument("--mathlib-root", type=Path, required=True)
    phase6_train.add_argument("--workload", type=Path, required=True)
    phase6_train.add_argument("--candidate", type=Path, required=True)
    phase6_train.add_argument("--adapter-dir", type=Path, required=True)
    phase6_train.add_argument("--mode", choices=("base", "adapter"), required=True)
    phase6_train.add_argument("--output-dir", type=Path, required=True)
    phase6_train.add_argument("--workers", type=int)
    phase6_train.add_argument("--timeout", type=float)
    phase6_train.add_argument(
        "--config", type=Path, default=root / "config/phase6-eval.json"
    )
    phase6_train.add_argument(
        "--phase2-config", type=Path, default=root / "config/phase2-mathlib.json"
    )

    phase6_test = subparsers.add_parser(
        "phase6-minif2f-test",
        help="evaluate base or reference SFT on the complete miniF2F test split",
    )
    phase6_test.add_argument("--benchmark-root", type=Path, required=True)
    phase6_test.add_argument("--candidate", type=Path, required=True)
    phase6_test.add_argument("--adapter-dir", type=Path, required=True)
    phase6_test.add_argument("--mode", choices=("base", "adapter"), required=True)
    phase6_test.add_argument("--output-dir", type=Path, required=True)
    phase6_test.add_argument("--workers", type=int)
    phase6_test.add_argument("--timeout", type=float)
    phase6_test.add_argument(
        "--config", type=Path, default=root / "config/phase6-eval.json"
    )

    phase6_evidence = subparsers.add_parser(
        "phase6-evidence", help="write final compact Phase 6 comparison evidence"
    )
    phase6_evidence.add_argument(
        "--artifact-dir", type=Path, default=root / "artifacts/phase6"
    )
    phase6_evidence.add_argument(
        "--phase5-heldout-comparison",
        type=Path,
        default=root / "artifacts/phase5/heldout-comparison.json",
    )
    phase6_evidence.add_argument(
        "--phase5-heldout-base-dir",
        type=Path,
        default=root / "artifacts/phase5/heldout/base",
    )
    phase6_evidence.add_argument(
        "--phase5-heldout-adapter-dir",
        type=Path,
        default=root / "artifacts/phase5/heldout/adapter",
    )
    phase6_evidence.add_argument(
        "--phase1-validation-base-summary",
        type=Path,
        default=root / "evidence/phase1/baseline/summary.json",
    )
    phase6_evidence.add_argument(
        "--phase5-validation-adapter-evidence",
        type=Path,
        default=root / "evidence/phase5/minif2f.json",
    )
    phase6_evidence.add_argument("--evidence-dir", type=Path, required=True)
    phase6_evidence.add_argument(
        "--config", type=Path, default=root / "config/phase6-eval.json"
    )

    sft2_preflight = subparsers.add_parser(
        "sft2-preflight", help="validate immutable-parent SFT-2 continuation"
    )
    sft2_preflight.add_argument("--workload", type=Path, required=True)
    sft2_preflight.add_argument("--parent-adapter-dir", type=Path, required=True)
    sft2_preflight.add_argument("--candidate", type=Path, required=True)
    sft2_preflight.add_argument("--output", type=Path, required=True)
    sft2_preflight.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )

    sft2_checkpoint = subparsers.add_parser(
        "sft2-checkpoint-a-evidence",
        help="write compact pre-training SFT-2 integrity evidence",
    )
    sft2_checkpoint.add_argument("--workload", type=Path, required=True)
    sft2_checkpoint.add_argument("--preflight", type=Path, required=True)
    sft2_checkpoint.add_argument("--output", type=Path, required=True)
    sft2_checkpoint.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )

    sft2_train = subparsers.add_parser(
        "sft2-train", help="run or resume the fixed one-pass SFT-2 stage"
    )
    sft2_train.add_argument("--workload", type=Path, required=True)
    sft2_train.add_argument("--parent-adapter-dir", type=Path, required=True)
    sft2_train.add_argument("--candidate", type=Path, required=True)
    sft2_train.add_argument("--output-dir", type=Path, required=True)
    sft2_train.add_argument("--resume-from-checkpoint", type=Path)
    sft2_train.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )

    sft2_reload = subparsers.add_parser(
        "sft2-adapter-reload", help="reload the fixed SFT-2 Q4 endpoint"
    )
    sft2_reload.add_argument("--workload", type=Path, required=True)
    sft2_reload.add_argument("--training", type=Path, required=True)
    sft2_reload.add_argument("--adapter-dir", type=Path, required=True)
    sft2_reload.add_argument("--output", type=Path, required=True)
    sft2_reload.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )

    sft2_train_eval = subparsers.add_parser(
        "sft2-train512", help="evaluate the fixed SFT-2 endpoint on train512"
    )
    sft2_train_eval.add_argument("--dataset-dir", type=Path, required=True)
    sft2_train_eval.add_argument("--mathlib-root", type=Path, required=True)
    sft2_train_eval.add_argument("--workload", type=Path, required=True)
    sft2_train_eval.add_argument("--training", type=Path, required=True)
    sft2_train_eval.add_argument("--adapter-dir", type=Path, required=True)
    sft2_train_eval.add_argument("--output-dir", type=Path, required=True)
    sft2_train_eval.add_argument("--workers", type=int)
    sft2_train_eval.add_argument("--timeout", type=float)
    sft2_train_eval.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )
    sft2_train_eval.add_argument(
        "--phase2-config", type=Path, default=root / "config/phase2-mathlib.json"
    )

    sft2_train_reverify = subparsers.add_parser(
        "sft2-train512-reverify",
        help="retry only transient verifier results from stored SFT-2 train512 output",
    )
    sft2_train_reverify.add_argument("--dataset-dir", type=Path, required=True)
    sft2_train_reverify.add_argument("--mathlib-root", type=Path, required=True)
    sft2_train_reverify.add_argument("--workload", type=Path, required=True)
    sft2_train_reverify.add_argument("--training", type=Path, required=True)
    sft2_train_reverify.add_argument("--adapter-dir", type=Path, required=True)
    sft2_train_reverify.add_argument("--output-dir", type=Path, required=True)
    sft2_train_reverify.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )
    sft2_train_reverify.add_argument(
        "--phase2-config", type=Path, default=root / "config/phase2-mathlib.json"
    )

    sft2_heldout = subparsers.add_parser(
        "sft2-heldout512", help="evaluate the fixed SFT-2 endpoint on heldout512"
    )
    sft2_heldout.add_argument("--dataset-dir", type=Path, required=True)
    sft2_heldout.add_argument("--mathlib-root", type=Path, required=True)
    sft2_heldout.add_argument("--workload", type=Path, required=True)
    sft2_heldout.add_argument("--training", type=Path, required=True)
    sft2_heldout.add_argument("--adapter-dir", type=Path, required=True)
    sft2_heldout.add_argument("--output-dir", type=Path, required=True)
    sft2_heldout.add_argument("--workers", type=int)
    sft2_heldout.add_argument("--timeout", type=float)
    sft2_heldout.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )
    sft2_heldout.add_argument(
        "--phase2-config", type=Path, default=root / "config/phase2-mathlib.json"
    )

    sft2_minif2f = subparsers.add_parser(
        "sft2-minif2f-validation",
        help="evaluate the fixed SFT-2 endpoint on miniF2F validation",
    )
    sft2_minif2f.add_argument("--benchmark-root", type=Path, required=True)
    sft2_minif2f.add_argument("--training", type=Path, required=True)
    sft2_minif2f.add_argument("--adapter-dir", type=Path, required=True)
    sft2_minif2f.add_argument("--output-dir", type=Path, required=True)
    sft2_minif2f.add_argument("--workers", type=int)
    sft2_minif2f.add_argument("--timeout", type=float)
    sft2_minif2f.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )

    sft2_minif2f_reverify = subparsers.add_parser(
        "sft2-minif2f-validation-reverify",
        help="retry only transient verifier results from stored SFT-2 miniF2F validation",
    )
    sft2_minif2f_reverify.add_argument("--benchmark-root", type=Path, required=True)
    sft2_minif2f_reverify.add_argument("--training", type=Path, required=True)
    sft2_minif2f_reverify.add_argument("--adapter-dir", type=Path, required=True)
    sft2_minif2f_reverify.add_argument("--output-dir", type=Path, required=True)
    sft2_minif2f_reverify.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )

    sft2_evidence = subparsers.add_parser(
        "sft2-evidence", help="write compact final SFT-2 comparison evidence"
    )
    sft2_evidence.add_argument(
        "--artifact-dir", type=Path, default=root / "artifacts/sft2"
    )
    sft2_evidence.add_argument(
        "--reference-train-dir",
        type=Path,
        default=root / "artifacts/phase6/train/adapter",
    )
    sft2_evidence.add_argument(
        "--reference-heldout-dir",
        type=Path,
        default=root / "artifacts/phase5/heldout/adapter",
    )
    sft2_evidence.add_argument(
        "--reference-minif2f-dir",
        type=Path,
        default=root / "artifacts/phase5/minif2f",
    )
    sft2_evidence.add_argument(
        "--phase6-comparison",
        type=Path,
        default=root / "evidence/phase6/comparison.json",
    )
    sft2_evidence.add_argument("--evidence-dir", type=Path, required=True)
    sft2_evidence.add_argument(
        "--config", type=Path, default=root / "config/sft2-ablation.json"
    )
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

    if args.command == "gpt53-spark-preflight":
        summary = run_preflight(GPT53Config.load(args.config), args.output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "gpt53-spark-assess":
        _, _, summary = run_assessment(
            GPT53Config.load(args.config),
            benchmark_root=args.benchmark_root,
            workload_id=args.workload,
            preflight_dir=args.preflight_dir,
            output_dir=args.output_dir,
            resume=args.resume,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "gpt53-spark-evidence":
        comparison = write_gpt53_evidence(
            GPT53Config.load(args.config),
            preflight_dir=args.preflight_dir,
            dev16_dir=args.dev16_dir,
            full_dir=args.full_dir,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "phase2-loader-smoke":
        dataset = load_phase2_dataset(args.artifact_dir)
        counts = {split: len(dataset[split]) for split in dataset}
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0

    if args.command == "phase2-verify":
        config = Phase2Config.load(args.config)
        evidence = verify_phase2_sample(
            config,
            args.artifact_dir,
            args.mathlib_root,
            args.output,
            workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(evidence["summary"], indent=2, sort_keys=True))
        return 0

    if args.command == "phase2-evidence":
        write_compact_evidence(
            args.artifact_dir,
            args.evidence_dir,
            verification_path=args.verification,
        )
        print(str(args.evidence_dir))
        return 0

    if args.command == "riemann-materialize":
        summary = materialize_riemann_data(
            args.phase2_artifact_dir,
            args.output_dir,
            RiemannDataConfig.load(args.config),
            RiemannAtlasConfig.load(args.atlas_config),
            external_root=args.external_root,
            phase2_snapshot_dir=args.phase2_snapshot_dir,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "riemann-validate":
        summary = validate_materialized_riemann_data(args.data_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "phase3-materialize":
        config = Phase3Config.load(args.config)
        tokenizer = load_pinned_tokenizer(config)
        examples, eligible = select_overfit_workload(
            load_phase2_train_records(args.artifact_dir), tokenizer, config
        )
        write_phase3_workload(
            args.output,
            config,
            examples,
            eligible_examples=eligible,
            eos_token_id=int(tokenizer.eos_token_id),
        )
        print(
            json.dumps(
                {
                    "workload_id": config.workload["id"],
                    "eligible_examples": eligible,
                    "selected_examples": len(examples),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "phase3-preflight":
        value = run_training_preflight(
            Phase3Config.load(args.config), args.workload, args.output
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase3-train":
        value = run_overfit_training(
            Phase3Config.load(args.config),
            args.workload,
            args.output_dir,
            target_step=args.target_step,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase3-adapter-reload":
        value = run_adapter_reload_check(
            Phase3Config.load(args.config),
            args.workload,
            args.adapter_dir,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase3-memorization":
        value = run_vllm_memorization(
            Phase3Config.load(args.config),
            args.workload,
            args.adapter_dir,
            args.output,
            optimizer_step=args.optimizer_step,
        )
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "exact_matches": value["exact_matches"],
                    "examples": value["examples"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "phase3-adapter-smoke":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_adapter_minif2f_smoke(
            Phase3Config.load(args.config),
            args.benchmark_root,
            args.adapter_dir,
            args.output_dir,
            timeout_seconds=args.timeout,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase3-semantic-verify":
        evidence = run_phase3_semantic_verification(
            Phase3Config.load(args.config),
            Phase2Config.load(args.phase2_config),
            args.dataset_dir,
            args.mathlib_root,
            args.memorization,
            args.training,
            args.output,
            optimizer_step=args.optimizer_step,
            workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(evidence["summary"], indent=2, sort_keys=True))
        return 0

    if args.command == "phase3-evidence":
        write_phase3_evidence(args.artifact_dir, args.evidence_dir)
        print(str(args.evidence_dir))
        return 0

    if args.command == "phase4-materialize":
        config = Phase4Config.load(args.config)
        workloads = materialize_phase4_workloads(args.artifact_dir, config)
        write_phase4_workloads(args.output, config, workloads)
        print(
            json.dumps(
                {
                    "eligible_examples": workloads.eligible_counts,
                    "selected_examples": {
                        "train": len(workloads.train),
                        "validation": len(workloads.validation),
                        "heldout": len(workloads.heldout),
                    },
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "phase4-preflight":
        value = run_phase4_preflight(
            Phase4Config.load(args.config), args.workload, args.output
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase4-train":
        value = run_phase4_training(
            Phase4Config.load(args.config),
            args.workload,
            args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase4-adapter-reload":
        value = run_phase4_adapter_reload(
            Phase4Config.load(args.config),
            args.workload,
            args.training,
            args.adapter_dir,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase4-heldout":
        if args.mode == "base" and args.adapter_dir is not None:
            print("--adapter-dir is forbidden in base mode")
            return 2
        if args.mode == "adapter" and args.adapter_dir is None:
            print("--adapter-dir is required in adapter mode")
            return 2
        _, _, summary = run_phase4_heldout(
            Phase4Config.load(args.config),
            Phase2Config.load(args.phase2_config),
            args.dataset_dir,
            args.mathlib_root,
            args.workload,
            args.training,
            args.output_dir,
            adapter_dir=args.adapter_dir,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase4-heldout-compare":
        value = compare_phase4_heldout_runs(
            args.training, args.base_dir, args.adapter_dir, args.output
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase4-minif2f":
        _, _, summary = run_phase4_minif2f(
            Phase4Config.load(args.config),
            args.benchmark_root,
            args.training,
            args.adapter_dir,
            args.output_dir,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase4-evidence":
        write_phase4_evidence(args.artifact_dir, args.evidence_dir)
        print(str(args.evidence_dir))
        return 0

    if args.command == "phase5-materialize":
        config = Phase5Config.load(args.config)
        workloads = materialize_phase5_workloads(args.artifact_dir, config)
        write_phase5_workloads(args.output, config, workloads)
        print(
            json.dumps(
                {
                    "input_examples": workloads.input_counts,
                    "eligible_examples": workloads.eligible_counts,
                    "overlength_examples": {
                        name: len(items) for name, items in workloads.overlength.items()
                    },
                    "selected_examples": {
                        "train": len(workloads.train),
                        "validation": len(workloads.validation),
                        "heldout": len(workloads.heldout),
                    },
                    "trajectory": workloads.trajectory.to_dict(),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "phase5-preflight":
        value = run_phase5_preflight(
            Phase5Config.load(args.config), args.workload, args.output
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase5-train":
        value = run_phase5_training(
            Phase5Config.load(args.config),
            args.workload,
            args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase5-adapter-reload":
        value = run_phase5_adapter_reload(
            Phase5Config.load(args.config),
            args.workload,
            args.training,
            args.adapter_dir,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase5-heldout":
        if args.mode == "base" and args.adapter_dir is not None:
            print("--adapter-dir is forbidden in base mode")
            return 2
        if args.mode == "adapter" and args.adapter_dir is None:
            print("--adapter-dir is required in adapter mode")
            return 2
        _, _, summary = run_phase5_heldout(
            Phase5Config.load(args.config),
            Phase2Config.load(args.phase2_config),
            args.dataset_dir,
            args.mathlib_root,
            args.workload,
            args.training,
            args.output_dir,
            adapter_dir=args.adapter_dir,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase5-heldout-compare":
        value = compare_phase5_heldout_runs(
            args.training, args.base_dir, args.adapter_dir, args.output
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase5-minif2f":
        _, _, summary = run_phase5_minif2f(
            Phase5Config.load(args.config),
            args.benchmark_root,
            args.training,
            args.adapter_dir,
            args.output_dir,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase5-evidence":
        write_phase5_evidence(args.artifact_dir, args.evidence_dir)
        print(str(args.evidence_dir))
        return 0

    if args.command == "phase6-freeze":
        value = freeze_reference_candidate(
            Phase6Config.load(args.config),
            args.adapter_dir,
            args.phase5_training_evidence,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "phase6-materialize":
        config = Phase6Config.load(args.config)
        tokenizer = load_pinned_tokenizer(config)  # type: ignore[arg-type]
        value = materialize_phase6_train_workload(
            config,
            args.dataset_dir,
            args.phase5_workload_evidence,
            tokenizer,
        )
        write_phase6_train_workload(args.output, value)
        print(
            json.dumps(
                {
                    "workload_id": value["workload_id"],
                    "selected_examples": len(value["selected_record_ids"]),
                    "selected_record_ids_sha256": value["selected_record_ids_sha256"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "phase6-checkpoint-a-evidence":
        write_phase6_checkpoint_a_evidence(
            Phase6Config.load(args.config),
            args.candidate,
            args.train_workload,
            args.benchmark_root,
            args.evidence_dir,
        )
        print(str(args.evidence_dir))
        return 0

    if args.command == "phase6-train":
        _, _, summary = run_phase6_train(
            Phase6Config.load(args.config),
            Phase2Config.load(args.phase2_config),
            args.dataset_dir,
            args.mathlib_root,
            args.workload,
            args.candidate,
            args.adapter_dir,
            args.output_dir,
            mode=args.mode,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase6-minif2f-test":
        _, _, summary = run_phase6_minif2f_test(
            Phase6Config.load(args.config),
            args.benchmark_root,
            args.candidate,
            args.adapter_dir,
            args.output_dir,
            mode=args.mode,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase6-evidence":
        value = write_phase6_final_evidence(
            Phase6Config.load(args.config),
            args.artifact_dir,
            args.phase5_heldout_comparison,
            args.phase5_heldout_base_dir,
            args.phase5_heldout_adapter_dir,
            args.phase1_validation_base_summary,
            args.phase5_validation_adapter_evidence,
            args.evidence_dir,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "sft2-preflight":
        value = run_sft2_preflight(
            SFT2Config.load(args.config),
            args.workload,
            args.parent_adapter_dir,
            args.candidate,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "sft2-checkpoint-a-evidence":
        value = write_sft2_checkpoint_a_evidence(
            SFT2Config.load(args.config),
            args.workload,
            args.preflight,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "sft2-train":
        value = run_sft2_training(
            SFT2Config.load(args.config),
            args.workload,
            args.parent_adapter_dir,
            args.candidate,
            args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "sft2-adapter-reload":
        value = run_sft2_adapter_reload(
            SFT2Config.load(args.config),
            args.workload,
            args.training,
            args.adapter_dir,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "sft2-train512":
        _, _, summary = run_sft2_train512(
            SFT2Config.load(args.config),
            Phase2Config.load(args.phase2_config),
            args.dataset_dir,
            args.mathlib_root,
            args.workload,
            args.training,
            args.adapter_dir,
            args.output_dir,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "sft2-train512-reverify":
        _, _, summary = reverify_sft2_train512(
            SFT2Config.load(args.config),
            Phase2Config.load(args.phase2_config),
            args.dataset_dir,
            args.mathlib_root,
            args.workload,
            args.training,
            args.adapter_dir,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "sft2-heldout512":
        _, _, summary = run_sft2_heldout512(
            SFT2Config.load(args.config),
            Phase2Config.load(args.phase2_config),
            args.dataset_dir,
            args.mathlib_root,
            args.workload,
            args.training,
            args.adapter_dir,
            args.output_dir,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "sft2-minif2f-validation":
        _, _, summary = run_sft2_minif2f_validation(
            SFT2Config.load(args.config),
            args.benchmark_root,
            args.training,
            args.adapter_dir,
            args.output_dir,
            verification_workers=args.workers,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "sft2-minif2f-validation-reverify":
        _, _, summary = reverify_sft2_minif2f_validation(
            SFT2Config.load(args.config),
            args.benchmark_root,
            args.training,
            args.adapter_dir,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "sft2-evidence":
        value = write_sft2_final_evidence(
            SFT2Config.load(args.config),
            args.artifact_dir,
            args.reference_train_dir,
            args.reference_heldout_dir,
            args.reference_minif2f_dir,
            args.phase6_comparison,
            args.evidence_dir,
        )
        print(json.dumps(value, indent=2))
        return 0

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
