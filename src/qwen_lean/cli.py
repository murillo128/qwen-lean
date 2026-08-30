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
from .deepseek_prover_assessment import (
    load_assessment_config as load_deepseek_prover_config,
)
from .deepseek_prover_assessment import (
    run_strict_assessment as run_deepseek_prover_assessment,
)
from .deepseek_prover_assessment import (
    write_compact_evidence as write_deepseek_prover_evidence,
)
from .goedel_assessment import (
    run_assessment as run_goedel_assessment,
    run_preflight as run_goedel_preflight,
    write_compact_evidence as write_goedel_evidence,
)
from .gpt53_assessment import (
    GPT53Config,
    run_assessment,
    run_preflight,
)
from .gpt53_assessment import (
    write_compact_evidence as write_gpt53_evidence,
)
from .minif2f import Phase1Config
from .ministral3_assessment import (
    Ministral3AssessmentConfig,
    reverify_assessment as reverify_ministral3_assessment,
    run_assessment as run_ministral3_assessment,
    run_preflight as run_ministral3_preflight,
    write_compact_evidence as write_ministral3_evidence,
)
from .olmo3_assessment import (
    load_assessment_config as load_olmo3_config,
)
from .olmo3_assessment import run_preflight as run_olmo3_preflight
from .olmo3_assessment import (
    run_strict_assessment as run_olmo3_assessment,
)
from .olmo3_assessment import (
    write_compact_evidence as write_olmo3_evidence,
)
from .qwen3_posttrained_assessment import (
    load_assessment_config as load_qwen3_posttrained_config,
)
from .qwen3_posttrained_assessment import (
    run_strict_assessment as run_qwen3_posttrained_assessment,
)
from .qwen3_posttrained_assessment import (
    write_compact_evidence as write_qwen3_posttrained_evidence,
)
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
from .qwen35_9b_posttrained_assessment import (
    run_assessment as run_qwen35_9b_posttrained_assessment,
)
from .qwen35_9b_posttrained_assessment import (
    run_precision_preflight as run_qwen35_9b_posttrained_preflight,
)
from .qwen35_9b_posttrained_assessment import (
    write_compact_evidence as write_qwen35_9b_posttrained_evidence,
)
from .qwen36_27b_assessment import (
    Qwen36AssessmentConfig,
    run_assessment as run_qwen36_27b_assessment,
    run_preflight as run_qwen36_27b_preflight,
    write_blocker_evidence as write_qwen36_27b_blocker_evidence,
    write_compact_evidence as write_qwen36_27b_evidence,
)
from .qwen35_4b_posttrained_assessment import (
    load_assessment_config as load_qwen35_assessment_config,
    run_preflight as run_qwen35_preflight,
    run_strict_assessment as run_qwen35_strict_assessment,
    write_compact_evidence as write_qwen35_evidence,
)
from .qwen35_4b_base_assessment import (
    run_assessment as run_qwen35_4b_base_assessment,
)
from .qwen35_4b_base_assessment import (
    write_compact_evidence as write_qwen35_4b_base_evidence,
)
from .native_thinking_assessment import (
    NativeThinkingConfig,
    run_generation as run_native_thinking_generation,
    run_preflight as run_native_thinking_preflight,
    run_verification as run_native_thinking_verification,
    write_final_evidence as write_native_thinking_evidence,
)
from .counterfactual_forking_assessment import (
    CounterfactualForkingConfig,
    run_counterfactual_preflight,
    run_fork_generation as run_counterfactual_generation,
    run_fork_verification as run_counterfactual_verification,
    write_final_evidence as write_counterfactual_evidence,
)
from .full_context_forking_diagnostic import (
    FullContextForkingConfig,
    run_calibration_probe as run_full_context_calibration_probe,
    run_context_calibration as run_full_context_calibration,
    run_full_context_generation,
    run_full_context_verification,
    write_full_context_evidence,
)
from .qwen35_posttrained_assessment import (
    Qwen35AssessmentConfig,
    run_assessment as run_qwen35_posttrained_assessment,
    run_preflight as run_qwen35_posttrained_preflight,
    write_compact_evidence as write_qwen35_posttrained_evidence,
)
from .qwen35_9b_base_assessment import (
    Qwen35BaseAssessmentConfig,
    WORKLOADS,
    run_assessment as run_qwen35_9b_base_assessment,
    run_preflight as run_qwen35_9b_base_preflight,
    write_compact_evidence as write_qwen35_9b_base_evidence,
)
from .riemann_data import (
    RiemannAtlasConfig,
    RiemannDataConfig,
    materialize_riemann_data,
    validate_materialized_riemann_data,
)
from .qwen35_9b_riemann_assessment import (
    RiemannAssessmentConfig,
    run_generation as run_riemann_generation,
    run_preflight as run_riemann_preflight,
    run_verification as run_riemann_verification,
    write_compact_evidence as write_riemann_evidence,
)
from .riemann_assessment import (
    run_assessment as run_riemann_qwen35_4b_assessment,
)
from .riemann_assessment import run_preflight as run_riemann_qwen35_4b_preflight
from .riemann_assessment import (
    write_compact_evidence as write_riemann_qwen35_4b_evidence,
)
from .qwen35_assessment import (
    run_assessment as run_qwen35_base_assessment,
)
from .qwen35_assessment import (
    run_preflight as run_qwen35_base_preflight,
)
from .qwen35_assessment import (
    write_compact_evidence as write_qwen35_base_evidence,
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

    qwen3_assess = subparsers.add_parser(
        "qwen3-posttrained-assess",
        help="run the strict Qwen3-8B official post-trained assessment",
    )
    qwen3_assess.add_argument("--benchmark-root", type=Path, required=True)
    qwen3_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen3-8b-posttrained-assessment.json",
    )
    qwen3_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    qwen3_assess.add_argument("--output-dir", type=Path, required=True)
    qwen3_assess.add_argument("--verification-workers", type=int, default=8)

    qwen3_evidence = subparsers.add_parser(
        "qwen3-posttrained-evidence",
        help="write compact Qwen3-8B post-trained comparison evidence",
    )
    qwen3_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen3-8b-posttrained-assessment.json",
    )
    qwen3_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/qwen3-8b/dev16"
    )
    qwen3_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/qwen3-8b/full"
    )
    qwen3_evidence.add_argument(
        "--base-dir", type=Path, default=root / "evidence/phase1/baseline"
    )
    qwen3_evidence.add_argument(
        "--reference",
        type=Path,
        default=root / "evidence/phase5/minif2f.json",
    )
    qwen3_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/qwen3-8b-posttrained"
    )

    deepseek_assess = subparsers.add_parser(
        "deepseek-prover-assess",
        help="run the strict DeepSeek-Prover-V2-7B Lean assessment",
    )
    deepseek_assess.add_argument("--benchmark-root", type=Path, required=True)
    deepseek_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/deepseek-prover-v2-7b-assessment.json",
    )
    deepseek_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    deepseek_assess.add_argument("--output-dir", type=Path, required=True)
    deepseek_assess.add_argument("--verification-workers", type=int, default=8)

    deepseek_evidence = subparsers.add_parser(
        "deepseek-prover-evidence",
        help="write compact DeepSeek-Prover-V2-7B comparison evidence",
    )
    deepseek_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/deepseek-prover-v2-7b-assessment.json",
    )
    deepseek_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/deepseek-prover/dev16"
    )
    deepseek_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/deepseek-prover/full"
    )
    deepseek_evidence.add_argument(
        "--base-dir", type=Path, default=root / "evidence/phase1/baseline"
    )
    deepseek_evidence.add_argument(
        "--reference", type=Path, default=root / "evidence/phase5/minif2f.json"
    )
    deepseek_evidence.add_argument(
        "--qwen3-posttrained",
        type=Path,
        default=root / "evidence/qwen3-8b-posttrained/full.json",
    )
    deepseek_evidence.add_argument(
        "--qwen35-4b-base",
        type=Path,
        default=root / "evidence/qwen35-4b-base/full.json",
    )
    deepseek_evidence.add_argument(
        "--goedel",
        type=Path,
        default=root / "evidence/goedel-prover-v2-8b/full.json",
    )
    deepseek_evidence.add_argument(
        "--evidence-dir",
        type=Path,
        default=root / "evidence/deepseek-prover-v2-7b",
    )

    olmo3_preflight = subparsers.add_parser(
        "olmo3-preflight",
        help="run the local BF16 OLMo 3 7B compatibility preflight",
    )
    olmo3_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/olmo3-7b-assessment.json",
    )
    olmo3_preflight.add_argument(
        "--output", type=Path, default=root / "artifacts/olmo3-7b/preflight.json"
    )

    olmo3_assess = subparsers.add_parser(
        "olmo3-assess", help="run the strict OLMo 3 7B miniF2F assessment"
    )
    olmo3_assess.add_argument("--benchmark-root", type=Path, required=True)
    olmo3_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/olmo3-7b-assessment.json",
    )
    olmo3_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    olmo3_assess.add_argument("--output-dir", type=Path, required=True)
    olmo3_assess.add_argument("--verification-workers", type=int, default=8)

    olmo3_evidence = subparsers.add_parser(
        "olmo3-evidence", help="write compact OLMo 3 7B assessment evidence"
    )
    olmo3_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/olmo3-7b-assessment.json",
    )
    olmo3_evidence.add_argument(
        "--preflight",
        type=Path,
        default=root / "artifacts/olmo3-7b/preflight.json",
    )
    olmo3_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/olmo3-7b/dev16"
    )
    olmo3_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/olmo3-7b/full"
    )
    olmo3_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/olmo3-7b"
    )

    qwen35_9b_posttrained_preflight = subparsers.add_parser(
        "qwen35-9b-preflight",
        help="establish the BF16 or frozen 4-bit Qwen3.5-9B precision lane",
    )
    qwen35_9b_posttrained_preflight.add_argument(
        "--benchmark-root", type=Path, required=True
    )
    qwen35_9b_posttrained_preflight.add_argument(
        "--config", type=Path, default=root / "config/qwen35-9b-assessment.json"
    )
    qwen35_9b_posttrained_preflight.add_argument(
        "--lane", required=True, choices=("bf16", "bitsandbytes-4bit")
    )
    qwen35_9b_posttrained_preflight.add_argument(
        "--output", type=Path, required=True
    )

    qwen35_9b_posttrained_assess = subparsers.add_parser(
        "qwen35-9b-assess",
        help="run the strict Qwen3.5-9B miniF2F casting lane",
    )
    qwen35_9b_posttrained_assess.add_argument(
        "--benchmark-root", type=Path, required=True
    )
    qwen35_9b_posttrained_assess.add_argument(
        "--config", type=Path, default=root / "config/qwen35-9b-assessment.json"
    )
    qwen35_9b_posttrained_assess.add_argument(
        "--preflight", type=Path, required=True
    )
    qwen35_9b_posttrained_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    qwen35_9b_posttrained_assess.add_argument(
        "--output-dir", type=Path, required=True
    )
    qwen35_9b_posttrained_assess.add_argument("--timeout", type=float)
    qwen35_9b_posttrained_assess.add_argument(
        "--verification-workers", type=int, default=8
    )

    qwen35_9b_posttrained_evidence = subparsers.add_parser(
        "qwen35-9b-evidence",
        help="write compact strict-lane Qwen3.5-9B comparison evidence",
    )
    qwen35_9b_posttrained_evidence.add_argument(
        "--config", type=Path, default=root / "config/qwen35-9b-assessment.json"
    )
    qwen35_9b_posttrained_evidence.add_argument(
        "--preflight", type=Path, default=root / "artifacts/qwen35-9b/preflight.json"
    )
    qwen35_9b_posttrained_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/qwen35-9b/dev16"
    )
    qwen35_9b_posttrained_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/qwen35-9b/full"
    )
    qwen35_9b_posttrained_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/qwen35-9b"
    )

    qwen36_27b_preflight = subparsers.add_parser(
        "qwen36-27b-preflight",
        help="run the frozen Qwen3.6-27B 4-bit local-Ada preflight",
    )
    qwen36_27b_preflight.add_argument("--benchmark-root", type=Path, required=True)
    qwen36_27b_preflight.add_argument("--model-snapshot", type=Path, required=True)
    qwen36_27b_preflight.add_argument(
        "--config", type=Path, default=root / "config/qwen36-27b-assessment.json"
    )
    qwen36_27b_preflight.add_argument("--output", type=Path, required=True)

    qwen36_27b_assess = subparsers.add_parser(
        "qwen36-27b-assess",
        help="run the frozen Qwen3.6-27B 4-bit strict miniF2F assessment",
    )
    qwen36_27b_assess.add_argument("--benchmark-root", type=Path, required=True)
    qwen36_27b_assess.add_argument("--model-snapshot", type=Path, required=True)
    qwen36_27b_assess.add_argument("--preflight", type=Path, required=True)
    qwen36_27b_assess.add_argument(
        "--config", type=Path, default=root / "config/qwen36-27b-assessment.json"
    )
    qwen36_27b_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    qwen36_27b_assess.add_argument("--output-dir", type=Path, required=True)

    qwen36_27b_evidence = subparsers.add_parser(
        "qwen36-27b-evidence",
        help="write compact Qwen3.6-27B 4-bit comparison evidence",
    )
    qwen36_27b_evidence.add_argument(
        "--config", type=Path, default=root / "config/qwen36-27b-assessment.json"
    )
    qwen36_27b_evidence.add_argument("--preflight", type=Path, required=True)
    qwen36_27b_evidence.add_argument("--dev16-dir", type=Path, required=True)
    qwen36_27b_evidence.add_argument("--full-dir", type=Path, required=True)
    qwen36_27b_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/qwen36-27b"
    )

    qwen36_27b_blocker_evidence = subparsers.add_parser(
        "qwen36-27b-blocker-evidence",
        help="write compact evidence for a frozen-lane Stage 0 hardware blocker",
    )
    qwen36_27b_blocker_evidence.add_argument(
        "--config", type=Path, default=root / "config/qwen36-27b-assessment.json"
    )
    qwen36_27b_blocker_evidence.add_argument(
        "--preflight", type=Path, required=True
    )
    qwen36_27b_blocker_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/qwen36-27b"
    )

    qwen35_preflight = subparsers.add_parser(
        "qwen35-4b-preflight",
        help="run the frozen one-task Qwen3.5-4B BF16 compatibility preflight",
    )
    qwen35_preflight.add_argument("--benchmark-root", type=Path, required=True)
    qwen35_preflight.add_argument(
        "--config", type=Path, default=root / "config/qwen35-4b-assessment.json"
    )
    qwen35_preflight.add_argument("--output-dir", type=Path, required=True)
    qwen35_preflight.add_argument("--verification-workers", type=int, default=1)

    qwen35_assess = subparsers.add_parser(
        "qwen35-4b-assess",
        help="run the frozen Qwen3.5-4B strict raw-continuation assessment",
    )
    qwen35_assess.add_argument("--benchmark-root", type=Path, required=True)
    qwen35_assess.add_argument(
        "--config", type=Path, default=root / "config/qwen35-4b-assessment.json"
    )
    qwen35_assess.add_argument(
        "--workload",
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
        required=True,
    )
    qwen35_assess.add_argument("--output-dir", type=Path, required=True)
    qwen35_assess.add_argument("--verification-workers", type=int, default=8)

    qwen35_evidence = subparsers.add_parser(
        "qwen35-4b-evidence",
        help="write compact Qwen3.5-4B casting evidence from retained local runs",
    )
    qwen35_evidence.add_argument(
        "--config", type=Path, default=root / "config/qwen35-4b-assessment.json"
    )
    qwen35_evidence.add_argument(
        "--preflight-dir",
        type=Path,
        default=root / "artifacts/qwen35-4b/preflight",
    )
    qwen35_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/qwen35-4b/dev16"
    )
    qwen35_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/qwen35-4b/full"
    )
    qwen35_evidence.add_argument(
        "--reference-summary",
        type=Path,
        default=root / "evidence/phase5/minif2f.json",
    )
    qwen35_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/qwen35-4b"
    )

    native_thinking_preflight = subparsers.add_parser(
        "qwen35-native-thinking-preflight",
        help="run the frozen native-thinking pre-inference gate",
    )
    native_thinking_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-native-thinking-ab.json",
    )
    native_thinking_preflight.add_argument("--mathia-root", type=Path, required=True)
    native_thinking_preflight.add_argument(
        "--minif2f-root", type=Path, required=True
    )
    native_thinking_preflight.add_argument(
        "--mathlib-root", type=Path, default=root
    )
    native_thinking_preflight.add_argument("--artifact-dir", type=Path, required=True)
    native_thinking_preflight.add_argument(
        "--output",
        type=Path,
        default=root / "evidence/qwen35-native-thinking/pre-inference.json",
    )

    native_thinking_generate = subparsers.add_parser(
        "qwen35-native-thinking-generate",
        help="run or resume one frozen native-thinking generation arm",
    )
    native_thinking_generate.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-native-thinking-ab.json",
    )
    native_thinking_generate.add_argument("--mathia-root", type=Path, required=True)
    native_thinking_generate.add_argument("--arm", choices=("t0", "t1"), required=True)
    native_thinking_generate.add_argument("--artifact-dir", type=Path, required=True)

    native_thinking_verify = subparsers.add_parser(
        "qwen35-native-thinking-verify",
        help="run or resume final-channel-only Lean verification",
    )
    native_thinking_verify.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-native-thinking-ab.json",
    )
    native_thinking_verify.add_argument("--mathia-root", type=Path, required=True)
    native_thinking_verify.add_argument("--minif2f-root", type=Path, required=True)
    native_thinking_verify.add_argument("--mathlib-root", type=Path, default=root)
    native_thinking_verify.add_argument("--artifact-dir", type=Path, required=True)
    native_thinking_verify.add_argument("--workers", type=int)

    native_thinking_evidence = subparsers.add_parser(
        "qwen35-native-thinking-evidence",
        help="write compact paired quality, interface, diversity, and cost evidence",
    )
    native_thinking_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-native-thinking-ab.json",
    )
    native_thinking_evidence.add_argument("--mathia-root", type=Path, required=True)
    native_thinking_evidence.add_argument("--artifact-dir", type=Path, required=True)
    native_thinking_evidence.add_argument(
        "--preflight",
        type=Path,
        default=root / "evidence/qwen35-native-thinking/pre-inference.json",
    )
    native_thinking_evidence.add_argument(
        "--evidence-dir",
        type=Path,
        default=root / "evidence/qwen35-native-thinking",
    )

    counterfactual_preflight = subparsers.add_parser(
        "qwen35-counterfactual-preflight",
        help="run exact-token parent parity and two bounded real fork probes",
    )
    counterfactual_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-counterfactual-forking.json",
    )
    counterfactual_preflight.add_argument("--mathia-root", type=Path, required=True)
    counterfactual_preflight.add_argument(
        "--parent-generations", type=Path, required=True
    )
    counterfactual_preflight.add_argument(
        "--parent-package",
        type=Path,
        help="pinned GitHub Release archive when using the compact parent JSONL",
    )
    counterfactual_preflight.add_argument(
        "--minif2f-root", type=Path, required=True
    )
    counterfactual_preflight.add_argument(
        "--mathlib-root", type=Path, default=root
    )
    counterfactual_preflight.add_argument("--artifact-dir", type=Path, required=True)
    counterfactual_preflight.add_argument(
        "--output",
        type=Path,
        default=root / "evidence/qwen35-counterfactual-forking/pre-inference.json",
    )
    counterfactual_preflight.add_argument("--workers", type=int)

    counterfactual_generate = subparsers.add_parser(
        "qwen35-counterfactual-generate",
        help="run or resume discovery or matched-budget confirmation forks",
    )
    counterfactual_generate.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-counterfactual-forking.json",
    )
    counterfactual_generate.add_argument("--mathia-root", type=Path, required=True)
    counterfactual_generate.add_argument(
        "--parent-generations", type=Path, required=True
    )
    counterfactual_generate.add_argument(
        "--parent-package",
        type=Path,
        help="pinned GitHub Release archive when using the compact parent JSONL",
    )
    counterfactual_generate.add_argument("--artifact-dir", type=Path, required=True)
    counterfactual_generate.add_argument(
        "--phase", choices=("discovery", "confirmation"), required=True
    )

    counterfactual_verify = subparsers.add_parser(
        "qwen35-counterfactual-verify",
        help="run or resume exact-final-channel Lean verification for fork branches",
    )
    counterfactual_verify.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-counterfactual-forking.json",
    )
    counterfactual_verify.add_argument("--mathia-root", type=Path, required=True)
    counterfactual_verify.add_argument(
        "--parent-generations", type=Path, required=True
    )
    counterfactual_verify.add_argument(
        "--parent-package",
        type=Path,
        help="pinned GitHub Release archive when using the compact parent JSONL",
    )
    counterfactual_verify.add_argument(
        "--minif2f-root", type=Path, required=True
    )
    counterfactual_verify.add_argument(
        "--mathlib-root", type=Path, default=root
    )
    counterfactual_verify.add_argument("--artifact-dir", type=Path, required=True)
    counterfactual_verify.add_argument(
        "--phase", choices=("discovery", "confirmation"), required=True
    )
    counterfactual_verify.add_argument("--workers", type=int)

    counterfactual_evidence = subparsers.add_parser(
        "qwen35-counterfactual-evidence",
        help="write compact discovery and matched-budget confirmation evidence",
    )
    counterfactual_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-counterfactual-forking.json",
    )
    counterfactual_evidence.add_argument("--mathia-root", type=Path, required=True)
    counterfactual_evidence.add_argument(
        "--parent-generations", type=Path, required=True
    )
    counterfactual_evidence.add_argument(
        "--parent-package",
        type=Path,
        help="pinned GitHub Release archive when using the compact parent JSONL",
    )
    counterfactual_evidence.add_argument("--artifact-dir", type=Path, required=True)
    counterfactual_evidence.add_argument(
        "--preflight",
        type=Path,
        default=root / "evidence/qwen35-counterfactual-forking/pre-inference.json",
    )
    counterfactual_evidence.add_argument(
        "--evidence-dir",
        type=Path,
        default=root / "evidence/qwen35-counterfactual-forking",
    )

    full_context_probe = subparsers.add_parser(
        "qwen35-full-context-calibration-probe",
        help=argparse.SUPPRESS,
    )
    full_context_probe.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-full-context-forking.json",
    )
    full_context_probe.add_argument("--mathia-root", type=Path, required=True)
    full_context_probe.add_argument("--parent-generations", type=Path, required=True)
    full_context_probe.add_argument("--parent-package", type=Path)
    full_context_probe.add_argument("--context-length", type=int, required=True)
    full_context_probe.add_argument("--seed", type=int, required=True)
    full_context_probe.add_argument("--output", type=Path, required=True)

    full_context_calibrate = subparsers.add_parser(
        "qwen35-full-context-calibrate",
        help="calibrate and repeat-confirm the real sustainable local-GPU context",
    )
    full_context_calibrate.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-full-context-forking.json",
    )
    full_context_calibrate.add_argument("--mathia-root", type=Path, required=True)
    full_context_calibrate.add_argument(
        "--parent-generations", type=Path, required=True
    )
    full_context_calibrate.add_argument("--parent-package", type=Path)
    full_context_calibrate.add_argument("--artifact-dir", type=Path, required=True)
    full_context_calibrate.add_argument(
        "--output",
        type=Path,
        default=root / "evidence/qwen35-full-context-forking/calibration-rtx4000.json",
    )

    full_context_generate = subparsers.add_parser(
        "qwen35-full-context-generate",
        help="run or resume the 42 frozen full-context branches",
    )
    full_context_generate.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-full-context-forking.json",
    )
    full_context_generate.add_argument("--mathia-root", type=Path, required=True)
    full_context_generate.add_argument("--parent-generations", type=Path, required=True)
    full_context_generate.add_argument("--parent-package", type=Path)
    full_context_generate.add_argument("--artifact-dir", type=Path, required=True)
    full_context_generate.add_argument(
        "--calibration",
        type=Path,
        default=root / "evidence/qwen35-full-context-forking/calibration-rtx4000.json",
    )
    full_context_generate.add_argument(
        "--checkpoint-review",
        type=Path,
        default=(
            root / "evidence/qwen35-full-context-forking/calibration-review-rtx4000.json"
        ),
    )

    full_context_verify = subparsers.add_parser(
        "qwen35-full-context-verify",
        help="run or resume exact-final-channel Lean checks for all 42 branches",
    )
    full_context_verify.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-full-context-forking.json",
    )
    full_context_verify.add_argument("--mathia-root", type=Path, required=True)
    full_context_verify.add_argument("--parent-generations", type=Path, required=True)
    full_context_verify.add_argument("--parent-package", type=Path)
    full_context_verify.add_argument("--artifact-dir", type=Path, required=True)
    full_context_verify.add_argument(
        "--calibration",
        type=Path,
        default=root / "evidence/qwen35-full-context-forking/calibration-rtx4000.json",
    )
    full_context_verify.add_argument(
        "--checkpoint-review",
        type=Path,
        default=(
            root / "evidence/qwen35-full-context-forking/calibration-review-rtx4000.json"
        ),
    )
    full_context_verify.add_argument("--minif2f-root", type=Path, required=True)
    full_context_verify.add_argument("--mathlib-root", type=Path, default=root)
    full_context_verify.add_argument("--workers", type=int)

    full_context_evidence = subparsers.add_parser(
        "qwen35-full-context-evidence",
        help="write compact scoring-excluded full-context diagnostic evidence",
    )
    full_context_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-full-context-forking.json",
    )
    full_context_evidence.add_argument("--mathia-root", type=Path, required=True)
    full_context_evidence.add_argument("--parent-generations", type=Path, required=True)
    full_context_evidence.add_argument("--parent-package", type=Path)
    full_context_evidence.add_argument("--artifact-dir", type=Path, required=True)
    full_context_evidence.add_argument(
        "--calibration",
        type=Path,
        default=root / "evidence/qwen35-full-context-forking/calibration-rtx4000.json",
    )
    full_context_evidence.add_argument(
        "--checkpoint-review",
        type=Path,
        default=(
            root / "evidence/qwen35-full-context-forking/calibration-review-rtx4000.json"
        ),
    )
    full_context_evidence.add_argument(
        "--output",
        type=Path,
        default=root / "evidence/qwen35-full-context-forking/results.json",
    )

    qwen35_4b_base_assess = subparsers.add_parser(
        "qwen35-4b-base-assess",
        help="run the strict local-GPU Qwen3.5-4B-Base foundation assessment",
    )
    qwen35_4b_base_assess.add_argument(
        "--benchmark-root", type=Path, required=True
    )
    qwen35_4b_base_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-4b-base-assessment.json",
    )
    qwen35_4b_base_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    qwen35_4b_base_assess.add_argument(
        "--output-dir", type=Path, required=True
    )
    qwen35_4b_base_assess.add_argument(
        "--verification-workers", type=int, default=8
    )

    qwen35_4b_base_evidence = subparsers.add_parser(
        "qwen35-4b-base-evidence",
        help="write compact Qwen3.5-4B-Base assessment evidence",
    )
    qwen35_4b_base_evidence.add_argument(
        "--benchmark-root", type=Path, required=True
    )
    qwen35_4b_base_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-4b-base-assessment.json",
    )
    qwen35_4b_base_evidence.add_argument(
        "--environment-validation", type=Path, required=True
    )
    qwen35_4b_base_evidence.add_argument(
        "--dev16-dir", type=Path, required=True
    )
    qwen35_4b_base_evidence.add_argument(
        "--full-dir", type=Path, required=True
    )
    qwen35_4b_base_evidence.add_argument(
        "--evidence-dir", type=Path, required=True
    )

    riemann_qwen35_4b_preflight = subparsers.add_parser(
        "riemann-qwen35-4b-preflight",
        help="validate the frozen Qwen3.5-4B-Base Riemann casting inputs",
    )
    riemann_qwen35_4b_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-4b-base-riemann-assessment.json",
    )
    riemann_qwen35_4b_preflight.add_argument(
        "--domain-config",
        type=Path,
        default=root / "config/riemann-domain-breakdown.json",
    )
    riemann_qwen35_4b_preflight.add_argument(
        "--mathlib-root", type=Path, required=True
    )
    riemann_qwen35_4b_preflight.add_argument("--output", type=Path, required=True)

    riemann_qwen35_4b_assess = subparsers.add_parser(
        "riemann-qwen35-4b-assess",
        help="run the complete local-GPU Qwen3.5-4B-Base Riemann casting",
    )
    riemann_qwen35_4b_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-4b-base-riemann-assessment.json",
    )
    riemann_qwen35_4b_assess.add_argument(
        "--domain-config",
        type=Path,
        default=root / "config/riemann-domain-breakdown.json",
    )
    riemann_qwen35_4b_assess.add_argument(
        "--mathlib-root", type=Path, required=True
    )
    riemann_qwen35_4b_assess.add_argument(
        "--preflight", type=Path, required=True
    )
    riemann_qwen35_4b_assess.add_argument(
        "--output-dir", type=Path, required=True
    )
    riemann_qwen35_4b_assess.add_argument(
        "--verification-workers", type=int, default=8
    )

    riemann_qwen35_4b_evidence = subparsers.add_parser(
        "riemann-qwen35-4b-evidence",
        help="write compact Qwen3.5-4B-Base Riemann casting evidence",
    )
    riemann_qwen35_4b_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-4b-base-riemann-assessment.json",
    )
    riemann_qwen35_4b_evidence.add_argument(
        "--domain-config",
        type=Path,
        default=root / "config/riemann-domain-breakdown.json",
    )
    riemann_qwen35_4b_evidence.add_argument(
        "--preflight", type=Path, required=True
    )
    riemann_qwen35_4b_evidence.add_argument(
        "--artifact-dir", type=Path, required=True
    )
    riemann_qwen35_4b_evidence.add_argument(
        "--evidence-dir", type=Path, required=True
    )

    riemann_deepseek_preflight = subparsers.add_parser(
        "riemann-deepseek-preflight",
        help="validate the frozen DeepSeek specialist-parent Riemann inputs",
    )
    riemann_deepseek_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/deepseek-prover-v2-7b-riemann-assessment.json",
    )
    riemann_deepseek_preflight.add_argument(
        "--domain-config",
        type=Path,
        default=root / "config/riemann-domain-breakdown.json",
    )
    riemann_deepseek_preflight.add_argument(
        "--mathlib-root", type=Path, required=True
    )
    riemann_deepseek_preflight.add_argument("--output", type=Path, required=True)

    riemann_deepseek_assess = subparsers.add_parser(
        "riemann-deepseek-assess",
        help="run the complete local-GPU DeepSeek specialist-parent Riemann casting",
    )
    riemann_deepseek_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/deepseek-prover-v2-7b-riemann-assessment.json",
    )
    riemann_deepseek_assess.add_argument(
        "--domain-config",
        type=Path,
        default=root / "config/riemann-domain-breakdown.json",
    )
    riemann_deepseek_assess.add_argument(
        "--mathlib-root", type=Path, required=True
    )
    riemann_deepseek_assess.add_argument("--preflight", type=Path, required=True)
    riemann_deepseek_assess.add_argument("--output-dir", type=Path, required=True)
    riemann_deepseek_assess.add_argument(
        "--verification-workers", type=int, default=8
    )

    riemann_deepseek_evidence = subparsers.add_parser(
        "riemann-deepseek-evidence",
        help="write compact DeepSeek specialist-parent Riemann casting evidence",
    )
    riemann_deepseek_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/deepseek-prover-v2-7b-riemann-assessment.json",
    )
    riemann_deepseek_evidence.add_argument(
        "--domain-config",
        type=Path,
        default=root / "config/riemann-domain-breakdown.json",
    )
    riemann_deepseek_evidence.add_argument("--preflight", type=Path, required=True)
    riemann_deepseek_evidence.add_argument(
        "--artifact-dir", type=Path, required=True
    )
    riemann_deepseek_evidence.add_argument(
        "--evidence-dir", type=Path, required=True
    )
    riemann_deepseek_evidence.add_argument(
        "--qwen35-4b-outcomes",
        type=Path,
        default=root / "evidence/riemann-qwen35-4b-base/task-outcomes.jsonl",
    )
    riemann_deepseek_evidence.add_argument(
        "--qwen35-9b-outcomes", type=Path
    )
    riemann_deepseek_evidence.add_argument(
        "--execution-limitation", action="append", default=[]
    )

    qwen35_preflight = subparsers.add_parser(
        "qwen35-preflight",
        help="prove pinned Qwen3.5 BF16/text-only compatibility and GPU memory",
    )
    qwen35_preflight.add_argument(
        "--config", type=Path, default=root / "config/qwen35-2b-base-assessment.json"
    )
    qwen35_preflight.add_argument(
        "--output", type=Path, default=root / "artifacts/qwen35-2b-base/preflight.json"
    )

    qwen35_assess = subparsers.add_parser(
        "qwen35-assess", help="run the strict local-GPU Qwen3.5 assessment"
    )
    qwen35_assess.add_argument("--benchmark-root", type=Path, required=True)
    qwen35_assess.add_argument(
        "--config", type=Path, default=root / "config/qwen35-2b-base-assessment.json"
    )
    qwen35_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    qwen35_assess.add_argument("--output-dir", type=Path, required=True)
    qwen35_assess.add_argument("--timeout", type=float)
    qwen35_assess.add_argument("--verification-workers", type=int, default=8)

    qwen35_evidence = subparsers.add_parser(
        "qwen35-evidence", help="write compact Qwen3.5 assessment evidence"
    )
    qwen35_evidence.add_argument(
        "--config", type=Path, default=root / "config/qwen35-2b-base-assessment.json"
    )
    qwen35_evidence.add_argument(
        "--preflight", type=Path, default=root / "artifacts/qwen35-2b-base/preflight.json"
    )
    qwen35_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/qwen35-2b-base/dev16"
    )
    qwen35_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/qwen35-2b-base/full"
    )
    qwen35_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/qwen35-2b-base"
    )

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

    ministral_preflight = subparsers.add_parser(
        "ministral3-8b-base-preflight",
        help="freeze the local BF16 or NF4 Ministral 3 assessment lane",
    )
    ministral_preflight.add_argument("--benchmark-root", type=Path, required=True)
    ministral_preflight.add_argument("--model-snapshot", type=Path, required=True)
    ministral_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/ministral3-8b-base-assessment.json",
    )
    ministral_preflight.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/ministral3-8b-base/preflight.json",
    )

    ministral_assess = subparsers.add_parser(
        "ministral3-8b-base-assess",
        help="run strict local Ministral 3 whole-proof assessment",
    )
    ministral_assess.add_argument("--benchmark-root", type=Path, required=True)
    ministral_assess.add_argument("--model-snapshot", type=Path, required=True)
    ministral_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/ministral3-8b-base-assessment.json",
    )
    ministral_assess.add_argument(
        "--preflight",
        type=Path,
        default=root / "artifacts/ministral3-8b-base/preflight.json",
    )
    ministral_assess.add_argument(
        "--workload", required=True, choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1")
    )
    ministral_assess.add_argument("--output-dir", type=Path, required=True)

    ministral_reverify = subparsers.add_parser(
        "ministral3-8b-base-reverify",
        help="reverify unchanged Ministral 3 candidates after an environment failure",
    )
    ministral_reverify.add_argument("--benchmark-root", type=Path, required=True)
    ministral_reverify.add_argument(
        "--config",
        type=Path,
        default=root / "config/ministral3-8b-base-assessment.json",
    )
    ministral_reverify.add_argument("--input-dir", type=Path, required=True)
    ministral_reverify.add_argument("--output-dir", type=Path, required=True)

    ministral_evidence = subparsers.add_parser(
        "ministral3-8b-base-evidence",
        help="write compact Ministral 3 assessment evidence",
    )
    ministral_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/ministral3-8b-base-assessment.json",
    )
    ministral_evidence.add_argument(
        "--preflight",
        type=Path,
        default=root / "artifacts/ministral3-8b-base/preflight.json",
    )
    ministral_evidence.add_argument(
        "--dev16-dir",
        type=Path,
        default=root / "artifacts/ministral3-8b-base/dev16",
    )
    ministral_evidence.add_argument(
        "--full-dir",
        type=Path,
        default=root / "artifacts/ministral3-8b-base/full",
    )
    ministral_evidence.add_argument(
        "--evidence-dir",
        type=Path,
        default=root / "evidence/ministral3-8b-base",
    )

    qwen35_9b_preflight = subparsers.add_parser(
        "qwen35-9b-base-preflight",
        help="resolve the BF16/4-bit local-Ada lane for Qwen3.5-9B-Base",
    )
    qwen35_9b_preflight.add_argument("--benchmark-root", type=Path, required=True)
    qwen35_9b_preflight.add_argument("--model-snapshot", type=Path, required=True)
    qwen35_9b_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-9b-base-assessment.json",
    )
    qwen35_9b_preflight.add_argument("--output", type=Path, required=True)

    qwen35_9b_assess = subparsers.add_parser(
        "qwen35-9b-base-assess",
        help="run the strict Qwen3.5-9B-Base miniF2F assessment",
    )
    qwen35_9b_assess.add_argument("--benchmark-root", type=Path, required=True)
    qwen35_9b_assess.add_argument("--model-snapshot", type=Path, required=True)
    qwen35_9b_assess.add_argument("--preflight", type=Path, required=True)
    qwen35_9b_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-9b-base-assessment.json",
    )
    qwen35_9b_assess.add_argument(
        "--workload", required=True, choices=WORKLOADS
    )
    qwen35_9b_assess.add_argument("--output-dir", type=Path, required=True)

    qwen35_9b_evidence = subparsers.add_parser(
        "qwen35-9b-base-evidence",
        help="write compact evidence from the strict Qwen3.5-9B-Base run",
    )
    qwen35_9b_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-9b-base-assessment.json",
    )
    qwen35_9b_evidence.add_argument("--preflight", type=Path, required=True)
    qwen35_9b_evidence.add_argument("--dev16-dir", type=Path, required=True)
    qwen35_9b_evidence.add_argument("--full-dir", type=Path, required=True)
    qwen35_9b_evidence.add_argument(
        "--evidence-dir",
        type=Path,
        default=root / "evidence/qwen35-9b-base",
    )

    riemann_preflight = subparsers.add_parser(
        "qwen35-9b-riemann-preflight",
        help="preflight the frozen Qwen3.5-9B-Base Riemann assessment",
    )
    riemann_preflight.add_argument("--repository-root", type=Path, default=root)
    riemann_preflight.add_argument("--mathlib-root", type=Path, required=True)
    riemann_preflight.add_argument("--lean-environment-root", type=Path, required=True)
    riemann_preflight.add_argument("--model-snapshot", type=Path, required=True)
    riemann_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-9b-riemann-assessment.json",
    )
    riemann_preflight.add_argument("--output", type=Path, required=True)

    riemann_generate = subparsers.add_parser(
        "qwen35-9b-riemann-generate",
        help="generate the frozen 556 by 4 Riemann continuations",
    )
    riemann_generate.add_argument("--repository-root", type=Path, default=root)
    riemann_generate.add_argument("--model-snapshot", type=Path, required=True)
    riemann_generate.add_argument("--preflight", type=Path, required=True)
    riemann_generate.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-9b-riemann-assessment.json",
    )
    riemann_generate.add_argument("--output-dir", type=Path, required=True)

    riemann_verify = subparsers.add_parser(
        "qwen35-9b-riemann-verify",
        help="verify a frozen Riemann generation artifact without regeneration",
    )
    riemann_verify.add_argument("--repository-root", type=Path, default=root)
    riemann_verify.add_argument("--mathlib-root", type=Path, required=True)
    riemann_verify.add_argument("--lean-environment-root", type=Path, required=True)
    riemann_verify.add_argument("--preflight", type=Path, required=True)
    riemann_verify.add_argument("--generation-dir", type=Path, required=True)
    riemann_verify.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-9b-riemann-assessment.json",
    )
    riemann_verify.add_argument("--output-dir", type=Path, required=True)

    riemann_evidence = subparsers.add_parser(
        "qwen35-9b-riemann-evidence",
        help="write compact domain-aware evidence from the frozen Riemann run",
    )
    riemann_evidence.add_argument("--repository-root", type=Path, default=root)
    riemann_evidence.add_argument("--preflight", type=Path, required=True)
    riemann_evidence.add_argument("--generation-dir", type=Path, required=True)
    riemann_evidence.add_argument("--artifact-dir", type=Path, required=True)
    riemann_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/qwen35-9b-riemann-assessment.json",
    )
    riemann_evidence.add_argument(
        "--domain-config",
        type=Path,
        default=root / "config/riemann-domain-breakdown.json",
    )
    riemann_evidence.add_argument(
        "--evidence-dir",
        type=Path,
        default=root / "evidence/qwen35-9b-riemann",
    )

    qwen35_preflight = subparsers.add_parser(
        "qwen35-2b-preflight",
        help="run the real BF16 Qwen3.5-2B compatibility and memory preflight",
    )
    qwen35_preflight.add_argument("--benchmark-root", type=Path, required=True)
    qwen35_preflight.add_argument(
        "--config", type=Path, default=root / "config/qwen35-2b-assessment.json"
    )
    qwen35_preflight.add_argument(
        "--output-dir", type=Path, default=root / "artifacts/qwen35-2b/preflight"
    )

    qwen35_assess = subparsers.add_parser(
        "qwen35-2b-assess",
        help="run strict raw-continuation Qwen3.5-2B miniF2F generation",
    )
    qwen35_assess.add_argument("--benchmark-root", type=Path, required=True)
    qwen35_assess.add_argument(
        "--config", type=Path, default=root / "config/qwen35-2b-assessment.json"
    )
    qwen35_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    qwen35_assess.add_argument(
        "--preflight-dir", type=Path, default=root / "artifacts/qwen35-2b/preflight"
    )
    qwen35_assess.add_argument("--output-dir", type=Path, required=True)
    qwen35_assess.add_argument("--verification-workers", type=int, default=8)

    qwen35_evidence = subparsers.add_parser(
        "qwen35-2b-evidence",
        help="write compact Qwen3.5-2B assessment and reference comparison",
    )
    qwen35_evidence.add_argument(
        "--config", type=Path, default=root / "config/qwen35-2b-assessment.json"
    )
    qwen35_evidence.add_argument(
        "--preflight-dir", type=Path, default=root / "artifacts/qwen35-2b/preflight"
    )
    qwen35_evidence.add_argument(
        "--dev16-dir", type=Path, default=root / "artifacts/qwen35-2b/dev16"
    )
    qwen35_evidence.add_argument(
        "--full-dir", type=Path, default=root / "artifacts/qwen35-2b/full"
    )
    qwen35_evidence.add_argument(
        "--reference-sft-evidence",
        type=Path,
        default=root / "evidence/phase5/minif2f.json",
    )
    qwen35_evidence.add_argument(
        "--evidence-dir", type=Path, default=root / "evidence/qwen35-2b"
    )

    goedel_preflight = subparsers.add_parser(
        "goedel-preflight",
        help="validate the pinned Goedel-Prover snapshot and strict environment",
    )
    goedel_preflight.add_argument("--benchmark-root", type=Path, required=True)
    goedel_preflight.add_argument("--model-snapshot", type=Path, required=True)
    goedel_preflight.add_argument(
        "--config",
        type=Path,
        default=root / "config/goedel-prover-v2-assessment.json",
    )
    goedel_preflight.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/goedel-prover-v2/preflight.json",
    )

    goedel_assess = subparsers.add_parser(
        "goedel-assess",
        help="run strict local Goedel-Prover-V2 miniF2F generation and verification",
    )
    goedel_assess.add_argument("--benchmark-root", type=Path, required=True)
    goedel_assess.add_argument(
        "--config",
        type=Path,
        default=root / "config/goedel-prover-v2-assessment.json",
    )
    goedel_assess.add_argument(
        "--preflight",
        type=Path,
        default=root / "artifacts/goedel-prover-v2/preflight.json",
    )
    goedel_assess.add_argument(
        "--workload",
        required=True,
        choices=("minif2f-valid-dev16-v1", "minif2f-valid-v1"),
    )
    goedel_assess.add_argument("--output-dir", type=Path, required=True)

    goedel_evidence = subparsers.add_parser(
        "goedel-evidence", help="write compact Goedel-Prover-V2 assessment evidence"
    )
    goedel_evidence.add_argument(
        "--config",
        type=Path,
        default=root / "config/goedel-prover-v2-assessment.json",
    )
    goedel_evidence.add_argument(
        "--preflight",
        type=Path,
        default=root / "artifacts/goedel-prover-v2/preflight.json",
    )
    goedel_evidence.add_argument(
        "--dev16-dir",
        type=Path,
        default=root / "artifacts/goedel-prover-v2/dev16",
    )
    goedel_evidence.add_argument(
        "--full-dir",
        type=Path,
        default=root / "artifacts/goedel-prover-v2/full",
    )
    goedel_evidence.add_argument(
        "--evidence-dir",
        type=Path,
        default=root / "evidence/goedel-prover-v2-8b",
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

    if args.command == "qwen3-posttrained-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_qwen3_posttrained_assessment(
            load_qwen3_posttrained_config(args.config),
            args.benchmark_root,
            args.workload,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen3-posttrained-evidence":
        comparison = write_qwen3_posttrained_evidence(
            load_qwen3_posttrained_config(args.config),
            dev16_dir=args.dev16_dir,
            full_dir=args.full_dir,
            base_dir=args.base_dir,
            reference_path=args.reference,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "deepseek-prover-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_deepseek_prover_assessment(
            load_deepseek_prover_config(args.config),
            args.benchmark_root,
            args.workload,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "deepseek-prover-evidence":
        comparison = write_deepseek_prover_evidence(
            load_deepseek_prover_config(args.config),
            dev16_dir=args.dev16_dir,
            full_dir=args.full_dir,
            base_dir=args.base_dir,
            reference_path=args.reference,
            qwen3_posttrained_path=args.qwen3_posttrained,
            qwen35_4b_base_path=args.qwen35_4b_base,
            goedel_path=args.goedel,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "olmo3-preflight":
        evidence = run_olmo3_preflight(
            load_olmo3_config(args.config), args.output
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "olmo3-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_olmo3_assessment(
            load_olmo3_config(args.config),
            args.benchmark_root,
            args.workload,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "olmo3-evidence":
        evidence = write_olmo3_evidence(
            load_olmo3_config(args.config),
            preflight_path=args.preflight,
            dev16_dir=args.dev16_dir,
            full_dir=args.full_dir,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-9b-preflight":
        state = run_qwen35_9b_posttrained_preflight(
            Phase1Config.load(args.config),
            args.benchmark_root,
            args.output,
            lane=args.lane,
        )
        print(json.dumps(state, indent=2))
        return 0 if state["status"] == "passed" else 1

    if args.command == "qwen35-9b-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        config = Phase1Config.load(args.config)
        timeout = (
            float(config.value["verifier"]["timeout_seconds"])
            if args.timeout is None
            else args.timeout
        )
        _, _, summary = run_qwen35_9b_posttrained_assessment(
            config,
            args.benchmark_root,
            args.preflight,
            args.workload,
            args.output_dir,
            verification_workers=args.verification_workers,
            timeout_seconds=timeout,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-9b-evidence":
        comparison = write_qwen35_9b_posttrained_evidence(
            Phase1Config.load(args.config),
            args.preflight,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "qwen36-27b-preflight":
        evidence = run_qwen36_27b_preflight(
            Qwen36AssessmentConfig.load(args.config),
            args.benchmark_root,
            args.model_snapshot,
            args.output,
        )
        print(json.dumps(evidence, indent=2))
        return 0 if evidence["status"] == "passed" else 1

    if args.command == "qwen36-27b-assess":
        _, _, summary = run_qwen36_27b_assessment(
            Qwen36AssessmentConfig.load(args.config),
            args.benchmark_root,
            args.model_snapshot,
            args.preflight,
            args.workload,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen36-27b-evidence":
        comparison = write_qwen36_27b_evidence(
            Qwen36AssessmentConfig.load(args.config),
            args.preflight,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "qwen36-27b-blocker-evidence":
        evidence = write_qwen36_27b_blocker_evidence(
            Qwen36AssessmentConfig.load(args.config),
            args.preflight,
            args.evidence_dir,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-4b-preflight":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_qwen35_preflight(
            load_qwen35_assessment_config(args.config),
            args.benchmark_root,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-4b-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_qwen35_strict_assessment(
            load_qwen35_assessment_config(args.config),
            args.benchmark_root,
            args.workload,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-4b-evidence":
        comparison = write_qwen35_evidence(
            load_qwen35_assessment_config(args.config),
            preflight_dir=args.preflight_dir,
            dev16_dir=args.dev16_dir,
            full_dir=args.full_dir,
            reference_summary_path=args.reference_summary,
            evidence_dir=args.evidence_dir,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "qwen35-native-thinking-preflight":
        evidence = run_native_thinking_preflight(
            NativeThinkingConfig.load(args.config),
            args.mathia_root,
            args.artifact_dir,
            args.output,
            project_roots={
                "minif2f-valid-clean-v2": args.minif2f_root,
                "fresh-composition-valid-v2": args.mathlib_root,
            },
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-native-thinking-generate":
        summary = run_native_thinking_generation(
            NativeThinkingConfig.load(args.config),
            args.mathia_root,
            args.arm,
            args.artifact_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-native-thinking-verify":
        if args.workers is not None and args.workers < 1:
            print("--workers must be positive")
            return 2
        summary = run_native_thinking_verification(
            NativeThinkingConfig.load(args.config),
            args.mathia_root,
            args.artifact_dir,
            project_roots={
                "minif2f-valid-clean-v2": args.minif2f_root,
                "fresh-composition-valid-v2": args.mathlib_root,
            },
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-native-thinking-evidence":
        evidence = write_native_thinking_evidence(
            NativeThinkingConfig.load(args.config),
            args.mathia_root,
            args.artifact_dir,
            args.preflight,
            args.evidence_dir,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-counterfactual-preflight":
        if args.workers is not None and args.workers < 1:
            print("--workers must be positive")
            return 2
        summary = run_counterfactual_preflight(
            CounterfactualForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            args.output,
            project_roots={
                "minif2f-valid-clean-v2": args.minif2f_root,
                "fresh-composition-valid-v2": args.mathlib_root,
            },
            workers=args.workers,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-counterfactual-generate":
        summary = run_counterfactual_generation(
            CounterfactualForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            phase=args.phase,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-counterfactual-verify":
        if args.workers is not None and args.workers < 1:
            print("--workers must be positive")
            return 2
        summary = run_counterfactual_verification(
            CounterfactualForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            phase=args.phase,
            project_roots={
                "minif2f-valid-clean-v2": args.minif2f_root,
                "fresh-composition-valid-v2": args.mathlib_root,
            },
            workers=args.workers,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-counterfactual-evidence":
        evidence = write_counterfactual_evidence(
            CounterfactualForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            args.preflight,
            args.evidence_dir,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-full-context-calibration-probe":
        result = run_full_context_calibration_probe(
            FullContextForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.output,
            context_length=args.context_length,
            seed=args.seed,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "qwen35-full-context-calibrate":
        evidence = run_full_context_calibration(
            FullContextForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            args.output,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-full-context-generate":
        summary = run_full_context_generation(
            FullContextForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            args.calibration,
            args.checkpoint_review,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-full-context-verify":
        if args.workers is not None and args.workers < 1:
            print("--workers must be positive")
            return 2
        summary = run_full_context_verification(
            FullContextForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            args.calibration,
            args.checkpoint_review,
            project_roots={
                "minif2f-valid-clean-v2": args.minif2f_root,
                "fresh-composition-valid-v2": args.mathlib_root,
            },
            workers=args.workers,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-full-context-evidence":
        evidence = write_full_context_evidence(
            FullContextForkingConfig.load(args.config),
            args.mathia_root,
            args.parent_generations,
            args.artifact_dir,
            args.calibration,
            args.checkpoint_review,
            args.output,
            parent_release_package_path=args.parent_package,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-4b-base-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        config = Phase1Config.load(args.config)
        timeout = float(config.value["verifier"]["timeout_seconds"])
        _, _, summary = run_qwen35_4b_base_assessment(
            config,
            args.benchmark_root,
            args.workload,
            args.output_dir,
            timeout_seconds=timeout,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-4b-base-evidence":
        comparison = write_qwen35_4b_base_evidence(
            Phase1Config.load(args.config),
            args.benchmark_root,
            args.environment_validation,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "riemann-qwen35-4b-preflight":
        evidence = run_riemann_qwen35_4b_preflight(
            Phase1Config.load(args.config),
            _project_root(),
            args.domain_config,
            args.mathlib_root,
            args.output,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "riemann-qwen35-4b-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_riemann_qwen35_4b_assessment(
            Phase1Config.load(args.config),
            _project_root(),
            args.domain_config,
            args.mathlib_root,
            args.preflight,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "riemann-qwen35-4b-evidence":
        evidence = write_riemann_qwen35_4b_evidence(
            Phase1Config.load(args.config),
            _project_root(),
            args.domain_config,
            args.preflight,
            args.artifact_dir,
            args.evidence_dir,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "riemann-deepseek-preflight":
        evidence = run_riemann_qwen35_4b_preflight(
            Phase1Config.load(args.config),
            _project_root(),
            args.domain_config,
            args.mathlib_root,
            args.output,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "riemann-deepseek-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_riemann_qwen35_4b_assessment(
            Phase1Config.load(args.config),
            _project_root(),
            args.domain_config,
            args.mathlib_root,
            args.preflight,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "riemann-deepseek-evidence":
        references = {"qwen35-4b-base": args.qwen35_4b_outcomes}
        unavailable = {}
        if args.qwen35_9b_outcomes is None:
            unavailable["qwen35-9b-base"] = (
                "No accepted Qwen3.5-9B-Base Riemann task-outcome artifact is "
                "present on this issue's authoritative independent base; unmerged "
                "sibling results are not imported."
            )
        else:
            references["qwen35-9b-base"] = args.qwen35_9b_outcomes
        evidence = write_riemann_qwen35_4b_evidence(
            Phase1Config.load(args.config),
            _project_root(),
            args.domain_config,
            args.preflight,
            args.artifact_dir,
            args.evidence_dir,
            paired_reference_paths=references,
            unavailable_paired_references=unavailable,
            execution_limitations=args.execution_limitation,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-preflight":
        summary = run_qwen35_base_preflight(
            Phase1Config.load(args.config), args.output
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "qwen35-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        config = Phase1Config.load(args.config)
        timeout = (
            float(config.value["verifier"]["timeout_seconds"])
            if args.timeout is None
            else args.timeout
        )
        _, _, summary = run_qwen35_base_assessment(
            config,
            args.benchmark_root,
            args.workload,
            args.output_dir,
            timeout_seconds=timeout,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-evidence":
        outputs = write_qwen35_base_evidence(
            Phase1Config.load(args.config),
            args.preflight,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
        )
        print(json.dumps(outputs, indent=2))
        return 0

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

    if args.command == "ministral3-8b-base-preflight":
        evidence = run_ministral3_preflight(
            Ministral3AssessmentConfig.load(args.config),
            args.benchmark_root,
            args.model_snapshot,
            args.output,
        )
        print(json.dumps(evidence, indent=2))
        return 0 if evidence["status"] == "passed" else 1

    if args.command == "ministral3-8b-base-assess":
        _, _, summary = run_ministral3_assessment(
            Ministral3AssessmentConfig.load(args.config),
            args.benchmark_root,
            args.model_snapshot,
            args.preflight,
            args.workload,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "ministral3-8b-base-reverify":
        _, _, summary = reverify_ministral3_assessment(
            Ministral3AssessmentConfig.load(args.config),
            args.benchmark_root,
            args.input_dir,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "ministral3-8b-base-evidence":
        evidence = write_ministral3_evidence(
            Ministral3AssessmentConfig.load(args.config),
            args.preflight,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
        )
        print(json.dumps(evidence, indent=2))
        return 0

    if args.command == "qwen35-9b-base-preflight":
        evidence = run_qwen35_9b_base_preflight(
            Qwen35BaseAssessmentConfig.load(args.config),
            args.benchmark_root,
            args.model_snapshot,
            args.output,
        )
        print(json.dumps(evidence, indent=2))
        return 0 if evidence["status"] == "passed" else 1

    if args.command == "qwen35-9b-base-assess":
        _, _, summary = run_qwen35_9b_base_assessment(
            Qwen35BaseAssessmentConfig.load(args.config),
            args.benchmark_root,
            args.model_snapshot,
            args.preflight,
            args.workload,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-9b-base-evidence":
        evidence = write_qwen35_9b_base_evidence(
            Qwen35BaseAssessmentConfig.load(args.config),
            args.preflight,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
        )
        print(json.dumps(evidence["full"], indent=2))
        return 0

    if args.command == "qwen35-9b-riemann-preflight":
        evidence = run_riemann_preflight(
            RiemannAssessmentConfig.load(args.config),
            args.repository_root,
            args.mathlib_root,
            args.lean_environment_root,
            args.model_snapshot,
            args.output,
        )
        print(json.dumps(evidence, indent=2))
        return 0 if evidence["status"] == "passed" else 1

    if args.command == "qwen35-9b-riemann-generate":
        generation = run_riemann_generation(
            RiemannAssessmentConfig.load(args.config),
            args.repository_root,
            args.model_snapshot,
            args.preflight,
            args.output_dir,
        )
        print(json.dumps(generation, indent=2))
        return 0

    if args.command == "qwen35-9b-riemann-verify":
        summary = run_riemann_verification(
            RiemannAssessmentConfig.load(args.config),
            args.repository_root,
            args.mathlib_root,
            args.lean_environment_root,
            args.preflight,
            args.generation_dir,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-9b-riemann-evidence":
        evidence = write_riemann_evidence(
            RiemannAssessmentConfig.load(args.config),
            args.repository_root,
            args.domain_config,
            args.preflight,
            args.generation_dir,
            args.artifact_dir,
            args.evidence_dir,
        )
        print(json.dumps(evidence["full"], indent=2))
        return 0

    if args.command == "qwen35-2b-preflight":
        preflight = run_qwen35_posttrained_preflight(
            Qwen35AssessmentConfig.load(args.config),
            args.benchmark_root,
            args.output_dir,
        )
        print(json.dumps(preflight, indent=2))
        return 0

    if args.command == "qwen35-2b-assess":
        if args.verification_workers < 1:
            print("--verification-workers must be positive")
            return 2
        _, _, summary = run_qwen35_posttrained_assessment(
            Qwen35AssessmentConfig.load(args.config),
            args.benchmark_root,
            args.workload,
            args.preflight_dir,
            args.output_dir,
            verification_workers=args.verification_workers,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "qwen35-2b-evidence":
        comparison = write_qwen35_posttrained_evidence(
            Qwen35AssessmentConfig.load(args.config),
            args.preflight_dir,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
            args.reference_sft_evidence,
        )
        print(json.dumps(comparison, indent=2))
        return 0

    if args.command == "goedel-preflight":
        value = run_goedel_preflight(
            Phase1Config.load(args.config),
            args.benchmark_root,
            args.model_snapshot,
            args.output,
        )
        print(json.dumps(value, indent=2))
        return 0

    if args.command == "goedel-assess":
        _, _, summary = run_goedel_assessment(
            Phase1Config.load(args.config),
            args.benchmark_root,
            args.preflight,
            args.workload,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["complete"] else 1

    if args.command == "goedel-evidence":
        comparison = write_goedel_evidence(
            Phase1Config.load(args.config),
            args.preflight,
            args.dev16_dir,
            args.full_dir,
            args.evidence_dir,
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
