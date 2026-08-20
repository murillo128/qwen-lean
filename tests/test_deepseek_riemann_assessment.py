import hashlib
import json
from pathlib import Path

import pytest

from qwen_lean.minif2f import Phase1Config
from qwen_lean.riemann_assessment import (
    DEEPSEEK_PROFILE,
    EXPECTED_TASKS,
    _exact_mcnemar,
    _paired_analysis,
    assessment_profile,
    load_domain_config,
    load_validation_workload,
    validate_assessment_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/deepseek-prover-v2-7b-riemann-assessment.json"
DOMAIN_CONFIG_PATH = ROOT / "config/riemann-domain-breakdown.json"
QWEN4_OUTCOMES = ROOT / "evidence/riemann-qwen35-4b-base/task-outcomes.jsonl"


def test_config_freezes_issue_76_specialist_parent_contract() -> None:
    config = Phase1Config.load(CONFIG_PATH)

    validate_assessment_config(config)
    assert assessment_profile(config) is DEEPSEEK_PROFILE
    assert config.model == {
        "model_id": "deepseek-ai/DeepSeek-Prover-V2-7B",
        "model_revision": "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b",
        "tokenizer_id": "deepseek-ai/DeepSeek-Prover-V2-7B",
        "tokenizer_revision": "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b",
        "chat_template": None,
    }
    assert config.engine["version"] == "0.10.2"
    assert config.engine["dtype"] == "bfloat16"
    assert config.engine["quantization"] is None
    assert "language_model_only" not in config.engine
    assert config.value["assessment"]["selection_role"] == (
        "specialist-parent-comparator"
    )


def test_deepseek_config_rejects_qwen_runtime_substitution(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["engine"]["language_model_only"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the accepted DeepSeek"):
        validate_assessment_config(Phase1Config.load(path))


def test_deepseek_uses_exact_frozen_workload_and_qwen4_pairing() -> None:
    config = Phase1Config.load(CONFIG_PATH)
    domains = load_domain_config(DOMAIN_CONFIG_PATH)
    records, _ = load_validation_workload(config, ROOT, domains)
    assert len(records) == EXPECTED_TASKS

    current = [{"task_id": record.id, "solved": False} for record in records]
    paired = _paired_analysis(
        current,
        {"qwen35-4b-base": QWEN4_OUTCOMES},
        {
            "qwen35-9b-base": (
                "No accepted independent-base task outcomes are available."
            )
        },
    )

    qwen4 = paired["qwen35-4b-base"]
    assert qwen4["contingency"] == {
        "both_solved": 0,
        "current_only": 0,
        "reference_only": 13,
        "neither_solved": 543,
    }
    assert qwen4["exact_mcnemar_two_sided_p"] == 0.000244140625
    assert paired["qwen35-9b-base"]["status"] == "unavailable"
    assert _exact_mcnemar(0, 0) == 1.0


def test_pairing_accepts_committed_qwen9_compact_schema(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    domains = load_domain_config(DOMAIN_CONFIG_PATH)
    records, _ = load_validation_workload(config, ROOT, domains)
    current = [{"task_id": record.id, "solved": False} for record in records]

    reference_dir = tmp_path / "qwen35-9b-riemann"
    reference_dir.mkdir()
    outcomes = QWEN4_OUTCOMES.read_bytes()
    outcomes_path = reference_dir / "task-outcomes.jsonl"
    outcomes_path.write_bytes(outcomes)
    (reference_dir / "full.json").write_text(
        json.dumps(
            {
                "workload_id": "riemann-specialist-validation-v1",
                "task_count": 556,
                "candidate_count": 2224,
                "infrastructure_error_count": 0,
                "lane_id": "bf16-text-only-v1",
                "precision": "bfloat16",
                "quantization": None,
                "task_outcomes": {
                    "rows": 556,
                    "sha256": hashlib.sha256(outcomes).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    (reference_dir / "preflight.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "accepted_lane": "bf16-text-only-v1",
                "prompt_format_id": "whole-proof-v1",
                "chat_template": None,
                "prompt_transformation": None,
                "model": {
                    "model_id": "Qwen/Qwen3.5-9B-Base",
                    "model_revision": (
                        "68c46c4b3498877f3ef123c856ecfde50c39f404"
                    ),
                    "tokenizer_id": "Qwen/Qwen3.5-9B-Base",
                    "tokenizer_revision": (
                        "68c46c4b3498877f3ef123c856ecfde50c39f404"
                    ),
                },
                "workload": {
                    "corpus_id": "riemann-specialist-validation-v1",
                    "loaded_task_count": 556,
                },
            }
        ),
        encoding="utf-8",
    )

    paired = _paired_analysis(
        current,
        {"qwen35-9b-base": outcomes_path},
        {},
    )

    assert paired["qwen35-9b-base"]["status"] == "available"
    assert paired["qwen35-9b-base"]["contingency"] == {
        "both_solved": 0,
        "current_only": 0,
        "reference_only": 13,
        "neither_solved": 543,
    }
