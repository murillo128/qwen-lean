from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from qwen_lean.baseline import vllm_engine_kwargs
from qwen_lean.minif2f import Phase1Config
from qwen_lean.qwen35_4b_base_assessment import (
    run_assessment,
    validate_assessment_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qwen35_assessment_contract_is_pinned() -> None:
    config = Phase1Config.load(ROOT / "config/qwen35-4b-base-assessment.json")

    validate_assessment_config(config)

    assert config.sampling["candidates_per_task"] == 4
    assert config.engine["dtype"] == "bfloat16"
    assert config.engine["language_model_only"] is True


def test_qwen35_engine_uses_text_only_compatibility_lane(monkeypatch) -> None:
    config = Phase1Config.load(ROOT / "config/qwen35-4b-base-assessment.json")
    snapshot = "/cache/qwen35-pinned"
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_: snapshot),
    )

    kwargs = vllm_engine_kwargs(config, config.sampling, None)

    assert kwargs["language_model_only"] is True
    assert kwargs["model"] == snapshot
    assert kwargs["tokenizer"] == snapshot
    assert kwargs["trust_remote_code"] is False
    assert kwargs["revision"] == config.model["model_revision"]
    assert kwargs["tokenizer_revision"] == config.model["tokenizer_revision"]


def test_qwen35_assessment_rejects_verifier_timeout_override(tmp_path: Path) -> None:
    config = Phase1Config.load(ROOT / "config/qwen35-4b-base-assessment.json")

    with pytest.raises(ValueError, match="verifier timeout mismatch"):
        run_assessment(
            config,
            tmp_path,
            "minif2f-valid-dev16-v1",
            tmp_path / "output",
            timeout_seconds=31.0,
            verification_workers=1,
        )
