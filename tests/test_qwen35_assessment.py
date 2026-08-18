import json
from pathlib import Path

import pytest

from qwen_lean.baseline import vllm_engine_kwargs
from qwen_lean.minif2f import Phase1Config
from qwen_lean.qwen35_assessment import (
    BF16_LANE,
    FALLBACK_LANE,
    PREFLIGHT_SCHEMA_VERSION,
    _load_preflight_state,
    _numeric_summary,
    config_for_lane,
    validate_assessment_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> Phase1Config:
    return Phase1Config.load(ROOT / "config/qwen35-9b-assessment.json")


def test_qwen35_contract_freezes_strict_casting_and_precision_lanes() -> None:
    config = _config()

    validate_assessment_contract(config)
    bf16 = config_for_lane(config, BF16_LANE)
    fallback = config_for_lane(config, FALLBACK_LANE)

    assert bf16.engine["quantization"] is None
    assert fallback.engine["quantization"] == "bitsandbytes"
    assert fallback.sampling["candidates_per_task"] == 4
    assert fallback.sampling["max_new_tokens"] == 1024
    assert fallback.value["qwen35_assessment"]["chat_template"] is None


def test_qwen35_vllm_kwargs_disable_multimodal_loading_and_cpu_offload() -> None:
    config = config_for_lane(_config(), FALLBACK_LANE)

    kwargs = vllm_engine_kwargs(config, config.sampling, None)

    assert kwargs["language_model_only"] is True
    assert kwargs["cpu_offload_gb"] == 0.0
    assert kwargs["quantization"] == "bitsandbytes"
    assert kwargs["max_model_len"] == 2048


def test_four_bit_fallback_requires_one_recorded_failed_bf16_attempt(
    tmp_path: Path,
) -> None:
    config = _config()
    output = tmp_path / "preflight.json"
    state = _load_preflight_state(config, output, BF16_LANE)
    state.update(
        {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "failed",
            "attempts": [{"lane": BF16_LANE, "status": "failed"}],
            "selected_lane": None,
        }
    )
    output.write_text(json.dumps(state), encoding="utf-8")

    resumed = _load_preflight_state(config, output, FALLBACK_LANE)

    assert resumed["attempts"] == [{"lane": BF16_LANE, "status": "failed"}]


def test_four_bit_fallback_rejects_missing_bf16_attempt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed BF16"):
        _load_preflight_state(_config(), tmp_path / "missing.json", FALLBACK_LANE)


def test_numeric_summary_records_totals_and_percentiles() -> None:
    assert _numeric_summary([1, 2, 3, 4]) == {
        "count": 4,
        "total": 10,
        "minimum": 1,
        "mean": 2.5,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "maximum": 4,
    }
