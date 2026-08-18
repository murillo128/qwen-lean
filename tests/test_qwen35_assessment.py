import json
from pathlib import Path

import pytest

from qwen_lean.qwen35_assessment import (
    ENGINE_VERSION,
    EXPECTED_SAMPLING,
    MODEL_ID,
    MODEL_REVISION,
    GpuMemoryMonitor,
    Qwen35AssessmentConfig,
    _validate_preflight_evidence,
    generated_token_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-2b-assessment.json"


def test_config_freezes_strict_cross_model_contract() -> None:
    config = Qwen35AssessmentConfig.load(CONFIG_PATH)

    assert config.phase1.model == {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "add_special_tokens": False,
        "chat_template": None,
    }
    assert config.phase1.sampling == EXPECTED_SAMPLING
    assert config.phase1.engine["version"] == ENGINE_VERSION
    assert config.phase1.engine["dtype"] == "bfloat16"
    assert config.phase1.engine["quantization"] is None
    assert config.phase1.engine["language_model_only"] is True
    assert config.phase1.engine["use_flashinfer_sampler"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "model_revision", "main"),
        ("model", "chat_template", "native"),
        ("sampling", "candidates_per_task", 8),
        ("sampling", "top_k", 20),
        ("engine", "dtype", "float16"),
    ],
)
def test_config_rejects_contract_drift(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contract differs"):
        Qwen35AssessmentConfig.load(path)


def test_generated_token_summary_retains_total_and_distribution() -> None:
    assert generated_token_summary([1, 2, 3, 4]) == {
        "total": 10,
        "minimum": 1,
        "maximum": 4,
        "mean": 2.5,
        "median": 2.5,
        "p95_nearest_rank": 4,
    }


def test_gpu_memory_monitor_records_peak_and_increment() -> None:
    observations = iter([(100, 1000), (350, 1000)])
    monitor = GpuMemoryMonitor(
        0,
        interval_seconds=60,
        query=lambda _: next(observations),
    )

    with monitor:
        pass

    assert monitor.to_dict() == {
        "device_index": 0,
        "baseline_used_mib": 100,
        "peak_used_mib": 350,
        "peak_incremental_mib": 250,
        "device_total_mib": 1000,
        "sample_interval_seconds": 60,
        "sample_count": 2,
    }


def test_committed_preflight_retains_required_runtime_contract() -> None:
    preflight = json.loads(
        (ROOT / "evidence/qwen35-2b/preflight.json").read_text(encoding="utf-8")
    )

    _validate_preflight_evidence(preflight)


def test_preflight_rejects_nonlocal_runtime() -> None:
    preflight = json.loads(
        (ROOT / "evidence/qwen35-2b/preflight.json").read_text(encoding="utf-8")
    )
    preflight["runtime"]["inference_execution"] = "hosted"

    with pytest.raises(ValueError, match="local Ada BF16"):
        _validate_preflight_evidence(preflight)
