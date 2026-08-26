from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.generalist_v3 import GeneralistV3Config
from qwen_lean.generalist_v3_parity import (
    PARITY_EVIDENCE_SCHEMA_VERSION,
    PARITY_GATE_ID,
    validate_lora_parity_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v3_parity_gate_requires_every_requirement(tmp_path: Path) -> None:
    config = GeneralistV3Config.load(ROOT / "config/qwen35-4b-generalist-v3.json")
    path = tmp_path / "parity.json"
    value = {
        "schema_version": PARITY_EVIDENCE_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "status": "passed",
        "model": config.model,
        "target_regex": config.lora["target_regex"],
        "adapter": {"adapter_model_sha256": "a" * 64},
        "requirements": {"static": True, "functional": True},
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validate_lora_parity_gate(config, path)["status"] == "passed"
    value["requirements"]["functional"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed"):
        validate_lora_parity_gate(config, path)
