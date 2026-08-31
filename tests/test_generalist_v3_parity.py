from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from qwen_lean.generalist_v3 import GeneralistV3Config
from qwen_lean.generalist_v3_parity import (
    PARITY_EVIDENCE_SCHEMA_VERSION,
    PARITY_GATE_ID,
    validate_lora_parity_gate,
)
from qwen_lean.qwen35_vllm_lora import (
    GDN_QKV_OUTPUT_SIZES,
    qwen35_vllm_runtime_tensor_items,
)


ROOT = Path(__file__).resolve().parents[1]


def test_vllm_017_gdn_qkv_transform_is_exact() -> None:
    prefix = "base_model.model.model.layers.0.linear_attn.in_proj_qkv"
    lora_a = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    lora_b = torch.arange(
        sum(GDN_QKV_OUTPUT_SIZES) * 16, dtype=torch.float32
    ).reshape(sum(GDN_QKV_OUTPUT_SIZES), 16)
    a_items = qwen35_vllm_runtime_tensor_items(
        f"{prefix}.lora_A.weight", lora_a, split_gdn_qkv=True
    )
    b_items = qwen35_vllm_runtime_tensor_items(
        f"{prefix}.lora_B.weight", lora_b, split_gdn_qkv=True
    )
    assert [key.rsplit(".", 3)[-3] for key, _ in a_items] == [
        "in_proj_q",
        "in_proj_k",
        "in_proj_v",
    ]
    assert all(torch.equal(tensor, lora_a) for _, tensor in a_items)
    assert [tuple(tensor.shape) for _, tensor in b_items] == [
        (2048, 16),
        (2048, 16),
        (4096, 16),
    ]
    assert torch.equal(torch.cat([tensor for _, tensor in b_items]), lora_b)


def test_vllm_017_gdn_qkv_transform_rejects_wrong_shape() -> None:
    key = (
        "base_model.model.model.layers.0.linear_attn."
        "in_proj_qkv.lora_B.weight"
    )
    with pytest.raises(ValueError, match="QKV LoRA B shape differs"):
        qwen35_vllm_runtime_tensor_items(
            key, torch.zeros(10, 16), split_gdn_qkv=True
        )


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
