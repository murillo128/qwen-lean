from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.generalist_v2 import GeneralistV2Config
from qwen_lean.generalist_v2_evaluation import run_checkpoint_generation
from qwen_lean.generalist_v2_parity import (
    PARITY_REQUIRED_GATES,
    _arm_pair_summary,
    validate_lora_parity_gate,
)
from qwen_lean.qwen35_vllm_lora import qwen35_vllm_runtime_tensor_key

ROOT = Path(__file__).resolve().parents[1]


def test_qwen35_vllm_runtime_key_adds_text_wrapper_namespace() -> None:
    source = (
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"
    )
    assert qwen35_vllm_runtime_tensor_key(source) == (
        "base_model.model.model.language_model.layers.0.linear_attn."
        "in_proj_qkv.lora_A.weight"
    )
    with pytest.raises(ValueError, match="unexpected Qwen3.5 PEFT tensor name"):
        qwen35_vllm_runtime_tensor_key("base_model.model.visual.lora_A.weight")


def _adapter(tmp_path: Path, config: GeneralistV2Config) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": config.model["model_id"],
                "revision": config.model["model_revision"],
                "r": config.lora["r"],
                "target_modules": config.lora["target_regex"],
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    return adapter


def _passed_gate(tmp_path: Path, config: GeneralistV2Config) -> Path:
    path = tmp_path / "parity.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-lora-parity-evidence-v1",
                "gate_id": "qwen35-vllm-lora-parity-v1",
                "status": "passed",
                "model": {
                    "model_id": config.model["model_id"],
                    "model_revision": config.model["model_revision"],
                },
                "vllm": {
                    "version": "0.27.2rc1.dev203+g41f179b57",
                    "source_revision": "41f179b57aa8ab6f634f508128ce1f1efadd0eb1",
                },
                "target_regex": config.lora["target_regex"],
                "requirements": {
                    name: True for name in sorted(PARITY_REQUIRED_GATES)
                },
                "adapters": {
                    "Q2": {"adapter_model_sha256": "q2-adapter-hash"}
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parity_gate_rejects_any_partial_static_or_functional_requirement(
    tmp_path: Path,
) -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    gate = _passed_gate(tmp_path, config)

    accepted = validate_lora_parity_gate(config, gate)
    assert accepted["status"] == "passed"

    value = json.loads(gate.read_text(encoding="utf-8"))
    value["requirements"]["static_q2_complete"] = False
    gate.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed"):
        validate_lora_parity_gate(config, gate)

    value["requirements"]["static_q2_complete"] = True
    del value["requirements"]["static_overfit64_complete"]
    gate.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed"):
        validate_lora_parity_gate(config, gate)


def test_qwen35_adapter_generation_cannot_start_without_parity_gate(
    tmp_path: Path,
) -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    adapter = _adapter(tmp_path, config)

    with pytest.raises(ValueError, match="requires the passed LoRA parity gate"):
        run_checkpoint_generation(
            config,
            ROOT / "config/qwen35-4b-base-assessment.json",
            "fresh-composition-valid-v2",
            tmp_path / "package",
            tmp_path / "views",
            tmp_path / "output",
            checkpoint_id="Q2",
            adapter_dir=adapter,
        )


def test_arm_pair_records_output_and_logprob_adapter_effects() -> None:
    base = {
        "a": {
            "normalized_text_sha256": "same",
            "first_token_id": 1,
            "first_token_logprob": -1.0,
        },
        "b": {
            "normalized_text_sha256": "base",
            "first_token_id": 2,
            "first_token_logprob": -2.0,
        },
    }
    adapter = {
        "a": {
            "normalized_text_sha256": "same",
            "first_token_id": 1,
            "first_token_logprob": -0.5,
        },
        "b": {
            "normalized_text_sha256": "adapter",
            "first_token_id": 3,
            "first_token_logprob": -0.1,
        },
    }

    summary = _arm_pair_summary(base, adapter)

    assert summary["output_difference_count"] == 1
    assert summary["same_first_token_logprob_comparison_count"] == 1
    assert summary["maximum_same_first_token_logprob_delta"] == 0.5
