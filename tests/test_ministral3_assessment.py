from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_lean.ministral3_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    Ministral3AssessmentConfig,
    _convert_outputs,
    _is_memory_failure,
    _validate_preflight,
    validate_model_snapshot,
    vllm_engine_kwargs,
    vllm_sampling_kwargs,
)
from qwen_lean.schema import TaskRecord


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ministral3-8b-base-assessment.json"


def test_config_freezes_issue_52_contract() -> None:
    config = Ministral3AssessmentConfig.load(CONFIG)

    assert config.model == {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
    }
    assert config.sampling == {
        "candidates_per_task": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_new_tokens": 1024,
        "stop": "tokenizer_eos_or_token_limit",
        "seed": 0,
    }
    assert config.runtime["expected_cuda_device_name"] == (
        "NVIDIA RTX 4000 Ada Generation"
    )
    assert config.value["bf16_lane"]["cpu_offload_gb"] == 0
    assert config.value["fallback_lane"]["quantization_metadata"] == {
        "bits": 4,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "bfloat16",
        "conversion": "vllm online from pinned BF16 safetensors",
        "prequantized_checkpoint": False,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("sampling", "temperature"), 0.7),
        (("sampling", "candidates_per_task"), 8),
        (("model", "model_revision"), "main"),
        (("bf16_lane", "cpu_offload_gb"), 1),
        (("fallback_lane", "quantization"), "awq"),
    ],
)
def test_config_rejects_contract_drift(
    tmp_path: Path, path: tuple[str, str], value: object
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = value
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        Ministral3AssessmentConfig.load(candidate)


def test_snapshot_validation_requires_pinned_complete_shards(tmp_path: Path) -> None:
    config = Ministral3AssessmentConfig.load(CONFIG)
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing weight shards"):
        validate_model_snapshot(config, snapshot)
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"fixture")

    assert validate_model_snapshot(config, snapshot) == snapshot.resolve()


def test_vllm_arguments_preserve_raw_text_only_lane(tmp_path: Path) -> None:
    config = Ministral3AssessmentConfig.load(CONFIG)
    kwargs = vllm_engine_kwargs(config, tmp_path, config.value["fallback_lane"])

    assert kwargs["model"] == str(tmp_path)
    assert kwargs["tokenizer"] == str(tmp_path)
    assert kwargs["tokenizer_mode"] == "mistral"
    assert kwargs["language_model_only"] is True
    assert kwargs["cpu_offload_gb"] == 0.0
    assert kwargs["quantization"] == "bitsandbytes"
    assert kwargs["load_format"] == "bitsandbytes"
    assert kwargs["generation_config"] == "vllm"
    assert kwargs["mm_processor_cache_gb"] == 0
    assert kwargs["trust_remote_code"] is False
    assert vllm_sampling_kwargs(config.sampling) == {
        "n": 4,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_tokens": 1024,
        "seed": 0,
        "ignore_eos": False,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
    }


def test_quantized_preflight_requires_bf16_memory_failure(tmp_path: Path) -> None:
    config = Ministral3AssessmentConfig.load(CONFIG)
    snapshot = tmp_path / MODEL_REVISION
    preflight = {
        "schema_version": "ministral3-8b-base-preflight-v1",
        "status": "passed",
        "config_sha256": config.digest(),
        "model": config.model,
        "model_snapshot": str(snapshot),
        "accepted_lane": "bitsandbytes-nf4-online-v1",
        "bf16_attempt": {"status": "failed", "memory_failure": False},
        "fallback_attempt": {"status": "passed"},
    }

    with pytest.raises(ValueError, match="memory failure"):
        _validate_preflight(config, preflight, snapshot)
    preflight["bf16_attempt"]["memory_failure"] = True
    _validate_preflight(config, preflight, snapshot)


def test_convert_outputs_keeps_unmodified_continuations() -> None:
    task = TaskRecord(
        id="task",
        preamble="import Mathlib",
        declaration="theorem task : True",
        declaration_name="task",
    )
    prompt = "exact prompt"
    texts = ["  exact True.intro\n", "by\n  trivial", "```lean\ntrivial\n```", ""]
    outputs = [
        SimpleNamespace(
            prompt=prompt,
            metrics=None,
            outputs=[
                SimpleNamespace(
                    index=index,
                    text=text,
                    token_ids=list(range(index + 1)),
                    finish_reason="stop" if index < 3 else "length",
                )
                for index, text in enumerate(texts)
            ],
        )
    ]

    generated = _convert_outputs(
        [task], [prompt], outputs, 4.0, candidates_per_task=4
    )

    assert [item.text for item in generated] == texts
    assert [item.finish_reason for item in generated] == [
        "eos",
        "eos",
        "eos",
        "token_limit",
    ]
    assert [item.token_count for item in generated] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory",
        "No available memory for the cache blocks",
        "Free memory on device is insufficient",
    ],
)
def test_memory_failures_are_classified(message: str) -> None:
    assert _is_memory_failure(RuntimeError(message), message)


def test_config_digest_matches_file_bytes() -> None:
    config = Ministral3AssessmentConfig.load(CONFIG)
    assert config.digest() == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
