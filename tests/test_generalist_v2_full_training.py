from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.generalist_v2 import LORA_TARGET_REGEX
from qwen_lean.generalist_v2_full_training import (
    EXPECTED_QUARTER_STEPS,
    validate_quarter_checkpoint_inventory,
)


def _write_checkpoint(root: Path, checkpoint_id: str, step: int) -> None:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-4B-Base",
                "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": LORA_TARGET_REGEX,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(
        f"adapter-{checkpoint_id}".encode()
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        (checkpoint / name).write_bytes(name.encode())


def test_quarter_checkpoint_inventory_requires_exact_reloadable_adapters(
    tmp_path: Path,
) -> None:
    for checkpoint_id, step in EXPECTED_QUARTER_STEPS.items():
        _write_checkpoint(tmp_path, checkpoint_id, step)
    trajectory = {"checkpoint_optimizer_steps": EXPECTED_QUARTER_STEPS}

    inventory = validate_quarter_checkpoint_inventory(tmp_path, trajectory)

    assert set(inventory) == set(EXPECTED_QUARTER_STEPS)
    assert inventory["Q4"]["optimizer_step"] == 22852
    assert inventory["Q4"]["adapter_only"] is True
    (tmp_path / "checkpoint-11426/tokenizer.json").unlink()
    with pytest.raises(ValueError, match="Q2 is incomplete"):
        validate_quarter_checkpoint_inventory(tmp_path, trajectory)


def test_quarter_checkpoint_inventory_rejects_different_boundaries(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="quarter boundaries differ"):
        validate_quarter_checkpoint_inventory(
            tmp_path, {"checkpoint_optimizer_steps": {"Q1": 1}}
        )
