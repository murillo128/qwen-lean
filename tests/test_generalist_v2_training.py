from __future__ import annotations

from pathlib import Path
import pytest

from qwen_lean.generalist_v2 import (
    EXPECTED_LORA_MODULE_COUNTS,
    GeneralistV2Config,
)
from qwen_lean.generalist_v2_training import (
    choose_precision_lane,
    inspect_lora_targets,
    lora_target_summary,
    pad_weighted_target_only_batch,
    statement_weighted_causal_loss,
)


ROOT = Path(__file__).resolve().parents[1]


class _Linear:
    in_features = 64
    out_features = 96


class _Model:
    def __init__(self, *, vision_lookalike: bool = False):
        modules: list[tuple[str, object]] = [("", self)]
        full_layers = set(range(3, 32, 4))
        for layer in range(32):
            if layer in full_layers:
                for suffix in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    modules.append(
                        (f"model.layers.{layer}.self_attn.{suffix}", _Linear())
                    )
            else:
                for suffix in (
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_a",
                    "in_proj_b",
                    "out_proj",
                ):
                    modules.append(
                        (f"model.layers.{layer}.linear_attn.{suffix}", _Linear())
                    )
            for suffix in ("gate_proj", "up_proj", "down_proj"):
                modules.append((f"model.layers.{layer}.mlp.{suffix}", _Linear()))
        if vision_lookalike:
            modules.append(("model.visual.layers.0.q_proj", _Linear()))
        self._modules = modules

    def named_modules(self):
        return iter(self._modules)


def test_architecture_matcher_covers_all_three_text_families() -> None:
    matches = inspect_lora_targets(_Model())
    summary = lora_target_summary(matches)

    assert summary["matched_module_count"] == sum(EXPECTED_LORA_MODULE_COUNTS.values())
    assert summary["module_counts_by_suffix"] == dict(
        sorted(EXPECTED_LORA_MODULE_COUNTS.items())
    )
    assert summary["module_counts_by_family"] == {
        "full_attention": 32,
        "gated_deltanet": 120,
        "mlp": 96,
    }
    assert summary["trainable_lora_parameter_count"] == len(matches) * 16 * 160
    assert summary["vision_modules_matched"] == 0


def test_architecture_matcher_rejects_target_like_vision_module() -> None:
    with pytest.raises(RuntimeError, match="outside the text-decoder constraint"):
        inspect_lora_targets(_Model(vision_lookalike=True))


def test_precision_lane_uses_48_gib_gate_and_automatic_qlora_fallback() -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")

    assert (
        choose_precision_lane(config, device_total_memory_bytes=48 * 1024**3)
        == "bf16-lora"
    )
    assert (
        choose_precision_lane(config, device_total_memory_bytes=20 * 1024**3)
        == "nf4-qlora"
    )


def test_weighted_collator_masks_padding_and_preserves_example_weights() -> None:
    batch = pad_weighted_target_only_batch(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "attention_mask": [1, 1, 1],
                "example_weight": 0.25,
            },
            {
                "input_ids": [4, 5],
                "labels": [-100, 5],
                "attention_mask": [1, 1],
                "example_weight": 1.75,
            },
        ],
        pad_token_id=0,
    )

    assert batch == {
        "input_ids": [[1, 2, 3], [4, 5, 0]],
        "labels": [[-100, 2, 3], [-100, 5, -100]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
        "example_weight": [0.25, 1.75],
    }


def test_weighted_causal_loss_keeps_example_weight_across_micro_batches() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [[5.0, -5.0], [5.0, -5.0], [0.0, 0.0]],
            [[5.0, -5.0], [5.0, -5.0], [0.0, 0.0]],
        ]
    )
    labels = torch.tensor([[-100, 0, 0], [-100, 1, 1]])
    weights = torch.tensor([1.0, 3.0])

    loss = statement_weighted_causal_loss(
        logits, labels, weights, weight_normalizer=2.0
    )
    easy = torch.nn.functional.cross_entropy(
        logits[0, :2], labels[0, 1:], reduction="mean"
    )
    hard = torch.nn.functional.cross_entropy(
        logits[1, :2], labels[1, 1:], reduction="mean"
    )

    assert loss == pytest.approx(float((easy + 3 * hard) / 4))
