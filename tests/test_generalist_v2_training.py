from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest

from qwen_lean.generalist_v2 import (
    EXPECTED_LORA_MODULE_COUNTS,
    GeneralistProofVariant,
    GeneralistV2Config,
    WeightedTokenizedExample,
)
from qwen_lean.generalist_v2_dataset import DATASET_BINDING_SCHEMA_VERSION
from qwen_lean.generalist_v2_training import (
    GeneralistTrainingRuntime,
    build_weighted_sft_trainer,
    checkpointed_target_only_causal_loss,
    choose_precision_lane,
    configure_gradient_checkpointing,
    enable_sequence_chunked_mlp,
    inspect_gated_delta_rule_backend,
    inspect_lora_targets,
    lora_target_summary,
    pad_weighted_target_only_batch,
    resolve_near_maximum_variant,
    scale_single_example_causal_loss,
    select_overfit64_variants,
    select_smoke4096_variants,
    should_checkpoint_activations,
    should_offload_activations,
    statement_weighted_causal_loss,
    summarize_finite_optimizer_logs,
    summarize_overfit_curve,
    tokenize_weighted_training_selection,
    validate_bounded_training_evidence,
    validate_production_preflight_gate,
    validate_q0_training_gate,
)

ROOT = Path(__file__).resolve().parents[1]


class _Linear:
    in_features = 64
    out_features = 96


class _Tokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


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


def _variant(
    statement_id: str,
    proof_variant_id: str,
    *,
    synthetic: bool = False,
) -> GeneralistProofVariant:
    return GeneralistProofVariant(
        statement_id=statement_id,
        proof_variant_id=proof_variant_id,
        declaration_name="fixture",
        declaration="theorem fixture : True",
        completion="trivial",
        preamble="import Mathlib",
        split="train",
        optimizer_eligible=True,
        source_kind="synthetic" if synthetic else "real",
        generator_family="fixture-family" if synthetic else None,
        composition_class="direct" if synthetic else None,
        derivation_family_id=f"family-{statement_id}" if synthetic else None,
        domain_tags=(),
    )


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


def test_near_maximum_variant_resolves_exact_bound_training_row() -> None:
    records = [_variant("statement-1", "proof-1"), _variant("statement-2", "proof-2")]
    binding = {
        "schema_version": DATASET_BINDING_SCHEMA_VERSION,
        "serialization": {
            "lengths": {
                "selected_context_tokens": 32768,
                "maximum_variant": {
                    "statement_id": "statement-2",
                    "proof_variant_id": "proof-2",
                    "tokens": 19385,
                },
            }
        },
    }

    record, expected_tokens, context_tokens = resolve_near_maximum_variant(
        records, binding
    )

    assert record.proof_variant_id == "proof-2"
    assert expected_tokens == 19385
    assert context_tokens == 32768


def test_near_maximum_variant_rejects_non_binding_evidence() -> None:
    with pytest.raises(ValueError, match="needs Dataset-v2 binding"):
        resolve_near_maximum_variant([], {"schema_version": "wrong"})


def test_overfit64_selects_16_per_probe_stratum_and_all_variants() -> None:
    strata_names = (
        "real-generic",
        "real-prime-number-theory",
        "synthetic-generic-composition",
        "synthetic-prime-composition",
    )
    strata = {
        name: [f"{name}-{index:02d}" for index in range(64)] for name in strata_names
    }
    records = [
        _variant(
            statement_id,
            f"{statement_id}-proof-{variant}",
            synthetic=stratum.startswith("synthetic"),
        )
        for stratum, statement_ids in strata.items()
        for statement_id in statement_ids
        for variant in (1, 2)
    ]
    probe = {"id": "dataset-v2-train-probe", "strata": strata}

    first, metadata = select_overfit64_variants(records, probe)
    second, second_metadata = select_overfit64_variants(records, probe)

    assert [item.proof_variant_id for item in first] == [
        item.proof_variant_id for item in second
    ]
    assert metadata == second_metadata
    assert metadata["statement_count"] == 64
    assert metadata["proof_variant_count"] == 128
    assert metadata["statements_by_stratum"] == {name: 16 for name in strata_names}
    selected_statements = {item.statement_id for item in first}
    assert all(
        sum(item.statement_id == statement_id for item in first) == 2
        for statement_id in selected_statements
    )


def test_smoke4096_is_deterministic_and_keeps_full_statement_variants() -> None:
    records = [
        _variant(
            f"statement-{index:04d}",
            f"statement-{index:04d}-proof-{variant}",
            synthetic=index >= 4000,
        )
        for index in range(4100)
        for variant in (1, 2)
    ]

    first, metadata = select_smoke4096_variants(records)
    second, second_metadata = select_smoke4096_variants(records)

    assert [item.proof_variant_id for item in first] == [
        item.proof_variant_id for item in second
    ]
    assert metadata == second_metadata
    assert metadata["statement_count"] == 4096
    assert metadata["proof_variant_count"] == 8192
    assert set(metadata["statement_source_counts"]) == {"real", "synthetic"}
    selected_statements = {item.statement_id for item in first}
    assert all(
        sum(item.statement_id == statement_id for item in first) == 2
        for statement_id in selected_statements
    )


def test_selected_tokenization_uses_full_membership_weight_scale() -> None:
    real = [_variant(f"real-{index}", f"real-{index}-proof") for index in range(40)]
    synthetic = _variant("synthetic", "synthetic-proof", synthetic=True)
    examples, normalizer, metadata = tokenize_weighted_training_selection(
        [*real, synthetic],
        [synthetic],
        _Tokenizer(),
        maximum_sequence_tokens=32768,
    )

    assert len(examples) == 1
    assert examples[0].proof_variant_id == "synthetic-proof"
    assert examples[0].example_weight == pytest.approx(4.0)
    assert normalizer == pytest.approx(44 / 41)
    assert metadata["full_membership_weight_normalizer"] == pytest.approx(44 / 41)
    assert metadata["selected_example_weight"] == {
        "minimum": 4.0,
        "maximum": 4.0,
        "mean": 4.0,
    }
    assert metadata["truncated_or_dropped_variants"] == 0


def test_q0_gate_requires_all_complete_workloads(tmp_path: Path) -> None:
    expected = {
        "fresh-composition-valid-v2": 406,
        "minif2f-valid-clean-v2": 244,
        "dataset-v2-train-probe": 256,
        "riemann-fresh-valid-v2": 100,
    }
    evidence = {
        "schema_version": "generalist-v2-q0-evidence-v1",
        "checkpoint_id": "Q0",
        "model_id": "Qwen/Qwen3.5-4B-Base",
        "model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "candidates_per_task": 8,
        "selection_test_workloads_consulted": False,
        "riemann_used_for_selection": False,
        "workloads": {
            workload_id: {
                "task_count": task_count,
                "candidate_count": task_count * 8,
                "category_counts": {"generation_error": 0, "verifier_error": 0},
                "verified_counts": [0] * task_count,
            }
            for workload_id, task_count in expected.items()
        },
    }
    path = tmp_path / "q0.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    gate = validate_q0_training_gate(path)

    assert gate["workload_count"] == 4
    assert gate["complete_before_optimizer_update"] is True

    evidence["workloads"]["minif2f-valid-clean-v2"]["candidate_count"] -= 1
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="minif2f-valid-clean-v2"):
        validate_q0_training_gate(path)


def test_production_preflight_gate_is_bound_to_exact_q0(tmp_path: Path) -> None:
    evidence = {
        "schema_version": "generalist-v2-production-preflight-v1",
        "status": "passed",
        "model": {"model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b"},
        "selected_lane": "nf4-qlora",
        "dataset": {"binding_manifest_sha256": "a" * 64},
        "q0_gate": {"evidence_sha256": "q0-hash"},
        "update": {
            "loss_finite": True,
            "all_trainable_gradients_present": True,
            "all_gradients_finite": True,
            "adapter_parameter_changed": True,
            "frozen_parameter_unchanged": True,
            "only_intended_lora_parameters_trainable": True,
        },
        "runtime": {"headroom_passed": True},
    }
    path = tmp_path / "production.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    gate = validate_production_preflight_gate(path, {"evidence_sha256": "q0-hash"})

    assert gate["selected_lane"] == "nf4-qlora"
    assert gate["binding_manifest_sha256"] == "a" * 64
    assert gate["passed_before_training"] is True
    with pytest.raises(ValueError, match="passed production preflight"):
        validate_production_preflight_gate(path, {"evidence_sha256": "different-q0"})


def test_bounded_gate_requires_reload_evaluator_and_adapter(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    evidence = {
        "schema_version": "generalist-v2-bounded-training-v1",
        "status": "passed",
        "workload": {"workload_id": "generalist-v2-overfit64-v1"},
        "model": {"model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b"},
        "q0_gate": {"evidence_sha256": "q0-hash"},
        "production_preflight_gate": {"evidence_sha256": "preflight-hash"},
        "selected_lane": "nf4-qlora",
        "training": {
            "logs": {"covers_every_optimizer_step_exactly_once": True},
            "overfit_curve": {"strong_fit_loss_gate": True},
        },
        "adapter": {"path": "adapter"},
        "reload_generation_evaluator_gate": {
            "fresh_base_and_adapter_reload": True,
            "evaluator_infrastructure_errors": 0,
            "exact_target_count": 1,
            "lean_verified_count": 1,
        },
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    gate = validate_bounded_training_evidence(
        path,
        "generalist-v2-overfit64-v1",
        {"evidence_sha256": "q0-hash"},
        {"evidence_sha256": "preflight-hash"},
    )

    assert gate["passed_before_full_training"] is True
    evidence["reload_generation_evaluator_gate"]["lean_verified_count"] = 0
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="did not strongly fit"):
        validate_bounded_training_evidence(
            path,
            "generalist-v2-overfit64-v1",
            {"evidence_sha256": "q0-hash"},
            {"evidence_sha256": "preflight-hash"},
        )


def test_optimizer_log_summary_requires_every_finite_step() -> None:
    summary = summarize_finite_optimizer_logs(
        [
            {"step": 1, "loss": 2.0, "grad_norm": 1.0},
            {"step": 2, "loss": 1.0, "grad_norm": 0.5},
            {"step": 2, "train_runtime": 3.0},
        ],
        2,
    )

    assert summary["logged_optimizer_steps"] == 2
    assert summary["loss"] == {
        "minimum": 1.0,
        "maximum": 2.0,
        "mean": 1.5,
        "first": 2.0,
        "last": 1.0,
    }
    with pytest.raises(RuntimeError, match="every optimizer step"):
        summarize_finite_optimizer_logs([{"step": 2, "loss": 1.0, "grad_norm": 0.5}], 2)


def test_overfit_curve_compares_complete_matching_passes() -> None:
    history = [
        {"step": step, "loss": loss, "grad_norm": 1.0}
        for step, loss in enumerate(
            [4.0, 2.0, 1.0, 0.5, 0.08, 0.04, 0.02, 0.01], start=1
        )
    ]

    summary = summarize_overfit_curve(history, one_pass_optimizer_steps=4)

    assert summary["first_complete_pass_mean_loss"] == pytest.approx(1.875)
    assert summary["last_complete_pass_mean_loss"] == pytest.approx(0.0375)
    assert summary["strong_fit_loss_gate"] is True
    with pytest.raises(RuntimeError, match="did not strongly reduce"):
        summarize_overfit_curve(
            [
                {"step": step, "loss": loss}
                for step, loss in enumerate([1.0, 0.9, 0.8, 0.7], start=1)
            ],
            one_pass_optimizer_steps=2,
        )


def test_weighted_trainer_constructs_with_frozen_sequential_order(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real weighted trainer construction requires CUDA")
    pytest.importorskip("datasets")
    pytest.importorskip("trl")
    if importlib.metadata.version("trl") != "1.10.0":
        pytest.skip("trainer construction requires the isolated TRL 1.10 runtime")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    tokenizer_backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "proof": 3},
            unk_token="<unk>",
        )
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=4,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=1,
        )
    )
    runtime = GeneralistTrainingRuntime(
        model=model,
        tokenizer=tokenizer,
        lane="nf4-qlora",
        target_matches=(),
        trainable_parameter_names=(),
        trainable_parameter_count=0,
        total_parameter_count=sum(item.numel() for item in model.parameters()),
        quantized_linear_module_count=0,
    )
    example = WeightedTokenizedExample(
        statement_id="statement",
        proof_variant_id="proof",
        declaration_name="fixture",
        prompt="prompt",
        completion="proof",
        input_ids=(2, 3, 1),
        labels=(-100, 3, 1),
        attention_mask=(1, 1, 1),
        prompt_tokens=1,
        completion_tokens=1,
        example_weight=1.0,
    )

    trainer = build_weighted_sft_trainer(
        runtime,
        [example],
        GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json"),
        tmp_path,
        maximum_sequence_tokens=4096,
        save_quarter_checkpoints=False,
        maximum_optimizer_steps=1,
    )

    assert str(trainer.args.train_sampling_strategy).endswith("sequential")
    assert trainer.args.max_steps == 1
    assert any(
        type(callback).__name__ == "FiniteOptimizationCallback"
        for callback in trainer.callback_handler.callbacks
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


def test_native_single_example_loss_preserves_weight_without_logits() -> None:
    torch = pytest.importorskip("torch")

    loss = scale_single_example_causal_loss(
        torch.tensor(2.0), torch.tensor([3.0]), weight_normalizer=1.5
    )

    assert loss == pytest.approx(4.0)
    with pytest.raises(ValueError, match="micro-batch one"):
        scale_single_example_causal_loss(
            torch.tensor(2.0), torch.tensor([1.0, 1.0]), weight_normalizer=1.0
        )


def test_checkpointed_target_only_loss_matches_full_masked_causal_loss() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(3, 5, bias=False)
    hidden = torch.randn(1, 6, 3, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4]])

    observed = checkpointed_target_only_causal_loss(
        model, hidden, labels, chunk_tokens=2
    )
    full_logits = model.lm_head(hidden)
    expected = torch.nn.functional.cross_entropy(
        full_logits[:, :-1, :].reshape(-1, 5),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    observed.backward()

    assert observed.detach() == pytest.approx(float(expected.detach()))
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all().item())


def test_sequence_chunked_mlp_preserves_output_and_gradients() -> None:
    torch = pytest.importorskip("torch")

    class TinyMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(3, 4, bias=False)
            self.up_proj = torch.nn.Linear(3, 4, bias=False)
            self.down_proj = torch.nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))

    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TinyMLP()

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([TinyLayer() for _ in range(32)])

    model = TinyModel()
    inputs = torch.randn(1, 5, 3, requires_grad=True)
    expected = model.model.layers[0].mlp(inputs).detach()

    matched = enable_sequence_chunked_mlp(model, chunk_tokens=2)
    observed = model.model.layers[0].mlp(inputs)
    observed.sum().backward()

    assert matched == 32
    assert torch.allclose(observed.detach(), expected)
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all().item())


def test_activation_cpu_offload_is_disabled_for_training() -> None:
    assert should_offload_activations(4095) is False
    assert should_offload_activations(4096) is False
    assert should_offload_activations(32768) is False
    with pytest.raises(ValueError, match="sequence length"):
        should_offload_activations(0)


def test_gradient_checkpointing_is_reserved_for_nontrivial_sequences() -> None:
    assert should_checkpoint_activations(1023) is False
    assert should_checkpoint_activations(1024) is True
    with pytest.raises(ValueError, match="sequence length"):
        should_checkpoint_activations(0)

    class Model:
        is_gradient_checkpointing = True

        def gradient_checkpointing_enable(self, **kwargs) -> None:
            self.is_gradient_checkpointing = True

        def gradient_checkpointing_disable(self) -> None:
            self.is_gradient_checkpointing = False

    model = Model()
    assert configure_gradient_checkpointing(model, 100) is False
    assert configure_gradient_checkpointing(model, 2000) is True


def test_locked_fla_delta_rule_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("FLA DeltaNet equivalence requires CUDA")
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
    )

    backend = inspect_gated_delta_rule_backend()
    torch.manual_seed(7)
    shape = (1, 64, 2, 8)
    inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    decay = torch.nn.functional.logsigmoid(
        torch.randn((1, 64, 2), device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(torch.randn((1, 64, 2), device="cuda", dtype=torch.float32))

    def run(operation):
        arguments = [item.detach().clone().requires_grad_() for item in inputs]
        run_decay = decay.detach().clone().requires_grad_()
        run_beta = beta.detach().clone().requires_grad_()
        output, _ = operation(
            *arguments,
            g=run_decay,
            beta=run_beta,
            chunk_size=32,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        output.float().square().mean().backward()
        gradients = [
            item.grad.detach().float() for item in [*arguments, run_decay, run_beta]
        ]
        return output.detach().float(), gradients

    observed, observed_gradients = run(torch_chunk_gated_delta_rule)
    expected, expected_gradients = run(torch_chunk_gated_delta_rule.__wrapped__)

    assert backend["distribution_version"] == "0.5.2"
    assert torch.allclose(observed, expected, atol=0.005, rtol=0.005)
    assert all(
        torch.allclose(actual, reference, atol=2e-5, rtol=0.01)
        for actual, reference in zip(
            observed_gradients, expected_gradients, strict=True
        )
    )
