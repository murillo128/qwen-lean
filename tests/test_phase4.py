import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_lean.phase2_schema import (
    MathlibProofRecord,
    SourcePosition,
    SourceSpan,
    TokenLengths,
)
from qwen_lean.phase3 import IGNORE_INDEX, tokenize_sft_record
from qwen_lean.phase3_verification import reconstruct_generated_proof
from qwen_lean.phase4 import (
    HELDOUT_WORKLOAD_ID,
    TRAIN_WORKLOAD_ID,
    VALIDATION_WORKLOAD_ID,
    Phase4Config,
    select_heldout_workload,
    select_sft_workload,
)
from qwen_lean.phase4_evidence import _compact_training
from qwen_lean.phase4_inference import _phase1_config, heldout_generation_request
from qwen_lean.phase4_training import (
    select_validation_checkpoint,
    validate_phase4_resume_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    eos_token_id = 999_999
    pad_token_id = None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [index + 10 for index, _ in enumerate(text.split())]


def _span() -> SourceSpan:
    return SourceSpan(SourcePosition(1, 1), SourcePosition(1, 2))


def _record(
    index: int,
    *,
    split: str,
    declaration: str | None = None,
    completion: str = "exact True.intro",
) -> MathlibProofRecord:
    return MathlibProofRecord(
        schema_version="mathlib-whole-proof-v1",
        id=hashlib.sha256(f"phase4-record-{index}".encode()).hexdigest(),
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="81a5d257c8e410db227a6665ed08f64fea08e997",
        file_path=f"Mathlib/Test/{index}.lean",
        declaration_name=f"Example.phase4_{index}",
        declaration_kind="theorem",
        source_span=_span(),
        declaration_span=_span(),
        proof_span=_span(),
        declaration=declaration or f"theorem phase4_{index} : True",
        proof="by\n  exact True.intro",
        completion=completion,
        premises=(),
        file_group=f"Mathlib/Test/{index}.lean",
        component_id=hashlib.sha256(f"component-{index}".encode()).hexdigest(),
        split=split,  # type: ignore[arg-type]
        statement_fingerprint=hashlib.sha256(f"statement-{index}".encode()).hexdigest(),
        token_lengths=TokenLengths(1, 1, 2, 2, 2),
    )


def _small_config() -> Phase4Config:
    source = Phase4Config.load(ROOT / "config/phase4-smoke.json")
    value = copy.deepcopy(source.value)
    value["workloads"]["train"]["expected_examples"] = 4
    value["workloads"]["validation"]["expected_examples"] = 3
    value["workloads"]["heldout"]["expected_examples"] = 2
    value["workloads"]["train"]["maximum_sequence_tokens"] = 80
    value["workloads"]["validation"]["maximum_sequence_tokens"] = 80
    value["workloads"]["heldout"]["maximum_prompt_and_generation_tokens"] = 80
    value["heldout_generation"]["max_new_tokens"] = 20
    return Phase4Config(path=source.path, value=value)


def test_phase4_config_pins_fixed_qlora_and_evaluation_contracts() -> None:
    config = Phase4Config.load(ROOT / "config/phase4-smoke.json")

    assert config.workloads["train"]["id"] == TRAIN_WORKLOAD_ID
    assert config.workloads["validation"]["id"] == VALIDATION_WORKLOAD_ID
    assert config.workloads["heldout"]["id"] == HELDOUT_WORKLOAD_ID
    assert config.training["gradient_accumulation_steps"] == 8
    assert config.training["maximum_sequence_tokens"] == 1024
    assert config.training["maximum_optimizer_steps"] == 512
    assert config.training["checkpoint_candidates"] == [128, 256, 384, 512]
    assert config.training["lr_schedule"] == "cosine"
    assert config.lora["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_phase4_sft_selectors_are_deterministic_split_correct_and_bounded() -> None:
    config = _small_config()
    tokenizer = WordTokenizer()
    train_records = [_record(index, split="train") for index in range(8)]
    validation_records = [
        _record(index + 100, split="validation") for index in range(7)
    ]
    train_records.append(
        _record(
            90,
            split="train",
            declaration="theorem oversized : " + "True " * 100,
        )
    )

    train, train_eligible = select_sft_workload(
        train_records, tokenizer, config, "train"
    )
    reversed_train, reversed_train_eligible = select_sft_workload(
        reversed(train_records), tokenizer, config, "train"
    )
    validation, validation_eligible = select_sft_workload(
        validation_records, tokenizer, config, "validation"
    )

    assert train_eligible == reversed_train_eligible == 8
    assert [item.record_id for item in train] == [
        item.record_id for item in reversed_train
    ]
    assert len(train) == 4
    assert validation_eligible == 7
    assert len(validation) == 3
    assert all(len(item.input_ids) <= 80 for item in train + validation)
    assert not (
        {item.record_id for item in train} & {item.record_id for item in validation}
    )

    with pytest.raises(ValueError, match="contains validation"):
        select_sft_workload(validation_records, tokenizer, config, "train")


def test_phase4_validation_serialization_keeps_target_only_mask_and_eos() -> None:
    tokenizer = WordTokenizer()
    record = _record(1, split="validation", completion="exact True.intro  ")

    example = tokenize_sft_record(
        record,
        tokenizer,
        maximum_sequence_tokens=1024,
        expected_split="validation",
    )

    assert (
        example.labels[: example.prompt_tokens]
        == (IGNORE_INDEX,) * example.prompt_tokens
    )
    assert (
        example.labels[example.prompt_tokens : -1]
        == example.input_ids[example.prompt_tokens : -1]
    )
    assert example.input_ids[-1] == example.labels[-1] == tokenizer.eos_token_id
    assert example.prompt.endswith(" := by\n  ")
    assert example.completion == "exact True.intro  "


def test_phase4_overlength_sft_is_rejected_without_truncation() -> None:
    record = _record(
        2,
        split="train",
        declaration="theorem too_long : " + "True " * 1100,
    )

    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_sft_record(
            record,
            WordTokenizer(),
            maximum_sequence_tokens=1024,
        )


def test_phase4_heldout_selector_uses_prompt_budget_not_completion_or_outcome() -> None:
    config = _small_config()
    tokenizer = WordTokenizer()
    records = [
        _record(
            index + 200,
            split="heldout",
            completion="irrelevant_target " * (index + 1) * 100,
        )
        for index in range(5)
    ]

    selected, eligible = select_heldout_workload(records, tokenizer, config)
    reversed_selected, reversed_eligible = select_heldout_workload(
        reversed(records), tokenizer, config
    )

    assert eligible == reversed_eligible == 5
    assert [item.record_id for item in selected] == [
        item.record_id for item in reversed_selected
    ]
    assert len(selected) == 2
    assert all(item.prompt_tokens + 20 <= 80 for item in selected)


def test_phase4_checkpoint_selection_uses_validation_ce_and_ties_earlier() -> None:
    probes = [
        {"optimizer_step": 128, "mean_target_token_cross_entropy": 1.2},
        {"optimizer_step": 256, "mean_target_token_cross_entropy": 0.9},
        {"optimizer_step": 384, "mean_target_token_cross_entropy": 0.9},
        {"optimizer_step": 512, "mean_target_token_cross_entropy": 1.0},
    ]

    selection = select_validation_checkpoint(probes)

    assert selection["selected_optimizer_step"] == 256
    assert selection["heldout_or_minif2f_consulted"] is False
    with pytest.raises(ValueError, match="exactly the configured boundaries"):
        select_validation_checkpoint(probes[:-1])


def test_phase4_resume_requires_full_state_and_step256_data_position(
    tmp_path: Path,
) -> None:
    config = Phase4Config.load(ROOT / "config/phase4-smoke.json")
    checkpoint = tmp_path / "checkpoint-256"
    checkpoint.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
    ):
        (checkpoint / name).write_text("fixture", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 256, "epoch": 0.5}), encoding="utf-8"
    )

    metadata = validate_phase4_resume_checkpoint(config, checkpoint)

    assert metadata["optimizer_state_preserved"] is True
    assert metadata["scheduler_state_preserved"] is True
    assert metadata["rng_state_preserved"] is True
    assert metadata["data_position_preserved"] is True
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match="not full-state resumable"):
        validate_phase4_resume_checkpoint(config, checkpoint)


def test_phase4_base_and_adapter_requests_differ_only_by_adapter_enablement(
    tmp_path: Path,
) -> None:
    config = Phase4Config.load(ROOT / "config/phase4-smoke.json")
    adapter_dir = tmp_path / "checkpoint-256"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    base = heldout_generation_request(config, None)
    adapter = heldout_generation_request(config, adapter_dir)

    assert base["sampling"] == adapter["sampling"]
    assert base["adapter"] is None
    assert adapter["adapter"]["enabled"] is True
    assert "enable_lora" not in base["engine"]
    common_adapter_engine = {
        key: value
        for key, value in adapter["engine"].items()
        if key not in {"enable_lora", "max_lora_rank", "max_loras"}
    }
    assert base["engine"] == common_adapter_engine


def test_phase4_heldout_reconstruction_uses_existing_no_repair_path() -> None:
    record = replace(
        _record(3, split="heldout"),
        proof_span=SourceSpan(SourcePosition(1, 3), SourcePosition(2, 6)),
    )

    reconstructed = reconstruct_generated_proof(
        "AAby\n  oldZZ", record, "exact True.intro  \r\n"
    )

    assert reconstructed == "AAby\n  exact True.introZZ"


def test_phase4_minif2f_uses_exact_phase1_sampling_contract() -> None:
    config = Phase4Config.load(ROOT / "config/phase4-smoke.json")
    phase1 = _phase1_config(config)

    assert phase1.sampling == {
        "candidates_per_task": 8,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_new_tokens": 1024,
        "stop": "tokenizer_eos_or_token_limit",
        "seed": 0,
    }


def test_phase4_training_evidence_references_canonical_workload_ids() -> None:
    training = {
        "workloads": {
            name: {"selected_record_ids": [f"{name}-record"], "id": name}
            for name in ("train", "validation", "heldout")
        }
    }

    compact = _compact_training(training)

    assert training["workloads"]["train"]["selected_record_ids"] == ["train-record"]
    for name in ("train", "validation", "heldout"):
        assert "selected_record_ids" not in compact["workloads"][name]
        assert compact["workloads"][name]["selected_record_ids_reference"] == (
            f"workloads.json#workloads.{name}.selected_record_ids"
        )
