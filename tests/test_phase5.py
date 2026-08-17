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
from qwen_lean.phase4_inference import _phase1_config
from qwen_lean.phase5 import (
    HELDOUT_WORKLOAD_ID,
    TRAIN_WORKLOAD_ID,
    VALIDATION_WORKLOAD_ID,
    Phase5Config,
    derive_phase5_trajectory,
    ordered_record_ids_sha256,
    select_full_sft_workload,
    select_phase5_heldout_workload,
)
from qwen_lean.phase5_evidence import compact_phase5_workloads
from qwen_lean.phase5_inference import (
    _validate_accepted_phase1_base,
    phase5_heldout_generation_request,
)
from qwen_lean.phase5_training import (
    select_phase5_validation_checkpoint,
    validate_phase5_resume_checkpoint,
)
from qwen_lean.phase4_training import summarize_finite_training_logs


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
        id=hashlib.sha256(f"phase5-record-{index}".encode()).hexdigest(),
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="81a5d257c8e410db227a6665ed08f64fea08e997",
        file_path=f"Mathlib/Test/{index}.lean",
        declaration_name=f"Example.phase5_{index}",
        declaration_kind="theorem",
        source_span=_span(),
        declaration_span=_span(),
        proof_span=_span(),
        declaration=declaration or f"theorem phase5_{index} : True",
        proof="by\n  exact True.intro",
        completion=completion,
        premises=(),
        file_group=f"Mathlib/Test/{index}.lean",
        component_id=hashlib.sha256(f"component-{index}".encode()).hexdigest(),
        split=split,  # type: ignore[arg-type]
        statement_fingerprint=hashlib.sha256(f"statement-{index}".encode()).hexdigest(),
        token_lengths=TokenLengths(1, 1, 2, 2, 2),
    )


def _small_config() -> Phase5Config:
    source = Phase5Config.load(ROOT / "config/phase5-full.json")
    value = copy.deepcopy(source.value)
    value["workloads"]["train"]["expected_input_examples"] = 5
    value["workloads"]["validation"]["expected_input_examples"] = 4
    value["workloads"]["heldout"]["expected_input_examples"] = 5
    value["workloads"]["train"]["maximum_sequence_tokens"] = 80
    value["workloads"]["validation"]["maximum_sequence_tokens"] = 80
    value["workloads"]["heldout"]["expected_examples"] = 2
    value["workloads"]["heldout"]["maximum_prompt_and_generation_tokens"] = 80
    value["heldout_generation"]["max_new_tokens"] = 20
    return Phase5Config(path=source.path, value=value)


def test_phase5_config_pins_full_corpus_and_fixed_qlora_contract() -> None:
    config = Phase5Config.load(ROOT / "config/phase5-full.json")

    assert config.workloads["train"]["id"] == TRAIN_WORKLOAD_ID
    assert config.workloads["validation"]["id"] == VALIDATION_WORKLOAD_ID
    assert config.workloads["heldout"]["id"] == HELDOUT_WORKLOAD_ID
    assert config.workloads["train"]["selection"] == "all_eligible"
    assert config.workloads["validation"]["selection"] == "all_eligible"
    assert config.training["epochs"] == 1
    assert config.training["gradient_accumulation_steps"] == 8
    assert config.training["packing"] is False
    assert config.training["truncation"] is False
    assert config.value["minif2f"]["workload_id"] == "minif2f-valid-v1"


def test_phase5_full_sft_membership_uses_only_split_and_serialized_length() -> None:
    config = _small_config()
    tokenizer = WordTokenizer()
    records = [_record(index, split="train") for index in range(4)]
    overlength = _record(
        99,
        split="train",
        declaration="theorem oversized : " + "True " * 100,
    )
    records.append(overlength)

    selected, excluded, input_count = select_full_sft_workload(
        records, tokenizer, config, "train"
    )
    reversed_selected, reversed_excluded, reversed_count = select_full_sft_workload(
        reversed(records), tokenizer, config, "train"
    )

    assert input_count == reversed_count == 5
    assert {item.record_id for item in selected} == {
        item.record_id for item in reversed_selected
    }
    assert len(selected) == 4
    assert [item.record_id for item in excluded] == [overlength.id]
    assert [item.record_id for item in reversed_excluded] == [overlength.id]
    assert excluded[0].serialized_tokens > 80
    assert all(len(item.input_ids) <= 80 for item in selected)


def test_phase5_validation_is_target_only_and_rejected_as_training_input() -> None:
    config = _small_config()
    tokenizer = WordTokenizer()
    record = _record(10, split="validation", completion="exact True.intro  ")

    selected, excluded, count = select_full_sft_workload(
        [record], tokenizer, config, "validation"
    )

    assert count == 1 and not excluded
    example = selected[0]
    assert (
        example.labels[: example.prompt_tokens]
        == (IGNORE_INDEX,) * example.prompt_tokens
    )
    assert example.input_ids[-1] == example.labels[-1] == tokenizer.eos_token_id
    with pytest.raises(ValueError, match="contains validation"):
        select_full_sft_workload([record], tokenizer, config, "train")


def test_phase5_overlength_is_explicit_and_never_truncated() -> None:
    record = _record(
        11,
        split="train",
        declaration="theorem too_long : " + "True " * 1100,
    )
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_sft_record(
            record,
            WordTokenizer(),
            maximum_sequence_tokens=1024,
        )


def test_phase5_trajectory_derives_quarters_warmup_and_partial_final_update() -> None:
    trajectory = derive_phase5_trajectory(65, effective_batch_size=8)

    assert trajectory.maximum_optimizer_steps == 9
    assert trajectory.checkpoint_candidates == (3, 5, 7, 9)
    assert trajectory.mandatory_process_stop_step == 5
    assert trajectory.warmup_steps == 1
    assert trajectory.final_optimizer_update_examples == 1
    assert (
        8 * (trajectory.maximum_optimizer_steps - 1)
        + trajectory.final_optimizer_update_examples
        == 65
    )


def test_phase5_checkpoint_selection_uses_validation_ce_and_ties_earlier() -> None:
    probes = [
        {"optimizer_step": 3, "mean_target_token_cross_entropy": 1.2},
        {"optimizer_step": 5, "mean_target_token_cross_entropy": 0.9},
        {"optimizer_step": 7, "mean_target_token_cross_entropy": 0.9},
        {"optimizer_step": 9, "mean_target_token_cross_entropy": 1.0},
    ]

    selection = select_phase5_validation_checkpoint(probes, [3, 5, 7, 9])

    assert selection["selected_optimizer_step"] == 5
    assert selection["heldout_or_minif2f_consulted"] is False
    with pytest.raises(ValueError, match="exactly the configured boundaries"):
        select_phase5_validation_checkpoint(probes[:-1], [3, 5, 7, 9])


def test_phase5_training_log_summary_requires_finite_complete_step_coverage() -> None:
    summary = summarize_finite_training_logs(
        [
            {"step": 1, "loss": 1.25, "grad_norm": 0.75},
            {"step": 2, "loss": 1.0, "grad_norm": 0.5},
            {"step": 2, "train_runtime": 2.0},
        ],
        2,
    )

    assert summary["logged_optimizer_steps"] == 2
    assert summary["covers_every_optimizer_step_exactly_once"] is True
    assert summary["loss"] == {"minimum": 1.0, "maximum": 1.25, "mean": 1.125}
    with pytest.raises(RuntimeError, match="every optimizer step exactly once"):
        summarize_finite_training_logs([{"step": 2, "loss": 1.0, "grad_norm": 0.5}], 2)
    with pytest.raises(RuntimeError, match="non-finite"):
        summarize_finite_training_logs(
            [{"step": 1, "loss": float("nan"), "grad_norm": 0.5}], 1
        )


def test_phase5_q2_resume_rejects_missing_full_state(tmp_path: Path) -> None:
    config = Phase5Config.load(ROOT / "config/phase5-full.json")
    resolved = config.resolve_for_training_examples(65)
    checkpoint = tmp_path / "checkpoint-5"
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
        json.dumps({"global_step": 5, "epoch": 0.5}), encoding="utf-8"
    )

    metadata = validate_phase5_resume_checkpoint(resolved, checkpoint)

    assert metadata["optimizer_state_preserved"] is True
    assert metadata["scheduler_state_preserved"] is True
    assert metadata["rng_state_preserved"] is True
    assert metadata["data_position_preserved"] is True
    for state_file in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        state_path = checkpoint / state_file
        state_path.unlink()
        with pytest.raises(ValueError, match="not full-state resumable"):
            validate_phase5_resume_checkpoint(resolved, checkpoint)
        state_path.write_text("fixture", encoding="utf-8")

    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 4, "epoch": 0.4}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="global step is 4"):
        validate_phase5_resume_checkpoint(resolved, checkpoint)


def test_phase5_heldout_selector_is_deterministic_and_output_independent() -> None:
    config = _small_config()
    tokenizer = WordTokenizer()
    records = [
        _record(
            index + 100,
            split="heldout",
            completion="irrelevant_target " * (index + 1) * 100,
        )
        for index in range(5)
    ]

    selected, eligible, input_count = select_phase5_heldout_workload(
        records, tokenizer, config
    )
    reversed_selected, reversed_eligible, reversed_count = (
        select_phase5_heldout_workload(reversed(records), tokenizer, config)
    )

    assert input_count == reversed_count == 5
    assert eligible == reversed_eligible == 5
    assert [item.record_id for item in selected] == [
        item.record_id for item in reversed_selected
    ]
    assert len(selected) == 2
    assert all(item.prompt_tokens + 20 <= 80 for item in selected)


def test_phase5_base_and_adapter_requests_differ_only_by_adapter(
    tmp_path: Path,
) -> None:
    config = Phase5Config.load(ROOT / "config/phase5-full.json")
    adapter_dir = tmp_path / "checkpoint"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    base = phase5_heldout_generation_request(config, None)
    adapter = phase5_heldout_generation_request(config, adapter_dir)

    assert base["sampling"] == adapter["sampling"]
    assert base["adapter"] is None
    assert adapter["adapter"]["enabled"] is True
    assert "enable_lora" not in base["engine"]
    assert base["engine"] == {
        key: value
        for key, value in adapter["engine"].items()
        if key not in {"enable_lora", "max_lora_rank", "max_loras"}
    }


def test_phase5_source_reconstruction_remains_raw_no_repair() -> None:
    record = replace(
        _record(12, split="heldout"),
        proof_span=SourceSpan(SourcePosition(1, 3), SourcePosition(2, 6)),
    )

    reconstructed = reconstruct_generated_proof(
        "AAby\n  oldZZ", record, "exact True.intro  \r\n"
    )

    assert reconstructed == "AAby\n  exact True.introZZ"


def test_phase5_minif2f_uses_full_phase1_contract_and_accepted_base() -> None:
    config = Phase5Config.load(ROOT / "config/phase5-full.json")
    phase1 = _phase1_config(config)
    accepted_run = json.loads(
        (ROOT / config.value["minif2f"]["accepted_base_run"]).read_text(
            encoding="utf-8"
        )
    )
    accepted_summary = json.loads(
        (ROOT / config.value["minif2f"]["accepted_base_summary"]).read_text(
            encoding="utf-8"
        )
    )

    _validate_accepted_phase1_base(config, phase1, accepted_run, accepted_summary)

    assert phase1.value["workloads"]["minif2f-valid-v1"]["expected_task_count"] == 244
    assert phase1.sampling["candidates_per_task"] == 8
    assert phase1.sampling["max_new_tokens"] == 1024


def test_phase5_compact_workload_evidence_keeps_exclusions_and_heldout_ids() -> None:
    train_ids = ["train-a", "train-b"]
    validation_ids = ["validation-a"]
    heldout_ids = ["heldout-a", "heldout-b"]
    sft_example = {
        "input_ids": [1, 2],
        "completion_tokens": 1,
    }
    value = {
        "schema_version": "phase5-workloads-v1",
        "dataset_schema_version": "mathlib-whole-proof-v1",
        "serialization_id": "mathlib-sft-v1",
        "tokenizer_id": "tokenizer",
        "tokenizer_revision": "revision",
        "eos_token_id": 2,
        "trajectory": {"maximum_optimizer_steps": 1},
        "workloads": {
            "train": {
                "id": TRAIN_WORKLOAD_ID,
                "split": "train",
                "input_examples": 3,
                "eligible_examples": 2,
                "overlength_examples": 1,
                "overlength_records": [
                    {"record_id": "train-long", "serialized_tokens": 1025}
                ],
                "selected_record_ids": train_ids,
                "examples": [sft_example, sft_example],
            },
            "validation": {
                "id": VALIDATION_WORKLOAD_ID,
                "split": "validation",
                "input_examples": 1,
                "eligible_examples": 1,
                "overlength_examples": 0,
                "overlength_records": [],
                "selected_record_ids": validation_ids,
                "examples": [sft_example],
            },
            "heldout": {
                "id": HELDOUT_WORKLOAD_ID,
                "split": "heldout",
                "input_examples": 2,
                "eligible_examples": 2,
                "selected_record_ids": heldout_ids,
                "examples": [
                    {"prompt_tokens": 10},
                    {"prompt_tokens": 11},
                ],
            },
        },
    }

    compact = compact_phase5_workloads(value)

    assert "selected_record_ids" not in compact["workloads"]["train"]
    assert (
        compact["workloads"]["train"]["overlength_records"][0]["serialized_tokens"]
        == 1025
    )
    assert compact["workloads"]["heldout"]["selected_record_ids"] == heldout_ids
    assert compact["workloads"]["train"]["selected_record_ids_sha256"] == (
        ordered_record_ids_sha256(train_ids)
    )
