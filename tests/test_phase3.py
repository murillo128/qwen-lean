import copy
import hashlib
import json
from pathlib import Path

import pytest

from qwen_lean.baseline import LoRAAdapterSpec, vllm_engine_kwargs
from qwen_lean.minif2f import Phase1Config
from qwen_lean.phase2_schema import (
    MathlibProofRecord,
    SourcePosition,
    SourceSpan,
    TokenLengths,
)
from qwen_lean.phase3 import (
    BASE_MODEL_ID,
    BASE_REVISION,
    IGNORE_INDEX,
    OVERFIT_WORKLOAD_ID,
    Phase3Config,
    pad_target_only_batch,
    render_sft_prompt,
    select_overfit_workload,
    tokenize_sft_record,
)
from qwen_lean.phase3_training import (
    validate_resume_checkpoint,
    validate_training_boundary,
)
from qwen_lean.phase3_evidence import _amended_memorization_results
from qwen_lean.prompt import render_prompt
from qwen_lean.schema import TaskRecord


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
    index: int = 0, *, split: str = "train", declaration: str | None = None
) -> MathlibProofRecord:
    record_id = hashlib.sha256(f"record-{index}".encode()).hexdigest()
    return MathlibProofRecord(
        schema_version="mathlib-whole-proof-v1",
        id=record_id,
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="81a5d257c8e410db227a6665ed08f64fea08e997",
        file_path=f"Mathlib/Test/{index}.lean",
        declaration_name=f"Example.theorem_{index}",
        declaration_kind="theorem",
        source_span=_span(),
        declaration_span=_span(),
        proof_span=_span(),
        declaration=declaration or f"theorem theorem_{index} : True",
        proof="by\n  exact True.intro one two three four five six",
        completion="exact True.intro one two three four five six  ",
        premises=(),
        file_group=f"Mathlib/Test/{index}.lean",
        component_id=hashlib.sha256(f"component-{index}".encode()).hexdigest(),
        split=split,  # type: ignore[arg-type]
        statement_fingerprint=hashlib.sha256(f"statement-{index}".encode()).hexdigest(),
        token_lengths=TokenLengths(1, 1, 8, 9, 9),
    )


def _synthetic_config(records: list[MathlibProofRecord]) -> Phase3Config:
    source = Phase3Config.load(ROOT / "config/phase3-overfit.json")
    value = copy.deepcopy(source.value)
    prefix = f"{OVERFIT_WORKLOAD_ID}\0".encode()
    selected = sorted(
        (record.id for record in records),
        key=lambda record_id: hashlib.sha256(prefix + record_id.encode()).digest(),
    )[:64]
    value["workload"]["selected_record_ids"] = selected
    config = Phase3Config(path=source.path, value=value)
    config.validate()
    return config


def test_mathlib_sft_v1_uses_the_shared_proof_request_and_exact_completion() -> None:
    record = _record()
    task = TaskRecord(
        id=record.id,
        preamble="import Mathlib",
        declaration=record.declaration,
        declaration_name=record.declaration_name,
    )

    assert render_sft_prompt(record) == (
        "/- Complete the proof below.\n"
        "Return only Lean code continuing after `by`; do not use `sorry` or `admit`. -/\n"
        "theorem theorem_0 : True := by\n  "
    )
    assert render_prompt(task).endswith(render_sft_prompt(record))
    assert render_sft_prompt(record) + record.completion == (
        render_sft_prompt(record) + "exact True.intro one two three four five six  "
    )


def test_target_only_labels_supervise_completion_and_exactly_one_appended_eos() -> None:
    tokenizer = WordTokenizer()
    example = tokenize_sft_record(_record(), tokenizer, maximum_sequence_tokens=512)

    assert example.input_ids[-1] == tokenizer.eos_token_id
    assert example.input_ids.count(tokenizer.eos_token_id) == 1
    assert (
        example.labels[: example.prompt_tokens]
        == (IGNORE_INDEX,) * example.prompt_tokens
    )
    assert (
        example.labels[example.prompt_tokens : -1]
        == example.input_ids[example.prompt_tokens : -1]
    )
    assert example.labels[-1] == tokenizer.eos_token_id
    assert example.attention_mask == (1,) * len(example.input_ids)


def test_production_padding_path_masks_padding_without_changing_real_tokens() -> None:
    batch = pad_target_only_batch(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": [IGNORE_INDEX, 2, 3],
                "attention_mask": [1, 1, 1],
            },
            {
                "input_ids": [4, 5],
                "labels": [IGNORE_INDEX, 5],
                "attention_mask": [1, 1],
            },
        ],
        pad_token_id=999,
    )

    assert batch["input_ids"] == [[1, 2, 3], [4, 5, 999]]
    assert batch["labels"] == [[IGNORE_INDEX, 2, 3], [IGNORE_INDEX, 5, IGNORE_INDEX]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]


def test_overfit_selector_is_deterministic_train_only_and_length_bounded() -> None:
    records = [_record(index) for index in range(80)]
    config = _synthetic_config(records)
    tokenizer = WordTokenizer()

    selected, eligible = select_overfit_workload(records, tokenizer, config)
    reversed_selected, reversed_eligible = select_overfit_workload(
        reversed(records), tokenizer, config
    )

    assert eligible == reversed_eligible == 80
    assert [item.record_id for item in selected] == list(config.selected_record_ids)
    assert [item.record_id for item in reversed_selected] == list(
        config.selected_record_ids
    )
    assert all(item.completion_tokens == 8 for item in selected)
    assert all(len(item.input_ids) <= 512 for item in selected)

    with pytest.raises(ValueError, match="non-train"):
        select_overfit_workload(
            records + [_record(100, split="validation")], tokenizer, config
        )


def test_oversized_sft_example_is_rejected_instead_of_truncated() -> None:
    oversized = _record(declaration="theorem too_long : " + "True " * 600)

    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_sft_record(oversized, WordTokenizer(), maximum_sequence_tokens=512)


def test_phase3_config_pins_base_tokenizer_and_qlora_targets() -> None:
    config = Phase3Config.load(ROOT / "config/phase3-overfit.json")

    assert config.model["model_id"] == config.model["tokenizer_id"] == BASE_MODEL_ID
    assert (
        config.model["model_revision"]
        == config.model["tokenizer_revision"]
        == BASE_REVISION
    )
    assert config.lora["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert len(config.selected_record_ids) == len(set(config.selected_record_ids)) == 64


def test_amended_training_boundaries_are_fixed_and_resumable(tmp_path: Path) -> None:
    config = Phase3Config.load(ROOT / "config/phase3-overfit.json")

    for target_step in (100, 200, 300, 400, 500, 600):
        validate_training_boundary(config, target_step)
    for invalid_step in (0, 99, 150, 700):
        with pytest.raises(ValueError, match="target step"):
            validate_training_boundary(config, invalid_step)

    checkpoint = tmp_path / "checkpoint-100"
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
        '{"global_step": 100}\n', encoding="utf-8"
    )

    assert validate_resume_checkpoint(config, checkpoint, 200) == 100
    with pytest.raises(ValueError, match="exactly one 100-step boundary"):
        validate_resume_checkpoint(config, checkpoint, 300)


def test_amended_training_rejects_adapter_only_checkpoint(tmp_path: Path) -> None:
    config = Phase3Config.load(ROOT / "config/phase3-overfit.json")
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_text("fixture", encoding="utf-8")

    with pytest.raises(ValueError, match="not resumable"):
        validate_resume_checkpoint(config, checkpoint, 200)


def test_amended_evidence_recovers_boundary_from_result_filename(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "memorization-amended"
    directory.mkdir()
    for step in (100, 200):
        (directory / f"step-{step}.json").write_text(
            json.dumps({"optimizer_step": None, "exact_matches": step // 10}),
            encoding="utf-8",
        )

    values = _amended_memorization_results(tmp_path)

    assert [value["optimizer_step"] for value in values] == [100, 200]
    assert [value["exact_matches"] for value in values] == [10, 20]


def test_adapter_support_is_opt_in_and_metadata_cannot_impersonate_base(
    tmp_path: Path,
) -> None:
    phase1 = Phase1Config.load(ROOT / "config/phase1-minif2f.json")
    base_kwargs = vllm_engine_kwargs(phase1, phase1.sampling, None)

    assert "enable_lora" not in base_kwargs
    assert "max_lora_rank" not in base_kwargs

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter = LoRAAdapterSpec(
        adapter_id="phase3-overfit64-v1-lora",
        path=adapter_dir,
        rank=16,
        base_model_id=BASE_MODEL_ID,
        base_model_revision=BASE_REVISION,
    )
    adapter_kwargs = vllm_engine_kwargs(phase1, phase1.sampling, adapter)

    assert adapter_kwargs["enable_lora"] is True
    assert adapter_kwargs["max_lora_rank"] == 16
    assert adapter.metadata()["merged"] is False
    assert adapter.metadata()["adapter_id"] != adapter.metadata()["base_model_id"]

    impersonating = LoRAAdapterSpec(
        adapter_id=BASE_MODEL_ID,
        path=adapter_dir,
        rank=16,
        base_model_id=BASE_MODEL_ID,
        base_model_revision=BASE_REVISION,
    )
    with pytest.raises(ValueError, match="distinct"):
        impersonating.validate(phase1)
