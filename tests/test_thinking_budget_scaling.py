import hashlib
import json
from pathlib import Path

import pytest

from qwen_lean.native_thinking_assessment import (
    MathiaTask,
    NativeThinkingConfig,
)
from qwen_lean.thinking_budget_scaling import (
    FROZEN_SCALING_ARMS,
    FROZEN_SCALING_SAMPLING,
    GENERATION_SCHEMA,
    SelectedTask,
    ThinkingBudgetScalingConfig,
    _gate_candidate_identity,
    load_scaling_generation_records,
    reasoning_exit_category,
    select_scaling_tasks,
    validate_stage1_binding,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-thinking-budget-scaling.json"
STAGE1_CONFIG_PATH = ROOT / "config/qwen35-native-thinking-ab.json"


def test_config_freezes_stage2_amendment_and_stage1_binding() -> None:
    config = ThinkingBudgetScalingConfig.load(CONFIG_PATH)
    stage1 = NativeThinkingConfig.load(STAGE1_CONFIG_PATH)

    validate_stage1_binding(config, stage1)
    assert config.sampling == FROZEN_SCALING_SAMPLING
    assert config.arms == FROZEN_SCALING_ARMS
    assert tuple(config.arms) == ("B4", "B8", "B16")
    assert config.selection["tasks_per_workload"] == 8
    assert config.selection["prompt_token_ceiling"] == 4096


def test_config_rejects_budget_or_sampler_changes(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["arms"]["B8"]["total_output_ceiling"] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="arms differ"):
        ThinkingBudgetScalingConfig.load(changed)


def test_selection_is_first_eligible_per_workload_without_outcomes() -> None:
    config = ThinkingBudgetScalingConfig.load(CONFIG_PATH)
    stage1 = NativeThinkingConfig.load(STAGE1_CONFIG_PATH)
    tasks = []
    for index in range(10):
        tasks.append(_task(f"mini-{index}", "minif2f-valid-clean-v2"))
        tasks.append(_task(f"fresh-{index}", "fresh-composition-valid-v2"))
    tokenizer = _FakeTokenizer(over_ceiling={"mini-1", "fresh-0"})

    selected, evidence = select_scaling_tasks(config, stage1, tasks, tokenizer)

    assert [
        row.task.task_id for row in selected if row.task.workload.startswith("mini")
    ] == [
        "mini-0",
        "mini-2",
        "mini-3",
        "mini-4",
        "mini-5",
        "mini-6",
        "mini-7",
        "mini-8",
    ]
    assert [
        row.task.task_id for row in selected if row.task.workload.startswith("fresh")
    ] == [
        "fresh-1",
        "fresh-2",
        "fresh-3",
        "fresh-4",
        "fresh-5",
        "fresh-6",
        "fresh-7",
        "fresh-8",
    ]
    assert evidence["outcome_blind"] is True
    assert all("verification" not in row for row in evidence["selected_tasks"])


@pytest.mark.parametrize(
    ("reasoning", "budget", "final", "end_position", "expected"),
    [
        (6, 8, "exact True.intro", 7, "natural_before_budget"),
        (7, 8, "exact True.intro", 8, "forced_at_budget"),
        (0, 0, "exact True.intro", 1, "forced_at_budget"),
        (8, 8, None, 8, "no_final_transition"),
        (9, 8, "exact True.intro", 10, "budget_exceeded"),
    ],
)
def test_reasoning_exit_is_auditable(
    reasoning: int,
    budget: int,
    final: str | None,
    end_position: int,
    expected: str,
) -> None:
    assert (
        reasoning_exit_category(
            reasoning,
            budget,
            final,
            reasoning_end_position_token_count=end_position,
        )
        == expected
    )


def test_generation_loader_fails_closed_on_raw_token_mutation(
    tmp_path: Path,
) -> None:
    config = ThinkingBudgetScalingConfig.load(CONFIG_PATH)
    selected = _selected(_task("mini-0", "minif2f-valid-clean-v2"))
    candidate_id, identity = _gate_candidate_identity(config, selected, 8)
    token_ids = [1, 2, 3]
    record = {
        "schema_version": GENERATION_SCHEMA,
        "candidate_id": candidate_id,
        **identity,
        "is_runtime_gate": True,
        "raw_response_text": "response",
        "raw_response_sha256": _sha256_text("response"),
        "raw_response_token_ids": token_ids,
        "raw_response_token_ids_sha256": _sha256_json(token_ids),
        "raw_response_token_count": len(token_ids),
        "final_content": "proof",
        "final_content_sha256": _sha256_text("proof"),
    }
    path = tmp_path / "generations.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_scaling_generation_records(path)[0]["candidate_id"] == candidate_id

    record["raw_response_token_ids"] = [1, 2, 4]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw token-id hash mismatch"):
        load_scaling_generation_records(path)


class _FakeTokenizer:
    def __init__(self, *, over_ceiling: set[str]) -> None:
        self.over_ceiling = over_ceiling

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        assert add_generation_prompt is True
        assert enable_thinking is True
        text = messages[0]["content"]
        count = 5000 if any(value in text for value in self.over_ceiling) else 100
        return list(range(count)) if tokenize else f"rendered:{text}"


def _task(task_id: str, workload: str) -> MathiaTask:
    return MathiaTask(
        task_id=task_id,
        workload=workload,
        preamble="import Mathlib",
        declaration=f"theorem {task_id.replace('-', '_')} : True",
        declaration_name=task_id.replace("-", "_"),
        intuition=f"intuition {task_id}",
        intuition_sha256=f"intuition-{task_id}",
        theorem_sha256=f"theorem-{task_id}",
    )


def _selected(task: MathiaTask) -> SelectedTask:
    return SelectedTask(
        task=task,
        frozen_global_index=0,
        frozen_workload_index=0,
        user_message="prompt",
        user_message_sha256="prompt-sha",
        rendered_prompt="rendered",
        rendered_prompt_sha256="rendered-sha",
        rendered_prompt_token_count=100,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
