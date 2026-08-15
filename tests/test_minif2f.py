from pathlib import Path

from qwen_lean.minif2f import Phase1Config, materialize_validation_source
from qwen_lean.prompt import PROMPT_FORMAT_ID, render_prompt
from qwen_lean.schema import TaskRecord


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIXTURE = """\
/- Synthetic parser fixture. -/
import MiniF2F.ProblemImports

open scoped Real
open scoped Nat

/-- Primary declaration. -/
theorem primary_alpha (x : ℕ) : x = x := by
  sorry

theorem primary_alpha.variants.helper (x : ℕ) : x = x := by
  sorry

/-- Second primary declaration. -/
theorem primary_beta : True := by
  sorry
"""


def test_materialization_preserves_preamble_and_primary_declarations() -> None:
    tasks = materialize_validation_source(SOURCE_FIXTURE, expected_primary_task_count=2)

    assert [task.id for task in tasks] == ["primary_alpha", "primary_beta"]
    assert tasks[0].preamble == (
        "import MiniF2F.ProblemImports\n\nopen scoped Real\nopen scoped Nat"
    )
    assert tasks[0].declaration == "theorem primary_alpha (x : ℕ) : x = x"
    assert tasks[1].declaration == "theorem primary_beta : True"


def test_materialization_rejects_a_changed_primary_denominator() -> None:
    try:
        materialize_validation_source(SOURCE_FIXTURE, expected_primary_task_count=3)
    except ValueError as error:
        assert "expected 3 primary tasks, got 2" in str(error)
    else:
        raise AssertionError("changed miniF2F denominator was accepted")


def test_dev16_is_fixed_deterministically_and_is_a_manifest_subset() -> None:
    config = Phase1Config.load(ROOT / "config/phase1-minif2f.json")
    manifest_path = ROOT / "config/minif2f-valid-task-ids.txt"
    full_ids = [
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    tasks = [
        TaskRecord(
            id=task_id,
            preamble="import Init",
            declaration="theorem t : True",
            declaration_name=task_id,
        )
        for task_id in full_ids
    ]

    dev_tasks = config.select_workload("minif2f-valid-dev16-v1", tasks)
    full_tasks = config.select_workload("minif2f-valid-v1", tasks)

    assert len(full_tasks) == 244
    assert [task.id for task in dev_tasks] == sorted(full_ids)[:16]
    assert {task.id for task in dev_tasks} < {task.id for task in full_tasks}


def test_benchmark_prompt_remains_plain_whole_proof_v1() -> None:
    task = materialize_validation_source(
        SOURCE_FIXTURE, expected_primary_task_count=2
    )[0]
    prompt = render_prompt(task)

    assert PROMPT_FORMAT_ID == "whole-proof-v1"
    assert prompt.startswith(task.preamble)
    assert prompt.endswith(f"{task.declaration} := by\n  ")
    assert "<|im_start|>" not in prompt
    assert "```" not in prompt
