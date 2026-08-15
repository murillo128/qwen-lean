from qwen_lean.prompt import PROMPT_FORMAT_ID, reconstruct_source, render_prompt
from qwen_lean.schema import TaskRecord


TASK = TaskRecord(
    id="identity",
    preamble="import Init",
    declaration="theorem identity (P : Prop) (h : P) : P",
    declaration_name="identity",
)


def test_whole_proof_v1_rendering_is_exact() -> None:
    assert PROMPT_FORMAT_ID == "whole-proof-v1"
    assert render_prompt(TASK) == (
        "import Init\n\n"
        "/- Complete the proof below.\n"
        "Return only Lean code continuing after `by`; do not use `sorry` or `admit`. -/\n"
        "theorem identity (P : Prop) (h : P) : P := by\n  "
    )


def test_reconstruction_appends_raw_candidate_to_same_prefix() -> None:
    candidate = "```lean\nby\n  exact h\r\n"
    assert reconstruct_source(TASK, candidate) == render_prompt(TASK) + (
        "```lean\nby\n  exact h\n"
    )
