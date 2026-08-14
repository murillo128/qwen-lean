from __future__ import annotations

from .schema import TaskRecord


PROMPT_FORMAT_ID = "whole-proof-v1"
_INSTRUCTION = (
    "/- Complete the proof below.\n"
    "Return only Lean code continuing after `by`; do not use `sorry` or `admit`. -/"
)


def render_prompt(task: TaskRecord) -> str:
    """Render the exact plain code-completion prefix for a whole proof."""
    return f"{task.preamble}\n\n{_INSTRUCTION}\n{task.declaration} := by\n  "


def normalize_transport(candidate: str) -> str:
    """Normalize only line endings and trailing transport whitespace."""
    return candidate.replace("\r\n", "\n").replace("\r", "\n").rstrip(" \t\n")


def reconstruct_source(task: TaskRecord, candidate: str) -> str:
    """Append the candidate directly to the exact generation prefix."""
    return f"{render_prompt(task)}{normalize_transport(candidate)}\n"
