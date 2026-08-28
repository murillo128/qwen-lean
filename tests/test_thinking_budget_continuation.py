import hashlib
import json
from pathlib import Path

import pytest

from qwen_lean.native_thinking_assessment import MathiaTask, NativeThinkingConfig
from qwen_lean.thinking_budget_continuation import (
    CONTINUATION_CONFIG_SCHEMA,
    ThinkingBudgetContinuationConfig,
    _verify_dual_record,
    continuation_candidate_identity,
    load_continuation_generation_records,
    validate_continuation_binding,
)
from qwen_lean.thinking_budget_scaling import (
    GENERATION_SCHEMA,
    LEAN_WRAPPER_NORMALIZATION,
    SelectedTask,
    ThinkingBudgetScalingConfig,
    lean_wrapper_normalization_v1,
)
from qwen_lean.verifier import VerificationOutcome

ROOT = Path(__file__).resolve().parents[1]
CONTINUATION_CONFIG_PATH = ROOT / "config/qwen35-thinking-budget-continuation.json"
SCALING_CONFIG_PATH = ROOT / "config/qwen35-thinking-budget-scaling.json"
STAGE1_CONFIG_PATH = ROOT / "config/qwen35-native-thinking-ab.json"


def test_config_binds_historical_gate_and_frozen_scaling_contract() -> None:
    continuation = ThinkingBudgetContinuationConfig.load(CONTINUATION_CONFIG_PATH)
    scaling = ThinkingBudgetScalingConfig.load(SCALING_CONFIG_PATH)
    stage1 = NativeThinkingConfig.load(STAGE1_CONFIG_PATH)

    validate_continuation_binding(continuation, scaling, stage1)
    assert continuation.value["schema_version"] == CONTINUATION_CONFIG_SCHEMA
    assert continuation.continuation_gate["reasoning_budgets"] == [32, 64, 128]
    assert continuation.canonical_output["canonical_answer"] == ("parsed_final_exact")
    assert continuation.canonical_output["normalization"] == (
        LEAN_WRAPPER_NORMALIZATION
    )


def test_config_rejects_changed_normalization(tmp_path: Path) -> None:
    value = json.loads(CONTINUATION_CONFIG_PATH.read_text(encoding="utf-8"))
    value["canonical_output"]["normalization"] = "unfrozen-normalization"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical parsed-output contract"):
        ThinkingBudgetContinuationConfig.load(changed)


@pytest.mark.parametrize(
    ("parsed", "expected", "applied"),
    [
        (None, None, False),
        ("", "", False),
        ("exact True.intro", "exact True.intro", False),
        ("  bypass", "  bypass", False),
        ("by_exact", "by_exact", False),
        ("\n  by\n  exact True.intro", "\n  exact True.intro", True),
        ("by exact True.intro", " exact True.intro", True),
    ],
)
def test_wrapper_normalization_is_narrow_and_byte_preserving(
    parsed: str | None, expected: str | None, applied: bool
) -> None:
    normalized, observed_applied = lean_wrapper_normalization_v1(parsed)

    assert normalized == expected
    assert observed_applied is applied
    assert lean_wrapper_normalization_v1(normalized)[0] == normalized


def test_candidate_identity_binds_arm_budgets_and_continuation_contract() -> None:
    continuation = ThinkingBudgetContinuationConfig.load(CONTINUATION_CONFIG_PATH)
    scaling = ThinkingBudgetScalingConfig.load(SCALING_CONFIG_PATH)
    stage1 = NativeThinkingConfig.load(STAGE1_CONFIG_PATH)
    selected = _selected(_task("mini-0", "minif2f-valid-clean-v2"))

    b4_id, b4 = continuation_candidate_identity(
        continuation, scaling, stage1, selected, "B4"
    )
    b16_id, b16 = continuation_candidate_identity(
        continuation, scaling, stage1, selected, "B16"
    )

    assert b4_id != b16_id
    assert b4["seed"] == b16["seed"] == 0
    assert (b4["max_reasoning_tokens"], b4["total_output_ceiling"]) == (
        4096,
        8192,
    )
    assert (b16["max_reasoning_tokens"], b16["total_output_ceiling"]) == (
        16384,
        20480,
    )


def test_continuation_loader_fails_closed_on_normalized_mutation(
    tmp_path: Path,
) -> None:
    continuation = ThinkingBudgetContinuationConfig.load(CONTINUATION_CONFIG_PATH)
    scaling = ThinkingBudgetScalingConfig.load(SCALING_CONFIG_PATH)
    stage1 = NativeThinkingConfig.load(STAGE1_CONFIG_PATH)
    selected = _selected(_task("mini-0", "minif2f-valid-clean-v2"))
    candidate_id, identity = continuation_candidate_identity(
        continuation, scaling, stage1, selected, "B4"
    )
    parsed = "by\nexact True.intro"
    normalized = "\nexact True.intro"
    token_ids = [1, 2, 3]
    record = {
        "schema_version": GENERATION_SCHEMA,
        "candidate_id": candidate_id,
        **identity,
        "is_runtime_gate": False,
        "raw_response_text": "raw",
        "raw_response_sha256": _sha256_text("raw"),
        "raw_response_token_ids": token_ids,
        "raw_response_token_ids_sha256": _sha256_json(token_ids),
        "raw_response_token_count": len(token_ids),
        "final_content": parsed,
        "final_content_sha256": _sha256_text(parsed),
        "parsed_final_exact": parsed,
        "parsed_final_sha256": _sha256_text(parsed),
        "parsed_final_token_count": 3,
        "normalized_final_exact": normalized,
        "normalized_final_sha256": _sha256_text(normalized),
        "normalized_final_token_count": 2,
        "normalization_id": LEAN_WRAPPER_NORMALIZATION,
        "normalization_applied": True,
        "normalization_pass_count": 1,
        "normalization_idempotent": True,
    }
    path = tmp_path / "generations.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_continuation_generation_records(path)[0]["candidate_id"] == (
        candidate_id
    )

    record["normalized_final_exact"] = "\nexact False.elim"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        load_continuation_generation_records(path)


def test_dual_verification_submits_strict_and_normalized_bytes_once() -> None:
    task = _task("fresh-0", "fresh-composition-valid-v2")
    parsed = "by\nexact True.intro"
    normalized = "\nexact True.intro"
    generation = _generation(task, parsed, normalized, applied=True)
    verifier = _CapturingVerifier()

    record = _verify_dual_record(generation, task, verifier)  # type: ignore[arg-type]

    assert verifier.sources == [
        f"{task.preamble}\n\n{task.declaration} := by\n  {parsed}\n",
        f"{task.preamble}\n\n{task.declaration} := by\n  {normalized}\n",
    ]
    assert record["lean_invocation_count"] == 2
    assert record["shared_identical_submission"] is False
    assert record["verification_outcome_changed_by_normalization"] is True


def test_dual_verification_reuses_one_identical_submission() -> None:
    task = _task("fresh-0", "fresh-composition-valid-v2")
    final = "exact True.intro"
    generation = _generation(task, final, final, applied=False)
    verifier = _CapturingVerifier()

    record = _verify_dual_record(generation, task, verifier)  # type: ignore[arg-type]

    assert len(verifier.sources) == 1
    assert record["lean_invocation_count"] == 1
    assert record["shared_identical_submission"] is True


class _CapturingVerifier:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def _run_source(self, source: str) -> VerificationOutcome:
        self.sources.append(source)
        verified = len(self.sources) % 2 == 0
        return VerificationOutcome(
            category="verified" if verified else "lean_rejected",
            lean_exit_code=0 if verified else 1,
            diagnostics={"stdout": "", "stderr": ""},
            latency_seconds=0.01,
        )


def _generation(
    task: MathiaTask, parsed: str, normalized: str, *, applied: bool
) -> dict[str, object]:
    return {
        "candidate_id": "candidate",
        "arm": "B4",
        "workload": task.workload,
        "task_id": task.task_id,
        "candidate_index": 0,
        "seed": 0,
        "prompt_sha256": "prompt",
        "generation_config_sha256": "generation",
        "parsed_final_exact": parsed,
        "parsed_final_sha256": _sha256_text(parsed),
        "normalized_final_exact": normalized,
        "normalized_final_sha256": _sha256_text(normalized),
        "normalization_id": LEAN_WRAPPER_NORMALIZATION,
        "normalization_applied": applied,
    }


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
