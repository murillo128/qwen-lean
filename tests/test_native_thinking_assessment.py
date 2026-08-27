from pathlib import Path

import pytest

from qwen_lean.native_thinking_assessment import (
    EXPECTED_COUNTS,
    FROZEN_SAMPLING,
    FROZEN_USER_TEMPLATE,
    MODEL_ID,
    MODEL_REVISION,
    MathiaTask,
    NativeThinkingConfig,
    _apparent_natural_language,
    _finish_reason,
    _mcnemar_exact,
    _verify_generation_record,
    analyze_results,
    candidate_identity,
    generation_config_sha256,
    load_mathia_tasks,
    render_user_message,
)
from qwen_lean.verifier import VerificationOutcome

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-native-thinking-ab.json"
MATHIA_ROOT = ROOT.parent / "mathia"


def test_config_freezes_native_thinking_causal_contract() -> None:
    config = NativeThinkingConfig.load(CONFIG_PATH)

    assert config.model == {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "license": "Apache-2.0",
    }
    assert config.sampling == FROZEN_SAMPLING
    assert config.value["arms"] == {
        "t0": {"enable_thinking": False},
        "t1": {"enable_thinking": True},
    }
    assert config.value["prompt"]["user_template"] == FROZEN_USER_TEMPLATE
    assert config.engine["reasoning_parser"] == "qwen3"
    assert config.engine["dtype"] == "bfloat16"
    assert config.engine["quantization"] is None


@pytest.mark.skipif(
    not MATHIA_ROOT.exists(), reason="local accepted Mathia corpus absent"
)
def test_mathia_population_is_bound_to_frozen_files() -> None:
    config = NativeThinkingConfig.load(CONFIG_PATH)
    tasks, binding = load_mathia_tasks(config, MATHIA_ROOT)

    assert len(tasks) == 611
    assert binding["counts"] == EXPECTED_COUNTS
    assert {task.workload for task in tasks} == set(EXPECTED_COUNTS)
    assert all(task.intuition and task.declaration and task.preamble for task in tasks)


def test_user_prompt_is_exact_and_candidate_identity_is_arm_paired() -> None:
    config = NativeThinkingConfig.load(CONFIG_PATH)
    task = _task("task-a", "minif2f-valid-clean-v2")
    prompt = render_user_message(task)

    assert prompt == FROZEN_USER_TEMPLATE.replace(
        "<FROZEN_INTUITION_BYTES>", task.intuition
    ).replace("<UNCHANGED_DECLARATION>", task.declaration)
    assert prompt.count(task.intuition) == 1
    assert prompt.count(task.declaration) == 1

    t0_id, t0 = candidate_identity(
        config,
        arm="t0",
        task=task,
        prompt_sha256="prompt-hash",
        candidate_index=3,
    )
    t1_id, t1 = candidate_identity(
        config,
        arm="t1",
        task=task,
        prompt_sha256="prompt-hash",
        candidate_index=3,
    )
    assert t0_id != t1_id
    assert t0["seed"] == t1["seed"] == 3
    assert t0["prompt_sha256"] == t1["prompt_sha256"] == "prompt-hash"
    assert t0["generation_config_sha256"] == generation_config_sha256(config)


def test_verification_submits_exact_final_channel_without_repair() -> None:
    task = _task("task-a", "fresh-composition-valid-v2")
    final = "```lean\nexact True.intro\n```\nThis is mixed output.  "
    generation = {
        "candidate_id": "candidate-a",
        "arm": "t1",
        "workload": task.workload,
        "task_id": task.task_id,
        "candidate_index": 0,
        "seed": 0,
        "prompt_sha256": "prompt",
        "generation_config_sha256": "generation",
        "final_content": final,
        "final_content_sha256": "final-hash",
    }
    verifier = _CapturingVerifier()

    record = _verify_generation_record(generation, task, verifier)  # type: ignore[arg-type]

    assert verifier.source == (
        f"{task.preamble}\n\n{task.declaration} := by\n  {final}\n"
    )
    assert record["final_content_submitted_without_repair"] is True
    assert record["category"] == "lean_rejected"


def test_paired_analysis_reports_quality_interface_and_overlap() -> None:
    config = NativeThinkingConfig.load(CONFIG_PATH)
    tasks = [
        _task("mini", "minif2f-valid-clean-v2"),
        _task("fresh", "fresh-composition-valid-v2"),
    ]
    rows = []
    # T0 solves mini only; T1 solves fresh only. This yields a symmetric McNemar
    # table and zero paired coverage delta while changing output diversity.
    for arm in ("t0", "t1"):
        for task in tasks:
            for candidate_index in range(4):
                verified = candidate_index == 0 and (
                    (arm == "t0" and task.task_id == "mini")
                    or (arm == "t1" and task.task_id == "fresh")
                )
                text = (
                    f"exact proof_{arm}_{task.task_id}_{candidate_index}"
                    if arm == "t1"
                    else "exact shared"
                )
                rows.append(
                    {
                        "arm": arm,
                        "workload": task.workload,
                        "task_id": task.task_id,
                        "candidate_index": candidate_index,
                        "reasoning_content": "reason" if arm == "t1" else None,
                        "reasoning_token_count": 2 if arm == "t1" else 0,
                        "final_content": text,
                        "final_token_count": 3,
                        "raw_response_token_count": 5 if arm == "t1" else 3,
                        "finish_reason": "eos",
                        "verification": {
                            "category": "verified" if verified else "lean_rejected"
                        },
                    }
                )

    analysis = analyze_results(config, tasks, rows)
    combined = analysis["combined"]

    assert combined["arms"]["t0"]["pass_at_k"] == {
        "pass@1": 0.125,
        "pass@4": 0.5,
    }
    assert combined["arms"]["t1"]["pass_at_k"] == {
        "pass@1": 0.125,
        "pass@4": 0.5,
    }
    assert combined["paired"]["solved_at_4_overlap"] == {
        "t0_only": 1,
        "t1_only": 1,
        "both": 0,
        "neither": 0,
    }
    assert combined["paired"]["mcnemar_exact_two_sided"]["p_value"] == 1.0
    assert (
        combined["arms"]["t0"]["interface_diagnostics"]["reasoning_present"]["count"]
        == 0
    )
    assert (
        combined["arms"]["t1"]["interface_diagnostics"]["reasoning_present"]["count"]
        == 8
    )


def test_diagnostic_helpers_are_deterministic_and_nonrepairing() -> None:
    assert _finish_reason("stop") == "eos"
    assert _finish_reason("length") == "token_limit"
    assert _mcnemar_exact(0, 3) == pytest.approx(0.25)
    assert _apparent_natural_language("Therefore we have the desired result.")
    assert not _apparent_natural_language("exact Nat.add_comm _ _")


def _task(task_id: str, workload: str) -> MathiaTask:
    return MathiaTask(
        task_id=task_id,
        workload=workload,
        preamble="import Mathlib",
        declaration=f"theorem {task_id.replace('-', '_')} : True",
        declaration_name=task_id.replace("-", "_"),
        intuition="Use the identity element.",
        intuition_sha256="intuition-hash",
        theorem_sha256="theorem-hash",
    )


class _CapturingVerifier:
    source: str | None = None

    def _run_source(self, source: str) -> VerificationOutcome:
        self.source = source
        return VerificationOutcome(
            category="lean_rejected",
            lean_exit_code=1,
            diagnostics={"stdout": "", "stderr": "expected rejection"},
            latency_seconds=0.01,
        )
