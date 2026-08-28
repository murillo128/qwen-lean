import json
from pathlib import Path

import pytest

from qwen_lean.native_thinking_assessment import MathiaTask, NativeThinkingConfig
from qwen_lean.thinking_budget_natural_max import (
    ARM,
    CAPACITY_ATTEMPT_SCHEMA,
    NaturalMaxConfig,
    _natural_conclusion,
    _natural_paired_table,
    load_capacity_attempts,
    natural_candidate_identity,
    next_capacity_candidate,
)
from qwen_lean.thinking_budget_scaling import (
    SelectedTask,
    ThinkingBudgetScalingConfig,
)

ROOT = Path(__file__).resolve().parents[1]
NATURAL_CONFIG_PATH = ROOT / "config/qwen35-thinking-budget-natural-max.json"
SCALING_CONFIG_PATH = ROOT / "config/qwen35-thinking-budget-scaling.json"
STAGE1_CONFIG_PATH = ROOT / "config/qwen35-native-thinking-ab.json"


def test_natural_max_config_freezes_native_ceiling_and_no_budget() -> None:
    config = NaturalMaxConfig.load(NATURAL_CONFIG_PATH)

    assert config.capacity["model_native_context_ceiling"] == 262144
    assert config.capacity["lattice_quantum"] == 4096
    assert config.capacity["gpu_memory_utilization"] == 0.9
    assert config.capacity["max_num_seqs"] == 1
    assert config.arm["thinking_token_budget"] is None
    assert config.arm["reserved_final_allowance"] == 0


def test_capacity_search_starts_native_and_stops_on_native_success() -> None:
    config = NaturalMaxConfig.load(NATURAL_CONFIG_PATH)

    assert next_capacity_candidate(config, []) == 262144
    assert next_capacity_candidate(config, [_attempt(262144, "passed")]) is None


def test_capacity_search_validates_lower_then_bisects_lattice() -> None:
    config = NaturalMaxConfig.load(NATURAL_CONFIG_PATH)
    attempts = [_attempt(262144, "capacity_failed")]

    assert next_capacity_candidate(config, attempts) == 24576
    attempts.append(_attempt(24576, "passed"))
    assert next_capacity_candidate(config, attempts) == 143360
    attempts.append(_attempt(143360, "passed"))
    assert next_capacity_candidate(config, attempts) == 200704


def test_capacity_search_fails_if_historical_lower_bound_is_unstable() -> None:
    config = NaturalMaxConfig.load(NATURAL_CONFIG_PATH)

    with pytest.raises(RuntimeError, match="known-stable lower bound failed"):
        next_capacity_candidate(
            config,
            [
                _attempt(262144, "capacity_failed"),
                _attempt(24576, "capacity_failed"),
            ],
        )


def test_capacity_attempt_loader_rejects_duplicate_context(tmp_path: Path) -> None:
    config = NaturalMaxConfig.load(NATURAL_CONFIG_PATH)
    first = _attempt(262144, "capacity_failed", config=config)
    path = tmp_path / "capacity-attempts.jsonl"
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(first) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate natural-max capacity attempt"):
        load_capacity_attempts(config, path)


def test_candidate_identity_has_exact_context_remainder_and_no_budget(
    tmp_path: Path,
) -> None:
    config = NaturalMaxConfig.load(NATURAL_CONFIG_PATH)
    scaling = ThinkingBudgetScalingConfig.load(SCALING_CONFIG_PATH)
    stage1 = NativeThinkingConfig.load(STAGE1_CONFIG_PATH)
    selected = _selected(_task("mini-0", "minif2f-valid-clean-v2"))
    capacity_path = tmp_path / "capacity.json"
    capacity_path.write_text(
        json.dumps(
            {
                "machine_supported_context": 262144,
                "selected_gpu_memory_utilization": 0.9,
            }
        ),
        encoding="utf-8",
    )

    candidate_id, identity = natural_candidate_identity(
        config, scaling, stage1, selected, capacity_path
    )

    assert candidate_id.startswith("thinking-budget-natural-max-")
    assert identity["arm"] == ARM
    assert identity["thinking_token_budget"] is None
    assert identity["max_model_len"] == 262144
    assert identity["max_tokens"] == 262144 - selected.rendered_prompt_token_count


def test_paired_table_binds_forced_b16_and_natural_result() -> None:
    selected = _selected(_task("mini-0", "minif2f-valid-clean-v2"))
    b16 = {
        "candidate_id": "b16",
        "reasoning_token_count": 16383,
        "reasoning_exit": "forced_at_budget",
        "parsed_final_sha256": "old-final",
        "parsed_final_exact": "old",
        "finish_reason": "eos",
    }
    natural = {
        "candidate_id": "natural",
        "rendered_prompt_token_count": 100,
        "max_model_len": 262144,
        "max_tokens": 262044,
        "reasoning_token_count": 20000,
        "reasoning_exit": "natural_to_final",
        "reasoning_end_position_token_count": 20001,
        "parsed_final_sha256": "new-final",
        "parsed_final_exact": "exact True.intro",
        "parsed_final_token_count": 3,
        "normalized_final_token_count": 3,
        "normalization_applied": False,
        "finish_reason": "eos",
        "context_exhausted_no_final": False,
        "raw_response_token_count": 20004,
    }
    b16_verification = {
        "candidate_id": "b16",
        "deployed_normalized_interface": {"category": "lean_rejected"},
    }
    natural_verification = {
        "candidate_id": "natural",
        "strict_parsed_interface": {"category": "verified"},
        "deployed_normalized_interface": {"category": "verified"},
        "verification_outcome_changed_by_normalization": False,
    }

    table = _natural_paired_table(
        [selected],
        {"mini-0": b16},
        {"b16": b16_verification},
        {"mini-0": natural},
        {"mini-0": natural_verification},
    )

    assert table[0]["b16"]["forced_at_budget"] is True
    assert table[0]["bnat_max"]["reasoning_end_position_token_count"] == 20001
    assert table[0]["comparison"]["bnat_max_only_verified"] is True


def test_conclusion_detects_new_verified_capability_after_16k() -> None:
    row = {
        "task_id": "mini-0",
        "bnat_max": {
            "reasoning_end_position_token_count": 20001,
            "parsed_final_nonempty": True,
            "context_exhausted_no_final": False,
        },
        "comparison": {"bnat_max_only_verified": True},
    }

    conclusion = _natural_conclusion(
        {"machine_supported_context": 262144}, [row], [row]
    )

    assert conclusion["category"] == ("natural_long_thinking_adds_verified_capability")
    assert conclusion["new_verified_after_more_than_16k_reasoning"] == ["mini-0"]


def _attempt(
    length: int, status: str, *, config: NaturalMaxConfig | None = None
) -> dict[str, object]:
    if config is None:
        return {"requested_max_model_len": length, "status": status}
    identity = {
        "natural_max_config_sha256": _sha256(config.path),
        "runner_source_sha256": _sha256(
            ROOT / "src/qwen_lean/thinking_budget_natural_max.py"
        ),
        "requested_max_model_len": length,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 1,
        "smoke_decode_tokens": 8,
        "probe_task_id": "mini-0",
        "probe_rendered_prompt_sha256": "rendered-sha",
    }
    import hashlib

    payload = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    attempt_id = (
        "natural-max-capacity-"
        + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    )
    return {
        "schema_version": CAPACITY_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        **identity,
        "status": status,
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


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
