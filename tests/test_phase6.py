import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_lean.minif2f import (
    Phase1Config,
    materialize_benchmark_source,
    materialize_validation_source,
)
from qwen_lean.phase2_schema import (
    MathlibProofRecord,
    SourcePosition,
    SourceSpan,
    TokenLengths,
)
from qwen_lean.phase6 import (
    Phase6Config,
    differential_gap_bootstrap,
    generalization_gaps,
    paired_task_bootstrap,
    select_phase6_train_workload,
    summarize_phase6_train_results,
    target_exact,
    validate_adapter_artifact_files,
)
from qwen_lean.schema import CandidateResult

ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _span() -> SourceSpan:
    return SourceSpan(SourcePosition(1, 1), SourcePosition(1, 2))


def _record(
    index: int,
    *,
    completion: str = "exact True.intro",
    declaration: str | None = None,
) -> MathlibProofRecord:
    return MathlibProofRecord(
        schema_version="mathlib-whole-proof-v1",
        id=hashlib.sha256(f"phase6-record-{index}".encode()).hexdigest(),
        source_repository="https://github.com/leanprover-community/mathlib4",
        source_revision="81a5d257c8e410db227a6665ed08f64fea08e997",
        file_path=f"Mathlib/Test/{index}.lean",
        declaration_name=f"Example.phase6_{index}",
        declaration_kind="theorem",
        source_span=_span(),
        declaration_span=_span(),
        proof_span=_span(),
        declaration=declaration or f"theorem phase6_{index} : True",
        proof="by\n  exact True.intro",
        completion=completion,
        premises=(),
        file_group=f"Mathlib/Test/{index}.lean",
        component_id=hashlib.sha256(f"component-{index}".encode()).hexdigest(),
        split="train",
        statement_fingerprint=hashlib.sha256(f"statement-{index}".encode()).hexdigest(),
        token_lengths=TokenLengths(1, 1, 2, 2, 2),
    )


def _result(task_id: str, index: int, *, verified: bool, text: str) -> CandidateResult:
    return CandidateResult(
        task_id=task_id,
        candidate_id=f"model-{index}",
        candidate_index=index,
        candidate_text=text,
        category="verified" if verified else "lean_rejected",
        lean_exit_code=0 if verified else 1,
        diagnostics={"stdout": "", "stderr": ""},
        generation_latency_seconds=1.0,
        verification_latency_seconds=1.0,
        total_latency_seconds=2.0,
        generated_token_count=2,
        finish_reason="eos",
    )


def test_phase6_config_freezes_candidate_and_both_generation_contracts() -> None:
    config = Phase6Config.load(ROOT / "config/phase6-eval.json")

    assert config.reference["logical_id"] == "reference-sft-v1"
    assert config.reference["eligible_candidate_count"] == 1
    assert config.reference["phase6_metrics_may_change_identity"] is False
    assert config.train_generation["candidates_per_task"] == 4
    assert config.phase1_test_config().sampling["candidates_per_task"] == 8

    changed = copy.deepcopy(config.value)
    changed["reference_candidate"]["adapter"]["hub_revision"] = "mutable"
    with pytest.raises(ValueError, match="hub_revision"):
        Phase6Config(path=config.path, value=changed).validate()


def test_phase6_train_selector_uses_exact_membership_and_not_targets_or_order() -> None:
    records = [_record(index) for index in range(8)]
    member_ids = [record.id for record in records[:7]]
    changed_targets = [
        replace(record, completion="different target " * (index + 1))
        for index, record in enumerate(records)
    ]

    selected, eligible = select_phase6_train_workload(
        records,
        WordTokenizer(),
        phase5_ordered_member_ids=member_ids,
        expected_examples=4,
        generation_tokens=20,
        maximum_tokens=80,
    )
    reversed_selected, reversed_eligible = select_phase6_train_workload(
        reversed(changed_targets),
        WordTokenizer(),
        phase5_ordered_member_ids=member_ids,
        expected_examples=4,
        generation_tokens=20,
        maximum_tokens=80,
    )

    assert eligible == reversed_eligible == 7
    assert [item.record_id for item in selected] == [
        item.record_id for item in reversed_selected
    ]
    assert len(selected) == 4
    assert {item.record_id for item in selected} <= set(member_ids)
    assert records[7].id not in {item.record_id for item in selected}


def test_phase6_train_selector_enforces_fixed_prompt_generation_boundary() -> None:
    short = _record(1)
    long = _record(2, declaration="theorem too_long : " + "True " * 70)

    selected, eligible = select_phase6_train_workload(
        [short, long],
        WordTokenizer(),
        phase5_ordered_member_ids=[short.id, long.id],
        expected_examples=1,
        generation_tokens=20,
        maximum_tokens=60,
    )

    assert eligible == 1
    assert [item.record_id for item in selected] == [short.id]
    with pytest.raises(ValueError, match="only 1"):
        select_phase6_train_workload(
            [short, long],
            WordTokenizer(),
            phase5_ordered_member_ids=[short.id, long.id],
            expected_examples=2,
            generation_tokens=20,
            maximum_tokens=60,
        )


def test_phase6_exact_target_allows_only_transport_normalization() -> None:
    assert target_exact("exact h\r\n  ", "exact h\n") is True
    assert target_exact("by\n  exact h", "exact h") is False
    assert target_exact("exact  h", "exact h") is False
    assert target_exact("simpa using h", "exact h") is False


def test_phase6_exact_and_lean_metrics_remain_distinct() -> None:
    task_ids = ["a", "b"]
    results = [
        _result("a", 0, verified=True, text="target-a"),
        _result("a", 1, verified=True, text="alternative-a"),
        _result("a", 2, verified=False, text="bad-a-2"),
        _result("a", 3, verified=False, text="bad-a-3"),
        _result("b", 0, verified=True, text="alternative-b"),
        _result("b", 1, verified=False, text="bad-b-1"),
        _result("b", 2, verified=False, text="bad-b-2"),
        _result("b", 3, verified=False, text="bad-b-3"),
    ]
    exact = {
        (result.task_id, result.candidate_index): result.candidate_text == "target-a"
        for result in results
    }

    summary = summarize_phase6_train_results(
        results, expected_task_ids=task_ids, target_exact_by_candidate=exact
    )

    assert summary["pass_at_k"] == {"pass@1": 0.375, "pass@4": 1.0}
    assert summary["exact_target_pass_at_k"] == {"pass@1": 0.125, "pass@4": 0.5}
    assert summary["verified_non_exact_candidates"]["count"] == 2
    assert summary["exact_target_but_not_verified_count"] == 0
    assert summary["phase6_train_integrity_passed"] is True


def test_phase6_generalization_gap_formulas_are_fixed() -> None:
    gaps = generalization_gaps(
        {"pass@1": 0.1, "pass@4": 0.2},
        {"pass@1": 0.4, "pass@4": 0.6},
        {"pass@1": 0.05, "pass@4": 0.1},
        {"pass@1": 0.15, "pass@4": 0.25},
    )

    assert gaps["pass@1"] == {
        "base_train_gap": pytest.approx(0.05),
        "sft_train_gap": pytest.approx(0.25),
        "train_sft_lift": pytest.approx(0.3),
        "heldout_sft_lift": pytest.approx(0.1),
        "differential_gap": pytest.approx(0.2),
    }


def test_phase6_bootstrap_is_seeded_task_paired_and_workload_independent() -> None:
    base = [0, 1, 0, 2]
    adapter = [1, 1, 2, 3]
    first = paired_task_bootstrap(
        base,
        adapter,
        candidates_per_task=4,
        ks=(1, 4),
        resamples=100,
        seed=0,
    )
    second = paired_task_bootstrap(
        base,
        adapter,
        candidates_per_task=4,
        ks=(1, 4),
        resamples=100,
        seed=0,
    )
    identical = paired_task_bootstrap(
        base,
        base,
        candidates_per_task=4,
        ks=(1,),
        resamples=100,
        seed=0,
    )
    differential = differential_gap_bootstrap(
        base,
        adapter,
        [0, 0, 1],
        [0, 1, 1],
        resamples=100,
        seed=0,
    )

    assert first == second
    assert first["task_count"] == 4
    assert identical["metrics"]["pass@1"]["delta_adapter_minus_base"]["ci95"] == [
        0.0,
        0.0,
    ]
    assert differential["train_task_count"] == 4
    assert differential["heldout_task_count"] == 3


def test_minif2f_generic_materializer_keeps_validation_and_includes_solved_test_tasks() -> (
    None
):
    validation_source = """\
import MiniF2F.ProblemImports

theorem valid_a : True := by
  sorry

theorem valid_a.variants.helper : True := by
  sorry
"""
    test_source = """\
import MiniF2F.ProblemImports

theorem solved_test : 194 % 11 = 7 :=
  rfl

theorem placeholder_test : True := by
  sorry
"""

    old = materialize_validation_source(
        validation_source, expected_primary_task_count=1
    )
    new = materialize_benchmark_source(validation_source, expected_primary_task_count=1)
    test = materialize_benchmark_source(test_source, expected_primary_task_count=2)

    assert old == new
    assert [task.id for task in test] == ["solved_test", "placeholder_test"]
    assert test[0].declaration == "theorem solved_test : 194 % 11 = 7"


def test_phase6_test_manifest_is_distinct_and_pinned_to_244_tasks() -> None:
    test = Phase1Config.load(ROOT / "config/phase6-minif2f-test.json")
    valid = Phase1Config.load(ROOT / "config/phase1-minif2f.json")
    test_ids = [
        line
        for line in (ROOT / "config/minif2f-test-task-ids.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    valid_ids = [
        line
        for line in (ROOT / "config/minif2f-valid-task-ids.txt")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    ]

    assert test.benchmark["source_path"] == "MiniF2F/Test.lean"
    assert test.benchmark["revision"] == valid.benchmark["revision"]
    assert len(test_ids) == len(valid_ids) == 244
    assert test_ids != valid_ids
    assert test.sampling == valid.sampling
    assert test.engine == valid.engine


def test_phase6_adapter_file_binding_rejects_mismatch(tmp_path: Path) -> None:
    weights = tmp_path / "adapter_model.safetensors"
    adapter_config = tmp_path / "adapter_config.json"
    weights.write_bytes(b"weights")
    adapter_config.write_text("{}", encoding="utf-8")
    weights_hash = hashlib.sha256(b"weights").hexdigest()
    config_hash = hashlib.sha256(b"{}").hexdigest()

    observed = validate_adapter_artifact_files(
        tmp_path,
        expected_model_sha256=weights_hash,
        expected_config_sha256=config_hash,
    )

    assert observed["adapter_model_sha256"] == weights_hash
    with pytest.raises(ValueError, match="weights differ"):
        validate_adapter_artifact_files(
            tmp_path,
            expected_model_sha256="0" * 64,
            expected_config_sha256=config_hash,
        )
