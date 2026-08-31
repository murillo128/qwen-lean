from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from test_generalist_v2_dataset import _record

from qwen_lean.baseline import GeneratedCandidate
from qwen_lean.generalist_v2 import GeneralistV2Config
from qwen_lean.generalist_v2_dataset import generalist_variants, sha256_file
from qwen_lean.generalist_v2_evaluation import (
    _checkpoint_adapter_spec,
    _compact_checkpoint_workload,
    _compact_extended_workload,
    _compact_final_run,
    _compact_incomplete_deepseek_fresh,
    _deepseek_final_lane_phase1_config,
    _final_phase1_config,
    _normalized_verified_proof_sha256,
    _paired_solved,
    _source_position_verification_task,
    _summarize_extended_raw_candidate_evidence,
    _synthetic_task,
    _text_sha256,
    _write_extended_raw_candidate_evidence,
    compact_extended_validation_evidence,
)
from qwen_lean.metrics import pass_at_k
from qwen_lean.schema import CandidateResult, TaskRecord

ROOT = Path(__file__).resolve().parents[1]


def test_verified_proof_hash_ignores_comments_and_whitespace() -> None:
    assert _normalized_verified_proof_sha256(
        "by\n  /- irrelevant -/ exact True.intro"
    ) == _normalized_verified_proof_sha256("by exact   True.intro -- irrelevant\n")


def test_synthetic_evaluation_uses_persisted_import_context() -> None:
    record = _record()
    task = _synthetic_task(record)

    assert task.id == record.statement_id
    assert task.preamble == "import Mathlib"
    assert task.declaration == record.canonical_declaration
    assert generalist_variants(record)[0].completion == "trivial"


def test_source_position_verifier_preserves_exact_source_prefix(
    tmp_path: Path,
) -> None:
    source = (
        "import Mathlib\n\n"
        "namespace Fixture\n\n"
        "variable (localContext : True)\n\n"
        "theorem fixture : True := by\n"
        "  exact localContext\n"
    )
    source_path = tmp_path / ".lake/packages/mathlib/Fixture.lean"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    value = _record().to_dict()
    value["provenance"] = "real-mathlib"
    value["derivation_family_fingerprint"] = None
    value["generator_family"] = None
    value["structural_class"] = None
    value["normalized_proof_dag"] = None
    value["source_lemma_ids"] = []
    value["environment"]["repository"] = (
        "https://github.com/leanprover-community/mathlib4"
    )
    value["environment"]["revision"] = "revision"
    value["environment"]["file_path"] = "Fixture.lean"
    value["environment"]["source_span"] = {
        "start": {"line": 7, "column": 1},
        "end": {"line": 8, "column": 21},
    }
    record = type(_record()).from_dict(value)

    task = _source_position_verification_task(record, tmp_path)

    assert task.preamble.endswith("variable (localContext : True)")
    assert "theorem fixture" not in task.preamble
    assert task.declaration == "theorem fixture : True"


def test_checkpoint_adapter_spec_requires_pinned_unmerged_identity(
    tmp_path: Path,
) -> None:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-4B-Base",
                "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                "r": 16,
                "target_modules": config.lora["target_regex"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")

    adapter = _checkpoint_adapter_spec(config, "Q2", tmp_path)

    assert adapter.adapter_id == "qwen-lean-generalist-v2-q2"
    assert adapter.base_model_revision == config.model["model_revision"]
    with pytest.raises(ValueError, match="must be Q1-Q4"):
        _checkpoint_adapter_spec(config, "Q0", tmp_path)


def test_final_deepseek_config_uses_same_frozen_context_and_pinned_identity() -> None:
    generalist = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    phase1 = _final_phase1_config(
        ROOT / "config/deepseek-prover-v2-7b-generalist-v2.json",
        generalist,
        model_label="deepseek",
    )

    assert phase1.model["model_id"] == "deepseek-ai/DeepSeek-Prover-V2-7B"
    assert phase1.model["model_revision"] == (
        "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b"
    )
    assert phase1.engine["max_model_len"] == 24576
    assert phase1.engine["max_num_seqs"] == 4
    assert phase1.engine["cpu_offload_gb"] == 8.0
    assert phase1.value["assessment"]["context_census"]["scope"] == (
        "validation-only-before-checkpoint-freeze"
    )
    assert len(
        phase1.value["assessment"]["context_census"]["maximum_task"]["task_id"]
    ) == 64
    with pytest.raises(ValueError, match="context or BF16 offload"):
        _final_phase1_config(
            ROOT / "config/deepseek-prover-v2-7b-assessment.json",
            generalist,
            model_label="deepseek",
        )


def test_final_deepseek_lane_runtime_preserves_exact_context_without_full_offload() -> None:
    generalist = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    phase1 = _final_phase1_config(
        ROOT / "config/deepseek-prover-v2-7b-generalist-v2.json",
        generalist,
        model_label="deepseek",
    )

    lane = _deepseek_final_lane_phase1_config(
        phase1, "minif2f-valid-clean-v2"
    )

    assert lane.engine["max_model_len"] == 2048
    assert lane.engine["cpu_offload_gb"] == 0.0
    assert lane.value["assessment"]["lane_runtime"][
        "max_prompt_plus_generation_tokens"
    ] == 1510
    assert lane.sampling == phase1.sampling

    with pytest.raises(ValueError, match="unknown DeepSeek final workload"):
        _deepseek_final_lane_phase1_config(
            phase1, "fresh-composition-test-v2"
        )
    with pytest.raises(ValueError, match="unknown DeepSeek final workload"):
        _deepseek_final_lane_phase1_config(phase1, "riemann-fresh-test-v2")


def test_paired_historical_outcomes_use_exact_two_sided_mcnemar() -> None:
    paired = _paired_solved(
        [True, False, True, False],
        [True, True, False, True],
    )

    assert paired == {
        "both_solved": 1,
        "candidate_only": 2,
        "reference_only": 1,
        "neither_solved": 0,
        "paired_wins_candidate_minus_reference": 1,
        "exact_two_sided_mcnemar_p": 1.0,
    }


def test_compact_checkpoint_workload_binds_adapter_and_task_order(
    tmp_path: Path,
) -> None:
    adapter = {
        "adapter_id": "qwen-lean-generalist-v2-q2",
        "adapter_rank": 16,
        "base_model_id": "Qwen/Qwen3.5-4B-Base",
        "base_model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "adapter_config_sha256": "config-hash",
        "adapter_model_sha256": "adapter-hash",
    }
    (tmp_path / "generation-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-checkpoint-generation-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "ordered_task_ids_sha256": "task-hash",
                "candidate_count": 8,
                "generation_error_count": 0,
                "adapter": adapter,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run.json").write_text(
        json.dumps({"selected_adapter_binding": adapter}), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-checkpoint-verification-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "task_count": 1,
                "candidate_count": 8,
                "complete": True,
                "infrastructure_error_count": 0,
                "tasks_with_verified_candidate": 1,
                "pass_at_k": {"pass@1": 0.5, "pass@4": 1.0, "pass@8": 1.0},
                "category_counts": {"verified": 4, "lean_rejected": 4},
                "finish_reason_counts": {"eos": 8},
                "exact_target_candidate_count": 1,
                "exact_target_task_count": 1,
                "per_task": [{"task_id": "task-a", "verified_candidate_count": 4}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "results.jsonl").write_text("result\n", encoding="utf-8")
    (tmp_path / "generations.jsonl").write_text("generation\n", encoding="utf-8")
    generation_metadata = json.loads(
        (tmp_path / "generation-metadata.json").read_text(encoding="utf-8")
    )
    generation_metadata["generation_sha256"] = sha256_file(
        tmp_path / "generations.jsonl"
    )
    (tmp_path / "generation-metadata.json").write_text(
        json.dumps(generation_metadata), encoding="utf-8"
    )

    compact = _compact_checkpoint_workload(
        tmp_path, "Q2", "fixture", 1, "task-hash", "adapter-hash"
    )

    assert compact["verified_counts"] == [4]
    assert compact["per_task"] == [{"task_id": "task-a", "verified_candidate_count": 4}]
    assert compact["adapter_model_sha256"] == "adapter-hash"
    with pytest.raises(ValueError, match="selection workload is incomplete"):
        _compact_checkpoint_workload(
            tmp_path, "Q2", "fixture", 1, "different-task-hash", "adapter-hash"
        )


def test_compact_final_run_requires_frozen_identity_and_per_task_outcomes(
    tmp_path: Path,
) -> None:
    generation = {
        "schema_version": "generalist-v2-final-generation-v1",
        "evaluation_profile": "postselection-final-n8",
        "model_label": "base",
        "selected_checkpoint": "Q2",
        "workload_id": "fixture",
        "model_id": "Qwen/Qwen3.5-4B-Base",
        "model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "task_count": 1,
        "candidate_count": 8,
        "generation_error_count": 0,
        "prompt_format_id": "lean-sft-v2-raw-whole-proof",
        "sampling": {"candidates_per_task": 8},
        "inference_execution": "project-controlled-local-cuda",
        "first_complete_result_overwrite_protected": True,
        "final_only_workload": False,
        "ordered_task_ids_sha256": "task-hash",
        "adapter": None,
    }
    (tmp_path / "generation-metadata.json").write_text(
        json.dumps(generation), encoding="utf-8"
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "model_id": generation["model_id"],
                "model_revision": generation["model_revision"],
                "selected_adapter_binding": None,
                "candidates_per_task": 8,
                "runtime": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-final-verification-v1",
                "evaluation_profile": "postselection-final-n8",
                "model_label": "base",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "task_count": 1,
                "candidate_count": 8,
                "complete": True,
                "infrastructure_error_count": 0,
                "tasks_with_verified_candidate": {"count": 1, "rate": 1.0},
                "pass_at_k": {"pass@1": 0.25, "pass@4": 1.0, "pass@8": 1.0},
                "category_counts": {"verified": 2, "lean_rejected": 6},
                "finish_reason_counts": {"eos": 8},
                "per_task": [{"task_id": "task-a", "verified_candidate_count": 2}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "generations.jsonl").write_text("generation\n", encoding="utf-8")
    (tmp_path / "results.jsonl").write_text("result\n", encoding="utf-8")
    generation["generation_sha256"] = sha256_file(tmp_path / "generations.jsonl")
    (tmp_path / "generation-metadata.json").write_text(
        json.dumps(generation), encoding="utf-8"
    )

    compact = _compact_final_run(
        tmp_path,
        model_label="base",
        selected_checkpoint="Q2",
        workload_id="fixture",
        expected_task_count=1,
    )

    assert compact["per_task"] == [{"task_id": "task-a", "verified_candidate_count": 2}]
    assert compact["first_complete_result_overwrite_protected"] is True


def test_incomplete_deepseek_fresh_is_diagnostic_and_never_scored(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete.json"
    value = {
        "schema_version": "generalist-v2-deepseek-fresh-incomplete-v1",
        "status": "INCOMPLETE / DIAGNOSTIC ONLY / NOT FOR MODEL-QUALITY COMPARISON",
        "model_id": "deepseek-ai/DeepSeek-Prover-V2-7B",
        "model_revision": "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b",
        "workload_id": "fresh-composition-test-v2",
        "expected_task_count": 415,
        "expected_candidate_count": 3320,
        "decision_point_completed_candidates": 200,
        "last_observed_completed_candidates": 208,
        "partial_candidate_records_materialized": 0,
        "full_benchmark_metrics_computed": False,
        "extrapolation_performed": False,
        "gpu_released": True,
        "stop_reason": "deliberate compute/value triage",
        "serialization_note": "vLLM buffered results until full return",
        "raw_operational_log": {"sha256": "log-sha256"},
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    compact = _compact_incomplete_deepseek_fresh(path)

    assert compact["last_observed_completed_candidates"] == 208
    assert compact["full_benchmark_metrics_computed"] is False
    assert "pass_at_k" not in compact
    value["pass_at_k"] = {"pass@8": 0.0}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic contract"):
        _compact_incomplete_deepseek_fresh(path)


def test_compact_extended_workload_reports_curve_and_marginal_gains(
    tmp_path: Path,
) -> None:
    adapter = {"adapter_model_sha256": "adapter-hash"}
    (tmp_path / "generation-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-extended-generation-v1",
                "evaluation_profile": "extended-search-budget-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "ordered_task_ids_sha256": "task-hash",
                "candidate_count": 64,
                "generation_error_count": 0,
                "generate_all_candidates_without_early_stop": True,
                "adapter": adapter,
                "sampling": {"seed": 0},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run.json").write_text(
        json.dumps({"selected_adapter_binding": adapter, "candidates_per_task": 64}),
        encoding="utf-8",
    )
    pass_metrics = {f"pass@{k}": pass_at_k(64, 2, k) for k in (1, 2, 4, 8, 16, 32, 64)}
    solved = {
        "solved@1": 0,
        "solved@2": 0,
        "solved@4": 0,
        "solved@8": 0,
        "solved@16": 1,
        "solved@32": 1,
        "solved@64": 1,
    }
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "generalist-v2-extended-verification-v1",
                "evaluation_profile": "extended-search-budget-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "task_count": 1,
                "candidate_count": 64,
                "complete": True,
                "infrastructure_error_count": 0,
                "pass_at_k": pass_metrics,
                "tasks_solved_within_k": solved,
                "all_candidates_verified_without_early_stop": True,
                "category_counts": {"verified": 2, "lean_rejected": 62},
                "finish_reason_counts": {"eos": 64},
                "per_task": [{"task_id": "task-a", "verified_candidate_count": 2}],
            }
        ),
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw-candidates.jsonl.gz"
    raw_rows = []
    for index in range(64):
        verified = index in {9, 47}
        candidate_text = "by\n  exact True.intro" if verified else "by\n  contradiction"
        raw_rows.append(
            {
                "schema_version": "generalist-v2-extended-candidate-v1",
                "checkpoint_id": "Q2",
                "workload_id": "fixture",
                "model_id": "Qwen/Qwen3.5-4B-Base",
                "model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                "adapter_model_sha256": "adapter-hash",
                "task_id": "task-a",
                "candidate_id": f"model-{index}",
                "candidate_index": index,
                "sampling_seed": 0,
                "candidate_text": candidate_text,
                "candidate_text_sha256": _text_sha256(candidate_text),
                "generated_token_count": 4,
                "finish_reason": "eos",
                "generation_latency_seconds": 0.1,
                "category": "verified" if verified else "lean_rejected",
                "lean_exit_code": 0 if verified else 1,
                "diagnostics": {"stdout": "", "stderr": ""},
                "verification_latency_seconds": 0.1,
                "total_latency_seconds": 0.2,
                "normalized_proof_sha256": (
                    _normalized_verified_proof_sha256(candidate_text)
                    if verified
                    else None
                ),
            }
        )
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows)
        )
    density = _summarize_extended_raw_candidate_evidence(
        raw_path,
        checkpoint_id="Q2",
        workload_id="fixture",
        expected_task_ids=["task-a"],
        expected_adapter_model_sha256="adapter-hash",
        sampling_seed=0,
    )
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["extended_candidate_evidence"] = {
        "artifact": "raw-candidates.jsonl.gz",
        "sha256": sha256_file(raw_path),
        **density,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "results.jsonl").write_text("result\n", encoding="utf-8")
    (tmp_path / "generations.jsonl").write_text("generation\n", encoding="utf-8")

    compact = _compact_extended_workload(
        tmp_path, "Q2", "fixture", 1, "task-hash", "adapter-hash"
    )

    assert compact["marginal_pass_at_k"] == pytest.approx(
        {
            "delta_8_to_16": pass_metrics["pass@16"] - pass_metrics["pass@8"],
            "delta_16_to_32": pass_metrics["pass@32"] - pass_metrics["pass@16"],
            "delta_32_to_64": pass_metrics["pass@64"] - pass_metrics["pass@32"],
        }
    )
    assert compact["marginal_tasks_solved"] == {
        "delta_8_to_16": 1,
        "delta_16_to_32": 0,
        "delta_32_to_64": 0,
    }
    assert compact["raw_candidate_evidence"]["verified_candidate_count"] == 2
    assert compact["raw_candidate_evidence"]["unique_verified_proof_count"] == 1
    assert compact["raw_candidate_evidence"]["verified_duplication_fraction"] == 0.5
    raw_rows[0]["candidate_text_sha256"] = "tampered"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows)
        )
    with pytest.raises(ValueError, match="raw candidate identity"):
        _compact_extended_workload(
            tmp_path, "Q2", "fixture", 1, "task-hash", "adapter-hash"
        )


def test_extended_raw_candidates_are_deterministically_compressed(
    tmp_path: Path,
) -> None:
    task = TaskRecord(
        id="task-a",
        preamble="import Mathlib",
        declaration="theorem task_a : True",
        declaration_name="task_a",
    )
    generated = [
        GeneratedCandidate(
            task=task,
            candidate_index=index,
            text="by\n  contradiction",
            token_count=2,
            finish_reason="eos",
            generation_latency_seconds=0.1,
        )
        for index in range(64)
    ]
    results = [
        CandidateResult(
            task_id=task.id,
            candidate_id=f"model-{index}",
            candidate_index=index,
            candidate_text="by\n  contradiction",
            category="lean_rejected",
            lean_exit_code=1,
            diagnostics={"stdout": "", "stderr": ""},
            generation_latency_seconds=0.1,
            verification_latency_seconds=0.1,
            total_latency_seconds=0.2,
            generated_token_count=2,
            finish_reason="eos",
        )
        for index in range(64)
    ]
    metadata = {
        "checkpoint_id": "Q2",
        "workload_id": "fixture",
        "sampling": {"seed": 0},
        "adapter": {"adapter_model_sha256": "adapter-hash"},
    }
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    density = _write_extended_raw_candidate_evidence(
        first, generated, results, metadata
    )
    _write_extended_raw_candidate_evidence(second, generated, results, metadata)

    assert density["raw_candidate_count"] == 64
    assert density["verified_candidate_count"] == 0
    assert sha256_file(first) == sha256_file(second)
    assert len(list(gzip.open(first, "rt", encoding="utf-8"))) == 64


def test_extended_validation_uses_only_the_n8_frozen_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screening = {
        "schema_version": "generalist-v2-checkpoint-selection-v1",
        "status": "frozen",
        "selection": {
            "selected_checkpoint": "Q3",
            "strongest_runner_up": "Q2",
        },
        "checkpoints": {
            checkpoint: {
                "adapter_model_sha256": f"{checkpoint}-hash",
                "workloads": {
                    workload: {"ordered_task_ids_sha256": f"{workload}-hash"}
                    for workload in (
                        "fresh-composition-valid-v2",
                        "minif2f-valid-clean-v2",
                    )
                },
            }
            for checkpoint in ("Q1", "Q2", "Q3", "Q4")
        },
    }
    screening_path = tmp_path / "screening.json"
    screening_path.write_text(json.dumps(screening), encoding="utf-8")

    monkeypatch.setattr(
        "qwen_lean.generalist_v2_evaluation._compact_extended_workload",
        lambda *args, **kwargs: {"pass_at_k": {"pass@64": 0.5}},
    )
    output = tmp_path / "extended.json"
    evidence = compact_extended_validation_evidence(
        GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json"),
        screening_path,
        tmp_path / "runs",
        output,
    )

    assert evidence["evaluated_checkpoint"]["checkpoint_id"] == "Q3"
    assert evidence["base_control_evaluated_at_n64"] is False
    assert evidence["runner_up_evaluated_at_n64"] is False
    assert evidence["test_workloads_evaluated_at_n64"] is False
    screening["selection"]["selected_checkpoint"] = "Q0"
    screening_path.write_text(json.dumps(screening), encoding="utf-8")
    with pytest.raises(ValueError, match="complete n=8 screening"):
        compact_extended_validation_evidence(
            GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json"),
            screening_path,
            tmp_path / "runs",
            output,
        )
