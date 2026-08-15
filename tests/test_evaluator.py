from pathlib import Path

from qwen_lean.artifacts import read_artifacts
from qwen_lean.evaluator import run_fixture_evaluation


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_run_classifies_all_records_and_continues_after_failure(
    tmp_path: Path,
) -> None:
    metadata, results, mismatches = run_fixture_evaluation(
        ROOT / "fixtures/phase0.json", tmp_path, ROOT
    )

    assert not mismatches
    assert [result.candidate_id for result in results] == [
        "invalid-proof",
        "valid-core-proof",
        "placeholder-proof",
        "empty-proof",
        "valid-mathlib-proof",
    ]
    assert [result.category for result in results] == [
        "lean_rejected",
        "verified",
        "lean_rejected",
        "empty_candidate",
        "verified",
    ]
    assert metadata.prompt_format_id == "whole-proof-v1"
    assert metadata.task_source == "phase0-fixtures-v1"

    loaded_metadata, loaded_results = read_artifacts(tmp_path)
    assert loaded_metadata == metadata
    assert loaded_results == results
