from pathlib import Path

from qwen_lean.artifacts import read_artifacts, write_artifacts
from qwen_lean.schema import CandidateResult, RunMetadata


def test_result_artifacts_round_trip(tmp_path: Path) -> None:
    metadata = RunMetadata(
        candidate_source="fixture",
        task_source="fixture-v1",
        prompt_format_id="whole-proof-v1",
        lean_toolchain="leanprover/lean4:v4.32.0",
        mathlib_revision="v4.32.0",
        verifier_timeout_seconds=30.0,
    )
    result = CandidateResult(
        task_id="identity",
        candidate_id="candidate-0",
        candidate_index=0,
        candidate_text="exact x",
        category="verified",
        lean_exit_code=0,
        diagnostics={"stdout": "", "stderr": ""},
        generation_latency_seconds=None,
        verification_latency_seconds=0.1,
        total_latency_seconds=0.1,
    )
    write_artifacts(tmp_path, metadata, [result])

    loaded_metadata, loaded_results = read_artifacts(tmp_path)

    assert loaded_metadata == metadata
    assert loaded_results == [result]
