from pathlib import Path

from qwen_lean.artifacts import read_artifacts, write_artifacts
from qwen_lean.schema import (
    CandidateResult,
    PHASE1_RESULT_SCHEMA_VERSION,
    RunMetadata,
)


ROOT = Path(__file__).resolve().parents[1]


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


def test_phase0_committed_artifacts_remain_readable() -> None:
    metadata, results = read_artifacts(ROOT / "evidence/phase0/fixture")

    assert metadata.schema_version == "phase0-v1"
    assert results
    assert all(result.generated_token_count is None for result in results)
    assert all(result.finish_reason is None for result in results)


def test_phase1_multi_candidate_artifacts_retain_indices(tmp_path: Path) -> None:
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source="miniF2F-valid",
        prompt_format_id="whole-proof-v1",
        lean_toolchain="leanprover/lean4:v4.27.0",
        mathlib_revision="a3a10db0e9d66acbebf76c5e6a135066525ac900",
        verifier_timeout_seconds=30.0,
        candidates_per_task=8,
    )
    results = [
        CandidateResult(
            task_id="identity",
            candidate_id=f"model-{index}",
            candidate_index=index,
            candidate_text="exact x",
            category="verified",
            lean_exit_code=0,
            diagnostics={"stdout": "", "stderr": ""},
            generation_latency_seconds=0.2,
            verification_latency_seconds=0.1,
            total_latency_seconds=0.3,
            generated_token_count=2,
            finish_reason="eos",
        )
        for index in range(8)
    ]

    write_artifacts(tmp_path, metadata, results)
    _, loaded_results = read_artifacts(tmp_path)

    assert [result.candidate_index for result in loaded_results] == list(range(8))
