import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qwen_lean.schema import TaskRecord
from qwen_lean.verifier import LeanVerifier, VerificationOutcome


ROOT = Path(__file__).resolve().parents[1]
CORE_TASK = TaskRecord(
    id="identity",
    preamble="import Init",
    declaration="theorem identity (P : Prop) (h : P) : P",
    declaration_name="identity",
)


def test_valid_candidate_is_verified() -> None:
    outcome = LeanVerifier(ROOT).verify(CORE_TASK, "exact h")
    assert outcome.category == "verified", outcome.diagnostics
    assert outcome.lean_exit_code == 0


def test_lean_failure_is_rejected_with_diagnostics() -> None:
    outcome = LeanVerifier(ROOT).verify(CORE_TASK, "exact missing_hypothesis")
    assert outcome.category == "lean_rejected"
    assert outcome.lean_exit_code != 0
    assert "missing_hypothesis" in (
        outcome.diagnostics["stdout"] + outcome.diagnostics["stderr"]
    )


def test_empty_candidate_does_not_run_as_success() -> None:
    outcome = LeanVerifier(ROOT).verify(CORE_TASK, " \r\n")
    assert outcome.category == "empty_candidate"
    assert outcome.lean_exit_code is None


@pytest.mark.parametrize("candidate", ["sorry", "admit"])
def test_placeholder_is_not_verified(candidate: str) -> None:
    outcome = LeanVerifier(ROOT).verify(CORE_TASK, candidate)
    assert outcome.category == "lean_rejected", outcome.diagnostics


def test_broken_preamble_is_a_verifier_error() -> None:
    task = TaskRecord(
        id="missing-environment",
        preamble="import MissingPhaseZeroDependency",
        declaration="theorem unreachable (P : Prop) (h : P) : P",
        declaration_name="unreachable",
    )
    outcome = LeanVerifier(ROOT).verify(task, "exact h")
    assert outcome.category == "verifier_error"


def test_timed_out_preamble_is_a_verifier_error(tmp_path: Path) -> None:
    fake_lake = tmp_path / "slow-lake"
    fake_lake.write_text("#!/bin/sh\nsleep 1\n", encoding="utf-8")
    fake_lake.chmod(0o755)

    outcome = LeanVerifier(
        ROOT, timeout_seconds=0.01, lake_command=str(fake_lake)
    ).verify(CORE_TASK, "exact h")

    assert outcome.category == "verifier_error"
    assert "verifier environment probe failed" in outcome.diagnostics["stderr"]


def test_shared_verifier_serializes_one_preamble_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = LeanVerifier(ROOT)
    preamble_probe_count = 0

    def fake_run_source(
        source: str, *, started: float | None = None
    ) -> VerificationOutcome:
        nonlocal preamble_probe_count
        if "#check True" in source:
            preamble_probe_count += 1
            time.sleep(0.05)
        return VerificationOutcome("verified", 0, {"stdout": "", "stderr": ""}, 0.0)

    monkeypatch.setattr(verifier, "_run_source", fake_run_source)
    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _: verifier.verify(CORE_TASK, "exact h"), range(8)))

    assert preamble_probe_count == 1
    assert all(outcome.category == "verified" for outcome in outcomes)


def test_zero_exit_error_diagnostic_is_rejected(tmp_path: Path) -> None:
    fake_lake = tmp_path / "fake-lake"
    fake_lake.write_text(
        "#!/bin/sh\n"
        "if grep -q '#check True' \"$5\"; then exit 0; fi\n"
        "printf '%s:1:1: error: declaration uses '\"'\"'sorry'\"'\"'\\n' \"$5\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_lake.chmod(0o755)

    outcome = LeanVerifier(ROOT, lake_command=str(fake_lake)).verify(CORE_TASK, "sorry")

    assert outcome.category == "lean_rejected"
    assert outcome.lean_exit_code == 0
    assert "error: declaration uses 'sorry'" in outcome.diagnostics["stdout"]


def test_mathlib_candidate_uses_pinned_dependency() -> None:
    task = TaskRecord(
        id="add-zero",
        preamble="import Mathlib",
        declaration="theorem add_zero_fixture (n : ℕ) : n + 0 = n",
        declaration_name="add_zero_fixture",
    )
    outcome = LeanVerifier(ROOT).verify(task, "simpa using Nat.add_zero n")
    assert outcome.category == "verified", outcome.diagnostics
