from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .prompt import normalize_transport, reconstruct_source
from .schema import ResultCategory, TaskRecord


@dataclass(frozen=True)
class VerificationOutcome:
    category: ResultCategory
    lean_exit_code: int | None
    diagnostics: dict[str, str]
    latency_seconds: float


class LeanVerifier:
    """Verify isolated candidates in the repository's pinned Lake environment."""

    def __init__(
        self,
        project_root: Path,
        *,
        timeout_seconds: float = 30.0,
        lake_command: str = "lake",
    ) -> None:
        self.project_root = project_root.resolve()
        self.timeout_seconds = timeout_seconds
        self.lake_command = lake_command
        self._preamble_probes: dict[str, VerificationOutcome | None] = {}
        self._preamble_probe_lock = threading.Lock()

    def verify(self, task: TaskRecord, candidate: str) -> VerificationOutcome:
        started = time.perf_counter()
        if not normalize_transport(candidate):
            return VerificationOutcome(
                category="empty_candidate",
                lean_exit_code=None,
                diagnostics={"stdout": "", "stderr": "candidate is empty"},
                latency_seconds=time.perf_counter() - started,
            )

        probe = self._probe_preamble(task.preamble)
        if probe is not None:
            return VerificationOutcome(
                category=probe.category,
                lean_exit_code=probe.lean_exit_code,
                diagnostics={
                    "stdout": probe.diagnostics["stdout"],
                    "stderr": f"verifier environment probe failed: {probe.diagnostics['stderr']}",
                },
                latency_seconds=time.perf_counter() - started,
            )

        return self._run_source(reconstruct_source(task, candidate), started=started)

    def verify_raw_completion(
        self,
        *,
        preamble: str,
        model_input: str,
        candidate: str,
    ) -> VerificationOutcome:
        """Verify a raw completion after an arbitrary frozen model input."""

        started = time.perf_counter()
        normalized = normalize_transport(candidate)
        if not normalized:
            return VerificationOutcome(
                category="empty_candidate",
                lean_exit_code=None,
                diagnostics={"stdout": "", "stderr": "candidate is empty"},
                latency_seconds=time.perf_counter() - started,
            )
        probe = self._probe_preamble(preamble)
        if probe is not None:
            return VerificationOutcome(
                category=probe.category,
                lean_exit_code=probe.lean_exit_code,
                diagnostics={
                    "stdout": probe.diagnostics["stdout"],
                    "stderr": (
                        "verifier environment probe failed: "
                        f"{probe.diagnostics['stderr']}"
                    ),
                },
                latency_seconds=time.perf_counter() - started,
            )
        source = f"{preamble.rstrip()}\n\n{model_input}{normalized}\n"
        return self._run_source(source, started=started)

    def _probe_preamble(self, preamble: str) -> VerificationOutcome | None:
        return self.prime_preamble(preamble, timeout_seconds=self.timeout_seconds)

    def prime_preamble(
        self, preamble: str, *, timeout_seconds: float
    ) -> VerificationOutcome | None:
        """Validate and cache one shared preamble outside a candidate timeout."""

        with self._preamble_probe_lock:
            if preamble not in self._preamble_probes:
                outcome = self._run_source(
                    f"{preamble}\n\n#check True\n",
                    timeout_seconds=timeout_seconds,
                )
                if outcome.category != "verified":
                    outcome = VerificationOutcome(
                        category="verifier_error",
                        lean_exit_code=outcome.lean_exit_code,
                        diagnostics=outcome.diagnostics,
                        latency_seconds=outcome.latency_seconds,
                    )
                self._preamble_probes[preamble] = (
                    None if outcome.category == "verified" else outcome
                )
            return self._preamble_probes[preamble]

    def prime_task(
        self, task: TaskRecord, candidate: str, *, timeout_seconds: float
    ) -> VerificationOutcome | None:
        """Validate a full task and cache its source prefix as usable context."""

        outcome = self._run_source(
            reconstruct_source(task, candidate),
            timeout_seconds=timeout_seconds,
        )
        if outcome.category != "verified":
            return outcome
        with self._preamble_probe_lock:
            self._preamble_probes[task.preamble] = None
        return None

    def _run_source(
        self,
        source: str,
        *,
        started: float | None = None,
        timeout_seconds: float | None = None,
    ) -> VerificationOutcome:
        started = time.perf_counter() if started is None else started
        try:
            with tempfile.TemporaryDirectory(prefix="qwen-lean-") as temporary_dir:
                source_path = Path(temporary_dir) / "Candidate.lean"
                source_path.write_text(source, encoding="utf-8", newline="\n")
                process = subprocess.Popen(
                    [
                        self.lake_command,
                        "env",
                        "lean",
                        "-E",
                        "hasSorry",
                        str(source_path),
                    ],
                    cwd=self.project_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=(
                            self.timeout_seconds
                            if timeout_seconds is None
                            else timeout_seconds
                        )
                    )
                except subprocess.TimeoutExpired:
                    _terminate_process_group(process)
                    stdout, stderr = process.communicate()
                    return VerificationOutcome(
                        category="verifier_timeout",
                        lean_exit_code=None,
                        diagnostics={"stdout": stdout, "stderr": stderr},
                        latency_seconds=time.perf_counter() - started,
                    )
        except (OSError, ValueError) as error:
            return VerificationOutcome(
                category="verifier_error",
                lean_exit_code=None,
                diagnostics={"stdout": "", "stderr": str(error)},
                latency_seconds=time.perf_counter() - started,
            )

        diagnostics = {
            "stdout": stdout.replace(str(source_path), "Candidate.lean"),
            "stderr": stderr.replace(str(source_path), "Candidate.lean"),
        }
        has_error_diagnostic = any(
            ": error:" in line
            for stream in diagnostics.values()
            for line in stream.splitlines()
        )
        category: ResultCategory = (
            "verified"
            if process.returncode == 0 and not has_error_diagnostic
            else "lean_rejected"
        )
        return VerificationOutcome(
            category=category,
            lean_exit_code=process.returncode,
            diagnostics=diagnostics,
            latency_seconds=time.perf_counter() - started,
        )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.kill()
