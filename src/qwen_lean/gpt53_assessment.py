from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import queue
import re
import shlex
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .artifacts import write_artifacts
from .baseline import validate_minif2f_environment
from .metrics import summarize_results
from .minif2f import Phase1Config, materialize_benchmark_tasks
from .prompt import PROMPT_FORMAT_ID, render_prompt
from .schema import (
    GPT53_RESULT_SCHEMA_VERSION,
    CandidateResult,
    RunMetadata,
    TaskRecord,
)
from .verifier import LeanVerifier

CONFIG_SCHEMA_VERSION = "gpt53-spark-assessment-config-v1"
EVIDENCE_SCHEMA_VERSION = "gpt53-spark-assessment-evidence-v1"
MODEL_ID = "gpt-5.3-codex-spark"
REASONING_EFFORT = "xhigh"
ALLOWED_REASONING_EFFORTS = frozenset({"low", "xhigh"})
CANDIDATES_PER_TASK = 1
API_KEY_ENVIRONMENT_VARIABLES = frozenset({"OPENAI_API_KEY", "CODEX_API_KEY"})
FORBIDDEN_CHILD_EXECUTABLES = ("lean", "lake", "elan")
REQUIRED_EXEC_HELP_OPTIONS = (
    "--model",
    "--json",
    "--ephemeral",
    "--output-last-message",
)
REQUIRED_DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode_host",
    "computer_use",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "remote_plugin",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "tool_suggest",
    "view_image",
)

_DISALLOWED_EVENT_TYPE_FRAGMENTS = (
    "browser",
    "command_execution",
    "computer",
    "file_change",
    "image_generation",
    "mcp",
    "shell",
    "tool_call",
    "web_search",
)
_DISALLOWED_EVENT_KEYS = {
    "command",
    "mcp_server",
    "tool_call",
    "tool_name",
}
_MODEL_VALUE_KEYS = {
    "effective_model",
    "model",
    "model_id",
    "model_name",
    "requested_model",
    "selected_model",
}
_INTEGRITY_MARKERS = (
    re.compile(r"\bgpt-5\.3-codex(?!-spark)\b", re.IGNORECASE),
    re.compile(r"\bgpt-5\.6(?:-[a-z0-9_.-]+)?\b", re.IGNORECASE),
    re.compile(r"\bfall(?:ing)?\s+back\b", re.IGNORECASE),
    re.compile(r"\bfallback\b", re.IGNORECASE),
    re.compile(r"\bmigrat(?:e|ed|es|ing|ion)\b", re.IGNORECASE),
    re.compile(r"\bmodel\s+substitut(?:e|ed|es|ing|ion)\b", re.IGNORECASE),
)
_RETRYABLE_INFRASTRUCTURE_MARKERS = (
    re.compile(r"connection (?:reset|closed|refused)", re.IGNORECASE),
    re.compile(r"incomplete response returned", re.IGNORECASE),
    re.compile(r"stream disconnected before completion", re.IGNORECASE),
    re.compile(r"max_output_tokens", re.IGNORECASE),
    re.compile(r"ran out of room in the model's context window", re.IGNORECASE),
    re.compile(r"selected model is at capacity", re.IGNORECASE),
    re.compile(r"temporarily unavailable", re.IGNORECASE),
    re.compile(r"timed? out", re.IGNORECASE),
    re.compile(r"http (?:500|502|503|504)\b", re.IGNORECASE),
    re.compile(r"internal server error", re.IGNORECASE),
)


@dataclass(frozen=True)
class GPT53Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> GPT53Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unknown GPT assessment config schema: {value.get('schema_version')}"
            )
        config = cls(path=path.resolve(), value=value)
        config.validate()
        return config

    def validate(self) -> None:
        model = self.value.get("model", {})
        if model.get("id") != MODEL_ID:
            raise ValueError(f"Spark assessment model must be exactly {MODEL_ID}")
        reasoning_effort = model.get("reasoning_effort")
        if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
            raise ValueError(
                "Spark assessment reasoning effort must be explicitly selected from "
                f"{sorted(ALLOWED_REASONING_EFFORTS)}"
            )
        if reasoning_effort == "low" and self.value.get("artifact_namespace") != (
            "gpt53-spark-low"
        ):
            raise ValueError(
                "low reasoning assessment must use artifact_namespace gpt53-spark-low"
            )
        codex = self.value.get("codex", {})
        if tuple(codex.get("disabled_features", ())) != REQUIRED_DISABLED_FEATURES:
            raise ValueError("Spark assessment disabled-feature contract has changed")
        if float(codex.get("heartbeat_seconds", 0.0)) <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if int(codex.get("max_retryable_attempts", 0)) < 1:
            raise ValueError("max_retryable_attempts must be positive")
        workloads = self.value.get("workloads", {})
        expected = {
            "minif2f-valid-dev16-v1": 16,
            "minif2f-valid-v1": 244,
        }
        if workloads != expected:
            raise ValueError("Spark assessment workload contract has changed")
        prompt = self.value.get("prompt", {})
        if prompt.get("base_format_id") != PROMPT_FORMAT_ID:
            raise ValueError("Spark assessment must retain whole-proof-v1 semantics")
        if not str(prompt.get("instruction", "")).strip():
            raise ValueError("Spark assessment prompt instruction is empty")

    @property
    def project_root(self) -> Path:
        return self.path.parent.parent

    @property
    def phase1_config_path(self) -> Path:
        relative = Path(str(self.value["benchmark"]["phase1_config"]))
        return (self.project_root / relative).resolve()

    @property
    def prompt_instruction(self) -> str:
        return str(self.value["prompt"]["instruction"])

    @property
    def reasoning_effort(self) -> str:
        return str(self.value["model"]["reasoning_effort"])

    @property
    def artifact_namespace(self) -> str:
        return str(self.value.get("artifact_namespace", "gpt53-spark"))

    @property
    def heartbeat_seconds(self) -> float:
        return float(self.value["codex"]["heartbeat_seconds"])

    @property
    def max_retryable_attempts(self) -> int:
        return int(self.value["codex"]["max_retryable_attempts"])

    @property
    def fingerprint(self) -> str:
        return sha256_text(
            json.dumps(self.value, sort_keys=True, separators=(",", ":"))
        )


@dataclass(frozen=True)
class JsonlAudit:
    valid: bool
    violations: tuple[str, ...]
    event_counts: dict[str, int]
    item_type_counts: dict[str, int]
    tool_event_count: int
    thread_id: str | None
    usage: dict[str, int]
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": list(self.violations),
            "event_counts": self.event_counts,
            "item_type_counts": self.item_type_counts,
            "tool_event_count": self.tool_event_count,
            "thread_id": self.thread_id,
            "usage": self.usage,
            "event_count": self.event_count,
        }


@dataclass(frozen=True)
class ChildExecution:
    argv: tuple[str, ...]
    pid: int
    started_at: str
    ended_at: str
    elapsed_seconds: float
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    final_message_path: Path
    final_message_sha256: str | None
    final_message_bytes: int | None
    final_message_characters: int | None
    audit: JsonlAudit
    accepted: bool
    retryable: bool
    integrity_errors: tuple[str, ...]

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        def display(path: Path) -> str:
            if relative_to is None:
                return str(path)
            try:
                return str(path.relative_to(relative_to))
            except ValueError:
                return str(path)

        return {
            "argv": list(self.argv),
            "pid": self.pid,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": self.elapsed_seconds,
            "exit_code": self.exit_code,
            "stdout_jsonl_path": display(self.stdout_path),
            "stderr_path": display(self.stderr_path),
            "final_message_path": display(self.final_message_path),
            "final_message_sha256": self.final_message_sha256,
            "final_message_bytes": self.final_message_bytes,
            "final_message_characters": self.final_message_characters,
            "audit": self.audit.to_dict(),
            "accepted": self.accepted,
            "retryable": self.retryable,
            "integrity_errors": list(self.integrity_errors),
        }


class ProgressLogger:
    def __init__(self, path: Path, *, console: TextIO | None = None) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.console = console
        self._lock = threading.Lock()

    def emit(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", utc_now())
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
            rendered = format_progress(payload)
            if self.console is None:
                print(rendered, flush=True)
            else:
                print(rendered, file=self.console, flush=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def render_codex_prompt(config: GPT53Config, task: TaskRecord) -> str:
    return f"{config.prompt_instruction.rstrip()}\n\n{render_prompt(task)}"


def resolve_codex_binary() -> Path:
    resolved = shutil.which("codex")
    if resolved is None:
        raise RuntimeError("codex CLI is not available on PATH")
    return Path(resolved).resolve()


def build_child_argv(
    config: GPT53Config,
    *,
    codex_binary: Path,
    final_message_path: Path,
) -> list[str]:
    argv = [
        str(codex_binary.resolve()),
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--model",
        MODEL_ID,
        "-c",
        f'model_reasoning_effort="{config.reasoning_effort}"',
    ]
    for feature in config.value["codex"]["disabled_features"]:
        argv.extend(("--disable", str(feature)))
    argv.extend(
        (
            "--json",
            "--output-last-message",
            str(final_message_path.resolve()),
            "-",
        )
    )
    validate_child_argv(argv, reasoning_effort=config.reasoning_effort)
    return argv


def validate_child_argv(
    argv: Iterable[str], *, reasoning_effort: str = REASONING_EFFORT
) -> None:
    values = list(argv)
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(f"unsupported Spark reasoning effort: {reasoning_effort}")
    if "--model" not in values:
        raise ValueError("nested Codex argv omits the required explicit model flag")
    model_index = values.index("--model")
    if model_index + 1 >= len(values) or values[model_index + 1] != MODEL_ID:
        raise ValueError(f"nested Codex argv must select exactly {MODEL_ID}")
    effort = f'model_reasoning_effort="{reasoning_effort}"'
    if "-c" not in values or effort not in values:
        raise ValueError(
            f"nested Codex argv omits the required {reasoning_effort} override"
        )
    effort_overrides = [
        value for value in values if value.startswith("model_reasoning_effort=")
    ]
    if effort_overrides != [effort]:
        raise ValueError(
            "nested Codex argv must contain exactly one reasoning override"
        )
    for required in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
    ):
        if required not in values:
            raise ValueError(f"nested Codex argv omits required option {required}")
    if "resume" in values:
        raise ValueError("nested candidate execution may not resume a session")


def sanitize_child_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    for variable in API_KEY_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    environment["PATH"] = sanitize_child_path(environment.get("PATH", ""))
    for executable in FORBIDDEN_CHILD_EXECUTABLES:
        if shutil.which(executable, path=environment["PATH"]) is not None:
            raise RuntimeError(
                f"sanitized child PATH still exposes forbidden executable {executable}"
            )
    return environment


def sanitize_child_path(value: str) -> str:
    retained: list[str] = []
    for raw_entry in value.split(os.pathsep):
        if not raw_entry:
            continue
        entry = Path(raw_entry)
        exposes_forbidden = any(
            (entry / executable).is_file() and os.access(entry / executable, os.X_OK)
            for executable in FORBIDDEN_CHILD_EXECUTABLES
        )
        if not exposes_forbidden and raw_entry not in retained:
            retained.append(raw_entry)
    return os.pathsep.join(retained)


def validate_isolated_workdir(
    workdir: Path, *, project_root: Path, benchmark_root: Path
) -> None:
    resolved = workdir.resolve()
    if _is_relative_to(resolved, project_root.resolve()):
        raise ValueError("child working directory is inside qwen-lean")
    if _is_relative_to(resolved, benchmark_root.resolve()):
        raise ValueError("child working directory is inside the miniF2F checkout")
    if any(resolved.iterdir()):
        raise ValueError("child working directory is not empty")


def validate_arm_artifact_path(
    config: GPT53Config, path: Path, *, evidence: bool = False
) -> None:
    if config.reasoning_effort != "low":
        return
    category = "evidence" if evidence else "artifacts"
    expected_root = (
        config.project_root / category
    ).resolve() / config.artifact_namespace
    resolved = path.resolve()
    if not _is_relative_to(resolved, expected_root):
        raise ValueError(
            f"low reasoning {category} path must remain under {expected_root}; got "
            f"{resolved}"
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def audit_jsonl_text(stdout_text: str, stderr_text: str = "") -> JsonlAudit:
    violations: list[str] = []
    event_counts: Counter[str] = Counter()
    item_type_counts: Counter[str] = Counter()
    tool_event_count = 0
    thread_id: str | None = None
    usage: dict[str, int] = {}
    events: list[dict[str, Any]] = []

    for line_number, line in enumerate(stdout_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            violations.append(f"stdout line {line_number} is not JSON: {error.msg}")
            continue
        if not isinstance(event, dict):
            violations.append(f"stdout line {line_number} is not a JSON object")
            continue
        events.append(event)
        event_type = str(event.get("type", "<missing>"))
        event_counts[event_type] += 1
        if event_type == "thread.started" and event.get("thread_id") is not None:
            thread_id = str(event["thread_id"])
        if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {
                str(key): int(value)
                for key, value in event["usage"].items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type", "<missing>"))
            item_type_counts[item_type] += 1
        event_tool_hits = _tool_event_hits(event)
        if event_tool_hits:
            tool_event_count += 1
            violations.append(
                f"disallowed external-tool event at line {line_number}: "
                + ", ".join(sorted(event_tool_hits))
            )
        violations.extend(_wrong_model_values(event, line_number=line_number))

    combined = f"{stdout_text}\n{stderr_text}"
    for marker in _INTEGRITY_MARKERS:
        match = marker.search(combined)
        if match is not None:
            violations.append(f"model integrity marker detected: {match.group(0)}")
    if not events:
        violations.append("JSONL audit stream contains no events")
    if event_counts["thread.started"] != 1:
        violations.append(
            "JSONL audit stream must contain exactly one thread.started event"
        )
    if event_counts["turn.completed"] != 1:
        violations.append(
            "JSONL audit stream must contain exactly one turn.completed event"
        )
    for failed_type in ("error", "turn.failed"):
        if event_counts[failed_type]:
            violations.append(f"JSONL audit stream contains {failed_type}")

    unique_violations = tuple(dict.fromkeys(violations))
    return JsonlAudit(
        valid=not unique_violations,
        violations=unique_violations,
        event_counts=dict(sorted(event_counts.items())),
        item_type_counts=dict(sorted(item_type_counts.items())),
        tool_event_count=tool_event_count,
        thread_id=thread_id,
        usage=usage,
        event_count=len(events),
    )


def _tool_event_hits(value: Any, *, path: str = "event") -> set[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lowered_key = str(key).lower()
            if lowered_key in _DISALLOWED_EVENT_KEYS and child not in (
                None,
                "",
                [],
                {},
            ):
                hits.add(child_path)
            if lowered_key == "type" and isinstance(child, str):
                lowered_type = child.lower()
                if any(
                    fragment in lowered_type
                    for fragment in _DISALLOWED_EVENT_TYPE_FRAGMENTS
                ):
                    hits.add(f"{child_path}={child}")
            hits.update(_tool_event_hits(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.update(_tool_event_hits(child, path=f"{path}[{index}]"))
    return hits


def _wrong_model_values(
    value: Any, *, line_number: int, path: str = "event"
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                str(key).lower() in _MODEL_VALUE_KEYS
                and isinstance(child, str)
                and child != MODEL_ID
            ):
                violations.append(
                    f"unexpected model value at line {line_number} {child_path}: {child}"
                )
            violations.extend(
                _wrong_model_values(child, line_number=line_number, path=child_path)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _wrong_model_values(
                    child, line_number=line_number, path=f"{path}[{index}]"
                )
            )
    return violations


def read_final_message(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def is_retryable_infrastructure_failure(
    *, exit_code: int, stderr_text: str, audit: JsonlAudit
) -> bool:
    if audit.tool_event_count or any(
        violation.startswith(("model integrity", "unexpected model"))
        for violation in audit.violations
    ):
        return False
    if exit_code == 0:
        return False
    return any(
        pattern.search(stderr_text) for pattern in _RETRYABLE_INFRASTRUCTURE_MARKERS
    )


def should_retry_result_category(category: str) -> bool:
    return category == "verifier_error"


def format_progress(event: Mapping[str, Any]) -> str:
    kind = str(event.get("event", "progress"))
    workload = str(event.get("workload_id", "preflight"))
    task = str(event.get("task_id", "preflight"))
    completed = event.get("completed", 0)
    total = event.get("total", 1)
    prefix = f"[{workload}] {task} candidate=0 progress={completed}/{total}"
    if kind == "candidate_started":
        return (
            f"{prefix} started pid={event.get('pid')} model={event.get('requested_model')} "
            f"effort={event.get('requested_reasoning_effort')} "
            f"argv={event.get('argv')}"
        )
    if kind == "candidate_heartbeat":
        return (
            f"{prefix} running pid={event.get('pid')} "
            f"elapsed={event.get('elapsed_seconds')}s events={event.get('event_count')}"
        )
    if kind == "candidate_event":
        return (
            f"{prefix} event={event.get('codex_event_type')} "
            f"events={event.get('event_count')} thread={event.get('thread_id')}"
        )
    if kind == "candidate_completed":
        return (
            f"{prefix} completed exit={event.get('exit_code')} "
            f"elapsed={event.get('elapsed_seconds')}s tools={event.get('tool_event_count')} "
            f"usage={event.get('usage')} sha256={event.get('final_message_sha256')}"
        )
    if kind == "candidate_retry":
        return (
            f"{prefix} retrying infrastructure-invalid attempt={event.get('attempt')}"
        )
    if kind == "candidate_reused":
        return f"{prefix} reused exact accepted artifact"
    return f"{prefix} {kind}"


def run_nested_codex(
    config: GPT53Config,
    *,
    prompt: str,
    codex_binary: Path,
    attempt_dir: Path,
    project_root: Path,
    benchmark_root: Path,
    workload_id: str,
    task_id: str,
    completed: int,
    total: int,
    logger: ProgressLogger,
) -> ChildExecution:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = (attempt_dir / "events.jsonl").resolve()
    stderr_path = (attempt_dir / "stderr.log").resolve()
    final_message_path = (attempt_dir / "final-message.txt").resolve()
    argv = build_child_argv(
        config,
        codex_binary=codex_binary,
        final_message_path=final_message_path,
    )
    environment = sanitize_child_environment()
    started_at = utc_now()
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="qwen-lean-gpt53-spark-") as temporary_dir:
        workdir = Path(temporary_dir).resolve()
        validate_isolated_workdir(
            workdir, project_root=project_root, benchmark_root=benchmark_root
        )
        process = subprocess.Popen(
            argv,
            cwd=workdir,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        logger.emit(
            {
                "event": "candidate_started",
                "workload_id": workload_id,
                "task_id": task_id,
                "candidate_index": 0,
                "completed": completed,
                "total": total,
                "requested_model": MODEL_ID,
                "requested_reasoning_effort": config.reasoning_effort,
                "argv": shlex.join(argv),
                "pid": process.pid,
                "started_at": started_at,
                "working_directory_isolated": True,
                "sanitized_path": environment["PATH"],
                "api_key_variables_removed": sorted(API_KEY_ENVIRONMENT_VARIABLES),
                "stdout_jsonl_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "final_message_path": str(final_message_path),
            }
        )

        lines: queue.Queue[tuple[str, str]] = queue.Queue()
        stdout_thread = threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_path, "stdout", lines),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_path, "stderr", lines),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        process.stdin.write(prompt)
        process.stdin.close()

        event_count = 0
        thread_id: str | None = None
        next_heartbeat = time.monotonic() + config.heartbeat_seconds
        while (
            process.poll() is None
            or stdout_thread.is_alive()
            or stderr_thread.is_alive()
            or not lines.empty()
        ):
            try:
                stream_name, line = lines.get(timeout=0.25)
            except queue.Empty:
                stream_name = ""
                line = ""
            if stream_name == "stdout" and line.strip():
                event_count += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event_type = "invalid_json"
                else:
                    event_type = str(event.get("type", "<missing>"))
                    if event_type == "thread.started" and event.get("thread_id"):
                        thread_id = str(event["thread_id"])
                logger.emit(
                    {
                        "event": "candidate_event",
                        "workload_id": workload_id,
                        "task_id": task_id,
                        "candidate_index": 0,
                        "completed": completed,
                        "total": total,
                        "pid": process.pid,
                        "codex_event_type": event_type,
                        "event_count": event_count,
                        "thread_id": thread_id,
                    }
                )
            now = time.monotonic()
            if process.poll() is None and now >= next_heartbeat:
                logger.emit(
                    {
                        "event": "candidate_heartbeat",
                        "workload_id": workload_id,
                        "task_id": task_id,
                        "candidate_index": 0,
                        "completed": completed,
                        "total": total,
                        "pid": process.pid,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "event_count": event_count,
                        "thread_id": thread_id,
                    }
                )
                next_heartbeat = now + config.heartbeat_seconds

        exit_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    elapsed_seconds = time.perf_counter() - started
    ended_at = utc_now()
    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    audit = audit_jsonl_text(stdout_text, stderr_text)
    integrity_errors = list(audit.violations)
    if exit_code != 0:
        integrity_errors.append(f"nested Codex exited with code {exit_code}")
    final_bytes: bytes | None = None
    final_characters: int | None = None
    if final_message_path.is_file():
        final_bytes = final_message_path.read_bytes()
        try:
            final_characters = len(final_bytes.decode("utf-8"))
        except UnicodeDecodeError as error:
            integrity_errors.append(f"final message is not UTF-8: {error}")
    else:
        integrity_errors.append("nested Codex did not write the final message artifact")
    accepted = not integrity_errors
    retryable = is_retryable_infrastructure_failure(
        exit_code=exit_code,
        stderr_text=f"{stdout_text}\n{stderr_text}",
        audit=audit,
    )
    execution = ChildExecution(
        argv=tuple(argv),
        pid=process.pid,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed_seconds,
        exit_code=exit_code,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        final_message_path=final_message_path,
        final_message_sha256=None if final_bytes is None else sha256_bytes(final_bytes),
        final_message_bytes=None if final_bytes is None else len(final_bytes),
        final_message_characters=final_characters,
        audit=audit,
        accepted=accepted,
        retryable=retryable,
        integrity_errors=tuple(dict.fromkeys(integrity_errors)),
    )
    logger.emit(
        {
            "event": "candidate_completed",
            "workload_id": workload_id,
            "task_id": task_id,
            "candidate_index": 0,
            "completed": completed + int(accepted),
            "total": total,
            "requested_model": MODEL_ID,
            "requested_reasoning_effort": config.reasoning_effort,
            "pid": process.pid,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "exit_code": exit_code,
            "thread_id": audit.thread_id,
            "event_count": audit.event_count,
            "event_counts": audit.event_counts,
            "item_type_counts": audit.item_type_counts,
            "tool_event_count": audit.tool_event_count,
            "usage": audit.usage,
            "stdout_jsonl_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "final_message_path": str(final_message_path),
            "final_message_sha256": execution.final_message_sha256,
            "accepted": accepted,
            "retryable": retryable,
            "integrity_errors": list(execution.integrity_errors),
        }
    )
    return execution


def _drain_stream(
    stream: TextIO,
    path: Path,
    stream_name: str,
    lines: queue.Queue[tuple[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for line in stream:
            output.write(line)
            output.flush()
            lines.put((stream_name, line))


def run_preflight(config: GPT53Config, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    validate_arm_artifact_path(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    codex_binary = resolve_codex_binary()
    environment = sanitize_child_environment()
    command_evidence: dict[str, dict[str, Any]] = {}
    for name, argv in (
        ("version", [str(codex_binary), "--version"]),
        ("login_status", [str(codex_binary), "login", "status"]),
        ("exec_help", [str(codex_binary), "exec", "--help"]),
    ):
        completed = subprocess.run(
            argv,
            cwd=config.project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        (output_dir / f"{name}.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output_dir / f"{name}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        command_evidence[name] = {
            "argv": argv,
            "exit_code": completed.returncode,
            "stdout_sha256": sha256_text(completed.stdout),
            "stderr_sha256": sha256_text(completed.stderr),
        }

    version_text = (
        (output_dir / "version.stdout.txt").read_text(encoding="utf-8").strip()
    )
    login_text = (
        (output_dir / "login_status.stdout.txt").read_text(encoding="utf-8")
        + (output_dir / "login_status.stderr.txt").read_text(encoding="utf-8")
    ).strip()
    help_text = (output_dir / "exec_help.stdout.txt").read_text(encoding="utf-8") + (
        output_dir / "exec_help.stderr.txt"
    ).read_text(encoding="utf-8")
    preconditions: list[str] = []
    if any(item["exit_code"] != 0 for item in command_evidence.values()):
        preconditions.append("one or more Codex CLI inspection commands failed")
    if "Logged in using ChatGPT" not in login_text:
        preconditions.append("Codex CLI is not authenticated using ChatGPT")
    missing_options = [
        option for option in REQUIRED_EXEC_HELP_OPTIONS if option not in help_text
    ]
    if missing_options:
        preconditions.append(f"codex exec help is missing options: {missing_options}")

    logger = ProgressLogger(output_dir / "run-log.jsonl")
    execution: ChildExecution | None = None
    if not preconditions:
        execution = run_nested_codex(
            config,
            prompt=(
                "Do not use any tools or external resources. Reason internally, then "
                "return exactly the lowercase text `ok` and nothing else."
            ),
            codex_binary=codex_binary,
            attempt_dir=output_dir / "nested-attempt-1",
            project_root=config.project_root,
            benchmark_root=config.project_root,
            workload_id="gpt53-spark-preflight-v1",
            task_id="trivial-model-pin",
            completed=0,
            total=1,
            logger=logger,
        )
        if not execution.accepted:
            preconditions.extend(execution.integrity_errors)

    summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed" if not preconditions else "failed",
        "config_fingerprint": config.fingerprint,
        "cli_version": version_text,
        "auth_mode": "ChatGPT"
        if "Logged in using ChatGPT" in login_text
        else "unproved",
        "required_exec_help_options": {
            option: option in help_text for option in REQUIRED_EXEC_HELP_OPTIONS
        },
        "requested_model": MODEL_ID,
        "requested_reasoning_effort": config.reasoning_effort,
        "api_key_environment_variables_removed": sorted(API_KEY_ENVIRONMENT_VARIABLES),
        "forbidden_child_executables": list(FORBIDDEN_CHILD_EXECUTABLES),
        "disabled_features": list(REQUIRED_DISABLED_FEATURES),
        "command_evidence": command_evidence,
        "nested_execution": None
        if execution is None
        else execution.to_dict(relative_to=output_dir),
        "failures": list(dict.fromkeys(preconditions)),
    }
    write_json(output_dir / "summary.json", summary)
    if summary["status"] != "passed":
        raise RuntimeError(
            "GPT-5.3-Codex Spark preflight failed: " + "; ".join(preconditions)
        )
    return summary


def assessment_contract_fingerprint(
    config: GPT53Config, phase1_config: Phase1Config, cli_version: str
) -> str:
    payload = {
        "config_fingerprint": config.fingerprint,
        "phase1_config_sha256": sha256_bytes(phase1_config.path.read_bytes()),
        "cli_version": cli_version,
        "model": MODEL_ID,
        "reasoning_effort": config.reasoning_effort,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def validate_preflight(
    config: GPT53Config, preflight_dir: Path, *, cli_version: str
) -> dict[str, Any]:
    summary = json.loads((preflight_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError("Spark assessment preflight has not passed")
    expected = {
        "config_fingerprint": config.fingerprint,
        "cli_version": cli_version,
        "auth_mode": "ChatGPT",
        "requested_model": MODEL_ID,
        "requested_reasoning_effort": config.reasoning_effort,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"Spark assessment preflight {key} differs from current run"
            )
    nested = summary.get("nested_execution") or {}
    if not nested.get("accepted"):
        raise ValueError("Spark assessment preflight nested execution was not accepted")
    audit = nested.get("audit") or {}
    if not audit.get("valid") or audit.get("tool_event_count") != 0:
        raise ValueError("Spark assessment preflight audit is not valid and tool-free")
    return summary


def run_assessment(
    config: GPT53Config,
    *,
    benchmark_root: Path,
    workload_id: str,
    preflight_dir: Path,
    output_dir: Path,
    resume: bool,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    if workload_id not in config.value["workloads"]:
        raise ValueError(f"unsupported Spark assessment workload: {workload_id}")
    output_dir = output_dir.resolve()
    validate_arm_artifact_path(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = benchmark_root.resolve()
    codex_binary = resolve_codex_binary()
    cli_version = _checked_cli_version(codex_binary, config.project_root)
    preflight = validate_preflight(config, preflight_dir, cli_version=cli_version)
    phase1_config = Phase1Config.load(config.phase1_config_path)
    all_tasks = materialize_benchmark_tasks(phase1_config, benchmark_root)
    tasks = phase1_config.select_workload(workload_id, all_tasks)
    expected_count = int(config.value["workloads"][workload_id])
    if len(tasks) != expected_count:
        raise ValueError(
            f"Spark assessment workload {workload_id} expected {expected_count} tasks, "
            f"got {len(tasks)}"
        )
    timeout_seconds = float(phase1_config.value["verifier"]["timeout_seconds"])
    environment_validation = validate_minif2f_environment(
        phase1_config,
        benchmark_root,
        timeout_seconds=timeout_seconds,
    )
    verifier_environment = environment_validation["verifier_environment"]
    contract_fingerprint = assessment_contract_fingerprint(
        config, phase1_config, cli_version
    )
    logger = ProgressLogger(output_dir / "run-log.jsonl")
    verifier = LeanVerifier(benchmark_root, timeout_seconds=timeout_seconds)
    results: list[CandidateResult] = []
    records: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for global_index, task in enumerate(tasks):
        candidate_dir = output_dir / "candidates" / task.id
        record_path = candidate_dir / "candidate.json"
        failure_path = candidate_dir / "failure.json"
        prompt = render_codex_prompt(config, task)
        prompt_sha256 = sha256_text(prompt)
        if resume and record_path.is_file():
            record, result = load_existing_candidate(
                record_path,
                output_dir=output_dir,
                contract_fingerprint=contract_fingerprint,
                task=task,
                prompt_sha256=prompt_sha256,
                reasoning_effort=config.reasoning_effort,
            )
            if should_retry_result_category(result.category):
                logger.emit(
                    {
                        "event": "candidate_verification_retry",
                        "workload_id": workload_id,
                        "task_id": task.id,
                        "candidate_index": 0,
                        "completed": global_index,
                        "total": len(tasks),
                        "prior_category": result.category,
                        "raw_candidate_sha256": record["raw_candidate_sha256"],
                    }
                )
                verification_started = time.perf_counter()
                outcome = verifier.verify(task, result.candidate_text)
                verification_elapsed = time.perf_counter() - verification_started
                result = CandidateResult(
                    task_id=result.task_id,
                    candidate_id=result.candidate_id,
                    candidate_index=result.candidate_index,
                    candidate_text=result.candidate_text,
                    category=outcome.category,
                    lean_exit_code=outcome.lean_exit_code,
                    diagnostics=outcome.diagnostics,
                    generation_latency_seconds=result.generation_latency_seconds,
                    verification_latency_seconds=outcome.latency_seconds,
                    total_latency_seconds=(result.generation_latency_seconds or 0.0)
                    + verification_elapsed,
                    generated_token_count=result.generated_token_count,
                    finish_reason=result.finish_reason,
                )
                verification = {
                    "category": outcome.category,
                    "lean_exit_code": outcome.lean_exit_code,
                    "latency_seconds": outcome.latency_seconds,
                }
                verification_attempts = list(
                    record.get("verification_attempts") or [record["verification"]]
                )
                verification_attempts.append(verification)
                record["verification"] = verification
                record["verification_attempt_count"] = len(verification_attempts)
                record["verification_attempts"] = verification_attempts
                record["result"] = result.to_dict()
                write_json(record_path, record)
                if should_retry_result_category(outcome.category):
                    raise RuntimeError(
                        f"unresolved verifier infrastructure outcome for {task.id}: "
                        f"{outcome.category}; preserved raw candidate was reverified "
                        "without regeneration"
                    )
                failure_path.unlink(missing_ok=True)
            records.append(record)
            results.append(result)
            logger.emit(
                {
                    "event": "candidate_reused",
                    "workload_id": workload_id,
                    "task_id": task.id,
                    "candidate_index": 0,
                    "completed": global_index + 1,
                    "total": len(tasks),
                }
            )
            continue
        if record_path.exists():
            raise ValueError(
                f"candidate artifact already exists for {task.id}; use --resume only "
                "after verifying the exact contract"
            )

        prior_attempts: list[dict[str, Any]] = []
        if failure_path.is_file():
            if not resume:
                raise ValueError(
                    f"failed candidate artifact already exists for {task.id}; "
                    "use --resume to preserve and reclassify it"
                )
            prior_attempts = load_prior_retryable_attempts(
                failure_path,
                output_dir=output_dir,
                contract_fingerprint=contract_fingerprint,
                workload_id=workload_id,
                task_id=task.id,
                prompt_sha256=prompt_sha256,
            )
            if len(prior_attempts) >= config.max_retryable_attempts:
                raise RuntimeError(
                    f"retryable attempt budget exhausted for {task.id}: "
                    f"{len(prior_attempts)}/{config.max_retryable_attempts}"
                )
            logger.emit(
                {
                    "event": "candidate_retry",
                    "workload_id": workload_id,
                    "task_id": task.id,
                    "candidate_index": 0,
                    "completed": global_index,
                    "total": len(tasks),
                    "attempt": len(prior_attempts) + 1,
                    "prior_integrity_errors": prior_attempts[-1]["integrity_errors"],
                }
            )

        attempt_records = list(prior_attempts)
        execution: ChildExecution | None = None
        for attempt in range(
            len(prior_attempts) + 1, config.max_retryable_attempts + 1
        ):
            execution = run_nested_codex(
                config,
                prompt=prompt,
                codex_binary=codex_binary,
                attempt_dir=candidate_dir / f"attempt-{attempt}",
                project_root=config.project_root,
                benchmark_root=benchmark_root,
                workload_id=workload_id,
                task_id=task.id,
                completed=global_index,
                total=len(tasks),
                logger=logger,
            )
            attempt_records.append(execution.to_dict(relative_to=output_dir))
            if execution.accepted:
                break
            failure = {
                "schema_version": GPT53_RESULT_SCHEMA_VERSION,
                "status": "execution_contract_failure",
                "contract_fingerprint": contract_fingerprint,
                "workload_id": workload_id,
                "task_id": task.id,
                "candidate_index": 0,
                "prompt_sha256": prompt_sha256,
                "attempts": attempt_records,
            }
            write_json(failure_path, failure)
            if not execution.retryable or attempt == config.max_retryable_attempts:
                raise RuntimeError(
                    f"invalid nested execution for {task.id}: "
                    + "; ".join(execution.integrity_errors)
                )
            logger.emit(
                {
                    "event": "candidate_retry",
                    "workload_id": workload_id,
                    "task_id": task.id,
                    "candidate_index": 0,
                    "completed": global_index,
                    "total": len(tasks),
                    "attempt": attempt + 1,
                    "prior_integrity_errors": list(execution.integrity_errors),
                }
            )

        assert execution is not None and execution.accepted
        candidate_text = read_final_message(execution.final_message_path)
        verification_started = time.perf_counter()
        outcome = verifier.verify(task, candidate_text)
        verification_elapsed = time.perf_counter() - verification_started
        result = CandidateResult(
            task_id=task.id,
            candidate_id="gpt53-spark-0",
            candidate_index=0,
            candidate_text=candidate_text,
            category=outcome.category,
            lean_exit_code=outcome.lean_exit_code,
            diagnostics=outcome.diagnostics,
            generation_latency_seconds=execution.elapsed_seconds,
            verification_latency_seconds=outcome.latency_seconds,
            total_latency_seconds=execution.elapsed_seconds + verification_elapsed,
            generated_token_count=None,
            finish_reason="turn_completed",
        )
        record = {
            "schema_version": GPT53_RESULT_SCHEMA_VERSION,
            "status": "accepted",
            "contract_fingerprint": contract_fingerprint,
            "workload_id": workload_id,
            "task_id": task.id,
            "candidate_index": 0,
            "prompt_sha256": prompt_sha256,
            "requested_model": MODEL_ID,
            "requested_reasoning_effort": config.reasoning_effort,
            "attempt_count": len(attempt_records),
            "attempts": attempt_records,
            "accepted_attempt": len(attempt_records),
            "raw_candidate_path": str(
                execution.final_message_path.relative_to(output_dir)
            ),
            "raw_candidate_sha256": execution.final_message_sha256,
            "raw_candidate_bytes": execution.final_message_bytes,
            "raw_candidate_characters": execution.final_message_characters,
            "verification": {
                "category": outcome.category,
                "lean_exit_code": outcome.lean_exit_code,
                "latency_seconds": outcome.latency_seconds,
            },
            "verification_attempt_count": 1,
            "verification_attempts": [
                {
                    "category": outcome.category,
                    "lean_exit_code": outcome.lean_exit_code,
                    "latency_seconds": outcome.latency_seconds,
                }
            ],
            "result": result.to_dict(),
        }
        write_json(record_path, record)
        records.append(record)
        results.append(result)
        if should_retry_result_category(outcome.category):
            raise RuntimeError(
                f"unresolved verifier infrastructure outcome for {task.id}: "
                f"{outcome.category}; raw candidate was preserved and must not be regenerated"
            )
        failure_path.unlink(missing_ok=True)

    summary = summarize_results(
        results,
        expected_task_ids=[task.id for task in tasks],
        candidates_per_task=CANDIDATES_PER_TASK,
        ks=(1,),
    )
    summary["workload_id"] = workload_id
    summary["run_wall_time_seconds"] = time.perf_counter() - run_started
    summary["execution_integrity"] = summarize_execution_records(
        records, reasoning_effort=config.reasoning_effort
    )
    if summary["candidate_count"] != expected_count:
        summary["complete"] = False
        summary["completeness_errors"].append(
            f"expected {expected_count} Spark candidates, got {summary['candidate_count']}"
        )
    if not summary["execution_integrity"]["valid"]:
        summary["complete"] = False
        summary["completeness_errors"].extend(
            summary["execution_integrity"]["failures"]
        )

    metadata = RunMetadata(
        schema_version=GPT53_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source=(
            f"{phase1_config.benchmark['repository']}@"
            f"{phase1_config.benchmark['revision']}:"
            f"{phase1_config.benchmark['source_path']}"
        ),
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=str(verifier_environment["lean_toolchain"]),
        mathlib_revision=str(verifier_environment["dependencies"]["mathlib"]),
        verifier_timeout_seconds=timeout_seconds,
        model_id=MODEL_ID,
        workload_id=workload_id,
        benchmark_split=str(phase1_config.benchmark["split"]),
        benchmark_repository=str(phase1_config.benchmark["repository"]),
        benchmark_revision=str(phase1_config.benchmark["revision"]),
        verifier_environment=verifier_environment,
        candidates_per_task=CANDIDATES_PER_TASK,
        inference_engine="codex-cli",
        inference_engine_version=cli_version,
        generation_settings={
            "reasoning_effort": config.reasoning_effort,
            "candidates_per_task": CANDIDATES_PER_TASK,
            "prompt_adapter_id": str(config.value["prompt"]["id"]),
            "prompt_instruction_sha256": sha256_text(config.prompt_instruction),
            "raw_final_message_used_without_repair": True,
            "nested_command_template": sanitized_command_template(config),
            "disabled_features": list(REQUIRED_DISABLED_FEATURES),
        },
        runtime={
            "python": platform.python_version(),
            "authentication_mode": "ChatGPT",
            "inference_execution": "nested_codex_cli_chatgpt_entitlement",
            "contract_fingerprint": contract_fingerprint,
            "preflight_summary_sha256": sha256_bytes(
                (preflight_dir / "summary.json").read_bytes()
            ),
            "preflight_nested_thread_id": preflight["nested_execution"]["audit"][
                "thread_id"
            ],
            "run_log": str((output_dir / "run-log.jsonl").resolve()),
        },
    )
    write_artifacts(output_dir, metadata, results, summary=summary)
    if not summary["complete"]:
        raise RuntimeError(
            "Spark assessment workload is incomplete: "
            + "; ".join(summary["completeness_errors"])
        )
    return metadata, results, summary


def _checked_cli_version(codex_binary: Path, cwd: Path) -> str:
    completed = subprocess.run(
        [str(codex_binary), "--version"],
        cwd=cwd,
        env=sanitize_child_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"codex --version failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def load_existing_candidate(
    record_path: Path,
    *,
    output_dir: Path,
    contract_fingerprint: str,
    task: TaskRecord,
    prompt_sha256: str,
    reasoning_effort: str = REASONING_EFFORT,
) -> tuple[dict[str, Any], CandidateResult]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": GPT53_RESULT_SCHEMA_VERSION,
        "status": "accepted",
        "contract_fingerprint": contract_fingerprint,
        "task_id": task.id,
        "candidate_index": 0,
        "prompt_sha256": prompt_sha256,
        "requested_model": MODEL_ID,
        "requested_reasoning_effort": reasoning_effort,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"stale candidate artifact {record_path}: {key} differs")
    attempts = record.get("attempts") or []
    accepted_index = int(record.get("accepted_attempt", 0)) - 1
    if accepted_index not in range(len(attempts)):
        raise ValueError(f"candidate artifact has no accepted attempt: {record_path}")
    accepted = attempts[accepted_index]
    audit = accepted.get("audit") or {}
    if (
        not accepted.get("accepted")
        or not audit.get("valid")
        or audit.get("tool_event_count") != 0
    ):
        raise ValueError(f"candidate artifact failed integrity audit: {record_path}")
    raw_path = (output_dir / str(record["raw_candidate_path"])).resolve()
    if not _is_relative_to(raw_path, output_dir.resolve()):
        raise ValueError("candidate artifact path escapes the run directory")
    raw_bytes = raw_path.read_bytes()
    if sha256_bytes(raw_bytes) != record.get("raw_candidate_sha256"):
        raise ValueError(f"candidate artifact hash differs: {record_path}")
    result = CandidateResult.from_dict(record["result"])
    if result.candidate_text.encode("utf-8") != raw_bytes:
        raise ValueError(
            f"stored result repaired or changed raw candidate: {record_path}"
        )
    if result.category == "generation_error":
        raise ValueError("accepted candidate artifact contains a generation failure")
    return record, result


def load_prior_retryable_attempts(
    failure_path: Path,
    *,
    output_dir: Path,
    contract_fingerprint: str,
    workload_id: str,
    task_id: str,
    prompt_sha256: str,
) -> list[dict[str, Any]]:
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": GPT53_RESULT_SCHEMA_VERSION,
        "status": "execution_contract_failure",
        "contract_fingerprint": contract_fingerprint,
        "workload_id": workload_id,
        "task_id": task_id,
        "candidate_index": 0,
        "prompt_sha256": prompt_sha256,
    }
    for key, value in expected.items():
        if failure.get(key) != value:
            raise ValueError(f"stale failed candidate artifact: {key} differs")
    attempts = list(failure.get("attempts") or [])
    if not attempts:
        raise ValueError("failed candidate artifact contains no attempts")
    for attempt in attempts:
        if attempt.get("accepted"):
            raise ValueError("failed candidate artifact contains an accepted attempt")
        audit_value = attempt.get("audit") or {}
        if audit_value.get("tool_event_count"):
            raise ValueError(
                "tool-use failure is not eligible for infrastructure retry"
            )

    last_attempt = attempts[-1]
    stdout_path = _resolve_run_artifact_path(
        output_dir, str(last_attempt["stdout_jsonl_path"])
    )
    stderr_path = _resolve_run_artifact_path(
        output_dir, str(last_attempt["stderr_path"])
    )
    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    audit = audit_jsonl_text(stdout_text, stderr_text)
    if not is_retryable_infrastructure_failure(
        exit_code=int(last_attempt["exit_code"]),
        stderr_text=f"{stdout_text}\n{stderr_text}",
        audit=audit,
    ):
        raise ValueError("failed candidate artifact is not retryable infrastructure")
    return attempts


def _resolve_run_artifact_path(output_dir: Path, relative_path: str) -> Path:
    path = (output_dir / relative_path).resolve()
    if not _is_relative_to(path, output_dir.resolve()):
        raise ValueError("run artifact path escapes the output directory")
    return path


def summarize_execution_records(
    records: Iterable[Mapping[str, Any]],
    *,
    reasoning_effort: str = REASONING_EFFORT,
) -> dict[str, Any]:
    materialized = list(records)
    failures: list[str] = []
    accepted_executions: list[Mapping[str, Any]] = []
    attempt_count = 0
    for record in materialized:
        attempts = list(record.get("attempts") or [])
        attempt_count += len(attempts)
        accepted_index = int(record.get("accepted_attempt", 0)) - 1
        if record.get("status") != "accepted" or accepted_index not in range(
            len(attempts)
        ):
            failures.append(f"{record.get('task_id')}: no accepted execution")
            continue
        accepted = attempts[accepted_index]
        audit = accepted.get("audit") or {}
        if not accepted.get("accepted") or not audit.get("valid"):
            failures.append(f"{record.get('task_id')}: invalid accepted execution")
        if audit.get("tool_event_count") != 0:
            failures.append(
                f"{record.get('task_id')}: tool event in accepted execution"
            )
        if accepted.get("exit_code") != 0:
            failures.append(f"{record.get('task_id')}: nonzero accepted child exit")
        accepted_executions.append(accepted)

    usage_keys = sorted(
        {
            key
            for execution in accepted_executions
            for key in (execution.get("audit", {}).get("usage", {}) or {})
        }
    )
    usage = {
        key: sum(
            int(execution.get("audit", {}).get("usage", {}).get(key, 0))
            for execution in accepted_executions
        )
        for key in usage_keys
    }
    latencies = [float(item["elapsed_seconds"]) for item in accepted_executions]
    byte_lengths = [
        int(item["final_message_bytes"])
        for item in accepted_executions
        if item.get("final_message_bytes") is not None
    ]
    event_types = Counter(
        event_type
        for execution in accepted_executions
        for event_type, count in execution.get("audit", {})
        .get("event_counts", {})
        .items()
        for _ in range(int(count))
    )
    return {
        "valid": not failures and len(accepted_executions) == len(materialized),
        "failures": failures,
        "requested_model": MODEL_ID,
        "requested_reasoning_effort": reasoning_effort,
        "accepted_candidate_execution_count": len(accepted_executions),
        "total_child_attempt_count": attempt_count,
        "retry_count": attempt_count - len(accepted_executions),
        "child_failure_count": attempt_count - len(accepted_executions),
        "tool_event_count": sum(
            int(item.get("audit", {}).get("tool_event_count", 0))
            for item in accepted_executions
        ),
        "event_type_counts": dict(sorted(event_types.items())),
        "usage_totals": usage,
        "latency_seconds": numeric_summary(latencies),
        "output_bytes": numeric_summary(byte_lengths),
        "raw_final_messages_used_without_repair": True,
    }


def numeric_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    materialized = list(values)
    if not materialized:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(materialized),
        "min": min(materialized),
        "max": max(materialized),
        "mean": statistics.fmean(materialized),
        "median": statistics.median(materialized),
    }


def sanitized_command_template(config: GPT53Config) -> list[str]:
    return build_child_argv(
        config,
        codex_binary=Path("/ABSOLUTE/PATH/TO/codex"),
        final_message_path=Path("/ISOLATED/ARTIFACT/final-message.txt"),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_compact_evidence(
    config: GPT53Config,
    *,
    preflight_dir: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    if config.reasoning_effort == "low":
        for path in (preflight_dir, dev16_dir, full_dir):
            validate_arm_artifact_path(config, path)
        validate_arm_artifact_path(config, evidence_dir, evidence=True)
    preflight = json.loads((preflight_dir / "summary.json").read_text(encoding="utf-8"))
    dev16 = _load_complete_summary(dev16_dir, "minif2f-valid-dev16-v1", 16)
    full = _load_complete_summary(full_dir, "minif2f-valid-v1", 244)
    if preflight.get("status") != "passed":
        raise ValueError(
            "cannot write evidence from a failed Spark assessment preflight"
        )
    if preflight.get("config_fingerprint") != config.fingerprint:
        raise ValueError("preflight evidence uses a different Spark assessment config")

    evidence_dir = evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    preflight_compact = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": preflight["status"],
        "cli_version": preflight["cli_version"],
        "auth_mode": preflight["auth_mode"],
        "requested_model": preflight["requested_model"],
        "requested_reasoning_effort": preflight["requested_reasoning_effort"],
        "required_exec_help_options": preflight["required_exec_help_options"],
        "api_key_environment_variables_removed": preflight[
            "api_key_environment_variables_removed"
        ],
        "forbidden_child_executables": preflight["forbidden_child_executables"],
        "disabled_features": preflight["disabled_features"],
        "command_template": sanitized_command_template(config),
        "nested_execution": _compact_execution(preflight["nested_execution"]),
        "accepted_model_substitution_or_fallback_markers": 0,
    }
    dev_compact = compact_workload_evidence(dev16, dev16_dir)
    full_compact = compact_workload_evidence(full, full_dir)
    if config.reasoning_effort == "low":
        return _write_low_evidence(
            config,
            preflight_compact=preflight_compact,
            dev_compact=dev_compact,
            full_compact=full_compact,
            full_dir=full_dir,
            evidence_dir=evidence_dir,
        )

    phase1_summary_path = config.project_root / str(
        config.value["comparison"]["phase1_base_summary"]
    )
    phase6_validation_path = config.project_root / str(
        config.value["comparison"]["reference_sft_validation"]
    )
    phase1 = json.loads(phase1_summary_path.read_text(encoding="utf-8"))
    phase6 = json.loads(phase6_validation_path.read_text(encoding="utf-8"))
    base_pass1 = float(phase1["pass_at_k"]["pass@1"])
    reference_pass1 = float(phase6["adapter"]["pass_at_k"]["pass@1"])
    gpt_pass1 = float(full["pass_at_k"]["pass@1"])
    comparison = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "workload_id": "minif2f-valid-v1",
        "candidate_budget": {
            "gpt53_codex_spark": {"candidates_per_task": 1, "candidate_count": 244},
            "qwen3_8b_base": {
                "candidates_per_task": int(phase1["candidates_per_task"]),
                "candidate_count": int(phase1["candidate_count"]),
            },
            "reference_sft_v1": {
                "candidates_per_task": int(phase6["adapter"]["candidates_per_task"]),
                "candidate_count": int(phase6["adapter"]["candidate_count"]),
            },
        },
        "pass_at_1": {
            "gpt-5.3-codex-spark-xhigh-one-shot": gpt_pass1,
            "Qwen/Qwen3-8B-Base": base_pass1,
            "reference-sft-v1": reference_pass1,
        },
        "delta_gpt53_spark_minus_qwen3_8b_base": gpt_pass1 - base_pass1,
        "delta_gpt53_spark_minus_reference_sft_v1": gpt_pass1 - reference_pass1,
        "comparison_caveat": (
            "GPT-5.3-Codex Spark used one isolated xhigh reasoning execution per theorem; "
            "the Qwen base and reference-sft-v1 pass@1 estimates come from the accepted "
            "Phase 1 sampling contract with eight stochastic candidates per task. All three "
            "use the same raw-continuation Lean verifier, but their generation procedures are "
            "not identical."
        ),
        "source_evidence": {
            "qwen3_8b_base": str(phase1_summary_path.relative_to(config.project_root)),
            "reference_sft_v1": str(
                phase6_validation_path.relative_to(config.project_root)
            ),
        },
    }
    write_json(evidence_dir / "preflight.json", preflight_compact)
    write_json(evidence_dir / "dev16.json", dev_compact)
    write_json(evidence_dir / "full.json", full_compact)
    write_json(evidence_dir / "comparison.json", comparison)
    readme = f"""# GPT-5.3-Codex Spark one-shot assessment

**ACCEPTED:** all benchmark candidates used fresh nested Codex CLI executions with the explicit `gpt-5.3-codex-spark` model pin, `xhigh` reasoning override, ChatGPT authentication, an isolated empty working directory, a sanitized PATH without Lean/Lake/Elan, disabled tool surfaces, and raw final-message verification without repair.

**OBSERVED:** dev16 completed {dev16["candidate_count"]}/{dev16["task_count"]} candidates with pass@1 {dev16["pass_at_k"]["pass@1"]:.6f}. The complete miniF2F validation run completed {full["candidate_count"]}/{full["task_count"]} candidates with {full["category_counts"]["verified"]} verified proofs and pass@1 {gpt_pass1:.6f}.

**ACCEPTED:** the unchanged Phase 1 miniF2F verifier timeout is 30 seconds. The full run's {full["verifier_timeout_count"]} `verifier_timeout` outcomes are unsuccessful proof attempts, not infrastructure errors; they count in the 244-candidate denominator and do not authorize candidate regeneration or verification retry.

**OBSERVED:** accepted executions contained {full["execution_integrity"]["tool_event_count"]} external-tool events. Child-process accounting records {full["execution_integrity"]["retry_count"]} infrastructure retries separately from proof outcomes, and accepted executions contained zero detected non-Spark GPT-5.3, GPT-5.6, model-migration, substitution, or fallback markers. Full JSONL event streams and raw final messages remain in ignored local `artifacts/` storage.

**OBSERVED:** on the same 244-task validation set, accepted Qwen3-8B-Base pass@1 was {base_pass1:.6f} and `reference-sft-v1` pass@1 was {reference_pass1:.6f}. Spark used one isolated xhigh reasoning execution rather than the Phase 1 eight-sample stochastic generation procedure, so this is verifier-aligned but not an identical inference process.
"""
    (evidence_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison


def _write_low_evidence(
    config: GPT53Config,
    *,
    preflight_compact: Mapping[str, Any],
    dev_compact: Mapping[str, Any],
    full_compact: Mapping[str, Any],
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    comparison_config = config.value["comparison"]
    xhigh_dir = (
        config.project_root / str(comparison_config["xhigh_full_artifacts"])
    ).resolve()
    xhigh_evidence_path = (
        config.project_root / str(comparison_config["xhigh_full_evidence"])
    ).resolve()
    expected_xhigh_evidence_sha256 = str(
        comparison_config["xhigh_full_evidence_sha256"]
    )
    actual_xhigh_evidence_sha256 = sha256_bytes(xhigh_evidence_path.read_bytes())
    if actual_xhigh_evidence_sha256 != expected_xhigh_evidence_sha256:
        raise ValueError(
            "accepted xhigh compact evidence hash differs from the low config"
        )
    xhigh_evidence = json.loads(xhigh_evidence_path.read_text(encoding="utf-8"))
    expected_xhigh_pass1 = float(comparison_config["xhigh_pass_at_1"])
    if (
        xhigh_evidence.get("execution_integrity", {}).get("requested_reasoning_effort")
        != "xhigh"
        or xhigh_evidence.get("candidate_count") != 244
        or float(xhigh_evidence["pass_at_k"]["pass@1"]) != expected_xhigh_pass1
    ):
        raise ValueError("accepted xhigh compact evidence violates the frozen control")

    expected_ids = _primary_task_ids(config)
    low_records = load_accepted_candidate_records(
        full_dir, expected_ids=expected_ids, reasoning_effort="low"
    )
    xhigh_records = load_accepted_candidate_records(
        xhigh_dir, expected_ids=expected_ids, reasoning_effort="xhigh"
    )
    xhigh_manifest_sha256 = candidate_records_manifest_sha256(xhigh_dir)
    if xhigh_manifest_sha256 != str(
        comparison_config["xhigh_candidate_records_manifest_sha256"]
    ):
        raise ValueError("accepted xhigh candidate-record manifest hash differs")

    low_outcomes = {
        task_id: str(record["result"]["category"])
        for task_id, record in low_records.items()
    }
    xhigh_outcomes = {
        task_id: str(record["result"]["category"])
        for task_id, record in xhigh_records.items()
    }
    comparison = paired_outcome_comparison(
        low_outcomes,
        xhigh_outcomes,
        expected_task_ids=expected_ids,
        xhigh_pass_at_1=expected_xhigh_pass1,
    )
    comparison["source_integrity"].update(
        {
            "xhigh_compact_evidence": str(
                xhigh_evidence_path.relative_to(config.project_root)
            ),
            "xhigh_compact_evidence_sha256": actual_xhigh_evidence_sha256,
            "xhigh_candidate_records": str(xhigh_dir.relative_to(config.project_root)),
            "xhigh_candidate_records_manifest_sha256": xhigh_manifest_sha256,
            "low_candidate_records_manifest_sha256": (
                candidate_records_manifest_sha256(full_dir)
            ),
        }
    )
    low_pass1 = float(full_compact["pass_at_k"]["pass@1"])
    if comparison["aggregate"]["low_pass_at_1"] != low_pass1:
        raise ValueError("low candidate records disagree with the complete run summary")

    compute = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "workload_id": "minif2f-valid-v1",
        "arms": {
            "low": runtime_arm_summary(low_records, full_compact),
            "xhigh": runtime_arm_summary(xhigh_records, xhigh_evidence),
        },
        "usage_semantics": (
            "Codex CLI turn.completed usage fields are compared by identical field name "
            "under the same model, prompt, candidate count, CLI version, and task IDs."
        ),
        "interpretation_scope": (
            "Descriptive evidence from one frozen reasoning-effort ablation; it does not "
            "identify a compute-independent model-quality effect or establish causality "
            "beyond this low-versus-xhigh procedure."
        ),
    }
    low_reasoning = int(
        compute["arms"]["low"]["usage"]["reasoning_output_tokens"]["total"]
    )
    xhigh_reasoning = int(
        compute["arms"]["xhigh"]["usage"]["reasoning_output_tokens"]["total"]
    )
    compute["comparison"] = {
        "reasoning_output_token_reduction": xhigh_reasoning - low_reasoning,
        "reasoning_output_token_reduction_fraction": (
            (xhigh_reasoning - low_reasoning) / xhigh_reasoning
        ),
        "verified_proof_delta_xhigh_minus_low": (
            comparison["paired_outcomes"]["solved_only_by_xhigh"]
            - comparison["paired_outcomes"]["solved_only_by_low"]
        ),
    }

    write_json(evidence_dir / "preflight.json", preflight_compact)
    write_json(evidence_dir / "dev16.json", dev_compact)
    write_json(evidence_dir / "full.json", full_compact)
    write_json(evidence_dir / "comparison.json", comparison)
    write_json(evidence_dir / "compute.json", compute)

    low_verified = int(full_compact["category_counts"]["verified"])
    xhigh_verified = int(xhigh_evidence["category_counts"]["verified"])
    reduction_fraction = float(
        compute["comparison"]["reasoning_output_token_reduction_fraction"]
    )
    compute_assessment = (
        "materially reduced"
        if reduction_fraction >= 0.1
        else "did not materially reduce"
    )
    proof_assessment = (
        "decreased"
        if low_verified < xhigh_verified
        else "increased"
        if low_verified > xhigh_verified
        else "was unchanged"
    )
    readme = f"""# GPT-5.3-Codex Spark low-reasoning ablation

**ACCEPTED:** all low-arm candidates used fresh nested Codex CLI executions with the explicit `gpt-5.3-codex-spark` model pin, `low` reasoning override, ChatGPT authentication, an isolated empty working directory, a sanitized PATH without Lean/Lake/Elan, disabled tool surfaces, and raw final-message verification without repair.

**OBSERVED:** dev16 completed {dev_compact["candidate_count"]}/{dev_compact["task_count"]} candidates with pass@1 {dev_compact["pass_at_k"]["pass@1"]:.6f}. The complete miniF2F validation run completed {full_compact["candidate_count"]}/{full_compact["task_count"]} candidates with {low_verified} verified proofs and pass@1 {low_pass1:.6f}.

**OBSERVED:** the frozen xhigh control verified {xhigh_verified}/244 tasks (pass@1 {expected_xhigh_pass1:.6f}). Low and xhigh solved {comparison["paired_outcomes"]["solved_by_both"]} tasks in common; {comparison["paired_outcomes"]["solved_only_by_xhigh"]} were solved only by xhigh, {comparison["paired_outcomes"]["solved_only_by_low"]} only by low, and {comparison["paired_outcomes"]["solved_by_neither"]} by neither. The exact two-sided McNemar p-value is {comparison["paired_binary_test"]["p_value"]:.6g}.

**OBSERVED:** reducing reasoning effort {compute_assessment} reported test-time reasoning output: {xhigh_reasoning:,} xhigh tokens versus {low_reasoning:,} low tokens ({reduction_fraction:.2%} reduction). Verified proof success {proof_assessment} from {xhigh_verified} to {low_verified}. Latency, other usage fields, verifier timeouts, retries, and proof-per-token efficiencies are recorded in `compute.json`.

**ACCEPTED:** `lean_rejected`, `empty_candidate`, and `verifier_timeout` remain unsuccessful proof attempts and never authorize candidate regeneration. Accepted executions contained {full_compact["execution_integrity"]["tool_event_count"]} external-tool events and {full_compact["execution_integrity"]["retry_count"]} bounded generation-infrastructure retries.

**ACCEPTED:** this is descriptive evidence from one frozen low-versus-xhigh effort ablation. The relative ratio and efficiency values are not compute-independent model-quality claims, and no causal claim is made beyond this exact procedure.
"""
    (evidence_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison


def _primary_task_ids(config: GPT53Config) -> list[str]:
    phase1_config = Phase1Config.load(config.phase1_config_path)
    manifest_path = phase1_config.path.parent / str(
        phase1_config.benchmark["primary_task_manifest"]
    )
    task_ids = [
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(task_ids) != 244 or len(set(task_ids)) != 244:
        raise ValueError(
            "pinned miniF2F validation manifest must contain 244 unique IDs"
        )
    return task_ids


def candidate_records_manifest_sha256(run_dir: Path) -> str:
    candidate_root = run_dir.resolve() / "candidates"
    paths = sorted(candidate_root.glob("*/candidate.json"))
    manifest = "".join(
        f"{path.relative_to(run_dir.resolve())} {sha256_bytes(path.read_bytes())}\n"
        for path in paths
    )
    return sha256_text(manifest)


def load_accepted_candidate_records(
    run_dir: Path,
    *,
    expected_ids: Iterable[str],
    reasoning_effort: str,
) -> dict[str, dict[str, Any]]:
    ordered_ids = list(expected_ids)
    candidate_root = run_dir.resolve() / "candidates"
    paths = sorted(candidate_root.glob("*/candidate.json"))
    actual_ids = [path.parent.name for path in paths]
    if set(actual_ids) != set(ordered_ids) or len(actual_ids) != len(ordered_ids):
        raise ValueError(
            "paired comparison requires exactly the pinned 244 task IDs in each arm"
        )
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        task_id = path.parent.name
        attempts = list(record.get("attempts") or [])
        accepted_index = int(record.get("accepted_attempt", 0)) - 1
        if (
            record.get("status") != "accepted"
            or record.get("task_id") != task_id
            or record.get("candidate_index") != 0
            or record.get("requested_model") != MODEL_ID
            or record.get("requested_reasoning_effort") != reasoning_effort
            or accepted_index not in range(len(attempts))
        ):
            raise ValueError(f"invalid paired candidate record: {path}")
        accepted = attempts[accepted_index]
        audit = accepted.get("audit") or {}
        if (
            not accepted.get("accepted")
            or accepted.get("exit_code") != 0
            or not audit.get("valid")
            or audit.get("tool_event_count") != 0
        ):
            raise ValueError(f"paired candidate record failed execution audit: {path}")
        result = record.get("result") or {}
        if result.get("task_id") != task_id or result.get("candidate_index") != 0:
            raise ValueError(f"paired candidate result identity differs: {path}")
        records[task_id] = record
    return {task_id: records[task_id] for task_id in ordered_ids}


def paired_outcome_comparison(
    low_outcomes: Mapping[str, str],
    xhigh_outcomes: Mapping[str, str],
    *,
    expected_task_ids: Iterable[str],
    xhigh_pass_at_1: float,
) -> dict[str, Any]:
    task_ids = list(expected_task_ids)
    expected = set(task_ids)
    if (
        len(task_ids) != len(expected)
        or set(low_outcomes) != expected
        or set(xhigh_outcomes) != expected
    ):
        raise ValueError("paired comparison task IDs differ from the pinned manifest")
    groups = {
        "solved_by_both": [],
        "solved_only_by_xhigh": [],
        "solved_only_by_low": [],
        "solved_by_neither": [],
    }
    for task_id in task_ids:
        low_verified = low_outcomes[task_id] == "verified"
        xhigh_verified = xhigh_outcomes[task_id] == "verified"
        if low_verified and xhigh_verified:
            group = "solved_by_both"
        elif xhigh_verified:
            group = "solved_only_by_xhigh"
        elif low_verified:
            group = "solved_only_by_low"
        else:
            group = "solved_by_neither"
        groups[group].append(task_id)
    counts = {key: len(value) for key, value in groups.items()}
    xhigh_verified_count = counts["solved_by_both"] + counts["solved_only_by_xhigh"]
    low_verified_count = counts["solved_by_both"] + counts["solved_only_by_low"]
    observed_xhigh_pass1 = xhigh_verified_count / len(task_ids)
    if observed_xhigh_pass1 != xhigh_pass_at_1:
        raise ValueError("frozen xhigh task outcomes disagree with accepted pass@1")
    discordant = counts["solved_only_by_xhigh"] + counts["solved_only_by_low"]
    tail = min(counts["solved_only_by_xhigh"], counts["solved_only_by_low"])
    p_value = (
        1.0
        if discordant == 0
        else min(
            1.0,
            2.0
            * sum(math.comb(discordant, k) for k in range(tail + 1))
            / (2**discordant),
        )
    )
    low_pass1 = low_verified_count / len(task_ids)
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "workload_id": "minif2f-valid-v1",
        "task_count": len(task_ids),
        "aggregate": {
            "low_pass_at_1": low_pass1,
            "xhigh_pass_at_1": xhigh_pass_at_1,
            "absolute_delta_xhigh_minus_low": xhigh_pass_at_1 - low_pass1,
            "descriptive_ratio_low_over_xhigh": low_pass1 / xhigh_pass_at_1,
        },
        "paired_outcome_table": {
            "low_fail_xhigh_fail": counts["solved_by_neither"],
            "low_fail_xhigh_verified": counts["solved_only_by_xhigh"],
            "low_verified_xhigh_fail": counts["solved_only_by_low"],
            "low_verified_xhigh_verified": counts["solved_by_both"],
        },
        "paired_outcomes": counts,
        "task_ids_by_paired_outcome": groups,
        "paired_binary_test": {
            "method": "exact_two_sided_mcnemar_binomial",
            "discordant_pair_count": discordant,
            "p_value": p_value,
            "interpretation": "descriptive_uncertainty_not_a_hard_success_gate",
        },
        "source_integrity": {
            "same_ordered_task_ids": True,
            "ordered_task_ids_sha256": sha256_text("\n".join(task_ids) + "\n"),
            "xhigh_control_reused_without_regeneration": True,
        },
        "interpretation_caveat": (
            "The ratio and paired test describe this single frozen reasoning-effort "
            "ablation and are not compute-independent model-quality claims."
        ),
    }


def runtime_arm_summary(
    records: Mapping[str, Mapping[str, Any]], compact: Mapping[str, Any]
) -> dict[str, Any]:
    accepted: list[Mapping[str, Any]] = []
    verified_count = 0
    for record in records.values():
        attempts = list(record["attempts"])
        accepted.append(attempts[int(record["accepted_attempt"]) - 1])
        verified_count += int(record["result"]["category"] == "verified")
    usage_keys = sorted(
        {
            key
            for execution in accepted
            for key in (execution.get("audit", {}).get("usage", {}) or {})
        }
    )
    usage = {
        key: distribution_summary(
            int(execution.get("audit", {}).get("usage", {}).get(key, 0))
            for execution in accepted
        )
        for key in usage_keys
    }
    reasoning = usage.get("reasoning_output_tokens")
    efficiency: dict[str, float | int | None] = {
        "verified_proofs": verified_count,
        "verified_proofs_per_1m_reasoning_tokens": None,
        "reasoning_tokens_per_verified_proof": None,
        "mean_reasoning_tokens_per_task": None,
    }
    if reasoning is not None and int(reasoning["total"]) > 0:
        reasoning_total = int(reasoning["total"])
        efficiency["verified_proofs_per_1m_reasoning_tokens"] = (
            verified_count * 1_000_000 / reasoning_total
        )
        efficiency["reasoning_tokens_per_verified_proof"] = (
            None if verified_count == 0 else reasoning_total / verified_count
        )
        efficiency["mean_reasoning_tokens_per_task"] = float(reasoning["mean"])
    return {
        "candidate_count": len(accepted),
        "verified_count": verified_count,
        "usage": usage,
        "accepted_child_latency_seconds": distribution_summary(
            float(execution["elapsed_seconds"]) for execution in accepted
        ),
        "verifier_timeout_count": int(compact["verifier_timeout_count"]),
        "infrastructure_retry_count": int(
            compact["child_process_retry_accounting"]["retry_count"]
        ),
        "efficiency": efficiency,
    }


def distribution_summary(
    values: Iterable[float | int],
) -> dict[str, float | int | None]:
    materialized = sorted(values)
    if not materialized:
        return {
            "count": 0,
            "total": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    rank = (len(materialized) - 1) * 0.95
    lower = math.floor(rank)
    upper = math.ceil(rank)
    p95 = float(materialized[lower])
    if upper != lower:
        p95 += (float(materialized[upper]) - p95) * (rank - lower)
    return {
        "count": len(materialized),
        "total": sum(materialized),
        "min": materialized[0],
        "max": materialized[-1],
        "mean": statistics.fmean(materialized),
        "median": statistics.median(materialized),
        "p95": p95,
    }


def _load_complete_summary(
    path: Path, workload_id: str, expected_count: int
) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("workload_id") != workload_id:
        raise ValueError(f"unexpected workload in {path}")
    if not summary.get("complete"):
        raise ValueError(f"incomplete Spark assessment workload in {path}")
    if (
        summary.get("task_count") != expected_count
        or summary.get("candidate_count") != expected_count
    ):
        raise ValueError(f"unexpected Spark assessment denominator in {path}")
    integrity = summary.get("execution_integrity") or {}
    if (
        not integrity.get("valid")
        or integrity.get("accepted_candidate_execution_count") != expected_count
        or integrity.get("tool_event_count") != 0
    ):
        raise ValueError(f"invalid Spark assessment execution integrity in {path}")
    return summary


def compact_workload_evidence(
    summary: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    progress = sample_progress_events(run_dir / "run-log.jsonl")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    verifier_timeout_seconds = float(run["verifier_timeout_seconds"])
    if verifier_timeout_seconds != 30.0:
        raise ValueError(
            "Spark assessment evidence requires the frozen 30-second verifier timeout"
        )
    execution_integrity = summary["execution_integrity"]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "workload_id": summary["workload_id"],
        "task_count": summary["task_count"],
        "candidate_count": summary["candidate_count"],
        "candidates_per_task": summary["candidates_per_task"],
        "tasks_with_verified_candidate": summary["tasks_with_verified_candidate"],
        "pass_at_k": summary["pass_at_k"],
        "category_counts": summary["category_counts"],
        "verifier_timeout_count": summary["verifier_timeout_count"],
        "infrastructure_error_count": summary["infrastructure_error_count"],
        "verifier_policy": {
            "timeout_seconds": verifier_timeout_seconds,
            "verifier_timeout_semantics": "unsuccessful_proof_attempt",
            "verifier_timeout_is_infrastructure_error": False,
            "verifier_timeout_triggers_candidate_regeneration": False,
            "verifier_timeout_triggers_verification_retry": False,
        },
        "child_process_retry_accounting": {
            "accepted_candidate_execution_count": execution_integrity[
                "accepted_candidate_execution_count"
            ],
            "total_child_attempt_count": execution_integrity[
                "total_child_attempt_count"
            ],
            "child_failure_count": execution_integrity["child_failure_count"],
            "retry_count": execution_integrity["retry_count"],
        },
        "execution_integrity": execution_integrity,
        "progress_log_excerpt": progress,
    }


def sample_progress_events(path: Path) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict[str, Any]] = []
    for wanted in ("candidate_started", "candidate_heartbeat", "candidate_completed"):
        match = next((event for event in events if event.get("event") == wanted), None)
        if match is None:
            continue
        selected.append(
            {
                key: match.get(key)
                for key in (
                    "event",
                    "timestamp",
                    "workload_id",
                    "task_id",
                    "candidate_index",
                    "completed",
                    "total",
                    "requested_model",
                    "requested_reasoning_effort",
                    "pid",
                    "elapsed_seconds",
                    "event_count",
                    "thread_id",
                    "exit_code",
                    "tool_event_count",
                    "usage",
                    "final_message_sha256",
                )
                if key in match
            }
        )
    if not any(item["event"] == "candidate_completed" for item in selected):
        raise ValueError("progress log contains no candidate completion event")
    return selected


def _compact_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    audit = value["audit"]
    return {
        "accepted": value["accepted"],
        "exit_code": value["exit_code"],
        "elapsed_seconds": value["elapsed_seconds"],
        "final_message_sha256": value["final_message_sha256"],
        "thread_id": audit["thread_id"],
        "event_counts": audit["event_counts"],
        "item_type_counts": audit["item_type_counts"],
        "tool_event_count": audit["tool_event_count"],
        "usage": audit["usage"],
        "integrity_errors": value["integrity_errors"],
    }
