from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset_v2 import sha256_file
from .mathia_prompt_ab import (
    EXPECTED_CANDIDATES_PER_TASK,
    EXPECTED_CANDIDATES_TOTAL,
    VERIFIER_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    WORKLOAD_IDS,
    BoundTask,
    PromptABConfig,
    _atomic_write,
    _canonical_json_bytes,
    _load_q0_reference,
    _sha256_json,
    _sha256_text,
    inventory_generations,
    inventory_verifications,
    validate_execution_manifest,
    verifier_environment_identities,
)
from .prompt import normalize_transport, render_prompt, render_proof_request
from .schema import TaskRecord
from .verifier import LeanVerifier, VerificationOutcome

ANALYSIS_SCHEMA_VERSION = "mathia-q0-b-regression-analysis-v3"
RAW_B_SCHEMA_VERSION = "mathia-q0-b-regression-raw-b-candidate-v1"
RAW_Q0_SCHEMA_VERSION = "mathia-q0-b-regression-q0-verified-candidate-v1"
TRANSFORMED_SCHEMA_VERSION = "mathia-q0-b-regression-transformed-candidate-v1"
ORACLE_SCHEMA_VERSION = "mathia-q0-b-regression-single-node-oracle-v1"
EXPECTED_REGRESSION_TASKS = 23
EXPECTED_REGRESSION_COUNTS = {
    "minif2f-valid-clean-v2": 17,
    "fresh-composition-valid-v2": 6,
}
RECOVERY_ARCHIVE_SHA256 = (
    "aeac05f215c9882456a712de341593a19a7a7253da7e4cebff64e015301d9182"
)

_LEADING_BY = re.compile(r"\A(?P<leading>[ \t\r\n]*)by(?=\s|$)")
_MARKDOWN_FENCE = re.compile(
    r"\A\s*```(?:lean|lean4)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_LEAN3_BEGIN_END = re.compile(
    r"\A(?P<leading>\s*)begin(?=\s)(?P<body>.*?)\bend(?P<trailing>\s*)\Z",
    re.DOTALL,
)
_NATURAL_LANGUAGE_PREFIX = re.compile(
    r"\A(?P<leading>[ \t]*)(?:Here(?:'s| is) (?:the )?(?:Lean )?proof|"
    r"The (?:Lean )?proof is|Proof|Answer):?[ \t]*\r?\n",
    re.IGNORECASE,
)
_NATURAL_LANGUAGE_SUFFIX = re.compile(
    r"\r?\n[ \t]*(?:This (?:completes|proves) the proof\.?|QED\.?|"
    r"Explanation:.*|Note:.*)[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)
_SORRY_ADMIT = re.compile(r"\b(?:sorry|admit)\b")
_THEOREM_REPETITION = re.compile(r"(?m)^\s*(?:theorem|lemma)\s+[A-Za-z0-9_'.]+")
_UNKNOWN_REFERENCE = re.compile(
    r"Unknown (?P<kind>identifier|constant|declaration) "
    r"`(?P<name>[^`\r\n]+)`",
    re.IGNORECASE,
)
_UNKNOWN_REFERENCE_AT = re.compile(
    r"Candidate\.lean:(?P<line>[0-9]+):(?P<column>[0-9]+):[^\r\n]*?"
    r"Unknown (?P<kind>identifier|constant|declaration) "
    r"`(?P<name>[^`\r\n]+)`",
    re.IGNORECASE,
)
_LEAN_ERROR_AT = re.compile(
    r"Candidate\.lean:(?P<line>[0-9]+):(?P<column>[0-9]+): "
    r"error(?:\([^\r\n)]*\))?: (?P<message>[^\r\n]*)",
    re.IGNORECASE,
)
_QUALIFIED_LEAN_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_'])"
    r"(?:[A-Za-z_][A-Za-z0-9_']*\.)+[A-Za-z_][A-Za-z0-9_']*"
)
_DECLARATION_BINDER = re.compile(
    r"[({]\s*(?P<names>[A-Za-z_][A-Za-z0-9_'₀-₉]*"
    r"(?:\s+[A-Za-z_][A-Za-z0-9_'₀-₉]*)*)\s*:"
)
_LOCAL_PREMISE_NAME = re.compile(
    r"h(?:[0-9₀-₉]+|[A-Za-z](?:[0-9₀-₉])?)?\Z"
)

_DIAGNOSTIC_CATEGORY_PATTERNS = (
    (
        "unknown_identifier",
        re.compile(r"Unknown identifier `[^`\r\n]+`", re.IGNORECASE),
    ),
    (
        "unknown_constant",
        re.compile(r"Unknown constant `[^`\r\n]+`", re.IGNORECASE),
    ),
    (
        "unknown_declaration",
        re.compile(r"Unknown declaration `[^`\r\n]+`", re.IGNORECASE),
    ),
    ("unsolved_goals", re.compile(r"\bunsolved goals?\b", re.IGNORECASE)),
    (
        "no_goals_to_be_solved",
        re.compile(r"\bNo goals to be solved\b", re.IGNORECASE),
    ),
    (
        "type_mismatch",
        re.compile(r"\b(?:application )?type mismatch\b", re.IGNORECASE),
    ),
    (
        "elaboration_error",
        re.compile(
            r"\b(?:failed to synthesize|cannot synthesize|invalid field|"
            r"function expected|failed to infer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "syntax_error",
        re.compile(
            r"\b(?:unexpected token|unexpected end of input|unexpected identifier|"
            r"unterminated|Invalid `end`)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tactic_failure",
        re.compile(
            r"\b(?:unknown tactic|Tactic `[^`]+` failed|linarith failed|"
            r"made no progress|failed to close the goal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sorry_or_admit_rejected",
        re.compile(r"declaration uses `(?:sorry|admit)`", re.IGNORECASE),
    ),
)

_TACTIC_FAMILIES = (
    "aesop",
    "apply",
    "by_contra",
    "constructor",
    "decide",
    "exact",
    "field_simp",
    "fin_cases",
    "have",
    "induction",
    "interval_cases",
    "intro",
    "linarith",
    "native_decide",
    "nlinarith",
    "norm_num",
    "omega",
    "positivity",
    "ring",
    "ring_nf",
    "rw",
    "simp",
)

_STRUCTURAL_CLASSES = ("direct", "branching", "deep")
_TRANSITION_BUCKETS = ("both", "q0_only", "b_only", "neither")
_ORACLE_OUTCOMES = (
    "oracle_closes_parent",
    "oracle_advances_then_fails",
    "oracle_reaches_second_unknown",
    "oracle_no_material_progress",
    "oracle_not_testable",
)
_ANTI_VACUITY_FLAGS = (
    "root_target_restatement",
    "current_subgoal_oracle",
    "strict_intermediate_fact",
    "equivalence_not_determined",
)
_LEAN_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_'.\u2080-\u2089]*|[0-9]+|[^\s]"
)

_DIAGNOSTIC_PRELUDE = r'''
namespace QwenLeanIssue93Diagnostic

open Lean Elab Term Tactic Meta

elab "__qwenCaptureAndClose" : tactic => do
  let goals <- getUnsolvedGoals
  logInfo m!"QWEN_STATE_BEGIN\n{goalsToMessageData goals}\nQWEN_STATE_END"
  for goal in goals do
    goal.withContext do
      goal.assign (<- mkSorry (<- goal.getType) true)
  setGoals []

syntax "__qwenOracle" str : term
elab_rules : term <= expectedType
  | `(__qwenOracle $name:str) => do
    let expectedType <- instantiateMVars expectedType
    if expectedType.hasExprMVar then
      tryPostpone
    let unknown := name.getString
    let exactDeclarationExists := (<- getEnv).contains unknown.toName
    let proposition <- isProp expectedType
    logInfo m!"QWEN_ORACLE_BEGIN\nunknown: {unknown}\ntype_begin\n{expectedType}\ntype_end\nis_prop: {proposition}\nexact_declaration_exists: {exactDeclarationExists}\ncurrent_defeq: not_checked\nroot_defeq: not_checked\nQWEN_ORACLE_END"
    mkSorry expectedType true

syntax "__qwenOracleCompare" str "," term "," term : term
elab_rules : term <= expectedType
  | `(__qwenOracleCompare $name:str, $current, $root) => do
    let expectedType <- instantiateMVars expectedType
    if expectedType.hasExprMVar then
      tryPostpone
    let unknown := name.getString
    let currentType <- elabType current
    let rootType <- elabType root
    let currentDefEq <- isDefEq expectedType currentType
    let rootDefEq <- isDefEq expectedType rootType
    let exactDeclarationExists := (<- getEnv).contains unknown.toName
    let proposition <- isProp expectedType
    logInfo m!"QWEN_ORACLE_BEGIN\nunknown: {unknown}\ntype_begin\n{expectedType}\ntype_end\nis_prop: {proposition}\nexact_declaration_exists: {exactDeclarationExists}\ncurrent_defeq: {currentDefEq}\nroot_defeq: {rootDefEq}\nQWEN_ORACLE_END"
    mkSorry expectedType true

syntax "#qwenCheckDecl" str : command
elab_rules : command
  | `(#qwenCheckDecl $name:str) => do
    let candidate := name.getString
    let observed := (<- getEnv).contains candidate.toName
    logInfo m!"QWEN_DECL|{candidate}|{observed}"

end QwenLeanIssue93Diagnostic
'''.strip()

_STATE_MESSAGE = re.compile(
    r"QWEN_STATE_BEGIN\n(?P<state>.*?)\nQWEN_STATE_END", re.DOTALL
)
_ORACLE_MESSAGE = re.compile(
    r"QWEN_ORACLE_BEGIN\n(?P<body>.*?)\nQWEN_ORACLE_END", re.DOTALL
)
_DECLARATION_CHECK_MESSAGE = re.compile(
    r"QWEN_DECL\|(?P<name>[^|\r\n]+)\|(?P<exists>true|false)"
)


@dataclass(frozen=True)
class MechanicalVariant:
    transform_sequence: tuple[str, ...]
    transformed_text: str
    transformed_sha256: str


@dataclass(frozen=True)
class UnknownReferenceSite:
    workload: str
    task_id: str
    task_ordinal: int
    candidate_index: int
    candidate_id: str
    raw_sha256: str
    raw_text: str
    unknown_name: str
    unknown_kind: str
    source_line: int
    source_column: int
    raw_start: int
    raw_end: int
    command_start: int
    command_end: int


@dataclass(frozen=True)
class DiagnosticRun:
    category: str
    lean_exit_code: int | None
    stdout: str
    stderr: str
    latency_seconds: float


def _iter_jsonl_bytes(path: Path) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            yield line_number, line, value


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = _jsonl_payload(rows)
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def structural_transition_evidence(
    manifest: Mapping[str, Any],
    verification_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        q0_solved = int(task["q0_verified_candidate_count"]) > 0
        b_solved = any(
            verification_by_id[str(candidate_id)]["category"] == "verified"
            for candidate_id in task["candidate_slots"]["B"]
        )
        if q0_solved and b_solved:
            transition = "both"
        elif q0_solved:
            transition = "q0_only"
        elif b_solved:
            transition = "b_only"
        else:
            transition = "neither"
        structural_class = task.get("metadata", {}).get("structural_class")
        if structural_class not in _STRUCTURAL_CLASSES:
            structural_class = None
        members.append(
            {
                "ordinal": int(task["ordinal"]),
                "workload": str(task["workload_id"]),
                "task_id": str(task["task_id"]),
                "structural_class": structural_class,
                "q0_solved_at_8": q0_solved,
                "b_solved_at_8": b_solved,
                "transition_bucket": transition,
            }
        )

    by_class: dict[str, Any] = {}
    for structural_class in _STRUCTURAL_CLASSES:
        selected = [
            row for row in members if row["structural_class"] == structural_class
        ]
        counts = Counter(str(row["transition_bucket"]) for row in selected)
        by_class[structural_class] = {
            "availability": "available",
            "task_count": len(selected),
            "transition_counts": {
                bucket: counts[bucket] for bucket in _TRANSITION_BUCKETS
            },
        }
    unavailable = [row for row in members if row["structural_class"] is None]
    unavailable_counts = Counter(
        str(row["transition_bucket"]) for row in unavailable
    )
    classified_workloads = Counter(
        str(row["workload"])
        for row in members
        if row["structural_class"] is not None
    )
    unavailable_workloads = Counter(
        str(row["workload"])
        for row in members
        if row["structural_class"] is None
    )
    return {
        "population_task_count": len(members),
        "classification_source": {
            "field": "execution-manifest.tasks[].metadata.structural_class",
            "rule": (
                "Use only the frozen Dataset-v2 fresh-composition value when it is "
                "exactly direct, branching, or deep; preserve all other values as "
                "unavailable without inferring from Q0/B outcomes or output length."
            ),
            "outcome_independent": True,
        },
        "coverage": {
            "classified_task_count": len(members) - len(unavailable),
            "unavailable_task_count": len(unavailable),
            "classified_workloads": dict(sorted(classified_workloads.items())),
            "unavailable_workloads": dict(sorted(unavailable_workloads.items())),
            "multi_step": (
                "not_available: no pre-existing multi-step structural class is present "
                "in the frozen 611-task manifest"
            ),
        },
        "aggregate_by_structural_class": {
            "direct": by_class["direct"],
            "multi-step": {
                "availability": "not_available",
                "task_count": None,
                "transition_counts": None,
            },
            "branching": by_class["branching"],
            "deep": by_class["deep"],
            "unavailable": {
                "availability": "structural_class_null",
                "task_count": len(unavailable),
                "transition_counts": {
                    bucket: unavailable_counts[bucket]
                    for bucket in _TRANSITION_BUCKETS
                },
            },
        },
        "members": members,
        "sampling_caveat": (
            "Paired solved@8 transitions are descriptive outcomes from stochastic n=8 "
            "samples; this stratification does not establish a causal guidance effect."
        ),
    }


def _run_diagnostic_source(
    project_root: Path,
    source: str,
    *,
    timeout_seconds: float,
) -> DiagnosticRun:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="qwen-lean-issue93-") as temporary_dir:
            source_path = Path(temporary_dir) / "Candidate.lean"
            source_path.write_text(source, encoding="utf-8", newline="\n")
            process = subprocess.Popen(
                ["lake", "env", "lean", str(source_path)],
                cwd=project_root.resolve(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
                stdout, stderr = process.communicate()
                return DiagnosticRun(
                    category="verifier_timeout",
                    lean_exit_code=None,
                    stdout=stdout.replace(str(source_path), "Candidate.lean"),
                    stderr=stderr.replace(str(source_path), "Candidate.lean"),
                    latency_seconds=time.perf_counter() - started,
                )
    except (OSError, ValueError) as error:
        return DiagnosticRun(
            category="verifier_error",
            lean_exit_code=None,
            stdout="",
            stderr=str(error),
            latency_seconds=time.perf_counter() - started,
        )

    stdout = stdout.replace(str(source_path), "Candidate.lean")
    stderr = stderr.replace(str(source_path), "Candidate.lean")
    has_error = any(
        ": error" in line for stream in (stdout, stderr) for line in stream.splitlines()
    )
    return DiagnosticRun(
        category=(
            "verified" if process.returncode == 0 and not has_error else "lean_rejected"
        ),
        lean_exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        latency_seconds=time.perf_counter() - started,
    )


def _instrumented_source(
    task: TaskRecord,
    candidate: str,
    *,
    declaration_queries: Sequence[str] = (),
) -> str:
    queries = "\n".join(
        f"#qwenCheckDecl {json.dumps(name, ensure_ascii=False)}"
        for name in declaration_queries
    )
    middle = _DIAGNOSTIC_PRELUDE + ("\n" + queries if queries else "")
    return (
        f"{task.preamble}\n\n{middle}\n\n{render_proof_request(task.declaration)}"
        f"{normalize_transport(candidate)}\n"
    )


def _plain_source(task: TaskRecord, candidate: str) -> str:
    return f"{render_prompt(task)}{normalize_transport(candidate)}\n"


def _combined_diagnostics(run: DiagnosticRun) -> str:
    return run.stdout + "\n" + run.stderr


def _parse_state(run: DiagnosticRun) -> dict[str, Any] | None:
    match = _STATE_MESSAGE.search(_combined_diagnostics(run))
    if match is None:
        return None
    state = match.group("state").strip()
    target_matches = list(re.finditer(r"(?m)^\u22a2\s?", state))
    if not target_matches:
        return None
    first_target = target_matches[0]
    next_case = re.search(r"\n\ncase\s", state[first_target.end() :])
    target_end = (
        first_target.end() + next_case.start()
        if next_case is not None
        else len(state)
    )
    context = state[: first_target.start()].strip()
    context_lines = [
        line
        for line in context.splitlines()
        if line.strip() and not line.startswith("case ")
    ]
    focused_goal = state[first_target.end() : target_end].strip()
    return {
        "proof_state": state,
        "proof_state_sha256": _sha256_text(state),
        "open_goal_count": len(target_matches),
        "focused_goal": focused_goal,
        "focused_goal_sha256": _sha256_text(focused_goal),
        "focused_local_context": context,
        "focused_local_context_lines": context_lines,
    }


def _parse_oracle_message(run: DiagnosticRun) -> dict[str, Any] | None:
    match = _ORACLE_MESSAGE.search(_combined_diagnostics(run))
    if match is None:
        return None
    body = match.group("body")
    type_match = re.search(r"type_begin\n(?P<type>.*?)\ntype_end", body, re.DOTALL)
    if type_match is None:
        return None

    def value(label: str) -> str | None:
        observed = re.search(rf"(?m)^{re.escape(label)}: ([^\r\n]+)$", body)
        return observed.group(1) if observed is not None else None

    def boolean(label: str) -> bool | None:
        observed = value(label)
        if observed == "true":
            return True
        if observed == "false":
            return False
        return None

    inferred_type = type_match.group("type").strip()
    return {
        "unknown_name": value("unknown"),
        "inferred_type": inferred_type,
        "inferred_type_contains_metavariables": (
            re.search(r"\?m\.[0-9]+|\?_", inferred_type) is not None
        ),
        "inferred_type_is_proposition": boolean("is_prop"),
        "exact_unknown_declaration_exists": boolean(
            "exact_declaration_exists"
        ),
        "definitionally_equal_to_current_goal": boolean("current_defeq"),
        "definitionally_equal_to_root_target": boolean("root_defeq"),
    }


def _line_column_offset(text: str, line: int, column: int, needle: str) -> int | None:
    lines = text.splitlines(keepends=True)
    if line < 1 or line > len(lines):
        return None
    line_text = lines[line - 1]
    starts = [match.start() for match in re.finditer(re.escape(needle), line_text)]
    if not starts:
        return None
    expected = max(column - 1, 0)
    start = min(starts, key=lambda value: abs(value - expected))
    return sum(len(value) for value in lines[: line - 1]) + start


def _locate_first_unknown_site(
    task: Mapping[str, Any],
    bound: BoundTask,
    raw: Mapping[str, Any],
    official_result: Mapping[str, Any],
) -> tuple[UnknownReferenceSite | None, str | None]:
    raw_text = str(raw["raw_text"])
    normalized_text = normalize_transport(raw_text)
    if "\r" in raw_text or not raw_text.startswith(normalized_text):
        return None, "raw continuation has non-suffix transport normalization"
    diagnostics = _diagnostic_text(official_result)
    observations = list(_UNKNOWN_REFERENCE_AT.finditer(diagnostics))
    if not observations:
        return None, "official unknown-reference diagnostic has no source position"
    official_source = _plain_source(bound.task, raw_text)
    source_prefix_length = len(render_prompt(bound.task))
    located: list[tuple[int, re.Match[str]]] = []
    for observation in observations:
        offset = _line_column_offset(
            official_source,
            int(observation.group("line")),
            int(observation.group("column")),
            observation.group("name"),
        )
        if offset is not None and offset >= source_prefix_length:
            located.append((offset, observation))
    if not located:
        return None, "unknown-reference source position cannot be bound to raw text"
    source_offset, observation = min(located, key=lambda item: item[0])
    raw_start = source_offset - source_prefix_length
    unknown_name = observation.group("name")
    raw_end = raw_start + len(unknown_name)
    if raw_text[raw_start:raw_end] != unknown_name:
        return None, "unknown-reference source span differs from raw text"
    command_start = raw_text.rfind("\n", 0, raw_start) + 1
    command_end = raw_text.find("\n", raw_end)
    if command_end < 0:
        command_end = len(raw_text)
    return (
        UnknownReferenceSite(
            workload=str(task["workload_id"]),
            task_id=str(task["task_id"]),
            task_ordinal=int(task["ordinal"]),
            candidate_index=int(raw["candidate_index"]),
            candidate_id=str(raw["candidate_id"]),
            raw_sha256=str(raw["raw_sha256"]),
            raw_text=raw_text,
            unknown_name=unknown_name,
            unknown_kind=observation.group("kind").lower(),
            source_line=int(observation.group("line")),
            source_column=int(observation.group("column")),
            raw_start=raw_start,
            raw_end=raw_end,
            command_start=command_start,
            command_end=command_end,
        ),
        None,
    )


def _oracle_span_ends(site: UnknownReferenceSite) -> list[int]:
    ends = [site.raw_end]
    depth = 0
    index = site.raw_end
    last_nonspace = site.raw_end
    while index < site.command_end:
        char = site.raw_text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                break
            depth -= 1
            if depth == 0:
                ends.append(index + 1)
        elif depth == 0 and char in ",;":
            break
        if not char.isspace():
            last_nonspace = index + 1
        elif depth == 0 and last_nonspace > ends[-1]:
            ends.append(last_nonspace)
        index += 1
    if last_nonspace > ends[-1]:
        ends.append(last_nonspace)
    if re.search(r"\bfun\b.*=>", site.raw_text[site.raw_end : site.command_end]):
        ends.append(site.command_end)
    return sorted({end for end in ends if end <= site.command_end})


def _text_size(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "serialized_character_count": len(value),
        "serialized_utf8_byte_count": len(value.encode("utf-8")),
        "lexical_token_count": len(_LEAN_TOKEN.findall(value)),
    }


def _bounded_declaration_candidates(site: UnknownReferenceSite, declaration: str) -> list[str]:
    name = site.unknown_name
    candidates = {name}
    parts = name.split(".")
    if parts:
        capitalized = parts.copy()
        capitalized[0] = capitalized[0][:1].upper() + capitalized[0][1:]
        candidates.add(".".join(capitalized))
        pascal = parts.copy()
        pascal[0] = "".join(
            component[:1].upper() + component[1:]
            for component in pascal[0].split("_")
            if component
        )
        candidates.add(".".join(pascal))
    if "." not in name:
        namespace_sources = declaration + "\n" + site.raw_text
        namespaces = {
            identifier.split(".", 1)[0]
            for identifier in _qualified_lean_identifiers(namespace_sources)
        }
        candidates.update(f"{namespace}.{name}" for namespace in namespaces)
    return sorted(candidate for candidate in candidates if 0 < len(candidate) <= 256)[:32]


def _syntactic_reference_usage(site: UnknownReferenceSite) -> str:
    before = site.raw_text[site.command_start : site.raw_start]
    after = site.raw_text[site.raw_end : site.command_end]
    if before.strip() == "":
        return "command_head_reference_kind_not_determined"
    if re.match(r"\s+(?:\(|\{|\[|[A-Za-z0-9_'\u2080-\u2089])", after):
        return "term_application"
    if "." in site.unknown_name:
        return "namespace_qualified_term_reference"
    return "term_reference"


def _candidate_formal_obligation_rule(site: UnknownReferenceSite) -> str | None:
    before = site.raw_text[site.command_start : site.raw_start]
    if re.search(r"\bapply\s*$", before):
        return "unknown_reference_is_apply_tactic_head"
    if re.search(r"\b(?:exact|refine)\s*$", before):
        return "unknown_reference_is_exact_or_refine_proof_term_head"
    if re.search(r"\brw\s*\[[^\]\n]*$", before):
        return "unknown_reference_is_rewrite_rule"
    if re.search(r"\b(?:linarith|nlinarith)\s*\[[^\]\n]*$", before):
        return "unknown_reference_is_arithmetic_tactic_lemma_argument"
    return None


def _explicit_argument_text(site: UnknownReferenceSite) -> str:
    expression_end = max(_oracle_span_ends(site))
    return site.raw_text[site.raw_end:expression_end]


def _ordered_line_prefixes(raw_text: str) -> list[dict[str, Any]]:
    boundaries = [match.end() for match in re.finditer(r"\n", raw_text)]
    if not boundaries or boundaries[-1] != len(raw_text):
        boundaries.append(len(raw_text))
    return [
        {
            "prefix_index": index,
            "end_character_offset": end,
            "utf8_byte_count": len(raw_text[:end].encode("utf-8")),
            "prefix_sha256": _sha256_text(raw_text[:end]),
        }
        for index, end in enumerate(boundaries)
    ]


def reconstruct_regression_tasks(
    manifest: Mapping[str, Any],
    verification_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        if int(task["q0_verified_candidate_count"]) <= 0:
            continue
        categories = [
            str(verification_by_id[str(candidate_id)]["category"])
            for candidate_id in task["candidate_slots"]["B"]
        ]
        if "verified" not in categories:
            regressions.append(dict(task))
    return regressions


def candidate_format_markers(raw_text: str) -> dict[str, bool]:
    return {
        "duplicated_by": _LEADING_BY.search(raw_text) is not None,
        "theorem_repetition": _THEOREM_REPETITION.search(raw_text) is not None,
        "markdown_fence": "```" in raw_text,
        "lean3_begin": re.match(r"\A\s*begin(?=\s)", raw_text) is not None,
        "natural_language_contamination": (
            _NATURAL_LANGUAGE_PREFIX.search(raw_text) is not None
            or _NATURAL_LANGUAGE_SUFFIX.search(raw_text) is not None
        ),
        "sorry_or_admit": _SORRY_ADMIT.search(raw_text) is not None,
    }


def _one_step_transforms(raw_text: str, declaration: str) -> list[tuple[str, str]]:
    transformed: list[tuple[str, str]] = []

    match = _LEADING_BY.search(raw_text)
    if match is not None:
        transformed.append(
            (
                "strip_leading_duplicated_by",
                match.group("leading") + raw_text[match.end() :],
            )
        )

    match = _MARKDOWN_FENCE.fullmatch(raw_text)
    if match is not None:
        transformed.append(("unwrap_markdown_fence", match.group("body")))

    stripped = raw_text.lstrip()
    repeated_prefix = f"{declaration} := by"
    if stripped.startswith(repeated_prefix):
        transformed.append(
            (
                "remove_exact_repeated_theorem_declaration",
                raw_text[: len(raw_text) - len(stripped)]
                + stripped[len(repeated_prefix) :],
            )
        )

    match = _LEAN3_BEGIN_END.fullmatch(raw_text)
    if match is not None:
        transformed.append(
            (
                "unwrap_lean3_begin_end",
                match.group("leading")
                + match.group("body")
                + match.group("trailing"),
            )
        )

    match = _NATURAL_LANGUAGE_PREFIX.search(raw_text)
    if match is not None:
        transformed.append(
            (
                "remove_whitelisted_natural_language_prefix",
                match.group("leading") + raw_text[match.end() :],
            )
        )

    match = _NATURAL_LANGUAGE_SUFFIX.search(raw_text)
    if match is not None:
        transformed.append(
            ("remove_whitelisted_natural_language_suffix", raw_text[: match.start()])
        )

    return [(name, text) for name, text in transformed if text != raw_text]


def mechanical_variants(raw_text: str, declaration: str) -> list[MechanicalVariant]:
    queue: deque[tuple[tuple[str, ...], str]] = deque([((), raw_text)])
    seen_text = {raw_text}
    variants: list[MechanicalVariant] = []
    while queue:
        sequence, current = queue.popleft()
        if len(sequence) >= 4:
            continue
        for transform_name, transformed_text in _one_step_transforms(
            current, declaration
        ):
            if transform_name in sequence or transformed_text in seen_text:
                continue
            seen_text.add(transformed_text)
            transformed_sequence = (*sequence, transform_name)
            variants.append(
                MechanicalVariant(
                    transform_sequence=transformed_sequence,
                    transformed_text=transformed_text,
                    transformed_sha256=_sha256_text(transformed_text),
                )
            )
            queue.append((transformed_sequence, transformed_text))
    return sorted(
        variants,
        key=lambda variant: (
            len(variant.transform_sequence),
            variant.transform_sequence,
            variant.transformed_sha256,
        ),
    )


def _tactic_families(text: str) -> set[str]:
    return {
        tactic
        for tactic in _TACTIC_FAMILIES
        if re.search(rf"\b{re.escape(tactic)}\b", text)
    }


def _diagnostic_text(result: Mapping[str, Any]) -> str:
    diagnostics = result.get("diagnostics", {})
    return (
        str(diagnostics.get("stdout", ""))
        + "\n"
        + str(diagnostics.get("stderr", ""))
    )


def candidate_diagnostic_observations(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    text = _diagnostic_text(result)
    unknown_references = sorted(
        {
            (match.group("kind").lower(), match.group("name"))
            for match in _UNKNOWN_REFERENCE.finditer(text)
        }
    )
    return {
        "diagnostic_categories": [
            category
            for category, pattern in _DIAGNOSTIC_CATEGORY_PATTERNS
            if pattern.search(text) is not None
        ],
        "unknown_references": [
            {"kind": kind, "name": name} for kind, name in unknown_references
        ],
        "diagnostic_text_sha256": _sha256_text(text),
    }


def _qualified_lean_identifiers(text: str) -> set[str]:
    return set(_QUALIFIED_LEAN_IDENTIFIER.findall(text))


def _obvious_local_premise_identifiers(declaration: str) -> set[str]:
    return {
        name
        for match in _DECLARATION_BINDER.finditer(declaration)
        for name in match.group("names").split()
        if _LOCAL_PREMISE_NAME.fullmatch(name) is not None
    }


def _contains_lean_identifier(text: str, identifier: str) -> bool:
    return (
        re.search(
            rf"(?<![A-Za-z0-9_'₀-₉]){re.escape(identifier)}"
            rf"(?![A-Za-z0-9_'₀-₉])",
            text,
        )
        is not None
    )


def _load_q0_candidates(
    q0_root: Path,
    q0_evidence: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, dict[str, str]],
]:
    candidates: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for workload_id in WORKLOAD_IDS:
        workload_evidence = q0_evidence["workloads"][workload_id]
        workload_root = q0_root / workload_id
        generation_path = workload_root / "generations.jsonl"
        result_path = workload_root / "results.jsonl"
        expected_generation_sha = str(workload_evidence["generation_sha256"])
        expected_results_sha = str(workload_evidence["results_sha256"])
        if sha256_file(generation_path) != expected_generation_sha:
            raise ValueError(f"Q0 generation bytes differ for {workload_id}")
        if sha256_file(result_path) != expected_results_sha:
            raise ValueError(f"Q0 result bytes differ for {workload_id}")
        source_hashes[workload_id] = {
            "generations_sha256": expected_generation_sha,
            "results_sha256": expected_results_sha,
            "generation_metadata_sha256": sha256_file(
                workload_root / "generation-metadata.json"
            ),
            "run_sha256": sha256_file(workload_root / "run.json"),
            "summary_sha256": sha256_file(workload_root / "summary.json"),
        }
        for line_number, raw_line, generation in _iter_jsonl_bytes(generation_path):
            task = generation.get("task", {})
            key = (
                workload_id,
                str(task.get("id")),
                int(generation["candidate_index"]),
            )
            if key in candidates:
                raise ValueError(f"duplicate Q0 generation candidate: {key}")
            raw_text = generation.get("text")
            if not isinstance(raw_text, str):
                raise ValueError(f"Q0 raw continuation is not text: {key}")
            candidates[key] = {
                "generation": generation,
                "generation_line_number": line_number,
                "generation_row_sha256": hashlib.sha256(raw_line).hexdigest(),
                "raw_sha256": _sha256_text(raw_text),
            }
        result_count = 0
        for line_number, raw_line, result in _iter_jsonl_bytes(result_path):
            key = (
                workload_id,
                str(result["task_id"]),
                int(result["candidate_index"]),
            )
            candidate = candidates.get(key)
            if candidate is None or "result" in candidate:
                raise ValueError(f"Q0 result identity differs: {key}")
            generation = candidate["generation"]
            if (
                result.get("candidate_text") != generation.get("text")
                or result.get("finish_reason") != generation.get("finish_reason")
                or int(result.get("generated_token_count", -1))
                != int(generation.get("token_count", -2))
                or result.get("category")
                not in {
                    "verified",
                    "lean_rejected",
                    "verifier_timeout",
                    "verifier_error",
                }
            ):
                raise ValueError(f"Q0 generation/result bytes differ: {key}")
            candidate.update(
                {
                    "result": result,
                    "result_line_number": line_number,
                    "result_row_sha256": hashlib.sha256(raw_line).hexdigest(),
                }
            )
            result_count += 1
        expected_candidates = int(workload_evidence["candidate_count"])
        observed = [key for key in candidates if key[0] == workload_id]
        if len(observed) != expected_candidates or result_count != expected_candidates:
            raise ValueError(f"Q0 candidate count differs for {workload_id}")
    if any("result" not in candidate for candidate in candidates.values()):
        raise ValueError("Q0 candidate without authoritative classification")
    return candidates, source_hashes


def _raw_b_rows(
    regressions: Sequence[Mapping[str, Any]],
    generation_inventory: Mapping[str, Any],
    verification_inventory: Mapping[str, Any],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    shard_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in generation_inventory["shards"]
    }
    verification_hashes = dict(verification_inventory["file_hashes"])
    rows: list[dict[str, Any]] = []
    for task in regressions:
        for candidate_id_value in task["candidate_slots"]["B"]:
            candidate_id = str(candidate_id_value)
            generation = generation_inventory["candidates_by_id"][candidate_id]
            verification = verification_inventory["results_by_id"][candidate_id]
            raw_text = generation["raw_continuation"]
            if not isinstance(raw_text, str):
                raise ValueError(f"Arm-B raw continuation is not text: {candidate_id}")
            verification_path = str(
                (
                    Path("verifications")
                    / "B"
                    / str(task["workload_id"])
                    / f"{candidate_id}.json"
                )
            )
            rows.append(
                {
                    "schema_version": RAW_B_SCHEMA_VERSION,
                    "workload": task["workload_id"],
                    "task_id": task["task_id"],
                    "task_ordinal": task["ordinal"],
                    "candidate_index": generation["candidate_index"],
                    "candidate_id": candidate_id,
                    "raw_text": raw_text,
                    "raw_sha256": generation["raw_continuation_sha256"],
                    "finish_reason": generation["finish_reason"],
                    "generated_token_count": generation["token_count"],
                    "official_verifier_classification": verification["category"],
                    "official_lean_exit_code": verification["lean_exit_code"],
                    "official_verifier_environment_sha256": verification[
                        "verifier_environment_sha256"
                    ],
                    "official_verification_result_sha256": verification_hashes[
                        verification_path
                    ],
                    "source_generation_shard": generation["generation_shard"],
                    "source_generation_shard_sha256": shard_hashes[
                        generation["generation_shard"]
                    ],
                    "manifest_sha256": manifest_sha256,
                }
            )
    if len(rows) != EXPECTED_REGRESSION_TASKS * EXPECTED_CANDIDATES_PER_TASK:
        raise ValueError("regression Arm-B raw candidate count differs")
    if any(row["raw_sha256"] != _sha256_text(row["raw_text"]) for row in rows):
        raise ValueError("regression Arm-B raw text hash differs")
    return rows


def _raw_q0_rows(
    regressions: Sequence[Mapping[str, Any]],
    q0_candidates: Mapping[tuple[str, str, int], Mapping[str, Any]],
    q0_evidence_sha256: str,
    q0_source_hashes: Mapping[str, Mapping[str, str]],
    model_revision: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in regressions:
        workload_id = str(task["workload_id"])
        task_id = str(task["task_id"])
        selected = [
            candidate
            for (workload, observed_task, _), candidate in q0_candidates.items()
            if workload == workload_id
            and observed_task == task_id
            and candidate["result"]["category"] == "verified"
        ]
        selected.sort(key=lambda candidate: candidate["generation"]["candidate_index"])
        if len(selected) != int(task["q0_verified_candidate_count"]):
            raise ValueError(f"Q0 verified count differs for regression task: {task_id}")
        for candidate in selected:
            generation = candidate["generation"]
            result = candidate["result"]
            candidate_index = int(generation["candidate_index"])
            stable_identity = "q0-candidate-" + _sha256_json(
                {
                    "workload": workload_id,
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "model_revision": model_revision,
                    "generation_sha256": q0_source_hashes[workload_id][
                        "generations_sha256"
                    ],
                }
            )
            rows.append(
                {
                    "schema_version": RAW_Q0_SCHEMA_VERSION,
                    "workload": workload_id,
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "candidate_id": stable_identity,
                    "source_candidate_id": result["candidate_id"],
                    "source_model_revision": model_revision,
                    "raw_text": generation["text"],
                    "raw_sha256": candidate["raw_sha256"],
                    "finish_reason": generation["finish_reason"],
                    "generated_token_count": generation["token_count"],
                    "authoritative_q0_classification": result["category"],
                    "authoritative_lean_exit_code": result["lean_exit_code"],
                    "authoritative_diagnostics": result["diagnostics"],
                    "task": generation["task"],
                    "source_generation_file_sha256": q0_source_hashes[workload_id][
                        "generations_sha256"
                    ],
                    "source_result_file_sha256": q0_source_hashes[workload_id][
                        "results_sha256"
                    ],
                    "source_generation_line_number": candidate[
                        "generation_line_number"
                    ],
                    "source_generation_row_sha256": candidate[
                        "generation_row_sha256"
                    ],
                    "source_result_line_number": candidate["result_line_number"],
                    "source_result_row_sha256": candidate["result_row_sha256"],
                    "q0_evidence_sha256": q0_evidence_sha256,
                    "q0_raw_recovery_archive_sha256": RECOVERY_ARCHIVE_SHA256,
                }
            )
    if any(row["raw_sha256"] != _sha256_text(row["raw_text"]) for row in rows):
        raise ValueError("regression Q0 raw text hash differs")
    return rows


def _verify_variants(
    variant_jobs: Sequence[dict[str, Any]],
    tasks_by_id: Mapping[tuple[str, str], BoundTask],
    config: PromptABConfig,
    lean_project_roots: Mapping[str, Path],
    *,
    workers: int,
) -> dict[str, VerificationOutcome]:
    verifiers = {
        workload_id: LeanVerifier(
            lean_project_roots[workload_id],
            timeout_seconds=float(config.verifier["timeout_seconds"]),
        )
        for workload_id in WORKLOAD_IDS
    }
    variant_task_keys = {
        (str(job["workload"]), str(job["task_id"])) for job in variant_jobs
    }
    for workload_id in WORKLOAD_IDS:
        preambles = {
            bound.task.preamble
            for key, bound in tasks_by_id.items()
            if key in variant_task_keys and key[0] == workload_id
        }
        for preamble in sorted(preambles):
            failure = verifiers[workload_id].prime_preamble(
                preamble,
                timeout_seconds=VERIFIER_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
            )
            if failure is not None:
                raise RuntimeError(
                    f"regression verifier environment probe failed: {workload_id}: "
                    f"{failure.category}"
                )

    outcomes: dict[str, VerificationOutcome] = {}

    def verify(job: Mapping[str, Any]) -> tuple[str, VerificationOutcome]:
        key = (str(job["workload"]), str(job["task_id"]))
        return str(job["variant_id"]), verifiers[key[0]].verify(
            tasks_by_id[key].task, str(job["transformed_text"])
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(verify, job) for job in variant_jobs]
        for future in as_completed(futures):
            variant_id, outcome = future.result()
            outcomes[variant_id] = outcome
    return outcomes


def _build_variant_jobs(
    regressions: Sequence[Mapping[str, Any]],
    raw_b_rows: Sequence[Mapping[str, Any]],
    tasks_by_id: Mapping[tuple[str, str], BoundTask],
) -> list[dict[str, Any]]:
    raw_by_id = {str(row["candidate_id"]): row for row in raw_b_rows}
    jobs: list[dict[str, Any]] = []
    for task in regressions:
        key = (str(task["workload_id"]), str(task["task_id"]))
        declaration = tasks_by_id[key].task.declaration
        for candidate_id_value in task["candidate_slots"]["B"]:
            candidate_id = str(candidate_id_value)
            raw = raw_by_id[candidate_id]
            for variant in mechanical_variants(str(raw["raw_text"]), declaration):
                variant_id = "mechanical-variant-" + _sha256_json(
                    {
                        "source_raw_sha256": raw["raw_sha256"],
                        "transform_sequence": variant.transform_sequence,
                        "transformed_sha256": variant.transformed_sha256,
                    }
                )
                jobs.append(
                    {
                        "variant_id": variant_id,
                        "workload": key[0],
                        "task_id": key[1],
                        "candidate_id": candidate_id,
                        "candidate_index": raw["candidate_index"],
                        "source_raw_sha256": raw["raw_sha256"],
                        "transform_sequence": list(variant.transform_sequence),
                        "transformed_text": variant.transformed_text,
                        "transformed_sha256": variant.transformed_sha256,
                    }
                )
    jobs.sort(
        key=lambda job: (
            next(
                int(task["ordinal"])
                for task in regressions
                if task["task_id"] == job["task_id"]
                and task["workload_id"] == job["workload"]
            ),
            int(job["candidate_index"]),
            tuple(job["transform_sequence"]),
            str(job["variant_id"]),
        )
    )
    return jobs


def _diagnostic_run_payload(run: DiagnosticRun) -> dict[str, Any]:
    return {
        "category": run.category,
        "lean_exit_code": run.lean_exit_code,
        "diagnostics": {"stdout": run.stdout, "stderr": run.stderr},
    }


def _first_error(
    run: DiagnosticRun,
    *,
    at_or_after: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    matches = list(_LEAN_ERROR_AT.finditer(_combined_diagnostics(run)))
    if at_or_after is not None:
        matches = [
            match
            for match in matches
            if (int(match.group("line")), int(match.group("column")))
            >= at_or_after
        ]
    if not matches:
        return None
    match = min(
        matches,
        key=lambda value: (int(value.group("line")), int(value.group("column"))),
    )
    return {
        "line": int(match.group("line")),
        "column": int(match.group("column")),
        "message": match.group("message"),
    }


def _furthest_prefix_from_error(
    task: TaskRecord,
    transformed_text: str,
    error: Mapping[str, Any] | None,
) -> str:
    if error is None:
        return transformed_text
    source = _plain_source(task, transformed_text)
    lines = source.splitlines(keepends=True)
    line = int(error["line"])
    if line < 1 or line > len(lines):
        return ""
    source_offset = sum(len(value) for value in lines[: line - 1])
    raw_offset = source_offset - len(render_prompt(task))
    return transformed_text[: max(0, min(raw_offset, len(transformed_text)))]


def _structural_progress(
    root_state: Mapping[str, Any], current_state: Mapping[str, Any]
) -> bool:
    return not (
        int(root_state["open_goal_count"]) == int(current_state["open_goal_count"])
        and root_state["focused_goal"] == current_state["focused_goal"]
        and root_state["focused_local_context_lines"]
        == current_state["focused_local_context_lines"]
    )


def _build_unknown_reference_evidence(
    regressions: Sequence[Mapping[str, Any]],
    raw_b_rows: Sequence[Mapping[str, Any]],
    tasks_by_id: Mapping[tuple[str, str], BoundTask],
    verification_by_id: Mapping[str, Mapping[str, Any]],
    config: PromptABConfig,
    lean_project_roots: Mapping[str, Path],
    environment_sha256_by_workload: Mapping[str, str],
    *,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_by_id = {str(row["candidate_id"]): row for row in raw_b_rows}
    specs: list[dict[str, Any]] = []
    total_unknown_occurrences = 0
    for task in regressions:
        key = (str(task["workload_id"]), str(task["task_id"]))
        for candidate_id_value in task["candidate_slots"]["B"]:
            candidate_id = str(candidate_id_value)
            result = verification_by_id[candidate_id]
            observations = candidate_diagnostic_observations(result)
            references = observations["unknown_references"]
            total_unknown_occurrences += len(references)
            if not references:
                continue
            raw = raw_by_id[candidate_id]
            site, location_failure = _locate_first_unknown_site(
                task, tasks_by_id[key], raw, result
            )
            specs.append(
                {
                    "task": task,
                    "key": key,
                    "raw": raw,
                    "result": result,
                    "all_unknown_references": references,
                    "site": site,
                    "location_failure": location_failure,
                }
            )

    located_sites = [spec["site"] for spec in specs if spec["site"] is not None]
    queries_by_task: dict[tuple[str, str], set[str]] = {}
    for site in located_sites:
        assert isinstance(site, UnknownReferenceSite)
        bound = tasks_by_id[(site.workload, site.task_id)]
        queries_by_task.setdefault((site.workload, site.task_id), set()).update(
            _bounded_declaration_candidates(site, bound.task.declaration)
        )

    timeout_seconds = float(config.verifier["timeout_seconds"])

    def root_job(key: tuple[str, str]) -> tuple[tuple[str, str], dict[str, Any]]:
        bound = tasks_by_id[key]
        queries = sorted(queries_by_task[key])
        run = _run_diagnostic_source(
            lean_project_roots[key[0]],
            _instrumented_source(
                bound.task,
                "__qwenCaptureAndClose",
                declaration_queries=queries,
            ),
            timeout_seconds=timeout_seconds,
        )
        declaration_checks = {
            match.group("name"): match.group("exists") == "true"
            for match in _DECLARATION_CHECK_MESSAGE.finditer(
                _combined_diagnostics(run)
            )
        }
        return key, {
            "state": _parse_state(run),
            "declaration_checks": declaration_checks,
            "run": run,
        }

    root_results: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(root_job, key) for key in sorted(queries_by_task)]
        for future in as_completed(futures):
            key, result = future.result()
            root_results[key] = result

    def candidate_job(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        task = spec["task"]
        raw = spec["raw"]
        key = spec["key"]
        bound = tasks_by_id[key]
        site = spec["site"]
        base = {
            "workload": key[0],
            "task_id": key[1],
            "candidate_index": int(raw["candidate_index"]),
            "candidate_id": str(raw["candidate_id"]),
            "source_raw_sha256": str(raw["raw_sha256"]),
            "all_official_unknown_references": spec["all_unknown_references"],
            "official_verifier_classification": spec["result"]["category"],
            "official_verification_result_sha256": raw[
                "official_verification_result_sha256"
            ],
            "verifier_environment_sha256": environment_sha256_by_workload[key[0]],
            "scoring_excluded": True,
        }
        if site is None:
            reason = str(spec["location_failure"])
            first_reference = (
                spec["all_unknown_references"][0]
                if spec["all_unknown_references"]
                else None
            )
            evidence = {
                **base,
                "first_unknown_call_site_reconstructed": False,
                "first_unknown_reference": first_reference,
                "availability_limitation": reason,
                "candidate_node_status": "candidate_node_not_determined",
                "candidate_node_not_determined_reason": (
                    "the first unknown-reference call site is unavailable"
                ),
                "candidate_formal_obligation": None,
                "oracle_outcome": "oracle_not_testable",
            }
            oracle = {
                "schema_version": ORACLE_SCHEMA_VERSION,
                **base,
                "first_unknown_reference": first_reference,
                "oracle_outcome": "oracle_not_testable",
                "oracle_not_testable_reason": reason,
                "transformed_text": None,
                "transformed_sha256": None,
            }
            return evidence, oracle

        assert isinstance(site, UnknownReferenceSite)
        root_result = root_results.get(key)
        root_state = root_result["state"] if root_result is not None else None
        declaration_candidates = _bounded_declaration_candidates(
            site, bound.task.declaration
        )
        declaration_checks = (
            root_result["declaration_checks"] if root_result is not None else {}
        )
        known_matches = sorted(
            name for name in declaration_candidates if declaration_checks.get(name)
        )
        exact_raw_prefix = site.raw_text[: site.command_start]
        failing_action = site.raw_text[site.command_start : site.command_end]
        failing_indentation_match = re.match(r"[ \t]*", failing_action)
        failing_indentation = (
            failing_indentation_match.group(0)
            if failing_indentation_match is not None
            else "  "
        )
        capture_candidate = exact_raw_prefix + (
            "" if exact_raw_prefix.endswith("\n") else "\n"
        ) + failing_indentation + "__qwenCaptureAndClose"
        prefix_run = _run_diagnostic_source(
            lean_project_roots[key[0]],
            _instrumented_source(bound.task, capture_candidate),
            timeout_seconds=timeout_seconds,
        )
        current_state = _parse_state(prefix_run)
        prefix_closes_parent = False
        if current_state is None:
            prefix_only = _run_diagnostic_source(
                lean_project_roots[key[0]],
                _plain_source(bound.task, exact_raw_prefix),
                timeout_seconds=timeout_seconds,
            )
            prefix_closes_parent = prefix_only.category == "verified"
        else:
            prefix_only = None

        candidate_node_rule = _candidate_formal_obligation_rule(site)
        candidate_formal_obligation = None
        if current_state is not None and candidate_node_rule is not None:
            candidate_formal_obligation = {
                "source_binding": {
                    "workload": key[0],
                    "task_id": key[1],
                    "candidate_id": site.candidate_id,
                    "source_raw_sha256": site.raw_sha256,
                    "official_source_line": site.source_line,
                    "official_source_column": site.source_column,
                },
                "public_context_sha256": task["public_context_sha256"],
                "verifier_environment_sha256": environment_sha256_by_workload[
                    key[0]
                ],
                "extraction_rule": candidate_node_rule,
                "exact_goal_before_call": current_state["focused_goal"],
                "local_context_before_call": current_state[
                    "focused_local_context"
                ],
                "explicit_argument_text": _explicit_argument_text(site),
                "interpretation": "not_determined",
            }
        if candidate_formal_obligation is not None:
            candidate_node_status = "candidate_formal_obligation_extracted"
            candidate_node_not_determined_reason = None
        elif current_state is None:
            candidate_node_status = "candidate_node_not_determined"
            candidate_node_not_determined_reason = (
                "the proof state immediately before the failing action is unavailable"
            )
        else:
            candidate_node_status = "candidate_node_not_determined"
            candidate_node_not_determined_reason = (
                "a theorem/lemma application position is not mechanically established "
                "by the bounded syntax rule"
            )

        raw_window_start = max(0, site.raw_start - 160)
        raw_window_end = min(len(site.raw_text), site.raw_end + 160)
        evidence_base = {
            **base,
            "first_unknown_call_site_reconstructed": True,
            "first_unknown_reference": {
                "name": site.unknown_name,
                "kind": site.unknown_kind,
            },
            "source_span": {
                "official_source_line": site.source_line,
                "official_source_column": site.source_column,
                "raw_start_character_offset": site.raw_start,
                "raw_end_character_offset": site.raw_end,
            },
            "raw_text_around_reference": site.raw_text[
                raw_window_start:raw_window_end
            ],
            "failing_action": failing_action,
            "candidate_prefix_immediately_before_failing_action": exact_raw_prefix,
            "candidate_prefix_sha256": _sha256_text(exact_raw_prefix),
            "candidate_prefix_accepted_by_lean": current_state is not None,
            "candidate_prefix_closes_parent_before_failing_action": (
                prefix_closes_parent
            ),
            "prefix_replay_result": _diagnostic_run_payload(prefix_run),
            "proof_state_immediately_before_failing_action": current_state,
            "already_closed_goal_count": None,
            "already_closed_goal_count_availability": (
                "not observable from the frozen tactic-state snapshot"
            ),
            "syntactic_reference_usage": _syntactic_reference_usage(site),
            "exact_text_after_reference_in_failing_action": site.raw_text[
                site.raw_end : site.command_end
            ],
            "bounded_declaration_candidates_checked": declaration_candidates,
            "known_nearby_or_qualified_declaration_matches": known_matches,
            "mechanical_reference_evidence_class": (
                "exact_bounded_declaration_match"
                if known_matches
                else (
                    "candidate_formal_obligation_extracted"
                    if candidate_formal_obligation is not None
                    else "undetermined"
                )
            ),
            "declaration_match_rule": (
                "Exact frozen-environment membership only over the original name, a "
                "capitalized/PascalCase first namespace component, and exact namespace "
                "prefixes already present in the task declaration or candidate."
            ),
            "candidate_node_status": candidate_node_status,
            "candidate_node_not_determined_reason": (
                candidate_node_not_determined_reason
            ),
            "candidate_formal_obligation": candidate_formal_obligation,
        }
        if root_state is None or current_state is None:
            reason = (
                "root proof state unavailable"
                if root_state is None
                else (
                    "prefix already closes parent before failing action"
                    if prefix_closes_parent
                    else "prefix state is not Lean-replayable at the line boundary"
                )
            )
            evidence = {
                **evidence_base,
                "availability_limitation": reason,
                "oracle_outcome": "oracle_not_testable",
            }
            oracle = {
                "schema_version": ORACLE_SCHEMA_VERSION,
                **base,
                "first_unknown_reference": evidence_base[
                    "first_unknown_reference"
                ],
                "candidate_prefix_sha256": evidence_base["candidate_prefix_sha256"],
                "oracle_outcome": "oracle_not_testable",
                "oracle_not_testable_reason": reason,
                "transformed_text": None,
                "transformed_sha256": None,
            }
            return evidence, oracle

        selected_end: int | None = None
        base_oracle_message: dict[str, Any] | None = None
        oracle_probe_run: DiagnosticRun | None = None
        oracle_run: DiagnosticRun | None = None
        transformed_text: str | None = None
        unknown_literal = json.dumps(site.unknown_name, ensure_ascii=False)
        for span_end in _oracle_span_ends(site):
            sentinel = f"(__qwenOracle {unknown_literal})"
            probe_text = (
                site.raw_text[: site.raw_start]
                + sentinel
                + site.raw_text[span_end:]
            )
            observed_run = _run_diagnostic_source(
                lean_project_roots[key[0]],
                _instrumented_source(bound.task, probe_text),
                timeout_seconds=timeout_seconds,
            )
            observed_message = _parse_oracle_message(observed_run)
            if (
                observed_message is not None
                and not observed_message["inferred_type_contains_metavariables"]
            ):
                candidate_transformed_text = (
                    site.raw_text[: site.raw_start]
                    + "(by sorry)"
                    + site.raw_text[span_end:]
                )
                candidate_oracle_run = _run_diagnostic_source(
                    lean_project_roots[key[0]],
                    _plain_source(bound.task, candidate_transformed_text),
                    timeout_seconds=timeout_seconds,
                )
                same_site_syntax_error = any(
                    int(error.group("line")) == site.source_line
                    and "unexpected token" in error.group("message").lower()
                    for error in _LEAN_ERROR_AT.finditer(
                        _combined_diagnostics(candidate_oracle_run)
                    )
                )
                if same_site_syntax_error:
                    continue
                selected_end = span_end
                base_oracle_message = observed_message
                oracle_probe_run = observed_run
                oracle_run = candidate_oracle_run
                transformed_text = candidate_transformed_text
                break

        if (
            selected_end is None
            or base_oracle_message is None
            or oracle_probe_run is None
            or oracle_run is None
            or transformed_text is None
        ):
            reason = "contextual oracle type cannot be inferred for any bounded expression span"
            evidence = {
                **evidence_base,
                "root_proof_state": root_state,
                "availability_limitation": reason,
                "oracle_outcome": "oracle_not_testable",
            }
            oracle = {
                "schema_version": ORACLE_SCHEMA_VERSION,
                **base,
                "first_unknown_reference": evidence_base[
                    "first_unknown_reference"
                ],
                "candidate_prefix_sha256": evidence_base["candidate_prefix_sha256"],
                "oracle_outcome": "oracle_not_testable",
                "oracle_not_testable_reason": reason,
                "transformed_text": None,
                "transformed_sha256": None,
            }
            return evidence, oracle

        replaced_expression = site.raw_text[site.raw_start:selected_end]
        transformed_sha256 = _sha256_text(transformed_text)

        compare_sentinel = (
            f"(__qwenOracleCompare {unknown_literal}, "
            f"({current_state['focused_goal']}), ({root_state['focused_goal']}))"
        )
        compare_text = (
            site.raw_text[: site.raw_start]
            + compare_sentinel
            + site.raw_text[selected_end:]
        )
        compare_run = _run_diagnostic_source(
            lean_project_roots[key[0]],
            _instrumented_source(bound.task, compare_text),
            timeout_seconds=timeout_seconds,
        )
        comparison = _parse_oracle_message(compare_run)
        oracle_message = dict(base_oracle_message)
        if (
            comparison is not None
            and not comparison["inferred_type_contains_metavariables"]
        ):
            oracle_message.update(
                {
                    "definitionally_equal_to_current_goal": comparison[
                        "definitionally_equal_to_current_goal"
                    ],
                    "definitionally_equal_to_root_target": comparison[
                        "definitionally_equal_to_root_target"
                    ],
                }
            )

        later_unknowns = [
            {
                "name": match.group("name"),
                "kind": match.group("kind").lower(),
                "line": int(match.group("line")),
                "column": int(match.group("column")),
            }
            for match in _UNKNOWN_REFERENCE_AT.finditer(
                _combined_diagnostics(oracle_run)
            )
            if match.group("name") != site.unknown_name
        ]
        next_error = _first_error(
            oracle_run,
            at_or_after=(site.source_line, site.source_column),
        )
        if oracle_run.category == "verified":
            outcome = "oracle_closes_parent"
        elif later_unknowns:
            outcome = "oracle_reaches_second_unknown"
        elif next_error is not None and (
            int(next_error["line"]), int(next_error["column"])
        ) > (site.source_line, site.source_column):
            outcome = "oracle_advances_then_fails"
        else:
            outcome = "oracle_no_material_progress"

        progress = _structural_progress(root_state, current_state)
        new_context_lines = [
            line
            for line in current_state["focused_local_context_lines"]
            if line not in root_state["focused_local_context_lines"]
        ]
        root_defeq = oracle_message["definitionally_equal_to_root_target"]
        current_defeq = oracle_message["definitionally_equal_to_current_goal"]
        if root_defeq is True and not progress:
            anti_vacuity_flag = "root_target_restatement"
        elif current_defeq is True and progress:
            anti_vacuity_flag = "current_subgoal_oracle"
        elif (
            oracle_message["inferred_type_is_proposition"] is True
            and root_defeq is False
            and current_defeq is False
        ):
            anti_vacuity_flag = "strict_intermediate_fact"
        else:
            anti_vacuity_flag = "equivalence_not_determined"

        furthest_prefix = _furthest_prefix_from_error(
            bound.task, transformed_text, next_error
        )
        candidate_node = None
        if anti_vacuity_flag in {
            "current_subgoal_oracle",
            "strict_intermediate_fact",
        }:
            candidate_node = {
                "source_binding": {
                    "workload": key[0],
                    "task_id": key[1],
                    "candidate_id": site.candidate_id,
                    "source_raw_sha256": site.raw_sha256,
                    "official_source_line": site.source_line,
                    "official_source_column": site.source_column,
                },
                "public_context_sha256": task["public_context_sha256"],
                "verifier_environment_sha256": environment_sha256_by_workload[
                    key[0]
                ],
                "local_context": current_state["focused_local_context"],
                "exact_proposition": oracle_message["inferred_type"],
                "interpretation": "not_determined",
            }
        mechanical_reference_class = (
            "exact_bounded_declaration_match"
            if known_matches
            else (
                "candidate_formal_obligation_extracted"
                if candidate_formal_obligation is not None
                else (
                    "oracle_candidate_node_extracted"
                    if candidate_node is not None
                    else "undetermined"
                )
            )
        )

        evidence = {
            **evidence_base,
            "root_proof_state": root_state,
            "oracle_expression_span": {
                "raw_start_character_offset": site.raw_start,
                "raw_end_character_offset": selected_end,
                "exact_replaced_expression": replaced_expression,
                "exact_supplied_argument_text": replaced_expression[
                    len(site.unknown_name) :
                ],
            },
            "oracle_inferred_type": oracle_message["inferred_type"],
            "oracle_type_is_proposition": oracle_message[
                "inferred_type_is_proposition"
            ],
            "oracle_definitionally_equal_to_root_target": root_defeq,
            "oracle_definitionally_equal_to_current_goal": current_defeq,
            "oracle_equivalence_check_available": comparison is not None,
            "call_occurs_before_observable_proof_state_progress": not progress,
            "new_local_context_lines_before_call": new_context_lines,
            "root_target_restatement_with_same_effective_context": (
                root_defeq is True
                and not progress
                and not new_context_lines
            ),
            "size_diagnostics": {
                "root_target": _text_size(root_state["focused_goal"]),
                "current_goal": _text_size(current_state["focused_goal"]),
                "oracle_type": _text_size(oracle_message["inferred_type"]),
            },
            "anti_vacuity_flag": anti_vacuity_flag,
            "oracle_outcome": outcome,
            "second_unknown_references_after_oracle": later_unknowns,
            "next_failure_after_oracle": next_error,
            "furthest_valid_prefix_after_oracle": furthest_prefix,
            "furthest_valid_prefix_after_oracle_sha256": _sha256_text(
                furthest_prefix
            ),
            "oracle_transform_sha256": transformed_sha256,
            "candidate_node": candidate_node,
            "mechanical_reference_evidence_class": mechanical_reference_class,
        }
        oracle = {
            "schema_version": ORACLE_SCHEMA_VERSION,
            **base,
            "first_unknown_reference": evidence_base["first_unknown_reference"],
            "source_span": evidence_base["source_span"],
            "candidate_prefix_sha256": evidence_base["candidate_prefix_sha256"],
            "candidate_prefix_accepted_by_lean": True,
            "proof_state_immediately_before_failing_action": current_state,
            "exact_replaced_expression": replaced_expression,
            "replacement": "(by sorry)",
            "single_node_only": True,
            "transformed_text": transformed_text,
            "transformed_sha256": transformed_sha256,
            "inferred_oracle_type": oracle_message["inferred_type"],
            "anti_vacuity_flag": anti_vacuity_flag,
            "oracle_outcome": outcome,
            "second_unknown_references_after_oracle": later_unknowns,
            "next_failure_after_oracle": next_error,
            "furthest_valid_prefix_after_oracle": furthest_prefix,
            "furthest_valid_prefix_after_oracle_sha256": _sha256_text(
                furthest_prefix
            ),
            "oracle_verification": _diagnostic_run_payload(oracle_run),
            "oracle_probe_diagnostics_sha256": _sha256_text(
                _combined_diagnostics(oracle_probe_run)
            ),
            "equivalence_probe_diagnostics_sha256": _sha256_text(
                _combined_diagnostics(compare_run)
            ),
            "candidate_node": candidate_node,
            "mechanical_reference_evidence_class": mechanical_reference_class,
        }
        return evidence, oracle

    evidence_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(candidate_job, spec) for spec in specs]
        for future in as_completed(futures):
            evidence, oracle = future.result()
            evidence_rows.append(evidence)
            oracle_rows.append(oracle)
    evidence_rows.sort(
        key=lambda row: (
            next(
                int(task["ordinal"])
                for task in regressions
                if task["workload_id"] == row["workload"]
                and task["task_id"] == row["task_id"]
            ),
            int(row["candidate_index"]),
        )
    )
    oracle_rows.sort(
        key=lambda row: (
            next(
                int(task["ordinal"])
                for task in regressions
                if task["workload_id"] == row["workload"]
                and task["task_id"] == row["task_id"]
            ),
            int(row["candidate_index"]),
        )
    )
    outcome_counts = Counter(str(row["oracle_outcome"]) for row in evidence_rows)
    flag_counts = Counter(
        str(row["anti_vacuity_flag"])
        for row in evidence_rows
        if row.get("anti_vacuity_flag") is not None
    )
    candidate_nodes = [
        row["candidate_node"]
        for row in evidence_rows
        if row.get("candidate_node") is not None
    ]
    candidate_formal_obligations = [
        row["candidate_formal_obligation"]
        for row in evidence_rows
        if row.get("candidate_formal_obligation") is not None
    ]
    reference_class_counts = Counter(
        str(row.get("mechanical_reference_evidence_class", "undetermined"))
        for row in evidence_rows
    )
    return (
        {
            "official_unknown_reference_candidate_count": len(specs),
            "official_unknown_reference_occurrence_count": (
                total_unknown_occurrences
            ),
            "first_call_site_reconstructed_count": sum(
                bool(row["first_unknown_call_site_reconstructed"])
                for row in evidence_rows
            ),
            "prefix_replay_accepted_count": sum(
                bool(row.get("candidate_prefix_accepted_by_lean"))
                for row in evidence_rows
            ),
            "oracle_outcome_counts": {
                outcome: outcome_counts[outcome] for outcome in _ORACLE_OUTCOMES
            },
            "anti_vacuity_flag_counts": {
                flag: flag_counts[flag] for flag in _ANTI_VACUITY_FLAGS
            },
            "candidate_nodes_for_future_prove_the_node_experiment": candidate_nodes,
            "candidate_formal_obligations_from_unknown_call_sites": (
                candidate_formal_obligations
            ),
            "candidate_node_status_counts": dict(
                sorted(
                    Counter(
                        str(row["candidate_node_status"])
                        for row in evidence_rows
                    ).items()
                )
            ),
            "mechanical_reference_evidence_class_counts": dict(
                sorted(reference_class_counts.items())
            ),
            "call_sites": evidence_rows,
            "availability_and_scope": {
                "diagnostic_timeout_seconds": timeout_seconds,
                "first_unknown_only": True,
                "recursive_oracle_injection": False,
                "official_rescoring": False,
                "model_inference_or_regeneration": False,
                "semantic_api_or_decomposition_judgment": "not_determined",
                "candidate_formal_obligation_rule": (
                    "Preserve the captured focused goal only when the unknown is "
                    "mechanically the head after apply/exact/refine, a rw rule, or an "
                    "explicit linarith/nlinarith lemma argument; all other positions "
                    "are candidate_node_not_determined."
                ),
            },
        },
        oracle_rows,
    )


def _stopping_control_evidence(
    task_rows: Sequence[Mapping[str, Any]],
    raw_b_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw_by_id = {str(row["candidate_id"]): row for row in raw_b_rows}
    no_goals: list[dict[str, Any]] = []
    for task in task_rows:
        for candidate in task["b_candidate_diagnostics"]:
            if "no_goals_to_be_solved" not in candidate["diagnostic_categories"]:
                continue
            raw = raw_by_id[str(candidate["candidate_id"])]
            no_goals.append(
                {
                    "workload": task["workload"],
                    "task_id": task["task_id"],
                    "candidate_index": candidate["candidate_index"],
                    "candidate_id": candidate["candidate_id"],
                    "source_raw_sha256": raw["raw_sha256"],
                    "ordered_line_prefixes": _ordered_line_prefixes(
                        str(raw["raw_text"])
                    ),
                    "prefixes_lean_verified": False,
                    "recovery_classification_changed": False,
                }
            )
    no_goals.sort(
        key=lambda row: (
            str(row["workload"]),
            str(row["task_id"]),
            int(row["candidate_index"]),
        )
    )
    token_limited = [
        {
            "workload": row["workload"],
            "task_id": row["task_id"],
            "candidate_index": row["candidate_index"],
            "candidate_id": row["candidate_id"],
            "source_raw_sha256": row["raw_sha256"],
            "generated_token_count": row["generated_token_count"],
        }
        for row in raw_b_rows
        if row["finish_reason"] == "token_limit"
    ]
    return {
        "no_goals_to_be_solved": {
            "official_candidate_count": len(no_goals),
            "candidates": no_goals,
            "scope": (
                "Ordered raw line-boundary prefix hashes only; no prefix is classified "
                "as recovered or Lean-verified by this diagnostic."
            ),
        },
        "token_limit": {
            "candidate_count": len(token_limited),
            "task_count": len(
                {(row["workload"], row["task_id"]) for row in token_limited}
            ),
            "candidates": token_limited,
        },
        "separation_rule": (
            "No-goals control evidence, token-limit stopping evidence, unknown-reference "
            "evidence, and output-format recoveries are reported independently."
        ),
    }


def _diagnostic_tags(
    raw_rows: Sequence[Mapping[str, Any]],
    official_results: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    intuition_text: str,
    declaration: str,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    q0_family_sets = [
        _tactic_families(str(row["raw_text"])) for row in q0_rows
    ]
    q0_families = set().union(*q0_family_sets)
    b_family_sets = [_tactic_families(str(row["raw_text"])) for row in raw_rows]
    b_families = set().union(*b_family_sets)
    intuition_families = _tactic_families(intuition_text)
    candidate_diagnostics = [
        {
            "candidate_index": raw["candidate_index"],
            "candidate_id": raw["candidate_id"],
            "raw_sha256": raw["raw_sha256"],
            "official_verifier_classification": result["category"],
            "official_lean_exit_code": result["lean_exit_code"],
            **candidate_diagnostic_observations(result),
        }
        for raw, result in zip(raw_rows, official_results, strict=True)
    ]
    unknown_reference_candidate_count = sum(
        any(
            category
            in {"unknown_identifier", "unknown_constant", "unknown_declaration"}
            for category in row["diagnostic_categories"]
        )
        for row in candidate_diagnostics
    )
    q0_lengths = [
        {
            "candidate_index": int(row["candidate_index"]),
            "candidate_id": str(row["candidate_id"]),
            "raw_sha256": str(row["raw_sha256"]),
            "generated_token_count": int(row["generated_token_count"]),
            "character_count": len(str(row["raw_text"])),
            "utf8_byte_count": len(str(row["raw_text"]).encode("utf-8")),
        }
        for row in q0_rows
    ]
    shortest_q0_token_index = min(
        range(len(q0_rows)),
        key=lambda index: (
            q0_lengths[index]["generated_token_count"],
            q0_lengths[index]["candidate_index"],
        ),
    )
    shortest_q0_character_index = min(
        range(len(q0_rows)),
        key=lambda index: (
            q0_lengths[index]["character_count"],
            q0_lengths[index]["candidate_index"],
        ),
    )
    shortest_q0_tokens = q0_lengths[shortest_q0_token_index][
        "generated_token_count"
    ]
    shortest_q0_families = q0_family_sets[shortest_q0_token_index]
    q0_texts = [str(row["raw_text"]) for row in q0_rows]
    b_texts = [str(row["raw_text"]) for row in raw_rows]
    obvious_premises = _obvious_local_premise_identifiers(declaration)
    q0_premises = {
        premise
        for premise in obvious_premises
        if any(_contains_lean_identifier(text, premise) for text in q0_texts)
    }
    b_premises = {
        premise
        for premise in obvious_premises
        if any(_contains_lean_identifier(text, premise) for text in b_texts)
    }
    q0_qualified_identifiers = set().union(
        *(_qualified_lean_identifiers(text) for text in q0_texts)
    )
    b_qualified_identifiers = set().union(
        *(_qualified_lean_identifiers(text) for text in b_texts)
    )
    tags: list[str] = []
    if any(row["finish_reason"] == "token_limit" for row in raw_rows):
        tags.append("incomplete_or_token_limited")
    if unknown_reference_candidate_count:
        tags.append("hallucinated_lemma_or_api")
    if q0_families and b_families:
        overlap = len(q0_families & b_families) / len(q0_families)
        if overlap < 0.5:
            tags.append("different_proof_family_from_q0")
    if (
        shortest_q0_tokens <= 64
        and shortest_q0_families
        and not any(
            shortest_q0_families <= b_families
            for b_families in b_family_sets
        )
    ):
        tags.append("lost_simple_q0_strategy")
    if intuition_families:
        if intuition_families & b_families:
            tags.append("guidance_followed_but_formalization_failed")
        else:
            tags.append("guidance_seems_ignored")
    if not tags:
        tags.append("cannot_determine")
    features = {
        "q0_verified_candidate_lengths": q0_lengths,
        "shortest_q0_verified_by_generated_tokens": q0_lengths[
            shortest_q0_token_index
        ],
        "shortest_q0_verified_by_character_count": q0_lengths[
            shortest_q0_character_index
        ],
        "q0_tactic_families": sorted(q0_families),
        "q0_tactic_families_by_verified_candidate": [
            {
                "candidate_index": int(row["candidate_index"]),
                "raw_sha256": str(row["raw_sha256"]),
                "families": sorted(families),
            }
            for row, families in zip(q0_rows, q0_family_sets, strict=True)
        ],
        "b_tactic_families": sorted(b_families),
        "b_tactic_families_by_candidate": [
            {
                "candidate_index": int(row["candidate_index"]),
                "candidate_id": str(row["candidate_id"]),
                "raw_sha256": str(row["raw_sha256"]),
                "families": sorted(families),
            }
            for row, families in zip(raw_rows, b_family_sets, strict=True)
        ],
        "q0_b_shared_tactic_families": sorted(q0_families & b_families),
        "q0_only_tactic_families": sorted(q0_families - b_families),
        "b_only_tactic_families": sorted(b_families - q0_families),
        "intuition_explicit_tactic_families": sorted(intuition_families),
        "intuition_tactic_families_also_observed_in_b": sorted(
            intuition_families & b_families
        ),
        "intuition_tactic_families_not_observed_in_b": sorted(
            intuition_families - b_families
        ),
        "shortest_q0_verified_generated_tokens": shortest_q0_tokens,
        "shortest_q0_verified_character_count": q0_lengths[
            shortest_q0_character_index
        ]["character_count"],
        "shortest_q0_verified_tactic_families": sorted(shortest_q0_families),
        "q0_b_tactic_family_overlap_fraction": (
            len(q0_families & b_families) / len(q0_families)
            if q0_families
            else None
        ),
        "official_unknown_reference_candidate_count": (
            unknown_reference_candidate_count
        ),
        "obvious_local_premise_identifiers_in_declaration": sorted(
            obvious_premises
        ),
        "q0_obvious_local_premise_identifiers_observed": sorted(q0_premises),
        "b_obvious_local_premise_identifiers_observed": sorted(b_premises),
        "q0_b_shared_obvious_local_premise_identifiers": sorted(
            q0_premises & b_premises
        ),
        "q0_only_obvious_local_premise_identifiers": sorted(
            q0_premises - b_premises
        ),
        "q0_qualified_lean_identifiers": sorted(q0_qualified_identifiers),
        "q0_qualified_identifiers_also_observed_in_b": sorted(
            q0_qualified_identifiers & b_qualified_identifiers
        ),
        "q0_qualified_identifiers_not_observed_in_b": sorted(
            q0_qualified_identifiers - b_qualified_identifiers
        ),
        "q0_qualified_identifier_overlap_fraction": (
            len(q0_qualified_identifiers & b_qualified_identifiers)
            / len(q0_qualified_identifiers)
            if q0_qualified_identifiers
            else None
        ),
        "direct_text_comparison_scope": (
            "Exact lexical occurrence only; no semantic equivalence, route quality, "
            "architectural ownership, or causal attribution is inferred."
        ),
    }
    return tags, features, candidate_diagnostics


def _task_analysis(
    regressions: Sequence[Mapping[str, Any]],
    raw_b_rows: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    transformed_rows: Sequence[Mapping[str, Any]],
    tasks_by_id: Mapping[tuple[str, str], BoundTask],
    verification_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_b_by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    q0_by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    transformed_by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for rows, target in (
        (raw_b_rows, raw_b_by_task),
        (q0_rows, q0_by_task),
        (transformed_rows, transformed_by_task),
    ):
        for row in rows:
            target.setdefault((str(row["workload"]), str(row["task_id"])), []).append(
                row
            )
    analyses: list[dict[str, Any]] = []
    for task in regressions:
        key = (str(task["workload_id"]), str(task["task_id"]))
        bound = tasks_by_id[key]
        raw_rows = raw_b_by_task[key]
        q0_task_rows = q0_by_task[key]
        variants = transformed_by_task.get(key, [])
        recoveries = [
            row for row in variants if row["diagnostic_verifier_classification"] == "verified"
        ]
        inconclusive = [
            row
            for row in variants
            if row["diagnostic_verifier_classification"]
            in {"verifier_timeout", "verifier_error"}
        ]
        if recoveries:
            classification = "FORMAT_ONLY"
        elif inconclusive:
            classification = "INCONCLUSIVE"
        else:
            classification = "CONTENT_OR_SEARCH"
        official_results = [
            verification_by_id[str(candidate_id)]
            for candidate_id in task["candidate_slots"]["B"]
        ]
        computed_tags, features, candidate_diagnostics = _diagnostic_tags(
            raw_rows,
            official_results,
            q0_task_rows,
            bound.intuition_text,
            bound.task.declaration,
        )
        tags = computed_tags if classification == "CONTENT_OR_SEARCH" else []
        diagnostic_category_counts = Counter(
            category
            for row in candidate_diagnostics
            for category in row["diagnostic_categories"]
        )
        category_counts = Counter(str(row["official_verifier_classification"]) for row in raw_rows)
        finish_counts = Counter(str(row["finish_reason"]) for row in raw_rows)
        if classification == "FORMAT_ONLY":
            note = (
                f"{len(recoveries)} of 8 raw B candidates become Lean-valid only after "
                "a recorded superficial wrapper transform; official raw classifications "
                "remain unchanged."
            )
        elif classification == "INCONCLUSIVE":
            note = (
                f"No transformed candidate verified, but {len(inconclusive)} strict "
                "mechanical variant result is infrastructurally inconclusive."
            )
        else:
            observations: list[str] = []
            if "incomplete_or_token_limited" in tags:
                observations.append(
                    f"{finish_counts['token_limit']}/8 candidates hit the token limit"
                )
            if "hallucinated_lemma_or_api" in tags:
                observations.append(
                    f"{features['official_unknown_reference_candidate_count']}/8 "
                    "official diagnostics report an unknown identifier/API"
                )
            if "different_proof_family_from_q0" in tags:
                observations.append(
                    "the union of B tactic families covers only "
                    f"{features['q0_b_tactic_family_overlap_fraction']:.0%} of the "
                    "verified-Q0 families"
                )
            if "lost_simple_q0_strategy" in tags:
                observations.append(
                    f"the shortest Q0 proof uses {features['shortest_q0_verified_generated_tokens']} "
                    "tokens and its complete tactic-family set is absent from every B "
                    "candidate"
                )
            if "guidance_followed_but_formalization_failed" in tags:
                observations.append(
                    "B attempts the tactic family named explicitly by the intuition, "
                    "but every attempt is Lean-rejected"
                )
            if "guidance_seems_ignored" in tags:
                observations.append(
                    "the tactic family named explicitly by the intuition is absent from B"
                )
            if "cannot_determine" in tags:
                observations.append(
                    "no narrower deterministic observable tag is supportable"
                )
            note = (
                f"All 8 raw B candidates and all {len(variants)} applicable strict "
                "wrapper variants fail under the frozen verifier; "
                + "; ".join(observations)
                + "."
            )
        analyses.append(
            {
                "ordinal": task["ordinal"],
                "workload": key[0],
                "task_id": key[1],
                "declaration_name": task["declaration_name"],
                "public_context": bound.task.preamble,
                "public_context_sha256": task["public_context_sha256"],
                "declaration": bound.task.declaration,
                "declaration_sha256": task["declaration_sha256"],
                "frozen_mathia_intuition": bound.intuition_text,
                "frozen_mathia_intuition_sha256": bound.intuition_sha256,
                "q0_verified_candidate_count": len(q0_task_rows),
                "q0_verified_candidate_raw_sha256": [
                    row["raw_sha256"] for row in q0_task_rows
                ],
                "b_candidate_result_distribution": dict(sorted(category_counts.items())),
                "b_finish_reason_distribution": dict(sorted(finish_counts.items())),
                "b_token_limit_candidate_count": finish_counts["token_limit"],
                "b_official_diagnostic_category_counts": dict(
                    sorted(diagnostic_category_counts.items())
                ),
                "b_candidate_diagnostics": candidate_diagnostics,
                "b_candidate_markers": [
                    {
                        "candidate_index": row["candidate_index"],
                        "candidate_id": row["candidate_id"],
                        "raw_sha256": row["raw_sha256"],
                        **candidate_format_markers(str(row["raw_text"])),
                    }
                    for row in raw_rows
                ],
                "mechanical_variants_tested": len(variants),
                "recoveries": [
                    {
                        key: row[key]
                        for key in (
                            "candidate_index",
                            "candidate_id",
                            "source_raw_sha256",
                            "transform_sequence",
                            "transformed_sha256",
                            "diagnostic_verifier_classification",
                            "diagnostic_lean_exit_code",
                            "verifier_environment_sha256",
                        )
                    }
                    for row in recoveries
                ],
                "primary_classification": classification,
                "diagnostic_tags": tags,
                "observable_comparison_features": features,
                "note": note,
            }
        )
    return analyses


def _aggregate(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classification_counts = Counter(
        str(row["primary_classification"]) for row in task_rows
    )
    recoveries = [recovery for row in task_rows for recovery in row["recoveries"]]
    recoverable_candidates = {
        (str(row["workload"]), str(row["task_id"]), str(recovery["candidate_id"]))
        for row in task_rows
        for recovery in row["recoveries"]
    }
    format_types = Counter(
        transform
        for recovery in recoveries
        for transform in recovery["transform_sequence"]
    )
    tag_counts = Counter(
        tag
        for row in task_rows
        if row["primary_classification"] == "CONTENT_OR_SEARCH"
        for tag in row["diagnostic_tags"]
    )
    total = len(task_rows)
    format_only = classification_counts["FORMAT_ONLY"]
    content = classification_counts["CONTENT_OR_SEARCH"]
    diagnostic_candidate_counts = Counter(
        category
        for row in task_rows
        for candidate in row["b_candidate_diagnostics"]
        for category in candidate["diagnostic_categories"]
    )
    diagnostic_task_counts = Counter(
        category
        for row in task_rows
        for category in {
            category
            for candidate in row["b_candidate_diagnostics"]
            for category in candidate["diagnostic_categories"]
        }
    )
    unknown_reference_kind_counts = Counter(
        reference["kind"]
        for row in task_rows
        for candidate in row["b_candidate_diagnostics"]
        for reference in candidate["unknown_references"]
    )
    format_marker_candidate_counts = Counter(
        marker
        for row in task_rows
        for candidate in row["b_candidate_markers"]
        for marker in (
            "duplicated_by",
            "theorem_repetition",
            "markdown_fence",
            "lean3_begin",
            "natural_language_contamination",
            "sorry_or_admit",
        )
        if candidate[marker]
    )
    lost_simple_examples = [
        str(row["task_id"])
        for row in task_rows
        if "lost_simple_q0_strategy" in row["diagnostic_tags"]
    ]
    translation_examples = [
        str(row["task_id"])
        for row in task_rows
        if "guidance_followed_but_formalization_failed" in row["diagnostic_tags"]
    ]
    return {
        "total_regressions": total,
        "workload_counts": dict(sorted(Counter(str(row["workload"]) for row in task_rows).items())),
        "classification_counts": {
            key: classification_counts[key]
            for key in ("FORMAT_ONLY", "CONTENT_OR_SEARCH", "INCONCLUSIVE")
        },
        "classification_fractions": {
            key: classification_counts[key] / total
            for key in ("FORMAT_ONLY", "CONTENT_OR_SEARCH", "INCONCLUSIVE")
        },
        "tasks_with_mechanically_recoverable_b_proof": sum(
            bool(row["recoveries"]) for row in task_rows
        ),
        "mechanically_recoverable_candidate_proofs": len(recoverable_candidates),
        "verified_mechanical_variants": len(recoveries),
        "format_error_type_counts_among_recoveries": dict(sorted(format_types.items())),
        "content_or_search_diagnostic_tag_counts": dict(sorted(tag_counts.items())),
        "observed_diagnostic_facts": {
            "final_output_formatting_recovery_lower_bound": {
                "task_count": format_only,
                "task_fraction": format_only / total,
                "unique_candidate_proof_count": len(recoverable_candidates),
                "verified_mechanical_variant_count": len(recoveries),
            },
            "content_or_search_after_mechanical_cleanup": {
                "task_count": content,
                "task_fraction": content / total,
            },
            "inconclusive_task_count": classification_counts["INCONCLUSIVE"],
            "arm_b_candidate_count": total * EXPECTED_CANDIDATES_PER_TASK,
            "arm_b_token_limit_candidate_count": sum(
                int(row["b_token_limit_candidate_count"]) for row in task_rows
            ),
            "tasks_with_arm_b_token_limit": sum(
                int(row["b_token_limit_candidate_count"]) > 0 for row in task_rows
            ),
            "format_marker_candidate_counts": dict(
                sorted(format_marker_candidate_counts.items())
            ),
            "official_diagnostic_category_candidate_counts": dict(
                sorted(diagnostic_candidate_counts.items())
            ),
            "official_diagnostic_category_task_counts": dict(
                sorted(diagnostic_task_counts.items())
            ),
            "unknown_reference_kind_occurrence_counts": dict(
                sorted(unknown_reference_kind_counts.items())
            ),
            "lost_simple_q0_strategy_examples": lost_simple_examples,
            "intuition_tactic_family_also_observed_in_b_examples": (
                translation_examples
            ),
            "interpretation_boundary": (
                "Architectural ownership, causal attribution, and future-training "
                "recommendations are intentionally deferred to ChatGPT/user review."
            ),
        },
    }


def render_regression_analysis(analysis: Mapping[str, Any]) -> str:
    aggregate = analysis["aggregate"]
    counts = aggregate["classification_counts"]
    facts = aggregate["observed_diagnostic_facts"]
    bindings = analysis["source_bindings"]
    structural = analysis["structural_transition_evidence"]
    unknown = analysis["unknown_reference_evidence"]
    stopping = analysis["stopping_control_evidence"]
    lines = [
        "# Q0-pass / Arm-B-fail regression retrospective",
        "",
        "**OBSERVED, scoring-excluded:** This diagnostic reuses frozen #78 Q0 and "
        "#86 Arm-B candidates. It does not regenerate, repair for scoring, or modify "
        "any official classification or metric.",
        "",
        f"The exact regression set contains {aggregate['total_regressions']} tasks: "
        f"{aggregate['workload_counts']['minif2f-valid-clean-v2']} MiniF2F and "
        f"{aggregate['workload_counts']['fresh-composition-valid-v2']} fresh composition.",
        "",
        "| primary classification | tasks | fraction |",
        "| --- | ---: | ---: |",
    ]
    for classification in ("FORMAT_ONLY", "CONTENT_OR_SEARCH", "INCONCLUSIVE"):
        lines.append(
            f"| {classification} | {counts[classification]} | "
            f"{aggregate['classification_fractions'][classification]:.1%} |"
        )
    lines.extend(
        [
            "",
            "Mechanically recoverable: "
            f"{aggregate['tasks_with_mechanically_recoverable_b_proof']} tasks and "
            f"{aggregate['mechanically_recoverable_candidate_proofs']} unique candidate "
            f"proofs ({aggregate['verified_mechanical_variants']} verified transform "
            "variants).",
            "",
            "## Method and provenance",
            "",
            "The regression membership is reconstructed from the frozen manifest and "
            "all official Arm-B verification records. Only the six declared superficial "
            "wrapper removals (and deterministic compositions of at most four removals) "
            "are tested. A task is `FORMAT_ONLY` only when Lean accepts at least one "
            "transformed candidate in the exact workload environment.",
            "",
            f"- execution manifest SHA-256: `{bindings['manifest_sha256']}`",
            f"- #86 verification result-set SHA-256: "
            f"`{bindings['issue_86_verification_result_set_sha256']}`",
            f"- Q0 compact evidence SHA-256: `{bindings['q0_evidence_sha256']}`",
            f"- Q0 recovery archive SHA-256: "
            f"`{bindings['q0_raw_recovery_archive_sha256']}`",
            f"- verifier environment-set SHA-256: "
            f"`{bindings['verifier_environment_set_sha256']}`",
            "",
            "## Per-task evidence",
            "",
            "| workload | task | Q0 verified | B finish eos/token limit | variants | classification | diagnostic categories | tags |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in analysis["tasks"]:
        finish = row["b_finish_reason_distribution"]
        lines.append(
            f"| {row['workload']} | `{row['task_id']}` | "
            f"{row['q0_verified_candidate_count']} | {finish.get('eos', 0)}/"
            f"{finish.get('token_limit', 0)} | {row['mechanical_variants_tested']} | "
            f"{row['primary_classification']} | "
            f"{', '.join(row['b_official_diagnostic_category_counts']) or '—'} | "
            f"{', '.join(row['diagnostic_tags']) or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Observed diagnostic facts",
            "",
            "- Final-output formatting lower bound: "
            f"{facts['final_output_formatting_recovery_lower_bound']['task_count']}/"
            f"{aggregate['total_regressions']} tasks and "
            f"{facts['final_output_formatting_recovery_lower_bound']['unique_candidate_proof_count']} "
            "candidate proofs are mechanically recoverable.",
            "- Not recovered by the frozen mechanical transform set: "
            f"{facts['content_or_search_after_mechanical_cleanup']['task_count']}/"
            f"{aggregate['total_regressions']} tasks; "
            f"{facts['inconclusive_task_count']} tasks are inconclusive.",
            "- Token-limit observations: "
            f"{facts['arm_b_token_limit_candidate_count']}/"
            f"{facts['arm_b_candidate_count']} B candidates across "
            f"{facts['tasks_with_arm_b_token_limit']} tasks.",
            "- Official diagnostic categories by candidate: "
            + (", ".join(
                f"`{category}`={count}"
                for category, count in facts[
                    "official_diagnostic_category_candidate_counts"
                ].items()
            ) or "none detected by the declared rules")
            + ".",
            "- Exact output-format markers by candidate: "
            + (", ".join(
                f"`{marker}`={count}"
                for marker, count in facts["format_marker_candidate_counts"].items()
            ) or "none")
            + ".",
            "- Observable short-Q0/B tactic-family divergences: "
            + (", ".join(
                f"`{task_id}`"
                for task_id in facts[
                    "lost_simple_q0_strategy_examples"
                ]
            ) or "none under the deterministic tag rule")
            + ".",
            "- Tasks where an explicit intuition tactic family also appears in B: "
            + (", ".join(
                f"`{task_id}`"
                for task_id in facts[
                    "intuition_tactic_family_also_observed_in_b_examples"
                ]
            ) or "none under the deterministic tag rule")
            + ".",
            "- " + facts["interpretation_boundary"],
            "",
            "The Q0-pass/B-fail selection compares two stochastic n=8 samples from "
            "different prompt-conditioned distributions. FORMAT_ONLY is directly "
            "confirmed by Lean; CONTENT_OR_SEARCH establishes an observable distribution "
            "change after strict wrapper transforms, not causal harm from the intuition. "
            "No architectural owner or future training method is assigned here.",
            "",
            "## Permanent raw evidence boundary",
            "",
            "`q0-b-regressions/raw-b-candidates.jsonl` contains exactly 184 untouched "
            "Arm-B continuations. `q0-b-regressions/q0-verified-candidates.jsonl` "
            f"contains all {analysis['committed_evidence']['q0_verified_candidates']['row_count']} "
            "authoritative verified Q0 continuations for the same tasks. "
            "All transformed candidates are separate in "
            "`q0-b-regressions/transformed-b-candidates.jsonl` and reference their "
            "source raw SHA-256.",
            "",
            "A fresh checkout can audit the committed subset with "
            "`uv run pytest -q tests/test_mathia_prompt_ab_regressions.py`. Recomputing "
            "the Lean diagnostic additionally requires the frozen #86 artifact root, "
            "both frozen Lean projects, and the hash-matching Q0 recovery archive; the "
            "complete CLI is available through "
            "`python -m qwen_lean.mathia_prompt_ab_regressions --help`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Structural Q0/B transitions",
            "",
            f"The frozen structural field covers "
            f"{structural['coverage']['classified_task_count']}/"
            f"{structural['population_task_count']} matched tasks. The remaining "
            f"{structural['coverage']['unavailable_task_count']} MiniF2F tasks have a "
            "null structural class; no class is inferred for them. `multi-step` is not "
            "present as a pre-existing class in the frozen manifest.",
            "",
            "| structural class | coverage | both | Q0 only | B only | neither |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for structural_class in ("direct", "multi-step", "branching", "deep"):
        row = structural["aggregate_by_structural_class"][structural_class]
        transition_counts = row["transition_counts"]
        if transition_counts is None:
            lines.append(
                f"| {structural_class} | unavailable | — | — | — | — |"
            )
        else:
            lines.append(
                f"| {structural_class} | {row['task_count']} | "
                f"{transition_counts['both']} | {transition_counts['q0_only']} | "
                f"{transition_counts['b_only']} | {transition_counts['neither']} |"
            )
    unavailable = structural["aggregate_by_structural_class"]["unavailable"]
    unavailable_counts = unavailable["transition_counts"]
    lines.extend(
        [
            f"| unavailable (MiniF2F) | {unavailable['task_count']} | "
            f"{unavailable_counts['both']} | {unavailable_counts['q0_only']} | "
            f"{unavailable_counts['b_only']} | {unavailable_counts['neither']} |",
            "",
            "These are paired descriptive counts only; the report does not value one "
            "structural class over another or infer causal guidance effects.",
            "",
            "## Unknown-reference call sites and single-node oracle",
            "",
            f"Official diagnostics contain "
            f"{unknown['official_unknown_reference_occurrence_count']} unknown-reference "
            f"occurrences across {unknown['official_unknown_reference_candidate_count']} "
            "Arm-B candidates. The first call site is reconstructed for "
            f"{unknown['first_call_site_reconstructed_count']} candidates and the prefix "
            f"replays through the captured state for "
            f"{unknown['prefix_replay_accepted_count']} candidates.",
            "",
            "| single-node oracle outcome | candidates |",
            "| --- | ---: |",
        ]
    )
    for outcome in _ORACLE_OUTCOMES:
        lines.append(
            f"| {outcome} | {unknown['oracle_outcome_counts'][outcome]} |"
        )
    lines.extend(
        [
            "",
            "Anti-vacuity flags: "
            + ", ".join(
                f"`{flag}`={unknown['anti_vacuity_flag_counts'][flag]}"
                for flag in _ANTI_VACUITY_FLAGS
            )
            + ".",
            "",
            "Bounded mechanical reference evidence classes: "
            + ", ".join(
                f"`{classification}`={count}"
                for classification, count in unknown[
                    "mechanical_reference_evidence_class_counts"
                ].items()
            )
            + ". Only `exact_bounded_declaration_match` denotes a frozen-environment "
            "name match; candidate obligations are syntax/state extractions. Neither "
            "is a semantic API attribution.",
            "",
            "Candidate-node extraction status: "
            + ", ".join(
                f"`{status}`={count}"
                for status, count in unknown["candidate_node_status_counts"].items()
            )
            + ". The extracted objects preserve only the exact goal/context before a "
            "mechanically identified proof-producing call; node quality remains "
            "undetermined.",
            "",
            "Every intervention is scoring-excluded, replaces at most the first "
            "unknown expression with `(by sorry)`, and never supplies a second unknown. "
            "The evidence records exact prefix/state/type bindings and does not decide "
            "whether a missing node is useful or which architecture component owns it.",
            "",
            "## Stopping and control evidence",
            "",
            f"`No goals to be solved` occurs in "
            f"{stopping['no_goals_to_be_solved']['official_candidate_count']} official "
            "candidate diagnostics. Ordered raw line-prefix hashes are retained for a "
            "future authorized prefix-recovery study, but none is marked recovered here. "
            f"Token limits remain separate: {stopping['token_limit']['candidate_count']} "
            f"candidates across {stopping['token_limit']['task_count']} tasks.",
            "",
            "## Availability and interpretation boundary",
            "",
            "- MiniF2F structural classes: unavailable in the frozen manifest.",
            "- `multi-step` structural class: unavailable in the frozen manifest.",
            "- Call sites whose line-boundary prefix or contextual type cannot be "
            "reconstructed are recorded as `oracle_not_testable`.",
            "- Closed-goal history before a call is not exposed by the frozen tactic-state "
            "snapshot and is recorded as unavailable rather than estimated.",
            "- Architectural, causal, node-quality, and training interpretation remains "
            "deferred to ChatGPT/user review.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_regression_analysis(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
    manifest_path: Path,
    artifact_root: Path,
    q0_root: Path,
    lean_project_roots: Mapping[str, Path],
    raw_b_output: Path,
    q0_verified_output: Path,
    transformed_output: Path,
    oracle_output: Path,
    analysis_output: Path,
    readme_output: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    manifest, bound_tasks = validate_execution_manifest(
        config, dataset_root, mathia_root, repository_root, manifest_path
    )
    manifest_sha256 = sha256_file(manifest_path)
    generation_inventory = inventory_generations(
        manifest, artifact_root, manifest_sha256
    )
    if generation_inventory["completed_candidate_count"] != EXPECTED_CANDIDATES_TOTAL:
        raise ValueError("#86 generation inventory is incomplete")
    environment_bundle = verifier_environment_identities(config, lean_project_roots)
    verification_inventory = inventory_verifications(
        manifest,
        artifact_root,
        manifest_sha256,
        environment_bundle["environment_sha256_by_workload"],
        generation_inventory,
    )
    if verification_inventory["completed_verification_count"] != EXPECTED_CANDIDATES_TOTAL:
        raise ValueError("#86 verification inventory is incomplete")
    structural_transitions = structural_transition_evidence(
        manifest, verification_inventory["results_by_id"]
    )
    regressions = reconstruct_regression_tasks(
        manifest, verification_inventory["results_by_id"]
    )
    counts = Counter(str(task["workload_id"]) for task in regressions)
    if len(regressions) != EXPECTED_REGRESSION_TASKS or counts != Counter(
        EXPECTED_REGRESSION_COUNTS
    ):
        raise ValueError(
            f"Q0-pass/B-fail regression population differs: {len(regressions)} {counts}"
        )

    q0_evidence, q0_evidence_bytes = _load_q0_reference(config, repository_root)
    q0_evidence_sha256 = hashlib.sha256(q0_evidence_bytes).hexdigest()
    q0_candidates, q0_source_hashes = _load_q0_candidates(q0_root, q0_evidence)
    raw_b_rows = _raw_b_rows(
        regressions,
        generation_inventory,
        verification_inventory,
        manifest_sha256,
    )
    q0_rows = _raw_q0_rows(
        regressions,
        q0_candidates,
        q0_evidence_sha256,
        q0_source_hashes,
        str(config.model["model_revision"]),
    )
    raw_b_sha256 = _write_jsonl(raw_b_output, raw_b_rows)
    q0_verified_sha256 = _write_jsonl(q0_verified_output, q0_rows)

    tasks_by_id = {
        (bound.workload_id, bound.task.id): bound for bound in bound_tasks
    }
    variant_jobs = _build_variant_jobs(
        regressions, raw_b_rows, tasks_by_id
    )
    outcomes = _verify_variants(
        variant_jobs,
        tasks_by_id,
        config,
        lean_project_roots,
        workers=workers,
    )
    transformed_rows = []
    for job in variant_jobs:
        outcome = outcomes[str(job["variant_id"])]
        transformed_rows.append(
            {
                "schema_version": TRANSFORMED_SCHEMA_VERSION,
                **job,
                "diagnostic_verifier_classification": outcome.category,
                "diagnostic_lean_exit_code": outcome.lean_exit_code,
                "diagnostic_diagnostics": outcome.diagnostics,
                "verifier_environment_sha256": environment_bundle[
                    "environment_sha256_by_workload"
                ][str(job["workload"])],
                "scoring_excluded": True,
            }
        )
    transformed_sha256 = _write_jsonl(transformed_output, transformed_rows)
    task_rows = _task_analysis(
        regressions,
        raw_b_rows,
        q0_rows,
        transformed_rows,
        tasks_by_id,
        verification_inventory["results_by_id"],
    )
    unknown_reference_evidence, oracle_rows = _build_unknown_reference_evidence(
        regressions,
        raw_b_rows,
        tasks_by_id,
        verification_inventory["results_by_id"],
        config,
        lean_project_roots,
        environment_bundle["environment_sha256_by_workload"],
        workers=workers,
    )
    oracle_sha256 = _write_jsonl(oracle_output, oracle_rows)
    stopping_control_evidence = _stopping_control_evidence(task_rows, raw_b_rows)
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "decision_marker": "OBSERVED",
        "scoring_excluded": True,
        "official_86_results_modified": False,
        "model_inference_or_regeneration_performed": False,
        "sampling_caveat": (
            "Q0-pass/B-fail compares stochastic n=8 samples from different "
            "prompt-conditioned distributions. It identifies confirmed format failures "
            "and observable search/formalization changes, not causal harm from guidance."
        ),
        "source_bindings": {
            "manifest_sha256": manifest_sha256,
            "issue_86_results_sha256": sha256_file(
                repository_root / "evidence/mathia-prompt-ab/results.json"
            ),
            "issue_86_format_diagnostic_sha256": sha256_file(
                repository_root
                / "evidence/mathia-prompt-ab/format-contamination-diagnostic.json"
            ),
            "issue_86_verification_result_set_sha256": verification_inventory[
                "result_set_sha256"
            ],
            "q0_evidence_sha256": q0_evidence_sha256,
            "q0_raw_recovery_archive_sha256": RECOVERY_ARCHIVE_SHA256,
            "q0_workloads": q0_source_hashes,
            "verifier_environment_set_sha256": environment_bundle[
                "environment_set_sha256"
            ],
            "verifier_environment_sha256_by_workload": environment_bundle[
                "environment_sha256_by_workload"
            ],
            "diagnostic_prelude_sha256": _sha256_text(_DIAGNOSTIC_PRELUDE),
        },
        "committed_evidence": {
            "raw_b_candidates": {
                "path": str(
                    raw_b_output.resolve().relative_to(repository_root.resolve())
                ),
                "row_count": len(raw_b_rows),
                "sha256": raw_b_sha256,
            },
            "q0_verified_candidates": {
                "path": str(
                    q0_verified_output.resolve().relative_to(repository_root.resolve())
                ),
                "row_count": len(q0_rows),
                "sha256": q0_verified_sha256,
            },
            "transformed_b_candidates": {
                "path": str(
                    transformed_output.resolve().relative_to(repository_root.resolve())
                ),
                "row_count": len(transformed_rows),
                "sha256": transformed_sha256,
            },
            "single_node_oracle_candidates": {
                "path": str(
                    oracle_output.resolve().relative_to(repository_root.resolve())
                ),
                "row_count": len(oracle_rows),
                "sha256": oracle_sha256,
            },
        },
        "mechanical_transform_contract": {
            "maximum_composed_transforms": 4,
            "transforms": [
                "strip_leading_duplicated_by",
                "unwrap_markdown_fence",
                "remove_exact_repeated_theorem_declaration",
                "unwrap_lean3_begin_end",
                "remove_whitelisted_natural_language_prefix",
                "remove_whitelisted_natural_language_suffix",
            ],
            "semantic_repair_permitted": False,
        },
        "diagnostic_tag_rules": {
            "incomplete_or_token_limited": "at least one B candidate hit token_limit",
            "hallucinated_lemma_or_api": "official Lean diagnostics report an unknown identifier/constant/declaration",
            "different_proof_family_from_q0": "B covers under half of the tactic families observed in verified Q0 candidates",
            "lost_simple_q0_strategy": "the shortest verified Q0 proof uses at most 64 generated tokens and no B candidate contains all of its tactic families",
            "guidance_followed_but_formalization_failed": "an explicit tactic family named in the intuition also occurs in B",
            "guidance_seems_ignored": "the intuition names a tactic family absent from B",
            "cannot_determine": "none of the deterministic observable rules above applies",
        },
        "official_diagnostic_category_rules": {
            "unknown_identifier": "official diagnostic contains the exact phrase Unknown identifier",
            "unknown_constant": "official diagnostic contains the exact phrase Unknown constant",
            "unknown_declaration": "official diagnostic contains the exact phrase Unknown declaration",
            "unsolved_goals": "official diagnostic contains unsolved goal or unsolved goals",
            "no_goals_to_be_solved": "official diagnostic contains the exact phrase No goals to be solved",
            "type_mismatch": "official diagnostic explicitly reports type mismatch or application type mismatch",
            "elaboration_error": "official diagnostic explicitly reports failed/cannot synthesize, invalid field, function expected, or failed to infer",
            "syntax_error": "official diagnostic explicitly reports an unexpected token/end/identifier, unterminated input, or invalid end",
            "tactic_failure": "official diagnostic explicitly reports unknown tactic, tactic failure, linarith failure, no progress, or failure to close a goal",
            "sorry_or_admit_rejected": "official diagnostic explicitly says the declaration uses sorry or admit",
        },
        "observable_text_comparison_rules": {
            "tactic_families": (
                "exact lexical occurrence of a closed, declared tactic-family vocabulary"
            ),
            "obvious_local_premises": (
                "conventional declaration binder identifiers h, h+number, or h+one-letter "
                "suffix, compared by exact identifier occurrence in Q0 and B text"
            ),
            "qualified_lean_identifiers": (
                "qualified dotted identifiers observed in verified Q0 text, compared "
                "by exact lexical occurrence in B text"
            ),
            "semantic_equivalence_or_ownership_inferred": False,
        },
        "structural_transition_evidence": structural_transitions,
        "unknown_reference_evidence": unknown_reference_evidence,
        "stopping_control_evidence": stopping_control_evidence,
        "aggregate": _aggregate(task_rows),
        "tasks": task_rows,
    }
    _atomic_write(analysis_output, _canonical_json_bytes(analysis, pretty=True))
    _atomic_write(readme_output, render_regression_analysis(analysis).encode("utf-8"))
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scoring-excluded Q0-pass/Arm-B-fail retrospective for issue #93"
    )
    parser.add_argument("--config", type=Path, default=Path("config/mathia-prompt-ab.json"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mathia-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--q0-root", type=Path, required=True)
    parser.add_argument("--minif2f-project-root", type=Path, required=True)
    parser.add_argument("--fresh-project-root", type=Path, required=True)
    parser.add_argument("--raw-b-output", type=Path, required=True)
    parser.add_argument("--q0-verified-output", type=Path, required=True)
    parser.add_argument("--transformed-output", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--readme-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = run_regression_analysis(
        PromptABConfig.load(args.config),
        args.dataset_root,
        args.mathia_root,
        args.repository_root.resolve(),
        args.manifest,
        args.artifacts,
        args.q0_root,
        {
            "minif2f-valid-clean-v2": args.minif2f_project_root,
            "fresh-composition-valid-v2": args.fresh_project_root,
        },
        args.raw_b_output,
        args.q0_verified_output,
        args.transformed_output,
        args.oracle_output,
        args.analysis_output,
        args.readme_output,
        workers=args.workers,
    )
    print(json.dumps(analysis["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
