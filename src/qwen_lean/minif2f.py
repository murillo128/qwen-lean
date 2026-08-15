from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import TaskRecord


PHASE1_CONFIG_SCHEMA_VERSION = "phase1-config-v1"
_THEOREM_PATTERN = re.compile(
    r"(?ms)^theorem\s+(?P<name>[^\s(:]+)(?P<tail>.*?)\s*:=\s*by\n\s+sorry\s*(?=\n|\Z)"
)


@dataclass(frozen=True)
class Phase1Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Phase1Config:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != PHASE1_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unknown Phase 1 config schema: {value.get('schema_version')}")
        return cls(path=path.resolve(), value=value)

    @property
    def benchmark(self) -> dict[str, Any]:
        return self.value["benchmark"]

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.value["sampling"]

    @property
    def engine(self) -> dict[str, Any]:
        return self.value["engine"]

    def select_workload(
        self, workload_id: str, tasks: list[TaskRecord]
    ) -> list[TaskRecord]:
        try:
            workload = self.value["workloads"][workload_id]
        except KeyError as error:
            raise ValueError(f"unknown workload id: {workload_id}") from error

        tasks_by_id = {task.id: task for task in tasks}
        if workload["selection"] == "all":
            selected = tasks
        elif workload["selection"] == "explicit_ids":
            ids = [str(task_id) for task_id in workload["task_ids"]]
            missing = sorted(set(ids) - tasks_by_id.keys())
            if missing:
                raise ValueError(f"workload {workload_id} has unknown task ids: {missing}")
            selected = [tasks_by_id[task_id] for task_id in ids]
        else:
            raise ValueError(
                f"unknown selection for workload {workload_id}: {workload['selection']}"
            )

        expected = int(workload["expected_task_count"])
        if len(selected) != expected:
            raise ValueError(
                f"workload {workload_id} expected {expected} tasks, got {len(selected)}"
            )
        return selected


def validate_benchmark_checkout(config: Phase1Config, benchmark_root: Path) -> str:
    root = benchmark_root.resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"not a readable git checkout: {root}: {completed.stderr.strip()}")
    actual_revision = completed.stdout.strip()
    expected_revision = str(config.benchmark["revision"])
    if actual_revision != expected_revision:
        raise ValueError(
            f"miniF2F revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    return actual_revision


def materialize_validation_tasks(
    config: Phase1Config, benchmark_root: Path
) -> list[TaskRecord]:
    validate_benchmark_checkout(config, benchmark_root)
    source_path = benchmark_root.resolve() / str(config.benchmark["source_path"])
    source = source_path.read_text(encoding="utf-8")
    tasks = materialize_validation_source(
        source,
        expected_primary_task_count=int(config.benchmark["expected_primary_task_count"]),
    )
    manifest_path = config.path.parent / str(config.benchmark["primary_task_manifest"])
    expected_ids = [
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    actual_ids = [task.id for task in tasks]
    if actual_ids != expected_ids:
        raise ValueError("materialized miniF2F primary task IDs differ from pinned manifest")
    return tasks


def materialize_validation_source(
    source: str, *, expected_primary_task_count: int
) -> list[TaskRecord]:
    preamble = _extract_preamble(source)

    tasks: list[TaskRecord] = []
    declaration_names: set[str] = set()
    for match in _THEOREM_PATTERN.finditer(source):
        name = match.group("name")
        if ".variants." in name:
            continue
        declaration = f"theorem {name}{match.group('tail')}".rstrip()
        if name in declaration_names:
            raise ValueError(f"duplicate primary theorem declaration: {name}")
        declaration_names.add(name)
        tasks.append(
            TaskRecord(
                id=name,
                preamble=preamble,
                declaration=declaration,
                declaration_name=name,
            )
        )

    if len(tasks) != expected_primary_task_count:
        raise ValueError(
            "miniF2F validation contract expected "
            f"{expected_primary_task_count} primary tasks, got {len(tasks)}"
        )
    return tasks


def verifier_environment_metadata(
    config: Phase1Config, benchmark_root: Path
) -> dict[str, Any]:
    root = benchmark_root.resolve()
    validate_benchmark_checkout(config, root)
    toolchain = (root / "lean-toolchain").read_text(encoding="utf-8").strip()
    manifest = json.loads((root / "lake-manifest.json").read_text(encoding="utf-8"))
    manifest_dependencies = {
        package["name"]: package["rev"]
        for package in manifest["packages"]
        if package["name"] in {"formal_conjectures", "mathlib"}
    }
    dependencies: dict[str, str] = {}
    for name, expected_revision in manifest_dependencies.items():
        dependency_root = root / ".lake" / "packages" / name
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=dependency_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"missing or unreadable miniF2F dependency {name}: "
                f"{completed.stderr.strip()}"
            )
        actual_revision = completed.stdout.strip()
        if actual_revision != expected_revision:
            raise ValueError(
                f"miniF2F dependency {name} revision mismatch: "
                f"expected {expected_revision}, got {actual_revision}"
            )
        dependencies[name] = actual_revision
    expected_toolchain = str(config.benchmark["lean_toolchain"])
    if toolchain != expected_toolchain:
        raise ValueError(
            f"miniF2F Lean toolchain mismatch: expected {expected_toolchain}, got {toolchain}"
        )
    return {
        "project": str(config.benchmark["repository"]),
        "project_revision": str(config.benchmark["revision"]),
        "lean_toolchain": toolchain,
        "dependencies": dependencies,
    }


def _extract_preamble(source: str) -> str:
    lines = source.splitlines()
    try:
        first_import = next(index for index, line in enumerate(lines) if line.startswith("import "))
    except StopIteration as error:
        raise ValueError("miniF2F validation source has no import preamble") from error

    preamble_lines: list[str] = []
    for line in lines[first_import:]:
        if line.startswith("/--") or line.startswith("theorem "):
            break
        preamble_lines.append(line)
    preamble = "\n".join(preamble_lines).rstrip()
    if not preamble:
        raise ValueError("miniF2F validation source has an empty preamble")
    return preamble
