"""Exact-token counterfactual forks of frozen native-thinking trajectories."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .baseline import _GpuMemoryMonitor
from .counterfactual_forking_handoff import HANDOFF_SCHEMA
from .native_thinking_assessment import (
    GENERATION_RECORD_SCHEMA,
    MATHIA_FREEZE_ID,
    MODEL_ID,
    MODEL_REVISION,
    VLLM_VERSION,
    WORKLOADS,
    MathiaTask,
    NativeThinkingConfig,
    _append_jsonl,
    _atomic_write_json,
    _distribution,
    _file_sha256,
    _finish_reason,
    _load_tokenizer,
    _package_versions,
    _resolve_model_snapshot,
    _sha256_json,
    _sha256_text,
    generation_config_sha256,
    load_mathia_tasks,
    render_user_message,
    validate_lean_environments,
)
from .verifier import LeanVerifier, VerificationOutcome

CONFIG_SCHEMA = "qwen35-counterfactual-forking-config-v1"
FORK_GENERATION_SCHEMA = "qwen35-counterfactual-fork-generation-v1"
FORK_VERIFICATION_SCHEMA = "qwen35-counterfactual-fork-verification-v1"
CONFIRMATION_PLAN_SCHEMA = "qwen35-counterfactual-confirmation-plan-v1"
PREINFERENCE_SCHEMA = "qwen35-counterfactual-pre-inference-v1"
FINAL_EVIDENCE_SCHEMA = "qwen35-counterfactual-forking-results-v1"
HANDOFF_COMMIT = "bcd72d5203d82e27d50e42ec6d2d2afa061c2504"
HANDOFF_MANIFEST_SHA256 = (
    "e17eef9ea8fdd566908b6c70b6305ee9f671d62f924e3f2ee677fe9f4bb33f3a"
)
RELEASE_TRANSPORT = {
    "release_tag": "issue-92-counterfactual-parent-handoff",
    "release_url": (
        "https://github.com/murillo128/qwen-lean/releases/tag/"
        "issue-92-counterfactual-parent-handoff"
    ),
    "asset_name": "qwen-lean-issue92-counterfactual-parent-handoff.tar.zst",
    "asset_size_bytes": 539601,
    "asset_sha256": (
        "2418ed7694c95b970075c0c170e26e796638d0ec34749a5f420fdc39acb052ef"
    ),
    "package_metadata_schema": (
        "qwen-lean-issue92-counterfactual-parent-handoff-package-v1"
    ),
    "package_metadata_sha256": (
        "ee8145ca6adb40fd28cd704273618cb9f8cf0c7f50f0dd61d8aa760537d890c5"
    ),
    "sha256sums_sha256": (
        "7a5073bbbeccbd2c704ab1b2eb1f8ea9434e59ece8b0b553cf7134bec22c8cda"
    ),
    "compact_generations_sha256": (
        "e0383d34d4c59833c70582178dabcd28c7e90e6c7642cbf09a488189d79e4210"
    ),
    "record_count": 120,
    "extraction_order": "frozen handoff manifest candidate order",
}
TOTAL_GENERATION_BUDGET = 4096
LOCAL_GPU_MEMORY_UTILIZATION = 0.89
QWEN35_EOS_TOKEN_ID = 248046
QWEN35_EOS_TOKEN_TEXT = "<|im_end|>"
FORK_FRACTIONS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
DISCOVERY_SEEDS = tuple(range(100, 106))
CONFIRMATION_SEEDS = tuple(range(1000, 1012))
MATHEMATICAL_VERIFIER_CATEGORIES = {
    "verified",
    "lean_rejected",
    "empty_candidate",
}
EXPECTED_RUNTIME_VERSIONS = {
    "nvidia-ml-py": "13.610.43",
    "torch": "2.13.0+cu132",
    "transformers": "5.15.0",
    "vllm": VLLM_VERSION,
}


@dataclass(frozen=True)
class CounterfactualForkingConfig:
    path: Path
    value: dict[str, Any]
    native: NativeThinkingConfig
    repository_root: Path

    @classmethod
    def load(cls, path: Path) -> CounterfactualForkingConfig:
        resolved = path.resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        repository_root = resolved.parents[1]
        native_path = repository_root / str(value.get("native_config_path", ""))
        config = cls(
            path=resolved,
            value=value,
            native=NativeThinkingConfig.load(native_path),
            repository_root=repository_root,
        )
        validate_counterfactual_config(config)
        return config

    @property
    def handoff(self) -> dict[str, Any]:
        return self.value["handoff"]

    @property
    def parent_selection(self) -> dict[str, Any]:
        return self.value["parent_selection"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.value["execution"]

    @property
    def forks(self) -> dict[str, Any]:
        return self.value["forks"]

    @property
    def discovery(self) -> dict[str, Any]:
        return self.value["discovery"]

    @property
    def confirmation(self) -> dict[str, Any]:
        return self.value["confirmation"]

    @property
    def preflight(self) -> dict[str, Any]:
        return self.value["preflight"]

    @property
    def manifest_path(self) -> Path:
        return self.repository_root / str(self.handoff["manifest_path"])


@dataclass(frozen=True)
class ForkState:
    label: str
    fraction: float
    prefix_len: int


@dataclass(frozen=True)
class ParentTrajectory:
    ordinal: int
    task: MathiaTask
    handoff: dict[str, Any]
    record: dict[str, Any]
    record_sha256: str
    raw_response_token_ids: tuple[int, ...]
    rendered_prompt_token_ids: tuple[int, ...]
    rendered_prompt_sha256: str
    states: tuple[ForkState, ...]
    parser_parity: dict[str, Any]


@dataclass(frozen=True)
class ForkRequest:
    phase: str
    parent: ParentTrajectory
    state: ForkState
    seed: int
    max_tokens: int
    branch_id: str
    generation_config_sha256: str
    fork_prompt_token_ids: tuple[int, ...]
    interval_id: str | None = None
    interval_side: str | None = None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "parent_candidate_id": self.parent.handoff["candidate_id"],
            "workload": self.parent.task.workload,
            "task_id": self.parent.task.task_id,
            "fork_state": self.state.label,
            "fork_fraction": self.state.fraction,
            "fork_prefix_len": self.state.prefix_len,
            "branch_seed": self.seed,
            "max_tokens": self.max_tokens,
            "model_revision": MODEL_REVISION,
            "fork_generation_config_sha256": self.generation_config_sha256,
            "interval_id": self.interval_id,
            "interval_side": self.interval_side,
        }


def validate_counterfactual_config(config: CounterfactualForkingConfig) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "native_config_path": "config/qwen35-native-thinking-ab.json",
        "handoff": {
            "commit": HANDOFF_COMMIT,
            "manifest_path": (
                "evidence/qwen35-native-thinking/counterfactual-forking-handoff.json"
            ),
            "manifest_sha256": HANDOFF_MANIFEST_SHA256,
            "raw_generation_artifact_path": (
                "artifacts/qwen35-native-thinking/full/generations.jsonl"
            ),
            "release_transport": RELEASE_TRANSPORT,
        },
        "execution": {
            "gpu_memory_utilization": LOCAL_GPU_MEMORY_UTILIZATION,
        },
        "parent_selection": {
            "candidate_index": 0,
            "expected_task_count": 30,
            "minimum_reasoning_tokens": 512,
            "minimum_eligible_tasks": 24,
        },
        "forks": {
            "total_generation_budget": TOTAL_GENERATION_BUDGET,
            "fractions": list(FORK_FRACTIONS),
        },
        "discovery": {"seeds": list(DISCOVERY_SEEDS)},
        "confirmation": {
            "seeds": list(CONFIRMATION_SEEDS),
            "minimum_absolute_discovery_difference": 2 / 6,
            "maximum_positive_intervals": 10,
            "maximum_negative_intervals": 10,
            "stable_absolute_difference": 2 / 12,
        },
        "preflight": {
            "probes": [
                {"parent_ordinal": 0, "state": "P90", "seed": 9000},
                {"parent_ordinal": 1, "state": "P90", "seed": 9001},
            ]
        },
    }
    if config.value != expected:
        raise ValueError(
            "counterfactual-forking config differs from the frozen contract"
        )
    if generation_config_sha256(config.native) != (
        "b30d52a83b5c179ceb53a012351271f71f245d5a52b6b77013c42d226d1e8820"
    ):
        raise ValueError("#89 native generation contract hash changed")
    if _file_sha256(config.manifest_path) != HANDOFF_MANIFEST_SHA256:
        raise ValueError("counterfactual handoff manifest bytes changed")


def fork_generation_config_sha256(config: CounterfactualForkingConfig) -> str:
    return _sha256_json(
        {
            "counterfactual_config": config.value,
            "native_generation_config_sha256": generation_config_sha256(config.native),
            "model_revision": MODEL_REVISION,
        }
    )


def load_handoff_manifest(config: CounterfactualForkingConfig) -> dict[str, Any]:
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != HANDOFF_SCHEMA:
        raise ValueError("counterfactual handoff schema changed")
    selection = manifest.get("selection", {})
    if selection != {
        "arm": "t1",
        "candidate_count": 120,
        "candidates_per_task": 4,
        "policy": "first_frozen_task_order_with_all_candidates_durable",
        "quality_independent": True,
        "selected_tasks": selection.get("selected_tasks"),
        "task_count": 30,
    }:
        raise ValueError("counterfactual handoff selection contract changed")
    if len(selection.get("selected_tasks", [])) != 30:
        raise ValueError("counterfactual handoff must identify exactly 30 tasks")
    source = manifest.get("source", {})
    if source.get("durable_generation_artifact_path") != config.handoff.get(
        "raw_generation_artifact_path"
    ):
        raise ValueError("handoff raw artifact location changed")
    if source.get("generation_config_sha256") != generation_config_sha256(
        config.native
    ):
        raise ValueError("handoff native generation config binding changed")
    if source.get("mathia_freeze_id") != MATHIA_FREEZE_ID:
        raise ValueError("handoff Mathia freeze binding changed")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 120:
        raise ValueError("counterfactual handoff candidate population changed")
    return manifest


def validate_handoff_records(
    manifest: dict[str, Any],
    generations_path: Path,
    *,
    release_package_path: Path | None = None,
    release_transport: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    records, _ = _validate_handoff_records_with_transport(
        manifest,
        generations_path,
        release_package_path=release_package_path,
        release_transport=release_transport,
    )
    return records


def _validate_handoff_records_with_transport(
    manifest: dict[str, Any],
    generations_path: Path,
    *,
    release_package_path: Path | None,
    release_transport: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate the source JSONL or its byte-preserving compact Release package."""
    candidates = manifest["candidates"]
    expected_by_line: dict[int, dict[str, Any]] = {}
    expected_ids: set[str] = set()
    for candidate in candidates:
        line_number = int(candidate["raw_generation_jsonl_line_number"])
        candidate_id = str(candidate["candidate_id"])
        if line_number in expected_by_line:
            raise ValueError(f"duplicate handoff JSONL line: {line_number}")
        if candidate_id in expected_ids:
            raise ValueError(f"duplicate handoff candidate: {candidate_id}")
        expected_by_line[line_number] = candidate
        expected_ids.add(candidate_id)

    if not generations_path.is_file():
        raise FileNotFoundError(
            "required #89 raw generation artifact is absent: "
            f"{generations_path.resolve()}"
        )
    maximum_source_line = max(expected_by_line)
    compact_lines: list[str] = []
    source_lines: list[tuple[dict[str, Any], str]] = []
    observed_line_count = 0
    source_detected = False
    with generations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            observed_line_count = line_number
            raw_line = line.removesuffix("\n")
            if line_number <= len(candidates):
                compact_lines.append(raw_line)
            expected = expected_by_line.get(line_number)
            if expected is not None:
                source_lines.append((expected, raw_line))
            if line_number >= maximum_source_line and len(source_lines) == len(
                expected_by_line
            ):
                source_detected = True
                break
    if source_detected:
        selected_lines = source_lines
        transport = {
            "mode": "source_append_only_jsonl",
            "referenced_original_line_minimum": min(expected_by_line),
            "referenced_original_line_maximum": maximum_source_line,
            "observed_line_count_at_least": observed_line_count,
            "later_appends_allowed": True,
        }
    elif observed_line_count == len(candidates):
        if release_package_path is None or release_transport is None:
            raise ValueError(
                "compact #89 parent JSONL requires its pinned Release package "
                "and transport metadata"
            )
        transport = _validate_release_package(
            manifest,
            generations_path,
            release_package_path,
            release_transport,
        )
        selected_lines = list(zip(candidates, compact_lines, strict=True))
    else:
        raise ValueError(
            "#89 artifact is neither the append-only source JSONL nor the exact "
            f"{len(candidates)}-record compact Release payload"
        )

    located: dict[str, dict[str, Any]] = {}
    for expected, raw_line in selected_lines:
        observed_record_sha256 = _sha256_text(raw_line)
        if observed_record_sha256 != expected["raw_generation_record_sha256"]:
            raise ValueError(
                "referenced #89 record bytes changed at original line "
                f"{expected['raw_generation_jsonl_line_number']}: "
                f"{expected['candidate_id']}"
            )
        record = json.loads(raw_line)
        _validate_handoff_record(expected, record)
        if record.get("arm") != "t1" or record.get("enable_thinking") is not True:
            raise ValueError(
                f"#89 record is not a native-thinking T1 parent: "
                f"{expected['candidate_id']}"
            )
        if record.get("model_revision") != MODEL_REVISION:
            raise ValueError(
                f"#89 parent model revision changed: {expected['candidate_id']}"
            )
        if record.get("generation_config_sha256") != manifest["source"].get(
            "generation_config_sha256"
        ):
            raise ValueError(
                f"#89 parent generation config changed: {expected['candidate_id']}"
            )
        located[str(expected["candidate_id"])] = record
    missing = sorted(expected_ids - set(located))
    if missing:
        raise ValueError(f"#89 artifact is missing referenced records: {missing[:5]}")
    transport["validated_record_count"] = len(located)
    transport["record_hashes_match_handoff"] = True
    return located, transport


def _validate_release_package(
    manifest: dict[str, Any],
    generations_path: Path,
    release_package_path: Path,
    release_transport: Mapping[str, Any],
) -> dict[str, Any]:
    expected_asset_name = str(release_transport["asset_name"])
    if not release_package_path.is_file():
        raise FileNotFoundError(
            f"pinned #89 Release package is absent: {release_package_path.resolve()}"
        )
    if release_package_path.name != expected_asset_name:
        raise ValueError("#89 Release asset name differs from the pinned transport")
    if release_package_path.stat().st_size != int(
        release_transport["asset_size_bytes"]
    ):
        raise ValueError("#89 Release asset size differs from the pinned transport")
    if _file_sha256(release_package_path) != release_transport["asset_sha256"]:
        raise ValueError("#89 Release asset SHA-256 differs from the pinned transport")

    package_dir = generations_path.parent
    metadata_path = package_dir / "metadata.json"
    sums_path = package_dir / "SHA256SUMS"
    packaged_manifest_path = package_dir / "counterfactual-forking-handoff.json"
    for path in (metadata_path, sums_path, packaged_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"#89 Release package lacks {path.name}")
    if _file_sha256(metadata_path) != release_transport["package_metadata_sha256"]:
        raise ValueError("#89 Release package metadata hash changed")
    if _file_sha256(sums_path) != release_transport["sha256sums_sha256"]:
        raise ValueError("#89 Release SHA256SUMS hash changed")
    if (
        _file_sha256(generations_path)
        != release_transport["compact_generations_sha256"]
    ):
        raise ValueError("#89 compact parent JSONL hash changed")
    if _file_sha256(packaged_manifest_path) != HANDOFF_MANIFEST_SHA256:
        raise ValueError("#89 packaged handoff manifest hash changed")

    checksums = _load_sha256sums(sums_path)
    expected_checksums = {
        "counterfactual-forking-handoff.json": HANDOFF_MANIFEST_SHA256,
        "generations.jsonl": str(release_transport["compact_generations_sha256"]),
        "metadata.json": str(release_transport["package_metadata_sha256"]),
    }
    if checksums != expected_checksums:
        raise ValueError("#89 Release SHA256SUMS payload differs from the pinned files")
    observed_paths = {
        "counterfactual-forking-handoff.json": packaged_manifest_path,
        "generations.jsonl": generations_path,
        "metadata.json": metadata_path,
    }
    for filename, expected_sha256 in checksums.items():
        if _file_sha256(observed_paths[filename]) != expected_sha256:
            raise ValueError(f"#89 Release checksum failed for {filename}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    if int(release_transport["record_count"]) != len(candidates):
        raise ValueError("#89 Release transport record count changed")
    task_ids = {str(candidate["task_id"]) for candidate in candidates}
    line_numbers = [
        int(candidate["raw_generation_jsonl_line_number"]) for candidate in candidates
    ]
    required_metadata = {
        "schema_version": release_transport["package_metadata_schema"],
        "producer_issue": 89,
        "consumer_issue": 92,
        "handoff_commit": HANDOFF_COMMIT,
        "record_count": len(candidates),
        "task_count": len(task_ids),
        "candidate_count_per_task": len(candidates) // len(task_ids),
        "extraction_order": release_transport["extraction_order"],
        "source_generation_artifact": (
            "artifacts/qwen35-native-thinking/full/generations.jsonl"
        ),
        "referenced_jsonl_line_range": {
            "minimum": min(line_numbers),
            "maximum": max(line_numbers),
        },
    }
    for field, expected in required_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"#89 Release package metadata changed: {field}")
    integrity = metadata.get("integrity", {})
    required_integrity = {
        "status": "PASS",
        "byte_preserved_original_jsonl_lines": True,
        "candidates_regenerated": False,
        "records_reconstructed_from_parsed_json": False,
        "reasoning_text_retokenized": False,
        "fail_closed": True,
    }
    for field, expected in required_integrity.items():
        if integrity.get(field) != expected:
            raise ValueError(f"#89 Release integrity metadata changed: {field}")
    required_hashes = {
        "raw_generation_record_sha256",
        "raw_response_sha256",
        "raw_response_token_ids_sha256",
    }
    if not required_hashes.issubset(set(integrity.get("validated_hashes", []))):
        raise ValueError("#89 Release metadata lacks required hash validations")
    payload_files = {
        str(row["filename"]): {
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in metadata.get("payload_files", [])
    }
    expected_payload_files = {
        "generations.jsonl": {
            "sha256": str(release_transport["compact_generations_sha256"]),
            "size_bytes": generations_path.stat().st_size,
        },
        "counterfactual-forking-handoff.json": {
            "sha256": HANDOFF_MANIFEST_SHA256,
            "size_bytes": packaged_manifest_path.stat().st_size,
        },
    }
    if payload_files != expected_payload_files:
        raise ValueError("#89 Release payload metadata differs from extracted files")
    return {
        "mode": "immutable_github_release_compact_jsonl",
        "release_tag": release_transport["release_tag"],
        "release_url": release_transport["release_url"],
        "asset_name": expected_asset_name,
        "asset_size_bytes": release_package_path.stat().st_size,
        "asset_sha256": release_transport["asset_sha256"],
        "package_metadata_schema": metadata["schema_version"],
        "package_metadata_sha256": release_transport["package_metadata_sha256"],
        "compact_generations_sha256": release_transport["compact_generations_sha256"],
        "extraction_order": release_transport["extraction_order"],
    }


def _load_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.removeprefix("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or Path(filename).name != filename
            or filename in checksums
        ):
            raise ValueError("invalid #89 Release SHA256SUMS entry")
        checksums[filename] = digest
    return checksums


def _validate_handoff_record(expected: dict[str, Any], record: dict[str, Any]) -> None:
    if record.get("schema_version") != GENERATION_RECORD_SCHEMA:
        raise ValueError(f"unknown #89 record schema: {expected['candidate_id']}")
    fields = (
        "candidate_id",
        "workload",
        "task_id",
        "candidate_index",
        "seed",
        "prompt_sha256",
        "rendered_prompt_sha256",
        "raw_response_sha256",
        "raw_response_token_count",
        "reasoning_token_count",
        "final_token_count",
        "finish_reason",
    )
    for field in fields:
        if record.get(field) != expected.get(field):
            raise ValueError(
                f"#89 handoff field mismatch for {expected['candidate_id']}: {field}"
            )
    token_ids = record.get("raw_response_token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in token_ids
    ):
        raise ValueError(f"invalid #89 token ids: {expected['candidate_id']}")
    if len(token_ids) != int(expected["raw_response_token_count"]):
        raise ValueError(f"#89 token count changed: {expected['candidate_id']}")
    if _sha256_json(token_ids) != expected["raw_response_token_ids_sha256"]:
        raise ValueError(f"#89 token-id hash changed: {expected['candidate_id']}")
    raw_text = record.get("raw_response_text")
    if (
        not isinstance(raw_text, str)
        or _sha256_text(raw_text) != expected["raw_response_sha256"]
    ):
        raise ValueError(f"#89 raw-response hash changed: {expected['candidate_id']}")
    for field in ("reasoning_content", "final_content"):
        if field not in record:
            raise ValueError(f"#89 record lacks {field}: {expected['candidate_id']}")


def fork_states(reasoning_token_count: int) -> tuple[ForkState, ...]:
    if reasoning_token_count < 2:
        raise ValueError("reasoning trajectory is too short to fork")
    states = [ForkState(label="P0", fraction=0.0, prefix_len=0)]
    previous = 0
    fractions = list(FORK_FRACTIONS)
    for index, fraction in enumerate(fractions):
        raw = math.floor(fraction * reasoning_token_count)
        remaining_positions = len(fractions) - index - 1
        upper = reasoning_token_count - 1 - remaining_positions
        position = max(previous + 1, min(raw, upper))
        label = f"P{round(fraction * 100)}"
        states.append(ForkState(label=label, fraction=fraction, prefix_len=position))
        previous = position
    positions = [state.prefix_len for state in states[1:]]
    if positions != sorted(set(positions)) or not all(
        1 <= position < reasoning_token_count for position in positions
    ):
        raise AssertionError("fork checkpoint clamping failed")
    return tuple(states)


def _render_t1_prompt_token_ids(
    tokenizer: Any, user_message: str
) -> tuple[str, tuple[int, ...]]:
    messages = [{"role": "user", "content": user_message}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    token_ids = tokenized["input_ids"] if isinstance(tokenized, Mapping) else tokenized
    return str(rendered), tuple(int(value) for value in token_ids)


def _parse_combined_response(
    config: CounterfactualForkingConfig,
    tokenizer: Any,
    user_message: str,
    token_ids: Sequence[int],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.reasoning import ReasoningParserManager

    raw_text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    parser_class = ReasoningParserManager.get_reasoning_parser(
        str(config.native.engine["reasoning_parser"])
    )
    parser = parser_class(
        tokenizer,
        chat_template_kwargs={"enable_thinking": True},
    )
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=max_tokens,
        temperature=float(config.native.sampling["temperature"]),
        top_p=float(config.native.sampling["top_p"]),
        include_reasoning=True,
    )
    reasoning, final_content = parser.extract_reasoning(raw_text, request)
    reasoning_count = int(parser.count_reasoning_tokens(list(token_ids)))
    content_ids = (
        [] if final_content is None else parser.extract_content_ids(list(token_ids))
    )
    final_count = len(content_ids)
    final_content_is_exact_raw_suffix = final_content is None or raw_text.endswith(
        final_content
    )
    parser_terminal_token_ids: list[int] = []
    parser_terminal_text = ""
    if final_content is None:
        final_content_parity = "no_final_content"
    elif final_content_is_exact_raw_suffix:
        final_content_parity = "exact_raw_suffix"
    else:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        eos_token_text = (
            tokenizer.decode([int(eos_token_id)], skip_special_tokens=False)
            if eos_token_id is not None
            else ""
        )
        content_without_eos = content_ids[:-1]
        decoded_content_without_eos = tokenizer.decode(
            content_without_eos, skip_special_tokens=False
        )
        decoded_content_with_eos = tokenizer.decode(
            content_ids, skip_special_tokens=False
        )
        if (
            eos_token_id == QWEN35_EOS_TOKEN_ID
            and eos_token_text == QWEN35_EOS_TOKEN_TEXT
            and content_ids
            and content_ids[-1] == QWEN35_EOS_TOKEN_ID
            and decoded_content_without_eos == final_content
            and decoded_content_with_eos == final_content + QWEN35_EOS_TOKEN_TEXT
            and raw_text.endswith(decoded_content_with_eos)
        ):
            final_content_parity = "exact_before_terminal_eos"
            parser_terminal_token_ids = [QWEN35_EOS_TOKEN_ID]
            parser_terminal_text = QWEN35_EOS_TOKEN_TEXT
        else:
            final_content_parity = "mismatch"
    return {
        "raw_text": raw_text,
        "reasoning_content": reasoning,
        "reasoning_token_count": reasoning_count,
        "final_content": final_content,
        "final_token_count": final_count,
        "final_content_is_exact_raw_suffix": final_content_is_exact_raw_suffix,
        "final_content_parity": final_content_parity,
        "terminal_token_ids": parser_terminal_token_ids,
        "terminal_text": parser_terminal_text,
    }


def materialize_parent_trajectories(
    config: CounterfactualForkingConfig,
    mathia_root: Path,
    generations_path: Path,
    tokenizer: Any,
    *,
    release_package_path: Path | None = None,
    parser: Callable[
        [CounterfactualForkingConfig, Any, str, Sequence[int]], dict[str, Any]
    ]
    | None = None,
) -> tuple[list[ParentTrajectory], dict[str, Any]]:
    manifest = load_handoff_manifest(config)
    records_by_id, parent_transport = _validate_handoff_records_with_transport(
        manifest,
        generations_path,
        release_package_path=release_package_path,
        release_transport=config.handoff["release_transport"],
    )
    tasks, mathia_binding = load_mathia_tasks(config.native, mathia_root)
    tasks_by_id = {task.task_id: task for task in tasks}
    selected_index = int(config.parent_selection["candidate_index"])
    selected = [
        candidate
        for candidate in manifest["candidates"]
        if int(candidate["candidate_index"]) == selected_index
    ]
    selected_task_order = [
        str(row["task_id"]) for row in manifest["selection"]["selected_tasks"]
    ]
    if [str(row["task_id"]) for row in selected] != selected_task_order:
        raise ValueError("candidate-index-0 parent order differs from frozen handoff")
    if len(selected) != int(config.parent_selection["expected_task_count"]):
        raise ValueError("candidate-index-0 parent count changed")

    parents: list[ParentTrajectory] = []
    parity_rejected: list[dict[str, Any]] = []
    ineligible_short: list[dict[str, Any]] = []
    for ordinal, handoff in enumerate(selected):
        candidate_id = str(handoff["candidate_id"])
        record = records_by_id[candidate_id]
        task_id = str(handoff["task_id"])
        task = tasks_by_id.get(task_id)
        if task is None or task.workload != handoff["workload"]:
            raise ValueError(
                f"handoff task is absent from frozen Mathia input: {task_id}"
            )
        user_message = render_user_message(task)
        rendered, prompt_token_ids = _render_t1_prompt_token_ids(
            tokenizer, user_message
        )
        prompt_sha256 = _sha256_text(user_message)
        rendered_sha256 = _sha256_text(rendered)
        parity_failures: list[str] = []
        if prompt_sha256 != record.get("prompt_sha256"):
            parity_failures.append("prompt_sha256")
        if rendered_sha256 != record.get("rendered_prompt_sha256"):
            parity_failures.append("rendered_prompt_sha256")
        if len(prompt_token_ids) != int(record.get("rendered_prompt_token_count", -1)):
            parity_failures.append("rendered_prompt_token_count")
        raw_token_ids = tuple(int(value) for value in record["raw_response_token_ids"])
        parsed = (
            _parse_combined_response(
                config,
                tokenizer,
                user_message,
                raw_token_ids,
                max_tokens=TOTAL_GENERATION_BUDGET,
            )
            if parser is None
            else parser(config, tokenizer, user_message, raw_token_ids)
        )
        if parsed["raw_text"] != record.get("raw_response_text"):
            parity_failures.append("raw_token_decode")
        if parsed["reasoning_content"] != record.get("reasoning_content"):
            parity_failures.append("reasoning_content")
        if parsed["final_content"] != record.get("final_content"):
            parity_failures.append("final_content")
        if int(parsed["reasoning_token_count"]) != int(
            record.get("reasoning_token_count", -1)
        ):
            parity_failures.append("reasoning_token_count")
        if int(parsed["final_token_count"]) != int(record.get("final_token_count", -1)):
            parity_failures.append("final_token_count")
        if not parsed["final_content_is_exact_raw_suffix"]:
            parity_failures.append("channel_boundary")
        if record.get("reasoning_content") is not None and _sha256_text(
            str(record["reasoning_content"])
        ) != record.get("reasoning_content_sha256"):
            parity_failures.append("stored_reasoning_hash")
        if record.get("final_content") is not None and _sha256_text(
            str(record["final_content"])
        ) != record.get("final_content_sha256"):
            parity_failures.append("stored_final_hash")
        if parity_failures:
            parity_rejected.append(
                {"candidate_id": candidate_id, "failures": parity_failures}
            )
            continue
        reasoning_count = int(record["reasoning_token_count"])
        if reasoning_count < int(config.parent_selection["minimum_reasoning_tokens"]):
            ineligible_short.append(
                {
                    "candidate_id": candidate_id,
                    "failures": ["minimum_reasoning_tokens"],
                }
            )
            continue
        states = fork_states(reasoning_count)
        parents.append(
            ParentTrajectory(
                ordinal=ordinal,
                task=task,
                handoff=handoff,
                record=record,
                record_sha256=str(handoff["raw_generation_record_sha256"]),
                raw_response_token_ids=raw_token_ids,
                rendered_prompt_token_ids=prompt_token_ids,
                rendered_prompt_sha256=rendered_sha256,
                states=states,
                parser_parity={
                    "status": "passed",
                    "raw_response_sha256": record["raw_response_sha256"],
                    "reasoning_content_sha256": record.get("reasoning_content_sha256"),
                    "final_content_sha256": record.get("final_content_sha256"),
                    "reasoning_token_count": reasoning_count,
                    "final_token_count": int(record["final_token_count"]),
                    "finish_reason": record["finish_reason"],
                    "final_state": (
                        "none" if record.get("final_content") is None else "present"
                    ),
                },
            )
        )

    if parity_rejected:
        raise RuntimeError(
            "one or more selected parents failed the exact token/parser parity "
            f"gate; do not approximate: {parity_rejected}"
        )
    minimum = int(config.parent_selection["minimum_eligible_tasks"])
    if len(parents) < minimum:
        raise RuntimeError(
            f"only {len(parents)}/30 parents have at least "
            f"{config.parent_selection['minimum_reasoning_tokens']} reasoning "
            f"tokens; minimum is {minimum}; ineligible={ineligible_short}"
        )
    return parents, {
        "handoff_commit": HANDOFF_COMMIT,
        "handoff_manifest_sha256": HANDOFF_MANIFEST_SHA256,
        "referenced_record_count": len(records_by_id),
        "selected_parent_count": len(selected),
        "eligible_parent_count": len(parents),
        "parser_parity_failures": parity_rejected,
        "ineligible_short_parents": ineligible_short,
        "mathia_binding": mathia_binding,
        "parent_transport": parent_transport,
    }


def _request(
    config: CounterfactualForkingConfig,
    *,
    phase: str,
    parent: ParentTrajectory,
    state: ForkState,
    seed: int,
    max_tokens: int,
    interval_id: str | None = None,
    interval_side: str | None = None,
) -> ForkRequest:
    generation_hash = fork_generation_config_sha256(config)
    payload = {
        "phase": phase,
        "parent_candidate_id": parent.handoff["candidate_id"],
        "workload": parent.task.workload,
        "task_id": parent.task.task_id,
        "fork_state": state.label,
        "fork_fraction": state.fraction,
        "fork_prefix_len": state.prefix_len,
        "branch_seed": seed,
        "max_tokens": max_tokens,
        "model_revision": MODEL_REVISION,
        "fork_generation_config_sha256": generation_hash,
        "interval_id": interval_id,
        "interval_side": interval_side,
    }
    branch_id = "counterfactual-fork-" + _sha256_json(payload)[:32]
    prefix = parent.raw_response_token_ids[: state.prefix_len]
    fork_prompt_token_ids = parent.rendered_prompt_token_ids + prefix
    return ForkRequest(
        phase=phase,
        parent=parent,
        state=state,
        seed=seed,
        max_tokens=max_tokens,
        branch_id=branch_id,
        generation_config_sha256=generation_hash,
        fork_prompt_token_ids=fork_prompt_token_ids,
        interval_id=interval_id,
        interval_side=interval_side,
    )


def discovery_requests(
    config: CounterfactualForkingConfig, parents: Sequence[ParentTrajectory]
) -> list[ForkRequest]:
    return [
        _request(
            config,
            phase="discovery",
            parent=parent,
            state=state,
            seed=seed,
            max_tokens=TOTAL_GENERATION_BUDGET - state.prefix_len,
        )
        for parent in parents
        for state in parent.states
        for seed in DISCOVERY_SEEDS
    ]


def preflight_requests(
    config: CounterfactualForkingConfig, parents: Sequence[ParentTrajectory]
) -> list[ForkRequest]:
    requests: list[ForkRequest] = []
    for probe in config.preflight["probes"]:
        parent = parents[int(probe["parent_ordinal"])]
        states = {state.label: state for state in parent.states}
        state = states[str(probe["state"])]
        requests.append(
            _request(
                config,
                phase="preflight",
                parent=parent,
                state=state,
                seed=int(probe["seed"]),
                max_tokens=TOTAL_GENERATION_BUDGET - state.prefix_len,
            )
        )
    return requests


def _phase_generation_path(artifact_dir: Path, phase: str) -> Path:
    if phase not in {"preflight", "discovery", "confirmation"}:
        raise ValueError(f"unknown counterfactual generation phase: {phase}")
    return artifact_dir / f"{phase}-generations.jsonl"


def _phase_verification_path(artifact_dir: Path, phase: str) -> Path:
    if phase not in {"preflight", "discovery", "confirmation"}:
        raise ValueError(f"unknown counterfactual verification phase: {phase}")
    return artifact_dir / f"{phase}-verifications.jsonl"


def _restart_safe_jsonl_lines(path: Path) -> list[str]:
    """Read an append-only journal, preserving and dropping only a torn tail."""
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        complete_end = payload.rfind(b"\n") + 1
        interrupted_tail = payload[complete_end:]
        tail_sha256 = hashlib.sha256(interrupted_tail).hexdigest()
        recovery_path = path.with_name(
            f"{path.name}.interrupted-tail-{tail_sha256[:16]}.bin"
        )
        if recovery_path.exists():
            if recovery_path.read_bytes() != interrupted_tail:
                raise RuntimeError(
                    f"restart recovery sidecar changed unexpectedly: {recovery_path}"
                )
        else:
            with recovery_path.open("xb") as recovery:
                recovery.write(interrupted_tail)
                recovery.flush()
                os.fsync(recovery.fileno())
        with path.open("r+b") as journal:
            journal.truncate(complete_end)
            journal.flush()
            os.fsync(journal.fileno())
        payload = payload[:complete_end]
    return [line.decode("utf-8") for line in payload.splitlines() if line]


def load_fork_generation_records(
    path: Path, expected_requests: Sequence[ForkRequest]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    expected = {request.branch_id: request for request in expected_requests}
    if len(expected) != len(expected_requests):
        raise AssertionError("expected counterfactual branch identities are not unique")
    records_by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for line in _restart_safe_jsonl_lines(path):
        if not line:
            continue
        record = json.loads(line)
        branch_id = str(record.get("branch_id"))
        request = expected.get(branch_id)
        if request is None:
            raise ValueError(
                f"persisted branch is not in the frozen request set: {branch_id}"
            )
        if branch_id in seen:
            raise ValueError(f"duplicate persisted counterfactual branch: {branch_id}")
        seen.add(branch_id)
        _validate_fork_generation_record(record, request)
        records_by_id[branch_id] = record
    return [
        records_by_id[request.branch_id]
        for request in expected_requests
        if request.branch_id in records_by_id
    ]


def _validate_fork_generation_record(
    record: dict[str, Any], request: ForkRequest
) -> None:
    if record.get("schema_version") != FORK_GENERATION_SCHEMA:
        raise ValueError(
            f"unknown counterfactual generation schema: {request.branch_id}"
        )
    if record.get("branch_id") != request.branch_id:
        raise ValueError(f"counterfactual branch identity changed: {request.branch_id}")
    for field, expected in request.identity_payload().items():
        if record.get(field) != expected:
            raise ValueError(
                f"persisted branch identity field changed for "
                f"{request.branch_id}: {field}"
            )
    parent = request.parent
    bindings = {
        "parent_raw_generation_record_sha256": parent.record_sha256,
        "parent_raw_response_sha256": parent.handoff["raw_response_sha256"],
        "parent_raw_response_token_ids_sha256": parent.handoff[
            "raw_response_token_ids_sha256"
        ],
        "rendered_prompt_sha256": parent.rendered_prompt_sha256,
        "rendered_prompt_token_count": len(parent.rendered_prompt_token_ids),
        "fork_prompt_token_ids_sha256": _sha256_json(
            list(request.fork_prompt_token_ids)
        ),
        "fork_prompt_token_count": len(request.fork_prompt_token_ids),
        "reasoning_prefix_token_ids_sha256": _sha256_json(
            list(parent.raw_response_token_ids[: request.state.prefix_len])
        ),
    }
    for field, expected in bindings.items():
        if record.get(field) != expected:
            raise ValueError(
                f"persisted branch binding changed for {request.branch_id}: {field}"
            )
    suffix_ids = record.get("suffix_response_token_ids")
    if not isinstance(suffix_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in suffix_ids
    ):
        raise ValueError(f"invalid persisted suffix token ids: {request.branch_id}")
    if len(suffix_ids) != int(record.get("suffix_response_token_count", -1)):
        raise ValueError(f"persisted suffix token count changed: {request.branch_id}")
    if _sha256_json(suffix_ids) != record.get("suffix_response_token_ids_sha256"):
        raise ValueError(f"persisted suffix token hash changed: {request.branch_id}")
    suffix_text = record.get("suffix_response_text")
    if not isinstance(suffix_text, str) or _sha256_text(suffix_text) != record.get(
        "suffix_response_sha256"
    ):
        raise ValueError(f"persisted suffix response changed: {request.branch_id}")
    text_parity = record.get("suffix_text_vllm_parity")
    if text_parity is not None:
        emitted_length = int(record.get("vllm_emitted_text_length", -1))
        emitted_sha256 = record.get("vllm_emitted_text_sha256")
        if text_parity == "exact":
            expected_emitted_text = suffix_text
        elif text_parity == "token_limit_trailing_incomplete_unicode":
            if record.get("raw_finish_reason") != "length" or not suffix_text.endswith(
                "\ufffd"
            ):
                raise ValueError(
                    f"invalid token-limit text parity: {request.branch_id}"
                )
            expected_emitted_text = suffix_text[:-1]
        elif text_parity == "diagnostic_mismatch_token_ids_authoritative":
            expected_emitted_text = record.get("vllm_emitted_text")
            if not isinstance(expected_emitted_text, str) or (
                expected_emitted_text == suffix_text
            ):
                raise ValueError(
                    f"invalid diagnostic vLLM text mismatch: {request.branch_id}"
                )
            expected_mismatch_at = next(
                (
                    index
                    for index, (decoded, emitted) in enumerate(
                        zip(suffix_text, expected_emitted_text, strict=False)
                    )
                    if decoded != emitted
                ),
                min(len(suffix_text), len(expected_emitted_text)),
            )
            if record.get("vllm_text_first_mismatch_index") != (expected_mismatch_at):
                raise ValueError(
                    f"vLLM text mismatch position changed: {request.branch_id}"
                )
        else:
            raise ValueError(f"unknown vLLM text parity: {request.branch_id}")
        if emitted_length != len(expected_emitted_text) or emitted_sha256 != (
            _sha256_text(expected_emitted_text)
        ):
            raise ValueError(f"vLLM emitted text binding changed: {request.branch_id}")
    combined_text = record.get("combined_response_text")
    if not isinstance(combined_text, str) or _sha256_text(combined_text) != record.get(
        "combined_response_sha256"
    ):
        raise ValueError(f"persisted combined response changed: {request.branch_id}")
    reasoning = record.get("reasoning_content")
    if reasoning is not None and _sha256_text(str(reasoning)) != record.get(
        "reasoning_content_sha256"
    ):
        raise ValueError(f"persisted reasoning content changed: {request.branch_id}")
    final = record.get("final_content")
    if final is not None and _sha256_text(str(final)) != record.get(
        "final_content_sha256"
    ):
        raise ValueError(f"persisted final content changed: {request.branch_id}")
    parser_parity = record.get("parser_final_content_parity")
    if parser_parity is None:
        # Records written before the terminal-EOS distinction are valid only
        # under the original, stricter raw-suffix invariant.
        if not record.get("parser_final_content_is_exact_raw_suffix"):
            raise ValueError(f"parser changed final bytes: {request.branch_id}")
    else:
        parser_terminal_token_ids = record.get("parser_terminal_token_ids")
        parser_terminal_text = record.get("parser_terminal_text")
        parser_exact_raw_suffix = record.get("parser_final_content_is_exact_raw_suffix")
        if parser_parity == "no_final_content":
            valid_parser_parity = (
                final is None
                and parser_exact_raw_suffix is True
                and parser_terminal_token_ids == []
                and parser_terminal_text == ""
            )
        elif parser_parity == "exact_raw_suffix":
            valid_parser_parity = (
                isinstance(final, str)
                and parser_exact_raw_suffix is True
                and parser_terminal_token_ids == []
                and parser_terminal_text == ""
                and combined_text.endswith(final)
            )
        elif parser_parity == "exact_before_terminal_eos":
            valid_parser_parity = (
                isinstance(final, str)
                and parser_exact_raw_suffix is False
                and parser_terminal_token_ids == [QWEN35_EOS_TOKEN_ID]
                and parser_terminal_text == QWEN35_EOS_TOKEN_TEXT
                and suffix_ids[-1:] == [QWEN35_EOS_TOKEN_ID]
                and combined_text.endswith(final + QWEN35_EOS_TOKEN_TEXT)
            )
        else:
            valid_parser_parity = False
        if not valid_parser_parity:
            raise ValueError(f"parser changed final bytes: {request.branch_id}")
    if record.get("inference_input_kind") != "prompt_token_ids":
        raise ValueError(
            f"branch did not use direct token-ID input: {request.branch_id}"
        )


def _fork_local_runtime(config: CounterfactualForkingConfig) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("counterfactual generation requires torch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual generation requires project-local CUDA")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    # Compute capability 8.9 is the architecture-level Ada identity. Consumer
    # RTX 40-series device strings need not contain the literal word "Ada".
    if capability != (8, 9):
        raise RuntimeError(
            "counterfactual generation requires the project Ada GPU, got "
            f"{name} with compute capability {capability}"
        )
    properties = torch.cuda.get_device_properties(device)
    return {
        "inference_execution": "project_controlled_local_cuda",
        "cuda_device_index": int(device),
        "cuda_device": name,
        "cuda_device_capability": list(capability),
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": str(torch.version.cuda),
        "ada_validation": "cuda_compute_capability_8_9",
        "native_expected_device_name_fragment": config.native.engine[
            "expected_cuda_device_name_fragment"
        ],
    }


def _validated_package_versions() -> dict[str, str]:
    versions = _package_versions()
    if versions != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError(
            "counterfactual runtime package versions differ from #89: "
            f"{versions} != {EXPECTED_RUNTIME_VERSIONS}"
        )
    return versions


def _assert_no_other_compute_process(device_index: int) -> None:
    try:
        import pynvml
    except ImportError as error:
        raise RuntimeError("GPU exclusivity gate requires nvidia-ml-py") from error
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        competing = sorted(
            int(process.pid) for process in processes if int(process.pid) != os.getpid()
        )
    finally:
        pynvml.nvmlShutdown()
    if competing:
        raise RuntimeError(
            "project GPU already has another compute process; refusing competing "
            f"counterfactual inference (pids={competing})"
        )


def _configure_fork_runtime() -> None:
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _engine_args(config: CounterfactualForkingConfig, snapshot_path: Path) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs

    native = config.native
    return AsyncEngineArgs(
        model=str(snapshot_path),
        tokenizer=str(snapshot_path),
        revision=str(native.model["model_revision"]),
        tokenizer_revision=str(native.model["tokenizer_revision"]),
        dtype=str(native.engine["dtype"]),
        tensor_parallel_size=int(native.engine["tensor_parallel_size"]),
        gpu_memory_utilization=float(config.execution["gpu_memory_utilization"]),
        max_model_len=int(native.engine["max_model_len"]),
        max_num_seqs=int(native.engine["max_num_seqs"]),
        enforce_eager=bool(native.engine["enforce_eager"]),
        quantization=native.engine["quantization"],
        language_model_only=bool(native.engine["language_model_only"]),
        reasoning_parser=str(native.engine["reasoning_parser"]),
        generation_config="vllm",
        enable_log_requests=False,
        disable_log_stats=False,
    )


async def _run_async_fork_generation(
    config: CounterfactualForkingConfig,
    tokenizer: Any,
    requests: Sequence[ForkRequest],
    generation_path: Path,
    snapshot_path: Path,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(_engine_args(config, snapshot_path))
    queue: asyncio.Queue[ForkRequest] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    persisted: list[dict[str, Any]] = []
    persisted_lock = asyncio.Lock()
    progress_every = max(1, min(50, len(requests) // 20 or 1))

    async def worker() -> None:
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            started = time.perf_counter()
            last_output: Any | None = None
            params = SamplingParams(
                n=1,
                temperature=float(config.native.sampling["temperature"]),
                top_p=float(config.native.sampling["top_p"]),
                top_k=int(config.native.sampling["top_k"]),
                min_p=float(config.native.sampling["min_p"]),
                presence_penalty=float(config.native.sampling["presence_penalty"]),
                repetition_penalty=float(config.native.sampling["repetition_penalty"]),
                max_tokens=request.max_tokens,
                seed=request.seed,
                skip_special_tokens=False,
            )
            try:
                token_prompt = TokensPrompt(
                    prompt_token_ids=list(request.fork_prompt_token_ids)
                )
                async for output in engine.generate(
                    token_prompt,
                    params,
                    request_id=request.branch_id,
                    reasoning_ended=False,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": {"enable_thinking": True}
                    },
                ):
                    last_output = output
                if last_output is None or not last_output.finished:
                    raise RuntimeError(
                        f"vLLM fork request did not complete: {request.branch_id}"
                    )
                if len(last_output.outputs) != 1:
                    raise RuntimeError(
                        "vLLM returned an unexpected completion count for "
                        f"{request.branch_id}: {len(last_output.outputs)}"
                    )
                record = _fork_generation_record(
                    config,
                    tokenizer,
                    request,
                    last_output.outputs[0],
                    latency_seconds=time.perf_counter() - started,
                )
                async with persisted_lock:
                    _append_jsonl(generation_path, record)
                    persisted.append(record)
                    if len(persisted) % progress_every == 0 or len(persisted) == len(
                        requests
                    ):
                        print(
                            json.dumps(
                                {
                                    "phase": f"{request.phase}_generation",
                                    "completed_branches": len(persisted),
                                    "total_branches": len(requests),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(
            min(int(config.native.engine["max_in_flight_requests"]), len(requests))
        )
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        engine.shutdown()
        gc.collect()
    return persisted


def _fork_generation_record(
    config: CounterfactualForkingConfig,
    tokenizer: Any,
    request: ForkRequest,
    completion: Any,
    *,
    latency_seconds: float,
) -> dict[str, Any]:
    suffix_ids = [int(value) for value in completion.token_ids]
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=False)
    completion_text = str(completion.text)
    raw_finish_reason = (
        None if completion.finish_reason is None else str(completion.finish_reason)
    )
    if suffix_text == completion_text:
        suffix_text_vllm_parity = "exact"
        vllm_text_first_mismatch_index = None
        vllm_emitted_text = None
    elif raw_finish_reason == "length" and suffix_text == completion_text + "\ufffd":
        suffix_text_vllm_parity = "token_limit_trailing_incomplete_unicode"
        vllm_text_first_mismatch_index = len(completion_text)
        vllm_emitted_text = None
    else:
        suffix_text_vllm_parity = "diagnostic_mismatch_token_ids_authoritative"
        vllm_text_first_mismatch_index = next(
            (
                index
                for index, (decoded, emitted) in enumerate(
                    zip(suffix_text, completion_text, strict=False)
                )
                if decoded != emitted
            ),
            min(len(suffix_text), len(completion_text)),
        )
        vllm_emitted_text = completion_text
    parent_prefix = list(
        request.parent.raw_response_token_ids[: request.state.prefix_len]
    )
    combined_ids = parent_prefix + suffix_ids
    parsed = _parse_combined_response(
        config,
        tokenizer,
        render_user_message(request.parent.task),
        combined_ids,
        max_tokens=request.max_tokens,
    )
    final_content = parsed["final_content"]
    record = {
        "schema_version": FORK_GENERATION_SCHEMA,
        "branch_id": request.branch_id,
        **request.identity_payload(),
        "parent_raw_generation_record_sha256": request.parent.record_sha256,
        "parent_raw_response_sha256": request.parent.handoff["raw_response_sha256"],
        "parent_raw_response_token_ids_sha256": request.parent.handoff[
            "raw_response_token_ids_sha256"
        ],
        "rendered_prompt_sha256": request.parent.rendered_prompt_sha256,
        "rendered_prompt_token_count": len(request.parent.rendered_prompt_token_ids),
        "reasoning_prefix_token_ids_sha256": _sha256_json(parent_prefix),
        "fork_prompt_token_ids_sha256": _sha256_json(
            list(request.fork_prompt_token_ids)
        ),
        "fork_prompt_token_count": len(request.fork_prompt_token_ids),
        "inference_input_kind": "prompt_token_ids",
        "suffix_response_text": suffix_text,
        "suffix_response_sha256": _sha256_text(suffix_text),
        "suffix_response_token_ids": suffix_ids,
        "suffix_response_token_ids_sha256": _sha256_json(suffix_ids),
        "suffix_response_token_count": len(suffix_ids),
        "suffix_text_vllm_parity": suffix_text_vllm_parity,
        "vllm_emitted_text_sha256": _sha256_text(completion_text),
        "vllm_emitted_text_length": len(completion_text),
        "vllm_emitted_text": vllm_emitted_text,
        "vllm_text_first_mismatch_index": vllm_text_first_mismatch_index,
        "combined_response_text": parsed["raw_text"],
        "combined_response_sha256": _sha256_text(parsed["raw_text"]),
        "combined_response_token_count": len(combined_ids),
        "reasoning_content": parsed["reasoning_content"],
        "reasoning_content_sha256": (
            None
            if parsed["reasoning_content"] is None
            else _sha256_text(parsed["reasoning_content"])
        ),
        "reasoning_token_count": int(parsed["reasoning_token_count"]),
        "final_content": final_content,
        "final_content_sha256": (
            None if final_content is None else _sha256_text(final_content)
        ),
        "final_token_count": int(parsed["final_token_count"]),
        "final_production_status": (
            "empty" if final_content is None or final_content == "" else "nonempty"
        ),
        "parser_final_content_is_exact_raw_suffix": parsed[
            "final_content_is_exact_raw_suffix"
        ],
        "parser_final_content_parity": parsed["final_content_parity"],
        "parser_terminal_token_ids": parsed["terminal_token_ids"],
        "parser_terminal_text": parsed["terminal_text"],
        "finish_reason": _finish_reason(raw_finish_reason),
        "raw_finish_reason": raw_finish_reason,
        "generation_latency_seconds": latency_seconds,
        "request_id": request.branch_id,
    }
    _validate_fork_generation_record(record, request)
    return record


def run_fork_generation(
    config: CounterfactualForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    *,
    phase: str,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    snapshot_path = _resolve_model_snapshot(config.native)
    tokenizer = _load_tokenizer(config.native, snapshot_path=snapshot_path)
    parents, integrity = materialize_parent_trajectories(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        release_package_path=parent_release_package_path,
    )
    if phase == "discovery":
        expected = discovery_requests(config, parents)
    elif phase == "preflight":
        expected = preflight_requests(config, parents)
    elif phase == "confirmation":
        plan = load_or_build_confirmation_plan(config, parents, artifact_dir)
        expected = confirmation_requests(config, parents, plan)
    else:
        raise ValueError(f"unknown counterfactual generation phase: {phase}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_path = _phase_generation_path(artifact_dir, phase)
    prior = load_fork_generation_records(generation_path, expected)
    completed = {str(record["branch_id"]) for record in prior}
    pending = [request for request in expected if request.branch_id not in completed]
    if not pending:
        return {
            "status": "already_complete",
            "phase": phase,
            "expected_branches": len(expected),
            "new_branches": 0,
            "integrity": integrity,
        }

    _configure_fork_runtime()
    runtime = _fork_local_runtime(config)
    runtime.update(
        {
            "engine_gpu_memory_utilization": config.execution["gpu_memory_utilization"],
            "parent_engine_gpu_memory_utilization": config.native.engine[
                "gpu_memory_utilization"
            ],
        }
    )
    runtime_versions = _validated_package_versions()
    _assert_no_other_compute_process(int(runtime["cuda_device_index"]))
    monitor = _GpuMemoryMonitor(int(runtime["cuda_device_index"]), required=True)
    started = time.perf_counter()
    monitor.start()
    status = "failed"
    new_records: list[dict[str, Any]] = []
    error_text: str | None = None
    try:
        new_records = asyncio.run(
            _run_async_fork_generation(
                config,
                tokenizer,
                pending,
                generation_path,
                snapshot_path,
            )
        )
        status = "completed"
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        runtime.update(monitor.stop())
        runtime.update(
            {
                "schema_version": "qwen35-counterfactual-generation-segment-v1",
                "phase": phase,
                "status": status,
                "requested_branch_count": len(pending),
                "persisted_branch_count": len(new_records),
                "segment_wall_time_seconds": time.perf_counter() - started,
                "error": error_text,
                "package_versions": runtime_versions,
                "model_snapshot_revision": snapshot_path.name,
            }
        )
        _append_jsonl(artifact_dir / "generation-segments.jsonl", runtime)
    return {
        "status": status,
        "phase": phase,
        "expected_branches": len(expected),
        "new_branches": len(new_records),
        "integrity": integrity,
        "runtime": runtime,
    }


def load_fork_verification_records(
    path: Path, expected_requests: Sequence[ForkRequest]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    expected = {request.branch_id: request for request in expected_requests}
    attempts: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for line in _restart_safe_jsonl_lines(path):
        if not line:
            continue
        record = json.loads(line)
        branch_id = str(record.get("branch_id"))
        request = expected.get(branch_id)
        if request is None:
            raise ValueError(f"persisted verification has unknown branch: {branch_id}")
        if record.get("schema_version") != FORK_VERIFICATION_SCHEMA:
            raise ValueError(f"unknown counterfactual verification schema: {branch_id}")
        if int(record.get("attempt_index", -1)) != attempts[branch_id]:
            raise ValueError(f"non-sequential verification attempts: {branch_id}")
        attempts[branch_id] += 1
        bindings = {
            "phase": request.phase,
            "workload": request.parent.task.workload,
            "task_id": request.parent.task.task_id,
            "fork_state": request.state.label,
            "fork_prefix_len": request.state.prefix_len,
            "branch_seed": request.seed,
            "fork_generation_config_sha256": request.generation_config_sha256,
        }
        for field, expected_value in bindings.items():
            if record.get(field) != expected_value:
                raise ValueError(
                    f"verification binding changed for {branch_id}: {field}"
                )
        records.append(record)
    return records


def latest_verifications(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["branch_id"])] = record
    return latest


def _verify_fork_generation_record(
    generation: dict[str, Any],
    task: MathiaTask,
    verifier: LeanVerifier,
    *,
    attempt_index: int,
) -> dict[str, Any]:
    if generation["task_id"] != task.task_id:
        raise ValueError("counterfactual generation/task mismatch")
    final_content = generation.get("final_content")
    if final_content is None or final_content == "":
        outcome = VerificationOutcome(
            category="empty_candidate",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": "native final content is empty"},
            latency_seconds=0.0,
        )
    else:
        source = f"{task.preamble}\n\n{task.declaration} := by\n  {final_content}\n"
        outcome = verifier._run_source(source)
    return {
        "schema_version": FORK_VERIFICATION_SCHEMA,
        "branch_id": generation["branch_id"],
        "attempt_index": attempt_index,
        "phase": generation["phase"],
        "workload": generation["workload"],
        "task_id": generation["task_id"],
        "fork_state": generation["fork_state"],
        "fork_prefix_len": generation["fork_prefix_len"],
        "branch_seed": generation["branch_seed"],
        "fork_generation_config_sha256": generation["fork_generation_config_sha256"],
        "final_content_sha256": generation["final_content_sha256"],
        "final_content_submitted_without_repair": True,
        "category": outcome.category,
        "lean_success": outcome.category == "verified",
        "lean_exit_code": outcome.lean_exit_code,
        "diagnostics": outcome.diagnostics,
        "verification_latency_seconds": outcome.latency_seconds,
    }


def _expected_phase_requests(
    config: CounterfactualForkingConfig,
    parents: Sequence[ParentTrajectory],
    artifact_dir: Path,
    phase: str,
) -> list[ForkRequest]:
    if phase == "preflight":
        return preflight_requests(config, parents)
    if phase == "discovery":
        return discovery_requests(config, parents)
    if phase == "confirmation":
        plan = load_or_build_confirmation_plan(config, parents, artifact_dir)
        return confirmation_requests(config, parents, plan)
    raise ValueError(f"unknown counterfactual phase: {phase}")


def run_fork_verification(
    config: CounterfactualForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    *,
    phase: str,
    project_roots: Mapping[str, Path],
    workers: int | None = None,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    snapshot_path = _resolve_model_snapshot(config.native)
    tokenizer = _load_tokenizer(config.native, snapshot_path=snapshot_path)
    parents, integrity = materialize_parent_trajectories(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        release_package_path=parent_release_package_path,
    )
    expected = _expected_phase_requests(config, parents, artifact_dir, phase)
    generation_records = load_fork_generation_records(
        _phase_generation_path(artifact_dir, phase), expected
    )
    if len(generation_records) != len(expected):
        raise RuntimeError(
            f"{phase} generation is incomplete: "
            f"{len(generation_records)}/{len(expected)}"
        )
    tasks = [parent.task for parent in parents]
    tasks_by_id = {task.task_id: task for task in tasks}
    environment_tasks, _ = load_mathia_tasks(config.native, mathia_root)
    environments = validate_lean_environments(
        config.native, environment_tasks, project_roots
    )
    verification_path = _phase_verification_path(artifact_dir, phase)
    prior = load_fork_verification_records(verification_path, expected)
    latest = latest_verifications(prior)
    pending = [
        generation
        for generation in generation_records
        if str(generation["branch_id"]) not in latest
        or str(latest[str(generation["branch_id"])]["category"])
        not in MATHEMATICAL_VERIFIER_CATEGORIES
    ]
    if not pending:
        return {
            "status": "already_complete",
            "phase": phase,
            "generation_branches": len(generation_records),
            "new_verifications": 0,
            "integrity": integrity,
            "environments": environments,
        }
    verifiers = {
        workload: LeanVerifier(
            project_roots[workload],
            timeout_seconds=float(config.native.verifier["timeout_seconds"]),
        )
        for workload in WORKLOADS
    }
    worker_count = int(
        config.native.verifier["workers"] if workers is None else workers
    )
    if worker_count < 1:
        raise ValueError("verification worker count must be positive")
    attempts = Counter(str(record["branch_id"]) for record in prior)
    started = time.perf_counter()
    new_count = 0
    progress_every = max(1, min(50, len(pending) // 20 or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _verify_fork_generation_record,
                generation,
                tasks_by_id[str(generation["task_id"])],
                verifiers[str(generation["workload"])],
                attempt_index=attempts[str(generation["branch_id"])],
            ): str(generation["branch_id"])
            for generation in pending
        }
        for future in as_completed(futures):
            record = future.result()
            _append_jsonl(verification_path, record)
            new_count += 1
            if new_count % progress_every == 0 or new_count == len(pending):
                print(
                    json.dumps(
                        {
                            "phase": f"{phase}_verification",
                            "completed_branches": new_count,
                            "total_branches": len(pending),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    segment = {
        "schema_version": "qwen35-counterfactual-verification-segment-v1",
        "phase": phase,
        "status": "completed",
        "candidate_count": new_count,
        "workers": worker_count,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _append_jsonl(artifact_dir / "verification-segments.jsonl", segment)
    return {
        "status": "completed",
        "phase": phase,
        "generation_branches": len(generation_records),
        "new_verifications": new_count,
        "integrity": integrity,
        "environments": environments,
        "runtime": segment,
    }


def _complete_phase_rows(
    generation_records: Sequence[dict[str, Any]],
    verification_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = latest_verifications(verification_records)
    rows: list[dict[str, Any]] = []
    for generation in generation_records:
        branch_id = str(generation["branch_id"])
        verification = latest.get(branch_id)
        if verification is None:
            raise RuntimeError(f"branch lacks Lean verification: {branch_id}")
        if verification["category"] not in MATHEMATICAL_VERIFIER_CATEGORIES:
            raise RuntimeError(
                f"branch has unresolved verifier infrastructure failure: "
                f"{branch_id} ({verification['category']})"
            )
        rows.append({**generation, "verification": verification})
    return rows


def discovery_prefix_values(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), str(row["fork_state"]))].append(row)
    values: list[dict[str, Any]] = []
    for (task_id, state), state_rows in grouped.items():
        if len(state_rows) != len(DISCOVERY_SEEDS):
            raise ValueError(f"discovery state lacks six branches: {task_id}/{state}")
        seeds = sorted(int(row["branch_seed"]) for row in state_rows)
        if seeds != list(DISCOVERY_SEEDS):
            raise ValueError(f"discovery seeds changed: {task_id}/{state}")
        exemplar = state_rows[0]
        verified = sum(
            row["verification"]["category"] == "verified" for row in state_rows
        )
        nonempty = sum(
            row["final_production_status"] == "nonempty" for row in state_rows
        )
        values.append(
            {
                "workload": exemplar["workload"],
                "task_id": task_id,
                "parent_candidate_id": exemplar["parent_candidate_id"],
                "fork_state": state,
                "fork_fraction": exemplar["fork_fraction"],
                "fork_prefix_len": exemplar["fork_prefix_len"],
                "remaining_budget": exemplar["max_tokens"],
                "verified_branches": verified,
                "nonempty_final_branches": nonempty,
                "branch_count": len(state_rows),
                "V_op": verified / len(state_rows),
                "F": nonempty / len(state_rows),
            }
        )
    values.sort(
        key=lambda row: (
            str(row["task_id"]),
            float(row["fork_fraction"]),
        )
    )
    return values


def select_confirmation_intervals(
    prefix_values: Sequence[dict[str, Any]],
    *,
    task_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in prefix_values:
        by_task[str(value["task_id"])].append(value)
    order = list(task_order) if task_order is not None else sorted(by_task)
    order_index = {task_id: index for index, task_id in enumerate(order)}
    intervals: list[dict[str, Any]] = []
    for task_id, values in by_task.items():
        ordered = sorted(values, key=lambda row: float(row["fork_fraction"]))
        if len(ordered) != 7:
            raise ValueError(f"task lacks seven discovery states: {task_id}")
        for interval_index, (before, after) in enumerate(pairwise(ordered), start=1):
            delta = float(after["V_op"]) - float(before["V_op"])
            intervals.append(
                {
                    "interval_id": (
                        f"{task_id}:{before['fork_state']}->{after['fork_state']}"
                    ),
                    "workload": after["workload"],
                    "task_id": task_id,
                    "parent_candidate_id": after["parent_candidate_id"],
                    "before_state": before["fork_state"],
                    "before_fraction": before["fork_fraction"],
                    "before_prefix_len": before["fork_prefix_len"],
                    "after_state": after["fork_state"],
                    "after_fraction": after["fork_fraction"],
                    "after_prefix_len": after["fork_prefix_len"],
                    "matched_remaining_budget": (
                        TOTAL_GENERATION_BUDGET - int(after["fork_prefix_len"])
                    ),
                    "Delta_op": delta,
                    "interval_index": interval_index,
                }
            )
    threshold = 2 / 6
    positives = [row for row in intervals if float(row["Delta_op"]) >= threshold]
    negatives = [row for row in intervals if float(row["Delta_op"]) <= -threshold]
    tie_key = lambda row: (
        -abs(float(row["Delta_op"])),
        order_index[str(row["task_id"])],
        int(row["interval_index"]),
    )
    selected = sorted(positives, key=tie_key)[:10] + sorted(negatives, key=tie_key)[:10]
    for selection_index, row in enumerate(selected):
        row["selection_index"] = selection_index
        row["selection_rule"] = (
            "top-10-per-sign-with-absolute-discovery-difference-at-least-2-of-6"
        )
    return selected


def build_confirmation_plan(
    config: CounterfactualForkingConfig,
    parents: Sequence[ParentTrajectory],
    discovery_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    values = discovery_prefix_values(discovery_rows)
    selected = select_confirmation_intervals(
        values, task_order=[parent.task.task_id for parent in parents]
    )
    return {
        "schema_version": CONFIRMATION_PLAN_SCHEMA,
        "fork_generation_config_sha256": fork_generation_config_sha256(config),
        "discovery_generation_records_sha256": _sha256_json(
            [
                {
                    "branch_id": row["branch_id"],
                    "suffix_response_token_ids_sha256": row[
                        "suffix_response_token_ids_sha256"
                    ],
                    "final_content_sha256": row["final_content_sha256"],
                    "verification_category": row["verification"]["category"],
                }
                for row in discovery_rows
            ]
        ),
        "selection_threshold": 2 / 6,
        "maximum_positive_intervals": 10,
        "maximum_negative_intervals": 10,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "selected_interval_count": len(selected),
        "selected_intervals": selected,
    }


def load_or_build_confirmation_plan(
    config: CounterfactualForkingConfig,
    parents: Sequence[ParentTrajectory],
    artifact_dir: Path,
) -> dict[str, Any]:
    discovery_expected = discovery_requests(config, parents)
    generations = load_fork_generation_records(
        _phase_generation_path(artifact_dir, "discovery"), discovery_expected
    )
    if len(generations) != len(discovery_expected):
        raise RuntimeError("cannot select confirmation intervals before discovery")
    verifications = load_fork_verification_records(
        _phase_verification_path(artifact_dir, "discovery"), discovery_expected
    )
    rows = _complete_phase_rows(generations, verifications)
    expected_plan = build_confirmation_plan(config, parents, rows)
    plan_path = artifact_dir / "confirmation-plan.json"
    if plan_path.exists():
        observed = json.loads(plan_path.read_text(encoding="utf-8"))
        if observed != expected_plan:
            raise ValueError(
                "frozen confirmation plan differs from completed discovery evidence"
            )
        return observed
    _atomic_write_json(plan_path, expected_plan)
    return expected_plan


def confirmation_requests(
    config: CounterfactualForkingConfig,
    parents: Sequence[ParentTrajectory],
    plan: dict[str, Any],
) -> list[ForkRequest]:
    if plan.get("schema_version") != CONFIRMATION_PLAN_SCHEMA:
        raise ValueError("unknown confirmation plan schema")
    parents_by_task = {parent.task.task_id: parent for parent in parents}
    requests: list[ForkRequest] = []
    for interval in plan["selected_intervals"]:
        parent = parents_by_task[str(interval["task_id"])]
        states = {state.label: state for state in parent.states}
        budget = int(interval["matched_remaining_budget"])
        after = states[str(interval["after_state"])]
        if budget != TOTAL_GENERATION_BUDGET - after.prefix_len:
            raise ValueError(
                f"confirmation matched budget changed: {interval['interval_id']}"
            )
        for side, state_name in (
            ("before", str(interval["before_state"])),
            ("after", str(interval["after_state"])),
        ):
            state = states[state_name]
            for seed in CONFIRMATION_SEEDS:
                requests.append(
                    _request(
                        config,
                        phase="confirmation",
                        parent=parent,
                        state=state,
                        seed=seed,
                        max_tokens=budget,
                        interval_id=str(interval["interval_id"]),
                        interval_side=side,
                    )
                )
    return requests


def _repository_binding(config: CounterfactualForkingConfig) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.repository_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", HANDOFF_COMMIT, head],
            cwd=config.repository_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if not ancestor:
        raise RuntimeError(
            f"current target {head} does not contain exact handoff {HANDOFF_COMMIT}"
        )
    return {
        "handoff_commit": HANDOFF_COMMIT,
        "handoff_is_ancestor": True,
    }


def _restart_safety_checks(
    config: CounterfactualForkingConfig,
    request: ForkRequest,
    record: dict[str, Any],
) -> dict[str, bool]:
    parent_tamper_rejected = False
    tampered = dict(record)
    tampered["parent_raw_generation_record_sha256"] = "0" * 64
    try:
        _validate_fork_generation_record(tampered, request)
    except ValueError:
        parent_tamper_rejected = True

    config_change_changes_identity = False
    changed_payload = request.identity_payload()
    changed_payload["fork_generation_config_sha256"] = "0" * 64
    changed_id = "counterfactual-fork-" + _sha256_json(changed_payload)[:32]
    config_change_changes_identity = changed_id != request.branch_id

    suffix_tamper_rejected = False
    tampered_suffix = dict(record)
    tampered_suffix["suffix_response_token_ids"] = list(
        record["suffix_response_token_ids"]
    ) + [0]
    try:
        _validate_fork_generation_record(tampered_suffix, request)
    except ValueError:
        suffix_tamper_rejected = True
    checks = {
        "parent_hash_tampering_rejected": parent_tamper_rejected,
        "configuration_change_changes_branch_identity": (
            config_change_changes_identity
        ),
        "persisted_suffix_tampering_rejected": suffix_tamper_rejected,
        "fork_config_hash_matches": (
            request.generation_config_sha256 == fork_generation_config_sha256(config)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"restart-safety self-check failed: {checks}")
    return checks


class _RejectingVerifier:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def _run_source(self, source: str) -> VerificationOutcome:
        self.sources.append(source)
        return VerificationOutcome(
            category="lean_rejected",
            lean_exit_code=1,
            diagnostics={"stdout": "", "stderr": _sha256_text(source)},
            latency_seconds=0.0,
        )


def _verification_semantics_controls() -> dict[str, Any]:
    task = MathiaTask(
        task_id="counterfactual-verification-control",
        workload="fresh-composition-valid-v2",
        preamble="import Mathlib",
        declaration="theorem counterfactual_verification_control : True",
        declaration_name="counterfactual_verification_control",
        intuition="control",
        intuition_sha256=_sha256_text("control"),
        theorem_sha256=_sha256_text(
            "theorem counterfactual_verification_control : True"
        ),
    )
    final = "exact True.intro  "
    generation = {
        "branch_id": "counterfactual-verification-final-control",
        "phase": "preflight",
        "workload": task.workload,
        "task_id": task.task_id,
        "fork_state": "P90",
        "fork_prefix_len": 1,
        "branch_seed": 0,
        "fork_generation_config_sha256": "control",
        "final_content": final,
        "final_content_sha256": _sha256_text(final),
    }
    final_verifier = _RejectingVerifier()
    final_record = _verify_fork_generation_record(
        generation,
        task,
        final_verifier,
        attempt_index=0,  # type: ignore[arg-type]
    )
    empty_generation = {
        **generation,
        "branch_id": "counterfactual-verification-empty-control",
        "final_content": None,
        "final_content_sha256": None,
    }
    empty_verifier = _RejectingVerifier()
    empty_record = _verify_fork_generation_record(
        empty_generation,
        task,
        empty_verifier,  # type: ignore[arg-type]
        attempt_index=0,
    )
    expected_source = f"{task.preamble}\n\n{task.declaration} := by\n  {final}\n"
    controls = {
        "nonempty_final_content_sha256": _sha256_text(final),
        "nonempty_submitted_without_repair": final_record[
            "final_content_submitted_without_repair"
        ],
        "nonempty_control_category": final_record["category"],
        "nonempty_submitted_source_sha256": _sha256_text(final_verifier.sources[0]),
        "nonempty_expected_source_sha256": _sha256_text(expected_source),
        "empty_control_category": empty_record["category"],
        "empty_control_lean_success": empty_record["lean_success"],
        "empty_control_lean_invocation_count": len(empty_verifier.sources),
    }
    if controls != {
        "nonempty_final_content_sha256": _sha256_text(final),
        "nonempty_submitted_without_repair": True,
        "nonempty_control_category": "lean_rejected",
        "nonempty_submitted_source_sha256": _sha256_text(expected_source),
        "nonempty_expected_source_sha256": _sha256_text(expected_source),
        "empty_control_category": "empty_candidate",
        "empty_control_lean_success": False,
        "empty_control_lean_invocation_count": 0,
    }:
        raise AssertionError(f"verification semantics controls failed: {controls}")
    return controls


def write_preinference_evidence(
    config: CounterfactualForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    output_path: Path,
    *,
    project_roots: Mapping[str, Path],
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    snapshot_path = _resolve_model_snapshot(config.native)
    tokenizer = _load_tokenizer(config.native, snapshot_path=snapshot_path)
    parents, integrity = materialize_parent_trajectories(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        release_package_path=parent_release_package_path,
    )
    requests = preflight_requests(config, parents)
    generations = load_fork_generation_records(
        _phase_generation_path(artifact_dir, "preflight"), requests
    )
    if len(generations) != len(requests):
        raise RuntimeError(
            "two real token-ID fork probes must complete before pre-inference evidence"
        )
    verification_records = load_fork_verification_records(
        _phase_verification_path(artifact_dir, "preflight"), requests
    )
    rows = _complete_phase_rows(generations, verification_records)
    environment_tasks, _ = load_mathia_tasks(config.native, mathia_root)
    environments = validate_lean_environments(
        config.native, environment_tasks, project_roots
    )
    runtime_versions = _validated_package_versions()
    evidence = {
        "schema_version": PREINFERENCE_SCHEMA,
        "status": "passed",
        "repository": _repository_binding(config),
        "handoff": {
            "manifest_path": str(
                config.manifest_path.relative_to(config.repository_root)
            ),
            "manifest_sha256": HANDOFF_MANIFEST_SHA256,
            "raw_generation_artifact_path": config.handoff[
                "raw_generation_artifact_path"
            ],
            "referenced_record_count": integrity["referenced_record_count"],
            "all_referenced_records_integrity": True,
            "transport": integrity["parent_transport"],
        },
        "parent_selection": {
            "candidate_index": 0,
            "selected_parent_count": integrity["selected_parent_count"],
            "eligible_parent_count": integrity["eligible_parent_count"],
            "minimum_required": config.parent_selection["minimum_eligible_tasks"],
            "minimum_reasoning_tokens": config.parent_selection[
                "minimum_reasoning_tokens"
            ],
            "selection_quality_independent": True,
            "parents": [
                {
                    "ordinal": parent.ordinal,
                    "workload": parent.task.workload,
                    "task_id": parent.task.task_id,
                    "candidate_id": parent.handoff["candidate_id"],
                    "raw_generation_record_sha256": parent.record_sha256,
                    "raw_response_token_ids_sha256": parent.handoff[
                        "raw_response_token_ids_sha256"
                    ],
                    "rendered_prompt_sha256": parent.rendered_prompt_sha256,
                    "rendered_prompt_token_count": len(
                        parent.rendered_prompt_token_ids
                    ),
                    "parser_parity": parent.parser_parity,
                    "forks": [
                        {
                            "state": state.label,
                            "fraction": state.fraction,
                            "prefix_len": state.prefix_len,
                            "remaining_budget": (
                                TOTAL_GENERATION_BUDGET - state.prefix_len
                            ),
                        }
                        for state in parent.states
                    ],
                }
                for parent in parents
            ],
        },
        "frozen_runtime": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "reasoning_parser": config.native.engine["reasoning_parser"],
            "dtype": config.native.engine["dtype"],
            "vllm_version": VLLM_VERSION,
            "engine_gpu_memory_utilization": config.execution["gpu_memory_utilization"],
            "parent_engine_gpu_memory_utilization": config.native.engine[
                "gpu_memory_utilization"
            ],
            "resolved_snapshot": snapshot_path.name,
            "package_versions": runtime_versions,
            "fork_generation_config_sha256": fork_generation_config_sha256(config),
        },
        "real_token_id_probes": [
            {
                "branch_id": row["branch_id"],
                "task_id": row["task_id"],
                "fork_state": row["fork_state"],
                "prefix_len": row["fork_prefix_len"],
                "remaining_budget": row["max_tokens"],
                "seed": row["branch_seed"],
                "inference_input_kind": row["inference_input_kind"],
                "fork_prompt_token_ids_sha256": row["fork_prompt_token_ids_sha256"],
                "suffix_response_token_ids_sha256": row[
                    "suffix_response_token_ids_sha256"
                ],
                "suffix_response_token_count": row["suffix_response_token_count"],
                "final_production_status": row["final_production_status"],
                "final_content_sha256": row["final_content_sha256"],
                "verifier_category": row["verification"]["category"],
                "final_submitted_without_repair": row["verification"][
                    "final_content_submitted_without_repair"
                ],
            }
            for row in rows
        ],
        "planned_scientific_volume": {
            "discovery_branches": len(discovery_requests(config, parents)),
            "maximum_confirmation_branches": 20 * 2 * len(CONFIRMATION_SEEDS),
            "maximum_total_branches": len(discovery_requests(config, parents))
            + 20 * 2 * len(CONFIRMATION_SEEDS),
            "preflight_probes_excluded": len(requests),
        },
        "verification_semantics": _verification_semantics_controls(),
        "restart_safety": _restart_safety_checks(config, requests[0], generations[0]),
        "lean_environments": {
            workload: {
                key: value
                for key, value in environment.items()
                if key
                in {
                    "lean_toolchain",
                    "mathlib_revision",
                    "known_valid_control",
                    "placeholder_control",
                }
            }
            for workload, environment in environments.items()
        },
        "technical_preflight_status": "passed_pending_independent_review",
    }
    _atomic_write_json(output_path, evidence)
    return evidence


def run_counterfactual_preflight(
    config: CounterfactualForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    output_path: Path,
    *,
    project_roots: Mapping[str, Path],
    workers: int | None = None,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    generation = run_fork_generation(
        config,
        mathia_root,
        parent_generations_path,
        artifact_dir,
        phase="preflight",
        parent_release_package_path=parent_release_package_path,
    )
    verification = run_fork_verification(
        config,
        mathia_root,
        parent_generations_path,
        artifact_dir,
        phase="preflight",
        project_roots=project_roots,
        workers=workers,
        parent_release_package_path=parent_release_package_path,
    )
    evidence = write_preinference_evidence(
        config,
        mathia_root,
        parent_generations_path,
        artifact_dir,
        output_path,
        project_roots=project_roots,
        parent_release_package_path=parent_release_package_path,
    )
    return {
        "generation": generation,
        "verification": verification,
        "evidence": evidence,
    }


def _distribution_compact(values: Sequence[float | int]) -> dict[str, Any]:
    result = _distribution(values)
    return {
        key: result[key]
        for key in ("count", "min", "mean", "median", "p90", "p95", "max")
    }


def _confirmation_results(
    plan: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_interval_side: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_interval_side[(str(row["interval_id"]), str(row["interval_side"]))].append(
            row
        )
    results: list[dict[str, Any]] = []
    for interval in plan["selected_intervals"]:
        interval_id = str(interval["interval_id"])
        sides = {
            side: by_interval_side[(interval_id, side)] for side in ("before", "after")
        }
        for side, side_rows in sides.items():
            if len(side_rows) != len(CONFIRMATION_SEEDS):
                raise ValueError(
                    f"confirmation interval lacks 12 {side} branches: {interval_id}"
                )
            if sorted(int(row["branch_seed"]) for row in side_rows) != list(
                CONFIRMATION_SEEDS
            ):
                raise ValueError(f"confirmation seeds changed: {interval_id}/{side}")
            if {int(row["max_tokens"]) for row in side_rows} != {
                int(interval["matched_remaining_budget"])
            }:
                raise ValueError(f"confirmation budget changed: {interval_id}/{side}")
        before_verified = sum(
            row["verification"]["category"] == "verified" for row in sides["before"]
        )
        after_verified = sum(
            row["verification"]["category"] == "verified" for row in sides["after"]
        )
        v_before = before_verified / len(CONFIRMATION_SEEDS)
        v_after = after_verified / len(CONFIRMATION_SEEDS)
        delta = v_after - v_before
        discovery_delta = float(interval["Delta_op"])
        same_sign = (discovery_delta > 0 and delta > 0) or (
            discovery_delta < 0 and delta < 0
        )
        stable_threshold = 2 / 12
        if discovery_delta > 0 and delta >= stable_threshold:
            classification = "stable_positive"
        elif discovery_delta < 0 and delta <= -stable_threshold:
            classification = "stable_negative"
        elif same_sign:
            classification = "same_sign_weak"
        else:
            classification = "not_replicated"
        results.append(
            {
                **interval,
                "confirmation_seeds": list(CONFIRMATION_SEEDS),
                "before_verified_branches": before_verified,
                "after_verified_branches": after_verified,
                "V_mb_before": v_before,
                "V_mb_after": v_after,
                "Delta_mb": delta,
                "discovery_sign_replicated": same_sign,
                "classification": classification,
            }
        )
    return results


def analyze_counterfactual_results(
    config: CounterfactualForkingConfig,
    parents: Sequence[ParentTrajectory],
    discovery_rows: Sequence[dict[str, Any]],
    plan: dict[str, Any],
    confirmation_rows: Sequence[dict[str, Any]],
    *,
    generation_segments: Sequence[dict[str, Any]] = (),
    verification_segments: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    prefix_values = discovery_prefix_values(discovery_rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in prefix_values:
        by_task[str(value["task_id"])].append(value)
    deltas: list[dict[str, Any]] = []
    task_ranges: list[dict[str, Any]] = []
    for task_id, values in by_task.items():
        ordered = sorted(values, key=lambda row: float(row["fork_fraction"]))
        v_values = [float(row["V_op"]) for row in ordered]
        task_ranges.append(
            {
                "task_id": task_id,
                "range": max(v_values) - min(v_values),
                "any_nonzero_variation": len(set(v_values)) > 1,
            }
        )
        for before, after in pairwise(ordered):
            deltas.append(
                {
                    "task_id": task_id,
                    "before_state": before["fork_state"],
                    "after_state": after["fork_state"],
                    "Delta_op": float(after["V_op"]) - float(before["V_op"]),
                }
            )
    confirmations = _confirmation_results(plan, confirmation_rows)
    stable = [
        row
        for row in confirmations
        if row["classification"] in {"stable_positive", "stable_negative"}
    ]
    stable_tasks = {str(row["task_id"]) for row in stable}
    varied_tasks = sum(row["any_nonzero_variation"] for row in task_ranges)
    large_range_tasks = sum(float(row["range"]) >= 0.5 for row in task_ranges)
    replicated = sum(row["discovery_sign_replicated"] for row in confirmations)
    total_confirmed = len(confirmations)
    total_verified = sum(
        row["verification"]["category"] == "verified"
        for row in [*discovery_rows, *confirmation_rows]
    )
    eligible = len(parents)
    promising_checks = {
        "at_least_25pct_tasks_range_ge_0_5": (
            large_range_tasks / eligible >= 0.25 if eligible else False
        ),
        "at_least_half_confirmed_signs_replicate": (
            replicated / total_confirmed >= 0.5 if total_confirmed else False
        ),
        "at_least_25pct_confirmed_intervals_stable": (
            len(stable) / total_confirmed >= 0.25 if total_confirmed else False
        ),
        "nonzero_lean_verified_branches": total_verified > 0,
    }
    all_rows = [*discovery_rows, *confirmation_rows]
    category_counts = Counter(str(row["verification"]["category"]) for row in all_rows)
    no_final_count = sum(row["final_production_status"] == "empty" for row in all_rows)
    all_zero_tasks = sum(
        all(float(value["V_op"]) == 0.0 for value in values)
        for values in by_task.values()
    )
    peak_values = [
        int(segment["gpu_memory_peak_bytes"])
        for segment in generation_segments
        if segment.get("gpu_memory_peak_bytes") is not None
    ]
    generation_segment_status_counts = Counter(
        str(segment.get("status", "unknown")) for segment in generation_segments
    )
    suffix_text_parity_counts = Counter(
        str(row.get("suffix_text_vllm_parity") or "legacy_unrecorded")
        for row in all_rows
    )
    parser_final_parity_counts = Counter(
        str(row.get("parser_final_content_parity") or "legacy_strict_raw_suffix")
        for row in all_rows
    )
    completed_rows = {str(row["branch_id"]): row for row in all_rows}
    retry_outcome_variation_branch_ids: list[str] = []
    parser_error_marker = "parser changed final bytes: "
    for segment in generation_segments:
        error = str(segment.get("error") or "")
        if parser_error_marker not in error:
            continue
        branch_id = error.rsplit(parser_error_marker, 1)[-1].strip()
        completed = completed_rows.get(branch_id)
        if completed is not None and completed["final_production_status"] == "empty":
            retry_outcome_variation_branch_ids.append(branch_id)
    runtime = {
        "total_generated_tokens": sum(
            int(row["suffix_response_token_count"]) for row in all_rows
        ),
        "generation_wall_time_seconds": sum(
            float(segment.get("segment_wall_time_seconds", 0.0))
            for segment in generation_segments
        ),
        "verification_wall_time_seconds": sum(
            float(segment.get("wall_time_seconds", 0.0))
            for segment in verification_segments
        ),
        "peak_gpu_memory_bytes": max(peak_values) if peak_values else None,
        "generation_segment_count": len(generation_segments),
        "generation_segment_status_counts": dict(
            sorted(generation_segment_status_counts.items())
        ),
        "suffix_text_vllm_parity_counts": dict(
            sorted(suffix_text_parity_counts.items())
        ),
        "parser_final_content_parity_counts": dict(
            sorted(parser_final_parity_counts.items())
        ),
        "retry_outcome_variation": {
            "observed": bool(retry_outcome_variation_branch_ids),
            "branch_ids": sorted(set(retry_outcome_variation_branch_ids)),
            "interpretation": (
                "fixed per-request seeds did not make interrupted asynchronous "
                "vLLM requests deterministic across process restarts"
                if retry_outcome_variation_branch_ids
                else None
            ),
        },
        "verifier_workload_count": len(all_rows),
        "verifier_category_counts": dict(sorted(category_counts.items())),
    }
    return {
        "eligible_task_count": eligible,
        "discovery": {
            "branch_count": len(discovery_rows),
            "per_prefix": prefix_values,
            "adjacent_deltas": deltas,
            "tasks_with_any_nonzero_variation": {
                "count": varied_tasks,
                "fraction": varied_tasks / eligible if eligible else None,
            },
            "tasks_with_range_ge_0_5": {
                "count": large_range_tasks,
                "fraction": large_range_tasks / eligible if eligible else None,
            },
            "task_ranges": task_ranges,
            "V_op_distribution": _distribution_compact(
                [float(row["V_op"]) for row in prefix_values]
            ),
            "F_distribution": _distribution_compact(
                [float(row["F"]) for row in prefix_values]
            ),
            "Delta_op_distribution": _distribution_compact(
                [float(row["Delta_op"]) for row in deltas]
            ),
            "all_zero_task_count": all_zero_tasks,
        },
        "confirmation": {
            "selected_interval_count": total_confirmed,
            "intervals": confirmations,
            "sign_replication": {
                "count": replicated,
                "fraction": replicated / total_confirmed if total_confirmed else None,
            },
            "stable_positive_count": sum(
                row["classification"] == "stable_positive" for row in confirmations
            ),
            "stable_negative_count": sum(
                row["classification"] == "stable_negative" for row in confirmations
            ),
            "tasks_with_stable_interval": {
                "count": len(stable_tasks),
                "fraction": len(stable_tasks) / eligible if eligible else None,
            },
            "Delta_mb_distribution": _distribution_compact(
                [float(row["Delta_mb"]) for row in confirmations]
            ),
        },
        "parent_diagnostics": {
            "finish_reason_counts": dict(
                sorted(
                    Counter(
                        parent.record["finish_reason"] for parent in parents
                    ).items()
                )
            ),
            "final_production_counts": dict(
                sorted(
                    Counter(
                        "none"
                        if parent.record.get("final_content") is None
                        else "present"
                        for parent in parents
                    ).items()
                )
            ),
        },
        "outcome_regimes": {
            "all_zero_task_count": all_zero_tasks,
            "no_final_branch_count": no_final_count,
            "lean_rejected_branch_count": category_counts["lean_rejected"],
            "verified_branch_count": category_counts["verified"],
            "empty_candidate_branch_count": category_counts["empty_candidate"],
        },
        "runtime": runtime,
        "promising_checks": promising_checks,
        "interpretation": (
            "promising_for_later_training_or_value_work"
            if all(promising_checks.values())
            else "weak_or_uninformative_under_frozen_model_interface_and_budget"
        ),
    }


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_final_evidence(
    config: CounterfactualForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    preflight_path: Path,
    evidence_dir: Path,
    *,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("schema_version") != PREINFERENCE_SCHEMA
        or preflight.get("status") != "passed"
    ):
        raise ValueError("final evidence requires passed pre-inference evidence")
    snapshot_path = _resolve_model_snapshot(config.native)
    tokenizer = _load_tokenizer(config.native, snapshot_path=snapshot_path)
    parents, integrity = materialize_parent_trajectories(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        release_package_path=parent_release_package_path,
    )
    discovery_expected = discovery_requests(config, parents)
    discovery_generations = load_fork_generation_records(
        _phase_generation_path(artifact_dir, "discovery"), discovery_expected
    )
    discovery_verifications = load_fork_verification_records(
        _phase_verification_path(artifact_dir, "discovery"), discovery_expected
    )
    discovery_rows = _complete_phase_rows(
        discovery_generations, discovery_verifications
    )
    if len(discovery_rows) != len(discovery_expected):
        raise RuntimeError("discovery is incomplete")
    plan = load_or_build_confirmation_plan(config, parents, artifact_dir)
    confirmation_expected = confirmation_requests(config, parents, plan)
    confirmation_generations = load_fork_generation_records(
        _phase_generation_path(artifact_dir, "confirmation"), confirmation_expected
    )
    confirmation_verifications = load_fork_verification_records(
        _phase_verification_path(artifact_dir, "confirmation"), confirmation_expected
    )
    confirmation_rows = _complete_phase_rows(
        confirmation_generations, confirmation_verifications
    )
    if len(confirmation_rows) != len(confirmation_expected):
        raise RuntimeError("confirmation is incomplete")
    generation_segments = _read_optional_jsonl(
        artifact_dir / "generation-segments.jsonl"
    )
    verification_segments = _read_optional_jsonl(
        artifact_dir / "verification-segments.jsonl"
    )
    scientific_phases = {"discovery", "confirmation"}
    generation_segments = [
        segment
        for segment in generation_segments
        if segment.get("phase") in scientific_phases
    ]
    verification_segments = [
        segment
        for segment in verification_segments
        if segment.get("phase") in scientific_phases
    ]
    analysis = analyze_counterfactual_results(
        config,
        parents,
        discovery_rows,
        plan,
        confirmation_rows,
        generation_segments=generation_segments,
        verification_segments=verification_segments,
    )
    runtime = analysis["runtime"]
    limitations = [
        "bounded 30-task candidate-index-0 Mathia-guided T1 sample",
        "token-quantile checkpoints are not semantic decision boundaries",
        "operational discovery deltas mix state with consumed token budget",
        "matched-budget confirmation uses 12 Bernoulli branches per side",
        "go/no-go thresholds are heuristics rather than statistical theorems",
        "negative or weak signal is retained without selecting a training method",
    ]
    failed_segments = int(runtime["generation_segment_status_counts"].get("failed", 0))
    if failed_segments:
        limitations.append(
            f"execution required restart after {failed_segments} failed or "
            "interrupted discovery generation segments; the completed "
            "restart-safe JSONL is the authoritative sample"
        )
    diagnostic_mismatches = int(
        runtime["suffix_text_vllm_parity_counts"].get(
            "diagnostic_mismatch_token_ids_authoritative", 0
        )
    )
    if diagnostic_mismatches:
        limitations.append(
            f"vLLM emitted-text diagnostics differed from tokenizer-decoded "
            f"authoritative token IDs in {diagnostic_mismatches} persisted "
            "branches; parsing and verification used the token-ID reconstruction"
        )
    if runtime["retry_outcome_variation"]["observed"]:
        limitations.append(
            "OBSERVED: fixed per-request seeds did not reproduce identical "
            "outcomes for at least one interrupted asynchronous vLLM request "
            "after process restart"
        )
    result = {
        "schema_version": FINAL_EVIDENCE_SCHEMA,
        "experiment_id": "qwen35-4b-mathia-counterfactual-forking-v1",
        "status": "completed",
        "configuration": {
            "fork_generation_config_sha256": fork_generation_config_sha256(config),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "native_generation_config_sha256": generation_config_sha256(config.native),
            "handoff_commit": HANDOFF_COMMIT,
            "handoff_manifest_sha256": HANDOFF_MANIFEST_SHA256,
            "mathia_freeze_id": MATHIA_FREEZE_ID,
            "discovery_seeds": list(DISCOVERY_SEEDS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "total_generation_budget": TOTAL_GENERATION_BUDGET,
            "fork_fractions": list(FORK_FRACTIONS),
            "engine_gpu_memory_utilization": config.execution["gpu_memory_utilization"],
            "parent_engine_gpu_memory_utilization": config.native.engine[
                "gpu_memory_utilization"
            ],
        },
        "integrity": {
            "referenced_parent_record_count": integrity["referenced_record_count"],
            "eligible_parent_count": integrity["eligible_parent_count"],
            "all_parent_parser_parity_passed": True,
            "parent_transport": integrity["parent_transport"],
            "pre_inference_evidence_sha256": _file_sha256(preflight_path),
        },
        "identities": {
            "parents": [
                {
                    "task_id": parent.task.task_id,
                    "candidate_id": parent.handoff["candidate_id"],
                    "raw_generation_record_sha256": parent.record_sha256,
                    "raw_response_token_ids_sha256": parent.handoff[
                        "raw_response_token_ids_sha256"
                    ],
                    "forks": [
                        {
                            "state": state.label,
                            "prefix_len": state.prefix_len,
                            "remaining_budget": TOTAL_GENERATION_BUDGET
                            - state.prefix_len,
                        }
                        for state in parent.states
                    ],
                }
                for parent in parents
            ],
            "discovery_branch_count": len(discovery_expected),
            "ordered_discovery_branch_ids_sha256": _sha256_json(
                [request.branch_id for request in discovery_expected]
            ),
            "confirmation_plan_sha256": _sha256_json(plan),
            "confirmation_branch_count": len(confirmation_expected),
            "ordered_confirmation_branch_ids_sha256": _sha256_json(
                [request.branch_id for request in confirmation_expected]
            ),
        },
        "confirmation_plan": plan,
        "analysis": analysis,
        "limitations": limitations,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(evidence_dir / "results.json", result)
    (evidence_dir / "README.md").write_text(
        _render_final_readme(result), encoding="utf-8", newline="\n"
    )
    return result


def _render_final_readme(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    discovery = analysis["discovery"]
    confirmation = analysis["confirmation"]
    limitations = "\n".join(f"- {limitation}" for limitation in result["limitations"])
    return f"""# Counterfactual forking signal

This directory contains compact evidence for issue #92. Raw suffix token arrays,
combined reasoning trajectories, model caches, and verifier traces remain outside
Git in the restart-safe artifact directory.

**OBSERVED:** `{analysis["interpretation"]}`.

- Eligible tasks: {analysis["eligible_task_count"]}
- Discovery branches: {discovery["branch_count"]}
- Tasks with operational value range at least 0.5: {discovery["tasks_with_range_ge_0_5"]["count"]}
- Confirmed intervals: {confirmation["selected_interval_count"]}
- Confirmation sign replication: {confirmation["sign_replication"]["fraction"]}
- Stable positive / negative intervals: {confirmation["stable_positive_count"]} / {confirmation["stable_negative_count"]}
- Lean-verified branches: {analysis["outcome_regimes"]["verified_branch_count"]}

Discovery `Delta_op` is descriptive, not causal: adjacent states have different
remaining token budgets. Confirmation compares both sides with the later
prefix's remaining budget and independent seeds. This measurement does not
authorize or select DPO, RLVR, value learning, replay, or planner changes.

## Material limitations

{limitations}
"""
