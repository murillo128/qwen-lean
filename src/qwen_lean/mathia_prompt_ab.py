from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .baseline import (
    GeneratedCandidate,
    _convert_vllm_outputs,
    _generation_error_records,
    _verify_candidate,
    vllm_engine_kwargs,
    vllm_sampling_kwargs,
)
from .dataset_v2 import sha256_file
from .metrics import pass_at_k
from .minif2f import Phase1Config
from .phase6 import paired_task_bootstrap
from .schema import RESULT_CATEGORIES, TaskRecord
from .verifier import LeanVerifier


CONFIG_SCHEMA_VERSION = "mathia-prompt-ab-config-v1"
MANIFEST_SCHEMA_VERSION = "mathia-prompt-ab-execution-manifest-v1"
GENERATION_SHARD_SCHEMA_VERSION = "mathia-prompt-ab-generation-shard-v1"
VERIFICATION_RESULT_SCHEMA_VERSION = "mathia-prompt-ab-verification-result-v1"
RESULTS_SCHEMA_VERSION = "mathia-prompt-ab-results-v1"
ARM_IDS = ("A", "B")
WORKLOAD_IDS = (
    "minif2f-valid-clean-v2",
    "fresh-composition-valid-v2",
)
EXPECTED_CANDIDATES_PER_TASK = 8
EXPECTED_TASKS = 611
EXPECTED_CANDIDATES_PER_ARM = EXPECTED_TASKS * EXPECTED_CANDIDATES_PER_TASK
EXPECTED_CANDIDATES_TOTAL = EXPECTED_CANDIDATES_PER_ARM * len(ARM_IDS)

_SOURCE_TASK_KEYS = {
    "artifact_role",
    "declaration",
    "declaration_name",
    "evaluation_only",
    "model_visible_theorem_sha256",
    "projection_provenance",
    "public_context",
    "schema_version",
    "task_id",
    "training_eligible",
    "upstream",
    "workload",
}


@dataclass(frozen=True)
class PromptABConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> PromptABConfig:
        value = _read_json(path)
        config = cls(path=path.resolve(), value=value)
        config.validate()
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def generation(self) -> dict[str, Any]:
        return self.value["generation"]

    @property
    def engine(self) -> dict[str, Any]:
        return self.value["engine"]

    @property
    def verifier(self) -> dict[str, Any]:
        return self.value["verifier"]

    def validate(self) -> None:
        value = self.value
        if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("unknown Mathia prompt A/B config schema")
        if value.get("experiment_id") != "qwen35-4b-base-mathia-prompt-ab-v1":
            raise ValueError("Mathia prompt A/B experiment identity differs")
        model = value.get("model", {})
        expected_model = (
            "Qwen/Qwen3.5-4B-Base",
            "1001bb4d826a52d1f399e183466143f4da7b741b",
        )
        if (
            model.get("model_id"),
            model.get("model_revision"),
            model.get("tokenizer_id"),
            model.get("tokenizer_revision"),
        ) != (*expected_model, *expected_model):
            raise ValueError("Mathia prompt A/B model/tokenizer binding differs")
        if (
            model.get("adapter") is not None
            or model.get("text_only") is not True
            or model.get("add_special_tokens") is not False
            or model.get("chat_template") is not None
        ):
            raise ValueError("Mathia prompt A/B must use raw text-only Base inference")
        generation = value.get("generation", {})
        expected_generation = {
            "candidates_per_task": 8,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": -1,
            "max_new_tokens": 1024,
            "stop": "tokenizer_eos_or_token_limit",
            "seed": 0,
            "candidate_seed_mapping": (
                "vllm-request-seed-0-output-index-0-through-7"
            ),
        }
        if generation != expected_generation:
            raise ValueError("Mathia prompt A/B generation contract differs from #78 Q0")
        engine = value.get("engine", {})
        required_engine = {
            "name": "vllm",
            "version": "0.27.2rc1.dev203+g41f179b57",
            "torch_version": "2.13.0+cu130",
            "transformers_version": "5.15.1",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.89,
            "max_model_len": 32768,
            "max_num_seqs": 16,
            "enforce_eager": True,
            "language_model_only": True,
            "resolve_pinned_snapshot": True,
            "use_flashinfer_sampler": False,
            "quantization": None,
            "cpu_offload_gb": 0.0,
            "expected_cuda_compute_capability": [8, 9],
        }
        if any(engine.get(key) != expected for key, expected in required_engine.items()):
            raise ValueError("Mathia prompt A/B vLLM contract differs from #78 Q0")
        if value.get("mathia", {}).get("accepted_counts") != {
            "minif2f-valid-clean-v2": 223,
            "fresh-composition-valid-v2": 388,
            "combined": 611,
        }:
            raise ValueError("Mathia prompt A/B accepted-intuition population differs")
        arms = value.get("arms", {})
        if arms.get("A") != {
            "id": "raw-intuition-context",
            "comment_prefix": "/- Mathematical intuition:\n",
            "comment_suffix": "\n-/",
        }:
            raise ValueError("arm A prompt wording differs")
        if arms.get("B") != {
            "id": "explicit-proof-task",
            "comment_prefix": (
                "/- Complete the Lean proof below.\n"
                "Use the mathematical intuition as high-level guidance for the proof.\n"
                "Return only Lean code continuing after `by`.\n"
                "Do not use `sorry` or `admit`.\n\n"
                "Mathematical intuition:\n"
            ),
            "comment_suffix": "\n-/",
        }:
            raise ValueError("arm B prompt wording differs")
        verifier = value.get("verifier", {})
        if verifier != {
            "timeout_seconds": 30.0,
            "environments": {
                "minif2f-valid-clean-v2": {
                    "project_repository": (
                        "https://github.com/google-deepmind/miniF2F"
                    ),
                    "project_revision": (
                        "f0a20e14c1eeccd859d51bb4c2b3ee487889c303"
                    ),
                    "lean_toolchain": "leanprover/lean4:v4.27.0",
                    "mathlib_revision": (
                        "a3a10db0e9d66acbebf76c5e6a135066525ac900"
                    ),
                },
                "fresh-composition-valid-v2": {
                    "project_repository": (
                        "https://github.com/AlexKontorovich/PrimeNumberTheoremAnd"
                    ),
                    "project_revision": (
                        "7715064f690d0689f30889846f4e2c5e7ec0c47e"
                    ),
                    "lean_toolchain": "leanprover/lean4:v4.32.2",
                    "mathlib_revision": (
                        "905b95818eb32af7874a58b427f50c1711a5e96c"
                    ),
                },
            },
        }:
            raise ValueError("Mathia prompt A/B verifier contract differs from #78 Q0")


@dataclass(frozen=True)
class BoundTask:
    ordinal: int
    workload_id: str
    task: TaskRecord
    intuition_id: str
    intuition_text: str
    intuition_sha256: str
    model_visible_theorem_sha256: str
    q0_verified_candidate_count: int
    metadata: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            yield value


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    return text.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed in {root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _validate_mathia_inputs(
    config: PromptABConfig, mathia_root: Path
) -> tuple[Path, dict[str, Any]]:
    root = mathia_root.resolve()
    expected_commit = str(config.value["mathia"]["accepted_main_commit"])
    _git_output(root, "cat-file", "-e", f"{expected_commit}^{{commit}}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_commit, "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("Mathia checkout does not contain the accepted PR #61 commit")
    corpus = root / "experiments/frontier_assisted_intuition_corpus_v1"
    paths = {
        "freeze": corpus / "freeze.json",
        "accepted_intuitions": corpus / "accepted_intuitions.jsonl",
        "source_tasks": corpus / "source_tasks.jsonl",
    }
    expected_hashes = {
        "freeze": config.value["mathia"]["freeze_sha256"],
        "accepted_intuitions": config.value["mathia"][
            "accepted_intuitions_sha256"
        ],
        "source_tasks": config.value["mathia"]["source_tasks_sha256"],
    }
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[key]:
            raise ValueError(f"frozen Mathia {key} bytes differ")
    freeze = _read_json(paths["freeze"])
    if (
        freeze.get("freeze_id") != config.value["mathia"]["freeze_id"]
        or freeze.get("decision") != "FRONTIER_ASSISTED_INTUITION_CORPUS_READY"
        or freeze.get("summary", {}).get("accepted") != EXPECTED_TASKS
        or freeze.get("artifacts", {})
        .get("accepted_intuitions.jsonl", {})
        .get("sha256")
        != expected_hashes["accepted_intuitions"]
        or freeze.get("artifacts", {}).get("source_tasks.jsonl", {}).get("sha256")
        != expected_hashes["source_tasks"]
    ):
        raise ValueError("Mathia freeze contract differs")
    return corpus, freeze


def _validate_dataset_inputs(
    config: PromptABConfig, dataset_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    root = dataset_root.resolve()
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != config.value["dataset_v2"]["manifest_sha256"]:
        raise ValueError("Dataset-v2 manifest bytes differ")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_id") != config.value["dataset_v2"]["dataset_id"]:
        raise ValueError("Dataset-v2 package identity differs")
    files = manifest.get("files", {})
    expected_files = {
        "records.jsonl.gz": config.value["dataset_v2"][
            "canonical_records_sha256"
        ],
        "minif2f-valid-clean-v2.jsonl": config.value["dataset_v2"][
            "minif2f_valid_sha256"
        ],
    }
    for name, expected_hash in expected_files.items():
        path = root / name
        if (
            files.get(name, {}).get("sha256") != expected_hash
            or sha256_file(path) != expected_hash
        ):
            raise ValueError(f"Dataset-v2 {name} bytes differ")
    minif2f = list(_iter_jsonl(root / "minif2f-valid-clean-v2.jsonl"))
    if len(minif2f) != 244 or len({row["task_id"] for row in minif2f}) != 244:
        raise ValueError("Dataset-v2 clean miniF2F membership differs")
    fresh: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(root / "records.jsonl.gz"):
        if row.get("provenance") == "synthetic" and row.get("role") == "validation":
            fresh[str(row["statement_id"])] = row
    if len(fresh) != 406:
        raise ValueError("Dataset-v2 fresh-composition validation membership differs")
    return minif2f, fresh


def _load_q0_reference(
    config: PromptABConfig, repository_root: Path
) -> tuple[dict[str, Any], bytes]:
    reference = config.value["q0_reference"]
    spec = f"{reference['commit']}:{reference['path']}"
    completed = subprocess.run(
        ["git", "show", spec],
        cwd=repository_root.resolve(),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot read authoritative #78 Q0 evidence: {spec}")
    payload = completed.stdout
    if _sha256_bytes(payload) != reference["sha256"]:
        raise ValueError("authoritative #78 Q0 evidence bytes differ")
    value = json.loads(payload)
    if (
        value.get("schema_version") != "generalist-v2-q0-evidence-v1"
        or value.get("checkpoint_id") != "Q0"
        or value.get("model_id") != config.model["model_id"]
        or value.get("model_revision") != config.model["model_revision"]
        or value.get("candidates_per_task") != EXPECTED_CANDIDATES_PER_TASK
    ):
        raise ValueError("authoritative #78 Q0 evidence contract differs")
    return value, payload


def _model_visible_theorem(public_context: str, declaration: str) -> str:
    if public_context:
        return (
            f"Public Lean context:\n{public_context}\n\n"
            f"Theorem declaration:\n{declaration}"
        )
    return f"Theorem declaration:\n{declaration}"


def _validate_source_row(row: Mapping[str, Any]) -> None:
    if set(row) != _SOURCE_TASK_KEYS:
        raise ValueError("Mathia source task fields differ from the clean projection")
    if (
        row.get("schema_version")
        != "frontier-assisted-intuition-source-task-v1"
        or row.get("artifact_role") != "frontier_assisted_reference"
        or row.get("evaluation_only") is not True
        or row.get("training_eligible") is not False
        or row.get("workload") not in WORKLOAD_IDS
    ):
        raise ValueError("Mathia source task boundary differs")
    projection = row.get("projection_provenance", {})
    if projection.get("prior_generation_outputs_copied") is not False:
        raise ValueError("Mathia source projection contains prior generation outputs")
    theorem = _model_visible_theorem(
        str(row["public_context"]), str(row["declaration"])
    )
    if _sha256_text(theorem) != row["model_visible_theorem_sha256"]:
        raise ValueError("Mathia model-visible theorem hash differs")


def _dataset_task_material(
    workload_id: str,
    task_id: str,
    minif2f_by_id: Mapping[str, dict[str, Any]],
    fresh_by_id: Mapping[str, dict[str, Any]],
) -> tuple[TaskRecord, dict[str, Any]]:
    if workload_id == "minif2f-valid-clean-v2":
        row = minif2f_by_id[task_id]
        task = TaskRecord(
            id=task_id,
            preamble=str(row["preamble"]),
            declaration=str(row["declaration"]),
            declaration_name=str(row["declaration_name"]),
        )
        metadata = {"structural_class": None, "generator_family": None}
    else:
        row = fresh_by_id[task_id]
        variants = row.get("proof_variants", [])
        if len(variants) != 1:
            raise ValueError("fresh-composition task must have one verifier-only proof")
        imports = tuple(dict.fromkeys(row.get("environment", {}).get("imports", [])))
        if not imports:
            raise ValueError("fresh-composition task has no persisted import context")
        task = TaskRecord(
            id=task_id,
            preamble="\n".join(f"import {module}" for module in imports),
            declaration=str(row["canonical_declaration"]),
            declaration_name=str(variants[0]["source_declaration_name"]),
        )
        metadata = {
            "structural_class": row.get("structural_class"),
            "generator_family": row.get("generator_family"),
            "derivation_family_fingerprint": row.get(
                "derivation_family_fingerprint"
            ),
            "topic_tags": list(row.get("topic_tags", [])),
        }
    return task, metadata


def bind_tasks(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
) -> tuple[list[BoundTask], dict[str, Any]]:
    corpus, freeze = _validate_mathia_inputs(config, mathia_root)
    minif2f, fresh = _validate_dataset_inputs(config, dataset_root)
    q0, q0_bytes = _load_q0_reference(config, repository_root)
    minif2f_by_id = {str(row["task_id"]): row for row in minif2f}

    all_ids = {
        "minif2f-valid-clean-v2": [str(row["task_id"]) for row in minif2f],
        "fresh-composition-valid-v2": sorted(fresh),
    }
    q0_counts: dict[str, dict[str, int]] = {}
    for workload_id in WORKLOAD_IDS:
        lane = q0.get("workloads", {}).get(workload_id, {})
        ids = all_ids[workload_id]
        counts = lane.get("verified_counts")
        if (
            lane.get("task_count") != len(ids)
            or lane.get("candidate_count")
            != len(ids) * EXPECTED_CANDIDATES_PER_TASK
            or lane.get("ordered_task_ids_sha256") != _ordered_ids_sha256(ids)
            or not isinstance(counts, list)
            or len(counts) != len(ids)
            or any(not isinstance(count, int) or not 0 <= count <= 8 for count in counts)
        ):
            raise ValueError(f"#78 Q0 task evidence differs for {workload_id}")
        q0_counts[workload_id] = dict(zip(ids, counts, strict=True))

    source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_jsonl(corpus / "source_tasks.jsonl"):
        _validate_source_row(row)
        key = (str(row["workload"]), str(row["task_id"]))
        if key in source_rows:
            raise ValueError(f"duplicate Mathia source task: {key}")
        source_rows[key] = row

    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_jsonl(corpus / "accepted_intuitions.jsonl"):
        key = (str(row.get("workload")), str(row.get("task_id")))
        text = str(row.get("text"))
        if (
            row.get("schema_version")
            != "frontier-assisted-intuition-accepted-v2"
            or row.get("artifact_role") != "frontier_assisted_reference"
            or row.get("evaluation_only") is not True
            or row.get("training_eligible") is not False
            or key[0] not in WORKLOAD_IDS
            or _sha256_text(text) != row.get("text_sha256")
            or "-/" in text
        ):
            raise ValueError(f"accepted Mathia intuition boundary differs: {key}")
        if key in accepted:
            raise ValueError(f"duplicate accepted Mathia intuition: {key}")
        accepted[key] = row
    observed_counts = Counter(workload for workload, _ in accepted)
    expected_counts = config.value["mathia"]["accepted_counts"]
    if (
        observed_counts != Counter(
            {
                "minif2f-valid-clean-v2": expected_counts[
                    "minif2f-valid-clean-v2"
                ],
                "fresh-composition-valid-v2": expected_counts[
                    "fresh-composition-valid-v2"
                ],
            }
        )
        or len(accepted) != expected_counts["combined"]
    ):
        raise ValueError("accepted Mathia intuition counts differ")

    bound: list[BoundTask] = []
    for workload_id in WORKLOAD_IDS:
        for task_id in all_ids[workload_id]:
            key = (workload_id, task_id)
            intuition = accepted.get(key)
            if intuition is None:
                continue
            source = source_rows.get(key)
            if source is None:
                raise ValueError(f"accepted intuition has no clean source task: {key}")
            task, metadata = _dataset_task_material(
                workload_id, task_id, minif2f_by_id, fresh
            )
            if (
                source["public_context"] != task.preamble
                or source["declaration"] != task.declaration
                or source["declaration_name"] != task.declaration_name
                or source["model_visible_theorem_sha256"]
                != intuition["model_visible_theorem_sha256"]
            ):
                raise ValueError(f"Mathia/qwen-lean theorem bytes differ: {key}")
            bound.append(
                BoundTask(
                    ordinal=len(bound),
                    workload_id=workload_id,
                    task=task,
                    intuition_id=str(intuition["intuition_id"]),
                    intuition_text=str(intuition["text"]),
                    intuition_sha256=str(intuition["text_sha256"]),
                    model_visible_theorem_sha256=str(
                        intuition["model_visible_theorem_sha256"]
                    ),
                    q0_verified_candidate_count=q0_counts[workload_id][task_id],
                    metadata=metadata,
                )
            )
    if len(bound) != EXPECTED_TASKS:
        raise ValueError("bound Mathia prompt A/B population differs")
    return bound, {
        "mathia_freeze": freeze,
        "q0_evidence_sha256": _sha256_bytes(q0_bytes),
        "full_workload_ordered_ids_sha256": {
            workload_id: _ordered_ids_sha256(all_ids[workload_id])
            for workload_id in WORKLOAD_IDS
        },
    }


def render_arm_prompt(config: PromptABConfig, bound: BoundTask, arm_id: str) -> str:
    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown Mathia prompt A/B arm: {arm_id}")
    arm = config.value["arms"][arm_id]
    comment = (
        str(arm["comment_prefix"])
        + bound.intuition_text
        + str(arm["comment_suffix"])
    )
    return (
        f"{bound.task.preamble}\n\n{comment}\n\n"
        f"{bound.task.declaration} := by\n  "
    )


def _generation_contract(config: PromptABConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "generation": config.generation,
        "engine": config.engine,
        "raw_continuation_extraction": True,
        "prompt_transport": "plain-text-no-chat-template",
    }


def candidate_identity(
    *,
    arm_id: str,
    workload_id: str,
    task_id: str,
    prompt_sha256: str,
    candidate_index: int,
    sampling_seed: int,
    model_revision: str,
    generation_config_sha256: str,
) -> str:
    value = {
        "arm_id": arm_id,
        "workload_id": workload_id,
        "task_id": task_id,
        "prompt_sha256": prompt_sha256,
        "candidate_index": candidate_index,
        "sampling_seed": sampling_seed,
        "model_revision": model_revision,
        "generation_config_sha256": generation_config_sha256,
    }
    return f"mathia-prompt-ab-candidate-{_sha256_json(value)}"


def build_execution_manifest(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    bound_tasks, bindings = bind_tasks(
        config, dataset_root, mathia_root, repository_root
    )
    generation_contract = _generation_contract(config)
    generation_hash = _sha256_json(generation_contract)
    instruction = (
        "Complete the Lean proof below.\n"
        "Use the mathematical intuition as high-level guidance for the proof.\n"
        "Return only Lean code continuing after `by`.\n"
        "Do not use `sorry` or `admit`."
    )
    tasks: list[dict[str, Any]] = []
    all_candidate_ids: set[str] = set()
    workload_counts: Counter[str] = Counter()
    for bound in bound_tasks:
        prompts = {
            arm_id: render_arm_prompt(config, bound, arm_id) for arm_id in ARM_IDS
        }
        expected_b = prompts["A"].replace(
            "/- Mathematical intuition:\n",
            f"/- {instruction}\n\nMathematical intuition:\n",
            1,
        )
        if prompts["B"] != expected_b:
            raise ValueError("arm B differs from arm A by more than the frozen instruction")
        prompt_hashes = {
            arm_id: _sha256_text(prompts[arm_id]) for arm_id in ARM_IDS
        }
        slots: dict[str, list[str]] = {}
        for arm_id in ARM_IDS:
            candidates = []
            for candidate_index in range(EXPECTED_CANDIDATES_PER_TASK):
                candidate_id = candidate_identity(
                    arm_id=arm_id,
                    workload_id=bound.workload_id,
                    task_id=bound.task.id,
                    prompt_sha256=prompt_hashes[arm_id],
                    candidate_index=candidate_index,
                    sampling_seed=int(config.generation["seed"]),
                    model_revision=str(config.model["model_revision"]),
                    generation_config_sha256=generation_hash,
                )
                if candidate_id in all_candidate_ids:
                    raise ValueError("candidate identity collision")
                all_candidate_ids.add(candidate_id)
                candidates.append(candidate_id)
            slots[arm_id] = candidates
        tasks.append(
            {
                "ordinal": bound.ordinal,
                "workload_id": bound.workload_id,
                "task_id": bound.task.id,
                "declaration_name": bound.task.declaration_name,
                "public_context_sha256": _sha256_text(bound.task.preamble),
                "declaration_sha256": _sha256_text(bound.task.declaration),
                "model_visible_theorem_sha256": (
                    bound.model_visible_theorem_sha256
                ),
                "intuition_id": bound.intuition_id,
                "intuition_sha256": bound.intuition_sha256,
                "prompt_sha256": prompt_hashes,
                "q0_verified_candidate_count": bound.q0_verified_candidate_count,
                "candidate_slots": slots,
                "metadata": bound.metadata,
            }
        )
        workload_counts[bound.workload_id] += 1
    if len(all_candidate_ids) != EXPECTED_CANDIDATES_TOTAL:
        raise ValueError("prospective candidate identity count differs")
    prompt_templates = {
        arm_id: {
            "comment_prefix": config.value["arms"][arm_id]["comment_prefix"],
            "comment_suffix": config.value["arms"][arm_id]["comment_suffix"],
            "template_sha256": _sha256_json(config.value["arms"][arm_id]),
        }
        for arm_id in ARM_IDS
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": config.value["experiment_id"],
        "config_sha256": sha256_file(config.path),
        "generation_config_sha256": generation_hash,
        "generation_contract": generation_contract,
        "prompt_templates": prompt_templates,
        "task_count": len(tasks),
        "task_counts": {
            workload_id: workload_counts[workload_id] for workload_id in WORKLOAD_IDS
        },
        "candidates_per_task": EXPECTED_CANDIDATES_PER_TASK,
        "prospective_candidate_counts": {
            "A": EXPECTED_CANDIDATES_PER_ARM,
            "B": EXPECTED_CANDIDATES_PER_ARM,
            "combined": EXPECTED_CANDIDATES_TOTAL,
        },
        "ordered_task_ids_sha256": _ordered_ids_sha256(
            [bound.task.id for bound in bound_tasks]
        ),
        "bindings": {
            "dataset_v2": config.value["dataset_v2"],
            "mathia": config.value["mathia"],
            "q0_reference": config.value["q0_reference"],
            "full_workload_ordered_ids_sha256": bindings[
                "full_workload_ordered_ids_sha256"
            ],
        },
        "prompt_integrity_gate": {
            "passed": True,
            "same_task_ids_and_order": True,
            "same_intuition_bytes_and_hashes": True,
            "same_declaration_bytes": True,
            "b_differs_only_by_instruction_sha256": _sha256_text(instruction),
            "model_visible_construction_fields": [
                "public_context",
                "frozen_arm_comment_prefix",
                "frozen_intuition_text",
                "frozen_arm_comment_suffix",
                "declaration",
                "literal_:=_by_continuation_prefix",
            ],
            "oracle_or_source_proof_loaded_into_prompt": False,
            "q0_result_loaded_into_prompt": False,
            "q4_or_deepseek_result_loaded_into_prompt": False,
            "final_test_information_loaded": False,
        },
        "tasks": tasks,
    }
    return manifest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_json(path: Path, value: Any) -> None:
    payload = _canonical_json_bytes(value, pretty=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    _atomic_write(path, payload)


def _replace_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json_bytes(value, pretty=True))


def materialize_execution_manifest(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = build_execution_manifest(
        config, dataset_root, mathia_root, repository_root
    )
    _write_once_json(output_path, manifest)
    return manifest


def validate_execution_manifest(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[BoundTask]]:
    observed = _read_json(manifest_path)
    expected = build_execution_manifest(
        config, dataset_root, mathia_root, repository_root
    )
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
        raise ValueError(
            "execution manifest no longer matches frozen model, prompts, membership, "
            "intuition hashes, Q0 reference, or candidate-seed mapping"
        )
    bound, _ = bind_tasks(config, dataset_root, mathia_root, repository_root)
    return observed, bound


def _task_shard_name(task_entry: Mapping[str, Any]) -> str:
    identity = _sha256_text(
        f"{task_entry['workload_id']}\0{task_entry['task_id']}"
    )[:16]
    return f"{int(task_entry['ordinal']):04d}-{identity}.json"


def _generation_shard_path(
    artifact_root: Path, arm_id: str, task_entry: Mapping[str, Any]
) -> Path:
    return (
        artifact_root
        / "generations"
        / arm_id
        / str(task_entry["workload_id"])
        / _task_shard_name(task_entry)
    )


def _verification_result_path(
    artifact_root: Path,
    arm_id: str,
    workload_id: str,
    candidate_id: str,
) -> Path:
    return (
        artifact_root
        / "verifications"
        / arm_id
        / workload_id
        / f"{candidate_id}.json"
    )


class _ArtifactLock:
    def __init__(self, path: Path, *, operation: str, manifest_sha256: str) -> None:
        self.path = path
        self.operation = operation
        self.manifest_sha256 = manifest_sha256
        self.handle: Any = None

    def __enter__(self) -> _ArtifactLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            raise RuntimeError(f"another {self.operation} process owns {self.path}") from error
        self.handle.seek(0)
        self.handle.truncate()
        json.dump(
            {
                "operation": self.operation,
                "manifest_sha256": self.manifest_sha256,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "started_at_utc": _utc_now(),
            },
            self.handle,
            sort_keys=True,
        )
        self.handle.write("\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _next_session_path(artifact_root: Path, operation: str) -> Path:
    directory = artifact_root / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in directory.glob(f"{operation}-*.json"):
        try:
            indices.append(int(path.stem.rsplit("-", 1)[1]))
        except ValueError:
            raise ValueError(f"malformed session artifact: {path}")
    return directory / f"{operation}-{max(indices, default=0) + 1:04d}.json"


def _manifest_tasks_by_ordinal(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    tasks = {int(task["ordinal"]): task for task in manifest["tasks"]}
    if sorted(tasks) != list(range(EXPECTED_TASKS)):
        raise ValueError("execution manifest ordinals are not complete and contiguous")
    return tasks


def _validate_generation_shard(
    shard: Mapping[str, Any],
    *,
    arm_id: str,
    task_entry: Mapping[str, Any],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    expected_slots = task_entry["candidate_slots"][arm_id]
    candidates = shard.get("candidates")
    if (
        shard.get("schema_version") != GENERATION_SHARD_SCHEMA_VERSION
        or shard.get("manifest_sha256") != manifest_sha256
        or shard.get("arm_id") != arm_id
        or shard.get("workload_id") != task_entry["workload_id"]
        or shard.get("task_id") != task_entry["task_id"]
        or shard.get("task_ordinal") != task_entry["ordinal"]
        or shard.get("prompt_sha256") != task_entry["prompt_sha256"][arm_id]
        or not isinstance(candidates, list)
        or len(candidates) != EXPECTED_CANDIDATES_PER_TASK
    ):
        raise ValueError(
            f"completed generation shard differs: {arm_id}/{task_entry['task_id']}"
        )
    for candidate_index, (observed, expected_id) in enumerate(
        zip(candidates, expected_slots, strict=True)
    ):
        raw_text = observed.get("raw_continuation")
        if (
            observed.get("candidate_id") != expected_id
            or observed.get("candidate_index") != candidate_index
            or observed.get("sampling_seed") != 0
            or not isinstance(raw_text, str)
            or observed.get("raw_continuation_sha256") != _sha256_text(raw_text)
            or observed.get("finish_reason")
            not in {"eos", "token_limit", "unknown", "generation_error"}
            or not isinstance(observed.get("token_count"), int)
            or observed.get("token_count") < 0
        ):
            raise ValueError(
                f"completed candidate bytes/identity differ: {expected_id}"
            )
    return list(candidates)


def inventory_generations(
    manifest: Mapping[str, Any], artifact_root: Path, manifest_sha256: str
) -> dict[str, Any]:
    seen: dict[str, tuple[str, str]] = {}
    shards: list[dict[str, Any]] = []
    completed_tasks: dict[str, set[int]] = {arm_id: set() for arm_id in ARM_IDS}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    failures = 0
    output_tokens = 0
    tasks_by_ordinal = _manifest_tasks_by_ordinal(manifest)
    for arm_id in ARM_IDS:
        for ordinal, task_entry in tasks_by_ordinal.items():
            path = _generation_shard_path(artifact_root, arm_id, task_entry)
            if not path.exists():
                continue
            shard = _read_json(path)
            candidates = _validate_generation_shard(
                shard,
                arm_id=arm_id,
                task_entry=task_entry,
                manifest_sha256=manifest_sha256,
            )
            file_hash = sha256_file(path)
            relative = str(path.relative_to(artifact_root))
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                previous = seen.get(candidate_id)
                if previous is not None:
                    if previous != (relative, file_hash):
                        raise ValueError(
                            f"duplicate completed candidate identity: {candidate_id}"
                        )
                    raise ValueError(
                        f"duplicate candidate transport evidence: {candidate_id}"
                    )
                seen[candidate_id] = (relative, file_hash)
                candidates_by_id[candidate_id] = {
                    **candidate,
                    "arm_id": arm_id,
                    "workload_id": task_entry["workload_id"],
                    "task_id": task_entry["task_id"],
                    "task_ordinal": ordinal,
                    "generation_shard": relative,
                }
                failures += candidate.get("generation_error") is not None
                output_tokens += int(candidate["token_count"])
            completed_tasks[arm_id].add(ordinal)
            shards.append(
                {
                    "path": relative,
                    "sha256": file_hash,
                    "arm_id": arm_id,
                    "workload_id": task_entry["workload_id"],
                    "task_id": task_entry["task_id"],
                    "candidate_count": len(candidates),
                }
            )
    shards.sort(key=lambda item: (item["arm_id"], item["path"]))
    return {
        "completed_candidate_count": len(seen),
        "completed_task_count": sum(len(value) for value in completed_tasks.values()),
        "completed_tasks_by_arm": {
            arm_id: sorted(completed_tasks[arm_id]) for arm_id in ARM_IDS
        },
        "generation_failure_count": failures,
        "cumulative_output_token_count": output_tokens,
        "shards": shards,
        "candidates_by_id": candidates_by_id,
    }


def _validate_cuda_device_identity(
    device_name: str,
    major: int,
    minor: int,
    expected_capability: Sequence[int],
) -> None:
    observed = [major, minor]
    if observed != list(expected_capability):
        raise RuntimeError(
            "project Ada GPU not detected: "
            f"{device_name} has compute capability {major}.{minor}, expected "
            f"{expected_capability[0]}.{expected_capability[1]}"
        )


def _runtime_identity(config: PromptABConfig) -> dict[str, Any]:
    try:
        import torch
        import transformers
        import vllm
    except ImportError as error:
        raise RuntimeError("Mathia prompt A/B generation runtime is not installed") from error
    if vllm.__version__ != config.engine["version"]:
        raise RuntimeError(
            f"vLLM version mismatch: expected {config.engine['version']}, "
            f"got {vllm.__version__}"
        )
    if torch.__version__ != config.engine["torch_version"]:
        raise RuntimeError(
            f"Torch version mismatch: expected {config.engine['torch_version']}, "
            f"got {torch.__version__}"
        )
    if transformers.__version__ != config.engine["transformers_version"]:
        raise RuntimeError(
            "Transformers version mismatch: expected "
            f"{config.engine['transformers_version']}, got {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Mathia prompt A/B generation requires local CUDA")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    _validate_cuda_device_identity(
        properties.name,
        properties.major,
        properties.minor,
        config.engine["expected_cuda_compute_capability"],
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm": vllm.__version__,
        "cuda_device_index": index,
        "cuda_device": properties.name,
        "cuda_device_capability": [properties.major, properties.minor],
        "cuda_device_total_memory_bytes": properties.total_memory,
        "inference_execution": "project-controlled-local-cuda",
        "hostname": socket.gethostname(),
    }


def _phase1_generation_config(config: PromptABConfig) -> Phase1Config:
    return Phase1Config(
        path=config.path,
        value={
            "schema_version": "phase1-config-v1",
            "model": {
                "model_id": config.model["model_id"],
                "model_revision": config.model["model_revision"],
                "tokenizer_id": config.model["tokenizer_id"],
                "tokenizer_revision": config.model["tokenizer_revision"],
                "add_special_tokens": False,
                "chat_template": None,
            },
            "sampling": dict(config.generation),
            "engine": dict(config.engine),
        },
    )


def _progress_snapshot(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    inventory: Mapping[str, Any],
    runtime_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    per_arm: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        completed_tasks = len(inventory["completed_tasks_by_arm"][arm_id])
        per_arm[arm_id] = {
            "completed_candidates": completed_tasks * EXPECTED_CANDIDATES_PER_TASK,
            "total_candidates": EXPECTED_CANDIDATES_PER_ARM,
            "completed_tasks": completed_tasks,
            "total_tasks": EXPECTED_TASKS,
        }
    return {
        "schema_version": "mathia-prompt-ab-generation-progress-v1",
        "manifest_sha256": manifest_sha256,
        "updated_at_utc": _utc_now(),
        "arms": per_arm,
        "combined": {
            "completed_candidates": inventory["completed_candidate_count"],
            "total_candidates": EXPECTED_CANDIDATES_TOTAL,
            "completed_tasks": inventory["completed_task_count"],
            "total_tasks": EXPECTED_TASKS * len(ARM_IDS),
        },
        "generation_failure_count": inventory["generation_failure_count"],
        "incomplete_candidate_count": (
            EXPECTED_CANDIDATES_TOTAL - inventory["completed_candidate_count"]
        ),
        "cumulative_output_token_count": inventory[
            "cumulative_output_token_count"
        ],
        "raw_output_shards": inventory["shards"],
        "runtime_identity": runtime_identity,
        "manifest_candidate_counts": manifest["prospective_candidate_counts"],
    }


def set_pause(artifact_root: Path, *, reason: str | None = None) -> Path:
    marker = artifact_root / "PAUSE"
    _replace_json(
        marker,
        {"requested_at_utc": _utc_now(), "reason": reason or "operator-requested"},
    )
    return marker


def clear_pause(artifact_root: Path) -> None:
    (artifact_root / "PAUSE").unlink(missing_ok=True)


def run_resumable_generation(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
    manifest_path: Path,
    artifact_root: Path,
    *,
    arms: Sequence[str] = ARM_IDS,
    chunk_tasks: int = 4,
    max_chunks: int | None = None,
    resume: bool,
) -> dict[str, Any]:
    if not resume:
        raise ValueError("generation requires explicit --resume acknowledgement")
    if chunk_tasks < 1:
        raise ValueError("generation chunk size must be positive")
    if max_chunks is not None and max_chunks < 1:
        raise ValueError("max chunks must be positive")
    resolved_arms = tuple(arms)
    if not resolved_arms or any(arm_id not in ARM_IDS for arm_id in resolved_arms):
        raise ValueError("generation arms must be A and/or B")
    manifest, bound_tasks = validate_execution_manifest(
        config, dataset_root, mathia_root, repository_root, manifest_path
    )
    manifest_sha256 = sha256_file(manifest_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    with _ArtifactLock(
        artifact_root / ".generation.lock",
        operation="generation",
        manifest_sha256=manifest_sha256,
    ):
        inventory = inventory_generations(manifest, artifact_root, manifest_sha256)
        missing: list[tuple[str, dict[str, Any], BoundTask]] = []
        task_entries = _manifest_tasks_by_ordinal(manifest)
        for arm_id in resolved_arms:
            completed = set(inventory["completed_tasks_by_arm"][arm_id])
            for bound in bound_tasks:
                if bound.ordinal not in completed:
                    missing.append((arm_id, task_entries[bound.ordinal], bound))
        if not missing:
            snapshot = _progress_snapshot(
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                inventory=inventory,
                runtime_identity=None,
            )
            _replace_json(artifact_root / "progress" / "generation.json", snapshot)
            return snapshot

        session_path = _next_session_path(artifact_root, "generation")
        session = {
            "schema_version": "mathia-prompt-ab-generation-session-v1",
            "manifest_sha256": manifest_sha256,
            "session_id": session_path.stem,
            "state": "starting",
            "started_at_utc": _utc_now(),
            "requested_arms": list(resolved_arms),
            "chunk_tasks": chunk_tasks,
            "completed_chunks": 0,
        }
        _replace_json(session_path, session)
        runtime = _runtime_identity(config)
        session.update({"state": "running", "runtime_identity": runtime})
        _replace_json(session_path, session)
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        phase1 = _phase1_generation_config(config)
        sampling = dict(config.generation)
        try:
            import vllm
            from vllm import LLM, SamplingParams

            llm = LLM(**vllm_engine_kwargs(phase1, sampling, None))
            sampling_params = SamplingParams(**vllm_sampling_kwargs(sampling))
            chunk_count = 0
            while missing:
                if (artifact_root / "PAUSE").exists():
                    session["state"] = "paused"
                    break
                arm_id = missing[0][0]
                chunk: list[tuple[str, dict[str, Any], BoundTask]] = []
                while missing and len(chunk) < chunk_tasks and missing[0][0] == arm_id:
                    chunk.append(missing.pop(0))
                prompts = [render_arm_prompt(config, item[2], arm_id) for item in chunk]
                tasks = [item[2].task for item in chunk]
                started = time.perf_counter()
                try:
                    outputs = llm.generate(
                        prompts,
                        sampling_params,
                        use_tqdm=True,
                    )
                    generated = _convert_vllm_outputs(
                        tasks,
                        prompts,
                        outputs,
                        time.perf_counter() - started,
                        sampling=sampling,
                    )
                except KeyboardInterrupt:
                    missing = chunk + missing
                    session["state"] = "interrupted"
                    break
                except Exception as error:
                    latency = time.perf_counter() - started
                    generated = _generation_error_records(
                        tasks,
                        sampling,
                        f"{type(error).__name__}: {error}",
                        latency,
                    )
                by_task: dict[str, list[GeneratedCandidate]] = defaultdict(list)
                for candidate in generated:
                    by_task[candidate.task.id].append(candidate)
                for _, task_entry, bound in chunk:
                    task_candidates = sorted(
                        by_task[bound.task.id], key=lambda item: item.candidate_index
                    )
                    if len(task_candidates) != EXPECTED_CANDIDATES_PER_TASK:
                        raise ValueError("vLLM chunk returned an incomplete task")
                    slots = task_entry["candidate_slots"][arm_id]
                    candidate_rows = []
                    for candidate, candidate_id in zip(
                        task_candidates, slots, strict=True
                    ):
                        candidate_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "candidate_index": candidate.candidate_index,
                                "sampling_seed": int(config.generation["seed"]),
                                "raw_continuation": candidate.text,
                                "raw_continuation_sha256": _sha256_text(candidate.text),
                                "token_count": candidate.token_count,
                                "finish_reason": candidate.finish_reason,
                                "generation_latency_seconds": (
                                    candidate.generation_latency_seconds
                                ),
                                "generation_error": candidate.generation_error,
                            }
                        )
                    shard = {
                        "schema_version": GENERATION_SHARD_SCHEMA_VERSION,
                        "manifest_sha256": manifest_sha256,
                        "session_id": session_path.stem,
                        "arm_id": arm_id,
                        "workload_id": task_entry["workload_id"],
                        "task_id": task_entry["task_id"],
                        "task_ordinal": task_entry["ordinal"],
                        "prompt_sha256": task_entry["prompt_sha256"][arm_id],
                        "generation_config_sha256": manifest[
                            "generation_config_sha256"
                        ],
                        "candidates": candidate_rows,
                    }
                    _write_once_json(
                        _generation_shard_path(artifact_root, arm_id, task_entry),
                        shard,
                    )
                chunk_count += 1
                session["completed_chunks"] = chunk_count
                inventory = inventory_generations(
                    manifest, artifact_root, manifest_sha256
                )
                snapshot = _progress_snapshot(
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    inventory=inventory,
                    runtime_identity=runtime,
                )
                _replace_json(
                    artifact_root / "progress" / "generation.json", snapshot
                )
                session["completed_candidates_after_chunk"] = inventory[
                    "completed_candidate_count"
                ]
                _replace_json(session_path, session)
                if max_chunks is not None and chunk_count >= max_chunks:
                    session["state"] = "bounded-stop"
                    break
            else:
                session["state"] = "completed"
            del llm
        except Exception as error:
            session["state"] = "failed"
            session["error"] = f"{type(error).__name__}: {error}"
            session["finished_at_utc"] = _utc_now()
            _replace_json(session_path, session)
            raise
        session["finished_at_utc"] = _utc_now()
        inventory = inventory_generations(manifest, artifact_root, manifest_sha256)
        session["completed_candidates_at_end"] = inventory[
            "completed_candidate_count"
        ]
        _replace_json(session_path, session)
        snapshot = _progress_snapshot(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            inventory=inventory,
            runtime_identity=runtime,
        )
        _replace_json(artifact_root / "progress" / "generation.json", snapshot)
        return snapshot


def verifier_environment_identity(
    config: PromptABConfig,
    workload_id: str,
    lean_project_root: Path,
) -> dict[str, Any]:
    if workload_id not in WORKLOAD_IDS:
        raise ValueError(f"unknown verifier workload: {workload_id}")
    contract = config.verifier["environments"][workload_id]
    root = lean_project_root.resolve()
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision != contract["project_revision"]:
        raise ValueError(
            f"verifier project revision mismatch: expected "
            f"{contract['project_revision']}, got {revision}"
        )
    toolchain = (root / "lean-toolchain").read_text(encoding="utf-8").strip()
    if toolchain != contract["lean_toolchain"]:
        raise ValueError("verifier Lean toolchain differs")
    lake_manifest = _read_json(root / "lake-manifest.json")
    mathlib_entries = [
        package for package in lake_manifest.get("packages", []) if package["name"] == "mathlib"
    ]
    if len(mathlib_entries) != 1:
        raise ValueError("verifier project does not bind exactly one mathlib")
    mathlib_revision = str(mathlib_entries[0]["rev"])
    if mathlib_revision != contract["mathlib_revision"]:
        raise ValueError("verifier mathlib manifest revision differs")
    mathlib_root = root / ".lake" / "packages" / "mathlib"
    actual_mathlib = _git_output(mathlib_root, "rev-parse", "HEAD")
    if actual_mathlib != mathlib_revision:
        raise ValueError("verifier mathlib checkout revision differs")
    return {
        "workload_id": workload_id,
        "project_repository": contract["project_repository"],
        "project_revision": revision,
        "lean_toolchain": toolchain,
        "mathlib_revision": mathlib_revision,
        "timeout_seconds": float(config.verifier["timeout_seconds"]),
        "classification": sorted(RESULT_CATEGORIES),
    }


def verifier_environment_identities(
    config: PromptABConfig,
    lean_project_roots: Mapping[str, Path],
) -> dict[str, Any]:
    if set(lean_project_roots) != set(WORKLOAD_IDS):
        raise ValueError("verifier project roots must bind both issue #86 workloads")
    environments = {
        workload_id: verifier_environment_identity(
            config, workload_id, lean_project_roots[workload_id]
        )
        for workload_id in WORKLOAD_IDS
    }
    environment_sha256_by_workload = {
        workload_id: _sha256_json(environments[workload_id])
        for workload_id in WORKLOAD_IDS
    }
    return {
        "environments": environments,
        "environment_sha256_by_workload": environment_sha256_by_workload,
        "environment_set_sha256": _sha256_json(environments),
    }


def _validate_verification_result(
    value: Mapping[str, Any],
    *,
    manifest_sha256: str,
    environment_sha256: str,
    generation: Mapping[str, Any],
) -> None:
    if (
        value.get("schema_version") != VERIFICATION_RESULT_SCHEMA_VERSION
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("candidate_id") != generation["candidate_id"]
        or value.get("arm_id") != generation["arm_id"]
        or value.get("workload_id") != generation["workload_id"]
        or value.get("task_id") != generation["task_id"]
        or value.get("candidate_index") != generation["candidate_index"]
        or value.get("raw_continuation_sha256")
        != generation["raw_continuation_sha256"]
        or value.get("verifier_environment_sha256") != environment_sha256
        or value.get("category") not in RESULT_CATEGORIES
        or not isinstance(value.get("diagnostics"), dict)
    ):
        raise ValueError(
            f"durable verification result differs: {generation['candidate_id']}"
        )


def inventory_verifications(
    manifest: Mapping[str, Any],
    artifact_root: Path,
    manifest_sha256: str,
    environment_sha256_by_workload: Mapping[str, str],
    generation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    expected = generation_inventory["candidates_by_id"]
    results_by_id: dict[str, dict[str, Any]] = {}
    file_hashes: list[tuple[str, str]] = []
    category_counts: Counter[str] = Counter()
    for candidate_id, generation in expected.items():
        path = _verification_result_path(
            artifact_root,
            str(generation["arm_id"]),
            str(generation["workload_id"]),
            candidate_id,
        )
        if not path.exists():
            continue
        value = _read_json(path)
        _validate_verification_result(
            value,
            manifest_sha256=manifest_sha256,
            environment_sha256=environment_sha256_by_workload[
                str(generation["workload_id"])
            ],
            generation=generation,
        )
        if candidate_id in results_by_id:
            raise ValueError(f"duplicate verification identity: {candidate_id}")
        results_by_id[candidate_id] = value
        category_counts[str(value["category"])] += 1
        file_hashes.append((str(path.relative_to(artifact_root)), sha256_file(path)))
    verification_root = artifact_root / "verifications"
    if verification_root.exists():
        observed_files = {path.resolve() for path in verification_root.rglob("*.json")}
        expected_files = {
            _verification_result_path(
                artifact_root,
                str(generation["arm_id"]),
                str(generation["workload_id"]),
                candidate_id,
            ).resolve()
            for candidate_id, generation in expected.items()
        }
        extras = observed_files - expected_files
        if extras:
            raise ValueError(f"unexpected verification result files: {sorted(extras)}")
    file_hashes.sort()
    return {
        "completed_verification_count": len(results_by_id),
        "results_by_id": results_by_id,
        "category_counts": {
            category: category_counts[category]
            for category in sorted(RESULT_CATEGORIES)
        },
        "result_set_sha256": _sha256_json(file_hashes),
        "file_hashes": file_hashes,
    }


def _verification_progress_snapshot(
    *,
    manifest_sha256: str,
    generation_inventory: Mapping[str, Any],
    verification_inventory: Mapping[str, Any],
    environment_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "mathia-prompt-ab-verification-progress-v1",
        "manifest_sha256": manifest_sha256,
        "updated_at_utc": _utc_now(),
        "generated_candidates": generation_inventory["completed_candidate_count"],
        "verified_candidates": verification_inventory[
            "completed_verification_count"
        ],
        "pending_generated_candidates": (
            generation_inventory["completed_candidate_count"]
            - verification_inventory["completed_verification_count"]
        ),
        "category_counts": verification_inventory["category_counts"],
        "verification_result_set_sha256": verification_inventory[
            "result_set_sha256"
        ],
        "verifier_environments": environment_bundle["environments"],
        "verifier_environment_sha256_by_workload": environment_bundle[
            "environment_sha256_by_workload"
        ],
        "verifier_environment_set_sha256": environment_bundle[
            "environment_set_sha256"
        ],
    }


def run_resumable_verification(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
    manifest_path: Path,
    artifact_root: Path,
    lean_project_roots: Mapping[str, Path],
    *,
    workers: int = 8,
    batch_candidates: int = 64,
    max_batches: int | None = None,
    resume: bool,
) -> dict[str, Any]:
    if not resume:
        raise ValueError("verification requires explicit --resume acknowledgement")
    if workers < 1 or batch_candidates < 1:
        raise ValueError("verification worker and batch counts must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max batches must be positive")
    manifest, bound_tasks = validate_execution_manifest(
        config, dataset_root, mathia_root, repository_root, manifest_path
    )
    manifest_sha256 = sha256_file(manifest_path)
    generation_inventory = inventory_generations(
        manifest, artifact_root, manifest_sha256
    )
    if generation_inventory["completed_candidate_count"] == 0:
        raise ValueError("verification has no durable generation candidates")
    environment_bundle = verifier_environment_identities(config, lean_project_roots)
    environment_sha256_by_workload = environment_bundle[
        "environment_sha256_by_workload"
    ]
    _write_once_json(
        artifact_root / "verifier-environments.json",
        {
            "schema_version": "mathia-prompt-ab-verifier-environments-v1",
            **environment_bundle,
        },
    )
    with _ArtifactLock(
        artifact_root / ".verification.lock",
        operation="verification",
        manifest_sha256=manifest_sha256,
    ):
        verification_inventory = inventory_verifications(
            manifest,
            artifact_root,
            manifest_sha256,
            environment_sha256_by_workload,
            generation_inventory,
        )
        completed = set(verification_inventory["results_by_id"])
        missing = [
            generation
            for candidate_id, generation in sorted(
                generation_inventory["candidates_by_id"].items(),
                key=lambda item: (
                    item[1]["arm_id"],
                    item[1]["task_ordinal"],
                    item[1]["candidate_index"],
                ),
            )
            if candidate_id not in completed
        ]
        if not missing:
            snapshot = _verification_progress_snapshot(
                manifest_sha256=manifest_sha256,
                generation_inventory=generation_inventory,
                verification_inventory=verification_inventory,
                environment_bundle=environment_bundle,
            )
            _replace_json(artifact_root / "progress" / "verification.json", snapshot)
            return snapshot
        session_path = _next_session_path(artifact_root, "verification")
        session = {
            "schema_version": "mathia-prompt-ab-verification-session-v1",
            "manifest_sha256": manifest_sha256,
            "session_id": session_path.stem,
            "state": "running",
            "started_at_utc": _utc_now(),
            "workers": workers,
            "batch_candidates": batch_candidates,
            "completed_batches": 0,
            "verifier_environment_sha256_by_workload": (
                environment_sha256_by_workload
            ),
            "verifier_environment_set_sha256": environment_bundle[
                "environment_set_sha256"
            ],
        }
        _replace_json(session_path, session)
        verifiers = {
            workload_id: LeanVerifier(
                lean_project_roots[workload_id],
                timeout_seconds=float(config.verifier["timeout_seconds"]),
            )
            for workload_id in WORKLOAD_IDS
        }
        tasks_by_ordinal = {bound.ordinal: bound.task for bound in bound_tasks}
        batch_count = 0
        try:
            while missing:
                if (artifact_root / "PAUSE").exists():
                    session["state"] = "paused"
                    break
                batch = missing[:batch_candidates]
                missing = missing[batch_candidates:]

                def verify_one(generation: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
                    task = tasks_by_ordinal[int(generation["task_ordinal"])]
                    workload_id = str(generation["workload_id"])
                    generated = GeneratedCandidate(
                        task=task,
                        candidate_index=int(generation["candidate_index"]),
                        text=str(generation["raw_continuation"]),
                        token_count=int(generation["token_count"]),
                        finish_reason=str(generation["finish_reason"]),
                        generation_latency_seconds=float(
                            generation["generation_latency_seconds"]
                        ),
                        generation_error=generation.get("generation_error"),
                    )
                    return dict(generation), _verify_candidate(
                        verifiers[workload_id], generated
                    )

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(verify_one, item) for item in batch]
                    for future in as_completed(futures):
                        generation, outcome = future.result()
                        value = {
                            "schema_version": VERIFICATION_RESULT_SCHEMA_VERSION,
                            "manifest_sha256": manifest_sha256,
                            "session_id": session_path.stem,
                            "candidate_id": generation["candidate_id"],
                            "arm_id": generation["arm_id"],
                            "workload_id": generation["workload_id"],
                            "task_id": generation["task_id"],
                            "candidate_index": generation["candidate_index"],
                            "raw_continuation_sha256": generation[
                                "raw_continuation_sha256"
                            ],
                            "verifier_environment_sha256": (
                                environment_sha256_by_workload[
                                    str(generation["workload_id"])
                                ]
                            ),
                            "category": outcome.category,
                            "lean_exit_code": outcome.lean_exit_code,
                            "diagnostics": outcome.diagnostics,
                            "verification_latency_seconds": (
                                outcome.verification_latency_seconds
                            ),
                            "total_latency_seconds": outcome.total_latency_seconds,
                        }
                        _write_once_json(
                            _verification_result_path(
                                artifact_root,
                                str(generation["arm_id"]),
                                str(generation["workload_id"]),
                                str(generation["candidate_id"]),
                            ),
                            value,
                        )
                batch_count += 1
                session["completed_batches"] = batch_count
                verification_inventory = inventory_verifications(
                    manifest,
                    artifact_root,
                    manifest_sha256,
                    environment_sha256_by_workload,
                    generation_inventory,
                )
                snapshot = _verification_progress_snapshot(
                    manifest_sha256=manifest_sha256,
                    generation_inventory=generation_inventory,
                    verification_inventory=verification_inventory,
                    environment_bundle=environment_bundle,
                )
                _replace_json(
                    artifact_root / "progress" / "verification.json", snapshot
                )
                session["completed_verifications_after_batch"] = (
                    verification_inventory["completed_verification_count"]
                )
                _replace_json(session_path, session)
                if max_batches is not None and batch_count >= max_batches:
                    session["state"] = "bounded-stop"
                    break
            else:
                session["state"] = "completed"
        except KeyboardInterrupt:
            session["state"] = "interrupted"
        except Exception as error:
            session["state"] = "failed"
            session["error"] = f"{type(error).__name__}: {error}"
            session["finished_at_utc"] = _utc_now()
            _replace_json(session_path, session)
            raise
        session["finished_at_utc"] = _utc_now()
        verification_inventory = inventory_verifications(
            manifest,
            artifact_root,
            manifest_sha256,
            environment_sha256_by_workload,
            generation_inventory,
        )
        session["completed_verifications_at_end"] = verification_inventory[
            "completed_verification_count"
        ]
        _replace_json(session_path, session)
        snapshot = _verification_progress_snapshot(
            manifest_sha256=manifest_sha256,
            generation_inventory=generation_inventory,
            verification_inventory=verification_inventory,
            environment_bundle=environment_bundle,
        )
        _replace_json(artifact_root / "progress" / "verification.json", snapshot)
        return snapshot


def _exact_two_sided_mcnemar(candidate_only: int, reference_only: int) -> float:
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    lower = min(candidate_only, reference_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _pass_metrics(counts: Sequence[int]) -> dict[str, float]:
    return {
        f"pass@{k}": fmean(
            pass_at_k(EXPECTED_CANDIDATES_PER_TASK, count, k) for count in counts
        )
        for k in (1, 4, 8)
    }


def _paired_comparison(
    reference: Sequence[int],
    candidate: Sequence[int],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("paired comparison requires equal non-empty task populations")
    if any(not 0 <= count <= 8 for count in (*reference, *candidate)):
        raise ValueError("paired verified counts must be within 0..8")
    both = sum(a > 0 and b > 0 for a, b in zip(reference, candidate, strict=True))
    candidate_only = sum(
        a == 0 and b > 0 for a, b in zip(reference, candidate, strict=True)
    )
    reference_only = sum(
        a > 0 and b == 0 for a, b in zip(reference, candidate, strict=True)
    )
    neither = len(reference) - both - candidate_only - reference_only
    return {
        "task_count": len(reference),
        "paired_outcomes": {
            "both_solved": both,
            "candidate_only": candidate_only,
            "reference_only": reference_only,
            "neither_solved": neither,
            "paired_solved_delta_count": candidate_only - reference_only,
            "paired_solved_delta_fraction": (
                candidate_only - reference_only
            )
            / len(reference),
        },
        "exact_two_sided_mcnemar_p": _exact_two_sided_mcnemar(
            candidate_only, reference_only
        ),
        "pass_at_k_delta_candidate_minus_reference": {
            key: _pass_metrics(candidate)[key] - _pass_metrics(reference)[key]
            for key in ("pass@1", "pass@4", "pass@8")
        },
        "paired_bootstrap": paired_task_bootstrap(
            reference,
            candidate,
            candidates_per_task=EXPECTED_CANDIDATES_PER_TASK,
            ks=(1, 4, 8),
            resamples=resamples,
            seed=seed,
        ),
    }


def _session_inventory(artifact_root: Path, operation: str) -> dict[str, Any]:
    rows = []
    for path in sorted((artifact_root / "sessions").glob(f"{operation}-*.json")):
        value = _read_json(path)
        rows.append(
            {
                "session_id": value.get("session_id", path.stem),
                "state": value.get("state"),
                "started_at_utc": value.get("started_at_utc"),
                "finished_at_utc": value.get("finished_at_utc"),
                "completed_chunks": value.get("completed_chunks"),
                "completed_batches": value.get("completed_batches"),
                "runtime_identity": value.get("runtime_identity"),
                "file_sha256": sha256_file(path),
            }
        )
    return {
        "session_count": len(rows),
        "restart_count": max(0, len(rows) - 1),
        "sessions": rows,
    }


def _arm_workload_summary(
    *,
    arm_id: str,
    task_entries: Sequence[Mapping[str, Any]],
    generation_by_id: Mapping[str, Mapping[str, Any]],
    verification_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[int]]:
    verified_counts: list[int] = []
    solved_within = {1: 0, 4: 0, 8: 0}
    category_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    generated_tokens: list[int] = []
    for task in task_entries:
        verified_indices = []
        for candidate_index, candidate_id_value in enumerate(
            task["candidate_slots"][arm_id]
        ):
            candidate_id = str(candidate_id_value)
            generation = generation_by_id[candidate_id]
            verification = verification_by_id[candidate_id]
            category_counts[str(verification["category"])] += 1
            finish_counts[str(generation["finish_reason"])] += 1
            generated_tokens.append(int(generation["token_count"]))
            if verification["category"] == "verified":
                verified_indices.append(candidate_index)
        verified_counts.append(len(verified_indices))
        for k in solved_within:
            solved_within[k] += any(index < k for index in verified_indices)
    candidate_count = len(task_entries) * EXPECTED_CANDIDATES_PER_TASK
    verified_candidate_count = sum(verified_counts)
    return (
        {
            "task_count": len(task_entries),
            "candidate_count": candidate_count,
            "pass_at_k": _pass_metrics(verified_counts),
            "tasks_solved_within_k": {
                f"solved@{k}": solved_within[k] for k in solved_within
            },
            "verified_candidate_count": verified_candidate_count,
            "verified_candidate_rate": verified_candidate_count / candidate_count,
            "category_counts": {
                category: category_counts[category]
                for category in sorted(RESULT_CATEGORIES)
            },
            "finish_reason_counts": dict(sorted(finish_counts.items())),
            "generated_token_count": {
                "total": sum(generated_tokens),
                "minimum": min(generated_tokens),
                "maximum": max(generated_tokens),
                "mean": fmean(generated_tokens),
            },
            "verified_counts": verified_counts,
        },
        verified_counts,
    )


def _recommendation(comparison: Mapping[str, Any]) -> dict[str, Any]:
    paired = comparison["combined"]["A_vs_B"]
    outcomes = paired["paired_outcomes"]
    p_value = float(paired["exact_two_sided_mcnemar_p"])
    if p_value < 0.05 and outcomes["candidate_only"] > outcomes["reference_only"]:
        result = "adopt-explicit-proof-task-interface"
        text = (
            "Arm B produced a statistically clear paired solved@8 advantage over "
            "Arm A; adopt the frozen explicit proof-task wording as the default "
            "qwen-lean intuition interface."
        )
    elif p_value < 0.05 and outcomes["reference_only"] > outcomes["candidate_only"]:
        result = "retain-raw-intuition-interface"
        text = (
            "Arm A produced a statistically clear paired solved@8 advantage over "
            "Arm B; retain the minimal raw-intuition interface."
        )
    else:
        result = "no-clear-ab-advantage-retain-minimal-interface"
        text = (
            "The paired A/B result does not establish a statistically clear "
            "solved@8 advantage for the explicit instruction. Retain the minimal "
            "raw-intuition interface by default and report the point estimates."
        )
    return {
        "decision_marker": "OBSERVED",
        "result": result,
        "text": text,
        "automatic_training_contract_change_authorized": False,
    }


def compact_results(
    config: PromptABConfig,
    dataset_root: Path,
    mathia_root: Path,
    repository_root: Path,
    manifest_path: Path,
    artifact_root: Path,
    lean_project_roots: Mapping[str, Path],
    output_path: Path,
    readme_path: Path | None = None,
) -> dict[str, Any]:
    manifest, _ = validate_execution_manifest(
        config, dataset_root, mathia_root, repository_root, manifest_path
    )
    manifest_sha256 = sha256_file(manifest_path)
    generation_inventory = inventory_generations(
        manifest, artifact_root, manifest_sha256
    )
    if generation_inventory["completed_candidate_count"] != EXPECTED_CANDIDATES_TOTAL:
        raise ValueError("cannot score before all 9,776 generation slots are durable")
    environment_bundle = verifier_environment_identities(config, lean_project_roots)
    environment_sha256_by_workload = environment_bundle[
        "environment_sha256_by_workload"
    ]
    verification_inventory = inventory_verifications(
        manifest,
        artifact_root,
        manifest_sha256,
        environment_sha256_by_workload,
        generation_inventory,
    )
    if verification_inventory["completed_verification_count"] != EXPECTED_CANDIDATES_TOTAL:
        raise ValueError("cannot score before all 9,776 classifications are durable")
    generation_by_id = generation_inventory["candidates_by_id"]
    verification_by_id = verification_inventory["results_by_id"]
    resamples = int(config.value["analysis"]["bootstrap_resamples"])
    seed = int(config.value["analysis"]["bootstrap_seed"])
    task_groups: dict[str, list[dict[str, Any]]] = {
        workload_id: [
            task for task in manifest["tasks"] if task["workload_id"] == workload_id
        ]
        for workload_id in WORKLOAD_IDS
    }
    task_groups["combined"] = list(manifest["tasks"])
    analyses: dict[str, Any] = {}
    for workload_id, tasks in task_groups.items():
        arms: dict[str, Any] = {}
        counts: dict[str, list[int]] = {}
        for arm_id in ARM_IDS:
            summary, verified_counts = _arm_workload_summary(
                arm_id=arm_id,
                task_entries=tasks,
                generation_by_id=generation_by_id,
                verification_by_id=verification_by_id,
            )
            arms[arm_id] = summary
            counts[arm_id] = verified_counts
        q0_counts = [int(task["q0_verified_candidate_count"]) for task in tasks]
        q0 = {
            "task_count": len(tasks),
            "pass_at_k": _pass_metrics(q0_counts),
            "tasks_solved_within_8": sum(count > 0 for count in q0_counts),
            "verified_counts": q0_counts,
            "regenerated": False,
        }
        analyses[workload_id] = {
            "arms": arms,
            "q0_reference": q0,
            "A_vs_B": _paired_comparison(
                counts["A"], counts["B"], resamples=resamples, seed=seed
            ),
            "Q0_vs_A": _paired_comparison(
                q0_counts, counts["A"], resamples=resamples, seed=seed
            ),
            "Q0_vs_B": _paired_comparison(
                q0_counts, counts["B"], resamples=resamples, seed=seed
            ),
        }
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "decision_marker": "OBSERVED",
        "manifest_sha256": manifest_sha256,
        "config_sha256": manifest["config_sha256"],
        "model": config.model,
        "generation_contract": manifest["generation_contract"],
        "verifier_environments": environment_bundle["environments"],
        "verifier_environment_sha256_by_workload": (
            environment_sha256_by_workload
        ),
        "verifier_environment_set_sha256": environment_bundle[
            "environment_set_sha256"
        ],
        "completion_integrity_gate": {
            "passed": True,
            "candidate_identities_per_arm": {
                "A": EXPECTED_CANDIDATES_PER_ARM,
                "B": EXPECTED_CANDIDATES_PER_ARM,
            },
            "durable_generation_results": generation_inventory[
                "completed_candidate_count"
            ],
            "durable_verification_results": verification_inventory[
                "completed_verification_count"
            ],
            "generation_failure_count": generation_inventory[
                "generation_failure_count"
            ],
            "duplicate_candidate_count": 0,
            "omitted_candidate_count": 0,
            "scoring_order": "immutable-manifest-order",
            "restart_or_shard_history_used_as_exclusion": False,
            "metrics_invariant_to_restart_history_by_construction": True,
        },
        "raw_artifact_binding": {
            "artifact_root": str(artifact_root.resolve()),
            "generation_shard_count": len(generation_inventory["shards"]),
            "generation_shards": generation_inventory["shards"],
            "verification_result_count": verification_inventory[
                "completed_verification_count"
            ],
            "verification_result_set_sha256": verification_inventory[
                "result_set_sha256"
            ],
        },
        "process_history": {
            "generation": _session_inventory(artifact_root, "generation"),
            "verification": _session_inventory(artifact_root, "verification"),
        },
        "workloads": {
            workload_id: analyses[workload_id] for workload_id in WORKLOAD_IDS
        },
        "combined": analyses["combined"],
    }
    results["recommendation"] = _recommendation(results)
    _replace_json(output_path, results)
    if readme_path is not None:
        _atomic_write(readme_path, render_results_readme(results).encode("utf-8"))
    return results


def render_results_readme(results: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3.5-4B-Base Mathia prompt A/B",
        "",
        "**OBSERVED:** This artifact reports issue #86 without regenerating Q0. "
        + str(results["recommendation"]["text"]),
        "",
        "| workload | arm | solved@8 | verified candidates | pass@1 | pass@4 | pass@8 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for workload_id in (*WORKLOAD_IDS, "combined"):
        lane = (
            results["combined"]
            if workload_id == "combined"
            else results["workloads"][workload_id]
        )
        for arm_id in ARM_IDS:
            arm = lane["arms"][arm_id]
            metrics = arm["pass_at_k"]
            lines.append(
                f"| {workload_id} | {arm_id} | "
                f"{arm['tasks_solved_within_k']['solved@8']}/{arm['task_count']} | "
                f"{arm['verified_candidate_count']}/{arm['candidate_count']} | "
                f"{metrics['pass@1']:.6f} | {metrics['pass@4']:.6f} | "
                f"{metrics['pass@8']:.6f} |"
            )
    paired = results["combined"]["A_vs_B"]
    outcomes = paired["paired_outcomes"]
    lines.extend(
        [
            "",
            "## Paired combined result",
            "",
            f"A-only/B-only/both/neither solved@8: "
            f"{outcomes['reference_only']}/{outcomes['candidate_only']}/"
            f"{outcomes['both_solved']}/{outcomes['neither_solved']}. Exact "
            f"two-sided McNemar p={paired['exact_two_sided_mcnemar_p']:.6g}.",
            "",
            "Q0 is the unchanged authoritative Dataset-v2 Base evidence from issue "
            "#78 restricted to the same 611 tasks. Raw generations and Lean outcomes "
            "remain in the bound outside-Git artifact root; the committed JSON binds "
            "their atomic shard/result hashes and restart history.",
            "",
            "No model was trained, no Q0 candidate was regenerated, and this result "
            "does not automatically change the training contract.",
        ]
    )
    return "\n".join(lines) + "\n"


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=Path("config/mathia-prompt-ab.json")
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mathia-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)


def _verifier_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--minif2f-project-root", type=Path, required=True)
    parser.add_argument("--fresh-project-root", type=Path, required=True)


def _verifier_project_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "minif2f-valid-clean-v2": args.minif2f_project_root,
        "fresh-composition-valid-v2": args.fresh_project_root,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable issue #86 Mathia prompt A/B execution"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    _common_paths(materialize)

    generate = subparsers.add_parser("generate")
    _common_paths(generate)
    generate.add_argument("--artifacts", type=Path, required=True)
    generate.add_argument("--arm", choices=("A", "B", "both"), default="both")
    generate.add_argument("--chunk-tasks", type=int, default=4)
    generate.add_argument("--max-chunks", type=int)
    generate.add_argument("--resume", action="store_true")

    verify = subparsers.add_parser("verify")
    _common_paths(verify)
    verify.add_argument("--artifacts", type=Path, required=True)
    _verifier_paths(verify)
    verify.add_argument("--workers", type=int, default=8)
    verify.add_argument("--batch-candidates", type=int, default=64)
    verify.add_argument("--max-batches", type=int)
    verify.add_argument("--resume", action="store_true")

    compact = subparsers.add_parser("compact")
    _common_paths(compact)
    compact.add_argument("--artifacts", type=Path, required=True)
    _verifier_paths(compact)
    compact.add_argument("--output", type=Path, required=True)
    compact.add_argument("--readme", type=Path)

    pause = subparsers.add_parser("pause")
    pause.add_argument("--artifacts", type=Path, required=True)
    pause.add_argument("--reason")
    unpause = subparsers.add_parser("unpause")
    unpause.add_argument("--artifacts", type=Path, required=True)
    return parser


def _cli_summary(command: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if command == "materialize":
        return {
            "schema_version": value["schema_version"],
            "task_count": value["task_count"],
            "task_counts": value["task_counts"],
            "prospective_candidate_counts": value["prospective_candidate_counts"],
            "ordered_task_ids_sha256": value["ordered_task_ids_sha256"],
            "generation_config_sha256": value["generation_config_sha256"],
            "prompt_integrity_gate": value["prompt_integrity_gate"],
        }
    if command == "generate":
        return {
            "combined": value["combined"],
            "arms": value["arms"],
            "generation_failure_count": value["generation_failure_count"],
            "cumulative_output_token_count": value[
                "cumulative_output_token_count"
            ],
            "runtime_identity": value["runtime_identity"],
        }
    if command == "verify":
        return {
            key: value[key]
            for key in (
                "generated_candidates",
                "verified_candidates",
                "pending_generated_candidates",
                "category_counts",
                "verification_result_set_sha256",
                "verifier_environment_sha256_by_workload",
                "verifier_environment_set_sha256",
            )
        }
    if command == "compact":
        return {
            "schema_version": value["schema_version"],
            "completion_integrity_gate": value["completion_integrity_gate"],
            "recommendation": value["recommendation"],
        }
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pause":
        path = set_pause(args.artifacts, reason=args.reason)
        print(json.dumps({"pause_marker": str(path)}, sort_keys=True))
        return 0
    if args.command == "unpause":
        clear_pause(args.artifacts)
        print(json.dumps({"paused": False}, sort_keys=True))
        return 0
    config = PromptABConfig.load(args.config)
    if args.command == "materialize":
        value = materialize_execution_manifest(
            config,
            args.dataset_root,
            args.mathia_root,
            args.repository_root,
            args.manifest,
        )
    elif args.command == "generate":
        arms = ARM_IDS if args.arm == "both" else (args.arm,)
        value = run_resumable_generation(
            config,
            args.dataset_root,
            args.mathia_root,
            args.repository_root,
            args.manifest,
            args.artifacts,
            arms=arms,
            chunk_tasks=args.chunk_tasks,
            max_chunks=args.max_chunks,
            resume=args.resume,
        )
    elif args.command == "verify":
        value = run_resumable_verification(
            config,
            args.dataset_root,
            args.mathia_root,
            args.repository_root,
            args.manifest,
            args.artifacts,
            _verifier_project_roots(args),
            workers=args.workers,
            batch_candidates=args.batch_candidates,
            max_batches=args.max_batches,
            resume=args.resume,
        )
    elif args.command == "compact":
        value = compact_results(
            config,
            args.dataset_root,
            args.mathia_root,
            args.repository_root,
            args.manifest,
            args.artifacts,
            _verifier_project_roots(args),
            args.output,
            args.readme,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(_cli_summary(args.command, value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
