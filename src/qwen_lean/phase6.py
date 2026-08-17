from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .metrics import pass_at_k, summarize_results
from .minif2f import Phase1Config
from .phase2_corpus import read_jsonl_records
from .phase2_schema import MathlibProofRecord
from .phase3 import BASE_MODEL_ID, BASE_REVISION, render_sft_prompt
from .phase5 import ordered_record_ids_sha256
from .prompt import normalize_transport
from .schema import CandidateResult

PHASE6_CONFIG_SCHEMA_VERSION = "phase6-config-v1"
PHASE6_WORKLOAD_SCHEMA_VERSION = "phase6-train-workload-v1"
PHASE6_CANDIDATE_SCHEMA_VERSION = "phase6-reference-candidate-v1"
TRAIN_WORKLOAD_ID = "phase6-train512-v1"
REFERENCE_SFT_ID = "reference-sft-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Phase6Config:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Phase6Config:
        value = _read_json(path)
        config = cls(path=path.resolve(), value=value)
        config.validate()
        return config

    @property
    def model(self) -> dict[str, Any]:
        return self.value["model"]

    @property
    def reference(self) -> dict[str, Any]:
        return self.value["reference_candidate"]

    @property
    def adapter(self) -> dict[str, Any]:
        return self.reference["adapter"]

    @property
    def train_workload(self) -> dict[str, Any]:
        return self.value["train_workload"]

    @property
    def train_generation(self) -> dict[str, Any]:
        return self.value["train_generation"]

    def phase1_test_config(self) -> Phase1Config:
        project_root = self.path.parents[1]
        return Phase1Config.load(
            project_root / str(self.value["minif2f_test"]["config"])
        )

    def validate(self) -> None:
        if self.value.get("schema_version") != PHASE6_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unknown Phase 6 config schema: {self.value.get('schema_version')}"
            )
        expected: tuple[tuple[tuple[str, ...], Any], ...] = (
            (("model", "model_id"), BASE_MODEL_ID),
            (("model", "model_revision"), BASE_REVISION),
            (("model", "tokenizer_id"), BASE_MODEL_ID),
            (("model", "tokenizer_revision"), BASE_REVISION),
            (("model", "add_special_tokens"), False),
            (("model", "chat_template"), None),
            (("reference_candidate", "logical_id"), REFERENCE_SFT_ID),
            (("reference_candidate", "selection_predates_phase6_outputs"), True),
            (("reference_candidate", "eligible_candidate_count"), 1),
            (("reference_candidate", "ineligible_optimizer_steps"), [2491, 4981, 7472]),
            (("reference_candidate", "phase6_metrics_may_change_identity"), False),
            (
                ("reference_candidate", "adapter", "artifact_id"),
                "phase5-train-full-v1-lora",
            ),
            (("reference_candidate", "adapter", "format"), "peft-lora"),
            (("reference_candidate", "adapter", "merged"), False),
            (("reference_candidate", "adapter", "selected_optimizer_step"), 9962),
            (
                ("reference_candidate", "adapter", "training_artifact_sha256"),
                "48d33bc2f276d6f8c22525a5cb30fafe8677da95e866dbf3f37116e78e8ae990",
            ),
            (
                ("reference_candidate", "adapter", "hub_repository"),
                "murillo2000/qwen3-8b-base-lean-sft-qlora",
            ),
            (
                ("reference_candidate", "adapter", "hub_revision"),
                "5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5",
            ),
            (
                ("reference_candidate", "adapter", "adapter_model_sha256"),
                "8aa50fa56f6a1d03a702abcaafc20e11d661a4a2ac935864bf5648411e5cdc58",
            ),
            (
                ("reference_candidate", "adapter", "adapter_config_sha256"),
                "4b7b513b216484554e05d3c75ecf0777ee1fbae94935e93d949d63cf4a76481c",
            ),
            (("reference_candidate", "adapter", "rank"), 16),
            (("reference_candidate", "adapter", "alpha"), 32),
            (("reference_candidate", "adapter", "dropout"), 0.0),
            (("phase5_inputs", "train_workload_id"), "phase5-train-full-v1"),
            (("phase5_inputs", "train_input_examples"), 80062),
            (("phase5_inputs", "train_eligible_examples"), 79696),
            (
                ("phase5_inputs", "train_ordered_ids_sha256"),
                "0ec5ef7e969774924384d80d04b5ea9ea6e0eabac9f38cf3deff9924c714d816",
            ),
            (("phase5_inputs", "heldout_workload_id"), "phase5-heldout512-v1"),
            (
                ("phase5_inputs", "heldout_ordered_ids_sha256"),
                "936091af02d7c58dfe172c10b91376afd1e144cb1586a1f07a0a924f8e0b194b",
            ),
            (("phase5_inputs", "minif2f_validation_workload_id"), "minif2f-valid-v1"),
            (("train_workload", "id"), TRAIN_WORKLOAD_ID),
            (("train_workload", "selection_hash_prefix"), f"{TRAIN_WORKLOAD_ID}\0"),
            (("train_workload", "expected_examples"), 512),
            (("train_workload", "maximum_prompt_and_generation_tokens"), 2048),
            (("train_generation", "candidates_per_task"), 4),
            (("train_generation", "do_sample"), True),
            (("train_generation", "temperature"), 0.8),
            (("train_generation", "top_p"), 0.95),
            (("train_generation", "top_k"), -1),
            (("train_generation", "max_new_tokens"), 1024),
            (("train_generation", "stop"), "tokenizer_eos_or_token_limit"),
            (("train_generation", "seed"), 0),
            (("minif2f_test", "workload_id"), "minif2f-test-v1"),
            (("minif2f_test", "expected_tasks"), 244),
            (("minif2f_test", "candidates_per_task"), 8),
            (("bootstrap", "resamples"), 10000),
            (("bootstrap", "seed"), 0),
            (("bootstrap", "interval_percentiles"), [2.5, 97.5]),
        )
        for path, wanted in expected:
            observed: Any = self.value
            for key in path:
                observed = observed[key]
            if observed != wanted:
                raise ValueError(
                    f"Phase 6 {'.'.join(path)} must be {wanted!r}, got {observed!r}"
                )
        if int(self.value["verification"]["workers"]) < 1:
            raise ValueError("Phase 6 verification workers must be positive")
        for key in ("train_timeout_seconds", "minif2f_timeout_seconds"):
            if float(self.value["verification"][key]) <= 0:
                raise ValueError(f"Phase 6 {key} must be positive")

        test = self.phase1_test_config()
        expected_test_model = {
            key: self.model[key]
            for key in (
                "model_id",
                "model_revision",
                "tokenizer_id",
                "tokenizer_revision",
            )
        }
        if test.model != expected_test_model:
            raise ValueError("Phase 6 miniF2F test model identity differs")
        if test.sampling != {
            "candidates_per_task": 8,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": -1,
            "max_new_tokens": 1024,
            "stop": "tokenizer_eos_or_token_limit",
            "seed": 0,
        }:
            raise ValueError("Phase 6 miniF2F test sampling differs from Phase 1")
        if (
            test.benchmark["source_path"] != "MiniF2F/Test.lean"
            or test.benchmark["split"] != "test"
            or int(test.benchmark["expected_primary_task_count"]) != 244
        ):
            raise ValueError("Phase 6 miniF2F test source contract differs")


@dataclass(frozen=True)
class Phase6TrainExample:
    record_id: str
    declaration_name: str
    prompt: str
    prompt_tokens: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Phase6TrainExample:
        return cls(
            record_id=str(value["record_id"]),
            declaration_name=str(value["declaration_name"]),
            prompt=str(value["prompt"]),
            prompt_tokens=int(value["prompt_tokens"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_adapter_artifact_files(
    adapter_dir: Path, *, expected_model_sha256: str, expected_config_sha256: str
) -> dict[str, Any]:
    root = adapter_dir.resolve()
    model_path = root / "adapter_model.safetensors"
    config_path = root / "adapter_config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise ValueError("Phase 6 adapter is missing PEFT weights or configuration")
    observed = {
        "adapter_model_sha256": _sha256(model_path),
        "adapter_config_sha256": _sha256(config_path),
    }
    if observed["adapter_model_sha256"] != expected_model_sha256:
        raise ValueError("Phase 6 adapter weights differ from the published artifact")
    if observed["adapter_config_sha256"] != expected_config_sha256:
        raise ValueError(
            "Phase 6 adapter configuration differs from the published artifact"
        )
    return observed


def freeze_reference_candidate(
    config: Phase6Config,
    adapter_dir: Path,
    phase5_training_evidence: Path,
    output: Path,
) -> dict[str, Any]:
    training = _read_json(phase5_training_evidence)
    selection = training.get("checkpoint_selection", {})
    binding = training.get("selected_adapter_binding", {})
    expected_binding = {
        "artifact_id": config.adapter["artifact_id"],
        "format": config.adapter["format"],
        "merged": config.adapter["merged"],
        "selected_optimizer_step": config.adapter["selected_optimizer_step"],
        "training_artifact_sha256": config.adapter["training_artifact_sha256"],
        "training_relative_path": "trainer-state/checkpoint-9962",
    }
    if binding != expected_binding:
        raise ValueError("Phase 5 selected adapter binding differs from Phase 6")
    if (
        training.get("status") != "passed"
        or training.get("model") != config.model
        or selection.get("selected_optimizer_step") != 9962
        or selection.get("candidate_steps") != [2491, 4981, 7472, 9962]
        or selection.get("heldout_or_minif2f_consulted") is not False
    ):
        raise ValueError("Phase 5 candidate selection evidence differs from Phase 6")

    hashes = validate_adapter_artifact_files(
        adapter_dir,
        expected_model_sha256=str(config.adapter["adapter_model_sha256"]),
        expected_config_sha256=str(config.adapter["adapter_config_sha256"]),
    )
    peft = _read_json(adapter_dir / "adapter_config.json")
    if (
        peft.get("base_model_name_or_path") != BASE_MODEL_ID
        or peft.get("revision") != BASE_REVISION
        or peft.get("peft_type") != "LORA"
        or peft.get("task_type") != "CAUSAL_LM"
        or int(peft.get("r", -1)) != int(config.adapter["rank"])
        or int(peft.get("lora_alpha", -1)) != int(config.adapter["alpha"])
        or float(peft.get("lora_dropout", -1)) != float(config.adapter["dropout"])
    ):
        raise ValueError("Phase 6 PEFT configuration differs from the frozen identity")

    resolved = adapter_dir.resolve()
    source_kind = (
        "immutable_hub_snapshot"
        if config.adapter["hub_revision"] in resolved.parts
        else "content-hash-proven-local-equivalent"
    )
    manifest = {
        "schema_version": PHASE6_CANDIDATE_SCHEMA_VERSION,
        "status": "frozen",
        "logical_id": REFERENCE_SFT_ID,
        "model": config.model,
        "adapter": {
            **config.adapter,
            **hashes,
            "local_source_kind": source_kind,
            "local_path_retained_outside_git": True,
        },
        "phase5_selected_adapter_binding": expected_binding,
        "selection": {
            "metric": selection["metric"],
            "rule": selection["rule"],
            "selected_before_phase6_train_generation": True,
            "selected_before_minif2f_test": True,
            "phase6_metrics_may_change_identity": False,
            "ineligible_optimizer_steps": [2491, 4981, 7472],
        },
        "candidate_set_size": 1,
    }
    _write_json(output, manifest)
    return manifest


def load_reference_candidate(
    config: Phase6Config, manifest_path: Path, adapter_dir: Path
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != PHASE6_CANDIDATE_SCHEMA_VERSION
        or manifest.get("status") != "frozen"
        or manifest.get("logical_id") != REFERENCE_SFT_ID
        or manifest.get("model") != config.model
        or int(manifest.get("candidate_set_size", -1)) != 1
    ):
        raise ValueError("Phase 6 reference candidate manifest differs")
    adapter = manifest.get("adapter", {})
    for key, expected in config.adapter.items():
        if adapter.get(key) != expected:
            raise ValueError(f"Phase 6 candidate adapter {key} differs")
    selection = manifest.get("selection", {})
    if (
        selection.get("selected_before_phase6_train_generation") is not True
        or selection.get("selected_before_minif2f_test") is not True
        or selection.get("phase6_metrics_may_change_identity") is not False
        or selection.get("ineligible_optimizer_steps") != [2491, 4981, 7472]
    ):
        raise ValueError("Phase 6 frozen-selection guarantee differs")
    validate_adapter_artifact_files(
        adapter_dir,
        expected_model_sha256=str(config.adapter["adapter_model_sha256"]),
        expected_config_sha256=str(config.adapter["adapter_config_sha256"]),
    )
    return manifest


def select_phase6_train_workload(
    records: Iterable[MathlibProofRecord],
    tokenizer: Any,
    *,
    phase5_ordered_member_ids: Sequence[str],
    expected_examples: int = 512,
    generation_tokens: int = 1024,
    maximum_tokens: int = 2048,
    selection_prefix: str = f"{TRAIN_WORKLOAD_ID}\0",
) -> tuple[list[Phase6TrainExample], int]:
    member_ids = [str(item) for item in phase5_ordered_member_ids]
    if len(member_ids) != len(set(member_ids)):
        raise ValueError("Phase 5 training membership contains duplicate record IDs")
    members = set(member_ids)
    records_by_id: dict[str, MathlibProofRecord] = {}
    for record in records:
        if record.id not in members:
            continue
        if record.split != "train":
            raise ValueError(f"Phase 6 member {record.id} is not in the train split")
        if record.id in records_by_id:
            raise ValueError(f"duplicate Phase 6 train record: {record.id}")
        records_by_id[record.id] = record
    missing = members - records_by_id.keys()
    if missing:
        raise ValueError(
            "Phase 6 train members are missing from Phase 2: "
            + ", ".join(sorted(missing)[:10])
        )

    eligible: list[tuple[bytes, Phase6TrainExample]] = []
    for record_id in member_ids:
        record = records_by_id[record_id]
        prompt = render_sft_prompt(record)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens + generation_tokens > maximum_tokens:
            continue
        eligible.append(
            (
                hashlib.sha256(
                    selection_prefix.encode("utf-8") + record.id.encode("utf-8")
                ).digest(),
                Phase6TrainExample(
                    record_id=record.id,
                    declaration_name=record.declaration_name,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                ),
            )
        )
    eligible.sort(key=lambda item: item[0])
    if len(eligible) < expected_examples:
        raise ValueError(
            f"{TRAIN_WORKLOAD_ID} requires {expected_examples} examples; "
            f"only {len(eligible)} exact Phase 5 members are eligible"
        )
    return [item for _, item in eligible[:expected_examples]], len(eligible)


def reconstruct_phase5_train_membership(
    config: Phase6Config,
    dataset_dir: Path,
    phase5_workload_evidence: Path,
) -> tuple[list[MathlibProofRecord], list[str], dict[str, Any]]:
    evidence = _read_json(phase5_workload_evidence)
    train = evidence.get("workloads", {}).get("train", {})
    expected = config.value["phase5_inputs"]
    if (
        evidence.get("cross_split_record_ids_disjoint") is not True
        or train.get("id") != expected["train_workload_id"]
        or int(train.get("input_examples", -1)) != int(expected["train_input_examples"])
        or int(train.get("eligible_examples", -1))
        != int(expected["train_eligible_examples"])
        or train.get("selected_record_ids_sha256")
        != expected["train_ordered_ids_sha256"]
    ):
        raise ValueError("Phase 5 compact training membership evidence differs")
    excluded = {str(item["record_id"]) for item in train["overlength_records"]}
    if len(excluded) != int(train["overlength_examples"]):
        raise ValueError("Phase 5 over-length exclusions contain duplicate IDs")
    records = list(read_jsonl_records(dataset_dir / "train.jsonl"))
    if len(records) != int(expected["train_input_examples"]):
        raise ValueError("Phase 2 train input count differs from Phase 5 evidence")
    ordered_ids = [record.id for record in records if record.id not in excluded]
    if (
        len(ordered_ids) != int(expected["train_eligible_examples"])
        or ordered_record_ids_sha256(ordered_ids)
        != expected["train_ordered_ids_sha256"]
    ):
        raise ValueError("reconstructed Phase 5 ordered train membership differs")
    return records, ordered_ids, train


def materialize_phase6_train_workload(
    config: Phase6Config,
    dataset_dir: Path,
    phase5_workload_evidence: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    records, membership, phase5_train = reconstruct_phase5_train_membership(
        config, dataset_dir, phase5_workload_evidence
    )
    selected, eligible_count = select_phase6_train_workload(
        records,
        tokenizer,
        phase5_ordered_member_ids=membership,
        expected_examples=int(config.train_workload["expected_examples"]),
        generation_tokens=int(config.train_generation["max_new_tokens"]),
        maximum_tokens=int(
            config.train_workload["maximum_prompt_and_generation_tokens"]
        ),
        selection_prefix=str(config.train_workload["selection_hash_prefix"]),
    )
    ids = [item.record_id for item in selected]
    value = {
        "schema_version": PHASE6_WORKLOAD_SCHEMA_VERSION,
        "workload_id": TRAIN_WORKLOAD_ID,
        "source_membership": {
            "workload_id": phase5_train["id"],
            "input_examples": phase5_train["input_examples"],
            "eligible_examples": phase5_train["eligible_examples"],
            "ordered_ids_sha256": phase5_train["selected_record_ids_sha256"],
            "reconstructed_from_phase2_and_exact_phase5_exclusions": True,
        },
        "selection": {
            "hash_prefix": config.train_workload["selection_hash_prefix"],
            "eligible_examples": eligible_count,
            "selected_examples": len(selected),
            "maximum_prompt_and_generation_tokens": config.train_workload[
                "maximum_prompt_and_generation_tokens"
            ],
            "max_new_tokens": config.train_generation["max_new_tokens"],
            "uses_target_or_model_output": False,
        },
        "selected_record_ids": ids,
        "selected_record_ids_sha256": ordered_record_ids_sha256(ids),
        "examples": [item.to_dict() for item in selected],
    }
    validate_phase6_train_workload(config, value)
    return value


def validate_phase6_train_workload(
    config: Phase6Config, value: Mapping[str, Any]
) -> list[Phase6TrainExample]:
    if value.get("schema_version") != PHASE6_WORKLOAD_SCHEMA_VERSION:
        raise ValueError("unknown Phase 6 train workload schema")
    if value.get("workload_id") != TRAIN_WORKLOAD_ID:
        raise ValueError("Phase 6 train workload ID differs")
    source = value["source_membership"]
    phase5 = config.value["phase5_inputs"]
    if (
        source.get("workload_id") != phase5["train_workload_id"]
        or source.get("ordered_ids_sha256") != phase5["train_ordered_ids_sha256"]
        or source.get("reconstructed_from_phase2_and_exact_phase5_exclusions")
        is not True
    ):
        raise ValueError("Phase 6 source membership differs from exact Phase 5 train")
    selection = value["selection"]
    if (
        selection.get("hash_prefix") != config.train_workload["selection_hash_prefix"]
        or int(selection.get("selected_examples", -1)) != 512
        or selection.get("uses_target_or_model_output") is not False
        or int(selection.get("maximum_prompt_and_generation_tokens", -1)) != 2048
        or int(selection.get("max_new_tokens", -1)) != 1024
    ):
        raise ValueError("Phase 6 train selection contract differs")
    examples = [Phase6TrainExample.from_dict(item) for item in value["examples"]]
    ids = [item.record_id for item in examples]
    if ids != value.get("selected_record_ids") or len(ids) != 512:
        raise ValueError("Phase 6 train selected record order or count differs")
    if ordered_record_ids_sha256(ids) != value.get("selected_record_ids_sha256"):
        raise ValueError("Phase 6 train selected record hash differs")
    if any(item.prompt_tokens + 1024 > 2048 for item in examples):
        raise ValueError("Phase 6 train workload exceeds the model-length boundary")
    return examples


def write_phase6_train_workload(path: Path, value: Mapping[str, Any]) -> None:
    _write_json(path, value)


def load_phase6_train_workload(
    path: Path, config: Phase6Config
) -> list[Phase6TrainExample]:
    value = _read_json(path)
    return validate_phase6_train_workload(config, value)


def target_exact(candidate: str, retained_target: str) -> bool:
    return normalize_transport(candidate) == normalize_transport(retained_target)


def summarize_phase6_train_results(
    results: Iterable[CandidateResult],
    *,
    expected_task_ids: list[str],
    target_exact_by_candidate: Mapping[tuple[str, int], bool],
) -> dict[str, Any]:
    materialized = list(results)
    expected_keys = {
        (result.task_id, result.candidate_index) for result in materialized
    }
    if set(target_exact_by_candidate) != expected_keys:
        raise ValueError(
            "Phase 6 exact-target accounting differs from candidate results"
        )
    summary = summarize_results(
        materialized,
        expected_task_ids=expected_task_ids,
        candidates_per_task=4,
        ks=(1, 4),
    )
    by_task: dict[str, list[CandidateResult]] = {
        task_id: [] for task_id in expected_task_ids
    }
    for result in materialized:
        by_task.setdefault(result.task_id, []).append(result)
    exact_counts: list[int] = []
    verified_non_exact_tasks = 0
    for task_item in summary["per_task"]:
        task_id = task_item["task_id"]
        task_results = by_task[task_id]
        exact_count = sum(
            target_exact_by_candidate[(task_id, result.candidate_index)]
            for result in task_results
        )
        verified_non_exact = sum(
            result.category == "verified"
            and not target_exact_by_candidate[(task_id, result.candidate_index)]
            for result in task_results
        )
        task_item["exact_target_candidate_count"] = exact_count
        task_item["verified_non_exact_candidate_count"] = verified_non_exact
        exact_counts.append(exact_count)
        verified_non_exact_tasks += verified_non_exact > 0

    exact_total = sum(exact_counts)
    verified_non_exact_total = sum(
        result.category == "verified"
        and not target_exact_by_candidate[(result.task_id, result.candidate_index)]
        for result in materialized
    )
    exact_rejected = sum(
        target_exact_by_candidate[(result.task_id, result.candidate_index)]
        and result.category != "verified"
        for result in materialized
    )
    token_counts = [int(result.generated_token_count or 0) for result in materialized]
    summary.update(
        {
            "exact_target_pass_at_k": {
                f"pass@{k}": fmean(pass_at_k(4, count, k) for count in exact_counts)
                for k in (1, 4)
            },
            "exact_target_candidates": {
                "count": exact_total,
                "fraction": exact_total / len(materialized) if materialized else 0.0,
            },
            "tasks_with_exact_target_candidate": {
                "count": sum(count > 0 for count in exact_counts),
                "fraction": (
                    sum(count > 0 for count in exact_counts) / len(expected_task_ids)
                    if expected_task_ids
                    else 0.0
                ),
            },
            "verified_non_exact_candidates": {
                "count": verified_non_exact_total,
                "fraction": (
                    verified_non_exact_total / len(materialized)
                    if materialized
                    else 0.0
                ),
            },
            "tasks_with_verified_non_exact_candidate": {
                "count": verified_non_exact_tasks,
                "fraction": (
                    verified_non_exact_tasks / len(expected_task_ids)
                    if expected_task_ids
                    else 0.0
                ),
            },
            "exact_target_but_not_verified_count": exact_rejected,
            "generated_token_counts": {
                "total": sum(token_counts),
                "mean": fmean(token_counts) if token_counts else None,
                "minimum": min(token_counts) if token_counts else None,
                "maximum": max(token_counts) if token_counts else None,
            },
        }
    )
    summary["phase6_train_integrity_passed"] = bool(
        summary["complete"]
        and summary["infrastructure_error_count"] == 0
        and summary["verifier_timeout_count"] == 0
        and exact_rejected == 0
    )
    return summary


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from no values")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _interval(values: Sequence[float]) -> list[float]:
    return [_percentile(values, 2.5), _percentile(values, 97.5)]


def paired_task_bootstrap(
    base_counts: Sequence[int],
    adapter_counts: Sequence[int],
    *,
    candidates_per_task: int,
    ks: Sequence[int],
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    if len(base_counts) != len(adapter_counts) or not base_counts:
        raise ValueError("paired bootstrap requires equal non-empty task counts")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    task_count = len(base_counts)
    rng = random.Random(seed)
    draws: dict[int, dict[str, list[float]]] = {
        k: {"base": [], "adapter": [], "delta": []} for k in ks
    }
    for _ in range(resamples):
        indices = [rng.randrange(task_count) for _ in range(task_count)]
        for k in ks:
            base = fmean(
                pass_at_k(candidates_per_task, base_counts[index], k)
                for index in indices
            )
            adapter = fmean(
                pass_at_k(candidates_per_task, adapter_counts[index], k)
                for index in indices
            )
            draws[k]["base"].append(base)
            draws[k]["adapter"].append(adapter)
            draws[k]["delta"].append(adapter - base)
    return {
        "method": "paired-task-percentile-bootstrap",
        "resamples": resamples,
        "seed": seed,
        "task_count": task_count,
        "interval_percentiles": [2.5, 97.5],
        "metrics": {
            f"pass@{k}": {
                "base": {
                    "estimate": fmean(
                        pass_at_k(candidates_per_task, count, k)
                        for count in base_counts
                    ),
                    "ci95": _interval(draws[k]["base"]),
                },
                "adapter": {
                    "estimate": fmean(
                        pass_at_k(candidates_per_task, count, k)
                        for count in adapter_counts
                    ),
                    "ci95": _interval(draws[k]["adapter"]),
                },
                "delta_adapter_minus_base": {
                    "estimate": fmean(
                        pass_at_k(candidates_per_task, adapter, k)
                        - pass_at_k(candidates_per_task, base, k)
                        for base, adapter in zip(
                            base_counts, adapter_counts, strict=True
                        )
                    ),
                    "ci95": _interval(draws[k]["delta"]),
                },
            }
            for k in ks
        },
    }


def generalization_gaps(
    train_base: Mapping[str, float],
    train_adapter: Mapping[str, float],
    heldout_base: Mapping[str, float],
    heldout_adapter: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in ("pass@1", "pass@4"):
        base_train_gap = float(train_base[key]) - float(heldout_base[key])
        sft_train_gap = float(train_adapter[key]) - float(heldout_adapter[key])
        train_lift = float(train_adapter[key]) - float(train_base[key])
        heldout_lift = float(heldout_adapter[key]) - float(heldout_base[key])
        result[key] = {
            "base_train_gap": base_train_gap,
            "sft_train_gap": sft_train_gap,
            "train_sft_lift": train_lift,
            "heldout_sft_lift": heldout_lift,
            "differential_gap": train_lift - heldout_lift,
        }
    return result


def differential_gap_bootstrap(
    train_base_counts: Sequence[int],
    train_adapter_counts: Sequence[int],
    heldout_base_counts: Sequence[int],
    heldout_adapter_counts: Sequence[int],
    *,
    candidates_per_task: int = 4,
    ks: Sequence[int] = (1, 4),
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    if (
        len(train_base_counts) != len(train_adapter_counts)
        or len(heldout_base_counts) != len(heldout_adapter_counts)
        or not train_base_counts
        or not heldout_base_counts
    ):
        raise ValueError("differential bootstrap requires paired non-empty workloads")
    rng = random.Random(seed)
    values: dict[int, list[float]] = {k: [] for k in ks}
    for _ in range(resamples):
        train_indices = [
            rng.randrange(len(train_base_counts)) for _ in range(len(train_base_counts))
        ]
        heldout_indices = [
            rng.randrange(len(heldout_base_counts))
            for _ in range(len(heldout_base_counts))
        ]
        for k in ks:
            train_lift = fmean(
                pass_at_k(candidates_per_task, train_adapter_counts[index], k)
                - pass_at_k(candidates_per_task, train_base_counts[index], k)
                for index in train_indices
            )
            heldout_lift = fmean(
                pass_at_k(candidates_per_task, heldout_adapter_counts[index], k)
                - pass_at_k(candidates_per_task, heldout_base_counts[index], k)
                for index in heldout_indices
            )
            values[k].append(train_lift - heldout_lift)
    return {
        "method": "independent-workload-paired-model-task-bootstrap",
        "resamples": resamples,
        "seed": seed,
        "train_task_count": len(train_base_counts),
        "heldout_task_count": len(heldout_base_counts),
        "interval_percentiles": [2.5, 97.5],
        "differential_gap": {f"pass@{k}": {"ci95": _interval(values[k])} for k in ks},
    }


def per_task_verified_counts(summary: Mapping[str, Any]) -> list[int]:
    return [int(item["verified_candidate_count"]) for item in summary["per_task"]]
