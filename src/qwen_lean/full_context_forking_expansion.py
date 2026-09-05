"""Conditional 29-task expansion for issue #98 full-context forks.

The expansion is authorized only after the one-task diagnostic has completed and
its GO checkpoint has been published.  Selection is the frozen issue-92 parent
order minus the already-run diagnostic parent; no outcome is consulted when
choosing tasks or constructing requests.
"""

from __future__ import annotations

import asyncio
import gc
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .counterfactual_forking_assessment import (
    MATHEMATICAL_VERIFIER_CATEGORIES,
    ForkRequest,
    ParentTrajectory,
    _assert_no_other_compute_process,
    _configure_fork_runtime,
    _fork_local_runtime,
    _verify_fork_generation_record,
    _validated_package_versions,
    latest_verifications,
    load_fork_verification_records,
    materialize_parent_trajectories,
)
from .full_context_forking_diagnostic import (
    BRANCH_ATTEMPT_SCHEMA,
    GENERATION_SEGMENT_SCHEMA,
    SCIENTIFIC_SEEDS,
    SCIENTIFIC_STATES,
    TARGET_PARENT_ID,
    TARGET_TASK_ID,
    FullContextForkingConfig,
    _branch_attempt_counts,
    _branch_attempt_summary,
    _load_calibration_evidence,
    _nvml_memory_snapshot,
    _require_active_authority,
    _run_full_context_branches,
    _validate_checkpoint_review,
    _validate_scientific_runtime_hardware,
    full_context_config_sha256,
    load_full_context_generation_records,
)
from .native_thinking_assessment import (
    MODEL_REVISION,
    _append_jsonl,
    _atomic_write_json,
    _file_sha256,
    _load_tokenizer,
    _resolve_model_snapshot,
    _sha256_json,
    load_mathia_tasks,
    validate_lean_environments,
)
from .verifier import LeanVerifier

EXPANSION_MANIFEST_SCHEMA = "qwen35-full-context-expansion-manifest-v1"
EXPANSION_RESULTS_SCHEMA = "qwen35-full-context-expansion-results-v1"
EXPANSION_PHASE = "full_context_expansion"
EXPANSION_PARENT_COUNT = 29
EXPANSION_BRANCH_COUNT = 1_218
FROZEN_PARENT_COUNT = 30
PRIOR_NONZERO_TASK_ID = "mathd_numbertheory_33"
GO_CHECKPOINT_COMMENT_ID = 5_551_173_927
GO_CHECKPOINT_URL = (
    "https://github.com/murillo128/qwen-lean/issues/98#issuecomment-5551173927"
)


def _materialize_expansion_parents(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    tokenizer: Any,
    *,
    parent_release_package_path: Path | None,
) -> tuple[list[ParentTrajectory], dict[str, Any]]:
    parents, integrity = materialize_parent_trajectories(
        config.counterfactual,
        mathia_root,
        parent_generations_path,
        tokenizer,
        release_package_path=parent_release_package_path,
    )
    if len(parents) != FROZEN_PARENT_COUNT:
        raise ValueError("frozen issue-92 parent population is not exactly 30")
    diagnostic = parents[0]
    if (
        diagnostic.ordinal != 0
        or diagnostic.task.task_id != TARGET_TASK_ID
        or diagnostic.handoff["candidate_id"] != TARGET_PARENT_ID
    ):
        raise ValueError("frozen issue-98 diagnostic parent is no longer ordinal zero")
    selected = parents[1:]
    if len(selected) != EXPANSION_PARENT_COUNT:
        raise AssertionError("issue-98 expansion parent count changed")
    if len({parent.task.task_id for parent in selected}) != len(selected):
        raise ValueError("issue-98 expansion contains duplicate task IDs")
    if len({parent.handoff["candidate_id"] for parent in selected}) != len(selected):
        raise ValueError("issue-98 expansion contains duplicate parent IDs")
    if TARGET_TASK_ID in {parent.task.task_id for parent in selected}:
        raise ValueError("diagnostic task leaked into the expansion")
    if sum(
        parent.task.task_id == PRIOR_NONZERO_TASK_ID for parent in selected
    ) != 1:
        raise ValueError("frozen prior-nonzero control task changed")
    return selected, integrity


def _manifest_parent_row(parent: ParentTrajectory) -> dict[str, Any]:
    handoff = parent.handoff
    return {
        "frozen_parent_ordinal": parent.ordinal,
        "candidate_id": handoff["candidate_id"],
        "workload": parent.task.workload,
        "task_id": parent.task.task_id,
        "candidate_index": handoff["candidate_index"],
        "seed": handoff["seed"],
        "prompt_sha256": handoff["prompt_sha256"],
        "rendered_prompt_sha256": handoff["rendered_prompt_sha256"],
        "raw_generation_record_sha256": handoff[
            "raw_generation_record_sha256"
        ],
        "raw_response_sha256": handoff["raw_response_sha256"],
        "raw_response_token_ids_sha256": handoff[
            "raw_response_token_ids_sha256"
        ],
        "raw_generation_jsonl_line_number": handoff[
            "raw_generation_jsonl_line_number"
        ],
        "raw_generation_artifact_path": handoff[
            "raw_generation_artifact_path"
        ],
        "raw_response_token_count": len(parent.raw_response_token_ids),
        "rendered_prompt_token_count": len(parent.rendered_prompt_token_ids),
        "frozen_prefix_lengths": {
            state.label: state.prefix_len for state in parent.states
        },
    }


def _validate_diagnostic_go(
    config: FullContextForkingConfig,
    diagnostic_evidence_path: Path,
    selected_context_length: int,
) -> dict[str, Any]:
    evidence = json.loads(diagnostic_evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "complete_scoring_excluded_diagnostic":
        raise ValueError("diagnostic evidence is not complete")
    target = evidence.get("diagnostic_target", {})
    if target != config.diagnostic_target:
        raise ValueError("diagnostic evidence target changed")
    if evidence.get("calibration", {}).get("selected_max_context_length") != (
        selected_context_length
    ):
        raise ValueError("diagnostic evidence uses a different calibrated M")
    outcomes = evidence.get("outcomes", {})
    if int(outcomes.get("nonempty_final_branch_count", -1)) < 6:
        raise ValueError("published GO rule-of-thumb is not supported by evidence")
    states_with_finals = sum(
        int(row.get("nonempty_final_branches", 0)) > 0
        for row in evidence.get("comparison", {}).get(
            "full_context_per_state", []
        )
    )
    if states_with_finals < 2:
        raise ValueError("published GO lacks finals across two prefix states")
    transitions = [
        int(row["reasoning_to_final_transition_token_index"])
        for row in evidence.get("branch_diagnostics", [])
        if row.get("reasoning_to_final_transition_token_index") is not None
    ]
    if not transitions or max(transitions) <= 4_096:
        raise ValueError("published GO lacks a transition beyond 4,096 tokens")
    return evidence


def build_expansion_manifest_payload(
    config: FullContextForkingConfig,
    parents: Sequence[ParentTrajectory],
    parent_integrity: Mapping[str, Any],
    calibration: Mapping[str, Any],
    calibration_evidence_path: Path,
    diagnostic_evidence: Mapping[str, Any],
    diagnostic_evidence_path: Path,
) -> dict[str, Any]:
    if len(parents) != EXPANSION_PARENT_COUNT:
        raise ValueError("expansion manifest requires exactly 29 parents")
    rows = [_manifest_parent_row(parent) for parent in parents]
    if [int(row["frozen_parent_ordinal"]) for row in rows] != list(range(1, 30)):
        raise ValueError("expansion parents differ from frozen ordinals 1..29")
    if TARGET_TASK_ID in {str(row["task_id"]) for row in rows}:
        raise ValueError("diagnostic task cannot appear in expansion manifest")
    selected = int(calibration["selected_max_context_length"])
    unfittable: list[dict[str, Any]] = []
    for parent in parents:
        rendered = len(parent.rendered_prompt_token_ids)
        for state in parent.states:
            max_tokens = selected - rendered - state.prefix_len
            if max_tokens < 1:
                unfittable.append(
                    {
                        "candidate_id": parent.handoff["candidate_id"],
                        "workload": parent.task.workload,
                        "task_id": parent.task.task_id,
                        "fork_state": state.label,
                        "rendered_prompt_token_count": rendered,
                        "frozen_prefix_token_count": state.prefix_len,
                        "max_new_tokens": max_tokens,
                        "affected_seed_count": len(SCIENTIFIC_SEEDS),
                        "reason": "no continuation budget under frozen M",
                    }
                )
    return {
        "schema_version": EXPANSION_MANIFEST_SCHEMA,
        "issue": 98,
        "decision": "GO",
        "go_checkpoint": {
            "comment_id": GO_CHECKPOINT_COMMENT_ID,
            "url": GO_CHECKPOINT_URL,
            "diagnostic_evidence_path": str(
                diagnostic_evidence_path.resolve().relative_to(
                    config.repository_root
                )
            ),
            "diagnostic_evidence_sha256": _file_sha256(
                diagnostic_evidence_path
            ),
            "diagnostic_classification": diagnostic_evidence["classification"],
            "diagnostic_nonempty_final_branch_count": diagnostic_evidence[
                "outcomes"
            ]["nonempty_final_branch_count"],
            "diagnostic_verified_branch_count": diagnostic_evidence["outcomes"][
                "verified_branch_count"
            ],
        },
        "selection": {
            "source": "frozen issue-92 30-task parent population",
            "rule": (
                "frozen parent order minus ordinal-zero diagnostic task; "
                "no outcome or quality filtering"
            ),
            "handoff_commit": parent_integrity["handoff_commit"],
            "handoff_manifest_sha256": parent_integrity[
                "handoff_manifest_sha256"
            ],
            "frozen_parent_count": FROZEN_PARENT_COUNT,
            "excluded_diagnostic": {
                "frozen_parent_ordinal": 0,
                "task_id": TARGET_TASK_ID,
                "candidate_id": TARGET_PARENT_ID,
            },
            "selected_parent_count": len(rows),
            "ordered_task_ids_sha256": _sha256_json(
                [row["task_id"] for row in rows]
            ),
            "ordered_parent_candidate_ids_sha256": _sha256_json(
                [row["candidate_id"] for row in rows]
            ),
            "ordered_parents": rows,
        },
        "calibration": {
            "selected_max_context_length": selected,
            "evidence_path": str(
                calibration_evidence_path.resolve().relative_to(
                    config.repository_root
                )
            ),
            "evidence_sha256": _file_sha256(calibration_evidence_path),
        },
        "request_plan": {
            "states": list(SCIENTIFIC_STATES),
            "seeds": list(SCIENTIFIC_SEEDS),
            "ordering": "manifest parent order, then state order, then seed order",
            "max_new_tokens_formula": (
                "M - rendered_prompt_token_count - frozen_prefix_token_count"
            ),
            "planned_branch_count": EXPANSION_BRANCH_COUNT,
            "unfittable_state_count": len(unfittable),
            "unfittable_request_count": sum(
                int(row["affected_seed_count"]) for row in unfittable
            ),
            "unfittable_requests": unfittable,
        },
        "scientific_contract": {
            "exact_stored_parent_token_ids": True,
            "prefix_positions_reused_from_issue92": True,
            "same_model_tokenizer_runtime_parser_sampling": True,
            "final_channel_only_lean_submission": True,
            "forced_stop_thinking": False,
            "repair_or_extraction": False,
            "quality_driven_retry": False,
            "append_only_restart_safe_artifacts": True,
            "hosted_inference": False,
        },
        "artifact_policy": {
            "raw_generations": "outside Git under the issue-98 artifact root",
            "manifest": "compact and committed before first expansion generation",
        },
    }


def _expansion_context(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    parent_release_package_path: Path | None,
    calibration_evidence_path: Path,
    diagnostic_evidence_path: Path,
) -> tuple[
    list[ParentTrajectory], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    _require_active_authority(config)
    calibration = _load_calibration_evidence(config, calibration_evidence_path)
    selected = int(calibration["selected_max_context_length"])
    diagnostic = _validate_diagnostic_go(
        config, diagnostic_evidence_path, selected
    )
    snapshot_path = _resolve_model_snapshot(config.counterfactual.native)
    tokenizer = _load_tokenizer(
        config.counterfactual.native, snapshot_path=snapshot_path
    )
    parents, integrity = _materialize_expansion_parents(
        config,
        mathia_root,
        parent_generations_path,
        tokenizer,
        parent_release_package_path=parent_release_package_path,
    )
    return parents, integrity, calibration, diagnostic


def write_expansion_manifest(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    calibration_evidence_path: Path,
    diagnostic_evidence_path: Path,
    output_path: Path,
    *,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    parents, integrity, calibration, diagnostic = _expansion_context(
        config,
        mathia_root,
        parent_generations_path,
        parent_release_package_path,
        calibration_evidence_path,
        diagnostic_evidence_path,
    )
    manifest = build_expansion_manifest_payload(
        config,
        parents,
        integrity,
        calibration,
        calibration_evidence_path,
        diagnostic,
        diagnostic_evidence_path,
    )
    _atomic_write_json(output_path, manifest)
    return manifest


def _load_expansion_manifest(
    config: FullContextForkingConfig,
    parents: Sequence[ParentTrajectory],
    integrity: Mapping[str, Any],
    calibration: Mapping[str, Any],
    calibration_evidence_path: Path,
    diagnostic: Mapping[str, Any],
    diagnostic_evidence_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_expansion_manifest_payload(
        config,
        parents,
        integrity,
        calibration,
        calibration_evidence_path,
        diagnostic,
        diagnostic_evidence_path,
    )
    if manifest != expected:
        raise ValueError("expansion manifest differs from frozen issue-92 order")
    return manifest


def _expansion_generation_hash(
    config: FullContextForkingConfig,
    calibration_evidence_path: Path,
    selected_context_length: int,
    manifest_path: Path,
) -> str:
    return _sha256_json(
        {
            "config_sha256": full_context_config_sha256(config),
            "calibration_evidence_sha256": _file_sha256(
                calibration_evidence_path
            ),
            "selected_max_context_length": selected_context_length,
            "expansion_manifest_sha256": _file_sha256(manifest_path),
            "go_checkpoint_comment_id": GO_CHECKPOINT_COMMENT_ID,
        }
    )


def expansion_requests(
    config: FullContextForkingConfig,
    parents: Sequence[ParentTrajectory],
    *,
    selected_context_length: int,
    calibration_evidence_path: Path,
    manifest_path: Path,
) -> tuple[list[ForkRequest], list[dict[str, Any]]]:
    generation_hash = _expansion_generation_hash(
        config,
        calibration_evidence_path,
        selected_context_length,
        manifest_path,
    )
    requests: list[ForkRequest] = []
    unfittable: list[dict[str, Any]] = []
    for parent in parents:
        states = {state.label: state for state in parent.states}
        rendered_prompt_count = len(parent.rendered_prompt_token_ids)
        for state_label in SCIENTIFIC_STATES:
            state = states[state_label]
            max_tokens = (
                selected_context_length
                - rendered_prompt_count
                - state.prefix_len
            )
            if max_tokens < 1:
                for seed in SCIENTIFIC_SEEDS:
                    unfittable.append(
                        {
                            "candidate_id": parent.handoff["candidate_id"],
                            "workload": parent.task.workload,
                            "task_id": parent.task.task_id,
                            "fork_state": state.label,
                            "branch_seed": seed,
                            "max_new_tokens": max_tokens,
                            "reason": "no continuation budget under frozen M",
                        }
                    )
                continue
            prefix = parent.raw_response_token_ids[: state.prefix_len]
            for seed in SCIENTIFIC_SEEDS:
                identity = {
                    "phase": EXPANSION_PHASE,
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
                    "interval_id": None,
                    "interval_side": None,
                }
                branch_id = (
                    "full-context-expansion-fork-"
                    + _sha256_json(identity)[:32]
                )
                requests.append(
                    ForkRequest(
                        phase=EXPANSION_PHASE,
                        parent=parent,
                        state=state,
                        seed=seed,
                        max_tokens=max_tokens,
                        branch_id=branch_id,
                        generation_config_sha256=generation_hash,
                        fork_prompt_token_ids=(
                            parent.rendered_prompt_token_ids + prefix
                        ),
                    )
                )
    if len(requests) + len(unfittable) != EXPANSION_BRANCH_COUNT:
        raise AssertionError("full-context expansion branch count changed")
    if len({request.branch_id for request in requests}) != len(requests):
        raise AssertionError("full-context expansion branch IDs are not unique")
    return requests, unfittable


def _validated_expansion_plan(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    parent_release_package_path: Path | None,
    calibration_evidence_path: Path,
    diagnostic_evidence_path: Path,
    manifest_path: Path,
) -> tuple[
    list[ParentTrajectory],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[ForkRequest],
    list[dict[str, Any]],
]:
    parents, integrity, calibration, diagnostic = _expansion_context(
        config,
        mathia_root,
        parent_generations_path,
        parent_release_package_path,
        calibration_evidence_path,
        diagnostic_evidence_path,
    )
    manifest = _load_expansion_manifest(
        config,
        parents,
        integrity,
        calibration,
        calibration_evidence_path,
        diagnostic,
        diagnostic_evidence_path,
        manifest_path,
    )
    requests, unfittable = expansion_requests(
        config,
        parents,
        selected_context_length=int(calibration["selected_max_context_length"]),
        calibration_evidence_path=calibration_evidence_path,
        manifest_path=manifest_path,
    )
    if int(manifest["request_plan"]["unfittable_request_count"]) != len(
        unfittable
    ):
        raise ValueError("manifest unfittable-request count changed")
    return parents, integrity, calibration, manifest, requests, unfittable


def run_expansion_generation(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    calibration_evidence_path: Path,
    checkpoint_review_path: Path,
    diagnostic_evidence_path: Path,
    manifest_path: Path,
    *,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    _require_active_authority(config)
    review = _validate_checkpoint_review(
        calibration_evidence_path, checkpoint_review_path
    )
    (
        _parents,
        integrity,
        calibration,
        manifest,
        expected,
        unfittable,
    ) = _validated_expansion_plan(
        config,
        mathia_root,
        parent_generations_path,
        parent_release_package_path,
        calibration_evidence_path,
        diagnostic_evidence_path,
        manifest_path,
    )
    selected = int(calibration["selected_max_context_length"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generation_path = artifact_dir / "generations.jsonl"
    attempt_path = artifact_dir / "branch-attempts.jsonl"
    prior = load_full_context_generation_records(
        generation_path, expected, selected_context_length=selected
    )
    completed = {str(record["branch_id"]) for record in prior}
    pending = [request for request in expected if request.branch_id not in completed]
    prior_hashes = {
        str(record["branch_id"]): _sha256_json(record) for record in prior
    }
    if not pending:
        _branch_attempt_counts(
            attempt_path, expected, persisted_record_hashes=prior_hashes
        )
        return {
            "status": "already_complete",
            "planned_branches": EXPANSION_BRANCH_COUNT,
            "expected_fittable_branches": len(expected),
            "unfittable_branches": len(unfittable),
            "new_branches": 0,
            "selected_max_context_length": selected,
            "manifest_sha256": _file_sha256(manifest_path),
            "integrity": integrity,
        }

    _configure_fork_runtime()
    runtime = _fork_local_runtime(config.counterfactual)
    runtime["engine_gpu_memory_utilization"] = config.execution[
        "gpu_memory_utilization"
    ]
    runtime["selected_max_context_length"] = selected
    runtime["package_versions"] = _validated_package_versions()
    device_index = int(runtime["cuda_device_index"])
    snapshot = _nvml_memory_snapshot(device_index)
    runtime["nvml_gpu_name"] = snapshot["gpu_name"]
    runtime["nvml_gpu_uuid"] = snapshot["gpu_uuid"]
    runtime["nvml_gpu_memory_total_bytes"] = snapshot[
        "gpu_memory_total_bytes"
    ]
    runtime["calibrated_hardware_binding"] = (
        _validate_scientific_runtime_hardware(config, runtime, calibration)
    )
    _assert_no_other_compute_process(device_index)

    started = time.perf_counter()
    status = "failed"
    error_text: str | None = None
    new_records: list[dict[str, Any]] = []
    try:
        snapshot_path = _resolve_model_snapshot(config.counterfactual.native)
        tokenizer = _load_tokenizer(
            config.counterfactual.native, snapshot_path=snapshot_path
        )
        new_records = asyncio.run(
            _run_full_context_branches(
                config,
                tokenizer,
                pending,
                expected,
                generation_path,
                attempt_path,
                snapshot_path,
                selected_context_length=selected,
                device_index=device_index,
                attempt_recovery_provenance=None,
                recovered_attempt_counts=None,
                persisted_record_hashes=prior_hashes,
            )
        )
        _branch_attempt_counts(
            attempt_path,
            expected,
            persisted_record_hashes={
                **prior_hashes,
                **{
                    str(record["branch_id"]): _sha256_json(record)
                    for record in new_records
                },
            },
        )
        status = "completed"
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        _append_jsonl(
            artifact_dir / "generation-segments.jsonl",
            {
                "schema_version": GENERATION_SEGMENT_SCHEMA,
                "status": status,
                "requested_branch_count": len(pending),
                "persisted_branch_count": len(new_records),
                "segment_wall_time_seconds": time.perf_counter() - started,
                "error": error_text,
                "selected_max_context_length": selected,
                "calibration_evidence_sha256": _file_sha256(
                    calibration_evidence_path
                ),
                "checkpoint_reviewed_commit": review["reviewed_commit"],
                "expansion_manifest_sha256": _file_sha256(manifest_path),
                "go_checkpoint_comment_id": manifest["go_checkpoint"][
                    "comment_id"
                ],
                "unfittable_branch_count": len(unfittable),
                "runtime": runtime,
            },
        )
        gc.collect()
    return {
        "status": status,
        "planned_branches": EXPANSION_BRANCH_COUNT,
        "expected_fittable_branches": len(expected),
        "unfittable_branches": len(unfittable),
        "new_branches": len(new_records),
        "selected_max_context_length": selected,
        "manifest_sha256": _file_sha256(manifest_path),
        "integrity": integrity,
        "runtime": runtime,
    }


def run_expansion_verification(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    calibration_evidence_path: Path,
    checkpoint_review_path: Path,
    diagnostic_evidence_path: Path,
    manifest_path: Path,
    *,
    project_roots: Mapping[str, Path],
    workers: int | None = None,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    _require_active_authority(config)
    _validate_checkpoint_review(calibration_evidence_path, checkpoint_review_path)
    (
        _parents,
        integrity,
        calibration,
        _manifest,
        expected,
        unfittable,
    ) = _validated_expansion_plan(
        config,
        mathia_root,
        parent_generations_path,
        parent_release_package_path,
        calibration_evidence_path,
        diagnostic_evidence_path,
        manifest_path,
    )
    selected = int(calibration["selected_max_context_length"])
    generations = load_full_context_generation_records(
        artifact_dir / "generations.jsonl",
        expected,
        selected_context_length=selected,
    )
    if len(generations) != len(expected):
        raise RuntimeError(
            f"expansion generation is incomplete: {len(generations)}/{len(expected)}"
        )
    _branch_attempt_counts(
        artifact_dir / "branch-attempts.jsonl",
        expected,
        persisted_record_hashes={
            str(record["branch_id"]): _sha256_json(record)
            for record in generations
        },
    )
    environment_tasks, _ = load_mathia_tasks(
        config.counterfactual.native, mathia_root
    )
    environments = validate_lean_environments(
        config.counterfactual.native, environment_tasks, project_roots
    )
    verification_path = artifact_dir / "verifications.jsonl"
    prior = load_fork_verification_records(verification_path, expected)
    latest = latest_verifications(prior)
    pending = [
        generation
        for generation in generations
        if str(generation["branch_id"]) not in latest
        or str(latest[str(generation["branch_id"])]["category"])
        not in MATHEMATICAL_VERIFIER_CATEGORIES
    ]
    if not pending:
        return {
            "status": "already_complete",
            "generation_branches": len(generations),
            "unfittable_branches": len(unfittable),
            "new_verifications": 0,
            "integrity": integrity,
            "environments": environments,
        }
    worker_count = int(
        config.counterfactual.native.verifier["workers"]
        if workers is None
        else workers
    )
    if worker_count < 1:
        raise ValueError("verification worker count must be positive")
    verifier = LeanVerifier(
        project_roots["minif2f-valid-clean-v2"],
        timeout_seconds=float(
            config.counterfactual.native.verifier["timeout_seconds"]
        ),
    )
    attempts = Counter(str(record["branch_id"]) for record in prior)
    started = time.perf_counter()
    new_count = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _verify_fork_generation_record,
                generation,
                next(
                    request.parent.task
                    for request in expected
                    if request.branch_id == generation["branch_id"]
                ),
                verifier,
                attempt_index=attempts[str(generation["branch_id"])],
            ): str(generation["branch_id"])
            for generation in pending
        }
        for future in as_completed(futures):
            record = future.result()
            _append_jsonl(verification_path, record)
            new_count += 1
            print(
                json.dumps(
                    {
                        "phase": "full_context_expansion_verification",
                        "completed": new_count,
                        "pending": len(pending),
                        "branch_id": record["branch_id"],
                        "category": record["category"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "status": "completed",
        "generation_branches": len(generations),
        "unfittable_branches": len(unfittable),
        "new_verifications": new_count,
        "verification_wall_time_seconds": time.perf_counter() - started,
        "integrity": integrity,
        "environments": environments,
    }


def _state_metrics(
    requests: Sequence[ForkRequest],
    generations: Sequence[dict[str, Any]],
    verifications: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in SCIENTIFIC_STATES:
        state_requests = [r for r in requests if r.state.label == state]
        ids = {r.branch_id for r in state_requests}
        state_generations = [r for r in generations if r["branch_id"] in ids]
        nonempty = sum(
            r["final_production_status"] == "nonempty"
            for r in state_generations
        )
        verified = sum(
            verifications[branch_id]["category"] == "verified"
            for branch_id in ids
        )
        rows.append(
            {
                "fork_state": state,
                "branch_count": len(state_requests),
                "nonempty_final_branches": nonempty,
                "verified_branches": verified,
                "F": nonempty / len(state_requests),
                "V": verified / len(state_requests),
            }
        )
    return rows


def _issue92_baselines(
    config: FullContextForkingConfig,
    parents: Sequence[ParentTrajectory],
) -> dict[str, dict[str, Any]]:
    path = config.repository_root / str(
        config.reviewed_target["counterfactual_results_path"]
    )
    if _file_sha256(path) != config.reviewed_target[
        "counterfactual_results_sha256"
    ]:
        raise ValueError("reviewed issue-92 evidence changed")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    source_rows = evidence["analysis"]["discovery"]["per_prefix"]
    baselines: dict[str, dict[str, Any]] = {}
    for parent in parents:
        candidate_id = str(parent.handoff["candidate_id"])
        rows = [
            row
            for row in source_rows
            if row["task_id"] == parent.task.task_id
            and row["parent_candidate_id"] == candidate_id
        ]
        if [row["fork_state"] for row in rows] != list(SCIENTIFIC_STATES):
            raise ValueError(
                f"issue-92 baseline state order changed: {parent.task.task_id}"
            )
        if any(int(row["branch_count"]) != len(SCIENTIFIC_SEEDS) for row in rows):
            raise ValueError(
                f"issue-92 baseline branch count changed: {parent.task.task_id}"
            )
        any_nonzero = any(
            int(row["nonempty_final_branches"]) > 0
            or int(row["verified_branches"]) > 0
            for row in rows
        )
        if (parent.task.task_id == PRIOR_NONZERO_TASK_ID) != any_nonzero:
            raise ValueError(
                "issue-92 prior-nonzero versus all-zero partition changed"
            )
        baselines[candidate_id] = {
            "had_any_nonzero_variation": any_nonzero,
            "per_state": [
                {
                    "fork_state": row["fork_state"],
                    "fork_prefix_len": row["fork_prefix_len"],
                    "branch_count": row["branch_count"],
                    "nonempty_final_branches": row[
                        "nonempty_final_branches"
                    ],
                    "verified_branches": row["verified_branches"],
                    "F": row["F"],
                    "V": row["V_op"],
                }
                for row in rows
            ],
        }
    return baselines


def write_expansion_evidence(
    config: FullContextForkingConfig,
    mathia_root: Path,
    parent_generations_path: Path,
    artifact_dir: Path,
    calibration_evidence_path: Path,
    diagnostic_evidence_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    parent_release_package_path: Path | None = None,
) -> dict[str, Any]:
    (
        parents,
        integrity,
        calibration,
        manifest,
        requests,
        unfittable,
    ) = _validated_expansion_plan(
        config,
        mathia_root,
        parent_generations_path,
        parent_release_package_path,
        calibration_evidence_path,
        diagnostic_evidence_path,
        manifest_path,
    )
    selected = int(calibration["selected_max_context_length"])
    generations = load_full_context_generation_records(
        artifact_dir / "generations.jsonl",
        requests,
        selected_context_length=selected,
    )
    if len(generations) != len(requests):
        raise RuntimeError("cannot write evidence from incomplete expansion generation")
    _branch_attempt_counts(
        artifact_dir / "branch-attempts.jsonl",
        requests,
        persisted_record_hashes={
            str(record["branch_id"]): _sha256_json(record)
            for record in generations
        },
    )
    verification_records = load_fork_verification_records(
        artifact_dir / "verifications.jsonl", requests
    )
    latest = latest_verifications(verification_records)
    if len(latest) != len(requests) or any(
        record["category"] not in MATHEMATICAL_VERIFIER_CATEGORIES
        for record in latest.values()
    ):
        raise RuntimeError("cannot write evidence from incomplete expansion verification")
    by_request = {request.branch_id: request for request in requests}
    issue92_baselines = _issue92_baselines(config, parents)
    by_parent: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for parent in parents:
        parent_requests = [
            request
            for request in requests
            if request.parent.handoff["candidate_id"]
            == parent.handoff["candidate_id"]
        ]
        ids = {request.branch_id for request in parent_requests}
        parent_generations = [
            generation
            for generation in generations
            if generation["branch_id"] in ids
        ]
        metrics = _state_metrics(parent_requests, parent_generations, latest)
        nonempty = sum(
            generation["final_production_status"] == "nonempty"
            for generation in parent_generations
        )
        verified = sum(
            latest[branch_id]["category"] == "verified" for branch_id in ids
        )
        by_parent.append(
            {
                "frozen_parent_ordinal": parent.ordinal,
                "candidate_id": parent.handoff["candidate_id"],
                "workload": parent.task.workload,
                "task_id": parent.task.task_id,
                "branch_count": len(parent_requests),
                "nonempty_final_branch_count": nonempty,
                "verified_branch_count": verified,
                "context_exhausted_no_final_count": sum(
                    generation["finish_reason"] == "token_limit"
                    and generation["final_production_status"] == "empty"
                    for generation in parent_generations
                ),
                "issue92": issue92_baselines[
                    str(parent.handoff["candidate_id"])
                ],
                "per_state": metrics,
            }
        )
    for generation in generations:
        transition = generation.get("reasoning_to_final_transition_token_index")
        if transition is None:
            continue
        request = by_request[str(generation["branch_id"])]
        transitions.append(
            {
                "branch_id": generation["branch_id"],
                "candidate_id": request.parent.handoff["candidate_id"],
                "task_id": request.parent.task.task_id,
                "fork_state": request.state.label,
                "branch_seed": request.seed,
                "transition_token_index": transition,
                "beyond_issue92_envelope": int(transition) > 4_096,
                "final_token_count": generation["final_token_count"],
                "verifier_category": latest[request.branch_id]["category"],
            }
        )
    prior = next(
        row for row in by_parent if row["task_id"] == PRIOR_NONZERO_TASK_ID
    )
    previously_zero = [
        row for row in by_parent if row["task_id"] != PRIOR_NONZERO_TASK_ID
    ]
    diagnostic = json.loads(diagnostic_evidence_path.read_text(encoding="utf-8"))
    expansion_per_state = _state_metrics(requests, generations, latest)
    diagnostic_per_state = {
        row["fork_state"]: row
        for row in diagnostic["comparison"]["full_context_per_state"]
    }
    all_30_per_state: list[dict[str, Any]] = []
    for row in expansion_per_state:
        diagnostic_row = diagnostic_per_state[row["fork_state"]]
        branch_count = int(row["branch_count"]) + int(
            diagnostic_row["branch_count"]
        )
        nonempty = int(row["nonempty_final_branches"]) + int(
            diagnostic_row["nonempty_final_branches"]
        )
        verified = int(row["verified_branches"]) + int(
            diagnostic_row["verified_branches"]
        )
        all_30_per_state.append(
            {
                "fork_state": row["fork_state"],
                "branch_count": branch_count,
                "nonempty_final_branches": nonempty,
                "verified_branches": verified,
                "F": nonempty / branch_count,
                "V": verified / branch_count,
            }
        )
    diagnostic_transitions = [
        {
            "branch_id": row["branch_id"],
            "task_id": TARGET_TASK_ID,
            "fork_state": row["fork_state"],
            "branch_seed": row["branch_seed"],
            "transition_token_index": row[
                "reasoning_to_final_transition_token_index"
            ],
            "beyond_issue92_envelope": int(
                row["reasoning_to_final_transition_token_index"]
            )
            > 4_096,
            "final_token_count": row["final_token_count"],
            "verifier_category": row["verifier_category"],
        }
        for row in diagnostic["branch_diagnostics"]
        if row["reasoning_to_final_transition_token_index"] is not None
    ]
    expansion_tasks_with_final = [
        row["task_id"]
        for row in by_parent
        if row["nonempty_final_branch_count"] > 0
    ]
    expansion_tasks_with_verified = [
        row["task_id"]
        for row in by_parent
        if row["verified_branch_count"] > 0
    ]
    all_30_tasks_with_final = [
        *(
            [TARGET_TASK_ID]
            if diagnostic["outcomes"]["nonempty_final_branch_count"] > 0
            else []
        ),
        *expansion_tasks_with_final,
    ]
    all_30_tasks_with_verified = [
        *(
            [TARGET_TASK_ID]
            if diagnostic["outcomes"]["verified_branch_count"] > 0
            else []
        ),
        *expansion_tasks_with_verified,
    ]
    diagnostic_generated_tokens = sum(
        int(row["generated_token_count"])
        for row in diagnostic["branch_diagnostics"]
    )
    diagnostic_generation_time = sum(
        float(row["generation_latency_seconds"])
        for row in diagnostic["branch_diagnostics"]
    )
    expansion_generated_tokens = sum(
        int(generation["total_generated_token_count"])
        for generation in generations
    )
    expansion_generation_time = sum(
        float(generation["generation_latency_seconds"])
        for generation in generations
    )
    expansion_peak_memory = max(
        int(generation["branch_gpu_memory"]["gpu_memory_peak_bytes"])
        for generation in generations
    )
    evidence = {
        "schema_version": EXPANSION_RESULTS_SCHEMA,
        "status": "complete_scoring_excluded_confirmation",
        "decision": "GO",
        "manifest": {
            "path": str(
                manifest_path.resolve().relative_to(config.repository_root)
            ),
            "sha256": _file_sha256(manifest_path),
            "go_checkpoint": manifest["go_checkpoint"],
        },
        "calibration": {
            "selected_max_context_length": selected,
            "evidence_sha256": _file_sha256(calibration_evidence_path),
        },
        "coverage": {
            "frozen_population_task_count": FROZEN_PARENT_COUNT,
            "diagnostic_task_count": 1,
            "expansion_task_count": len(parents),
            "planned_expansion_branch_count": EXPANSION_BRANCH_COUNT,
            "generated_expansion_branch_count": len(generations),
            "unfittable_expansion_branch_count": len(unfittable),
        },
        "groups": {
            "additional_previously_all_zero": {
                "task_count": len(previously_zero),
                "tasks": previously_zero,
            },
            "prior_nonzero_control": prior,
            "original_diagnostic": {
                "task_id": TARGET_TASK_ID,
                "evidence_path": str(
                    diagnostic_evidence_path.resolve().relative_to(
                        config.repository_root
                    )
                ),
                "evidence_sha256": _file_sha256(diagnostic_evidence_path),
                "classification": diagnostic["classification"],
                "outcomes": diagnostic["outcomes"],
                "per_state": diagnostic["comparison"][
                    "full_context_per_state"
                ],
                "reasoning_to_final_transitions": diagnostic_transitions,
                "regenerated": False,
            },
        },
        "expansion_aggregate": {
            "per_state": expansion_per_state,
            "tasks_with_any_nonempty_final": expansion_tasks_with_final,
            "tasks_with_any_verified_branch": expansion_tasks_with_verified,
            "verifier_category_counts": dict(
                sorted(
                    Counter(
                        str(record["category"]) for record in latest.values()
                    ).items()
                )
            ),
            "context_exhausted_no_final_count": sum(
                generation["finish_reason"] == "token_limit"
                and generation["final_production_status"] == "empty"
                for generation in generations
            ),
            "generated_token_count": expansion_generated_tokens,
            "generation_time_seconds": expansion_generation_time,
            "peak_gpu_memory_bytes": expansion_peak_memory,
        },
        "all_30_aggregate": {
            "task_count": FROZEN_PARENT_COUNT,
            "branch_count": EXPANSION_BRANCH_COUNT
            - len(unfittable)
            + int(diagnostic["scientific_contract"]["branch_count"]),
            "per_state": all_30_per_state,
            "tasks_with_any_nonempty_final": all_30_tasks_with_final,
            "tasks_with_any_verified_branch": all_30_tasks_with_verified,
            "generated_token_count": expansion_generated_tokens
            + diagnostic_generated_tokens,
            "generation_time_seconds": expansion_generation_time
            + diagnostic_generation_time,
            "peak_gpu_memory_bytes": max(
                expansion_peak_memory,
                max(
                    int(row["gpu_memory_peak_bytes"])
                    for row in diagnostic["branch_diagnostics"]
                ),
            ),
        },
        "reasoning_to_final_transitions": transitions,
        "branch_identity": {
            "ordered_branch_ids_sha256": _sha256_json(
                [request.branch_id for request in requests]
            ),
            "fork_generation_config_sha256": (
                requests[0].generation_config_sha256 if requests else None
            ),
            "raw_generations_sha256": _file_sha256(
                artifact_dir / "generations.jsonl"
            ),
            "raw_verifications_sha256": _file_sha256(
                artifact_dir / "verifications.jsonl"
            ),
        },
        "restart_safety": _branch_attempt_summary(
            artifact_dir / "branch-attempts.jsonl"
        ),
        "integrity": {
            "parent": integrity,
            "manifest_reconstructed_exactly": True,
            "raw_generation_records_validated": True,
            "issue92_results_sha256": config.reviewed_target[
                "counterfactual_results_sha256"
            ],
        },
        "limitations": [
            "This confirmation remains scoring-excluded and does not alter issue #92.",
            "The prior-nonzero mathd_numbertheory_33 task is reported separately.",
            "This result does not authorize DPO, RLVR, value-model training, or architecture changes.",
            "Raw generations and verifier diagnostics remain outside Git.",
        ],
    }
    _atomic_write_json(output_path, evidence)
    return evidence
