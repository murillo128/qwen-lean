from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .baseline import run_phase1_baseline
from .minif2f import Phase1Config
from .qwen3_posttrained_assessment import (
    STRICT_SAMPLING,
    _compact_run,
    _load_base_anchor,
    _load_reference_anchor,
)


MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"
MODEL_REVISION = "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b"
GOEDEL_MODEL_ID = "Goedel-LM/Goedel-Prover-V2-8B"
DEV16_WORKLOAD_ID = "minif2f-valid-dev16-v1"
FULL_WORKLOAD_ID = "minif2f-valid-v1"


def load_assessment_config(path: Path) -> Phase1Config:
    config = Phase1Config.load(path)
    validate_assessment_config(config)
    return config


def validate_assessment_config(config: Phase1Config) -> None:
    required = [
        (("benchmark", "revision"), "f0a20e14c1eeccd859d51bb4c2b3ee487889c303"),
        (("benchmark", "expected_primary_task_count"), 244),
        (("benchmark", "lean_toolchain"), "leanprover/lean4:v4.27.0"),
        (("model", "model_id"), MODEL_ID),
        (("model", "model_revision"), MODEL_REVISION),
        (("model", "tokenizer_id"), MODEL_ID),
        (("model", "tokenizer_revision"), MODEL_REVISION),
        (("engine", "name"), "vllm"),
        (("engine", "version"), "0.10.2"),
        (("engine", "dtype"), "bfloat16"),
        (("engine", "tensor_parallel_size"), 1),
        (("engine", "max_model_len"), 2048),
        (("engine", "max_num_seqs"), 8),
        (("engine", "enforce_eager"), True),
        (("engine", "quantization"), None),
        (("engine", "expected_cuda_device_name_fragment"), "Ada"),
        (("verifier", "timeout_seconds"), 30.0),
        (("assessment", "prompt_format_id"), "whole-proof-v1"),
        (("assessment", "chat_template"), None),
        (("assessment", "proof_extraction"), False),
        (("assessment", "verifier_feedback"), False),
        (("assessment", "repair"), False),
        (("assessment", "native_mode_diagnostic"), False),
        (("assessment", "environment_probe_timeout_seconds"), 120.0),
        (("assessment", "model_license"), "MIT"),
    ]
    for field_path, expected in required:
        value: Any = config.value
        for key in field_path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(
                    "missing DeepSeek-Prover-V2 assessment config field: "
                    + ".".join(field_path)
                )
            value = value[key]
        if value != expected:
            raise ValueError(
                "DeepSeek-Prover-V2 assessment requires "
                f"{'.'.join(field_path)}={expected!r}, got {value!r}"
            )

    if config.sampling != STRICT_SAMPLING:
        raise ValueError(
            "DeepSeek-Prover-V2 assessment sampling differs from the strict contract"
        )
    workloads = config.value.get("workloads", {})
    dev16 = workloads.get(DEV16_WORKLOAD_ID, {})
    full = workloads.get(FULL_WORKLOAD_ID, {})
    if (
        dev16.get("selection") != "explicit_ids"
        or dev16.get("expected_task_count") != 16
        or len(dev16.get("task_ids", [])) != 16
    ):
        raise ValueError("DeepSeek-Prover-V2 dev smoke must freeze exactly 16 tasks")
    if full.get("selection") != "all" or full.get("expected_task_count") != 244:
        raise ValueError(
            "DeepSeek-Prover-V2 full workload must contain all 244 validation tasks"
        )


def run_strict_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    workload_id: str,
    output_dir: Path,
    *,
    verification_workers: int,
) -> tuple[Any, Any, dict[str, Any]]:
    validate_assessment_config(config)
    if workload_id not in {DEV16_WORKLOAD_ID, FULL_WORKLOAD_ID}:
        raise ValueError(f"unsupported DeepSeek-Prover-V2 workload: {workload_id}")
    return run_phase1_baseline(
        config,
        benchmark_root,
        workload_id,
        output_dir,
        timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
        verification_workers=verification_workers,
        report_progress=True,
        environment_probe_timeout_seconds=float(
            config.value["assessment"]["environment_probe_timeout_seconds"]
        ),
    )


def write_compact_evidence(
    config: Phase1Config,
    *,
    dev16_dir: Path,
    full_dir: Path,
    base_dir: Path,
    reference_path: Path,
    qwen3_posttrained_path: Path,
    qwen35_4b_base_path: Path,
    goedel_path: Path | None,
    evidence_dir: Path,
) -> dict[str, Any]:
    validate_assessment_config(config)
    dev16 = _compact_run(
        config,
        dev16_dir,
        workload_id=DEV16_WORKLOAD_ID,
        expected_tasks=16,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        assessment_name="DeepSeek-Prover-V2-7B",
        run_evidence_schema="deepseek-prover-v2-7b-run-evidence-v1",
    )
    full = _compact_run(
        config,
        full_dir,
        workload_id=FULL_WORKLOAD_ID,
        expected_tasks=244,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        assessment_name="DeepSeek-Prover-V2-7B",
        run_evidence_schema="deepseek-prover-v2-7b-run-evidence-v1",
    )
    anchors = {
        "qwen3_8b_base": _load_base_anchor(base_dir),
        "qwen3_8b_posttrained": _load_qwen3_posttrained_anchor(
            qwen3_posttrained_path
        ),
        "qwen35_4b_base": _load_qwen35_4b_base_anchor(qwen35_4b_base_path),
        "reference_sft_v1": _load_reference_anchor(reference_path),
    }
    goedel = _load_optional_goedel_anchor(goedel_path)
    metrics: dict[str, dict[str, float]] = {}
    for key in ("pass@1", "pass@4"):
        strict_value = float(full["pass_at_k"][key])
        row = {"deepseek_prover_v2_7b": strict_value}
        for name, anchor in anchors.items():
            anchor_value = float(anchor["pass_at_k"][key])
            row[name] = anchor_value
            row[f"delta_deepseek_minus_{name}"] = strict_value - anchor_value
        if goedel["status"] == "available":
            goedel_value = float(goedel["pass_at_k"][key])
            row["goedel_prover_v2_8b"] = goedel_value
            row["delta_deepseek_minus_goedel"] = strict_value - goedel_value
        metrics[key] = row

    comparison = {
        "schema_version": "deepseek-prover-v2-7b-comparison-v1",
        "status": "passed",
        "workload_id": FULL_WORKLOAD_ID,
        "strict_model": MODEL_ID,
        "strict_model_revision": MODEL_REVISION,
        "anchors_regenerated": False,
        "candidate_budget_caveat": (
            "The strict DeepSeek, Qwen3-8B post-trained, and Qwen3.5-4B-Base "
            "lanes use four candidates per task. The historical Qwen3-8B-Base "
            "and reference-sft-v1 anchors use eight; their pass@1/pass@4 values "
            "use the same estimator and verifier but have a different finite "
            "sampling budget."
        ),
        "metrics": metrics,
        "accepted_anchors": anchors,
        "goedel_comparison": goedel,
        "strict_execution_integrity": {
            "task_count": full["task_count"],
            "candidate_count": full["candidate_count"],
            "infrastructure_error_count": full["infrastructure_error_count"],
            "verifier_timeout_count": full["verifier_timeout_count"],
            "raw_continuation": True,
            "chat_template": None,
            "proof_extraction": False,
            "verifier_feedback": False,
            "repair": False,
        },
    }

    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("dev16.json", dev16),
        ("full.json", full),
        ("comparison.json", comparison),
    ):
        (evidence_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (evidence_dir / "README.md").write_text(
        _render_readme(config, dev16, full, comparison), encoding="utf-8"
    )
    return comparison


def _load_qwen3_posttrained_anchor(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    model = value.get("model", {})
    _validate_four_candidate_anchor(value, "Qwen/Qwen3-8B")
    if (
        model.get("id") != "Qwen/Qwen3-8B"
        or model.get("revision") != "b968826d9c46dd6066d109eabc6255188de91218"
    ):
        raise ValueError("accepted Qwen3-8B post-trained anchor identity differs")
    return {
        "id": model["id"],
        "revision": model["revision"],
        "candidate_count": value["candidate_count"],
        "candidates_per_task": value["candidates_per_task"],
        "pass_at_k": {key: value["pass_at_k"][key] for key in ("pass@1", "pass@4")},
        "source": "evidence/qwen3-8b-posttrained/full.json",
    }


def _load_qwen35_4b_base_anchor(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("execution_identity", {})
    _validate_four_candidate_anchor(value, "Qwen/Qwen3.5-4B-Base")
    if (
        identity.get("model_id") != "Qwen/Qwen3.5-4B-Base"
        or identity.get("model_revision")
        != "1001bb4d826a52d1f399e183466143f4da7b741b"
    ):
        raise ValueError("accepted Qwen3.5-4B-Base anchor identity differs")
    return {
        "id": identity["model_id"],
        "revision": identity["model_revision"],
        "candidate_count": value["candidate_count"],
        "candidates_per_task": value["candidates_per_task"],
        "pass_at_k": {key: value["pass_at_k"][key] for key in ("pass@1", "pass@4")},
        "source": "evidence/qwen35-4b-base/full.json",
    }


def _validate_four_candidate_anchor(value: dict[str, Any], name: str) -> None:
    metrics = value.get("pass_at_k")
    if (
        value.get("task_count") != 244
        or value.get("candidate_count") != 976
        or value.get("candidates_per_task") != 4
        or value.get("infrastructure_error_count") != 0
        or not isinstance(metrics, dict)
        or any(key not in metrics for key in ("pass@1", "pass@4"))
    ):
        raise ValueError(f"accepted {name} anchor is incomplete or incompatible")


def _load_optional_goedel_anchor(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "id": GOEDEL_MODEL_ID,
            "status": "unavailable",
            "reason": (
                "No accepted qwen-lean-measured Goedel-Prover-V2-8B evidence "
                "is present in the authoritative repository; failed or partial "
                "sibling-run observations are not imported."
            ),
            "source": None,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    model = value.get("model", {})
    model_id = model.get("id", model.get("model_id"))
    metrics = value.get("pass_at_k", value.get("summary", {}).get("pass_at_k"))
    task_count = value.get("task_count", value.get("summary", {}).get("task_count"))
    infrastructure_errors = value.get(
        "infrastructure_error_count",
        value.get("summary", {}).get("infrastructure_error_count"),
    )
    if (
        model_id != GOEDEL_MODEL_ID
        or task_count != 244
        or infrastructure_errors != 0
        or not isinstance(metrics, dict)
        or any(key not in metrics for key in ("pass@1", "pass@4"))
    ):
        raise ValueError("Goedel anchor is not accepted compatible qwen-lean evidence")
    return {
        "id": GOEDEL_MODEL_ID,
        "status": "available",
        "revision": model.get("revision", model.get("model_revision")),
        "pass_at_k": {key: metrics[key] for key in ("pass@1", "pass@4")},
        "source": str(path),
    }


def _render_readme(
    config: Phase1Config,
    dev16: dict[str, Any],
    full: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    strict_pass = (
        f"{full['pass_at_k']['pass@1']:.6f}/{full['pass_at_k']['pass@4']:.6f}"
    )
    dev_pass = (
        f"{dev16['pass_at_k']['pass@1']:.6f}/"
        f"{dev16['pass_at_k']['pass@4']:.6f}"
    )
    compute = full["compute_per_solved_task"]
    compute_text = (
        "unavailable because no task had a verified candidate"
        if compute["run_wall_seconds"] is None
        else f"{compute['run_wall_seconds']:.2f} run-wall seconds"
    )
    anchors = comparison["accepted_anchors"]
    anchor_text = ", ".join(
        f"`{value['id']}` {value['pass_at_k']['pass@1']:.6f}/"
        f"{value['pass_at_k']['pass@4']:.6f}"
        for value in anchors.values()
    )
    return f"""# DeepSeek-Prover-V2-7B strict Lean specialist assessment

**OBSERVED:** `{MODEL_ID}` completed all {full['task_count']} miniF2F validation
tasks and {full['candidate_count']} raw candidates. It verified
{full['category_counts']['verified']} candidates across
{full['tasks_with_verified_candidate']['count']} tasks. pass@1/pass@4 were
{strict_pass}. The accepted qwen-lean anchors are {anchor_text}.

The dev16 smoke completed {dev16['candidate_count']} candidates with
pass@1/pass@4 of {dev_pass}. Both runs contain zero unresolved generation or
verifier infrastructure errors. The full run retains
{full['verifier_timeout_count']} `verifier_timeout` outcomes as unsuccessful
proofs. Compute per solved task was {compute_text}.

**ACCEPTED:** the primary score uses exact `whole-proof-v1` raw continuation,
four candidates per task, temperature 0.8, top-p 0.95, no top-k, 1,024 new
tokens, seed 0, no chat wrapper, no proof extraction, no verifier feedback, and
no repair. It ran in BF16 without quantization using local vLLM
`{config.engine['version']}` on {full['runtime']['cuda_device']}.

The model and tokenizer are pinned to `{MODEL_REVISION}`. No optional native
prompt/reasoning diagnostic was run. The repository's model license is MIT;
weights, caches, and raw candidates remain outside Git. Compact JSON retains
execution identity, category and finish-reason counts, token/latency summaries,
wall times, throughput, and compute per solved task.

`comparison.json` uses only accepted qwen-lean evidence. No accepted Goedel
measurement currently exists, so the Goedel entry is explicitly unavailable;
no failed or partial sibling-run observation was imported. Historical
`Qwen/Qwen3-8B-Base` and `reference-sft-v1` anchors used eight candidates while
the strict DeepSeek and other named Qwen lanes used four.
"""
