from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_v2 import sha256_file
from .generalist_v3 import (
    GeneralistV3Config,
    normalized_template_hash,
    summarize_canary_candidates,
)
from .schema import CandidateResult, TaskRecord
from .verifier import LeanVerifier


CANARY_RUN_SCHEMA_VERSION = "generalist-v3-canary-run-v1"
BASE_CANARY_EVIDENCE_SCHEMA_VERSION = "generalist-v3-base-canary-evidence-v1"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"unreadable verifier checkout: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _validate_verifier_environment(
    verifier_root: Path, dataset_manifest_path: Path
) -> dict[str, Any]:
    manifest = _read_json(dataset_manifest_path)
    expected = manifest["target_environment"]
    verifier_root = verifier_root.resolve()
    observed_head = _git_head(verifier_root)
    observed_toolchain = (verifier_root / "lean-toolchain").read_text(
        encoding="utf-8"
    ).strip()
    lake_manifest = _read_json(verifier_root / "lake-manifest.json")
    mathlib_package = next(
        item for item in lake_manifest["packages"] if item["name"] == "mathlib"
    )
    mathlib_root = verifier_root / ".lake/packages/mathlib"
    observed_mathlib = _git_head(mathlib_root)
    if observed_head != expected["host_revision"]:
        raise ValueError("generalist-v3 verifier host revision differs")
    if observed_toolchain != expected["lean_toolchain"]:
        raise ValueError("generalist-v3 verifier Lean toolchain differs")
    if (
        mathlib_package["rev"] != expected["mathlib_revision"]
        or observed_mathlib != expected["mathlib_revision"]
    ):
        raise ValueError("generalist-v3 verifier mathlib revision differs")
    return {
        "environment_id": expected["environment_id"],
        "project_root": str(verifier_root),
        "host_repository": expected["host_repository"],
        "host_revision": observed_head,
        "lean_toolchain": observed_toolchain,
        "mathlib_repository": expected["mathlib_repository"],
        "mathlib_revision": observed_mathlib,
    }


def _validate_canary_manifest(
    config: GeneralistV3Config, canary_path: Path
) -> dict[str, Any]:
    manifest = _read_json(canary_path)
    tasks = manifest.get("tasks", [])
    expected_sampling = {
        "candidates_per_task": config.evaluation["candidates_per_task"],
        **config.evaluation["sampling"],
    }
    if (
        manifest.get("schema_version") != "generalist-v3-canary-manifest-v1"
        or manifest.get("role") != "validation"
        or manifest.get("sealed") is not False
        or manifest.get("theorem_count") != 48
        or manifest.get("interface_task_count") != 96
        or manifest.get("interface_counts") != {"incremental": 48, "whole": 48}
        or manifest.get("sampling") != expected_sampling
        or len(tasks) != 96
        or len({str(task["task_id"]) for task in tasks}) != 96
        or manifest.get("ordered_tasks_sha256") != _sha256_json(tasks)
    ):
        raise ValueError("generalist-v3 validation canary manifest differs")
    for task in tasks:
        if task["interface"] not in {"whole", "incremental"}:
            raise ValueError("generalist-v3 validation canary interface differs")
        if task["model_input_sha256"] != hashlib.sha256(
            str(task["model_input"]).encode("utf-8")
        ).hexdigest():
            raise ValueError("generalist-v3 validation model input binding differs")
    return manifest


def _task_records(manifest: Mapping[str, Any]) -> list[TaskRecord]:
    return [
        TaskRecord(
            id=str(task["task_id"]),
            preamble=str(task["preamble"]),
            declaration=str(task["declaration"]),
            declaration_name=str(task["declaration_name"]),
        )
        for task in manifest["tasks"]
    ]


def _snapshot_path(config: GeneralistV3Config, model_snapshot: Path | None) -> Path:
    if model_snapshot is not None:
        snapshot = model_snapshot.resolve()
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=str(config.model["model_id"]),
                revision=str(config.model["model_revision"]),
                local_files_only=True,
            )
        ).resolve()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"model snapshot is incomplete: {snapshot}")
    if snapshot.name != str(config.model["model_revision"]):
        raise ValueError("generalist-v3 model snapshot revision differs")
    return snapshot


def _sampling_kwargs(sampling: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": int(sampling["candidates_per_task"]),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": int(sampling["top_k"]),
        "max_tokens": int(sampling["max_new_tokens"]),
        "seed": int(sampling["seed"]),
        "ignore_eos": False,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
    }


def _finish_reason(value: str | None) -> str:
    if value == "stop":
        return "eos"
    if value == "length":
        return "token_limit"
    return "unknown" if value is None else value


def _generate_candidates(
    config: GeneralistV3Config,
    manifest: Mapping[str, Any],
    snapshot: Path,
    *,
    adapter_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inference = config.evaluation["inference"]
    if inference["use_flashinfer_sampler"] is False:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != inference["engine_version"]:
        raise RuntimeError(
            "generalist-v3 vLLM version differs: "
            f"{vllm.__version__} != {inference['engine_version']}"
        )
    prompts = [str(task["model_input"]) for task in manifest["tasks"]]
    sampling = dict(manifest["sampling"])
    llm: Any | None = None
    lora_request: Any | None = None
    adapter_binding: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        engine_kwargs: dict[str, Any] = dict(
            model=str(snapshot),
            tokenizer=str(snapshot),
            dtype=str(inference["dtype"]),
            tensor_parallel_size=int(inference["tensor_parallel_size"]),
            gpu_memory_utilization=float(inference["gpu_memory_utilization"]),
            max_model_len=int(inference["max_model_len"]),
            max_num_seqs=int(inference["max_num_seqs"]),
            enforce_eager=bool(inference["enforce_eager"]),
            language_model_only=bool(inference["language_model_only"]),
            enable_prefix_caching=bool(inference["enable_prefix_caching"]),
            seed=int(sampling["seed"]),
            trust_remote_code=False,
        )
        if adapter_dir is not None:
            from vllm.lora.request import LoRARequest

            from .generalist_v3_parity import _adapter_identity
            from .qwen35_vllm_lora import (
                patch_qwen35_vllm_gdn_lora_mapping,
                prepare_qwen35_vllm_adapter,
            )

            identity = _adapter_identity(config, adapter_dir)
            compatibility = prepare_qwen35_vllm_adapter(
                adapter_dir, split_gdn_qkv=True
            )
            mapping_patch = patch_qwen35_vllm_gdn_lora_mapping(
                expected_version=str(inference["engine_version"])
            )
            runtime_adapter = Path(str(compatibility["runtime_adapter_dir"]))
            engine_kwargs.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": int(config.lora["r"]),
                    "max_loras": 1,
                    "worker_cls": (
                        "qwen_lean.qwen35_vllm_worker.Qwen35Vllm017Worker"
                    ),
                }
            )
            lora_request = LoRARequest(
                lora_name="qwen-lean-generalist-v3-checkpoint",
                lora_int_id=1,
                lora_path=str(runtime_adapter),
            )
            adapter_binding = {
                "identity": identity,
                "compatibility": compatibility,
                "vllm_gdn_mapping_patch": mapping_patch,
            }
        llm = LLM(**engine_kwargs)
        outputs = llm.generate(
            prompts,
            SamplingParams(**_sampling_kwargs(sampling)),
            use_tqdm=True,
            **({} if lora_request is None else {"lora_request": lora_request}),
        )
        wall_time = time.perf_counter() - started
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM canary output count differs")
        rows: list[dict[str, Any]] = []
        fallback_latency = wall_time / len(prompts)
        for task, prompt, request in zip(
            manifest["tasks"], prompts, outputs, strict=True
        ):
            completions = sorted(request.outputs, key=lambda item: item.index)
            indices = [item.index for item in completions]
            if (
                request.prompt != prompt
                or indices != list(range(int(sampling["candidates_per_task"])))
            ):
                raise RuntimeError(f"invalid vLLM output for {task['task_id']}")
            latency = fallback_latency
            metrics = request.metrics
            if (
                metrics is not None
                and metrics.finished_time is not None
                and metrics.finished_time >= metrics.arrival_time
            ):
                latency = metrics.finished_time - metrics.arrival_time
            for completion in completions:
                rows.append(
                    {
                        "task_id": str(task["task_id"]),
                        "candidate_id": f"model-{completion.index}",
                        "candidate_index": completion.index,
                        "candidate_text": completion.text,
                        "generated_token_count": len(completion.token_ids),
                        "finish_reason": _finish_reason(completion.finish_reason),
                        "generation_latency_seconds": latency,
                    }
                )
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    return rows, {
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "generation_wall_time_seconds": wall_time,
        "adapter": adapter_binding,
    }


def _verify_candidate(
    verifier: LeanVerifier,
    task: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        outcome = verifier.verify_raw_completion(
            preamble=str(task["preamble"]),
            model_input=str(task["model_input"]),
            candidate=str(generated["candidate_text"]),
        )
        result = CandidateResult(
            task_id=str(generated["task_id"]),
            candidate_id=str(generated["candidate_id"]),
            candidate_index=int(generated["candidate_index"]),
            candidate_text=str(generated["candidate_text"]),
            category=outcome.category,
            lean_exit_code=outcome.lean_exit_code,
            diagnostics=outcome.diagnostics,
            generation_latency_seconds=float(
                generated["generation_latency_seconds"]
            ),
            verification_latency_seconds=outcome.latency_seconds,
            total_latency_seconds=(
                float(generated["generation_latency_seconds"])
                + outcome.latency_seconds
            ),
            generated_token_count=int(generated["generated_token_count"]),
            finish_reason=str(generated["finish_reason"]),
        )
    except Exception as error:
        generation_latency = float(generated["generation_latency_seconds"])
        result = CandidateResult(
            task_id=str(generated["task_id"]),
            candidate_id=str(generated["candidate_id"]),
            candidate_index=int(generated["candidate_index"]),
            candidate_text=str(generated["candidate_text"]),
            category="verifier_error",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": f"{type(error).__name__}: {error}"},
            generation_latency_seconds=generation_latency,
            verification_latency_seconds=time.perf_counter() - started,
            total_latency_seconds=(
                generation_latency + time.perf_counter() - started
            ),
            generated_token_count=int(generated["generated_token_count"]),
            finish_reason=str(generated["finish_reason"]),
        )
    return result.to_dict()


def run_base_validation_canary(
    config: GeneralistV3Config,
    canary_path: Path,
    dataset_manifest_path: Path,
    verifier_root: Path,
    output_dir: Path,
    *,
    model_snapshot: Path | None = None,
    verification_workers: int = 8,
    preamble_probe_timeout_seconds: float = 300.0,
    adapter_dir: Path | None = None,
    checkpoint_id: str = "Base",
    parity_evidence_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _validate_canary_manifest(config, canary_path)
    environment = _validate_verifier_environment(
        verifier_root, dataset_manifest_path
    )
    tasks_by_id = {str(task["task_id"]): task for task in manifest["tasks"]}
    verifier = LeanVerifier(
        verifier_root,
        timeout_seconds=float(config.evaluation["verifier_timeout_seconds"]),
    )
    probe_tasks: dict[str, Mapping[str, Any]] = {}
    for task in manifest["tasks"]:
        probe_tasks.setdefault(str(task["preamble"]), task)
    target_probes = []
    for task in probe_tasks.values():
        probe_failure = verifier.prime_preamble(
            str(task["preamble"]), timeout_seconds=preamble_probe_timeout_seconds
        )
        if probe_failure is not None:
            raise RuntimeError("generalist-v3 verifier preamble probe failed")
        outcome = verifier.verify_raw_completion(
            preamble=str(task["preamble"]),
            model_input=str(task["model_input"]),
            candidate=str(task["target"]),
        )
        if outcome.category != "verified":
            raise RuntimeError(
                "generalist-v3 known-valid canary target failed: "
                f"{task['task_id']} as {outcome.category}"
            )
        target_probes.append(
            {
                "task_id": task["task_id"],
                "category": outcome.category,
                "lean_exit_code": outcome.lean_exit_code,
            }
        )

    parity_gate = None
    if adapter_dir is not None:
        if parity_evidence_path is None:
            raise ValueError("adapter validation canary requires the LoRA parity gate")
        from .generalist_v3_parity import validate_lora_parity_gate

        parity_gate = validate_lora_parity_gate(config, parity_evidence_path)
    elif checkpoint_id != "Base":
        raise ValueError("only Base may run without an adapter")
    snapshot = _snapshot_path(config, model_snapshot)
    generated, generation_runtime = _generate_candidates(
        config, manifest, snapshot, adapter_dir=adapter_dir
    )
    generation_path = output_dir / "generated-candidates.jsonl"
    _write_jsonl(generation_path, generated)
    verification_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=verification_workers) as executor:
        results = list(
            executor.map(
                lambda item: _verify_candidate(
                    verifier, tasks_by_id[str(item["task_id"])], item
                ),
                generated,
            )
        )
    verification_wall_time = time.perf_counter() - verification_started
    results.sort(key=lambda item: (str(item["task_id"]), int(item["candidate_index"])))
    results_path = output_dir / "candidates.jsonl"
    _write_jsonl(results_path, results)
    summary = summarize_canary_candidates(
        results,
        expected_task_ids=[str(task["task_id"]) for task in manifest["tasks"]],
        task_metadata=tasks_by_id,
    )
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    import torch

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    expected_gpu = str(
        config.evaluation["inference"]["expected_cuda_device_name_fragment"]
    )
    if expected_gpu not in device.name:
        raise RuntimeError(f"generalist-v3 evaluation requires Ada, got {device.name}")
    metadata = {
        "schema_version": CANARY_RUN_SCHEMA_VERSION,
        "status": "passed",
        "role": "validation",
        "checkpoint_id": checkpoint_id,
        "model": dict(config.model),
        "model_snapshot": str(snapshot),
        "canary_manifest_sha256": sha256_file(canary_path),
        "ordered_tasks_sha256": manifest["ordered_tasks_sha256"],
        "task_count": len(manifest["tasks"]),
        "candidate_count": len(results),
        "sampling": manifest["sampling"],
        "inference": dict(config.evaluation["inference"]),
        "adapter": generation_runtime.get("adapter"),
        "parity_gate": parity_gate,
        "runtime": {
            **generation_runtime,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda_device": device.name,
            "cuda_total_memory_bytes": device.total_memory,
            "verification_workers": verification_workers,
            "verification_wall_time_seconds": verification_wall_time,
        },
        "verifier_environment": environment,
        "known_valid_target_probes": target_probes,
        "generated_candidates_sha256": sha256_file(generation_path),
        "candidate_results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "category_counts": dict(Counter(str(item["category"]) for item in results)),
        "optimizer_updates": 0,
        "sealed_test_accessed": False,
    }
    _write_json(output_dir / "metadata.json", metadata)
    return {"metadata": metadata, "summary": summary}


def finalize_existing_base_canary(
    config: GeneralistV3Config,
    canary_path: Path,
    dataset_manifest_path: Path,
    verifier_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Recover metadata after a post-verification summary-only failure."""

    manifest = _validate_canary_manifest(config, canary_path)
    environment = _validate_verifier_environment(
        verifier_root, dataset_manifest_path
    )
    generation_path = output_dir / "generated-candidates.jsonl"
    results_path = output_dir / "candidates.jsonl"
    generated = _read_jsonl(generation_path)
    results = _read_jsonl(results_path)
    expected_pairs = {
        (str(task["task_id"]), index)
        for task in manifest["tasks"]
        for index in range(8)
    }
    observed_generation = {
        (str(item["task_id"]), int(item["candidate_index"])) for item in generated
    }
    observed_results = {
        (str(item["task_id"]), int(item["candidate_index"])) for item in results
    }
    if (
        len(generated) != 768
        or len(results) != 768
        or observed_generation != expected_pairs
        or observed_results != expected_pairs
        or any(
            item["category"] in {"generation_error", "verifier_error"}
            for item in results
        )
    ):
        raise ValueError("generalist-v3 Base canary recovery artifacts are incomplete")
    tasks_by_id = {str(task["task_id"]): task for task in manifest["tasks"]}
    summary = summarize_canary_candidates(
        results,
        expected_task_ids=[str(task["task_id"]) for task in manifest["tasks"]],
        task_metadata=tasks_by_id,
    )
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    verifier = LeanVerifier(
        verifier_root,
        timeout_seconds=float(config.evaluation["verifier_timeout_seconds"]),
    )
    probe_tasks: dict[str, Mapping[str, Any]] = {}
    for task in manifest["tasks"]:
        probe_tasks.setdefault(str(task["preamble"]), task)
    target_probes = []
    for task in probe_tasks.values():
        outcome = verifier.verify_raw_completion(
            preamble=str(task["preamble"]),
            model_input=str(task["model_input"]),
            candidate=str(task["target"]),
        )
        if outcome.category != "verified":
            raise RuntimeError("generalist-v3 recovered target probe failed")
        target_probes.append(
            {
                "task_id": task["task_id"],
                "category": outcome.category,
                "lean_exit_code": outcome.lean_exit_code,
            }
        )
    fallback_latencies = {
        float(item["generation_latency_seconds"]) for item in generated
    }
    generation_wall = (
        next(iter(fallback_latencies)) * len(manifest["tasks"])
        if len(fallback_latencies) == 1
        else None
    )
    import torch

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    metadata = {
        "schema_version": CANARY_RUN_SCHEMA_VERSION,
        "status": "passed",
        "role": "validation",
        "checkpoint_id": "Base",
        "model": dict(config.model),
        "model_snapshot": str(_snapshot_path(config, None)),
        "canary_manifest_sha256": sha256_file(canary_path),
        "ordered_tasks_sha256": manifest["ordered_tasks_sha256"],
        "task_count": len(manifest["tasks"]),
        "candidate_count": len(results),
        "sampling": manifest["sampling"],
        "inference": dict(config.evaluation["inference"]),
        "adapter": None,
        "parity_gate": None,
        "runtime": {
            "engine": "vllm",
            "engine_version": config.evaluation["inference"]["engine_version"],
            "generation_wall_time_seconds": generation_wall,
            "generation_wall_time_derivation": (
                "common fallback request latency multiplied by 96 tasks"
            ),
            "verification_wall_time_seconds": None,
            "candidate_verification_seconds_sum": sum(
                float(item["verification_latency_seconds"] or 0.0)
                for item in results
            ),
            "postprocessing_recovery": (
                "raw generation and verification completed; original summary "
                "failed on an unlexable rejected candidate"
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda_device": device.name,
            "cuda_total_memory_bytes": device.total_memory,
            "verification_workers": 8,
        },
        "verifier_environment": environment,
        "known_valid_target_probes": target_probes,
        "generated_candidates_sha256": sha256_file(generation_path),
        "candidate_results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "category_counts": dict(Counter(str(item["category"]) for item in results)),
        "optimizer_updates": 0,
        "sealed_test_accessed": False,
    }
    _write_json(output_dir / "metadata.json", metadata)
    return {"metadata": metadata, "summary": summary}


def _compact_summaries(
    manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task_metadata = {
        str(task["task_id"]): task for task in manifest["tasks"]
    }
    task_ids = [str(task["task_id"]) for task in manifest["tasks"]]
    summary = summarize_canary_candidates(
        results,
        expected_task_ids=task_ids,
        task_metadata=task_metadata,
    )
    structural = {}
    for structural_group in sorted(
        {str(task["structural_class"]) for task in manifest["tasks"]}
    ):
        group_ids = [
            str(task["task_id"])
            for task in manifest["tasks"]
            if task["structural_class"] == structural_group
        ]
        group_id_set = set(group_ids)
        group_results = [
            item for item in results if item["task_id"] in group_id_set
        ]
        structural[structural_group] = summarize_canary_candidates(
            group_results,
            expected_task_ids=group_ids,
            task_metadata=task_metadata,
        )
    by_task: dict[str, Any] = {}
    for task_id in task_ids:
        task = task_metadata[task_id]
        rows = [item for item in results if item["task_id"] == task_id]
        templates = sorted(
            {
                normalized_template_hash(
                    str(task.get("proof_prefix", ""))
                    + str(item["candidate_text"])
                )
                for item in rows
            }
        )
        by_task[task_id] = {
            "interface": task["interface"],
            "structural_group": task["structural_class"],
            "verified_candidates": sum(
                item["category"] == "verified" for item in rows
            ),
            "solved_at_8": any(item["category"] == "verified" for item in rows),
            "normalized_complete_output_templates": templates,
        }
    return summary, structural, by_task


def _prompt_length_evidence(
    config: GeneralistV3Config, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model["tokenizer_id"],
        revision=config.model["tokenizer_revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    rows = [
        {
            "task_id": str(task["task_id"]),
            "input_tokens": len(
                tokenizer.encode(str(task["model_input"]), add_special_tokens=False)
            ),
        }
        for task in manifest["tasks"]
    ]
    maximum = max(rows, key=lambda item: int(item["input_tokens"]))
    maximum_new = int(config.evaluation["sampling"]["max_new_tokens"])
    model_limit = int(config.evaluation["inference"]["max_model_len"])
    if int(maximum["input_tokens"]) + maximum_new > model_limit:
        raise ValueError("generalist-v3 canary exceeds the native evaluation context")
    lengths = [int(item["input_tokens"]) for item in rows]
    return {
        "minimum_input_tokens": min(lengths),
        "maximum_input_tokens": maximum["input_tokens"],
        "maximum_input_task_id": maximum["task_id"],
        "maximum_new_tokens": maximum_new,
        "native_model_context_tokens": model_limit,
        "tasks_above_training_execution_ceiling": sum(
            value > int(config.training["resolved_context_tokens"]) for value in lengths
        ),
        "truncated_tasks": 0,
    }


def compact_base_canary_evidence(
    config: GeneralistV3Config,
    canary_path: Path,
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = _validate_canary_manifest(config, canary_path)
    metadata_path = run_dir / "metadata.json"
    results_path = run_dir / "candidates.jsonl"
    summary_path = run_dir / "summary.json"
    metadata = _read_json(metadata_path)
    raw_summary = _read_json(summary_path)
    if (
        metadata.get("schema_version") != CANARY_RUN_SCHEMA_VERSION
        or metadata.get("status") != "passed"
        or metadata.get("model") != config.model
        or metadata.get("canary_manifest_sha256") != sha256_file(canary_path)
        or metadata.get("ordered_tasks_sha256") != manifest["ordered_tasks_sha256"]
        or metadata.get("candidate_count") != 768
        or metadata.get("candidate_results_sha256") != sha256_file(results_path)
        or metadata.get("summary_sha256") != sha256_file(summary_path)
        or metadata.get("optimizer_updates") != 0
        or metadata.get("sealed_test_accessed") is not False
    ):
        raise ValueError("generalist-v3 Base canary run binding differs")
    results = _read_jsonl(results_path)
    summary, structural, by_task = _compact_summaries(manifest, results)
    prompt_lengths = _prompt_length_evidence(config, manifest)
    if raw_summary.get("combined") != summary.get("combined"):
        # The compact pass may improve complete-output template accounting, but
        # coverage/category aggregates must remain identical to the raw run.
        comparable_keys = ("interface_task_count", "candidate_count", "solved_at_8", "verified_candidates", "verified_density")
        if any(
            raw_summary["combined"].get(key) != summary["combined"].get(key)
            for key in comparable_keys
        ):
            raise ValueError("generalist-v3 Base canary compact summary differs")
    evidence = {
        "schema_version": BASE_CANARY_EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "marker": "OBSERVED",
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "validation_canary_sha256": sha256_file(canary_path),
        "ordered_tasks_sha256": manifest["ordered_tasks_sha256"],
        "interface_tasks": 96,
        "candidates_per_task": 8,
        "candidate_results_sha256": metadata["candidate_results_sha256"],
        "raw_metadata_sha256": sha256_file(metadata_path),
        "raw_summary_sha256": sha256_file(summary_path),
        "summary": summary,
        "structural_summary": structural,
        "task_level": by_task,
        "prompt_lengths": prompt_lengths,
        "category_counts": metadata["category_counts"],
        "inference": metadata["inference"],
        "runtime": metadata["runtime"],
        "verifier_environment": metadata["verifier_environment"],
        "protocol": {
            "all_validation_compositions_evaluated": True,
            "truncation": False,
            "silent_exclusions": 0,
            "optimizer_updates": 0,
            "sealed_test_accessed": False,
        },
    }
    _write_json(output_path, evidence)
    return evidence


def compact_checkpoint_canary_evidence(
    config: GeneralistV3Config,
    canary_path: Path,
    run_dir: Path,
    base_evidence_path: Path,
    anchor_drift_path: Path,
    output_path: Path,
    *,
    configuration_id: str,
    optimizer_step: int,
) -> dict[str, Any]:
    from .generalist_v3 import evaluate_collapse_gates, positive_500_step_gate

    manifest = _validate_canary_manifest(config, canary_path)
    metadata_path = run_dir / "metadata.json"
    results_path = run_dir / "candidates.jsonl"
    summary_path = run_dir / "summary.json"
    metadata = _read_json(metadata_path)
    base = _read_json(base_evidence_path)
    drift = _read_json(anchor_drift_path)
    checkpoint_id = f"{configuration_id}-{optimizer_step}"
    if (
        metadata.get("schema_version") != CANARY_RUN_SCHEMA_VERSION
        or metadata.get("status") != "passed"
        or metadata.get("checkpoint_id") != checkpoint_id
        or metadata.get("adapter") is None
        or metadata.get("parity_gate", {}).get("status") != "passed"
        or metadata.get("candidate_results_sha256") != sha256_file(results_path)
        or metadata.get("summary_sha256") != sha256_file(summary_path)
        or base.get("schema_version") != BASE_CANARY_EVIDENCE_SCHEMA_VERSION
        or base.get("status") != "passed"
        or drift.get("schema_version") != "generalist-v3-anchor-drift-v1"
        or drift.get("configuration_id") != configuration_id
        or drift.get("optimizer_step") != optimizer_step
    ):
        raise ValueError("generalist-v3 checkpoint canary binding differs")
    results = _read_jsonl(results_path)
    summary, structural, task_level = _compact_summaries(manifest, results)
    prompt_lengths = _prompt_length_evidence(config, manifest)
    base_tasks = base["task_level"]
    base_solved = {task_id for task_id, row in base_tasks.items() if row["solved_at_8"]}
    current_solved = {
        task_id for task_id, row in task_level.items() if row["solved_at_8"]
    }
    retained = sorted(base_solved & current_solved)
    new_solves = sorted(current_solved - base_solved)
    overlap_rows = []
    for task_id in sorted(task_level):
        base_templates = set(
            base_tasks[task_id]["normalized_complete_output_templates"]
        )
        current_templates = set(
            task_level[task_id]["normalized_complete_output_templates"]
        )
        union = base_templates | current_templates
        overlap_rows.append(
            {
                "task_id": task_id,
                "shared_templates": len(base_templates & current_templates),
                "union_templates": len(union),
                "jaccard": len(base_templates & current_templates) / len(union),
            }
        )
    gates = evaluate_collapse_gates(
        config,
        summary,
        base["summary"],
        retained_base_solved=len(retained),
    )
    positive = (
        positive_500_step_gate(config, summary, base["summary"], gates)
        if optimizer_step == 500 and configuration_id != "C0"
        else None
    )
    evidence = {
        "schema_version": "generalist-v3-checkpoint-canary-evidence-v1",
        "status": "passed",
        "marker": "OBSERVED",
        "configuration_id": configuration_id,
        "optimizer_step": optimizer_step,
        "checkpoint_id": checkpoint_id,
        "adapter": metadata["adapter"],
        "parity_gate": metadata["parity_gate"],
        "validation_canary_sha256": sha256_file(canary_path),
        "candidate_results_sha256": metadata["candidate_results_sha256"],
        "summary": summary,
        "structural_summary": structural,
        "task_level": task_level,
        "prompt_lengths": prompt_lengths,
        "base_comparison": {
            "base_solved_interface_tasks": len(base_solved),
            "retained_base_solved_interface_tasks": len(retained),
            "retained_base_solved_task_ids": retained,
            "new_solved_interface_tasks": len(new_solves),
            "new_solved_task_ids": new_solves,
            "lost_base_solved_task_ids": sorted(base_solved - current_solved),
            "mean_task_template_jaccard": sum(item["jaccard"] for item in overlap_rows)
            / len(overlap_rows),
            "task_template_overlap": overlap_rows,
        },
        "anchor_drift": drift,
        "collapse_gates": gates,
        "positive_500_step_gate": positive,
        "category_counts": metadata["category_counts"],
        "protocol": {
            "all_validation_compositions_evaluated": True,
            "truncation": False,
            "silent_exclusions": 0,
            "sealed_test_accessed": False,
        },
    }
    _write_json(output_path, evidence)
    return evidence
