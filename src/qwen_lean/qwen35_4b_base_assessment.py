from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .artifacts import read_artifacts, write_artifacts
from .baseline import run_phase1_baseline
from .metrics import summarize_results
from .minif2f import Phase1Config, materialize_benchmark_tasks
from .schema import CandidateResult, RunMetadata


MODEL_ID = "Qwen/Qwen3.5-4B-Base"
MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
VLLM_VERSION = "0.27.2rc1.dev203+g41f179b57"
VLLM_SOURCE_REVISION = "41f179b57aa8ab6f634f508128ce1f1efadd0eb1"
WORKLOAD_TASK_COUNTS = {
    "minif2f-valid-dev16-v1": 16,
    "minif2f-valid-v1": 244,
}


def validate_assessment_config(config: Phase1Config) -> None:
    expected_model = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
    }
    expected_sampling = {
        "candidates_per_task": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "max_new_tokens": 1024,
        "stop": "tokenizer_eos_or_token_limit",
        "seed": 0,
    }
    expected_engine = {
        "name": "vllm",
        "version": VLLM_VERSION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "quantization": None,
        "language_model_only": True,
        "resolve_pinned_snapshot": True,
        "use_flashinfer_sampler": False,
        "worker_multiproc_method": "spawn",
    }
    for section, expected in (
        ("model", expected_model),
        ("sampling", expected_sampling),
        ("engine", expected_engine),
    ):
        actual = config.value[section]
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Qwen3.5 assessment {section} mismatch: {mismatches}")
    if config.value.get("assessment", {}).get("vllm_source_revision") != (
        VLLM_SOURCE_REVISION
    ):
        raise ValueError("Qwen3.5 assessment vLLM source revision mismatch")
    if config.value.get("verifier", {}).get("timeout_seconds") != 30.0:
        raise ValueError("Qwen3.5 assessment verifier timeout must be 30 seconds")


class _GpuMemorySampler:
    def __init__(self, device_index: int = 0, interval_seconds: float = 1.0):
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.samples_mib: list[int] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _GpuMemorySampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.device_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            self.errors.append(completed.stderr.strip() or "nvidia-smi failed")
            return
        try:
            self.samples_mib.append(int(completed.stdout.strip()))
        except ValueError:
            self.errors.append(f"unexpected nvidia-smi output: {completed.stdout!r}")


def run_assessment(
    config: Phase1Config,
    benchmark_root: Path,
    workload_id: str,
    output_dir: Path,
    *,
    timeout_seconds: float,
    verification_workers: int,
) -> tuple[RunMetadata, list[CandidateResult], dict[str, Any]]:
    validate_assessment_config(config)
    required_timeout = float(config.value["verifier"]["timeout_seconds"])
    if timeout_seconds != required_timeout:
        raise ValueError(
            "Qwen3.5-4B-Base verifier timeout mismatch: "
            f"{timeout_seconds!r} != {required_timeout!r}"
        )
    if workload_id not in WORKLOAD_TASK_COUNTS:
        raise ValueError(f"unknown Qwen3.5 assessment workload: {workload_id}")

    vllm_environment = {
        "VLLM_USE_FLASHINFER_SAMPLER": (
            "1" if config.engine["use_flashinfer_sampler"] else "0"
        ),
        "VLLM_WORKER_MULTIPROC_METHOD": str(
            config.engine["worker_multiproc_method"]
        ),
    }
    for key, value in vllm_environment.items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(f"{key}={existing!r} conflicts with required {value!r}")
        os.environ[key] = value

    with _GpuMemorySampler() as memory:
        metadata, results, summary = run_phase1_baseline(
            config,
            benchmark_root,
            workload_id,
            output_dir,
            timeout_seconds=timeout_seconds,
            verification_workers=verification_workers,
        )
    if not memory.samples_mib:
        raise RuntimeError(f"GPU peak-memory sampling failed: {memory.errors}")

    import huggingface_hub
    import transformers

    runtime = {
        **metadata.runtime,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "peak_gpu_memory_mib": max(memory.samples_mib),
        "peak_gpu_memory_bytes": max(memory.samples_mib) * 1024 * 1024,
        "gpu_memory_sample_count": len(memory.samples_mib),
        "gpu_memory_sampling_interval_seconds": memory.interval_seconds,
        "gpu_memory_measurement": "device memory.used sampled with nvidia-smi",
        "vllm_environment": vllm_environment,
    }
    updated_metadata = replace(metadata, runtime=runtime)
    write_artifacts(output_dir, updated_metadata, results, summary=summary)
    return updated_metadata, results, summary


def write_compact_evidence(
    config: Phase1Config,
    benchmark_root: Path,
    environment_validation_path: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    validate_assessment_config(config)
    all_tasks = materialize_benchmark_tasks(config, benchmark_root)
    dev16 = _compact_workload(config, all_tasks, dev16_dir, "minif2f-valid-dev16-v1")
    full = _compact_workload(config, all_tasks, full_dir, "minif2f-valid-v1")
    environment = json.loads(environment_validation_path.read_text(encoding="utf-8"))
    if environment["known_valid_candidate_category"] != "verified":
        raise ValueError("known-valid verifier probe did not pass")
    if environment["placeholder_candidate_category"] != "lean_rejected":
        raise ValueError("placeholder verifier probe was not rejected")

    preflight = {
        "schema_version": "qwen35-4b-base-preflight-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "inference_stack": dev16["inference_stack"],
        "gpu": dev16["gpu"],
        "precision_lane": "bfloat16",
        "bf16_dev16_fit_cleanly": True,
        "compatibility": {
            "required_change": True,
            "vllm_source_revision": VLLM_SOURCE_REVISION,
            "language_model_only": True,
            "model_artifact_resolution": "pinned_local_snapshot",
            "flashinfer_sampler": False,
            "worker_multiproc_method": "spawn",
            "scope": "isolated Qwen3.5 evaluation stack and transport settings",
        },
        "environment_validation": environment,
    }
    comparison = {
        "schema_version": "qwen35-4b-base-assessment-v1",
        "assessment_id": config.value["assessment"]["id"],
        "strict_lane": {
            "prompt_format_id": "whole-proof-v1",
            "candidate_handling": "raw_continuation_no_extraction_repair_or_feedback",
            "chat_template": None,
            "sampling": config.sampling,
            "verifier_timeout_seconds": config.value["verifier"]["timeout_seconds"],
            "verifier_timeout_semantics": "unsuccessful_proof_outcome",
        },
        "dev16": dev16,
        "full": full,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_dir / "preflight.json", preflight)
    _write_json(evidence_dir / "dev16.json", dev16)
    _write_json(evidence_dir / "full.json", full)
    _write_json(evidence_dir / "comparison.json", comparison)
    (evidence_dir / "README.md").write_text(
        _readme(dev16, full), encoding="utf-8"
    )
    return comparison


def _compact_workload(
    config: Phase1Config,
    all_tasks: list[Any],
    artifact_dir: Path,
    workload_id: str,
) -> dict[str, Any]:
    metadata, results = read_artifacts(artifact_dir)
    stored_summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    tasks = config.select_workload(workload_id, all_tasks)
    expected_task_count = WORKLOAD_TASK_COUNTS[workload_id]
    expected_candidate_count = expected_task_count * 4
    recomputed = summarize_results(
        results,
        expected_task_ids=[task.id for task in tasks],
        candidates_per_task=4,
    )
    for key in (
        "complete",
        "completeness_errors",
        "task_count",
        "candidate_count",
        "candidates_per_task",
        "tasks_with_verified_candidate",
        "pass_at_k",
        "category_counts",
        "finish_reason_counts",
        "verifier_timeout_count",
        "infrastructure_error_count",
    ):
        if stored_summary.get(key) != recomputed.get(key):
            raise ValueError(f"stored {workload_id} summary differs at {key}")
    if not recomputed["complete"] or recomputed["infrastructure_error_count"] != 0:
        raise ValueError(f"{workload_id} is not a complete infrastructure-clean run")
    if recomputed["task_count"] != expected_task_count:
        raise ValueError(f"{workload_id} task count mismatch")
    if recomputed["candidate_count"] != expected_candidate_count:
        raise ValueError(f"{workload_id} candidate count mismatch")
    _validate_run_contract(config, metadata, workload_id)

    token_counts = [result.generated_token_count for result in results]
    if any(value is None for value in token_counts):
        raise ValueError(f"{workload_id} has missing generated-token counts")
    tokens = [int(value) for value in token_counts if value is not None]
    generation_seconds = float(metadata.runtime["generation_wall_time_seconds"])
    verification_seconds = float(metadata.runtime["verification_wall_time_seconds"])
    run_seconds = float(stored_summary["run_wall_time_seconds"])
    solved = int(recomputed["tasks_with_verified_candidate"]["count"])
    generation_projection = [
        {
            "task_id": result.task_id,
            "candidate_id": result.candidate_id,
            "candidate_index": result.candidate_index,
            "candidate_text": result.candidate_text,
            "generation_latency_seconds": result.generation_latency_seconds,
            "generated_token_count": result.generated_token_count,
            "finish_reason": result.finish_reason,
        }
        for result in results
    ]
    generation_digest = hashlib.sha256(
        json.dumps(
            generation_projection, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "qwen35-4b-base-workload-v1",
        "workload_id": workload_id,
        "task_count": recomputed["task_count"],
        "candidate_count": recomputed["candidate_count"],
        "candidates_per_task": recomputed["candidates_per_task"],
        "tasks_with_verified_candidate": recomputed["tasks_with_verified_candidate"],
        "pass_at_k": recomputed["pass_at_k"],
        "category_counts": recomputed["category_counts"],
        "category_fractions": recomputed["category_fractions"],
        "finish_reason_counts": recomputed["finish_reason_counts"],
        "verifier_timeout_count": recomputed["verifier_timeout_count"],
        "verifier_timeout_semantics": "unsuccessful_proof_outcome",
        "infrastructure_error_count": recomputed["infrastructure_error_count"],
        "generated_tokens": {
            "total": sum(tokens),
            "minimum": min(tokens),
            "maximum": max(tokens),
            "mean": fmean(tokens),
            "median": median(tokens),
        },
        "timing_seconds": {
            "generation_wall": generation_seconds,
            "verification_wall": verification_seconds,
            "run_wall": run_seconds,
            "generation_candidate_latency_mean": recomputed["timing_seconds"][
                "generation_candidate_latency_mean"
            ],
            "verification_candidate_latency_mean": recomputed["timing_seconds"][
                "verification_latency_mean"
            ],
            "run_wall_per_solved_task": run_seconds / solved if solved else None,
            "generation_wall_per_solved_task": (
                generation_seconds / solved if solved else None
            ),
        },
        "throughput": {
            "generated_tokens_per_second": sum(tokens) / generation_seconds,
            "candidates_per_second": len(results) / generation_seconds,
        },
        "gpu": {
            key: metadata.runtime[key]
            for key in (
                "cuda_device",
                "cuda_device_capability",
                "cuda_device_total_memory_bytes",
                "peak_gpu_memory_mib",
                "peak_gpu_memory_bytes",
                "gpu_memory_sample_count",
                "gpu_memory_sampling_interval_seconds",
                "gpu_memory_measurement",
            )
        },
        "inference_stack": {
            "python": metadata.runtime["python"],
            "torch": metadata.runtime["torch"],
            "torch_cuda_version": metadata.runtime["torch_cuda_version"],
            "transformers": metadata.runtime["transformers"],
            "huggingface_hub": metadata.runtime["huggingface_hub"],
            "engine": metadata.inference_engine,
            "engine_version": metadata.inference_engine_version,
            "vllm_source_revision": VLLM_SOURCE_REVISION,
        },
        "execution_identity": {
            "model_id": metadata.model_id,
            "model_revision": metadata.model_revision,
            "tokenizer_id": metadata.tokenizer_id,
            "tokenizer_revision": metadata.tokenizer_revision,
            "prompt_format_id": metadata.prompt_format_id,
            "generation_settings": metadata.generation_settings,
            "benchmark_repository": metadata.benchmark_repository,
            "benchmark_revision": metadata.benchmark_revision,
            "benchmark_split": metadata.benchmark_split,
            "lean_toolchain": metadata.lean_toolchain,
            "mathlib_revision": metadata.mathlib_revision,
            "verifier_environment": metadata.verifier_environment,
            "inference_execution": metadata.runtime["inference_execution"],
            "vllm_environment": metadata.runtime["vllm_environment"],
        },
        "generation_identity_sha256": generation_digest,
    }


def _validate_run_contract(
    config: Phase1Config, metadata: RunMetadata, workload_id: str
) -> None:
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "workload_id": workload_id,
        "prompt_format_id": "whole-proof-v1",
        "candidates_per_task": 4,
        "inference_engine": "vllm",
        "inference_engine_version": VLLM_VERSION,
        "adapter_enabled": False,
    }
    for field, value in expected.items():
        if getattr(metadata, field) != value:
            raise ValueError(
                f"{workload_id} run contract mismatch for {field}: "
                f"{getattr(metadata, field)!r} != {value!r}"
            )
    required_timeout = float(config.value["verifier"]["timeout_seconds"])
    if metadata.verifier_timeout_seconds != required_timeout:
        raise ValueError(
            f"{workload_id} verifier timeout mismatch: "
            f"{metadata.verifier_timeout_seconds!r} != {required_timeout!r}"
        )
    settings = metadata.generation_settings or {}
    for key, value in {
        **config.sampling,
        "dtype": "bfloat16",
        "quantization": None,
        "language_model_only": True,
        "model_artifact_resolution": "pinned_local_snapshot",
        "chat_template": None,
        "prompt_transformation": None,
    }.items():
        if settings.get(key) != value:
            raise ValueError(f"{workload_id} generation setting mismatch: {key}")
    if metadata.runtime.get("inference_execution") != "local_cuda":
        raise ValueError(f"{workload_id} did not use local CUDA inference")
    if metadata.runtime.get("vllm_environment") != {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }:
        raise ValueError(f"{workload_id} vLLM environment mismatch")


def _readme(dev16: dict[str, Any], full: dict[str, Any]) -> str:
    full_solved = full["tasks_with_verified_candidate"]["count"]
    compute = full["timing_seconds"]["run_wall_per_solved_task"]
    compute_text = "not available (zero solved tasks)" if compute is None else f"{compute:.2f} s"
    return f"""# Qwen3.5-4B-Base foundation assessment

`OBSERVED`: the strict local-GPU assessment completed dev16 and all 244 miniF2F
validation tasks with four raw whole-proof candidates per task and zero unresolved
generation or verifier infrastructure errors.

| Workload | Tasks | Candidates | Solved tasks | pass@1 | pass@4 | Peak GPU MiB | Generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | {dev16['task_count']} | {dev16['candidate_count']} | {dev16['tasks_with_verified_candidate']['count']} | {dev16['pass_at_k']['pass@1']:.6f} | {dev16['pass_at_k']['pass@4']:.6f} | {dev16['gpu']['peak_gpu_memory_mib']} | {dev16['generated_tokens']['total']} |
| full validation | {full['task_count']} | {full['candidate_count']} | {full_solved} | {full['pass_at_k']['pass@1']:.6f} | {full['pass_at_k']['pass@4']:.6f} | {full['gpu']['peak_gpu_memory_mib']} | {full['generated_tokens']['total']} |

`ACCEPTED`: this casting assessment used `Qwen/Qwen3.5-4B-Base` and its
tokenizer at `{MODEL_REVISION}`, BF16, temperature 0.8, top-p 0.95, no top-k,
a 1,024-token generation limit, and seed 0. Prompts remained exact
`whole-proof-v1` raw continuations with no chat template, extraction, repair, or
Lean feedback. Verifier timeouts remain unsuccessful proof outcomes.

`OBSERVED`: the full run took {full['timing_seconds']['run_wall']:.2f} seconds
({full['timing_seconds']['generation_wall']:.2f} generation and
{full['timing_seconds']['verification_wall']:.2f} verification), generated
{full['throughput']['generated_tokens_per_second']:.2f} tokens/s, and used
{compute_text} of total measured run wall time per solved task. Device-level peak
memory was {full['gpu']['peak_gpu_memory_mib']} MiB on
{full['gpu']['cuda_device']}.

`OBSERVED`: Qwen3.5 support required the isolated vLLM build
`{VLLM_VERSION}` at `{VLLM_SOURCE_REVISION}`. Text-only
`language_model_only=true` omitted the unused vision encoder while preserving the
language model and raw continuation contract. The pinned Hub snapshot was
resolved to a local path to avoid revision loss inside the worker, and native
top-p sampling was selected because FlashInfer JIT required an unavailable CUDA
toolkit. Raw candidates, model caches, and bulky logs remain local and ignored
by Git.
"""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
