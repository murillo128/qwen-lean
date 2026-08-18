from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable

from .artifacts import read_artifacts, write_artifacts
from .baseline import (
    _generate_candidates,
    _local_cuda_runtime,
    run_phase1_baseline,
    validate_minif2f_environment,
)
from .minif2f import Phase1Config, materialize_benchmark_tasks
from .prompt import PROMPT_FORMAT_ID, render_prompt


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
ENGINE_NAME = "vllm"
ENGINE_VERSION = "0.23.0"
ASSESSMENT_SCHEMA_VERSION = "qwen35-2b-assessment-evidence-v1"
EXPECTED_SAMPLING = {
    "candidates_per_task": 4,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": -1,
    "max_new_tokens": 1024,
    "stop": "tokenizer_eos_or_token_limit",
    "seed": 0,
}
PACKAGE_NAMES = (
    "vllm",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
)

EXPECTED_STRICT_LANE = {
    "prompt_format_id": PROMPT_FORMAT_ID,
    "chat_template": None,
    "prompt_transformation": None,
    "proof_extraction": False,
    "verifier_feedback": False,
    "repair": False,
    "add_special_tokens": False,
    "tokenizer_default_matches_add_special_tokens_false": True,
}

EXPECTED_PREFLIGHT_ENGINE = {
    "name": ENGINE_NAME,
    "version": ENGINE_VERSION,
    "dtype": "bfloat16",
    "language_model_only": True,
    "sampler_backend": "native",
}


class Qwen35AssessmentConfig:
    def __init__(self, phase1: Phase1Config) -> None:
        self.phase1 = phase1
        self.path = phase1.path
        self.value = phase1.value
        self._validate()

    @classmethod
    def load(cls, path: Path) -> Qwen35AssessmentConfig:
        return cls(Phase1Config.load(path))

    @property
    def assessment(self) -> dict[str, Any]:
        return self.value["assessment"]

    def _validate(self) -> None:
        model = self.phase1.model
        required_model = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "add_special_tokens": False,
            "chat_template": None,
        }
        if model != required_model:
            raise ValueError("Qwen3.5 assessment model/tokenizer contract differs")
        if self.phase1.sampling != EXPECTED_SAMPLING:
            raise ValueError("Qwen3.5 assessment sampling contract differs")
        engine = self.phase1.engine
        required_engine = {
            "name": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 2048,
            "max_num_seqs": 32,
            "enforce_eager": True,
            "quantization": None,
            "language_model_only": True,
            "use_flashinfer_sampler": False,
            "expected_cuda_device_name_fragment": "Ada",
        }
        if engine != required_engine:
            raise ValueError("Qwen3.5 assessment inference-engine contract differs")
        expected_workloads = {
            "minif2f-valid-dev16-v1": 16,
            "minif2f-valid-v1": 244,
        }
        observed_workloads = {
            name: int(self.value["workloads"][name]["expected_task_count"])
            for name in expected_workloads
        }
        if observed_workloads != expected_workloads:
            raise ValueError("Qwen3.5 assessment workload contract differs")
        if int(self.assessment["preflight_max_new_tokens"]) < 1:
            raise ValueError("preflight_max_new_tokens must be positive")


MemoryQuery = Callable[[int], tuple[int, int]]


def _query_gpu_memory_mib(device_index: int) -> tuple[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi memory query failed: {completed.stderr.strip()}")
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"unexpected nvidia-smi memory output: {completed.stdout!r}")
    return int(fields[0]), int(fields[1])


class GpuMemoryMonitor:
    def __init__(
        self,
        device_index: int,
        *,
        interval_seconds: float = 0.5,
        query: MemoryQuery = _query_gpu_memory_mib,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU memory monitor interval must be positive")
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.query = query
        self.baseline_used_mib = 0
        self.peak_used_mib = 0
        self.total_mib = 0
        self.sample_count = 0
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        used, total = self.query(self.device_index)
        self.peak_used_mib = max(self.peak_used_mib, used)
        self.total_mib = total
        self.sample_count += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except Exception as error:  # Preserve the benchmark; report after it exits.
                self._error = error
                return

    def __enter__(self) -> GpuMemoryMonitor:
        self._sample()
        self.baseline_used_mib = self.peak_used_mib
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, error_type: object, error: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._error is None:
            self._sample()
        if error is None and self._error is not None:
            raise RuntimeError("GPU memory monitor failed") from self._error

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_index": self.device_index,
            "baseline_used_mib": self.baseline_used_mib,
            "peak_used_mib": self.peak_used_mib,
            "peak_incremental_mib": self.peak_used_mib - self.baseline_used_mib,
            "device_total_mib": self.total_mib,
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self.sample_count,
        }


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def _validate_package_versions(versions: dict[str, str]) -> None:
    observed = versions.get("vllm")
    if observed != ENGINE_VERSION:
        raise RuntimeError(
            f"vLLM version mismatch: expected {ENGINE_VERSION}, got {observed}"
        )


def _validate_preflight_evidence(preflight: dict[str, Any]) -> None:
    if preflight.get("status") != "passed":
        raise ValueError("Qwen3.5 assessment requires a passed real BF16 preflight")
    if preflight.get("model") != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
    }:
        raise ValueError("preflight model/tokenizer identity differs from contract")
    if preflight.get("strict_lane") != EXPECTED_STRICT_LANE:
        raise ValueError("preflight strict-lane evidence differs from contract")
    if preflight.get("engine") != EXPECTED_PREFLIGHT_ENGINE:
        raise ValueError("preflight engine evidence differs from contract")
    runtime = preflight.get("runtime", {})
    if (
        runtime.get("inference_execution") != "local_cuda"
        or runtime.get("bf16_supported") is not True
        or "Ada" not in str(runtime.get("cuda_device", ""))
    ):
        raise ValueError("preflight local Ada BF16 runtime differs from contract")
    _validate_package_versions(preflight.get("packages", {}))


def run_preflight(
    config: Qwen35AssessmentConfig,
    benchmark_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    environment = validate_minif2f_environment(
        config.phase1,
        benchmark_root,
        timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
    )
    runtime = _local_cuda_runtime(config.phase1)
    versions = package_versions()
    _validate_package_versions(versions)

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("project GPU does not support BF16")
    tasks = config.phase1.select_workload(
        "minif2f-valid-dev16-v1",
        materialize_benchmark_tasks(config.phase1, benchmark_root),
    )
    task = tasks[0]
    prompt = render_prompt(task)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    default_ids = tokenizer.encode(prompt)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if default_ids != raw_ids:
        raise RuntimeError("tokenizer default adds tokens to the raw continuation prompt")
    sampling = {
        **EXPECTED_SAMPLING,
        "candidates_per_task": 1,
        "max_new_tokens": int(config.assessment["preflight_max_new_tokens"]),
    }
    with GpuMemoryMonitor(int(runtime["cuda_device_index"])) as memory:
        generated, engine_version = _generate_candidates(
            config.phase1,
            [task],
            prompts=[prompt],
            sampling=sampling,
        )
    candidate = generated[0]
    if engine_version != ENGINE_VERSION:
        raise RuntimeError(
            f"preflight engine mismatch: expected {ENGINE_VERSION}, got {engine_version}"
        )
    if candidate.generation_error is not None:
        raise RuntimeError(f"real BF16 generation failed: {candidate.generation_error}")
    if candidate.token_count < 1:
        raise RuntimeError("real BF16 generation returned no tokens")

    evidence = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": "passed",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
        },
        "strict_lane": {
            **EXPECTED_STRICT_LANE,
        },
        "engine": {
            "name": ENGINE_NAME,
            "version": engine_version,
            "dtype": config.phase1.engine["dtype"],
            "language_model_only": config.phase1.engine["language_model_only"],
            "sampler_backend": "native",
        },
        "packages": versions,
        "runtime": {
            **runtime,
            "python": platform.python_version(),
            "bf16_supported": True,
            "gpu_memory": memory.to_dict(),
        },
        "benchmark_environment": environment,
        "generation_probe": {
            "task_id": task.id,
            "prompt_token_count": len(raw_ids),
            "generated_token_count": candidate.token_count,
            "finish_reason": candidate.finish_reason,
            "max_new_tokens": sampling["max_new_tokens"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def generated_token_summary(token_counts: list[int]) -> dict[str, Any]:
    if not token_counts:
        raise ValueError("generated token summary requires at least one candidate")
    ordered = sorted(token_counts)

    def percentile(fraction: float) -> int:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[min(len(ordered) - 1, index)]

    return {
        "total": sum(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": fmean(ordered),
        "median": median(ordered),
        "p95_nearest_rank": percentile(0.95),
    }


def run_assessment(
    config: Qwen35AssessmentConfig,
    benchmark_root: Path,
    workload_id: str,
    preflight_dir: Path,
    output_dir: Path,
    *,
    verification_workers: int,
) -> tuple[Any, list[Any], dict[str, Any]]:
    preflight = json.loads(
        (preflight_dir / "preflight.json").read_text(encoding="utf-8")
    )
    _validate_preflight_evidence(preflight)
    device_index = int(preflight["runtime"]["cuda_device_index"])
    with GpuMemoryMonitor(device_index) as memory:
        metadata, results, summary = run_phase1_baseline(
            config.phase1,
            benchmark_root,
            workload_id,
            output_dir,
            timeout_seconds=float(config.value["verifier"]["timeout_seconds"]),
            verification_workers=verification_workers,
        )
    token_summary = generated_token_summary(
        [int(result.generated_token_count or 0) for result in results]
    )
    generation_wall = float(metadata.runtime["generation_wall_time_seconds"])
    run_wall = float(summary["run_wall_time_seconds"])
    solved = int(summary["tasks_with_verified_candidate"]["count"])
    runtime = {
        **metadata.runtime,
        "gpu_memory": memory.to_dict(),
        "package_versions": package_versions(),
    }
    metadata = replace(metadata, runtime=runtime)
    summary = {
        **summary,
        "generated_tokens": token_summary,
        "generation_throughput_tokens_per_second": (
            token_summary["total"] / generation_wall
        ),
        "generation_seconds_per_solved_task": (
            generation_wall / solved if solved else None
        ),
        "run_seconds_per_solved_task": run_wall / solved if solved else None,
        "gpu_memory": memory.to_dict(),
        "verifier_timeout_semantics": "unsuccessful_proof_attempt",
        "verifier_timeout_is_infrastructure_error": False,
    }
    write_artifacts(output_dir, metadata, results, summary=summary)
    return metadata, results, summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_workload(
    config: Qwen35AssessmentConfig,
    artifact_dir: Path,
    workload_id: str,
) -> dict[str, Any]:
    metadata, results = read_artifacts(artifact_dir)
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    expected_tasks = int(
        config.value["workloads"][workload_id]["expected_task_count"]
    )
    expected_candidates = expected_tasks * EXPECTED_SAMPLING["candidates_per_task"]
    if metadata.model_id != MODEL_ID or metadata.model_revision != MODEL_REVISION:
        raise ValueError(f"{workload_id} model identity differs from contract")
    if (
        metadata.tokenizer_id != MODEL_ID
        or metadata.tokenizer_revision != MODEL_REVISION
    ):
        raise ValueError(f"{workload_id} tokenizer identity differs from contract")
    if metadata.prompt_format_id != PROMPT_FORMAT_ID:
        raise ValueError(f"{workload_id} prompt format differs from contract")
    if (
        metadata.inference_engine != ENGINE_NAME
        or metadata.inference_engine_version != ENGINE_VERSION
    ):
        raise ValueError(f"{workload_id} engine identity differs from contract")
    if metadata.workload_id != workload_id:
        raise ValueError(f"expected workload {workload_id}, got {metadata.workload_id}")
    if metadata.generation_settings is None:
        raise ValueError(f"{workload_id} has no generation settings")
    for key, expected in EXPECTED_SAMPLING.items():
        if metadata.generation_settings.get(key) != expected:
            raise ValueError(f"{workload_id} generation setting differs: {key}")
    if metadata.generation_settings.get("chat_template") is not None:
        raise ValueError(f"{workload_id} unexpectedly used a chat template")
    if metadata.generation_settings.get("prompt_transformation") is not None:
        raise ValueError(f"{workload_id} unexpectedly transformed prompts")
    if metadata.generation_settings.get("dtype") != "bfloat16":
        raise ValueError(f"{workload_id} did not use BF16")
    if metadata.generation_settings.get("add_special_tokens") is not False:
        raise ValueError(f"{workload_id} changed raw tokenizer input")
    if metadata.generation_settings.get("sampler_backend") != "native":
        raise ValueError(f"{workload_id} did not use the frozen sampler backend")
    if not summary.get("complete"):
        raise ValueError(f"{workload_id} is incomplete: {summary['completeness_errors']}")
    if int(summary["task_count"]) != expected_tasks:
        raise ValueError(f"{workload_id} task count differs from contract")
    if len(results) != expected_candidates:
        raise ValueError(f"{workload_id} candidate count differs from contract")
    if int(summary["infrastructure_error_count"]) != 0:
        raise ValueError(f"{workload_id} has unresolved infrastructure errors")
    pass_at_k = summary.get("pass_at_k")
    if set(pass_at_k or {}) != {"pass@1", "pass@4"}:
        raise ValueError(f"{workload_id} pass@k set differs from contract")
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": "passed",
        "workload_id": workload_id,
        "task_count": summary["task_count"],
        "candidate_count": summary["candidate_count"],
        "candidates_per_task": summary["candidates_per_task"],
        "verified_candidate_count": summary["category_counts"]["verified"],
        "tasks_with_verified_candidate": summary["tasks_with_verified_candidate"],
        "pass_at_k": pass_at_k,
        "category_counts": summary["category_counts"],
        "category_fractions": summary["category_fractions"],
        "finish_reason_counts": summary["finish_reason_counts"],
        "generated_tokens": summary["generated_tokens"],
        "timing_seconds": {
            **summary["timing_seconds"],
            "generation_wall": metadata.runtime["generation_wall_time_seconds"],
            "verification_wall": metadata.runtime["verification_wall_time_seconds"],
            "run_wall": summary["run_wall_time_seconds"],
            "generation_seconds_per_solved_task": summary[
                "generation_seconds_per_solved_task"
            ],
            "run_seconds_per_solved_task": summary["run_seconds_per_solved_task"],
        },
        "generation_throughput_tokens_per_second": summary[
            "generation_throughput_tokens_per_second"
        ],
        "gpu_memory": summary["gpu_memory"],
        "category_semantics": {
            "verifier_timeout": "unsuccessful_proof_attempt",
            "verifier_timeout_is_infrastructure_error": False,
        },
        "infrastructure_error_count": summary["infrastructure_error_count"],
        "candidate_records_manifest_sha256": _sha256(
            artifact_dir / "results.jsonl"
        ),
        "model": {
            "id": metadata.model_id,
            "revision": metadata.model_revision,
            "tokenizer_id": metadata.tokenizer_id,
            "tokenizer_revision": metadata.tokenizer_revision,
        },
        "benchmark": {
            "repository": metadata.benchmark_repository,
            "revision": metadata.benchmark_revision,
            "split": metadata.benchmark_split,
            "verifier_environment": metadata.verifier_environment,
            "verifier_timeout_seconds": metadata.verifier_timeout_seconds,
        },
        "generation_contract": metadata.generation_settings,
        "runtime": {
            "inference_execution": metadata.runtime["inference_execution"],
            "cuda_device": metadata.runtime["cuda_device"],
            "cuda_device_capability": metadata.runtime["cuda_device_capability"],
            "torch_cuda_version": metadata.runtime["torch_cuda_version"],
            "package_versions": metadata.runtime["package_versions"],
        },
    }


def write_compact_evidence(
    config: Qwen35AssessmentConfig,
    preflight_dir: Path,
    dev16_dir: Path,
    full_dir: Path,
    evidence_dir: Path,
    reference_sft_evidence: Path,
) -> dict[str, Any]:
    preflight = json.loads(
        (preflight_dir / "preflight.json").read_text(encoding="utf-8")
    )
    _validate_preflight_evidence(preflight)
    dev16 = _compact_workload(config, dev16_dir, "minif2f-valid-dev16-v1")
    full = _compact_workload(config, full_dir, "minif2f-valid-v1")
    reference = json.loads(reference_sft_evidence.read_text(encoding="utf-8"))
    reference_summary = reference["summary"]
    if not reference_summary.get("phase5_minif2f_passed"):
        raise ValueError("reference-sft-v1 evidence is not accepted")
    reference_metrics = reference_summary["pass_at_k"]
    current_metrics = full["pass_at_k"]
    comparison = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": "passed",
        "workload_id": "minif2f-valid-v1",
        "qwen35_2b": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "candidates_per_task": 4,
            "pass_at_k": current_metrics,
            "tasks_with_verified_candidate": full[
                "tasks_with_verified_candidate"
            ]["count"],
        },
        "reference_sft_v1": {
            "model_id": config.assessment["reference_sft"]["base_model_id"],
            "base_model_revision": config.assessment["reference_sft"][
                "base_model_revision"
            ],
            "adapter_id": reference_summary["adapter_id"],
            "adapter_hub_revision": config.assessment["reference_sft"][
                "adapter_hub_revision"
            ],
            "candidates_per_task": reference_summary["candidates_per_task"],
            "pass_at_k": {
                key: reference_metrics[key] for key in ("pass@1", "pass@4")
            },
            "tasks_with_verified_candidate": reference_summary[
                "tasks_with_verified_candidate"
            ]["count"],
            "source_evidence": str(config.assessment["reference_sft"]["evidence"]),
        },
        "delta_qwen35_2b_minus_reference_sft_v1": {
            key: current_metrics[key] - reference_metrics[key]
            for key in ("pass@1", "pass@4")
        },
        "ratio_qwen35_2b_over_reference_sft_v1": {
            key: current_metrics[key] / reference_metrics[key]
            if reference_metrics[key]
            else None
            for key in ("pass@1", "pass@4")
        },
        "comparison_caveat": (
            "Both lanes use whole-proof-v1, the same miniF2F validation tasks, "
            "temperature 0.8, top-p 0.95, no top-k, max_new_tokens 1024, seed 0, "
            "and unchanged Lean verification. Qwen3.5-2B uses the issue-mandated four "
            "candidates per task, while the retained reference-sft-v1 evidence uses eight."
        ),
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("preflight.json", preflight),
        ("dev16.json", dev16),
        ("full.json", full),
        ("comparison.json", comparison),
    ):
        (evidence_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    metrics = full["pass_at_k"]
    deltas = comparison["delta_qwen35_2b_minus_reference_sft_v1"]
    readme = f"""# Qwen3.5-2B efficiency baseline

**ACCEPTED:** the strict lane used the immutable official `{MODEL_ID}` checkpoint and tokenizer at `{MODEL_REVISION}` in BF16 on the project RTX 4000 Ada. Inputs were exact `whole-proof-v1` raw continuations with no chat wrapper, proof extraction, verifier feedback, or repair.

**OBSERVED:** dev16 completed {dev16['task_count']} tasks and {dev16['candidate_count']} candidates with pass@1 {dev16['pass_at_k']['pass@1']:.6f} and pass@4 {dev16['pass_at_k']['pass@4']:.6f}.

**OBSERVED:** the complete miniF2F validation run completed {full['task_count']} tasks and {full['candidate_count']} candidates. It verified {full['verified_candidate_count']} candidates across {full['tasks_with_verified_candidate']['count']} tasks, with pass@1 {metrics['pass@1']:.6f} and pass@4 {metrics['pass@4']:.6f}.

**OBSERVED:** versus retained `reference-sft-v1`, the Qwen3.5-2B pass@1/pass@4 deltas were {deltas['pass@1']:.6f}/{deltas['pass@4']:.6f}. The strict lane used four candidates per task as mandated here; the retained reference used eight, while the shared prompt, sampling parameters, tasks, and verifier semantics remained aligned.

**OBSERVED:** generation produced {full['generated_tokens']['total']} tokens in {full['timing_seconds']['generation_wall']:.3f} seconds ({full['generation_throughput_tokens_per_second']:.3f} tokens/second, including engine initialization). Peak observed GPU memory was {full['gpu_memory']['peak_used_mib']} MiB. Full latency, finish-reason, token, category, and compute-per-solved-task evidence is retained in `full.json`.

**ACCEPTED:** the run completed with zero generation/verifier infrastructure errors. `lean_rejected`, `empty_candidate`, and `verifier_timeout` are unsuccessful proof outcomes; verifier timeouts do not count as infrastructure errors or authorize regeneration.
"""
    (evidence_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison
