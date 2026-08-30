from __future__ import annotations

import asyncio
import gc
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline import _GpuMemoryMonitor
from .native_thinking_assessment import (
    MODEL_ID,
    MODEL_REVISION,
    WORKLOADS,
    MathiaTask,
    NativeThinkingConfig,
    _append_jsonl,
    _atomic_write_json,
    _configure_runtime,
    _distribution,
    _file_sha256,
    _finish_reason,
    _load_tokenizer,
    _local_runtime,
    _package_versions,
    _resolve_model_snapshot,
    _sha256_json,
    _sha256_text,
    load_mathia_tasks,
    validate_lean_environments,
)
from .thinking_budget_continuation import (
    LEAN_WRAPPER_NORMALIZATION,
    ThinkingBudgetContinuationConfig,
    _format_diagnostics,
    _verify_exact_final,
    canonical_parser_runtime_identity,
    load_continuation_generation_records,
    load_continuation_verification_records,
    validate_continuation_binding,
)
from .thinking_budget_scaling import (
    SelectedTask,
    ThinkingBudgetScalingConfig,
    _repository_relative_path,
    _sequence_positions,
    lean_wrapper_normalization_v1,
    select_scaling_tasks,
)
from .verifier import LeanVerifier

NATURAL_MAX_CONFIG_SCHEMA = "qwen35-thinking-budget-natural-max-config-v1"
CAPACITY_ATTEMPT_SCHEMA = "qwen35-thinking-budget-natural-max-capacity-attempt-v1"
CAPACITY_EVIDENCE_SCHEMA = "qwen35-thinking-budget-natural-max-capacity-v1"
GENERATION_SCHEMA = "qwen35-thinking-budget-natural-max-generation-v1"
VERIFICATION_SCHEMA = "qwen35-thinking-budget-natural-max-verification-v1"
RESULTS_SCHEMA = "qwen35-thinking-budget-natural-max-results-v1"
ARM = "BNAT-MAX"


@dataclass(frozen=True)
class NaturalMaxConfig:
    path: Path
    value: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> NaturalMaxConfig:
        config = cls(
            path=path.resolve(), value=json.loads(path.read_text(encoding="utf-8"))
        )
        validate_natural_max_config(config)
        return config

    @property
    def historical(self) -> dict[str, Any]:
        return self.value["historical_stage2"]

    @property
    def capacity(self) -> dict[str, Any]:
        return self.value["capacity_search"]

    @property
    def arm(self) -> dict[str, Any]:
        return self.value["arm"]

    @property
    def canonical_output(self) -> dict[str, Any]:
        return self.value["canonical_output"]


def validate_natural_max_config(config: NaturalMaxConfig) -> None:
    if config.value.get("schema_version") != NATURAL_MAX_CONFIG_SCHEMA:
        raise ValueError("unknown natural-max config schema")
    if config.value.get("experiment_id") != (
        "qwen35-4b-thinking-budget-natural-max-v1"
    ):
        raise ValueError("natural-max experiment id changed")
    if config.historical != {
        "continuation_config_path": ("config/qwen35-thinking-budget-continuation.json"),
        "continuation_config_sha256": (
            "9ce974ef55c57736cd42a47251af76e9bfd8f6d6780514057a97718ce8c3135e"
        ),
        "continuation_results_path": (
            "evidence/qwen35-thinking-budget-scaling/continuation-results.json"
        ),
        "continuation_results_sha256": (
            "c741a61829c8a5378501a239d13b0e62c7b8722b2d8f72dcd24d3de814f4aac6"
        ),
        "generations_sha256": (
            "fed3c3f866a67d57dae86b307c178091ad5db926078c3030a6bb56ee5ccc1112"
        ),
        "verifications_sha256": (
            "c170e62a6679f1e983447b8d58fa9182754b3ea0ce3d3917c866c8cb0c57c387"
        ),
        "preserve_unchanged": True,
    }:
        raise ValueError("historical Stage 2 binding changed")
    if config.capacity != {
        "model_native_context_ceiling": 262144,
        "known_stable_lower_bound": 24576,
        "lattice_quantum": 4096,
        "algorithm": (
            "try-native-then-validate-lower-and-bisect-largest-passing-lattice-point"
        ),
        "gpu_memory_utilization": 0.9,
        "gpu_memory_utilization_tuned": False,
        "max_num_seqs": 1,
        "max_in_flight_requests": 1,
        "smoke_decode_tokens": 8,
        "smoke_prompt_method": (
            "newline-token-prefix-plus-first-frozen-rendered-prompt"
        ),
        "smoke_total_tokens": "candidate-max-model-len",
        "smoke_sampling": "greedy-seed-0",
    }:
        raise ValueError("natural-max capacity-search contract changed")
    if config.arm != {
        "name": ARM,
        "enable_thinking": True,
        "thinking_token_budget": None,
        "candidate_index": 0,
        "seed": 0,
        "max_tokens": (
            "machine-supported-context-minus-exact-rendered-prompt-token-count"
        ),
        "reserved_final_allowance": 0,
    }:
        raise ValueError("natural-max arm contract changed")
    if config.canonical_output != {
        "parser": "qwen3",
        "parser_implementation_sha256": (
            "8a7ee658322de7b736ea5b0f802d70dd07a124b5878b4f8ad2f99eca8e1d35fb"
        ),
        "parser_adapter_sha256": (
            "2f9e31e13734b7df75d424f45cf39d723bef7c5415a9bc6851b121cbb6b4ae6d"
        ),
        "canonical_answer": "parsed_final_exact",
        "normalization": LEAN_WRAPPER_NORMALIZATION,
    }:
        raise ValueError("natural-max canonical-output contract changed")


def validate_natural_max_binding(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    historical_artifact_dir: Path,
) -> ThinkingBudgetContinuationConfig:
    repository_root = Path(__file__).resolve().parents[2]
    continuation_path = repository_root / config.historical["continuation_config_path"]
    results_path = repository_root / config.historical["continuation_results_path"]
    if (
        _file_sha256(continuation_path)
        != config.historical["continuation_config_sha256"]
    ):
        raise ValueError("immutable continuation config changed")
    if _file_sha256(results_path) != config.historical["continuation_results_sha256"]:
        raise ValueError("immutable continuation results changed")
    if (
        _file_sha256(historical_artifact_dir / "generations.jsonl")
        != (config.historical["generations_sha256"])
    ):
        raise ValueError("immutable B4/B8/B16 generations changed")
    if (
        _file_sha256(historical_artifact_dir / "verifications.jsonl")
        != (config.historical["verifications_sha256"])
    ):
        raise ValueError("immutable B4/B8/B16 verifications changed")
    continuation = ThinkingBudgetContinuationConfig.load(continuation_path)
    validate_continuation_binding(continuation, scaling, stage1)
    if (
        stage1.engine["dtype"] != "bfloat16"
        or stage1.engine["quantization"] is not None
    ):
        raise ValueError("natural-max must use unquantized BF16")
    if stage1.engine["reasoning_parser"] != "qwen3":
        raise ValueError("natural-max must use the qwen3 parser")
    identity = canonical_parser_runtime_identity(stage1)
    if (
        identity["implementation_source"]["sha256"]
        != (config.canonical_output["parser_implementation_sha256"])
    ):
        raise ValueError("natural-max parser implementation changed")
    if (
        identity["registered_adapter_source"]["sha256"]
        != (config.canonical_output["parser_adapter_sha256"])
    ):
        raise ValueError("natural-max parser adapter changed")
    return continuation


def next_capacity_candidate(
    config: NaturalMaxConfig, attempts: Sequence[Mapping[str, Any]]
) -> int | None:
    native = int(config.capacity["model_native_context_ceiling"])
    lower = int(config.capacity["known_stable_lower_bound"])
    quantum = int(config.capacity["lattice_quantum"])
    runner_source_sha256 = _file_sha256(Path(__file__))
    if any(
        row["status"] == "infrastructure_failed"
        and row.get("runner_source_sha256") == runner_source_sha256
        for row in attempts
    ):
        raise RuntimeError(
            "capacity runner infrastructure failed; correct it before retrying"
        )
    outcomes = {
        int(row["requested_max_model_len"]): row["status"]
        for row in attempts
        if row["status"] != "infrastructure_failed"
    }
    if native not in outcomes:
        return native
    if outcomes[native] == "passed":
        return None
    if lower not in outcomes:
        return lower
    if outcomes[lower] != "passed":
        raise RuntimeError("natural-max known-stable lower bound failed")
    passing = [value for value, status in outcomes.items() if status == "passed"]
    failing = [
        value for value, status in outcomes.items() if status == "capacity_failed"
    ]
    stable = max(passing)
    unstable = min(value for value in failing if value > stable)
    if unstable - stable <= quantum:
        return None
    midpoint_units = ((stable // quantum) + (unstable // quantum)) // 2
    candidate = midpoint_units * quantum
    if candidate <= stable:
        candidate = stable + quantum
    if candidate >= unstable:
        candidate = unstable - quantum
    if candidate in outcomes:
        raise RuntimeError("capacity search selected a duplicate lattice point")
    return candidate


def run_capacity_attempt(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    historical_artifact_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    validate_natural_max_binding(config, scaling, stage1, historical_artifact_dir)
    if output_path.exists():
        evidence = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_capacity_evidence(config, evidence, artifact_dir)
        return evidence
    tasks, mathia_binding = load_mathia_tasks(stage1, mathia_root)
    snapshot = _resolve_model_snapshot(stage1)
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    native_value = int(model_config["text_config"]["max_position_embeddings"])
    if native_value != int(config.capacity["model_native_context_ceiling"]):
        raise ValueError("model-native context ceiling changed")
    tokenizer = _load_tokenizer(stage1, snapshot_path=snapshot)
    selected, selection = select_scaling_tasks(scaling, stage1, tasks, tokenizer)
    probe = selected[0]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = artifact_dir / "capacity-attempts.jsonl"
    attempts = load_capacity_attempts(config, attempt_path)
    candidate = next_capacity_candidate(config, attempts)
    if candidate is not None:
        record = _run_one_capacity_probe(
            config, scaling, stage1, tokenizer, probe, snapshot, candidate
        )
        _append_jsonl(attempt_path, record)
        attempts = load_capacity_attempts(config, attempt_path)
    next_candidate = next_capacity_candidate(config, attempts)
    if next_candidate is not None:
        return {
            "schema_version": CAPACITY_EVIDENCE_SCHEMA,
            "status": "search_pending",
            "attempt_count": len(attempts),
            "next_max_model_len": next_candidate,
        }
    passing = [row for row in attempts if row["status"] == "passed"]
    if not passing:
        raise RuntimeError("capacity search completed without a passing context")
    selected_attempt = max(passing, key=lambda row: int(row["requested_max_model_len"]))
    evidence = {
        "schema_version": CAPACITY_EVIDENCE_SCHEMA,
        "status": "passed",
        "experiment_id": config.value["experiment_id"],
        "natural_max_config_sha256": _file_sha256(config.path),
        "historical_stage2": config.historical,
        "model": stage1.model,
        "runtime": {
            "engine": stage1.engine["name"],
            "version": stage1.engine["version"],
            "dtype": stage1.engine["dtype"],
            "quantization": stage1.engine["quantization"],
            "parser_runtime_identity": canonical_parser_runtime_identity(stage1),
        },
        "mathia_binding": {
            key: value for key, value in mathia_binding.items() if key != "corpus_root"
        },
        "selection_binding": {
            "ordered_selection_sha256": selection["ordered_selection_sha256"],
            "selected_task_count": len(selected),
            "probe_task_id": probe.task.task_id,
            "probe_rendered_prompt_sha256": probe.rendered_prompt_sha256,
        },
        "search_contract": config.capacity,
        "model_native_context_ceiling": native_value,
        "machine_supported_context": int(selected_attempt["requested_max_model_len"]),
        "selected_gpu_memory_utilization": float(
            selected_attempt["gpu_memory_utilization"]
        ),
        "selected_attempt_id": selected_attempt["attempt_id"],
        "attempts": [_compact_capacity_attempt(row) for row in attempts],
        "attempt_jsonl": {
            "path": _repository_relative_path(attempt_path),
            "sha256": _file_sha256(attempt_path),
            "record_count": len(attempts),
        },
        "checks": {
            "native_model_ceiling_exact": native_value == 262144,
            "single_sequence_execution": all(
                int(row["max_num_seqs"]) == 1 for row in attempts
            ),
            "selected_engine_length_not_lowered": int(
                selected_attempt["actual_max_model_len"]
            )
            == int(selected_attempt["requested_max_model_len"]),
            "selected_cache_capacity_sufficient": int(
                selected_attempt["kv_cache_size_tokens"]
            )
            >= int(selected_attempt["requested_max_model_len"]),
            "selected_near_capacity_smoke_completed": bool(
                selected_attempt["smoke_completed"]
            ),
            "selected_parser_replay_deterministic": bool(
                selected_attempt["parser_replay_deterministic"]
            ),
            "historical_stage2_preserved": True,
        },
        "scientific_generation_authorized": True,
        "scientific_generation_started": False,
        "scientific_generation_candidate_count": 0,
    }
    if not all(evidence["checks"].values()):
        raise RuntimeError("selected natural-max capacity failed an acceptance check")
    _atomic_write_json(output_path, evidence)
    return evidence


def _run_one_capacity_probe(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    tokenizer: Any,
    selected: SelectedTask,
    snapshot: Path,
    requested_max_model_len: int,
) -> dict[str, Any]:
    attempt_identity = {
        "natural_max_config_sha256": _file_sha256(config.path),
        "runner_source_sha256": _file_sha256(Path(__file__)),
        "requested_max_model_len": requested_max_model_len,
        "gpu_memory_utilization": float(config.capacity["gpu_memory_utilization"]),
        "max_num_seqs": int(config.capacity["max_num_seqs"]),
        "smoke_decode_tokens": int(config.capacity["smoke_decode_tokens"]),
        "probe_task_id": selected.task.task_id,
        "probe_rendered_prompt_sha256": selected.rendered_prompt_sha256,
    }
    attempt_id = "natural-max-capacity-" + _sha256_json(attempt_identity)[:32]
    _configure_runtime()
    runtime = _local_runtime(stage1)
    monitor = _GpuMemoryMonitor(int(runtime["cuda_device_index"]), required=True)
    started = time.perf_counter()
    monitor.start()
    status = "infrastructure_failed"
    details: dict[str, Any] = {}
    error_text: str | None = None
    try:
        details = asyncio.run(
            _capacity_probe_async(
                config,
                scaling,
                stage1,
                tokenizer,
                selected,
                snapshot,
                requested_max_model_len,
            )
        )
        status = "passed"
    except Exception as error:  # noqa: BLE001 - capacity failure is evidence
        error_text = f"{type(error).__name__}: {error}"
    finally:
        runtime.update(monitor.stop())
    error_category = _capacity_error_category(error_text)
    if error_category in {
        "cuda_oom",
        "kv_cache_capacity",
        "runtime_lowered_context",
        "runtime_stability",
    }:
        status = "capacity_failed"
    return {
        "schema_version": CAPACITY_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        **attempt_identity,
        "status": status,
        "error_category": error_category,
        "error": error_text,
        "max_in_flight_requests": int(config.capacity["max_in_flight_requests"]),
        "wall_time_seconds": time.perf_counter() - started,
        "package_versions": _package_versions(),
        "model_snapshot_revision": snapshot.name,
        **runtime,
        **details,
    }


async def _capacity_probe_async(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    tokenizer: Any,
    selected: SelectedTask,
    snapshot: Path,
    requested_max_model_len: int,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = _natural_engine_args(
        config, stage1, snapshot, requested_max_model_len
    )
    engine: AsyncLLM | None = None
    try:
        engine = AsyncLLM.from_engine_args(engine_args)
        actual_max_model_len = int(engine.model_config.max_model_len)
        cache = engine.vllm_config.cache_config
        cache_tokens = int(cache.kv_cache_size_tokens or 0)
        if actual_max_model_len != requested_max_model_len:
            raise RuntimeError(
                "vLLM silently changed max_model_len: "
                f"{requested_max_model_len} -> {actual_max_model_len}"
            )
        if cache_tokens < requested_max_model_len:
            raise RuntimeError(
                f"KV cache capacity {cache_tokens} < {requested_max_model_len}"
            )
        smoke_decode = int(config.capacity["smoke_decode_tokens"])
        smoke_prompt_count = requested_max_model_len - smoke_decode
        rendered_ids = tokenizer.encode(
            selected.rendered_prompt, add_special_tokens=False
        )
        if len(rendered_ids) != selected.rendered_prompt_token_count:
            raise RuntimeError("capacity smoke prompt token count changed")
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids != [198]:
            raise RuntimeError("capacity smoke newline filler token changed")
        filler_count = smoke_prompt_count - len(rendered_ids)
        if filler_count < 0:
            raise RuntimeError("capacity smoke context is shorter than frozen prompt")
        prompt_ids = [198] * filler_count + [int(value) for value in rendered_ids]
        params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=smoke_decode,
            seed=0,
            skip_special_tokens=False,
        )
        last_output: Any | None = None
        async for output in engine.generate(
            {"prompt_token_ids": prompt_ids},
            params,
            request_id=(f"natural-max-capacity-smoke-{requested_max_model_len}"),
            reasoning_parser_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
        ):
            last_output = output
        if last_output is None or not last_output.finished:
            raise RuntimeError("near-capacity prefill/decode smoke did not finish")
        completion = last_output.outputs[0]
        raw_text = str(completion.text)
        raw_ids = [int(value) for value in completion.token_ids]
        parser_class = ReasoningParserManager.get_reasoning_parser("qwen3")
        parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
        parser_request = ChatCompletionRequest(
            model=MODEL_ID,
            messages=[{"role": "user", "content": selected.user_message}],
            max_tokens=smoke_decode,
            temperature=0.0,
            include_reasoning=True,
        )
        reasoning, final = parser.extract_reasoning(raw_text, parser_request)
        replay_reasoning, replay_final = parser.extract_reasoning(
            raw_text, parser_request
        )
        if (reasoning, final) != (replay_reasoning, replay_final):
            raise RuntimeError("capacity smoke qwen3 replay is nondeterministic")
        return {
            "actual_max_model_len": actual_max_model_len,
            "kv_cache_size_tokens": cache_tokens,
            "kv_cache_max_concurrency": float(cache.kv_cache_max_concurrency or 0.0),
            "num_gpu_blocks": int(cache.num_gpu_blocks or 0),
            "block_size": int(cache.block_size or 0),
            "smoke_prompt_token_count": len(prompt_ids),
            "smoke_decode_token_budget": smoke_decode,
            "smoke_output_token_count": len(raw_ids),
            "smoke_total_token_count": len(prompt_ids) + len(raw_ids),
            "smoke_completed": True,
            "smoke_finish_reason": _finish_reason(
                None
                if completion.finish_reason is None
                else str(completion.finish_reason)
            ),
            "smoke_raw_response_sha256": _sha256_text(raw_text),
            "smoke_raw_response_text": raw_text,
            "smoke_raw_response_token_ids_sha256": _sha256_json(raw_ids),
            "smoke_raw_response_token_ids": raw_ids,
            "smoke_prompt_token_ids_sha256": _sha256_json(prompt_ids),
            "smoke_filler_token_id": 198,
            "smoke_filler_token_count": filler_count,
            "smoke_reasoning_content": reasoning,
            "smoke_reasoning_sha256": (
                None if reasoning is None else _sha256_text(reasoning)
            ),
            "smoke_final_content": final,
            "smoke_final_sha256": None if final is None else _sha256_text(final),
            "parser_replay_deterministic": True,
        }
    finally:
        if engine is not None:
            engine.shutdown()
        gc.collect()


def _natural_engine_args(
    config: NaturalMaxConfig,
    stage1: NativeThinkingConfig,
    snapshot: Path,
    max_model_len: int,
) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs

    return AsyncEngineArgs(
        model=str(snapshot),
        tokenizer=str(snapshot),
        revision=str(stage1.model["model_revision"]),
        tokenizer_revision=str(stage1.model["tokenizer_revision"]),
        dtype=str(stage1.engine["dtype"]),
        tensor_parallel_size=int(stage1.engine["tensor_parallel_size"]),
        gpu_memory_utilization=float(config.capacity["gpu_memory_utilization"]),
        max_model_len=max_model_len,
        max_num_seqs=int(config.capacity["max_num_seqs"]),
        enforce_eager=bool(stage1.engine["enforce_eager"]),
        quantization=stage1.engine["quantization"],
        language_model_only=bool(stage1.engine["language_model_only"]),
        reasoning_parser=str(stage1.engine["reasoning_parser"]),
        generation_config="vllm",
        enable_log_requests=False,
        disable_log_stats=False,
    )


def _capacity_error_category(error: str | None) -> str | None:
    if error is None:
        return None
    lowered = error.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "cuda_oom"
    if "silently changed max_model_len" in lowered:
        return "runtime_lowered_context"
    if "kv cache" in lowered or "max seq len" in lowered:
        return "kv_cache_capacity"
    if lowered.startswith(("typeerror:", "attributeerror:", "keyerror:")):
        return "implementation_error"
    return "runtime_stability"


def load_capacity_attempts(
    config: NaturalMaxConfig, path: Path
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    seen_ids: set[str] = set()
    seen_attempts: set[tuple[int, str]] = set()
    for record in records:
        if record.get("schema_version") != CAPACITY_ATTEMPT_SCHEMA:
            raise ValueError("unknown natural-max capacity-attempt schema")
        identity = {
            key: record[key]
            for key in (
                "natural_max_config_sha256",
                "runner_source_sha256",
                "requested_max_model_len",
                "gpu_memory_utilization",
                "max_num_seqs",
                "smoke_decode_tokens",
                "probe_task_id",
                "probe_rendered_prompt_sha256",
            )
        }
        attempt_id = "natural-max-capacity-" + _sha256_json(identity)[:32]
        if record["attempt_id"] != attempt_id:
            raise ValueError("natural-max capacity-attempt identity mismatch")
        length = int(record["requested_max_model_len"])
        attempt_key = (length, str(record["runner_source_sha256"]))
        if attempt_id in seen_ids or attempt_key in seen_attempts:
            raise ValueError("duplicate natural-max capacity attempt")
        seen_ids.add(attempt_id)
        seen_attempts.add(attempt_key)
        if record["natural_max_config_sha256"] != _file_sha256(config.path):
            raise ValueError("capacity attempt used another natural-max config")
        if record["status"] not in {
            "passed",
            "capacity_failed",
            "infrastructure_failed",
        }:
            raise ValueError("invalid natural-max capacity-attempt status")
        if record["status"] == "passed" and not (
            int(record["actual_max_model_len"]) == length
            and int(record["kv_cache_size_tokens"]) >= length
            and bool(record["smoke_completed"])
            and bool(record["parser_replay_deterministic"])
        ):
            raise ValueError("passing capacity attempt lacks acceptance evidence")
        if record["status"] == "passed":
            raw_text = str(record["smoke_raw_response_text"])
            raw_ids = [int(value) for value in record["smoke_raw_response_token_ids"]]
            if _sha256_text(raw_text) != record["smoke_raw_response_sha256"]:
                raise ValueError("capacity smoke raw-text hash mismatch")
            if _sha256_json(raw_ids) != record["smoke_raw_response_token_ids_sha256"]:
                raise ValueError("capacity smoke raw-token hash mismatch")
            if len(raw_ids) != int(record["smoke_output_token_count"]):
                raise ValueError("capacity smoke raw-token count mismatch")
            for name in ("reasoning", "final"):
                value = record[f"smoke_{name}_content"]
                expected_hash = None if value is None else _sha256_text(str(value))
                if record[f"smoke_{name}_sha256"] != expected_hash:
                    raise ValueError(f"capacity smoke {name} hash mismatch")
    return records


def _compact_capacity_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "attempt_id",
        "requested_max_model_len",
        "gpu_memory_utilization",
        "status",
        "error_category",
        "error",
        "actual_max_model_len",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
        "smoke_prompt_token_count",
        "smoke_decode_token_budget",
        "smoke_output_token_count",
        "smoke_total_token_count",
        "smoke_completed",
        "parser_replay_deterministic",
        "smoke_finish_reason",
        "gpu_memory_peak_bytes",
        "wall_time_seconds",
    )
    return {key: row.get(key) for key in keys}


def _validate_capacity_evidence(
    config: NaturalMaxConfig, evidence: Mapping[str, Any], artifact_dir: Path
) -> None:
    if evidence.get("schema_version") != CAPACITY_EVIDENCE_SCHEMA:
        raise ValueError("unknown natural-max capacity evidence schema")
    if evidence.get("status") != "passed" or not evidence.get(
        "scientific_generation_authorized"
    ):
        raise ValueError("natural-max capacity evidence has not passed")
    if evidence.get("natural_max_config_sha256") != _file_sha256(config.path):
        raise ValueError("capacity evidence used another natural-max config")
    if not all(bool(value) for value in evidence.get("checks", {}).values()):
        raise ValueError("capacity evidence contains a failed acceptance check")
    attempt_path = artifact_dir / "capacity-attempts.jsonl"
    attempts = load_capacity_attempts(config, attempt_path)
    binding = evidence["attempt_jsonl"]
    if _file_sha256(attempt_path) != binding["sha256"] or len(attempts) != int(
        binding["record_count"]
    ):
        raise ValueError("capacity attempt JSONL changed after selection")
    selected_id = str(evidence["selected_attempt_id"])
    selected = next((row for row in attempts if row["attempt_id"] == selected_id), None)
    if selected is None or selected["status"] != "passed":
        raise ValueError("selected natural-max capacity attempt is missing")
    if int(selected["requested_max_model_len"]) != int(
        evidence["machine_supported_context"]
    ):
        raise ValueError("selected machine context binding changed")


def natural_generation_config_sha256(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    capacity_path: Path,
) -> str:
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    return _sha256_json(
        {
            "natural_max_config_sha256": _file_sha256(config.path),
            "capacity_evidence_sha256": _file_sha256(capacity_path),
            "machine_supported_context": capacity["machine_supported_context"],
            "gpu_memory_utilization": capacity["selected_gpu_memory_utilization"],
            "model": stage1.model,
            "engine": {
                key: stage1.engine[key]
                for key in (
                    "name",
                    "version",
                    "reasoning_parser",
                    "dtype",
                    "tensor_parallel_size",
                    "enforce_eager",
                    "quantization",
                    "language_model_only",
                )
            },
            "sampling": scaling.sampling,
            "arm": config.arm,
            "canonical_output": config.canonical_output,
        }
    )


def natural_candidate_identity(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    selected: SelectedTask,
    capacity_path: Path,
) -> tuple[str, dict[str, Any]]:
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    machine_context = int(capacity["machine_supported_context"])
    max_tokens = machine_context - selected.rendered_prompt_token_count
    if max_tokens <= 0:
        raise ValueError("natural-max prompt leaves no generation budget")
    payload = {
        "arm": ARM,
        "enable_thinking": True,
        "thinking_token_budget": None,
        "workload": selected.task.workload,
        "task_id": selected.task.task_id,
        "prompt_sha256": selected.user_message_sha256,
        "rendered_prompt_sha256": selected.rendered_prompt_sha256,
        "rendered_prompt_token_count": selected.rendered_prompt_token_count,
        "candidate_index": 0,
        "seed": 0,
        "model_revision": MODEL_REVISION,
        "max_model_len": machine_context,
        "max_tokens": max_tokens,
        "generation_config_sha256": natural_generation_config_sha256(
            config, scaling, stage1, capacity_path
        ),
    }
    return "thinking-budget-natural-max-" + _sha256_json(payload)[:32], payload


def run_natural_generation(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    historical_artifact_dir: Path,
    capacity_path: Path,
) -> dict[str, Any]:
    validate_natural_max_binding(config, scaling, stage1, historical_artifact_dir)
    _validate_capacity_evidence(config, _read_json(capacity_path), artifact_dir)
    tasks, _ = load_mathia_tasks(stage1, mathia_root)
    snapshot = _resolve_model_snapshot(stage1)
    tokenizer = _load_tokenizer(stage1, snapshot_path=snapshot)
    selected, selection = select_scaling_tasks(scaling, stage1, tasks, tokenizer)
    generation_path = artifact_dir / "generations.jsonl"
    records = load_natural_generation_records(
        config, scaling, stage1, selected, capacity_path, generation_path, tokenizer
    )
    completed = {str(row["candidate_id"]) for row in records}
    requests = [
        (
            selected_task,
            *natural_candidate_identity(
                config, scaling, stage1, selected_task, capacity_path
            ),
        )
        for selected_task in selected
    ]
    pending = [row for row in requests if row[1] not in completed]
    if not pending:
        return {
            "status": "already_complete",
            "candidate_count": len(records),
            "new_candidates": 0,
            "selection": selection,
        }
    runtime = _execute_natural_requests(
        config,
        scaling,
        stage1,
        tokenizer,
        pending,
        snapshot,
        generation_path,
        artifact_dir / "generation-segments.jsonl",
    )
    return {
        "status": "completed",
        "candidate_count": len(requests),
        "new_candidates": int(runtime["persisted_candidate_count"]),
        "selection": selection,
        "runtime": runtime,
    }


def _execute_natural_requests(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    tokenizer: Any,
    requests: Sequence[tuple[SelectedTask, str, dict[str, Any]]],
    snapshot: Path,
    generation_path: Path,
    segment_path: Path,
) -> dict[str, Any]:
    _configure_runtime()
    runtime = _local_runtime(stage1)
    monitor = _GpuMemoryMonitor(int(runtime["cuda_device_index"]), required=True)
    started = time.perf_counter()
    monitor.start()
    status = "failed"
    starting_count = _jsonl_record_count(generation_path)
    persisted_count = 0
    error_text: str | None = None
    try:
        persisted_count = asyncio.run(
            _run_natural_async(
                config,
                scaling,
                stage1,
                tokenizer,
                requests,
                snapshot,
                generation_path,
            )
        )
        status = "completed"
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        persisted_count = _jsonl_record_count(generation_path) - starting_count
        runtime.update(monitor.stop())
        runtime.update(
            {
                "schema_version": "qwen35-thinking-budget-natural-max-segment-v1",
                "status": status,
                "arm": ARM,
                "requested_candidate_count": len(requests),
                "persisted_candidate_count": persisted_count,
                "segment_wall_time_seconds": time.perf_counter() - started,
                "error": error_text,
                "package_versions": _package_versions(),
                "model_snapshot_revision": snapshot.name,
                "max_model_len": int(requests[0][2]["max_model_len"]),
                "max_num_seqs": 1,
                "max_in_flight_requests": 1,
                "gpu_memory_utilization": float(
                    config.capacity["gpu_memory_utilization"]
                ),
            }
        )
        _append_jsonl(segment_path, runtime)
    return runtime


async def _run_natural_async(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    tokenizer: Any,
    requests: Sequence[tuple[SelectedTask, str, dict[str, Any]]],
    snapshot: Path,
    generation_path: Path,
) -> int:
    from vllm import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    max_model_len = int(requests[0][2]["max_model_len"])
    engine = AsyncLLM.from_engine_args(
        _natural_engine_args(config, stage1, snapshot, max_model_len)
    )
    persisted = 0
    try:
        if int(engine.model_config.max_model_len) != max_model_len:
            raise RuntimeError("natural-max engine changed selected max_model_len")
        if int(engine.vllm_config.cache_config.kv_cache_size_tokens or 0) < (
            max_model_len
        ):
            raise RuntimeError("natural-max engine cache is below selected context")
        for selected, candidate_id, identity in requests:
            started = time.perf_counter()
            params = SamplingParams(
                n=1,
                temperature=float(scaling.sampling["temperature"]),
                top_p=float(scaling.sampling["top_p"]),
                top_k=int(scaling.sampling["top_k"]),
                min_p=float(scaling.sampling["min_p"]),
                presence_penalty=float(scaling.sampling["presence_penalty"]),
                repetition_penalty=float(scaling.sampling["repetition_penalty"]),
                max_tokens=int(identity["max_tokens"]),
                seed=int(identity["seed"]),
                skip_special_tokens=False,
            )
            last_output: Any | None = None
            async for output in engine.generate(
                selected.rendered_prompt,
                params,
                request_id=candidate_id,
                prompt_text=selected.rendered_prompt,
                reasoning_parser_kwargs={
                    "chat_template_kwargs": {"enable_thinking": True}
                },
            ):
                last_output = output
            if last_output is None or not last_output.finished:
                raise RuntimeError(
                    f"natural-max request did not finish: {candidate_id}"
                )
            record = _natural_generation_record(
                scaling,
                stage1,
                tokenizer,
                selected,
                candidate_id,
                identity,
                last_output.outputs[0],
                latency_seconds=time.perf_counter() - started,
            )
            _append_jsonl(generation_path, record)
            persisted += 1
            print(
                json.dumps(
                    {
                        "phase": "thinking_budget_natural_max_generation",
                        "completed_candidates": persisted,
                        "pending_candidate_count": len(requests),
                        "candidate_id": candidate_id,
                        "reasoning_tokens": record["reasoning_token_count"],
                        "parsed_final_nonempty": bool(record["parsed_final_exact"]),
                        "finish_reason": record["finish_reason"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        engine.shutdown()
        gc.collect()
    return persisted


def _natural_generation_record(
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    tokenizer: Any,
    selected: SelectedTask,
    candidate_id: str,
    identity: Mapping[str, Any],
    completion: Any,
    *,
    latency_seconds: float,
) -> dict[str, Any]:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager

    token_ids = [int(value) for value in completion.token_ids]
    raw_text = str(completion.text)
    parser_class = ReasoningParserManager.get_reasoning_parser("qwen3")
    parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": selected.user_message}],
        max_tokens=int(identity["max_tokens"]),
        temperature=float(scaling.sampling["temperature"]),
        top_p=float(scaling.sampling["top_p"]),
        include_reasoning=True,
    )
    reasoning, parsed_final = parser.extract_reasoning(raw_text, request)
    normalized_final, normalization_applied = lean_wrapper_normalization_v1(
        parsed_final
    )
    normalized_second, _ = lean_wrapper_normalization_v1(normalized_final)
    end_ids = tokenizer.encode(str(parser.reasoning_end_str), add_special_tokens=False)
    end_positions = _sequence_positions(token_ids, end_ids)
    first_end = None if not end_positions else end_positions[0] + len(end_ids)
    raw_finish = (
        None if completion.finish_reason is None else str(completion.finish_reason)
    )
    finish_reason = _finish_reason(raw_finish)
    context_exhausted = (
        not parsed_final
        and finish_reason == "token_limit"
        and len(token_ids) == int(identity["max_tokens"])
    )
    if parsed_final:
        if first_end is None:
            raise RuntimeError(
                f"qwen3 returned a final without an auditable reasoning exit: "
                f"{candidate_id}"
            )
        reasoning_exit = "natural_to_final"
    elif context_exhausted:
        reasoning_exit = "context_exhausted_no_final"
    else:
        reasoning_exit = "ended_without_final"
    reasoning_count = int(parser.count_reasoning_tokens(token_ids))
    parsed_count = (
        0
        if parsed_final is None
        else len(tokenizer.encode(parsed_final, add_special_tokens=False))
    )
    normalized_count = (
        0
        if normalized_final is None
        else len(tokenizer.encode(normalized_final, add_special_tokens=False))
    )
    if len(token_ids) > int(identity["max_tokens"]):
        raise RuntimeError(f"natural-max request exceeded context: {candidate_id}")
    if candidate_id != "thinking-budget-natural-max-" + _sha256_json(identity)[:32]:
        raise AssertionError("natural-max candidate identity changed")
    return {
        "schema_version": GENERATION_SCHEMA,
        "candidate_id": candidate_id,
        **dict(identity),
        "intuition_sha256": selected.task.intuition_sha256,
        "theorem_sha256": selected.task.theorem_sha256,
        "raw_response_text": raw_text,
        "raw_response_sha256": _sha256_text(raw_text),
        "raw_response_token_ids": token_ids,
        "raw_response_token_ids_sha256": _sha256_json(token_ids),
        "raw_response_token_count": len(token_ids),
        "reasoning_content": reasoning,
        "reasoning_content_sha256": (
            None if reasoning is None else _sha256_text(reasoning)
        ),
        "reasoning_token_count": reasoning_count,
        "reasoning_exit": reasoning_exit,
        "reasoning_end_marker_token_positions": end_positions,
        "reasoning_end_position_token_count": first_end,
        "parsed_final_exact": parsed_final,
        "parsed_final_sha256": (
            None if parsed_final is None else _sha256_text(parsed_final)
        ),
        "parsed_final_token_count": parsed_count,
        "normalized_final_exact": normalized_final,
        "normalized_final_sha256": (
            None if normalized_final is None else _sha256_text(normalized_final)
        ),
        "normalized_final_token_count": normalized_count,
        "normalization_id": LEAN_WRAPPER_NORMALIZATION,
        "normalization_applied": normalization_applied,
        "normalization_pass_count": 1,
        "normalization_idempotent": normalized_second == normalized_final,
        "context_exhausted_no_final": context_exhausted,
        "parser_final_content_is_exact_raw_suffix": (
            parsed_final is None or raw_text.endswith(parsed_final)
        ),
        "finish_reason": finish_reason,
        "raw_finish_reason": raw_finish,
        "generation_latency_seconds": latency_seconds,
        "request_id": candidate_id,
    }


def load_natural_generation_records(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    selected: Sequence[SelectedTask],
    capacity_path: Path,
    path: Path,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    selected_by_task = {row.task.task_id: row for row in selected}
    expected = {
        natural_candidate_identity(config, scaling, stage1, row, capacity_path)[0]: row
        for row in selected
    }
    seen: set[str] = set()
    for record in records:
        if record.get("schema_version") != GENERATION_SCHEMA:
            raise ValueError("unknown natural-max generation schema")
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen or candidate_id not in expected:
            raise ValueError("duplicate or unexpected natural-max candidate")
        seen.add(candidate_id)
        selected_task = selected_by_task[str(record["task_id"])]
        expected_id, identity = natural_candidate_identity(
            config, scaling, stage1, selected_task, capacity_path
        )
        if candidate_id != expected_id or any(
            record.get(key) != value for key, value in identity.items()
        ):
            raise ValueError(f"natural-max identity mismatch: {candidate_id}")
        raw = str(record["raw_response_text"])
        ids = [int(value) for value in record["raw_response_token_ids"]]
        if _sha256_text(raw) != record["raw_response_sha256"]:
            raise ValueError(f"natural-max raw hash mismatch: {candidate_id}")
        if _sha256_json(ids) != record["raw_response_token_ids_sha256"]:
            raise ValueError(f"natural-max token-id hash mismatch: {candidate_id}")
        if len(ids) != int(record["raw_response_token_count"]):
            raise ValueError(f"natural-max raw count mismatch: {candidate_id}")
        for name in ("parsed_final", "normalized_final"):
            value = record[f"{name}_exact"]
            expected_hash = None if value is None else _sha256_text(str(value))
            if record[f"{name}_sha256"] != expected_hash:
                raise ValueError(f"natural-max {name} hash mismatch: {candidate_id}")
        normalized, applied = lean_wrapper_normalization_v1(
            record["parsed_final_exact"]
        )
        if normalized != record["normalized_final_exact"] or applied != bool(
            record["normalization_applied"]
        ):
            raise ValueError(f"natural-max normalization mismatch: {candidate_id}")
        if record["normalization_id"] != LEAN_WRAPPER_NORMALIZATION:
            raise ValueError(f"natural-max normalization id mismatch: {candidate_id}")
        if int(record["normalization_pass_count"]) != 1:
            raise ValueError(f"natural-max normalization pass mismatch: {candidate_id}")
        second_pass, _ = lean_wrapper_normalization_v1(record["normalized_final_exact"])
        if second_pass != record["normalized_final_exact"] or not bool(
            record["normalization_idempotent"]
        ):
            raise ValueError(
                f"natural-max normalization is not idempotent: {candidate_id}"
            )
        if tokenizer is not None:
            _natural_parser_replay(record, selected_task, tokenizer, scaling)
            parsed_final = record["parsed_final_exact"]
            normalized_final = record["normalized_final_exact"]
            parsed_count = (
                0
                if parsed_final is None
                else len(tokenizer.encode(parsed_final, add_special_tokens=False))
            )
            normalized_count = (
                0
                if normalized_final is None
                else len(tokenizer.encode(normalized_final, add_special_tokens=False))
            )
            if parsed_count != int(record["parsed_final_token_count"]):
                raise ValueError(
                    f"natural-max parsed-final count mismatch: {candidate_id}"
                )
            if normalized_count != int(record["normalized_final_token_count"]):
                raise ValueError(
                    f"natural-max normalized-final count mismatch: {candidate_id}"
                )
    return records


def _natural_parser_replay(
    record: Mapping[str, Any],
    selected: SelectedTask,
    tokenizer: Any,
    scaling: ThinkingBudgetScalingConfig,
) -> dict[str, Any]:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning import ReasoningParserManager

    parser_class = ReasoningParserManager.get_reasoning_parser("qwen3")
    parser = parser_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": selected.user_message}],
        max_tokens=int(record["max_tokens"]),
        temperature=float(scaling.sampling["temperature"]),
        top_p=float(scaling.sampling["top_p"]),
        include_reasoning=True,
    )
    reasoning, final = parser.extract_reasoning(
        str(record["raw_response_text"]), request
    )
    normalized, applied = lean_wrapper_normalization_v1(final)
    replay = {
        "parser_replay_exact": (
            reasoning == record["reasoning_content"]
            and final == record["parsed_final_exact"]
        ),
        "normalization_replay_exact": (
            normalized == record["normalized_final_exact"]
            and applied == bool(record["normalization_applied"])
        ),
    }
    if not all(replay.values()):
        raise ValueError(
            f"natural-max parser replay mismatch: {record['candidate_id']}"
        )
    return replay


def run_natural_verification(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    historical_artifact_dir: Path,
    capacity_path: Path,
    *,
    project_roots: Mapping[str, Path],
    workers: int | None = None,
) -> dict[str, Any]:
    validate_natural_max_binding(config, scaling, stage1, historical_artifact_dir)
    _validate_capacity_evidence(config, _read_json(capacity_path), artifact_dir)
    tasks, _ = load_mathia_tasks(stage1, mathia_root)
    tasks_by_id = {row.task_id: row for row in tasks}
    environments = validate_lean_environments(stage1, tasks, project_roots)
    tokenizer = _load_tokenizer(stage1)
    selected, _ = select_scaling_tasks(scaling, stage1, tasks, tokenizer)
    generations = load_natural_generation_records(
        config,
        scaling,
        stage1,
        selected,
        capacity_path,
        artifact_dir / "generations.jsonl",
        tokenizer,
    )
    if len(generations) != 16:
        raise RuntimeError("natural-max generation set is incomplete")
    verification_path = artifact_dir / "verifications.jsonl"
    prior = load_natural_verification_records(verification_path)
    completed = {str(row["candidate_id"]) for row in prior}
    pending = [row for row in generations if row["candidate_id"] not in completed]
    if not pending:
        return {
            "status": "already_complete",
            "candidate_count": len(generations),
            "new_verifications": 0,
            "environments": environments,
        }
    verifiers = {
        workload: LeanVerifier(
            project_roots[workload],
            timeout_seconds=float(stage1.verifier["timeout_seconds"]),
        )
        for workload in WORKLOADS
    }
    worker_count = int(stage1.verifier["workers"] if workers is None else workers)
    started = time.perf_counter()
    new_count = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _verify_natural_record,
                row,
                tasks_by_id[str(row["task_id"])],
                verifiers[str(row["workload"])],
            ): str(row["candidate_id"])
            for row in pending
        }
        for future in as_completed(futures):
            _append_jsonl(verification_path, future.result())
            new_count += 1
    segment = {
        "schema_version": "qwen35-thinking-budget-natural-max-verification-segment-v1",
        "status": "completed",
        "candidate_count": new_count,
        "workers": worker_count,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _append_jsonl(artifact_dir / "verification-segments.jsonl", segment)
    return {
        "status": "completed",
        "candidate_count": len(generations),
        "new_verifications": new_count,
        "environments": environments,
        "runtime": segment,
    }


def _verify_natural_record(
    generation: Mapping[str, Any], task: MathiaTask, verifier: LeanVerifier
) -> dict[str, Any]:
    parsed = generation["parsed_final_exact"]
    normalized = generation["normalized_final_exact"]
    strict = _verify_exact_final(parsed, task, verifier)
    shared = parsed == normalized
    deployed = strict if shared else _verify_exact_final(normalized, task, verifier)
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "candidate_id": generation["candidate_id"],
        "arm": ARM,
        "workload": generation["workload"],
        "task_id": generation["task_id"],
        "candidate_index": generation["candidate_index"],
        "seed": generation["seed"],
        "generation_config_sha256": generation["generation_config_sha256"],
        "parsed_final_sha256": generation["parsed_final_sha256"],
        "normalized_final_sha256": generation["normalized_final_sha256"],
        "strict_parsed_interface": strict,
        "deployed_normalized_interface": deployed,
        "shared_identical_submission": shared,
        "lean_invocation_count": 1 if shared else 2,
        "verification_outcome_changed_by_normalization": (
            strict["category"] != deployed["category"]
        ),
    }


def load_natural_verification_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    seen: set[str] = set()
    for record in records:
        if record.get("schema_version") != VERIFICATION_SCHEMA:
            raise ValueError("unknown natural-max verification schema")
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError("duplicate natural-max verification")
        seen.add(candidate_id)
        if (
            record["strict_parsed_interface"]["submitted_sha256"]
            != record["parsed_final_sha256"]
        ):
            raise ValueError("natural-max strict verification hash mismatch")
        if (
            record["deployed_normalized_interface"]["submitted_sha256"]
            != record["normalized_final_sha256"]
        ):
            raise ValueError("natural-max deployed verification hash mismatch")
        expected_invocations = 1 if record["shared_identical_submission"] else 2
        if int(record["lean_invocation_count"]) != expected_invocations:
            raise ValueError("natural-max Lean invocation count mismatch")
    return records


def write_natural_evidence(
    config: NaturalMaxConfig,
    scaling: ThinkingBudgetScalingConfig,
    stage1: NativeThinkingConfig,
    mathia_root: Path,
    artifact_dir: Path,
    historical_artifact_dir: Path,
    capacity_path: Path,
    evidence_dir: Path,
    *,
    project_roots: Mapping[str, Path],
) -> dict[str, Any]:
    validate_natural_max_binding(config, scaling, stage1, historical_artifact_dir)
    capacity = _read_json(capacity_path)
    _validate_capacity_evidence(config, capacity, artifact_dir)
    tasks, mathia_binding = load_mathia_tasks(stage1, mathia_root)
    tokenizer = _load_tokenizer(stage1)
    selected, selection = select_scaling_tasks(scaling, stage1, tasks, tokenizer)
    selected_by_task = {row.task.task_id: row for row in selected}
    generations = load_natural_generation_records(
        config,
        scaling,
        stage1,
        selected,
        capacity_path,
        artifact_dir / "generations.jsonl",
        tokenizer,
    )
    verifications = load_natural_verification_records(
        artifact_dir / "verifications.jsonl"
    )
    if len(generations) != 16 or len(verifications) != 16:
        raise RuntimeError("natural-max generation/verification set is incomplete")
    generation_by_id = {str(row["candidate_id"]): row for row in generations}
    verification_by_id = {str(row["candidate_id"]): row for row in verifications}
    if set(generation_by_id) != set(verification_by_id):
        raise RuntimeError("natural-max generation/verification identities differ")
    replay = [
        _natural_parser_replay(
            row, selected_by_task[str(row["task_id"])], tokenizer, scaling
        )
        for row in generations
    ]
    if not all(
        row["parser_replay_exact"] and row["normalization_replay_exact"]
        for row in replay
    ):
        raise RuntimeError("natural-max parser replay is incomplete")
    if any(
        marker in str(row["parsed_final_exact"] or "")
        for row in generations
        for marker in ("<think>", "</think>")
    ):
        raise RuntimeError("reasoning marker leaked into natural-max parsed final")
    historical_generations = load_continuation_generation_records(
        historical_artifact_dir / "generations.jsonl"
    )
    historical_verifications = load_continuation_verification_records(
        historical_artifact_dir / "verifications.jsonl"
    )
    b16_generation = {
        str(row["task_id"]): row
        for row in historical_generations
        if row["arm"] == "B16"
    }
    historical_verification_by_id = {
        str(row["candidate_id"]): row for row in historical_verifications
    }
    selected_task_ids = {row.task.task_id for row in selected}
    if set(b16_generation) != selected_task_ids:
        raise RuntimeError(
            "immutable B16 task population differs from frozen selection"
        )
    if any(
        str(row["candidate_id"]) not in historical_verification_by_id
        for row in b16_generation.values()
    ):
        raise RuntimeError("an immutable B16 candidate lacks frozen verification")
    natural_by_task = {str(row["task_id"]): row for row in generations}
    natural_verification_by_task = {
        str(row["task_id"]): verification_by_id[str(row["candidate_id"])]
        for row in generations
    }
    paired = _natural_paired_table(
        selected,
        b16_generation,
        historical_verification_by_id,
        natural_by_task,
        natural_verification_by_task,
    )
    summaries = {
        workload: _natural_summary(
            [row for row in paired if row["workload"] == workload]
        )
        for workload in WORKLOADS
    }
    summaries["combined"] = _natural_summary(paired)
    forced_subset = [row for row in paired if row["b16"]["forced_at_budget"]]
    natural_b16_subset = [row for row in paired if not row["b16"]["forced_at_budget"]]
    generation_segments = _read_jsonl(artifact_dir / "generation-segments.jsonl")
    verification_segments = _read_jsonl(artifact_dir / "verification-segments.jsonl")
    environments = validate_lean_environments(stage1, tasks, project_roots)
    conclusion = _natural_conclusion(capacity, paired, forced_subset)
    result = {
        "schema_version": RESULTS_SCHEMA,
        "status": "complete",
        "experiment_id": config.value["experiment_id"],
        "historical_stage2": {
            **config.historical,
            "preserved_unchanged": True,
            "historical_candidates_regenerated": 0,
        },
        "config_bindings": {
            "natural_max_config_sha256": _file_sha256(config.path),
            "capacity_evidence_sha256": _file_sha256(capacity_path),
            "scaling_config_sha256": _file_sha256(scaling.path),
            "stage1_config_sha256": _file_sha256(stage1.path),
        },
        "model": stage1.model,
        "engine": {
            **stage1.engine,
            "max_model_len": capacity["machine_supported_context"],
            "max_num_seqs": 1,
            "max_in_flight_requests": 1,
            "gpu_memory_utilization": capacity["selected_gpu_memory_utilization"],
        },
        "sampling": scaling.sampling,
        "arm": config.arm,
        "capacity": {
            "model_native_context_ceiling": capacity["model_native_context_ceiling"],
            "machine_supported_context": capacity["machine_supported_context"],
            "selected_attempt_id": capacity["selected_attempt_id"],
            "attempt_count": capacity["attempt_jsonl"]["record_count"],
            "evidence_sha256": _file_sha256(capacity_path),
        },
        "canonical_output": {
            **config.canonical_output,
            "parser_runtime_identity": canonical_parser_runtime_identity(stage1),
            "parser_replay_candidate_count": len(replay),
            "parser_replay_all_exact": True,
        },
        "mathia_binding": {
            key: value for key, value in mathia_binding.items() if key != "corpus_root"
        },
        "selection": selection,
        "lean_environments": {
            workload: {
                key: value
                for key, value in environment.items()
                if key not in {"project_root", "project_head"}
            }
            for workload, environment in environments.items()
        },
        "candidate_count": len(generations),
        "verification_count": len(verifications),
        "summaries": summaries,
        "b16_forced_subset": {
            "candidate_count": len(forced_subset),
            "fresh_composition_count": sum(
                row["workload"] == "fresh-composition-valid-v2" for row in forced_subset
            ),
            "summary": _natural_summary(forced_subset),
        },
        "b16_natural_subset": {
            "candidate_count": len(natural_b16_subset),
            "exact_parsed_final_reproduced_count": sum(
                row["comparison"]["exact_parsed_final_match"]
                for row in natural_b16_subset
            ),
            "deployed_category_reproduced_count": sum(
                row["comparison"]["deployed_category_match"]
                for row in natural_b16_subset
            ),
        },
        "paired_task_table": paired,
        "long_natural_exits": [
            {
                "workload": row["workload"],
                "task_id": row["task_id"],
                "reasoning_end_position_token_count": row["bnat_max"][
                    "reasoning_end_position_token_count"
                ],
                "deployed_lean_category": row["bnat_max"]["deployed_lean_category"],
            }
            for row in paired
            if row["bnat_max"]["reasoning_end_position_token_count"] is not None
            and int(row["bnat_max"]["reasoning_end_position_token_count"]) > 16384
            and row["bnat_max"]["parsed_final_nonempty"]
        ],
        "newly_verified_requiring_more_than_16k_reasoning": [
            row["task_id"]
            for row in paired
            if row["comparison"]["bnat_max_only_verified"]
            and row["bnat_max"]["reasoning_end_position_token_count"] is not None
            and int(row["bnat_max"]["reasoning_end_position_token_count"]) > 16384
        ],
        "cost": {
            "total_generated_tokens": sum(
                int(row["raw_response_token_count"]) for row in generations
            ),
            "total_reasoning_tokens": sum(
                int(row["reasoning_token_count"]) for row in generations
            ),
            "total_final_tokens": sum(
                int(row["parsed_final_token_count"]) for row in generations
            ),
            "total_normalized_final_tokens": sum(
                int(row["normalized_final_token_count"]) for row in generations
            ),
            "generation_wall_time_seconds": sum(
                float(row["segment_wall_time_seconds"]) for row in generation_segments
            ),
            "verification_wall_time_seconds": sum(
                float(row["wall_time_seconds"]) for row in verification_segments
            ),
            "peak_gpu_memory_bytes": max(
                int(row["gpu_memory_peak_bytes"]) for row in generation_segments
            ),
        },
        "conclusion": conclusion,
        "artifact_integrity": {
            "capacity_attempts_jsonl_sha256": _file_sha256(
                artifact_dir / "capacity-attempts.jsonl"
            ),
            "generations_jsonl_sha256": _file_sha256(
                artifact_dir / "generations.jsonl"
            ),
            "generation_segments_jsonl_sha256": _file_sha256(
                artifact_dir / "generation-segments.jsonl"
            ),
            "verifications_jsonl_sha256": _file_sha256(
                artifact_dir / "verifications.jsonl"
            ),
            "verification_segments_jsonl_sha256": _file_sha256(
                artifact_dir / "verification-segments.jsonl"
            ),
            "raw_artifacts_git_ignored": True,
        },
    }
    wall = float(result["cost"]["generation_wall_time_seconds"])
    result["cost"]["throughput_generated_tokens_per_wall_second"] = (
        float(result["cost"]["total_generated_tokens"]) / wall if wall else None
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(evidence_dir / "natural-max-results.json", result)
    (evidence_dir / "NATURAL_MAX.md").write_text(
        _render_natural_readme(result), encoding="utf-8"
    )
    return result


def _natural_paired_table(
    selected: Sequence[SelectedTask],
    b16_generation: Mapping[str, Mapping[str, Any]],
    b16_verification_by_id: Mapping[str, Mapping[str, Any]],
    natural_by_task: Mapping[str, Mapping[str, Any]],
    natural_verification_by_task: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected_task in selected:
        task_id = selected_task.task.task_id
        b16 = b16_generation[task_id]
        b16_verification = b16_verification_by_id[str(b16["candidate_id"])]
        natural = natural_by_task[task_id]
        natural_verification = natural_verification_by_task[task_id]
        b16_category = b16_verification["deployed_normalized_interface"]["category"]
        natural_category = natural_verification["deployed_normalized_interface"][
            "category"
        ]
        rows.append(
            {
                "workload": selected_task.task.workload,
                "task_id": task_id,
                "rendered_prompt_token_count": natural["rendered_prompt_token_count"],
                "b16": {
                    "candidate_id": b16["candidate_id"],
                    "reasoning_token_count": b16["reasoning_token_count"],
                    "reasoning_exit": b16["reasoning_exit"],
                    "forced_at_budget": b16["reasoning_exit"] == "forced_at_budget",
                    "parsed_final_sha256": b16["parsed_final_sha256"],
                    "parsed_final_nonempty": bool(b16["parsed_final_exact"]),
                    "finish_reason": b16["finish_reason"],
                    "deployed_lean_category": b16_category,
                },
                "bnat_max": {
                    "candidate_id": natural["candidate_id"],
                    "max_model_len": natural["max_model_len"],
                    "available_generation_budget": natural["max_tokens"],
                    "reasoning_token_count": natural["reasoning_token_count"],
                    "reasoning_exit": natural["reasoning_exit"],
                    "reasoning_end_position_token_count": natural[
                        "reasoning_end_position_token_count"
                    ],
                    "parsed_final_sha256": natural["parsed_final_sha256"],
                    "parsed_final_nonempty": bool(natural["parsed_final_exact"]),
                    "parsed_final_token_count": natural["parsed_final_token_count"],
                    "normalized_final_token_count": natural[
                        "normalized_final_token_count"
                    ],
                    "normalization_applied": natural["normalization_applied"],
                    "format_diagnostics": _format_diagnostics(
                        [str(natural["parsed_final_exact"] or "")]
                    ),
                    "finish_reason": natural["finish_reason"],
                    "context_exhausted_no_final": natural["context_exhausted_no_final"],
                    "strict_lean_category": natural_verification[
                        "strict_parsed_interface"
                    ]["category"],
                    "deployed_lean_category": natural_category,
                    "verification_outcome_changed_by_normalization": (
                        natural_verification[
                            "verification_outcome_changed_by_normalization"
                        ]
                    ),
                    "raw_response_token_count": natural["raw_response_token_count"],
                },
                "comparison": {
                    "exact_parsed_final_match": b16["parsed_final_sha256"]
                    == natural["parsed_final_sha256"],
                    "deployed_category_match": b16_category == natural_category,
                    "b16_verified": b16_category == "verified",
                    "bnat_max_verified": natural_category == "verified",
                    "bnat_max_only_verified": natural_category == "verified"
                    and b16_category != "verified",
                },
            }
        )
    return rows


def _natural_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    natural = [row["bnat_max"] for row in rows]
    deployed_categories = Counter(str(row["deployed_lean_category"]) for row in natural)
    strict_categories = Counter(str(row["strict_lean_category"]) for row in natural)
    format_counts = {
        key: sum(int(row["format_diagnostics"][key]) for row in natural)
        for key in (
            "markdown_fence",
            "repeated_declaration_or_by",
            "sorry_or_admit",
            "apparent_natural_language",
        )
    }
    return {
        "candidate_count": count,
        "reasoning_tokens": _distribution(
            [int(row["reasoning_token_count"]) for row in natural]
        ),
        "available_generation_budget": _distribution(
            [int(row["available_generation_budget"]) for row in natural]
        ),
        "reasoning_exit_counts": dict(
            sorted(Counter(str(row["reasoning_exit"]) for row in natural).items())
        ),
        "nonempty_parsed_final": _count_fraction(
            sum(bool(row["parsed_final_nonempty"]) for row in natural), count
        ),
        "context_exhausted_no_final": _count_fraction(
            sum(bool(row["context_exhausted_no_final"]) for row in natural), count
        ),
        "parsed_final_tokens": _distribution(
            [int(row["parsed_final_token_count"]) for row in natural]
        ),
        "finish_reason_counts": dict(
            sorted(Counter(str(row["finish_reason"]) for row in natural).items())
        ),
        "deployed_lean": {
            "verified": _count_fraction(deployed_categories["verified"], count),
            "category_counts": dict(sorted(deployed_categories.items())),
        },
        "strict_lean": {
            "verified": _count_fraction(strict_categories["verified"], count),
            "category_counts": dict(sorted(strict_categories.items())),
        },
        "bnat_max_only_verified_tasks": sorted(
            str(row["task_id"])
            for row in rows
            if row["comparison"]["bnat_max_only_verified"]
        ),
        "normalization_changed": _count_fraction(
            sum(bool(row["normalization_applied"]) for row in natural), count
        ),
        "verification_outcome_changed_by_normalization": _count_fraction(
            sum(
                bool(row["verification_outcome_changed_by_normalization"])
                for row in natural
            ),
            count,
        ),
        "format_diagnostics": format_counts,
    }


def _natural_conclusion(
    capacity: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    forced_subset: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    long_exits = [
        row
        for row in paired
        if row["bnat_max"]["reasoning_end_position_token_count"] is not None
        and int(row["bnat_max"]["reasoning_end_position_token_count"]) > 16384
        and row["bnat_max"]["parsed_final_nonempty"]
    ]
    new_long_verified = [
        row for row in long_exits if row["comparison"]["bnat_max_only_verified"]
    ]
    exhausted_forced = sum(
        bool(row["bnat_max"]["context_exhausted_no_final"]) for row in forced_subset
    )
    if new_long_verified:
        category = "natural_long_thinking_adds_verified_capability"
    elif long_exits:
        category = "natural_long_thinking_exits_but_no_new_lean_capability"
    elif int(capacity["machine_supported_context"]) <= 32768:
        category = "machine_context_ceiling_too_close_to_b16"
    elif forced_subset and exhausted_forced > len(forced_subset) / 2:
        category = "natural_thinking_consumes_available_context"
    else:
        category = "natural_max_no_direct_long_thinking_capability_signal"
    return {
        "category": category,
        "preflight_not_statistically_powered": True,
        "machine_supported_context": capacity["machine_supported_context"],
        "b16_forced_candidate_count": len(forced_subset),
        "b16_forced_context_exhausted_count": exhausted_forced,
        "long_natural_exit_count": len(long_exits),
        "bnat_max_only_verified_tasks": sorted(
            str(row["task_id"])
            for row in paired
            if row["comparison"]["bnat_max_only_verified"]
        ),
        "new_verified_after_more_than_16k_reasoning": sorted(
            str(row["task_id"]) for row in new_long_verified
        ),
        "larger_context_program_authorized": False,
    }


def _render_natural_readme(result: Mapping[str, Any]) -> str:
    combined = result["summaries"]["combined"]
    forced = result["b16_forced_subset"]
    return "\n".join(
        [
            "# Qwen3.5-4B BNAT-MAX natural-thinking continuation",
            "",
            (
                "**OBSERVED:** the exact local machine accepted "
                f"`max_model_len={result['capacity']['machine_supported_context']}` "
                "under the frozen BF16 runtime and completed all 16 paired "
                "BNAT-MAX candidates without a reasoning-token budget."
            ),
            "",
            (
                f"BNAT-MAX produced non-empty parsed finals for "
                f"{combined['nonempty_parsed_final']['count']}/16 candidates and "
                f"Lean verified {combined['deployed_lean']['verified']['count']}/16."
            ),
            "",
            (
                f"Within the immutable 11-candidate B16-forced subset, "
                f"{forced['summary']['context_exhausted_no_final']['count']} "
                "consumed the available context without final content."
            ),
            "",
            (f"**OBSERVED conclusion:** `{result['conclusion']['category']}`."),
            "",
            (
                "This is a 16-candidate capability check, not evidence of "
                "statistical superiority. Historical Stage 1 and B4/B8/B16 "
                "artifacts remain unchanged, and no larger program is authorized."
            ),
            "",
        ]
    )


def _count_fraction(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "fraction": count / total if total else 0.0}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _jsonl_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(bool(line) for line in path.read_text(encoding="utf-8").splitlines())
