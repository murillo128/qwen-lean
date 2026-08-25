from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .baseline import LoRAAdapterSpec, render_prompt, vllm_engine_kwargs
from .dataset_v2 import sha256_file
from .generalist_v2 import MODEL_ID, MODEL_REVISION, GeneralistV2Config
from .generalist_v2_dataset import _read_json
from .generalist_v2_evaluation import (
    _evaluation_phase1_config,
    _load_canonical_subset,
    _prime_verifiers,
    _validate_oracles,
    materialize_q0_workload,
)
from .generalist_v2_parity import (
    _adapter_identity,
    _arm_pair_summary,
    _audit_summary,
    _forward_sensitivity_summary,
    _hf_forward_and_generate,
    _results_by_probe,
    _verification_summary,
    _verify_output,
    _vllm_generate_arm,
    _vllm_source_binding,
    inspect_vllm_lora_worker,
)
from .prompt import normalize_transport
from .schema import TaskRecord

Q4_CANARY_GATE_ID = "qwen35-q4-final-inference-canary-v1"
Q4_CANARY_PROBE_SCHEMA_VERSION = "generalist-v2-q4-canary-probes-v1"
Q4_CANARY_RUNTIME_SCHEMA_VERSION = "generalist-v2-q4-canary-runtime-v1"
Q4_CANARY_RAW_VERIFICATION_SCHEMA_VERSION = (
    "generalist-v2-q4-canary-raw-verification-v1"
)
Q4_CANARY_EVIDENCE_SCHEMA_VERSION = "generalist-v2-q4-canary-evidence-v1"
Q4_FAILURE_DIAGNOSIS_SCHEMA_VERSION = (
    "generalist-v2-q4-fresh-test-failure-diagnosis-v1"
)
Q4_ADAPTER_MODEL_SHA256 = (
    "2398be7ac95db85d646bce66762abcea96487a93b7d92508ddcc274914ef470e"
)
Q4_CANARY_WORKLOAD_ID = "fresh-composition-valid-v2"
Q4_FINAL_WORKLOAD_ID = "fresh-composition-test-v2"
Q4_CANARY_STRUCTURE_CLASSES = ("direct", "branching", "deep")
Q4_CANARY_PROBES_PER_CLASS = 4
Q4_CANARY_PROBE_COUNT = 12
Q4_FAILURE_SAMPLE_COUNT = 100
Q4_CANARY_REQUIRED_GATES = {
    "exact_q4_adapter_bound",
    "known_positive_probe_set_complete",
    "hf_q4_tensor_load_complete",
    "vllm_runtime_transform_payload_complete",
    "vllm_static_mapping_complete",
    "hf_q4_forward_effect",
    "vllm_q4_inference_effect",
    "cross_backend_q4_consistency",
    "all_expected_outputs_present",
    "zero_verifier_infrastructure_errors",
    "no_base_equivalent_fallback",
}


def _json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Q4 canary artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row is not an object: {path}:{line_number}")
            yield value


def _require_q4_adapter(
    config: GeneralistV2Config, adapter_dir: Path
) -> dict[str, Any]:
    identity = _adapter_identity(config, adapter_dir, label="Q4")
    if identity["adapter_model_sha256"] != Q4_ADAPTER_MODEL_SHA256:
        raise ValueError("Q4 canary adapter hash differs from the amendment")
    return identity


def _known_positive_selection(
    results: Iterable[Mapping[str, Any]],
    task_metadata: Mapping[str, Mapping[str, Any]],
    *,
    per_class: int = Q4_CANARY_PROBES_PER_CLASS,
) -> tuple[list[str], dict[str, dict[str, Any]], int]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    result_count = 0
    for row in results:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_metadata:
            raise ValueError(f"extended Q4 result has unknown task: {task_id}")
        by_task[task_id].append(row)
        result_count += 1
    if len(by_task) != len(task_metadata):
        raise ValueError("extended Q4 result task membership differs")

    selected: list[str] = []
    observations: dict[str, dict[str, Any]] = {}
    for structural_class in Q4_CANARY_STRUCTURE_CLASSES:
        eligible = []
        for task_id, rows in by_task.items():
            if task_metadata[task_id].get("structural_class") != structural_class:
                continue
            verified = [row for row in rows if row.get("category") == "verified"]
            if verified:
                eligible.append((task_id, rows, verified))
        eligible.sort(key=lambda item: (-len(item[2]), item[0]))
        if len(eligible) < per_class:
            raise ValueError(
                f"Q4 canary lacks {per_class} known-positive {structural_class} tasks"
            )
        for task_id, rows, verified in eligible[:per_class]:
            selected.append(task_id)
            observations[task_id] = {
                "structural_class": structural_class,
                "prior_candidate_count": len(rows),
                "prior_verified_candidate_count": len(verified),
                "prior_verified_candidate_indices": sorted(
                    int(row["candidate_index"]) for row in verified
                ),
                "prior_verified_output_sha256s": sorted(
                    {
                        _text_sha256(normalize_transport(str(row["candidate_text"])))
                        for row in verified
                    }
                ),
            }
    return selected, observations, result_count


def build_q4_canary_probes(
    config: GeneralistV2Config,
    package_root: Path,
    view_dir: Path,
    extended_evidence_path: Path,
    extended_results_path: Path,
    extended_generation_metadata_path: Path,
    q4_adapter_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze 12 high-density known-positive Q4 validation probes."""

    config.validate()
    adapter = _require_q4_adapter(config, q4_adapter_dir)
    extended = _read_json(extended_evidence_path)
    generation_metadata = _read_json(extended_generation_metadata_path)
    workload = extended.get("evaluated_checkpoint", {}).get("workloads", {}).get(
        Q4_CANARY_WORKLOAD_ID, {}
    )
    if (
        extended.get("schema_version") != "generalist-v2-extended-validation-v1"
        or extended.get("status")
        != "selected-checkpoint-extended-validation-complete"
        or extended.get("screening_selected_checkpoint") != "Q4"
        or extended.get("evaluated_checkpoint", {}).get("adapter_model_sha256")
        != Q4_ADAPTER_MODEL_SHA256
        or workload.get("adapter_model_sha256") != Q4_ADAPTER_MODEL_SHA256
        or generation_metadata.get("workload_id") != Q4_CANARY_WORKLOAD_ID
        or generation_metadata.get("checkpoint_id") != "Q4"
        or generation_metadata.get("adapter", {}).get("adapter_model_sha256")
        != Q4_ADAPTER_MODEL_SHA256
        or int(generation_metadata.get("task_count", 0)) != 406
        or int(generation_metadata.get("candidate_count", 0)) != 406 * 64
        or int(workload.get("candidate_count", 0)) != 406 * 64
    ):
        raise ValueError("Q4 canary extended-validation binding differs")
    task_metadata = generation_metadata.get("task_metadata")
    if not isinstance(task_metadata, dict) or len(task_metadata) != 406:
        raise ValueError("Q4 canary task metadata differs")

    selected, observations, result_count = _known_positive_selection(
        _iter_jsonl(extended_results_path), task_metadata
    )
    if result_count != 406 * 64 or len(selected) != Q4_CANARY_PROBE_COUNT:
        raise ValueError("Q4 canary source result count differs")
    evidence_per_task = {
        str(item["task_id"]): item
        for item in workload.get("raw_candidate_evidence", {}).get("per_task", [])
    }
    for task_id in selected:
        source = evidence_per_task.get(task_id, {})
        if int(source.get("verified_candidate_count", -1)) != observations[task_id][
            "prior_verified_candidate_count"
        ]:
            raise ValueError("Q4 known-positive count differs from compact evidence")

    tasks, verification, _targets, materialized_metadata = materialize_q0_workload(
        Q4_CANARY_WORKLOAD_ID, package_root, view_dir
    )
    by_id = {task.id: task for task in tasks}
    selected_records = _load_canonical_subset(package_root, selected)
    probes = []
    for probe_index, task_id in enumerate(selected):
        task = by_id[task_id]
        prompt = render_prompt(task)
        metadata = materialized_metadata[task_id]
        if metadata["structural_class"] != observations[task_id]["structural_class"]:
            raise ValueError("Q4 canary structural class differs")
        probes.append(
            {
                "probe_id": f"q4-fresh-valid-{probe_index:02d}",
                "workload_id": Q4_CANARY_WORKLOAD_ID,
                "statement_id": task_id,
                "structural_class": metadata["structural_class"],
                "generator_family": metadata["generator_family"],
                "task": task.to_dict(),
                "verification_task": verification[task_id].to_dict(),
                "target_completions": [
                    variant.completion
                    for variant in selected_records[task_id].proof_variants
                ],
                "prompt": prompt,
                "prompt_sha256": _text_sha256(prompt),
                **observations[task_id],
            }
        )
    class_counts = Counter(str(item["structural_class"]) for item in probes)
    if class_counts != Counter({item: 4 for item in Q4_CANARY_STRUCTURE_CLASSES}):
        raise RuntimeError("Q4 canary structural balance differs")

    value = {
        "schema_version": Q4_CANARY_PROBE_SCHEMA_VERSION,
        "gate_id": Q4_CANARY_GATE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter": adapter,
        "workload_id": Q4_CANARY_WORKLOAD_ID,
        "selection": {
            "policy": (
                "top four tasks per direct/branching/deep class by prior Q4 n=64 "
                "Lean-verified candidate count, deterministic task-id tie break"
            ),
            "uses_validation_only": True,
            "known_positive_required": True,
            "probe_count": len(probes),
            "structural_class_counts": dict(sorted(class_counts.items())),
        },
        "deterministic_generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": int(config.evaluation["sampling"]["max_new_tokens"]),
            "eos_semantics": "tokenizer_eos_or_token_limit",
            "add_special_tokens": False,
        },
        "source_artifacts": {
            "extended_evidence_sha256": sha256_file(extended_evidence_path),
            "extended_results_sha256": sha256_file(extended_results_path),
            "extended_generation_metadata_sha256": sha256_file(
                extended_generation_metadata_path
            ),
            "prior_candidate_count": result_count,
        },
        "probes": probes,
        "ordered_probe_ids_sha256": _json_sha256(
            [str(item["probe_id"]) for item in probes]
        ),
    }
    _write_new_json(output, value)
    return value


def _load_q4_probe_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    probes = value.get("probes")
    if (
        value.get("schema_version") != Q4_CANARY_PROBE_SCHEMA_VERSION
        or value.get("gate_id") != Q4_CANARY_GATE_ID
        or value.get("model_id") != MODEL_ID
        or value.get("model_revision") != MODEL_REVISION
        or value.get("adapter", {}).get("adapter_model_sha256")
        != Q4_ADAPTER_MODEL_SHA256
        or not isinstance(probes, list)
        or len(probes) != Q4_CANARY_PROBE_COUNT
        or len({str(item.get("probe_id")) for item in probes})
        != Q4_CANARY_PROBE_COUNT
    ):
        raise ValueError("invalid Q4 canary probe manifest")
    for probe in probes:
        task = TaskRecord.from_dict(probe["task"])
        if (
            task.id != probe["statement_id"]
            or render_prompt(task) != probe["prompt"]
            or _text_sha256(str(probe["prompt"])) != probe["prompt_sha256"]
            or int(probe["prior_verified_candidate_count"]) < 1
        ):
            raise ValueError("Q4 canary probe binding differs")
    return value


def _hf_loaded_adapter_audit(model: Any, adapter_dir: Path) -> dict[str, Any]:
    import torch
    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        load_peft_weights,
    )

    source = load_peft_weights(str(adapter_dir), device="cpu")
    loaded = get_peft_model_state_dict(model, adapter_name="q4")
    source_keys = set(source)
    loaded_keys = set(loaded)
    missing = sorted(source_keys - loaded_keys)
    unexpected = sorted(loaded_keys - source_keys)
    mismatched = []
    for key in sorted(source_keys & loaded_keys):
        source_tensor = source[key].detach().cpu().contiguous()
        loaded_tensor = loaded[key].detach().cpu().contiguous()
        if source_tensor.dtype != loaded_tensor.dtype or not torch.equal(
            source_tensor, loaded_tensor
        ):
            mismatched.append(key)
    return {
        "status": "passed" if not (missing or unexpected or mismatched) else "failed",
        "source_tensor_count": len(source),
        "loaded_tensor_count": len(loaded),
        "missing_adapter_tensors": missing,
        "unexpected_adapter_tensors": unexpected,
        "mismatched_adapter_tensors": mismatched,
        "source_adapter_model_sha256": sha256_file(
            adapter_dir / "adapter_model.safetensors"
        ),
    }


def run_hf_q4_canary(
    config: GeneralistV2Config,
    probe_manifest_path: Path,
    q4_adapter_dir: Path,
    model_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    """Run deterministic BF16 Transformers Base/PEFT-Q4 canary arms."""

    config.validate()
    manifest = _load_q4_probe_manifest(probe_manifest_path)
    adapter = _require_q4_adapter(config, q4_adapter_dir)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("HF Q4 canary requires the training runtime") from error
    if not torch.cuda.is_available():
        raise RuntimeError("HF Q4 canary requires project-controlled local CUDA")

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        q4_adapter_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Q4 canary tokenizer lacks EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        base,
        q4_adapter_dir,
        adapter_name="q4",
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    model.eval()
    model.config.use_cache = True
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Q4 canary adapter unexpectedly remains trainable")
    load_audit = _hf_loaded_adapter_audit(model, q4_adapter_dir)
    if load_audit["status"] != "passed":
        raise RuntimeError("HF Q4 adapter tensor load audit failed")

    probes = manifest["probes"]
    max_new_tokens = int(manifest["deterministic_generation"]["max_new_tokens"])
    arms: dict[str, list[dict[str, Any]]] = {}
    logits: dict[tuple[str, str], Any] = {}
    for arm_id, adapter_name in (("hf_base", None), ("hf_q4", "q4")):
        rows = []
        for probe in probes:
            row, last_logits = _hf_forward_and_generate(
                model,
                tokenizer,
                probe,
                adapter_name=adapter_name,
                max_new_tokens=max_new_tokens,
            )
            rows.append(row)
            logits[(arm_id, str(probe["probe_id"]))] = last_logits
        arms[arm_id] = rows

    sensitivity = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        delta = logits[("hf_q4", probe_id)] - logits[("hf_base", probe_id)]
        sensitivity.append(
            {
                "probe_id": probe_id,
                "base_arm": "hf_base",
                "adapter_arm": "hf_q4",
                "maximum_absolute_logit_delta": float(delta.abs().max()),
                "mean_absolute_logit_delta": float(delta.abs().mean()),
                "l2_logit_delta": float(torch.linalg.vector_norm(delta)),
            }
        )
    value = {
        "schema_version": Q4_CANARY_RUNTIME_SCHEMA_VERSION,
        "gate_id": Q4_CANARY_GATE_ID,
        "runtime": "transformers-peft-bfloat16",
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter": adapter,
        "deterministic_generation": manifest["deterministic_generation"],
        "hf_adapter_load_audit": load_audit,
        "arms": arms,
        "forward_sensitivity": sensitivity,
        "execution": {
            "local_cuda": True,
            "cuda_device": properties.name,
            "cuda_device_total_memory_bytes": int(properties.total_memory),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device_index)
            ),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
        },
    }
    _write_new_json(output, value)
    del model, base, tokenizer, logits
    gc.collect()
    torch.cuda.empty_cache()
    return value


def run_vllm_q4_canary(
    config: GeneralistV2Config,
    base_evaluation_config: Path,
    probe_manifest_path: Path,
    q4_adapter_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Run deterministic BF16 vLLM Base/Q4 arms and the exact load audit."""

    config.validate()
    manifest = _load_q4_probe_manifest(probe_manifest_path)
    adapter = _require_q4_adapter(config, q4_adapter_dir)
    from .qwen35_vllm_lora import prepare_qwen35_vllm_adapter

    compatibility = prepare_qwen35_vllm_adapter(q4_adapter_dir)
    runtime_dir = Path(str(compatibility["runtime_adapter_dir"]))
    phase1 = _evaluation_phase1_config(base_evaluation_config, config)
    if phase1.engine.get(
        "use_flashinfer_sampler", phase1.engine.get("flashinfer_sampler")
    ) is False:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:
        raise RuntimeError("vLLM Q4 canary requires the screening runtime") from error
    if vllm.__version__ != str(phase1.engine["version"]):
        raise RuntimeError("vLLM Q4 canary runtime differs from final evaluator")

    deterministic = manifest["deterministic_generation"]
    engine_sampling = {
        "seed": 0,
        "candidates_per_task": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": int(deterministic["max_new_tokens"]),
    }
    adapter_spec = LoRAAdapterSpec(
        adapter_id="qwen-lean-generalist-v2-q4-final-canary",
        path=q4_adapter_dir.resolve(),
        rank=int(config.lora["r"]),
        base_model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        runtime_path=runtime_dir,
    )
    llm = LLM(**vllm_engine_kwargs(phase1, engine_sampling, adapter_spec))
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=int(deterministic["max_new_tokens"]),
        seed=0,
        ignore_eos=False,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        logprobs=8,
    )
    request = LoRARequest(
        lora_name="qwen-lean-generalist-v2-q4-final-canary",
        lora_int_id=4,
        lora_path=str(runtime_dir.resolve()),
    )
    llm.llm_engine.add_lora(request)
    audits = llm.collective_rpc(
        inspect_vllm_lora_worker,
        args=(
            4,
            str(q4_adapter_dir.resolve()),
            str(runtime_dir.resolve()),
            dict(config.lora["expected_module_counts"]),
            str(config.lora["target_regex"]),
        ),
    )
    failed = [audit for audit in audits if audit.get("status") != "passed"]
    if failed:
        diagnostic = {
            "schema_version": Q4_CANARY_RUNTIME_SCHEMA_VERSION,
            "gate_id": Q4_CANARY_GATE_ID,
            "status": "static-audit-failed-before-generation",
            "runtime": "vllm-lora-request-bfloat16",
            "probe_manifest_sha256": sha256_file(probe_manifest_path),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adapter": adapter,
            "vllm_adapter_compatibility": compatibility,
            "static_mapping_audits": audits,
            "arms": {},
        }
        _write_new_json(output, diagnostic)
        errors = [error for audit in failed for error in audit.get("errors", [])]
        raise RuntimeError(
            "vLLM Q4 static mapping audit failed: " + "; ".join(errors[:8])
        )
    probes = manifest["probes"]
    arms = {
        "vllm_base": _vllm_generate_arm(
            llm, sampling_params, probes, lora_request=None
        ),
        "vllm_q4": _vllm_generate_arm(
            llm, sampling_params, probes, lora_request=request
        ),
    }
    value = {
        "schema_version": Q4_CANARY_RUNTIME_SCHEMA_VERSION,
        "gate_id": Q4_CANARY_GATE_ID,
        "runtime": "vllm-lora-request-bfloat16",
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter": adapter,
        "vllm_adapter_compatibility": compatibility,
        "deterministic_generation": deterministic,
        "vllm": _vllm_source_binding(),
        "engine": phase1.engine,
        "static_mapping_audits": audits,
        "arms": arms,
        "execution": {
            "local_cuda": True,
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "audit_collective_rpc_pickle_enabled": True,
            "pytorch_native_sampler_forced": True,
        },
    }
    _write_new_json(output, value)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return value


def compact_q4_canary_evidence(
    config: GeneralistV2Config,
    probe_manifest_path: Path,
    hf_runtime_path: Path,
    vllm_runtime_path: Path,
    general_lean_project_root: Path,
    raw_verification_output: Path,
    output: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Lean-check all canary arms and enforce the blocking Q4 gate."""

    config.validate()
    manifest = _load_q4_probe_manifest(probe_manifest_path)
    hf_runtime = _read_json(hf_runtime_path)
    vllm_runtime = _read_json(vllm_runtime_path)
    manifest_sha = sha256_file(probe_manifest_path)
    for runtime, name in ((hf_runtime, "HF"), (vllm_runtime, "vLLM")):
        if (
            runtime.get("schema_version") != Q4_CANARY_RUNTIME_SCHEMA_VERSION
            or runtime.get("gate_id") != Q4_CANARY_GATE_ID
            or runtime.get("probe_manifest_sha256") != manifest_sha
            or runtime.get("model_id") != MODEL_ID
            or runtime.get("model_revision") != MODEL_REVISION
            or runtime.get("adapter", {}).get("adapter_model_sha256")
            != Q4_ADAPTER_MODEL_SHA256
        ):
            raise ValueError(f"{name} Q4 canary runtime binding differs")

    probes = {str(item["probe_id"]): item for item in manifest["probes"]}
    tasks = {
        str(probe["statement_id"]): TaskRecord.from_dict(probe["verification_task"])
        for probe in probes.values()
    }
    targets = {
        str(probe["statement_id"]): tuple(
            str(item) for item in probe["target_completions"]
        )
        for probe in probes.values()
    }
    verifiers, already_validated = _prime_verifiers(
        tasks,
        general_lean_project_root,
        candidate_timeout_seconds=float(
            config.evaluation["verifier_timeout_seconds"]
        ),
        workers=workers,
        targets=targets,
    )
    _validate_oracles(
        verifiers,
        tasks,
        targets,
        workers=workers,
        already_validated=already_validated,
    )
    arm_ids = ("hf_base", "hf_q4", "vllm_base", "vllm_q4")
    runtime_for_arm = {
        "hf_base": hf_runtime,
        "hf_q4": hf_runtime,
        "vllm_base": vllm_runtime,
        "vllm_q4": vllm_runtime,
    }
    jobs = []
    for arm_id in arm_ids:
        for probe_id, result in _results_by_probe(
            runtime_for_arm[arm_id], arm_id
        ).items():
            probe = probes[probe_id]
            verifier = verifiers[str(probe["verification_task"]["preamble"])]
            jobs.append((verifier, probe, arm_id, result))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        verification_rows = list(executor.map(lambda args: _verify_output(*args), jobs))
    raw_verification = {
        "schema_version": Q4_CANARY_RAW_VERIFICATION_SCHEMA_VERSION,
        "gate_id": Q4_CANARY_GATE_ID,
        "probe_manifest_sha256": manifest_sha,
        "hf_runtime_sha256": sha256_file(hf_runtime_path),
        "vllm_runtime_sha256": sha256_file(vllm_runtime_path),
        "rows": verification_rows,
    }
    _write_new_json(raw_verification_output, raw_verification)

    summaries = {
        arm_id: _verification_summary(verification_rows, arm_id)
        for arm_id in arm_ids
    }
    by_arm = {
        arm_id: _results_by_probe(runtime_for_arm[arm_id], arm_id)
        for arm_id in arm_ids
    }
    hf_pair = _arm_pair_summary(by_arm["hf_base"], by_arm["hf_q4"])
    vllm_pair = _arm_pair_summary(by_arm["vllm_base"], by_arm["vllm_q4"])
    hf_sensitivity = _forward_sensitivity_summary(hf_runtime, "hf_q4")
    cross_exact = sum(
        by_arm["hf_q4"][probe_id]["normalized_text_sha256"]
        == by_arm["vllm_q4"][probe_id]["normalized_text_sha256"]
        for probe_id in probes
    )
    verified_by_arm = {
        arm_id: {
            str(row["probe_id"])
            for row in verification_rows
            if row["arm_id"] == arm_id and row["lean_category"] == "verified"
        }
        for arm_id in arm_ids
    }
    common_q4_verified = verified_by_arm["hf_q4"] & verified_by_arm["vllm_q4"]
    compatibility = vllm_runtime.get("vllm_adapter_compatibility", {})
    static_audit = _audit_summary(vllm_runtime.get("static_mapping_audits"))
    hf_load = hf_runtime.get("hf_adapter_load_audit", {})
    expected_outputs = all(
        summaries[arm_id]["candidate_count"] == Q4_CANARY_PROBE_COUNT
        for arm_id in arm_ids
    )
    no_infrastructure_errors = all(
        summaries[arm_id]["infrastructure_error_count"] == 0
        for arm_id in arm_ids
    )
    hf_effect = hf_sensitivity["changed_probe_count"] == Q4_CANARY_PROBE_COUNT
    vllm_effect = (
        vllm_pair["output_difference_count"] > 0
        or vllm_pair["maximum_same_first_token_logprob_delta"] > 1e-6
    )
    requirements = {
        "exact_q4_adapter_bound": (
            manifest["adapter"]["adapter_model_sha256"]
            == Q4_ADAPTER_MODEL_SHA256
            and hf_runtime["adapter"]["adapter_model_sha256"]
            == Q4_ADAPTER_MODEL_SHA256
            and vllm_runtime["adapter"]["adapter_model_sha256"]
            == Q4_ADAPTER_MODEL_SHA256
        ),
        "known_positive_probe_set_complete": (
            len(probes) == Q4_CANARY_PROBE_COUNT
            and all(int(item["prior_verified_candidate_count"]) > 0 for item in probes.values())
            and Counter(item["structural_class"] for item in probes.values())
            == Counter({item: 4 for item in Q4_CANARY_STRUCTURE_CLASSES})
        ),
        "hf_q4_tensor_load_complete": (
            hf_load.get("status") == "passed"
            and hf_load.get("source_adapter_model_sha256")
            == Q4_ADAPTER_MODEL_SHA256
            and int(hf_load.get("source_tensor_count", 0)) == 496
            and int(hf_load.get("loaded_tensor_count", 0)) == 496
            and not hf_load.get("missing_adapter_tensors")
            and not hf_load.get("unexpected_adapter_tensors")
            and not hf_load.get("mismatched_adapter_tensors")
        ),
        "vllm_runtime_transform_payload_complete": (
            compatibility.get("source_adapter_model_sha256")
            == Q4_ADAPTER_MODEL_SHA256
            and compatibility.get("source_tensor_payload_sha256")
            == compatibility.get("runtime_tensor_payload_sha256")
            and int(compatibility.get("tensor_count", 0)) == 496
        ),
        "vllm_static_mapping_complete": (
            static_audit["status"] == "passed"
            and int(static_audit["raw_tensor_count"] or 0) == 496
            and int(static_audit["peft_module_count"] or 0) == 248
            and int(static_audit["runtime_module_count"] or 0) == 152
            and not static_audit["missing_loaded_runtime_modules"]
            and not static_audit["unexpected_loaded_runtime_modules"]
        ),
        "hf_q4_forward_effect": hf_effect,
        "vllm_q4_inference_effect": vllm_effect,
        "cross_backend_q4_consistency": (
            (cross_exact > 0 or len(common_q4_verified) > 0)
            and summaries["hf_q4"]["lean_verified_count"] > 0
            and summaries["vllm_q4"]["lean_verified_count"] > 0
        ),
        "all_expected_outputs_present": expected_outputs,
        "zero_verifier_infrastructure_errors": no_infrastructure_errors,
        "no_base_equivalent_fallback": hf_effect and vllm_effect,
    }
    passed = all(requirements.values())
    verification_by_key = {
        (str(row["arm_id"]), str(row["probe_id"])): row
        for row in verification_rows
    }
    outputs = []
    for probe_id in sorted(probes):
        probe = probes[probe_id]
        arms = {}
        for arm_id in arm_ids:
            runtime_row = by_arm[arm_id][probe_id]
            verified = verification_by_key[(arm_id, probe_id)]
            arms[arm_id] = {
                "raw_text": runtime_row["raw_text"],
                "raw_text_sha256": runtime_row["raw_text_sha256"],
                "normalized_text_sha256": runtime_row["normalized_text_sha256"],
                "generated_tokens": runtime_row["generated_tokens"],
                "finish_reason": runtime_row["finish_reason"],
                "first_token_id": runtime_row.get("first_token_id"),
                "first_token_logprob": runtime_row.get("first_token_logprob"),
                "next_token_top8": runtime_row.get("next_token_top8"),
                "last_token_logits_float32_sha256": runtime_row.get(
                    "last_token_logits_float32_sha256"
                ),
                "lean_category": verified["lean_category"],
                "lean_exit_code": verified["lean_exit_code"],
            }
        outputs.append(
            {
                "probe_id": probe_id,
                "statement_id": probe["statement_id"],
                "structural_class": probe["structural_class"],
                "prior_verified_candidate_count": probe[
                    "prior_verified_candidate_count"
                ],
                "arms": arms,
            }
        )
    value = {
        "schema_version": Q4_CANARY_EVIDENCE_SCHEMA_VERSION,
        "gate_id": Q4_CANARY_GATE_ID,
        "status": "passed" if passed else "failed",
        "classification": (
            "PASS: exact Q4 is loaded, active, and consistent across HF/PEFT and vLLM"
            if passed
            else "FAIL: exact Q4 inference integrity is not established"
        ),
        "model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION},
        "adapter_model_sha256": Q4_ADAPTER_MODEL_SHA256,
        "requirements": requirements,
        "probe_selection": manifest["selection"],
        "deterministic_generation": manifest["deterministic_generation"],
        "hf_adapter_load_audit": hf_load,
        "vllm_adapter_compatibility": {
            key: compatibility.get(key)
            for key in (
                "schema_version",
                "source_adapter_model_sha256",
                "runtime_adapter_model_sha256",
                "source_tensor_payload_sha256",
                "runtime_tensor_payload_sha256",
                "tensor_count",
            )
        },
        "vllm_static_mapping_audit": static_audit,
        "functional_parity": {
            "hf_base_vs_q4": {
                "pair": hf_pair,
                "forward_sensitivity": hf_sensitivity,
            },
            "vllm_base_vs_q4": {"pair": vllm_pair},
            "hf_q4_vs_vllm_q4": {
                "exact_output_match_count": cross_exact,
                "both_lean_verified_probe_count": len(common_q4_verified),
                "both_lean_verified_probe_ids": sorted(common_q4_verified),
            },
        },
        "arm_summaries": summaries,
        "outputs": outputs,
        "artifacts": {
            "probe_manifest_sha256": manifest_sha,
            "hf_runtime_sha256": sha256_file(hf_runtime_path),
            "vllm_runtime_sha256": sha256_file(vllm_runtime_path),
            "raw_verification_sha256": sha256_file(raw_verification_output),
            **manifest["source_artifacts"],
        },
    }
    _write_new_json(output, value)
    return value


_AT_CONSTANT = re.compile(r"@[^\s,()\[\]{}⟨⟩]+(?:\{[^}]*\})?")
_INTEGER = re.compile(r"(?<![A-Za-z_])\d+(?![A-Za-z_])")


def q4_failure_template(text: str) -> str:
    value = " ".join(normalize_transport(text).split())
    value = _AT_CONSTANT.sub("@CONST", value)
    value = _INTEGER.sub("N", value)
    return value


def q4_failure_signals(diagnostics: str, candidate_text: str) -> list[str]:
    lower = diagnostics.lower()
    signals = []
    if (
        "unexpected end of input" in lower
        or "unterminated" in lower
        or candidate_text.count("⟨") > candidate_text.count("⟩")
        or candidate_text.count("(") > candidate_text.count(")")
    ):
        signals.append("premature_eos_or_incomplete_proof")
    if any(
        marker in lower
        for marker in (
            "unexpected token",
            "invalid syntax",
            "parser error",
            "unexpected identifier",
        )
    ):
        signals.append("syntax_or_parser")
    if "unknownidentifier" in lower or "unknown constant" in lower:
        signals.append("unknown_or_mismatched_lemma")
    if "unsolved goals" in lower:
        signals.append("unsolved_goals")
    if (
        "not an inductive type" in lower
        or "invalid `⟨" in lower
        or "invalid constructor" in lower
        or "no goals to be solved" in lower
    ):
        signals.append("structurally_wrong_proof_or_template")
    if any(
        marker in lower
        for marker in (
            "type mismatch",
            "failed to synthesize",
            "failed to unify",
            "function expected",
            "application type mismatch",
            "invalid field",
            "type expected",
        )
    ):
        signals.append("type_or_elaboration")
    return signals or ["other_lean_rejection"]


def q4_primary_failure_category(signals: Sequence[str]) -> str:
    precedence = (
        "premature_eos_or_incomplete_proof",
        "syntax_or_parser",
        "structurally_wrong_proof_or_template",
        "unknown_or_mismatched_lemma",
        "unsolved_goals",
        "type_or_elaboration",
        "other_lean_rejection",
    )
    return next(item for item in precedence if item in signals)


def _proof_head(text: str) -> str:
    normalized = normalize_transport(text).lstrip()
    if normalized.startswith("exact ⟨"):
        return "exact-constructor"
    if normalized.startswith("exact @"):
        return "exact-explicit-constant"
    match = re.match(r"([A-Za-z_][A-Za-z0-9_']*)", normalized)
    return "empty" if match is None else match.group(1)


def _primary_diagnostic_line(diagnostics: str) -> str:
    for line in diagnostics.splitlines():
        if "error" in line.lower():
            return re.sub(r"^Candidate\.lean:\d+:\d+:\s*", "", line).strip()
    return ""


def diagnose_q4_fresh_test_failures(
    final_evidence_path: Path,
    generations_path: Path,
    results_path: Path,
    generation_metadata_path: Path,
    output: Path,
    *,
    sample_count: int = Q4_FAILURE_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Classify a deterministic sample from the existing Q4 final candidates."""

    final = _read_json(final_evidence_path)
    metadata = _read_json(generation_metadata_path)
    selected = final.get("workloads", {}).get(Q4_FINAL_WORKLOAD_ID, {}).get(
        "models", {}
    ).get("selected", {})
    generation_sha = sha256_file(generations_path)
    if (
        final.get("schema_version") != "generalist-v2-final-assessment-v1"
        or final.get("status") != "complete"
        or final.get("selected_checkpoint") != "Q4"
        or final.get("selected_adapter_model_sha256") != Q4_ADAPTER_MODEL_SHA256
        or metadata.get("schema_version") != "generalist-v2-final-generation-v1"
        or metadata.get("workload_id") != Q4_FINAL_WORKLOAD_ID
        or metadata.get("selected_checkpoint") != "Q4"
        or metadata.get("adapter", {}).get("adapter_model_sha256")
        != Q4_ADAPTER_MODEL_SHA256
        or metadata.get("generation_sha256") != generation_sha
        or selected.get("generation_sha256") != generation_sha
        or int(metadata.get("candidate_count", 0)) != 415 * 8
        or sample_count < 1
    ):
        raise ValueError("Q4 fresh-test diagnosis binding differs")

    generations = {}
    for row in _iter_jsonl(generations_path):
        key = (str(row["task"]["id"]), int(row["candidate_index"]))
        if key in generations:
            raise ValueError("duplicate Q4 fresh-test generation")
        generations[key] = row
    results = {}
    for row in _iter_jsonl(results_path):
        key = (str(row["task_id"]), int(row["candidate_index"]))
        if key in results:
            raise ValueError("duplicate Q4 fresh-test result")
        results[key] = row
    if (
        set(generations) != set(results)
        or len(generations) != 415 * 8
        or any(row.get("category") == "verified" for row in results.values())
    ):
        raise ValueError("Q4 fresh-test stored candidate membership differs")

    joined = []
    for key in sorted(generations):
        generation = generations[key]
        result = results[key]
        raw_text = str(generation["text"])
        if (
            normalize_transport(raw_text)
            != normalize_transport(str(result["candidate_text"]))
            or generation["finish_reason"] != result["finish_reason"]
            or int(generation["token_count"])
            != int(result["generated_token_count"])
        ):
            raise ValueError("Q4 fresh-test generation/result row differs")
        diagnostics = "\n".join(
            str(value)
            for value in result.get("diagnostics", {}).values()
            if value
        )
        normalized = normalize_transport(raw_text)
        template = q4_failure_template(normalized)
        signals = q4_failure_signals(diagnostics, normalized)
        joined.append(
            {
                "task_id": key[0],
                "candidate_index": key[1],
                "raw_text": raw_text,
                "normalized_text_sha256": _text_sha256(normalized),
                "template": template,
                "template_sha256": _text_sha256(template),
                "proof_head": _proof_head(normalized),
                "generated_token_count": int(generation["token_count"]),
                "finish_reason": str(generation["finish_reason"]),
                "diagnostics_sha256": _text_sha256(diagnostics),
                "primary_diagnostic": _primary_diagnostic_line(diagnostics),
                "failure_signals": signals,
                "primary_failure_category": q4_primary_failure_category(signals),
            }
        )
    if sample_count > len(joined):
        raise ValueError("Q4 failure sample exceeds stored candidate count")

    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    template_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        exact_groups[row["normalized_text_sha256"]].append(row)
        template_groups[row["template_sha256"]].append(row)
    ranked_sample = sorted(
        joined,
        key=lambda row: hashlib.sha256(
            f"{Q4_CANARY_GATE_ID}\0{row['task_id']}\0{row['candidate_index']}".encode()
        ).digest(),
    )[:sample_count]
    primary_counts = Counter(row["primary_failure_category"] for row in ranked_sample)
    signal_counts = Counter(
        signal for row in ranked_sample for signal in row["failure_signals"]
    )
    token_counts = [int(row["generated_token_count"]) for row in joined]
    finish_counts = Counter(str(row["finish_reason"]) for row in joined)
    proof_head_counts = Counter(str(row["proof_head"]) for row in joined)

    def recurrence(group: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "candidate_count": len(group),
            "task_count": len({str(row["task_id"]) for row in group}),
        }

    sample_rows = []
    for row in ranked_sample:
        sample_rows.append(
            {
                key: row[key]
                for key in (
                    "task_id",
                    "candidate_index",
                    "normalized_text_sha256",
                    "template_sha256",
                    "proof_head",
                    "generated_token_count",
                    "finish_reason",
                    "diagnostics_sha256",
                    "primary_diagnostic",
                    "failure_signals",
                    "primary_failure_category",
                )
            }
            | {
                "exact_output_recurrence": recurrence(
                    exact_groups[row["normalized_text_sha256"]]
                ),
                "template_recurrence": recurrence(
                    template_groups[row["template_sha256"]]
                ),
            }
        )
    top_templates = sorted(
        template_groups.values(),
        key=lambda group: (
            -len({str(row["task_id"]) for row in group}),
            -len(group),
            str(group[0]["template_sha256"]),
        ),
    )[:20]
    top_exact = sorted(
        exact_groups.values(),
        key=lambda group: (
            -len({str(row["task_id"]) for row in group}),
            -len(group),
            str(group[0]["normalized_text_sha256"]),
        ),
    )[:20]
    top_template_summary = [
        {
            "template": group[0]["template"],
            "template_sha256": group[0]["template_sha256"],
            **recurrence(group),
        }
        for group in top_templates
    ]
    short_count = sum(value <= 64 for value in token_counts)
    constructor_count = proof_head_counts["exact-constructor"]
    value = {
        "schema_version": Q4_FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        "status": "complete",
        "workload_id": Q4_FINAL_WORKLOAD_ID,
        "checkpoint_id": "Q4",
        "adapter_model_sha256": Q4_ADAPTER_MODEL_SHA256,
        "method": {
            "benchmark_regenerated": False,
            "source": "already-generated first complete Q4 final-test candidates",
            "sample_count": sample_count,
            "sample_policy": (
                "lowest SHA-256 of gate-id, task-id, candidate-index over all "
                "stored failed candidates"
            ),
            "classification_precedence": [
                "premature_eos_or_incomplete_proof",
                "syntax_or_parser",
                "structurally_wrong_proof_or_template",
                "unknown_or_mismatched_lemma",
                "unsolved_goals",
                "type_or_elaboration",
                "other_lean_rejection",
            ],
        },
        "full_candidate_diagnostics": {
            "candidate_count": len(joined),
            "verified_candidate_count": 0,
            "finish_reason_counts": dict(sorted(finish_counts.items())),
            "generated_tokens": {
                "minimum": min(token_counts),
                "maximum": max(token_counts),
                "mean": fmean(token_counts),
                "median": median(token_counts),
                "at_most_64_count": short_count,
                "at_most_64_fraction": short_count / len(joined),
            },
            "proof_head_counts": dict(sorted(proof_head_counts.items())),
            "exact_constructor_fraction": constructor_count / len(joined),
            "unique_exact_output_count": len(exact_groups),
            "repeated_exact_output_candidate_count": sum(
                len(group) for group in exact_groups.values() if len(group) > 1
            ),
            "unique_template_count": len(template_groups),
            "top_cross_task_templates": top_template_summary,
            "top_exact_output_recurrences": [
                {
                    "normalized_text_sha256": group[0][
                        "normalized_text_sha256"
                    ],
                    **recurrence(group),
                }
                for group in top_exact
            ],
        },
        "sample": {
            "primary_category_counts": dict(sorted(primary_counts.items())),
            "overlapping_signal_counts": dict(sorted(signal_counts.items())),
            "rows": sample_rows,
        },
        "interpretation": {
            "all_candidates_ended_by_eos": finish_counts == Counter({"eos": 3320}),
            "token_limit_or_truncation_explains_failures": False,
            "short_exact_constructor_mode_dominates": (
                short_count / len(joined) > 0.5
                and constructor_count / len(joined) > 0.5
            ),
            "classification": (
                "narrow learned proof-template / reduced-exploration failure pattern; "
                "adapter integrity is decided separately by the Q4 parity canary"
            ),
        },
        "artifacts": {
            "final_assessment_sha256": sha256_file(final_evidence_path),
            "generations_sha256": generation_sha,
            "results_sha256": sha256_file(results_path),
            "generation_metadata_sha256": sha256_file(generation_metadata_path),
            "sample_membership_sha256": _json_sha256(
                [
                    [row["task_id"], row["candidate_index"]]
                    for row in ranked_sample
                ]
            ),
        },
    }
    _write_new_json(output, value)
    return value


def validate_q4_canary_gate(path: Path) -> dict[str, Any]:
    evidence = _read_json(path)
    requirements = evidence.get("requirements")
    if (
        evidence.get("schema_version") != Q4_CANARY_EVIDENCE_SCHEMA_VERSION
        or evidence.get("gate_id") != Q4_CANARY_GATE_ID
        or evidence.get("status") != "passed"
        or evidence.get("adapter_model_sha256") != Q4_ADAPTER_MODEL_SHA256
        or not isinstance(requirements, Mapping)
        or set(requirements) != Q4_CANARY_REQUIRED_GATES
        or not all(bool(value) for value in requirements.values())
    ):
        raise ValueError("Q4 final inference canary gate has not passed")
    return {
        "gate_id": Q4_CANARY_GATE_ID,
        "status": "passed",
        "adapter_model_sha256": Q4_ADAPTER_MODEL_SHA256,
        "evidence_sha256": sha256_file(path),
    }
