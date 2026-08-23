from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from statistics import fmean
from typing import Any

from .baseline import LoRAAdapterSpec, render_prompt, vllm_engine_kwargs
from .dataset_v2 import sha256_file
from .generalist_v2 import MODEL_ID, MODEL_REVISION, GeneralistV2Config
from .generalist_v2_dataset import _read_json, dataset_record_preamble
from .generalist_v2_evaluation import (
    _evaluation_phase1_config,
    _load_canonical_subset,
    materialize_q0_workload,
)
from .prompt import normalize_transport
from .schema import TaskRecord
from .verifier import LeanVerifier

PARITY_PROBE_SCHEMA_VERSION = "generalist-v2-lora-parity-probes-v1"
PARITY_RUNTIME_SCHEMA_VERSION = "generalist-v2-lora-parity-runtime-v1"
PARITY_RAW_VERIFICATION_SCHEMA_VERSION = (
    "generalist-v2-lora-parity-raw-verification-v1"
)
PARITY_EVIDENCE_SCHEMA_VERSION = "generalist-v2-lora-parity-evidence-v1"
PARITY_GATE_ID = "qwen35-vllm-lora-parity-v1"
Q2_CHECKPOINT_ID = "Q2"
Q2_PROBE_COUNTS = {
    "dataset-v2-train-probe": 8,
    "minif2f-valid-clean-v2": 4,
    "fresh-composition-valid-v2": 4,
}
PARITY_REQUIRED_GATES = {
    "prior_evaluator_invalidated",
    "static_overfit64_complete",
    "static_q2_complete",
    "hf_known_positive_reproduced",
    "vllm_overfit_adapter_effect",
    "q2_hf_forward_effect",
    "q2_vllm_inference_effect",
    "all_expected_outputs_present",
    "zero_verifier_infrastructure_errors",
}


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return _text_sha256(serialized)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite parity artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _adapter_identity(
    config: GeneralistV2Config, adapter_dir: Path, *, label: str
) -> dict[str, Any]:
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_model_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_model_path.is_file():
        raise ValueError(f"{label} adapter is incomplete: {adapter_dir}")
    adapter_config = _read_json(adapter_config_path)
    expected = {
        "base_model_name_or_path": MODEL_ID,
        "revision": MODEL_REVISION,
        "r": int(config.lora["r"]),
        "target_modules": str(config.lora["target_regex"]),
    }
    observed = {
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        "revision": adapter_config.get("revision"),
        "r": int(adapter_config.get("r", -1)),
        "target_modules": adapter_config.get("target_modules"),
    }
    if observed != expected:
        raise ValueError(f"{label} adapter identity differs: {observed}")
    return {
        "label": label,
        **expected,
        "adapter_config_sha256": sha256_file(adapter_config_path),
        "adapter_model_sha256": sha256_file(adapter_model_path),
    }


def _rank_fixed_tasks(workload_id: str, tasks: Sequence[TaskRecord], count: int):
    if count < 1 or len(tasks) < count:
        raise ValueError(f"invalid parity probe count for {workload_id}: {count}")
    return sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            f"{PARITY_GATE_ID}\0{workload_id}\0{task.id}".encode()
        ).digest(),
    )[:count]


def _probe_record(
    *,
    probe_id: str,
    group: str,
    workload_id: str,
    task: TaskRecord,
    verification_task: TaskRecord,
    target_completions: Sequence[str],
    proof_variant_id: str | None = None,
    expected_adapter_candidate_sha256: str | None = None,
) -> dict[str, Any]:
    prompt = render_prompt(task)
    targets = [normalize_transport(item) for item in target_completions]
    return {
        "probe_id": probe_id,
        "group": group,
        "workload_id": workload_id,
        "statement_id": task.id,
        "proof_variant_id": proof_variant_id,
        "task": task.to_dict(),
        "verification_task": verification_task.to_dict(),
        "prompt": prompt,
        "prompt_sha256": _text_sha256(prompt),
        "target_completions": targets,
        "target_completion_sha256s": [_text_sha256(item) for item in targets],
        "expected_adapter_candidate_sha256": expected_adapter_candidate_sha256,
    }


def build_lora_parity_probes(
    config: GeneralistV2Config,
    package_root: Path,
    view_dir: Path,
    overfit_run_path: Path,
    general_lean_project_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze known-positive overfit probes and an outcome-blind Q2 smoke."""

    config.validate()
    overfit_run = _read_json(overfit_run_path)
    reload_gate = overfit_run.get("reload_generation_evaluator_gate", {})
    prior_results = reload_gate.get("results", [])
    if (
        overfit_run.get("status") != "passed"
        or reload_gate.get("generation_backend") != "transformers-greedy"
        or int(reload_gate.get("generation_probe_count", 0)) != 4
        or int(reload_gate.get("exact_target_count", 0)) != 4
        or int(reload_gate.get("lean_verified_count", 0)) != 4
        or not isinstance(prior_results, list)
        or len(prior_results) != 4
    ):
        raise ValueError("overfit64 evidence is not the established 4/4 reference")

    overfit_statement_ids = [str(item["statement_id"]) for item in prior_results]
    overfit_records = _load_canonical_subset(package_root, overfit_statement_ids)
    probes: list[dict[str, Any]] = []
    for index, prior in enumerate(prior_results):
        statement_id = str(prior["statement_id"])
        proof_variant_id = str(prior["proof_variant_id"])
        record = overfit_records[statement_id]
        if record.provenance != "synthetic":
            raise ValueError("overfit64 parity references a non-synthetic probe")
        variants = [
            item
            for item in record.proof_variants
            if item.proof_variant_id == proof_variant_id
        ]
        if len(variants) != 1:
            raise ValueError("overfit64 parity proof variant does not resolve once")
        variant = variants[0]
        task = TaskRecord(
            id=statement_id,
            preamble=dataset_record_preamble(record),
            declaration=record.canonical_declaration,
            declaration_name=variant.source_declaration_name,
        )
        probes.append(
            _probe_record(
                probe_id=f"overfit64-{index:02d}",
                group="overfit64-known-positive",
                workload_id="generalist-v2-overfit64-v1",
                task=task,
                verification_task=task,
                target_completions=(variant.completion,),
                proof_variant_id=proof_variant_id,
                expected_adapter_candidate_sha256=str(prior["candidate_sha256"]),
            )
        )

    q2_selection: dict[str, list[str]] = {}
    for workload_id, count in Q2_PROBE_COUNTS.items():
        tasks, verification, targets, _ = materialize_q0_workload(
            workload_id,
            package_root,
            view_dir,
            lean_project_root=(
                general_lean_project_root
                if workload_id == "dataset-v2-train-probe"
                else None
            ),
        )
        selected = _rank_fixed_tasks(workload_id, tasks, count)
        q2_selection[workload_id] = [item.id for item in selected]
        if workload_id == "fresh-composition-valid-v2":
            fresh_records = _load_canonical_subset(
                package_root, [item.id for item in selected]
            )
        else:
            fresh_records = {}
        for index, task in enumerate(selected):
            target_completions = list(targets.get(task.id, ()))
            if workload_id == "fresh-composition-valid-v2":
                target_completions = [
                    item.completion
                    for item in fresh_records[task.id].proof_variants
                ]
            probes.append(
                _probe_record(
                    probe_id=f"q2-{workload_id}-{index:02d}",
                    group="q2-activity-smoke",
                    workload_id=workload_id,
                    task=task,
                    verification_task=verification[task.id],
                    target_completions=target_completions,
                )
            )

    if len(probes) != 20 or len({item["probe_id"] for item in probes}) != 20:
        raise RuntimeError("parity probe manifest must contain 20 unique probes")
    value = {
        "schema_version": PARITY_PROBE_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "deterministic_generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": int(config.evaluation["sampling"]["max_new_tokens"]),
            "eos_semantics": "tokenizer_eos_or_token_limit",
            "add_special_tokens": False,
        },
        "selection": {
            "overfit64": "the exact four prior reload probes in stored evidence order",
            "q2": (
                "outcome-blind SHA-256 order keyed by gate/workload/task; "
                "8 train-probe + 4 clean miniF2F + 4 fresh composition"
            ),
            "q2_task_ids": q2_selection,
        },
        "overfit_run_sha256": sha256_file(overfit_run_path),
        "probes": probes,
        "ordered_probe_ids_sha256": _json_sha256(
            [item["probe_id"] for item in probes]
        ),
    }
    _write_new_json(output, value)
    return value


def _load_probe_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    probes = value.get("probes")
    if (
        value.get("schema_version") != PARITY_PROBE_SCHEMA_VERSION
        or value.get("gate_id") != PARITY_GATE_ID
        or value.get("model_id") != MODEL_ID
        or value.get("model_revision") != MODEL_REVISION
        or not isinstance(probes, list)
        or len(probes) != 20
        or len({str(item.get("probe_id")) for item in probes}) != 20
    ):
        raise ValueError("invalid generalist-v2 parity probe manifest")
    for probe in probes:
        task = TaskRecord.from_dict(probe["task"])
        if (
            render_prompt(task) != probe["prompt"]
            or _text_sha256(probe["prompt"]) != probe["prompt_sha256"]
        ):
            raise ValueError("parity probe prompt binding differs")
    return value


def _arm_probes(manifest: Mapping[str, Any], group: str) -> list[dict[str, Any]]:
    return [item for item in manifest["probes"] if item["group"] == group]


def _candidate_record(
    probe: Mapping[str, Any], raw_text: str, token_count: int, finish_reason: str
) -> dict[str, Any]:
    normalized = normalize_transport(raw_text)
    targets = [str(item) for item in probe["target_completions"]]
    return {
        "probe_id": probe["probe_id"],
        "statement_id": probe["statement_id"],
        "workload_id": probe["workload_id"],
        "raw_text": raw_text,
        "raw_text_sha256": _text_sha256(raw_text),
        "normalized_text": normalized,
        "normalized_text_sha256": _text_sha256(normalized),
        "generated_tokens": token_count,
        "finish_reason": finish_reason,
        "exact_target": normalized in targets,
    }


def _hf_forward_and_generate(
    model: Any,
    tokenizer: Any,
    probe: Mapping[str, Any],
    *,
    adapter_name: str | None,
    max_new_tokens: int,
) -> tuple[dict[str, Any], Any]:
    import torch

    if adapter_name is not None:
        model.set_adapter(adapter_name)
    device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    )
    inputs = tokenizer(
        probe["prompt"], add_special_tokens=False, return_tensors="pt"
    )
    prompt_tokens = int(inputs["input_ids"].shape[1])
    inputs = {key: value.to(device) for key, value in inputs.items()}
    adapter_context = (
        model.disable_adapter() if adapter_name is None else nullcontext()
    )
    started = time.perf_counter()
    with (
        adapter_context,
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        forward = model(**inputs, use_cache=False, logits_to_keep=1)
        last_logits = forward.logits[0, -1].float().cpu()
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    generated_ids = generated[0, prompt_tokens:].detach().cpu()
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    eos_token_id = int(tokenizer.eos_token_id)
    finish_reason = (
        "eos"
        if generated_ids.numel() and int(generated_ids[-1]) == eos_token_id
        else "token_limit"
    )
    record = _candidate_record(
        probe, raw_text, int(generated_ids.numel()), finish_reason
    )
    log_probs = torch.log_softmax(last_logits, dim=-1)
    top_values, top_ids = torch.topk(log_probs, k=8)
    record.update(
        {
            "prompt_tokens": prompt_tokens,
            "generation_wall_time_seconds": elapsed,
            "last_token_logits_float32_sha256": hashlib.sha256(
                last_logits.numpy().tobytes()
            ).hexdigest(),
            "next_token_top8": [
                {"token_id": int(token_id), "logprob": float(logprob)}
                for token_id, logprob in zip(
                    top_ids.tolist(), top_values.tolist(), strict=True
                )
            ],
        }
    )
    del forward, generated, generated_ids, inputs, log_probs, top_values, top_ids
    torch.cuda.empty_cache()
    return record, last_logits


def run_hf_lora_parity(
    config: GeneralistV2Config,
    probe_manifest_path: Path,
    overfit_adapter_dir: Path,
    q2_adapter_dir: Path,
    model_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    """Run deterministic Base/PEFT arms in the exact QLoRA training runtime."""

    config.validate()
    manifest = _load_probe_manifest(probe_manifest_path)
    overfit_identity = _adapter_identity(
        config, overfit_adapter_dir, label="overfit64"
    )
    q2_identity = _adapter_identity(config, q2_adapter_dir, label="Q2")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("HF parity requires the training runtime") from error
    if not torch.cuda.is_available():
        raise RuntimeError("HF parity requires project-controlled local CUDA")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    tokenizer = AutoTokenizer.from_pretrained(
        overfit_adapter_dir, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("parity tokenizer lacks EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fallback = config.precision["fallback"]
    base = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=bool(fallback["load_in_4bit"]),
            bnb_4bit_quant_type=str(fallback["quantization_type"]),
            bnb_4bit_use_double_quant=bool(fallback["double_quantization"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model = PeftModel.from_pretrained(
        base,
        overfit_adapter_dir,
        adapter_name="overfit64",
        is_trainable=False,
    )
    model.load_adapter(q2_adapter_dir, adapter_name="q2", is_trainable=False)
    model.eval()
    model.config.use_cache = True
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("parity adapters unexpectedly remain trainable")

    max_new_tokens = int(config.evaluation["sampling"]["max_new_tokens"])
    arms = (
        ("hf_base_overfit", "overfit64-known-positive", None),
        ("hf_overfit64", "overfit64-known-positive", "overfit64"),
        ("hf_base_q2", "q2-activity-smoke", None),
        ("hf_q2", "q2-activity-smoke", "q2"),
    )
    results: dict[str, list[dict[str, Any]]] = {}
    logits: dict[tuple[str, str], Any] = {}
    for arm_id, group, adapter_name in arms:
        arm_results = []
        for probe in _arm_probes(manifest, group):
            result, last_logits = _hf_forward_and_generate(
                model,
                tokenizer,
                probe,
                adapter_name=adapter_name,
                max_new_tokens=max_new_tokens,
            )
            arm_results.append(result)
            logits[(arm_id, str(probe["probe_id"]))] = last_logits
        results[arm_id] = arm_results

    sensitivities = []
    for base_arm, adapter_arm, group in (
        ("hf_base_overfit", "hf_overfit64", "overfit64-known-positive"),
        ("hf_base_q2", "hf_q2", "q2-activity-smoke"),
    ):
        for probe in _arm_probes(manifest, group):
            probe_id = str(probe["probe_id"])
            delta = logits[(adapter_arm, probe_id)] - logits[(base_arm, probe_id)]
            sensitivities.append(
                {
                    "probe_id": probe_id,
                    "base_arm": base_arm,
                    "adapter_arm": adapter_arm,
                    "maximum_absolute_logit_delta": float(delta.abs().max()),
                    "mean_absolute_logit_delta": float(delta.abs().mean()),
                    "l2_logit_delta": float(torch.linalg.vector_norm(delta)),
                }
            )
    value = {
        "schema_version": PARITY_RUNTIME_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "runtime": "transformers-peft-nf4-qlora",
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapters": {
            "overfit64": overfit_identity,
            "Q2": q2_identity,
        },
        "deterministic_generation": manifest["deterministic_generation"],
        "arms": results,
        "forward_sensitivity": sensitivities,
        "execution": {
            "local_cuda": True,
            "cuda_device": properties.name,
            "cuda_device_total_memory_bytes": int(properties.total_memory),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device_index)
            ),
            "peak_cuda_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device_index)
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        },
    }
    _write_new_json(output, value)
    del model, base, tokenizer, logits
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _shape_tree(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_shape_tree(item) for item in value]
    return list(value.shape)


def inspect_vllm_lora_worker(
    worker: Any,
    adapter_id: int,
    source_adapter_dir: str,
    runtime_adapter_dir: str,
    expected_suffix_counts: Mapping[str, int],
    expected_target_regex: str,
) -> dict[str, Any]:
    """Run inside the actual vLLM worker after an adapter is loaded."""

    import safetensors
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.lora.utils import parse_fine_tuned_lora_name

    model_runner = worker.model_runner
    worker_manager = model_runner.lora_manager
    manager = worker_manager._adapter_manager
    model = manager.model
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is not None:
        mapper = mapper.get_unstacked_mapper()

    adapter_path = Path(source_adapter_dir)
    runtime_adapter_path = Path(runtime_adapter_dir)
    adapter_config = json.loads(
        (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    tensor_path = adapter_path / "adapter_model.safetensors"
    tensor_pairs: dict[str, set[str]] = {}
    tensor_shapes: dict[str, dict[str, list[int]]] = {}
    mapped_names: dict[str, str] = {}
    runtime_tensor_path = runtime_adapter_path / "adapter_model.safetensors"
    with safetensors.safe_open(tensor_path, framework="pt", device="cpu") as source_handle:
        tensor_keys = list(source_handle.keys())
        with safetensors.safe_open(
            runtime_tensor_path, framework="pt", device="cpu"
        ) as runtime_handle:
            runtime_tensor_keys = set(runtime_handle.keys())
            from .qwen35_vllm_lora import qwen35_vllm_runtime_tensor_key

            for tensor_key in tensor_keys:
                runtime_tensor_key = qwen35_vllm_runtime_tensor_key(tensor_key)
                if runtime_tensor_key not in runtime_tensor_keys:
                    raise RuntimeError("derived vLLM adapter omitted a source tensor")
                module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_key)
                mapped_module_name, mapped_is_lora_a = parse_fine_tuned_lora_name(
                    runtime_tensor_key, mapper
                )
                if mapped_is_lora_a != is_lora_a:
                    raise RuntimeError("vLLM mapper changed the LoRA A/B side")
                prior_mapped_name = mapped_names.setdefault(
                    module_name, mapped_module_name
                )
                if prior_mapped_name != mapped_module_name:
                    raise RuntimeError("vLLM mapper is inconsistent across A/B tensors")
                side = "A" if is_lora_a else "B"
                tensor_pairs.setdefault(module_name, set()).add(side)
                tensor_shapes.setdefault(module_name, {})[side] = list(
                    source_handle.get_slice(tensor_key).get_shape()
                )
            if len(runtime_tensor_keys) != len(tensor_keys):
                raise RuntimeError("derived vLLM adapter has unexpected tensors")

    child_to_parent = {
        child: parent
        for parent, children in manager.packed_modules.items()
        for child in children
    }
    errors: list[str] = []
    mappings = []
    suffix_counts = Counter()
    family_counts = Counter()
    expected_runtime_modules: set[str] = set()
    for module_name in sorted(tensor_pairs):
        sides = tensor_pairs[module_name]
        mapped_module_name = mapped_names[module_name]
        suffix = module_name.rsplit(".", 1)[-1]
        suffix_counts[suffix] += 1
        family = (
            "full_attention"
            if ".self_attn." in module_name
            else (
                "gated_deltanet"
                if ".linear_attn." in module_name
                else "mlp" if ".mlp." in module_name else "unknown"
            )
        )
        family_counts[family] += 1
        runtime_module = child_to_parent.get(mapped_module_name, mapped_module_name)
        expected_runtime_modules.add(runtime_module)
        wrapper = manager.modules.get(runtime_module)
        packed = runtime_module != mapped_module_name
        if sides != {"A", "B"}:
            errors.append(f"incomplete A/B pair: {module_name} {sorted(sides)}")
        if re.fullmatch(expected_target_regex, module_name) is None:
            errors.append(f"PEFT module misses trained target regex: {module_name}")
        if wrapper is None:
            errors.append(f"mapped module has no vLLM LoRA wrapper: {module_name}")
        elif not isinstance(wrapper, BaseLayerWithLoRA):
            errors.append(f"mapped module wrapper is not LoRA-capable: {runtime_module}")
        mappings.append(
            {
                "peft_module": module_name,
                "mapped_module": mapped_module_name,
                "suffix": suffix,
                "family": family,
                "runtime_module": runtime_module,
                "packed": packed,
                "wrapper_class": None if wrapper is None else type(wrapper).__name__,
                "tensor_shapes": tensor_shapes[module_name],
            }
        )

    expected_counts = {str(key): int(value) for key, value in expected_suffix_counts.items()}
    if dict(suffix_counts) != expected_counts:
        errors.append(
            f"trained suffix counts differ: {dict(suffix_counts)} != {expected_counts}"
        )
    if adapter_config.get("target_modules") != expected_target_regex:
        errors.append("adapter target regex differs")
    if len(tensor_keys) != 2 * sum(expected_counts.values()):
        errors.append("adapter tensor count differs")

    registered = manager._registered_adapters[adapter_id]
    loaded_runtime_modules = set(registered.loras)
    missing_loaded = sorted(expected_runtime_modules - loaded_runtime_modules)
    unexpected_loaded = sorted(loaded_runtime_modules - expected_runtime_modules)
    if missing_loaded:
        errors.append(f"runtime adapter omitted {len(missing_loaded)} mapped modules")
    if unexpected_loaded:
        errors.append(f"runtime adapter has {len(unexpected_loaded)} unexpected modules")
    if adapter_id not in manager.lora_index_to_id:
        errors.append("runtime adapter is not active in a GPU LoRA slot")

    loaded_inventory = []
    for module_name in sorted(loaded_runtime_modules):
        weights = registered.loras[module_name]
        loaded_inventory.append(
            {
                "runtime_module": module_name,
                "lora_a_shapes": _shape_tree(weights.lora_a),
                "lora_b_shapes": _shape_tree(weights.lora_b),
            }
        )
        if weights.lora_a is None or weights.lora_b is None:
            errors.append(f"runtime module has incomplete weights: {module_name}")

    multimodal_config = worker.vllm_config.model_config.multimodal_config
    configured_language_model_only = bool(
        multimodal_config is not None
        and getattr(multimodal_config, "language_model_only", False)
    )
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "model_class": type(model).__name__,
        "model_is_text_only": configured_language_model_only,
        "configured_language_model_only": configured_language_model_only,
        "adapter_id": adapter_id,
        "adapter_rank": int(registered.rank),
        "raw_tensor_count": len(tensor_keys),
        "peft_module_count": len(tensor_pairs),
        "runtime_module_count": len(loaded_runtime_modules),
        "target_suffix_counts": dict(sorted(suffix_counts.items())),
        "target_family_counts": dict(sorted(family_counts.items())),
        "packed_modules_mapping": {
            key: list(value)
            for key, value in sorted(manager.packed_modules_mapping.items())
        },
        "supported_lora_module_suffixes": sorted(manager.supported_lora_modules),
        "registered_wrapper_count": len(manager.modules),
        "active_lora_slots": list(manager.lora_index_to_id),
        "module_mappings": mappings,
        "loaded_runtime_inventory": loaded_inventory,
        "missing_loaded_runtime_modules": missing_loaded,
        "unexpected_loaded_runtime_modules": unexpected_loaded,
    }


def _vllm_source_binding() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("vllm")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = None if direct_url_text is None else json.loads(direct_url_text)
    url = "" if direct_url is None else str(direct_url.get("url", ""))
    revision_match = re.search(r"/([0-9a-f]{40})/vllm-", url)
    return {
        "version": distribution.version,
        "direct_url": url,
        "source_revision": None if revision_match is None else revision_match.group(1),
    }


def _vllm_generate_arm(
    llm: Any,
    sampling_params: Any,
    probes: Sequence[Mapping[str, Any]],
    *,
    lora_request: Any | None,
) -> list[dict[str, Any]]:
    outputs = llm.generate(
        [str(item["prompt"]) for item in probes],
        sampling_params,
        use_tqdm=True,
        **({} if lora_request is None else {"lora_request": lora_request}),
    )
    if len(outputs) != len(probes):
        raise RuntimeError("vLLM parity output count differs")
    results = []
    for probe, request_output in zip(probes, outputs, strict=True):
        if len(request_output.outputs) != 1:
            raise RuntimeError("vLLM parity requires one greedy output per probe")
        completion = request_output.outputs[0]
        record = _candidate_record(
            probe,
            str(completion.text),
            len(completion.token_ids),
            "eos" if completion.finish_reason == "stop" else str(completion.finish_reason),
        )
        first_token_id = (
            None if not completion.token_ids else int(completion.token_ids[0])
        )
        first_token_logprob = None
        if first_token_id is not None and completion.logprobs:
            first = completion.logprobs[0]
            selected = first.get(first_token_id)
            if selected is not None:
                first_token_logprob = float(selected.logprob)
        record.update(
            {
                "prompt_tokens": len(request_output.prompt_token_ids),
                "first_token_id": first_token_id,
                "first_token_logprob": first_token_logprob,
            }
        )
        results.append(record)
    return results


def run_vllm_lora_parity(
    config: GeneralistV2Config,
    base_evaluation_config: Path,
    probe_manifest_path: Path,
    overfit_adapter_dir: Path,
    q2_adapter_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Run static mapping audits plus Base/LoRA arms in the screening engine."""

    config.validate()
    manifest = _load_probe_manifest(probe_manifest_path)
    overfit_identity = _adapter_identity(
        config, overfit_adapter_dir, label="overfit64"
    )
    q2_identity = _adapter_identity(config, q2_adapter_dir, label="Q2")
    from .qwen35_vllm_lora import prepare_qwen35_vllm_adapter

    overfit_compatibility = prepare_qwen35_vllm_adapter(overfit_adapter_dir)
    q2_compatibility = prepare_qwen35_vllm_adapter(q2_adapter_dir)
    overfit_runtime_dir = Path(str(overfit_compatibility["runtime_adapter_dir"]))
    q2_runtime_dir = Path(str(q2_compatibility["runtime_adapter_dir"]))
    phase1 = _evaluation_phase1_config(base_evaluation_config, config)
    if phase1.engine.get(
        "use_flashinfer_sampler", phase1.engine.get("flashinfer_sampler")
    ) is False:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # The audit callable executes only inside the project-local worker and
    # returns JSON-sized control metadata; vLLM otherwise rejects callables.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:
        raise RuntimeError("vLLM parity requires the screening runtime") from error
    if vllm.__version__ != str(phase1.engine["version"]):
        raise RuntimeError("vLLM parity runtime differs from screening config")
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
        adapter_id="qwen-lean-generalist-v2-overfit64-parity",
        path=overfit_adapter_dir.resolve(),
        rank=int(config.lora["r"]),
        base_model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        runtime_path=overfit_runtime_dir,
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
        logprobs=1,
    )
    overfit_probes = _arm_probes(manifest, "overfit64-known-positive")
    q2_probes = _arm_probes(manifest, "q2-activity-smoke")
    overfit_request = LoRARequest(
        lora_name="qwen-lean-generalist-v2-overfit64-parity",
        lora_int_id=1,
        lora_path=str(overfit_runtime_dir.resolve()),
    )
    llm.llm_engine.add_lora(overfit_request)
    overfit_audit = llm.collective_rpc(
        inspect_vllm_lora_worker,
        args=(
            1,
            str(overfit_adapter_dir.resolve()),
            str(overfit_runtime_dir.resolve()),
            dict(config.lora["expected_module_counts"]),
            str(config.lora["target_regex"]),
        ),
    )

    q2_request = LoRARequest(
        lora_name="qwen-lean-generalist-v2-q2-parity",
        lora_int_id=2,
        lora_path=str(q2_runtime_dir.resolve()),
    )
    llm.llm_engine.add_lora(q2_request)
    q2_audit = llm.collective_rpc(
        inspect_vllm_lora_worker,
        args=(
            2,
            str(q2_adapter_dir.resolve()),
            str(q2_runtime_dir.resolve()),
            dict(config.lora["expected_module_counts"]),
            str(config.lora["target_regex"]),
        ),
    )
    failed_audits = [
        item for item in (*overfit_audit, *q2_audit) if item.get("status") != "passed"
    ]
    if failed_audits:
        diagnostic = {
            "schema_version": PARITY_RUNTIME_SCHEMA_VERSION,
            "gate_id": PARITY_GATE_ID,
            "status": "static-audit-failed-before-generation",
            "runtime": "vllm-lora-request-bfloat16",
            "probe_manifest_sha256": sha256_file(probe_manifest_path),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adapters": {"overfit64": overfit_identity, "Q2": q2_identity},
            "vllm_adapter_compatibility": {
                "overfit64": overfit_compatibility,
                "Q2": q2_compatibility,
            },
            "vllm": _vllm_source_binding(),
            "engine": phase1.engine,
            "static_mapping_audits": {
                "overfit64": overfit_audit,
                "Q2": q2_audit,
            },
            "arms": {},
        }
        _write_new_json(output, diagnostic)
        errors = [error for audit in failed_audits for error in audit.get("errors", [])]
        raise RuntimeError(
            "vLLM static LoRA mapping audit failed before generation: "
            + "; ".join(errors[:8])
        )

    arms = {
        "vllm_base_overfit": _vllm_generate_arm(
            llm, sampling_params, overfit_probes, lora_request=None
        ),
        "vllm_base_q2": _vllm_generate_arm(
            llm, sampling_params, q2_probes, lora_request=None
        ),
    }
    arms["vllm_overfit64"] = _vllm_generate_arm(
        llm, sampling_params, overfit_probes, lora_request=overfit_request
    )
    arms["vllm_q2"] = _vllm_generate_arm(
        llm, sampling_params, q2_probes, lora_request=q2_request
    )
    source_binding = _vllm_source_binding()
    value = {
        "schema_version": PARITY_RUNTIME_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "runtime": "vllm-lora-request-bfloat16",
        "probe_manifest_sha256": sha256_file(probe_manifest_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapters": {
            "overfit64": overfit_identity,
            "Q2": q2_identity,
        },
        "vllm_adapter_compatibility": {
            "overfit64": overfit_compatibility,
            "Q2": q2_compatibility,
        },
        "deterministic_generation": deterministic,
        "vllm": source_binding,
        "engine": phase1.engine,
        "static_mapping_audits": {
            "overfit64": overfit_audit,
            "Q2": q2_audit,
        },
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


def _results_by_probe(
    runtime: Mapping[str, Any], arm_id: str
) -> dict[str, Mapping[str, Any]]:
    results = runtime.get("arms", {}).get(arm_id)
    if not isinstance(results, list):
        raise TypeError(f"parity runtime lacks arm: {arm_id}")
    by_probe = {str(item["probe_id"]): item for item in results}
    if len(by_probe) != len(results):
        raise ValueError(f"parity arm has duplicate probes: {arm_id}")
    return by_probe


def _arm_pair_summary(
    base_results: Mapping[str, Mapping[str, Any]],
    adapter_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(base_results) != set(adapter_results) or not base_results:
        raise ValueError("parity arm probe memberships differ")
    ids = sorted(base_results)
    output_differences = sum(
        base_results[probe_id]["normalized_text_sha256"]
        != adapter_results[probe_id]["normalized_text_sha256"]
        for probe_id in ids
    )
    same_first_token_logprob_differences = []
    for probe_id in ids:
        base = base_results[probe_id]
        adapter = adapter_results[probe_id]
        if (
            base.get("first_token_id") is not None
            and base.get("first_token_id") == adapter.get("first_token_id")
            and base.get("first_token_logprob") is not None
            and adapter.get("first_token_logprob") is not None
        ):
            same_first_token_logprob_differences.append(
                abs(
                    float(base["first_token_logprob"])
                    - float(adapter["first_token_logprob"])
                )
            )
    return {
        "probe_count": len(ids),
        "output_difference_count": output_differences,
        "same_first_token_logprob_comparison_count": len(
            same_first_token_logprob_differences
        ),
        "maximum_same_first_token_logprob_delta": max(
            same_first_token_logprob_differences, default=0.0
        ),
    }


def _audit_summary(audits: Any) -> dict[str, Any]:
    if not isinstance(audits, list) or len(audits) != 1:
        raise ValueError("single-GPU vLLM static audit count differs")
    audit = audits[0]
    return {
        "status": audit.get("status"),
        "errors": list(audit.get("errors", [])),
        "model_class": audit.get("model_class"),
        "model_is_text_only": audit.get("model_is_text_only"),
        "raw_tensor_count": audit.get("raw_tensor_count"),
        "peft_module_count": audit.get("peft_module_count"),
        "runtime_module_count": audit.get("runtime_module_count"),
        "target_suffix_counts": audit.get("target_suffix_counts"),
        "target_family_counts": audit.get("target_family_counts"),
        "registered_wrapper_count": audit.get("registered_wrapper_count"),
        "missing_loaded_runtime_modules": audit.get(
            "missing_loaded_runtime_modules"
        ),
        "unexpected_loaded_runtime_modules": audit.get(
            "unexpected_loaded_runtime_modules"
        ),
        "full_mapping_inventory_sha256": _json_sha256(audit.get("module_mappings")),
        "loaded_runtime_inventory_sha256": _json_sha256(
            audit.get("loaded_runtime_inventory")
        ),
    }


def _verify_output(
    verifier: LeanVerifier,
    probe: Mapping[str, Any],
    arm_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    outcome = verifier.verify(
        TaskRecord.from_dict(probe["verification_task"]),
        str(result["normalized_text"]),
    )
    return {
        "arm_id": arm_id,
        "probe_id": probe["probe_id"],
        "statement_id": probe["statement_id"],
        "workload_id": probe["workload_id"],
        "raw_text": result["raw_text"],
        "raw_text_sha256": result["raw_text_sha256"],
        "normalized_text": result["normalized_text"],
        "normalized_text_sha256": result["normalized_text_sha256"],
        "exact_target": bool(result["exact_target"]),
        "lean_category": outcome.category,
        "lean_exit_code": outcome.lean_exit_code,
        "lean_diagnostics": outcome.diagnostics,
        "verification_wall_time_seconds": outcome.latency_seconds,
    }


def _verification_summary(
    rows: Sequence[Mapping[str, Any]], arm_id: str
) -> dict[str, Any]:
    selected = [item for item in rows if item["arm_id"] == arm_id]
    categories = Counter(str(item["lean_category"]) for item in selected)
    return {
        "candidate_count": len(selected),
        "exact_target_count": sum(bool(item["exact_target"]) for item in selected),
        "lean_verified_count": categories["verified"],
        "lean_category_counts": dict(sorted(categories.items())),
        "infrastructure_error_count": categories["verifier_error"],
    }


def _forward_sensitivity_summary(
    hf_runtime: Mapping[str, Any], adapter_arm: str
) -> dict[str, Any]:
    rows = [
        item
        for item in hf_runtime.get("forward_sensitivity", [])
        if item.get("adapter_arm") == adapter_arm
    ]
    if not rows:
        raise ValueError(f"HF parity lacks forward sensitivity: {adapter_arm}")
    maxima = [float(item["maximum_absolute_logit_delta"]) for item in rows]
    return {
        "probe_count": len(rows),
        "changed_probe_count": sum(value > 0.0 for value in maxima),
        "maximum_absolute_logit_delta": max(maxima),
        "mean_of_maximum_absolute_logit_deltas": fmean(maxima),
    }


def compact_lora_parity_evidence(
    config: GeneralistV2Config,
    probe_manifest_path: Path,
    hf_runtime_path: Path,
    vllm_runtime_path: Path,
    prior_vllm_diagnostic_path: Path,
    general_lean_project_root: Path,
    minif2f_project_root: Path,
    raw_verification_output: Path,
    output: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Lean-check every arm and classify the blocking evaluator-validity gate."""

    config.validate()
    manifest = _load_probe_manifest(probe_manifest_path)
    hf_runtime = _read_json(hf_runtime_path)
    vllm_runtime = _read_json(vllm_runtime_path)
    prior_vllm_diagnostic = _read_json(prior_vllm_diagnostic_path)
    manifest_sha = sha256_file(probe_manifest_path)
    for runtime, name in ((hf_runtime, "HF"), (vllm_runtime, "vLLM")):
        if (
            runtime.get("schema_version") != PARITY_RUNTIME_SCHEMA_VERSION
            or runtime.get("gate_id") != PARITY_GATE_ID
            or runtime.get("probe_manifest_sha256") != manifest_sha
            or runtime.get("model_id") != MODEL_ID
            or runtime.get("model_revision") != MODEL_REVISION
        ):
            raise ValueError(f"{name} parity runtime binding differs")
    if hf_runtime.get("adapters") != vllm_runtime.get("adapters"):
        raise ValueError("HF and vLLM parity adapter bindings differ")
    prior_audits = prior_vllm_diagnostic.get("static_mapping_audits", {})
    prior_evaluator_invalidated = (
        prior_vllm_diagnostic.get("status")
        == "static-audit-failed-before-generation"
        and prior_vllm_diagnostic.get("probe_manifest_sha256") == manifest_sha
        and prior_vllm_diagnostic.get("model_id") == MODEL_ID
        and prior_vllm_diagnostic.get("model_revision") == MODEL_REVISION
        and prior_vllm_diagnostic.get("vllm") == vllm_runtime.get("vllm")
        and prior_vllm_diagnostic.get("arms") == {}
        and set(prior_audits) == {"overfit64", "Q2"}
        and all(
            len(audits) == 1
            and audits[0].get("status") == "failed"
            and len(audits[0].get("errors", [])) == 248
            for audits in prior_audits.values()
        )
    )
    if not prior_evaluator_invalidated:
        raise ValueError("prior no-op vLLM evaluator diagnostic binding differs")

    arm_ids = (
        "hf_base_overfit",
        "hf_overfit64",
        "hf_base_q2",
        "hf_q2",
        "vllm_base_overfit",
        "vllm_overfit64",
        "vllm_base_q2",
        "vllm_q2",
    )
    runtime_for_arm = {
        arm_id: hf_runtime if arm_id.startswith("hf_") else vllm_runtime
        for arm_id in arm_ids
    }
    probes = {str(item["probe_id"]): item for item in manifest["probes"]}
    general_verifier = LeanVerifier(
        general_lean_project_root,
        timeout_seconds=float(config.evaluation["verifier_timeout_seconds"]),
    )
    minif2f_verifier = LeanVerifier(
        minif2f_project_root,
        timeout_seconds=float(config.evaluation["verifier_timeout_seconds"]),
    )
    jobs = []
    for arm_id in arm_ids:
        for probe_id, result in _results_by_probe(
            runtime_for_arm[arm_id], arm_id
        ).items():
            probe = probes[probe_id]
            verifier = (
                minif2f_verifier
                if probe["workload_id"] == "minif2f-valid-clean-v2"
                else general_verifier
            )
            jobs.append((verifier, probe, arm_id, result))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        verification_rows = list(
            executor.map(lambda args: _verify_output(*args), jobs)
        )
    raw_verification = {
        "schema_version": PARITY_RAW_VERIFICATION_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
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
    hf_overfit_base = _results_by_probe(hf_runtime, "hf_base_overfit")
    hf_overfit_adapter = _results_by_probe(hf_runtime, "hf_overfit64")
    hf_q2_base = _results_by_probe(hf_runtime, "hf_base_q2")
    hf_q2_adapter = _results_by_probe(hf_runtime, "hf_q2")
    vllm_overfit_base = _results_by_probe(vllm_runtime, "vllm_base_overfit")
    vllm_overfit_adapter = _results_by_probe(vllm_runtime, "vllm_overfit64")
    vllm_q2_base = _results_by_probe(vllm_runtime, "vllm_base_q2")
    vllm_q2_adapter = _results_by_probe(vllm_runtime, "vllm_q2")
    static_audits = {
        label: _audit_summary(vllm_runtime["static_mapping_audits"][label])
        for label in ("overfit64", "Q2")
    }
    hf_overfit_pair = _arm_pair_summary(hf_overfit_base, hf_overfit_adapter)
    hf_q2_pair = _arm_pair_summary(hf_q2_base, hf_q2_adapter)
    vllm_overfit_pair = _arm_pair_summary(
        vllm_overfit_base, vllm_overfit_adapter
    )
    vllm_q2_pair = _arm_pair_summary(vllm_q2_base, vllm_q2_adapter)
    hf_overfit_sensitivity = _forward_sensitivity_summary(
        hf_runtime, "hf_overfit64"
    )
    hf_q2_sensitivity = _forward_sensitivity_summary(hf_runtime, "hf_q2")

    known_positive_hash_match_count = sum(
        hf_overfit_adapter[str(probe["probe_id"])]["normalized_text_sha256"]
        == probe["expected_adapter_candidate_sha256"]
        for probe in _arm_probes(manifest, "overfit64-known-positive")
    )
    vllm_hf_adapter_output_match_count = sum(
        vllm_overfit_adapter[probe_id]["normalized_text_sha256"]
        == hf_overfit_adapter[probe_id]["normalized_text_sha256"]
        for probe_id in hf_overfit_adapter
    )
    no_generation_errors = all(
        summaries[arm_id]["candidate_count"]
        == (4 if "overfit" in arm_id else 16)
        for arm_id in arm_ids
    )
    no_verifier_errors = all(
        summaries[arm_id]["infrastructure_error_count"] == 0
        for arm_id in arm_ids
    )
    requirements = {
        "prior_evaluator_invalidated": prior_evaluator_invalidated,
        "static_overfit64_complete": static_audits["overfit64"]["status"]
        == "passed",
        "static_q2_complete": static_audits["Q2"]["status"] == "passed",
        "hf_known_positive_reproduced": (
            known_positive_hash_match_count == 4
            and summaries["hf_overfit64"]["exact_target_count"] == 4
            and summaries["hf_overfit64"]["lean_verified_count"] == 4
            and hf_overfit_pair["output_difference_count"] > 0
            and hf_overfit_sensitivity["changed_probe_count"] > 0
        ),
        "vllm_overfit_adapter_effect": (
            summaries["vllm_overfit64"]["exact_target_count"] > 0
            and summaries["vllm_overfit64"]["lean_verified_count"] > 0
            and vllm_overfit_pair["output_difference_count"] > 0
            and vllm_hf_adapter_output_match_count > 0
        ),
        "q2_hf_forward_effect": hf_q2_sensitivity["changed_probe_count"] > 0,
        "q2_vllm_inference_effect": (
            vllm_q2_pair["output_difference_count"] > 0
            or vllm_q2_pair["maximum_same_first_token_logprob_delta"] > 1e-6
        ),
        "all_expected_outputs_present": no_generation_errors,
        "zero_verifier_infrastructure_errors": no_verifier_errors,
    }
    passed = all(requirements.values())
    compact_rows = [
        {
            key: row[key]
            for key in (
                "arm_id",
                "probe_id",
                "statement_id",
                "workload_id",
                "raw_text_sha256",
                "normalized_text_sha256",
                "exact_target",
                "lean_category",
                "lean_exit_code",
                "verification_wall_time_seconds",
            )
        }
        for row in verification_rows
    ]
    value = {
        "schema_version": PARITY_EVIDENCE_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "status": "passed" if passed else "failed",
        "classification": (
            "PASS: vLLM Qwen3.5 LoRA inference is valid"
            if passed
            else "FAIL: Qwen3.5 LoRA evaluator validity is not established"
        ),
        "model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION},
        "vllm": vllm_runtime["vllm"],
        "adapters": hf_runtime["adapters"],
        "vllm_adapter_compatibility": {
            label: {
                key: binding[key]
                for key in (
                    "schema_version",
                    "mapping_reason",
                    "source_tensor_prefix",
                    "runtime_tensor_prefix",
                    "source_adapter_model_sha256",
                    "runtime_adapter_model_sha256",
                    "source_tensor_payload_sha256",
                    "runtime_tensor_payload_sha256",
                    "tensor_count",
                )
            }
            for label, binding in vllm_runtime[
                "vllm_adapter_compatibility"
            ].items()
        },
        "target_regex": str(config.lora["target_regex"]),
        "requirements": requirements,
        "static_mapping_audits": static_audits,
        "functional_parity": {
            "hf_overfit64": {
                "pair": hf_overfit_pair,
                "forward_sensitivity": hf_overfit_sensitivity,
                "known_positive_hash_match_count": known_positive_hash_match_count,
            },
            "vllm_overfit64": {
                "pair": vllm_overfit_pair,
                "hf_adapter_output_match_count": vllm_hf_adapter_output_match_count,
            },
            "hf_q2": {
                "pair": hf_q2_pair,
                "forward_sensitivity": hf_q2_sensitivity,
            },
            "vllm_q2": {"pair": vllm_q2_pair},
        },
        "arm_summaries": summaries,
        "per_probe_outcomes": compact_rows,
        "artifacts": {
            "probe_manifest_sha256": manifest_sha,
            "hf_runtime_sha256": sha256_file(hf_runtime_path),
            "vllm_runtime_sha256": sha256_file(vllm_runtime_path),
            "prior_vllm_diagnostic_sha256": sha256_file(
                prior_vllm_diagnostic_path
            ),
            "raw_verification_sha256": sha256_file(raw_verification_output),
        },
        "invalidated_prior_screening": True,
        "prior_screening_classification": "INVALID FOR MODEL QUALITY",
        "q0_to_q2_flat_result_interpretation": "invalid-diagnostic-only",
        "required_followup": (
            "rerun Q1-Q4 screening from scratch with the corrected evaluator"
        ),
    }
    _write_new_json(output, value)
    return value


def validate_lora_parity_gate(
    config: GeneralistV2Config, parity_evidence_path: Path
) -> dict[str, Any]:
    config.validate()
    evidence = _read_json(parity_evidence_path)
    vllm = evidence.get("vllm", {})
    requirements = evidence.get("requirements", {})
    if (
        evidence.get("schema_version") != PARITY_EVIDENCE_SCHEMA_VERSION
        or evidence.get("gate_id") != PARITY_GATE_ID
        or evidence.get("status") != "passed"
        or evidence.get("model")
        != {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
        or evidence.get("target_regex") != str(config.lora["target_regex"])
        or vllm.get("version") != "0.27.2rc1.dev203+g41f179b57"
        or vllm.get("source_revision")
        != "41f179b57aa8ab6f634f508128ce1f1efadd0eb1"
        or not isinstance(requirements, Mapping)
        or set(requirements) != PARITY_REQUIRED_GATES
        or not all(bool(value) for value in requirements.values())
    ):
        raise ValueError("generalist-v2 LoRA inference parity gate has not passed")
    return {
        "gate_id": PARITY_GATE_ID,
        "status": "passed",
        "evidence_sha256": sha256_file(parity_evidence_path),
        "vllm_version": vllm["version"],
        "vllm_source_revision": vllm["source_revision"],
        "q2_adapter_model_sha256": evidence["adapters"]["Q2"][
            "adapter_model_sha256"
        ],
    }
