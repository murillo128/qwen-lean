from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import re
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_v2 import sha256_file
from .generalist_v3 import DatasetBinding, GeneralistV3Config, _read_json, _write_json
from .generalist_v3_training import _resolve_anchor_inputs
from .prompt import normalize_transport
from .qwen35_vllm_lora import (
    patch_qwen35_vllm_gdn_lora_mapping,
    prepare_qwen35_vllm_adapter,
    qwen35_vllm_runtime_tensor_items,
)


PARITY_GATE_ID = "qwen35-vllm-lora-parity-v2"
PARITY_HF_SCHEMA_VERSION = "generalist-v3-lora-parity-hf-v1"
PARITY_VLLM_SCHEMA_VERSION = "generalist-v3-lora-parity-vllm-v1"
PARITY_EVIDENCE_SCHEMA_VERSION = "generalist-v3-lora-parity-evidence-v1"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _adapter_identity(config: GeneralistV3Config, adapter_dir: Path) -> dict[str, Any]:
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_model_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_model_path.is_file():
        raise ValueError(f"generalist-v3 parity adapter is incomplete: {adapter_dir}")
    adapter_config = _read_json(adapter_config_path)
    observed = {
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        "revision": adapter_config.get("revision"),
        "r": int(adapter_config.get("r", -1)),
        "lora_alpha": int(adapter_config.get("lora_alpha", -1)),
        "target_modules": adapter_config.get("target_modules"),
    }
    expected = {
        "base_model_name_or_path": config.model["model_id"],
        "revision": config.model["model_revision"],
        "r": int(config.lora["r"]),
        "lora_alpha": int(config.lora["lora_alpha"]),
        "target_modules": config.lora["target_regex"],
    }
    if observed != expected:
        raise ValueError(f"generalist-v3 parity adapter identity differs: {observed}")
    return {
        **observed,
        "adapter_config_sha256": sha256_file(adapter_config_path),
        "adapter_model_sha256": sha256_file(adapter_model_path),
    }


def _probe_manifest(
    binding: DatasetBinding, anchor_manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor_manifest = _read_json(anchor_manifest_path)
    inputs = _resolve_anchor_inputs(binding, anchor_manifest)
    ranked = sorted(
        zip(anchor_manifest["anchors"], inputs, strict=True),
        key=lambda item: (
            int(item[0]["input_tokens"]),
            str(item[0]["example_id"]),
        ),
    )[:4]
    probes = [
        {
            "probe_id": f"anchor-{index}",
            "example_id": anchor["example_id"],
            "statement_id": anchor["statement_id"],
            "task_kind": anchor["task_kind"],
            "input_tokens": anchor["input_tokens"],
            "prompt": prompt,
            "prompt_sha256": _text_sha256(prompt),
        }
        for index, (anchor, prompt) in enumerate(ranked)
    ]
    return probes, anchor_manifest


def _hf_forward(model: Any, tokenizer: Any, prompt: str, *, adapter: bool) -> Any:
    import torch

    device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    )
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    context = nullcontext() if adapter else model.disable_adapter()
    with context, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    logits = output.logits[0, -1].float().cpu()
    del input_ids, output
    return logits


def build_and_run_hf_parity_sentinel(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    anchor_manifest_path: Path,
    adapter_dir: Path,
    output_path: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic non-quality sentinel and prove PEFT forward effect."""

    import torch

    from .generalist_v2_training import load_training_runtime, lora_target_summary

    if adapter_dir.exists() or output_path.exists():
        raise FileExistsError("generalist-v3 parity requires fresh HF artifacts")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    runtime = load_training_runtime(config, model_snapshot=model_snapshot)  # type: ignore[arg-type]
    changed_tensors = []
    with torch.no_grad():
        for name, parameter in runtime.model.named_parameters():
            if ".lora_B.default.weight" not in name:
                continue
            flat = parameter.view(-1)
            indices = torch.arange(flat.numel(), device=flat.device)
            flat.copy_(
                torch.where(
                    indices.remainder(2).eq(0),
                    torch.full_like(flat, 0.002),
                    torch.full_like(flat, -0.002),
                )
            )
            changed_tensors.append(name)
    if len(changed_tensors) != 248:
        raise RuntimeError("generalist-v3 parity sentinel changed module count differs")
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    runtime.model.save_pretrained(adapter_dir, safe_serialization=True)
    runtime.tokenizer.save_pretrained(adapter_dir)
    probes, anchor_manifest = _probe_manifest(binding, anchor_manifest_path)
    runtime.model.eval()
    rows = []
    for probe in probes:
        base_logits = _hf_forward(
            runtime.model, runtime.tokenizer, str(probe["prompt"]), adapter=False
        )
        adapter_logits = _hf_forward(
            runtime.model, runtime.tokenizer, str(probe["prompt"]), adapter=True
        )
        delta = adapter_logits - base_logits
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "prompt_sha256": probe["prompt_sha256"],
                "maximum_absolute_logit_delta": float(delta.abs().max()),
                "mean_absolute_logit_delta": float(delta.abs().mean()),
                "l2_logit_delta": float(torch.linalg.vector_norm(delta)),
                "base_argmax_token_id": int(base_logits.argmax()),
                "adapter_argmax_token_id": int(adapter_logits.argmax()),
            }
        )
    identity = _adapter_identity(config, adapter_dir)
    value = {
        "schema_version": PARITY_HF_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "status": "passed",
        "purpose": "deterministic non-quality LoRA transport sentinel",
        "model": dict(config.model),
        "adapter": identity,
        "anchor_manifest_sha256": sha256_file(anchor_manifest_path),
        "anchors_sha256": anchor_manifest["anchors_sha256"],
        "probes": probes,
        "forward_sensitivity": rows,
        "all_probes_changed": all(
            float(item["maximum_absolute_logit_delta"]) > 0.0 for item in rows
        ),
        "sentinel_initialization": {
            "selection": "every LoRA B tensor",
            "values": "alternating +0.002/-0.002 in flat tensor order",
            "changed_tensor_count": len(changed_tensors),
            "changed_tensor_names_sha256": hashlib.sha256(
                "\n".join(sorted(changed_tensors)).encode("utf-8")
            ).hexdigest(),
            "optimizer_created": False,
            "optimizer_updates": 0,
        },
        "lora": lora_target_summary(runtime.target_matches),
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    if not value["all_probes_changed"]:
        raise RuntimeError("generalist-v3 HF parity sentinel has no forward effect")
    _write_json(output_path, value)
    del runtime, rows
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _vllm_arm(
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
        raise RuntimeError("generalist-v3 vLLM parity output count differs")
    rows = []
    for probe, request in zip(probes, outputs, strict=True):
        if len(request.outputs) != 1:
            raise RuntimeError("generalist-v3 vLLM parity requires one output")
        output = request.outputs[0]
        first_token_id = None if not output.token_ids else int(output.token_ids[0])
        first_token_logprob = None
        if first_token_id is not None and output.logprobs:
            selected = output.logprobs[0].get(first_token_id)
            if selected is not None:
                first_token_logprob = float(selected.logprob)
        normalized = normalize_transport(str(output.text))
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "prompt_sha256": probe["prompt_sha256"],
                "candidate_text": str(output.text),
                "normalized_text_sha256": _text_sha256(normalized),
                "generated_tokens": len(output.token_ids),
                "finish_reason": (
                    "eos" if output.finish_reason == "stop" else output.finish_reason
                ),
                "first_token_id": first_token_id,
                "first_token_logprob": first_token_logprob,
            }
        )
    return rows


def _shape_tree(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_shape_tree(item) for item in value]
    return list(value.shape)


def inspect_vllm_v3_lora_worker(
    worker: Any,
    adapter_id: int,
    source_adapter_dir: str,
    runtime_adapter_dir: str,
    expected_suffix_counts: Mapping[str, int],
    expected_target_regex: str,
) -> dict[str, Any]:
    """Verify the source-to-derived-to-loaded mapping inside the vLLM worker."""

    import safetensors
    import torch
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.lora.utils import parse_fine_tuned_lora_name

    model_runner = worker.model_runner
    worker_manager = model_runner.lora_manager
    manager = worker_manager._adapter_manager
    model = manager.model
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is not None and hasattr(mapper, "get_unstacked_mapper"):
        mapper = mapper.get_unstacked_mapper()

    source_path = Path(source_adapter_dir)
    runtime_path = Path(runtime_adapter_dir)
    adapter_config = json.loads(
        (source_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []
    transform_errors: list[str] = []
    source_pairs: dict[str, set[str]] = {}
    source_shapes: dict[str, dict[str, list[int]]] = {}
    source_to_runtime: dict[str, set[str]] = {}
    runtime_pairs: dict[str, set[str]] = {}
    runtime_shapes: dict[str, dict[str, list[int]]] = {}
    source_tensor_path = source_path / "adapter_model.safetensors"
    runtime_tensor_path = runtime_path / "adapter_model.safetensors"
    with safetensors.safe_open(
        source_tensor_path, framework="pt", device="cpu"
    ) as source_handle:
        source_tensor_keys = list(source_handle.keys())
        with safetensors.safe_open(
            runtime_tensor_path, framework="pt", device="cpu"
        ) as runtime_handle:
            runtime_tensor_keys = set(runtime_handle.keys())
            expected_runtime_tensor_keys: set[str] = set()
            for source_key in source_tensor_keys:
                source_module, source_is_lora_a = parse_fine_tuned_lora_name(
                    source_key
                )
                source_side = "A" if source_is_lora_a else "B"
                source_pairs.setdefault(source_module, set()).add(source_side)
                source_shapes.setdefault(source_module, {})[source_side] = list(
                    source_handle.get_slice(source_key).get_shape()
                )
                expected_items = qwen35_vllm_runtime_tensor_items(
                    source_key,
                    source_handle.get_tensor(source_key),
                    split_gdn_qkv=True,
                )
                for runtime_key, expected_tensor in expected_items:
                    expected_runtime_tensor_keys.add(runtime_key)
                    runtime_module, runtime_is_lora_a = (
                        parse_fine_tuned_lora_name(runtime_key, mapper)
                    )
                    if runtime_is_lora_a != source_is_lora_a:
                        transform_errors.append(
                            f"LoRA A/B side changed: {source_key} -> {runtime_key}"
                        )
                    source_to_runtime.setdefault(source_module, set()).add(
                        runtime_module
                    )
                    runtime_side = "A" if runtime_is_lora_a else "B"
                    runtime_pairs.setdefault(runtime_module, set()).add(runtime_side)
                    runtime_shapes.setdefault(runtime_module, {})[runtime_side] = list(
                        expected_tensor.shape
                    )
                    if runtime_key not in runtime_tensor_keys:
                        transform_errors.append(
                            f"derived adapter omitted tensor: {runtime_key}"
                        )
                    elif not torch.equal(
                        expected_tensor, runtime_handle.get_tensor(runtime_key)
                    ):
                        transform_errors.append(
                            f"derived adapter changed tensor: {runtime_key}"
                        )
            unexpected_runtime_tensors = sorted(
                runtime_tensor_keys - expected_runtime_tensor_keys
            )
            if unexpected_runtime_tensors:
                transform_errors.append(
                    "derived adapter has unexpected tensors: "
                    + ", ".join(unexpected_runtime_tensors[:8])
                )

    suffix_counts = Counter()
    family_counts = Counter()
    for source_module, sides in sorted(source_pairs.items()):
        suffix = source_module.rsplit(".", 1)[-1]
        suffix_counts[suffix] += 1
        family_counts[
            "full_attention"
            if ".self_attn." in source_module
            else (
                "gated_deltanet"
                if ".linear_attn." in source_module
                else "mlp" if ".mlp." in source_module else "unknown"
            )
        ] += 1
        if sides != {"A", "B"}:
            errors.append(f"incomplete source A/B pair: {source_module}")
        if re.fullmatch(expected_target_regex, source_module) is None:
            errors.append(f"source module misses target regex: {source_module}")
    for runtime_module, sides in sorted(runtime_pairs.items()):
        if sides != {"A", "B"}:
            errors.append(f"incomplete runtime A/B pair: {runtime_module}")

    expected_counts = {
        str(key): int(value) for key, value in expected_suffix_counts.items()
    }
    if dict(suffix_counts) != expected_counts:
        errors.append(
            f"source suffix counts differ: {dict(suffix_counts)} != {expected_counts}"
        )
    if adapter_config.get("target_modules") != expected_target_regex:
        errors.append("adapter target regex differs")
    if len(source_tensor_keys) != 2 * sum(expected_counts.values()):
        errors.append("source adapter tensor count differs")
    errors.extend(transform_errors)

    child_to_parent = {
        child: parent
        for parent, children in manager.packed_modules.items()
        for child in children
    }
    expected_loaded_modules = {
        child_to_parent.get(runtime_module, runtime_module)
        for runtime_module in runtime_pairs
    }
    source_mappings = []
    for source_module in sorted(source_pairs):
        runtime_modules = sorted(source_to_runtime.get(source_module, set()))
        loaded_modules = sorted(
            {child_to_parent.get(item, item) for item in runtime_modules}
        )
        if not runtime_modules or not loaded_modules:
            errors.append(f"source module has no runtime mapping: {source_module}")
        source_mappings.append(
            {
                "source_module": source_module,
                "runtime_adapter_modules": runtime_modules,
                "loaded_runtime_modules": loaded_modules,
                "source_tensor_shapes": source_shapes[source_module],
            }
        )

    registered = manager._registered_adapters[adapter_id]
    loaded_runtime_modules = set(registered.loras)
    missing_loaded = sorted(expected_loaded_modules - loaded_runtime_modules)
    unexpected_loaded = sorted(loaded_runtime_modules - expected_loaded_modules)
    if missing_loaded:
        errors.append(f"runtime omitted {len(missing_loaded)} mapped modules")
    if unexpected_loaded:
        errors.append(f"runtime has {len(unexpected_loaded)} unexpected modules")
    if adapter_id not in manager.lora_index_to_id:
        errors.append("runtime adapter is not active in a GPU LoRA slot")

    loaded_inventory = []
    for module_name in sorted(loaded_runtime_modules):
        weights = registered.loras[module_name]
        wrapper = manager.modules.get(module_name)
        if wrapper is None or not isinstance(wrapper, BaseLayerWithLoRA):
            errors.append(f"loaded module has no LoRA wrapper: {module_name}")
        if weights.lora_a is None or weights.lora_b is None:
            errors.append(f"loaded module has incomplete weights: {module_name}")
        loaded_inventory.append(
            {
                "runtime_module": module_name,
                "wrapper_class": None if wrapper is None else type(wrapper).__name__,
                "lora_a_shapes": _shape_tree(weights.lora_a),
                "lora_b_shapes": _shape_tree(weights.lora_b),
            }
        )

    multimodal_config = worker.vllm_config.model_config.multimodal_config
    configured_language_model_only = bool(
        multimodal_config is not None
        and getattr(multimodal_config, "language_model_only", False)
    )
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "transform_errors": transform_errors,
        "semantic_transform_verified": not transform_errors,
        "source_modules_all_mapped": all(
            bool(source_to_runtime.get(item)) for item in source_pairs
        ),
        "model_class": type(model).__name__,
        "configured_language_model_only": configured_language_model_only,
        "adapter_id": adapter_id,
        "adapter_rank": int(registered.rank),
        "source_tensor_count": len(source_tensor_keys),
        "runtime_tensor_count": len(runtime_tensor_keys),
        "peft_module_count": len(source_pairs),
        "runtime_adapter_module_count": len(runtime_pairs),
        "loaded_runtime_module_count": len(loaded_runtime_modules),
        "target_suffix_counts": dict(sorted(suffix_counts.items())),
        "target_family_counts": dict(sorted(family_counts.items())),
        "packed_modules_mapping": {
            key: list(value)
            for key, value in sorted(manager.packed_modules_mapping.items())
        },
        "source_module_mappings": source_mappings,
        "runtime_adapter_tensor_shapes": runtime_shapes,
        "loaded_runtime_inventory": loaded_inventory,
        "missing_loaded_runtime_modules": missing_loaded,
        "unexpected_loaded_runtime_modules": unexpected_loaded,
    }


def run_vllm_parity_sentinel(
    config: GeneralistV3Config,
    hf_runtime_path: Path,
    adapter_dir: Path,
    output_path: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Audit all vLLM mappings and prove the transported sentinel is active."""

    if output_path.exists():
        raise FileExistsError("generalist-v3 parity requires a fresh vLLM artifact")
    hf_runtime = _read_json(hf_runtime_path)
    identity = _adapter_identity(config, adapter_dir)
    if (
        hf_runtime.get("schema_version") != PARITY_HF_SCHEMA_VERSION
        or hf_runtime.get("status") != "passed"
        or hf_runtime.get("adapter") != identity
    ):
        raise ValueError("generalist-v3 vLLM parity HF binding differs")
    compatibility = prepare_qwen35_vllm_adapter(
        adapter_dir, split_gdn_qkv=True
    )
    runtime_adapter_dir = Path(str(compatibility["runtime_adapter_dir"]))
    inference = config.evaluation["inference"]
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    import vllm
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if vllm.__version__ != inference["engine_version"]:
        raise RuntimeError("generalist-v3 parity vLLM runtime differs")
    mapping_patch = patch_qwen35_vllm_gdn_lora_mapping(
        expected_version=str(inference["engine_version"])
    )
    snapshot = (
        model_snapshot.resolve()
        if model_snapshot is not None
        else Path(
            snapshot_download(
                repo_id=config.model["model_id"],
                revision=config.model["model_revision"],
                local_files_only=True,
            )
        ).resolve()
    )
    probes = hf_runtime["probes"]
    llm = LLM(
        model=str(snapshot),
        tokenizer=str(snapshot),
        dtype=str(inference["dtype"]),
        tensor_parallel_size=int(inference["tensor_parallel_size"]),
        gpu_memory_utilization=float(inference["gpu_memory_utilization"]),
        max_model_len=4096,
        max_num_seqs=8,
        enforce_eager=True,
        language_model_only=True,
        enable_prefix_caching=True,
        seed=0,
        trust_remote_code=False,
        enable_lora=True,
        max_lora_rank=int(config.lora["r"]),
        max_loras=1,
        worker_cls="qwen_lean.qwen35_vllm_worker.Qwen35Vllm017Worker",
    )
    request = LoRARequest(
        lora_name="qwen-lean-v3-parity-sentinel",
        lora_int_id=1,
        lora_path=str(runtime_adapter_dir),
    )
    llm.llm_engine.add_lora(request)
    audits = llm.collective_rpc(
        inspect_vllm_v3_lora_worker,
        args=(
            1,
            str(adapter_dir.resolve()),
            str(runtime_adapter_dir.resolve()),
            dict(config.lora["expected_module_counts"]),
            str(config.lora["target_regex"]),
        ),
    )
    if (
        len(audits) != 1
        or audits[0].get("status") != "passed"
        or audits[0].get("peft_module_count") != 248
        or audits[0].get("runtime_adapter_module_count") != 296
        or audits[0].get("loaded_runtime_module_count") != 152
        or audits[0].get("semantic_transform_verified") is not True
        or audits[0].get("source_modules_all_mapped") is not True
    ):
        raise RuntimeError("generalist-v3 vLLM LoRA static mapping audit failed")
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=64,
        seed=0,
        ignore_eos=False,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        logprobs=1,
    )
    base = _vllm_arm(llm, sampling, probes, lora_request=None)
    adapter = _vllm_arm(llm, sampling, probes, lora_request=request)
    pairs = []
    for base_row, adapter_row in zip(base, adapter, strict=True):
        if base_row["probe_id"] != adapter_row["probe_id"]:
            raise RuntimeError("generalist-v3 parity probe order differs")
        logprob_delta = None
        if (
            base_row["first_token_id"] == adapter_row["first_token_id"]
            and base_row["first_token_logprob"] is not None
            and adapter_row["first_token_logprob"] is not None
        ):
            logprob_delta = abs(
                float(base_row["first_token_logprob"])
                - float(adapter_row["first_token_logprob"])
            )
        pairs.append(
            {
                "probe_id": base_row["probe_id"],
                "output_changed": (
                    base_row["normalized_text_sha256"]
                    != adapter_row["normalized_text_sha256"]
                ),
                "same_first_token_logprob_delta": logprob_delta,
            }
        )
    functional_effect = any(
        bool(item["output_changed"])
        or float(item["same_first_token_logprob_delta"] or 0.0) > 1e-6
        for item in pairs
    )
    value = {
        "schema_version": PARITY_VLLM_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "status": "passed" if functional_effect else "failed",
        "model": dict(config.model),
        "adapter": identity,
        "hf_runtime_sha256": sha256_file(hf_runtime_path),
        "compatibility": compatibility,
        "vllm_gdn_mapping_patch": mapping_patch,
        "static_mapping_audit": audits[0],
        "base_arm": base,
        "adapter_arm": adapter,
        "functional_pairs": pairs,
        "functional_effect": functional_effect,
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
        },
    }
    _write_json(output_path, value)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if not functional_effect:
        raise RuntimeError("generalist-v3 vLLM parity sentinel has no inference effect")
    return value


def compact_lora_parity_evidence(
    config: GeneralistV3Config,
    hf_runtime_path: Path,
    vllm_runtime_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    hf_runtime = _read_json(hf_runtime_path)
    vllm_runtime = _read_json(vllm_runtime_path)
    audit = vllm_runtime.get("static_mapping_audit", {})
    requirements = {
        "hf_forward_effect": hf_runtime.get("all_probes_changed") is True,
        "vllm_static_mapping_complete": (
            audit.get("status") == "passed"
            and audit.get("peft_module_count") == 248
            and audit.get("runtime_adapter_module_count") == 296
            and audit.get("loaded_runtime_module_count") == 152
            and audit.get("semantic_transform_verified") is True
            and audit.get("source_modules_all_mapped") is True
            and not audit.get("missing_loaded_runtime_modules")
            and not audit.get("unexpected_loaded_runtime_modules")
        ),
        "vllm_functional_effect": vllm_runtime.get("functional_effect") is True,
        "adapter_identity_matched": (
            hf_runtime.get("adapter") == vllm_runtime.get("adapter")
        ),
        "local_cuda_only": (
            hf_runtime.get("runtime", {}).get("execution")
            == "project-controlled-local-cuda"
            and vllm_runtime.get("runtime", {}).get("execution")
            == "project-controlled-local-cuda"
        ),
        "no_optimizer_updates": (
            hf_runtime.get("sentinel_initialization", {}).get("optimizer_updates")
            == 0
        ),
    }
    passed = all(requirements.values())
    value = {
        "schema_version": PARITY_EVIDENCE_SCHEMA_VERSION,
        "gate_id": PARITY_GATE_ID,
        "status": "passed" if passed else "failed",
        "classification": (
            "PASS: Qwen3.5 PEFT-to-vLLM LoRA transport is active and complete"
            if passed
            else "FAIL: Qwen3.5 LoRA inference integrity is not established"
        ),
        "model": dict(config.model),
        "target_regex": config.lora["target_regex"],
        "adapter": hf_runtime.get("adapter"),
        "requirements": requirements,
        "static_mapping": {
            "source_tensor_count": audit.get("source_tensor_count"),
            "runtime_tensor_count": audit.get("runtime_tensor_count"),
            "peft_module_count": audit.get("peft_module_count"),
            "runtime_adapter_module_count": audit.get(
                "runtime_adapter_module_count"
            ),
            "loaded_runtime_module_count": audit.get(
                "loaded_runtime_module_count"
            ),
            "semantic_transform_verified": audit.get(
                "semantic_transform_verified"
            ),
            "target_suffix_counts": audit.get("target_suffix_counts"),
            "target_family_counts": audit.get("target_family_counts"),
            "missing_loaded_runtime_modules": audit.get(
                "missing_loaded_runtime_modules"
            ),
            "unexpected_loaded_runtime_modules": audit.get(
                "unexpected_loaded_runtime_modules"
            ),
        },
        "functional": {
            "hf_forward_sensitivity": hf_runtime.get("forward_sensitivity"),
            "vllm_pairs": vllm_runtime.get("functional_pairs"),
        },
        "artifacts": {
            "hf_runtime_sha256": sha256_file(hf_runtime_path),
            "vllm_runtime_sha256": sha256_file(vllm_runtime_path),
            "source_adapter_model_sha256": hf_runtime["adapter"][
                "adapter_model_sha256"
            ],
            "runtime_adapter_model_sha256": vllm_runtime["compatibility"][
                "runtime_adapter_model_sha256"
            ],
            "source_tensor_payload_sha256": vllm_runtime["compatibility"][
                "source_tensor_payload_sha256"
            ],
            "runtime_tensor_payload_sha256": vllm_runtime["compatibility"][
                "runtime_tensor_payload_sha256"
            ],
        },
        "optimizer_updates": 0,
        "quality_interpretation": "none; deterministic transport sentinel only",
    }
    _write_json(output_path, value)
    if not passed:
        raise RuntimeError("generalist-v3 LoRA parity gate failed")
    return value


def validate_lora_parity_gate(
    config: GeneralistV3Config, evidence_path: Path
) -> dict[str, Any]:
    evidence = _read_json(evidence_path)
    requirements = evidence.get("requirements", {})
    if (
        evidence.get("schema_version") != PARITY_EVIDENCE_SCHEMA_VERSION
        or evidence.get("gate_id") != PARITY_GATE_ID
        or evidence.get("status") != "passed"
        or evidence.get("model") != config.model
        or evidence.get("target_regex") != config.lora["target_regex"]
        or not requirements
        or not all(bool(value) for value in requirements.values())
    ):
        raise ValueError("generalist-v3 LoRA inference parity gate has not passed")
    return {
        "gate_id": PARITY_GATE_ID,
        "status": "passed",
        "evidence_sha256": sha256_file(evidence_path),
        "sentinel_adapter_model_sha256": evidence["adapter"][
            "adapter_model_sha256"
        ],
    }
