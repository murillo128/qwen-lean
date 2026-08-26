from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

from .dataset_v2 import sha256_file
from .dataset_v3 import materialize_example
from .generalist_v2_training import (
    ACTIVATION_CPU_OFFLOAD,
    ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS,
    FULL_ATTENTION_SDPA_BACKEND,
    GRADIENT_CHECKPOINTING_USE_REENTRANT,
    LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
    LM_HEAD_LOSS_CHUNK_TOKENS,
    MLP_SEQUENCE_CHUNK_TOKENS,
    checkpointed_target_only_causal_loss,
    configure_gradient_checkpointing,
    load_training_runtime,
    lora_target_summary,
    should_offload_activations,
)
from .generalist_v3 import (
    DatasetBinding,
    GeneralistV3Config,
    _read_json,
    _write_json,
    load_execution_training_index,
    load_training_index,
    tokenize_materialized_example,
)


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def _maximum_materialized_example(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    execution_view: Mapping[str, Any],
) -> dict[str, Any]:
    maximum = execution_view.get("maximum_eligible_example", {})
    if (
        execution_view.get("schema_version")
        != "generalist-v3-training-execution-view-v1"
        or int(execution_view.get("maximum_sequence_tokens", 0))
        != int(config.training["resolved_context_tokens"])
        or int(maximum.get("sequence_tokens", 0)) > int(config.training["resolved_context_tokens"])
    ):
        raise ValueError("near-maximum preflight needs the frozen training execution view")
    records, examples = load_execution_training_index(config, binding, execution_view)
    statement_id = str(maximum["statement_id"])
    example_id = str(maximum["example_id"])
    matches = [item for item in examples[statement_id] if item.example_id == example_id]
    if len(matches) != 1:
        raise ValueError("near-maximum Dataset-v3 example does not resolve once")
    return materialize_example(records[statement_id], matches[0])


def run_no_update_near_max_preflight(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    execution_view_path: Path,
    output_path: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Run the exact near-maximum forward/backward without an optimizer update."""

    config.validate()
    execution_view = _read_json(execution_view_path)
    maximum = execution_view["maximum_eligible_example"]
    materialized = _maximum_materialized_example(config, binding, execution_view)
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as error:
        raise RuntimeError("generalist-v3 preflight requires its training runtime") from error
    if not torch.cuda.is_available():
        raise RuntimeError("generalist-v3 preflight requires project-controlled CUDA")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    runtime = load_training_runtime(config, model_snapshot=model_snapshot)  # type: ignore[arg-type]
    example = tokenize_materialized_example(
        materialized,
        runtime.tokenizer,
        maximum_sequence_tokens=int(config.training["resolved_context_tokens"]),
    )
    if len(example.input_ids) != int(maximum["sequence_tokens"]):
        raise RuntimeError("near-maximum tokenization differs from the frozen execution view")
    device = next(
        parameter.device
        for parameter in runtime.model.parameters()
        if parameter.device.type == "cuda"
    )
    input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([example.labels], dtype=torch.long, device=device)
    causal_model = (
        runtime.model.get_base_model()
        if hasattr(runtime.model, "get_base_model")
        else runtime.model
    )
    configure_gradient_checkpointing(causal_model, len(example.input_ids))
    runtime.model.train()
    for parameter in runtime.model.parameters():
        if parameter.requires_grad:
            parameter.grad = None
    activation_context = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if should_offload_activations(len(example.input_ids))
        else nullcontext()
    )
    with (
        activation_context,
        sdpa_kernel(SDPBackend.FLASH_ATTENTION),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        outputs = causal_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_size=LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
            use_cache=False,
        )
        loss = checkpointed_target_only_causal_loss(
            causal_model,
            outputs.last_hidden_state,
            labels,
        )
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("generalist-v3 near-maximum loss is non-finite")
    loss.backward()
    trainable = [
        parameter for parameter in runtime.model.parameters() if parameter.requires_grad
    ]
    if not trainable or any(parameter.grad is None for parameter in trainable):
        raise RuntimeError("generalist-v3 near-maximum gradients are missing")
    if any(not bool(torch.isfinite(parameter.grad).all().item()) for parameter in trainable):
        raise RuntimeError("generalist-v3 near-maximum gradients are non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable, float(config.training["maximum_gradient_norm"])
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError("generalist-v3 near-maximum gradient norm is non-finite")
    torch.cuda.synchronize(device_index)
    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_index))
    total_memory = int(properties.total_memory)
    value = {
        "schema_version": "generalist-v3-near-max-preflight-v2",
        "status": "passed",
        "model": config.model,
        "dataset_binding": binding.to_dict(),
        "training_execution_view_sha256": sha256_file(execution_view_path),
        "execution_view_identity_sha256": execution_view["execution_view_sha256"],
        "near_maximum_example": {
            "statement_id": example.statement_id,
            "example_id": example.proof_variant_id,
            "task_kind": example.declaration_name,
            "sequence_tokens": len(example.input_ids),
            "prompt_tokens": example.prompt_tokens,
            "target_tokens_excluding_eos": example.completion_tokens,
            "configured_context_tokens": int(config.training["resolved_context_tokens"]),
            "truncated_or_dropped": False,
        },
        "forward_backward": {
            "loss": float(loss.detach().item()),
            "loss_finite": True,
            "gradient_norm_before_clipping": float(gradient_norm.detach().item()),
            "all_trainable_gradients_present": True,
            "all_gradients_finite": True,
            "optimizer_created": False,
            "optimizer_update_run": False,
        },
        "lora": {
            "target_regex": config.lora["target_regex"],
            "rank": config.lora["r"],
            **lora_target_summary(runtime.target_matches),
        },
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "python": platform.python_version(),
            "cuda_device_index": device_index,
            "cuda_device": properties.name,
            "cuda_device_capability": [properties.major, properties.minor],
            "cuda_device_total_memory_bytes": total_memory,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "reserved_memory_headroom_bytes": total_memory - peak_reserved,
            "wall_time_seconds": time.perf_counter() - started,
            "torch_cuda_version": torch.version.cuda,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "gradient_checkpointing_use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT,
            "linear_attention_chunk_size": LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
            "full_attention_sdpa_backend": FULL_ATTENTION_SDPA_BACKEND,
            "activation_cpu_offload": ACTIVATION_CPU_OFFLOAD,
            "activation_cpu_offload_min_sequence_tokens": ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS,
            "lm_head_loss_chunk_tokens": LM_HEAD_LOSS_CHUNK_TOKENS,
            "mlp_sequence_chunk_tokens": MLP_SEQUENCE_CHUNK_TOKENS,
            "packages": _package_versions(
                (
                    "torch",
                    "transformers",
                    "peft",
                    "trl",
                    "bitsandbytes",
                    "accelerate",
                    "datasets",
                    "huggingface-hub",
                    "safetensors",
                    "flash-linear-attention",
                )
            ),
        },
    }
    _write_json(output_path, value)
    del labels, attention_mask, input_ids, outputs, loss, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _resolve_anchor_inputs(
    binding: DatasetBinding, anchor_manifest: Mapping[str, Any]
) -> list[str]:
    if (
        anchor_manifest.get("schema_version") != "generalist-v3-anchor-manifest-v1"
        or anchor_manifest.get("anchor_count") != 512
        or anchor_manifest.get("validation_or_test_anchors") != 0
    ):
        raise ValueError("generalist-v3 reference cache needs the frozen anchors")
    records, by_statement = load_training_index(binding)
    resolved: list[str] = []
    for anchor in anchor_manifest["anchors"]:
        statement_id = str(anchor["statement_id"])
        example_id = str(anchor["example_id"])
        matches = [item for item in by_statement[statement_id] if item.example_id == example_id]
        if len(matches) != 1:
            raise ValueError("generalist-v3 anchor does not resolve exactly once")
        model_input = str(materialize_example(records[statement_id], matches[0])["model_input"])
        if hashlib.sha256(model_input.encode()).hexdigest() != anchor["model_input_sha256"]:
            raise ValueError("generalist-v3 anchor input hash differs")
        resolved.append(model_input)
    return resolved


def cache_base_reference_logits(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    anchor_manifest_path: Path,
    logits_path: Path,
    metadata_path: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Cache exact full-vocabulary Base logits at each anchor's final input token."""

    config.validate()
    anchor_manifest = _read_json(anchor_manifest_path)
    inputs = _resolve_anchor_inputs(binding, anchor_manifest)
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("generalist-v3 reference cache requires the training runtime") from error
    if not torch.cuda.is_available():
        raise RuntimeError("generalist-v3 reference cache requires local CUDA")
    device_index = torch.cuda.current_device()
    source = str(model_snapshot) if model_snapshot is not None else str(config.model["model_id"])
    source_kwargs = (
        {"local_files_only": True}
        if model_snapshot is not None
        else {"revision": str(config.model["model_revision"]), "local_files_only": True}
    )
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **source_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        **source_kwargs,
    )
    if type(model).__name__ != config.model["architecture_class"]:
        raise RuntimeError("generalist-v3 reference model architecture differs")
    model.eval()
    cached = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, model_input in enumerate(inputs):
            input_ids = tokenizer.encode(
                model_input, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                chunk_size=LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
                use_cache=False,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1, :]).squeeze(0)
            cached.append(logits.detach().to(device="cpu", dtype=torch.float16))
            if (index + 1) % 64 == 0:
                print(json.dumps({"anchors_cached": index + 1, "total": 512}), flush=True)
    tensor = torch.stack(cached)
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"base_next_token_logits": tensor}, str(logits_path))
    value = {
        "schema_version": "generalist-v3-base-reference-logits-v1",
        "model_id": config.model["model_id"],
        "model_revision": config.model["model_revision"],
        "anchor_manifest_sha256": sha256_file(anchor_manifest_path),
        "anchor_count": tensor.shape[0],
        "vocabulary_size": tensor.shape[1],
        "dtype": "float16",
        "position": "final-input-next-token",
        "approximation": None,
        "logits_file": logits_path.name,
        "logits_sha256": sha256_file(logits_path),
        "wall_time_seconds": time.perf_counter() - started,
        "execution": "project-controlled-local-cuda",
        "packages": _package_versions(("torch", "transformers", "safetensors")),
    }
    _write_json(metadata_path, value)
    del model, tensor, cached
    gc.collect()
    torch.cuda.empty_cache()
    return value


def base_forward_kl(base_logits: Any, current_logits: Any) -> Any:
    try:
        import torch
        from torch.nn import functional
    except ImportError as error:
        raise RuntimeError("generalist-v3 KL requires PyTorch") from error
    if base_logits.shape != current_logits.shape or base_logits.ndim != 1:
        raise ValueError("generalist-v3 Base/current logits shapes differ")
    base_log_probabilities = functional.log_softmax(base_logits.float(), dim=-1)
    current_log_probabilities = functional.log_softmax(current_logits.float(), dim=-1)
    base_probabilities = base_log_probabilities.exp()
    value = torch.sum(
        base_probabilities * (base_log_probabilities - current_log_probabilities)
    )
    if not bool(torch.isfinite(value).item()) or float(value.detach().item()) < -1e-6:
        raise RuntimeError("generalist-v3 Base-preservation KL is invalid")
    return value.clamp_min(0.0)
