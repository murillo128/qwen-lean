from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    anchor_schedule,
    load_execution_training_index,
    load_training_index,
    tokenize_materialized_example,
)


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def _nvidia_smi_query(arguments: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": ["nvidia-smi", *arguments],
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def freeze_clean_gpu_baseline(output_path: Path) -> dict[str, Any]:
    """Record and enforce the final-Ada clean-GPU gate before model load."""

    gpu_query = _nvidia_smi_query(
        [
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = _nvidia_smi_query(
        [
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_query["returncode"] != 0 or not gpu_query["stdout"]:
        raise RuntimeError("clean-GPU baseline could not query NVIDIA GPU memory")
    gpu_rows = []
    for line in str(gpu_query["stdout"]).splitlines():
        parts = [item.strip() for item in line.split(",", 3)]
        if len(parts) != 4:
            raise RuntimeError("clean-GPU baseline received malformed GPU output")
        gpu_rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "total_memory_mib": int(parts[2]),
                "free_memory_mib": int(parts[3]),
            }
        )
    processes = []
    if process_query["returncode"] == 0 and process_query["stdout"]:
        for line in str(process_query["stdout"]).splitlines():
            parts = [item.strip() for item in line.split(",", 2)]
            if len(parts) == 3:
                processes.append(
                    {
                        "pid": int(parts[0]),
                        "process_name": parts[1],
                        "used_memory_mib": (
                            None if parts[2] in {"[N/A]", "N/A"} else int(parts[2])
                        ),
                    }
                )
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("clean-GPU baseline requires the training PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("clean-GPU baseline requires project-controlled CUDA")
    device_index = torch.cuda.current_device()
    if device_index >= len(gpu_rows):
        raise RuntimeError("clean-GPU baseline cannot bind the active CUDA device")
    gpu = gpu_rows[device_index]
    torch_free, torch_total = torch.cuda.mem_get_info(device_index)
    free_fraction = gpu["free_memory_mib"] / gpu["total_memory_mib"]
    clean = free_fraction >= 0.90 and not processes
    value = {
        "schema_version": "generalist-v3-clean-gpu-baseline-v1",
        "status": "passed" if clean else "environment-contamination / GPU-not-clean",
        "fresh_python_process_pid": os.getpid(),
        "model_load_started": False,
        "nvidia_smi": {
            "gpu_query": gpu_query,
            "compute_process_query": process_query,
            "gpus": gpu_rows,
            "compute_processes": processes,
        },
        "active_device": {
            **gpu,
            "nvidia_smi_free_fraction": free_fraction,
            "minimum_required_free_fraction": 0.90,
            "torch_cuda_mem_get_info_before_model_load": {
                "free_bytes": int(torch_free),
                "total_bytes": int(torch_total),
            },
            "torch_allocated_bytes_before_model_load": int(
                torch.cuda.memory_allocated(device_index)
            ),
            "torch_reserved_bytes_before_model_load": int(
                torch.cuda.memory_reserved(device_index)
            ),
        },
        "no_other_compute_processes": not processes,
        "clean_gpu_gate_passed": clean,
    }
    _write_json(output_path, value)
    if not clean:
        raise RuntimeError("environment-contamination / GPU-not-clean")
    return value


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
    clean_gpu_baseline_path: Path,
    output_path: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Run the exact near-maximum forward/backward without an optimizer update."""

    config.validate()
    execution_view = _read_json(execution_view_path)
    maximum = execution_view["maximum_eligible_example"]
    materialized = _maximum_materialized_example(config, binding, execution_view)
    clean_gpu_baseline = freeze_clean_gpu_baseline(clean_gpu_baseline_path)
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
    initial_free_memory = int(torch.cuda.mem_get_info(device_index)[0])
    started = time.perf_counter()
    runtime = load_training_runtime(config, model_snapshot=model_snapshot)  # type: ignore[arg-type]
    torch.cuda.synchronize(device_index)
    post_load_free_memory = int(torch.cuda.mem_get_info(device_index)[0])
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
    torch.cuda.synchronize(device_index)
    post_forward_free_memory = int(torch.cuda.mem_get_info(device_index)[0])
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("generalist-v3 near-maximum loss is non-finite")
    loss.backward()
    torch.cuda.synchronize(device_index)
    post_backward_free_memory = int(torch.cuda.mem_get_info(device_index)[0])
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
    minimum_allocated_headroom = 1 << 30
    allocated_headroom = total_memory - peak_allocated
    memory_gate_passed = allocated_headroom >= minimum_allocated_headroom
    value = {
        "schema_version": "generalist-v3-near-max-preflight-v4",
        "status": "passed" if memory_gate_passed else "failed-headroom",
        "model": config.model,
        "dataset_binding": binding.to_dict(),
        "training_execution_view_sha256": sha256_file(execution_view_path),
        "execution_view_identity_sha256": execution_view["execution_view_sha256"],
        "clean_gpu_baseline_sha256": sha256_file(clean_gpu_baseline_path),
        "clean_gpu_gate_passed": clean_gpu_baseline["clean_gpu_gate_passed"],
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
            "allocated_memory_headroom_bytes": allocated_headroom,
            "minimum_allocated_memory_headroom_bytes": minimum_allocated_headroom,
            "allocated_memory_headroom_gate_passed": memory_gate_passed,
            "reserved_memory_headroom_bytes": total_memory - peak_reserved,
            "free_memory_observations_bytes": {
                "before_model_load": initial_free_memory,
                "after_model_load": post_load_free_memory,
                "after_forward_and_loss": post_forward_free_memory,
                "after_backward": post_backward_free_memory,
            },
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
    if not memory_gate_passed:
        raise RuntimeError("generalist-v3 preflight leaves less than 1 GiB allocated headroom")
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


def _validate_training_gates(
    config: GeneralistV3Config,
    stage0_freeze_path: Path,
    base_canary_evidence_path: Path,
    parity_evidence_path: Path,
    stream_path: Path,
    stream_manifest_path: Path,
    anchor_manifest_path: Path,
    base_logits_path: Path,
    base_logits_metadata_path: Path,
) -> dict[str, Any]:
    from .generalist_v3_parity import validate_lora_parity_gate

    freeze = _read_json(stage0_freeze_path)
    base_canary = _read_json(base_canary_evidence_path)
    stream_manifest = _read_json(stream_manifest_path)
    anchor_manifest = _read_json(anchor_manifest_path)
    logits_metadata = _read_json(base_logits_metadata_path)
    parity = validate_lora_parity_gate(config, parity_evidence_path)
    if (
        freeze.get("schema_version") != "generalist-v3-stage0-freeze-v1"
        or freeze.get("config_sha256") != sha256_file(config.path)
        or freeze.get("optimizer_updates") != 0
        or base_canary.get("schema_version")
        != "generalist-v3-base-canary-evidence-v1"
        or base_canary.get("status") != "passed"
        or base_canary.get("protocol", {}).get("optimizer_updates") != 0
        or base_canary.get("protocol", {}).get("sealed_test_accessed") is not False
        or stream_manifest.get("schema_version")
        != "generalist-v3-training-stream-v1"
        or stream_manifest.get("gzip_file_sha256") != sha256_file(stream_path)
        or freeze.get("training_stream_manifest_sha256")
        != sha256_file(stream_manifest_path)
        or freeze.get("anchor_manifest_sha256") != sha256_file(anchor_manifest_path)
        or anchor_manifest.get("anchor_count") != 512
        or logits_metadata.get("schema_version")
        != "generalist-v3-base-reference-logits-v1"
        or logits_metadata.get("anchor_manifest_sha256")
        != sha256_file(anchor_manifest_path)
        or logits_metadata.get("logits_sha256") != sha256_file(base_logits_path)
    ):
        raise ValueError("generalist-v3 optimizer gates are incomplete or stale")
    return {
        "stage0_freeze_sha256": sha256_file(stage0_freeze_path),
        "base_canary_evidence_sha256": sha256_file(base_canary_evidence_path),
        "parity": parity,
        "training_stream_sha256": sha256_file(stream_path),
        "training_stream_manifest_sha256": sha256_file(stream_manifest_path),
        "anchor_manifest_sha256": sha256_file(anchor_manifest_path),
        "base_logits_sha256": sha256_file(base_logits_path),
        "base_logits_metadata_sha256": sha256_file(base_logits_metadata_path),
        "sealed_test_accessed": False,
    }


def compact_stage0_evidence(
    config: GeneralistV3Config,
    artifact_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    stage0 = artifact_root / "stage0"
    census_path = stage0 / "tokenizer-census.json"
    execution_path = stage0 / "training-execution-view.json"
    baseline_path = stage0 / "clean-gpu-baseline.json"
    preflight_path = stage0 / "near-max-preflight.json"
    structural_path = stage0 / "structural-sampling.json"
    anchors_path = stage0 / "anchor-manifest.json"
    logits_metadata_path = stage0 / "base-reference-logits.json"
    logits_path = stage0 / "base-reference-logits.safetensors"
    freeze_path = stage0 / "freeze.json"
    stream_manifest_path = artifact_root / "training-stream-manifest.json"
    stream_path = artifact_root / "training-stream.jsonl.gz"
    census = _read_json(census_path)
    execution = _read_json(execution_path)
    baseline = _read_json(baseline_path)
    preflight = _read_json(preflight_path)
    structural = _read_json(structural_path)
    anchors = _read_json(anchors_path)
    logits = _read_json(logits_metadata_path)
    freeze = _read_json(freeze_path)
    stream = _read_json(stream_manifest_path)
    if (
        census.get("execution_ceiling_tokens") != 16384
        or census.get("examples_above_execution_ceiling") != 51
        or execution.get("quarantined_example_count") != 51
        or execution.get("census_quarantine_identity_matched") is not True
        or execution.get("maximum_sequence_tokens") != 16384
        or baseline.get("clean_gpu_gate_passed") is not True
        or preflight.get("status") != "passed"
        or preflight.get("clean_gpu_gate_passed") is not True
        or preflight.get("forward_backward", {}).get("optimizer_update_run") is not False
        or preflight.get("runtime", {}).get("allocated_memory_headroom_gate_passed")
        is not True
        or freeze.get("config_sha256") != sha256_file(config.path)
        or freeze.get("training_stream_manifest_sha256")
        != sha256_file(stream_manifest_path)
        or stream.get("gzip_file_sha256") != sha256_file(stream_path)
        or stream.get("microbatches") != 64000
        or anchors.get("anchor_count") != 512
        or anchors.get("validation_or_test_anchors") != 0
        or logits.get("logits_sha256") != sha256_file(logits_path)
    ):
        raise ValueError("generalist-v3 compact Stage 0 evidence is stale")
    value = {
        "schema_version": "generalist-v3-stage0-evidence-v1",
        "status": "passed",
        "marker": "OBSERVED",
        "model": dict(config.model),
        "dataset_binding": execution["dataset_binding"],
        "execution_view": {
            "maximum_sequence_tokens": execution["maximum_sequence_tokens"],
            "independent_census_count": census["examples_above_execution_ceiling"],
            "independent_census_identity_sha256": census[
                "example_ids_above_execution_ceiling_sha256"
            ],
            "census_quarantine_identity_matched": execution[
                "census_quarantine_identity_matched"
            ],
            "quarantined_examples": execution["quarantined_example_count"],
            "quarantined_examples_sha256": execution[
                "quarantined_examples_sha256"
            ],
            "quarantined_unique_theorems": execution[
                "quarantined_unique_theorems"
            ],
            "affected_task_kind_counts": execution["affected_task_kind_counts"],
            "affected_structural_bucket_counts": execution[
                "affected_structural_bucket_counts"
            ],
            "excluded_theorems": execution["excluded_theorem_count"],
            "excluded_theorem_ids_sha256": execution[
                "excluded_theorem_ids_sha256"
            ],
            "retained_examples": execution["retained_example_count"],
            "retained_theorems": execution["retained_theorem_count"],
            "maximum_eligible_example": execution["maximum_eligible_example"],
            "execution_view_identity_sha256": execution["execution_view_sha256"],
            "dataset_v3_mutated": execution["dataset_v3_mutated"],
            "truncated_examples": execution["truncated_examples"],
            "silent_drops": execution["silent_drops"],
        },
        "clean_gpu_baseline": {
            "artifact_sha256": sha256_file(baseline_path),
            "nvidia_smi_free_fraction": baseline["active_device"][
                "nvidia_smi_free_fraction"
            ],
            "compute_processes": baseline["nvidia_smi"]["compute_processes"],
            "no_other_compute_processes": baseline["no_other_compute_processes"],
            "torch_mem_get_info_before_model_load": baseline["active_device"][
                "torch_cuda_mem_get_info_before_model_load"
            ],
        },
        "near_maximum_preflight": {
            "artifact_sha256": sha256_file(preflight_path),
            "example": preflight["near_maximum_example"],
            "forward_backward": preflight["forward_backward"],
            "peak_cuda_allocated_bytes": preflight["runtime"][
                "peak_cuda_allocated_bytes"
            ],
            "peak_cuda_reserved_bytes": preflight["runtime"][
                "peak_cuda_reserved_bytes"
            ],
            "allocated_memory_headroom_bytes": preflight["runtime"][
                "allocated_memory_headroom_bytes"
            ],
            "minimum_allocated_memory_headroom_bytes": preflight["runtime"][
                "minimum_allocated_memory_headroom_bytes"
            ],
            "free_memory_observations_bytes": preflight["runtime"][
                "free_memory_observations_bytes"
            ],
            "runtime": preflight["runtime"],
        },
        "structural_sampling": {
            "pre_quarantine_bucket_counts": structural[
                "pre_quarantine_bucket_counts"
            ],
            "post_quarantine_bucket_counts": structural[
                "post_quarantine_bucket_counts"
            ],
            "multipliers": structural["multipliers"],
            "theorem_to_bucket_mapping_sha256": structural[
                "theorem_to_bucket_mapping_sha256"
            ],
            "structural_sampling_sha256": structural[
                "structural_sampling_sha256"
            ],
        },
        "anchors_and_base_logits": {
            "anchor_manifest_sha256": sha256_file(anchors_path),
            "anchors_sha256": anchors["anchors_sha256"],
            "kind_counts": anchors["kind_counts"],
            "structural_bucket_counts": anchors["structural_bucket_counts"],
            "preferred_length_count": anchors["preferred_length_count"],
            "base_logits_metadata_sha256": sha256_file(logits_metadata_path),
            "base_logits_sha256": logits["logits_sha256"],
            "base_logits_dtype": logits["dtype"],
            "base_logits_vocabulary_size": logits["vocabulary_size"],
            "base_logits_approximation": logits["approximation"],
        },
        "training_stream": {
            "manifest_sha256": sha256_file(stream_manifest_path),
            "gzip_file_sha256": stream["gzip_file_sha256"],
            "canonical_rows_sha256": stream["canonical_rows_sha256"],
            "microbatches": stream["microbatches"],
            "optimizer_steps": stream["optimizer_steps"],
            "kind_counts": stream["kind_counts"],
            "structural_bucket_counts": stream["structural_bucket_counts"],
            "structural_distribution_agreement": stream[
                "structural_distribution_agreement"
            ],
        },
        "protocol": {
            "base_canary_generation_started": True,
            "adapter_training_started": False,
            "optimizer_updates": 0,
            "semantic_sealed_test_accessed": False,
        },
        "artifacts": {
            "tokenizer_census_sha256": sha256_file(census_path),
            "training_execution_view_sha256": sha256_file(execution_path),
            "structural_sampling_file_sha256": sha256_file(structural_path),
            "stage0_freeze_sha256": sha256_file(freeze_path),
        },
    }
    _write_json(output_path, value)
    return value


def _load_stream_prefix(path: Path, *, optimizer_steps: int) -> list[dict[str, Any]]:
    expected_rows = optimizer_steps * 8
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) == expected_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != expected_rows:
        raise ValueError("generalist-v3 training stream prefix is incomplete")
    for index, row in enumerate(rows):
        if (
            row.get("schema_version") != "generalist-v3-stream-row-v1"
            or row.get("stream_index") != index
            or row.get("optimizer_step") != index // 8 + 1
            or row.get("accumulation_index") != index % 8
            or hashlib.sha256(str(row["model_input"]).encode()).hexdigest()
            != row.get("model_input_sha256")
            or hashlib.sha256(str(row["target"]).encode()).hexdigest()
            != row.get("target_sha256")
        ):
            raise ValueError("generalist-v3 training stream row binding differs")
    return rows


def _tensorized_example(runtime: Any, materialized: Mapping[str, Any], maximum: int):
    import torch

    example = tokenize_materialized_example(
        materialized, runtime.tokenizer, maximum_sequence_tokens=maximum
    )
    device = next(
        parameter.device
        for parameter in runtime.model.parameters()
        if parameter.device.type == "cuda"
    )
    input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([example.labels], dtype=torch.long, device=device)
    return example, input_ids, labels


def _backward_sft_microbatch(
    runtime: Any,
    materialized: Mapping[str, Any],
    *,
    maximum_sequence_tokens: int,
    accumulation_steps: int,
) -> tuple[float, int, int]:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    example, input_ids, labels = _tensorized_example(
        runtime, materialized, maximum_sequence_tokens
    )
    causal_model = runtime.model.get_base_model()
    configure_gradient_checkpointing(causal_model, len(example.input_ids))
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
            attention_mask=torch.ones_like(input_ids),
            chunk_size=LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
            use_cache=False,
        )
        loss = checkpointed_target_only_causal_loss(
            causal_model, outputs.last_hidden_state, labels
        )
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("generalist-v3 SFT loss is non-finite")
    (loss / accumulation_steps).backward()
    value = float(loss.detach().item())
    tokens = len(example.input_ids)
    target_tokens = example.completion_tokens + 1
    del outputs, loss, labels, input_ids, example
    return value, tokens, target_tokens


def _backward_anchor_kl(
    runtime: Any,
    model_input: str,
    base_logits: Any,
    *,
    coefficient: float,
) -> tuple[float, int]:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    input_ids = runtime.tokenizer.encode(
        model_input, add_special_tokens=False, return_tensors="pt"
    ).to(runtime.model.device)
    causal_model = runtime.model.get_base_model()
    configure_gradient_checkpointing(causal_model, int(input_ids.shape[1]))
    with (
        sdpa_kernel(SDPBackend.FLASH_ATTENTION),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        outputs = causal_model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            chunk_size=LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
            use_cache=False,
        )
        current_logits = causal_model.lm_head(
            outputs.last_hidden_state[:, -1, :]
        ).squeeze(0)
        kl = base_forward_kl(base_logits.to(current_logits.device), current_logits)
        weighted = coefficient * kl
    weighted.backward()
    value = float(kl.detach().item())
    tokens = int(input_ids.shape[1])
    del outputs, current_logits, weighted, kl, input_ids
    return value, tokens


def _save_training_checkpoint(
    runtime: Any,
    optimizer: Any,
    output_dir: Path,
    *,
    configuration_id: str,
    optimizer_step: int,
    learning_rate: float,
    base_kl_lambda: float,
    stream_rows_consumed: int,
    gates: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    import torch

    checkpoint = output_dir / f"checkpoint-{optimizer_step}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    runtime.model.save_pretrained(checkpoint, safe_serialization=True)
    runtime.tokenizer.save_pretrained(checkpoint)
    state_path = checkpoint / "optimizer-rng-state.pt"
    torch.save(
        {
            "configuration_id": configuration_id,
            "optimizer_step": optimizer_step,
            "stream_rows_consumed": stream_rows_consumed,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        state_path,
    )
    metadata = {
        "schema_version": "generalist-v3-training-checkpoint-v1",
        "configuration_id": configuration_id,
        "optimizer_step": optimizer_step,
        "learning_rate": learning_rate,
        "base_kl_lambda": base_kl_lambda,
        "stream_rows_consumed": stream_rows_consumed,
        "adapter_model_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(checkpoint / "adapter_config.json"),
        "optimizer_rng_state_sha256": sha256_file(state_path),
        "training_log_sha256": sha256_file(log_path),
        "gates": dict(gates),
        "adapter_only": True,
        "sealed_test_accessed": False,
    }
    _write_json(checkpoint / "checkpoint.json", metadata)
    return metadata


def run_bounded_configuration_training(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    stage0_freeze_path: Path,
    base_canary_evidence_path: Path,
    parity_evidence_path: Path,
    stream_path: Path,
    stream_manifest_path: Path,
    anchor_manifest_path: Path,
    base_logits_path: Path,
    base_logits_metadata_path: Path,
    output_dir: Path,
    *,
    configuration_id: str,
    maximum_optimizer_steps: int = 500,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Run one paired bounded arm through the frozen 500-step boundary."""

    if configuration_id not in {"C0", "C1", "C2", "C3"}:
        raise ValueError("bounded training only permits C0/C1/C2/C3")
    if maximum_optimizer_steps != 500:
        raise ValueError("initial bounded training must stop exactly at 500 steps")
    if output_dir.exists():
        raise FileExistsError("generalist-v3 bounded training needs a fresh output")
    gates = _validate_training_gates(
        config,
        stage0_freeze_path,
        base_canary_evidence_path,
        parity_evidence_path,
        stream_path,
        stream_manifest_path,
        anchor_manifest_path,
        base_logits_path,
        base_logits_metadata_path,
    )
    configuration = config.training["configurations"][configuration_id]
    learning_rate = float(configuration["learning_rate"])
    coefficient = float(configuration["base_kl_lambda"])
    stream_rows = _load_stream_prefix(
        stream_path, optimizer_steps=maximum_optimizer_steps
    )
    anchor_manifest = _read_json(anchor_manifest_path)
    anchor_inputs = _resolve_anchor_inputs(binding, anchor_manifest)
    anchor_indices = anchor_schedule(512, maximum_optimizer_steps)
    try:
        import bitsandbytes as bnb
        import torch
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("generalist-v3 bounded training requires its runtime") from error
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    runtime = load_training_runtime(config, model_snapshot=model_snapshot)  # type: ignore[arg-type]
    runtime.model.train()
    trainables = [
        parameter for parameter in runtime.model.parameters() if parameter.requires_grad
    ]
    optimizer = bnb.optim.PagedAdamW8bit(
        trainables, lr=learning_rate, weight_decay=float(config.training["weight_decay"])
    )
    cached_logits = load_file(str(base_logits_path))["base_next_token_logits"]
    if cached_logits.shape != (512, 248320):
        raise RuntimeError("generalist-v3 cached Base logits shape differs")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training-log.jsonl"
    retained = set(int(item) for item in (100, 250, 500))
    checkpoints = {}
    maximum_sequence = int(config.training["resolved_context_tokens"])
    device_index = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
        for optimizer_step in range(1, maximum_optimizer_steps + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_rows = stream_rows[(optimizer_step - 1) * 8 : optimizer_step * 8]
            sft_losses = []
            sequence_tokens = []
            target_tokens = []
            for row in step_rows:
                loss, tokens, supervised = _backward_sft_microbatch(
                    runtime,
                    row,
                    maximum_sequence_tokens=maximum_sequence,
                    accumulation_steps=8,
                )
                sft_losses.append(loss)
                sequence_tokens.append(tokens)
                target_tokens.append(supervised)
            anchor_index = anchor_indices[optimizer_step - 1]
            anchor_kl, anchor_tokens = _backward_anchor_kl(
                runtime,
                anchor_inputs[anchor_index],
                cached_logits[anchor_index],
                coefficient=coefficient,
            )
            missing = [parameter for parameter in trainables if parameter.grad is None]
            if missing or any(
                not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in trainables
            ):
                raise RuntimeError("generalist-v3 bounded training gradients are invalid")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainables, float(config.training["maximum_gradient_norm"])
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("generalist-v3 bounded gradient norm is non-finite")
            step_lr = learning_rate * min(
                optimizer_step / int(config.training["warmup_steps"]), 1.0
            )
            for group in optimizer.param_groups:
                group["lr"] = step_lr
            optimizer.step()
            row = {
                "schema_version": "generalist-v3-training-log-row-v1",
                "configuration_id": configuration_id,
                "optimizer_step": optimizer_step,
                "stream_start_index": (optimizer_step - 1) * 8,
                "stream_end_index_exclusive": optimizer_step * 8,
                "sft_loss_mean": sum(sft_losses) / len(sft_losses),
                "sft_loss_microbatches": sft_losses,
                "preservation_kl": anchor_kl,
                "base_kl_lambda": coefficient,
                "weighted_preservation_kl": coefficient * anchor_kl,
                "gradient_norm_before_clipping": float(gradient_norm.detach().item()),
                "learning_rate": step_lr,
                "anchor_index": anchor_index,
                "anchor_input_tokens": anchor_tokens,
                "sequence_tokens": {
                    "minimum": min(sequence_tokens),
                    "maximum": max(sequence_tokens),
                    "sum": sum(sequence_tokens),
                },
                "supervised_tokens": sum(target_tokens),
                "wall_time_seconds": time.perf_counter() - step_started,
            }
            if not all(
                math.isfinite(float(row[key]))
                for key in (
                    "sft_loss_mean",
                    "preservation_kl",
                    "gradient_norm_before_clipping",
                )
            ):
                raise RuntimeError("generalist-v3 bounded training log is non-finite")
            log_handle.write(json.dumps(row, sort_keys=True) + "\n")
            log_handle.flush()
            if optimizer_step in retained:
                checkpoints[str(optimizer_step)] = _save_training_checkpoint(
                    runtime,
                    optimizer,
                    output_dir,
                    configuration_id=configuration_id,
                    optimizer_step=optimizer_step,
                    learning_rate=learning_rate,
                    base_kl_lambda=coefficient,
                    stream_rows_consumed=optimizer_step * 8,
                    gates=gates,
                    log_path=log_path,
                )
            print(
                json.dumps(
                    {
                        "configuration_id": configuration_id,
                        "optimizer_step": optimizer_step,
                        "sft_loss": row["sft_loss_mean"],
                        "preservation_kl": anchor_kl,
                        "gradient_norm": row["gradient_norm_before_clipping"],
                        "sequence_tokens_max": max(sequence_tokens),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device_index)
    evidence = {
        "schema_version": "generalist-v3-bounded-training-v1",
        "status": "passed",
        "configuration_id": configuration_id,
        "configuration": configuration,
        "optimizer_steps": maximum_optimizer_steps,
        "stream_rows_consumed": maximum_optimizer_steps * 8,
        "retained_steps": [100, 250, 500],
        "checkpoints": checkpoints,
        "gates": gates,
        "training_log_sha256": sha256_file(log_path),
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "lane": runtime.lane,
            "cuda_device": torch.cuda.get_device_name(device_index),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
            "wall_time_seconds": time.perf_counter() - started,
            "trainable_parameter_count": runtime.trainable_parameter_count,
            "optimizer": "bitsandbytes.PagedAdamW8bit",
        },
        "objective": "mean-of-8 target-only SFT + lambda * KL(p_Base||p_current)",
        "optimizer_updates": maximum_optimizer_steps,
        "sealed_test_accessed": False,
    }
    _write_json(output_dir / "training.json", evidence)
    del runtime, optimizer, trainables, cached_logits
    gc.collect()
    torch.cuda.empty_cache()
    return evidence


def measure_checkpoint_anchor_drift(
    config: GeneralistV3Config,
    binding: DatasetBinding,
    anchor_manifest_path: Path,
    base_logits_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    *,
    configuration_id: str,
    optimizer_step: int,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Measure the frozen 512-anchor next-token KL for one checkpoint."""

    try:
        import torch
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as error:
        raise RuntimeError("generalist-v3 anchor drift requires its runtime") from error
    checkpoint = _read_json(checkpoint_dir / "checkpoint.json")
    if (
        checkpoint.get("schema_version") != "generalist-v3-training-checkpoint-v1"
        or checkpoint.get("configuration_id") != configuration_id
        or checkpoint.get("optimizer_step") != optimizer_step
        or checkpoint.get("adapter_model_sha256")
        != sha256_file(checkpoint_dir / "adapter_model.safetensors")
    ):
        raise ValueError("generalist-v3 anchor drift checkpoint binding differs")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    runtime = load_training_runtime(config, model_snapshot=model_snapshot)  # type: ignore[arg-type]
    adapter_state = load_file(str(checkpoint_dir / "adapter_model.safetensors"))
    model_adapter_state = get_peft_model_state_dict(
        runtime.model, adapter_name="default"
    )
    missing_adapter_keys = sorted(set(model_adapter_state) - set(adapter_state))
    unexpected_adapter_keys = sorted(set(adapter_state) - set(model_adapter_state))
    shape_mismatches = sorted(
        key
        for key in set(model_adapter_state) & set(adapter_state)
        if tuple(model_adapter_state[key].shape) != tuple(adapter_state[key].shape)
    )
    if missing_adapter_keys or unexpected_adapter_keys or shape_mismatches:
        raise RuntimeError(
            "generalist-v3 checkpoint adapter state differs from the runtime: "
            f"missing_count={len(missing_adapter_keys)}, "
            f"unexpected_count={len(unexpected_adapter_keys)}, "
            f"shape_mismatch_count={len(shape_mismatches)}, "
            f"missing_sample={missing_adapter_keys[:5]}, "
            f"unexpected_sample={unexpected_adapter_keys[:5]}, "
            f"shape_mismatch_sample={shape_mismatches[:5]}"
        )
    load_result = set_peft_model_state_dict(
        runtime.model, adapter_state, adapter_name="default"
    )
    if load_result.unexpected_keys:
        raise RuntimeError(
            "generalist-v3 checkpoint adapter load had unexpected runtime keys: "
            f"count={len(load_result.unexpected_keys)}, "
            f"sample={sorted(load_result.unexpected_keys)[:5]}"
        )
    loaded_adapter_state = get_peft_model_state_dict(
        runtime.model, adapter_name="default"
    )
    value_mismatches = sorted(
        key
        for key in adapter_state
        if not torch.equal(
            adapter_state[key].detach().cpu(),
            loaded_adapter_state[key].detach().cpu(),
        )
    )
    if value_mismatches:
        raise RuntimeError(
            "generalist-v3 checkpoint adapter values did not load exactly: "
            f"count={len(value_mismatches)}, sample={value_mismatches[:5]}"
        )
    runtime.model.eval()
    anchor_manifest = _read_json(anchor_manifest_path)
    anchor_inputs = _resolve_anchor_inputs(binding, anchor_manifest)
    base_logits = load_file(str(base_logits_path))["base_next_token_logits"]
    causal_model = runtime.model.get_base_model()
    values = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, model_input in enumerate(anchor_inputs):
            input_ids = runtime.tokenizer.encode(
                model_input, add_special_tokens=False, return_tensors="pt"
            ).to(runtime.model.device)
            configure_gradient_checkpointing(causal_model, int(input_ids.shape[1]))
            with (
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                outputs = causal_model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    chunk_size=LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
                    use_cache=False,
                )
                current_logits = causal_model.lm_head(
                    outputs.last_hidden_state[:, -1, :]
                ).squeeze(0)
                kl = base_forward_kl(
                    base_logits[index].to(current_logits.device), current_logits
                )
            values.append(float(kl.item()))
            del outputs, current_logits, input_ids, kl
            if (index + 1) % 64 == 0:
                print(
                    json.dumps(
                        {"anchors_measured": index + 1, "total": len(anchor_inputs)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    ordered = sorted(values)
    value = {
        "schema_version": "generalist-v3-anchor-drift-v1",
        "status": "passed",
        "configuration_id": configuration_id,
        "optimizer_step": optimizer_step,
        "checkpoint_adapter_model_sha256": checkpoint["adapter_model_sha256"],
        "anchor_manifest_sha256": sha256_file(anchor_manifest_path),
        "base_logits_sha256": sha256_file(base_logits_path),
        "anchor_count": len(values),
        "mean_anchor_kl": sum(values) / len(values),
        "maximum_anchor_kl": max(values),
        "median_anchor_kl": ordered[len(ordered) // 2],
        "p95_anchor_kl": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "per_anchor_kl": values,
        "all_finite": all(math.isfinite(item) and item >= 0.0 for item in values),
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "wall_time_seconds": time.perf_counter() - started,
        },
        "sealed_test_accessed": False,
    }
    if not value["all_finite"]:
        raise RuntimeError("generalist-v3 checkpoint anchor drift is invalid")
    _write_json(output_path, value)
    del runtime, base_logits, adapter_state
    gc.collect()
    torch.cuda.empty_cache()
    return value
