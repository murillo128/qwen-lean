from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .dataset_v2 import sha256_file
from .generalist_v2 import (
    EXPECTED_LORA_MODULE_COUNTS,
    GENERALIST_SERIALIZATION_ID,
    LORA_TARGET_REGEX,
    LORA_TARGET_SUFFIXES,
    MODEL_ID,
    MODEL_REVISION,
    GeneralistProofVariant,
    GeneralistV2Config,
    WeightedTokenizedExample,
    compute_training_weights,
    deterministic_training_order,
    one_pass_membership_trajectory,
    render_generalist_prompt,
    serialization_length_evidence,
    tokenize_generalist_variant,
)
from .generalist_v2_dataset import (
    DATASET_BINDING_SCHEMA_VERSION,
    load_bound_training_variants,
)
from .phase3 import IGNORE_INDEX
from .prompt import normalize_transport
from .schema import TaskRecord
from .verifier import LeanVerifier

ARCHITECTURE_PREFLIGHT_SCHEMA_VERSION = "generalist-v2-architecture-preflight-v1"
RUNTIME_PREPARATION_SCHEMA_VERSION = "generalist-v2-runtime-preparation-v1"
PRODUCTION_PREFLIGHT_SCHEMA_VERSION = "generalist-v2-production-preflight-v1"
BOUNDED_TRAINING_SCHEMA_VERSION = "generalist-v2-bounded-training-v1"
MINIMUM_PRODUCTION_HEADROOM_BYTES = 512 * 1024**2
OVERFIT64_OPTIMIZER_STEPS = 600
GRADIENT_CHECKPOINTING_USE_REENTRANT = False
GRADIENT_CHECKPOINTING_MIN_SEQUENCE_TOKENS = 1024
LINEAR_ATTENTION_TRAINING_CHUNK_SIZE = 32
FULL_ATTENTION_SDPA_BACKEND = "FLASH_ATTENTION"
ACTIVATION_CPU_OFFLOAD = True
ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS = 4096
LM_HEAD_LOSS_CHUNK_TOKENS = 64
MLP_SEQUENCE_CHUNK_TOKENS = 1024
FLA_VERSION = "0.5.2"
FLA_UPSTREAM_TAG = "v0.5.2"
FLA_UPSTREAM_REVISION = "9c8e42e762fce087c27b673af4922795d9edb85e"
Q0_EVIDENCE_SCHEMA_VERSION = "generalist-v2-q0-evidence-v1"
Q0_EXPECTED_WORKLOADS = {
    "fresh-composition-valid-v2": 406,
    "minif2f-valid-clean-v2": 244,
    "dataset-v2-train-probe": 256,
    "riemann-fresh-valid-v2": 100,
}


@dataclass(frozen=True)
class LoraTargetMatch:
    path: str
    suffix: str
    family: str
    module_class: str
    input_features: int
    output_features: int
    lora_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "suffix": self.suffix,
            "family": self.family,
            "module_class": self.module_class,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "lora_parameter_count": self.lora_parameter_count,
        }


@dataclass
class GeneralistTrainingRuntime:
    model: Any
    tokenizer: Any
    lane: str
    target_matches: tuple[LoraTargetMatch, ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    total_parameter_count: int
    quantized_linear_module_count: int
    sequence_chunked_mlp_module_count: int = 0
    gated_delta_rule_backend: Mapping[str, Any] | None = None


def _target_family(path: str) -> str:
    if ".self_attn." in path:
        return "full_attention"
    if ".linear_attn." in path:
        return "gated_deltanet"
    if ".mlp." in path:
        return "mlp"
    raise ValueError(f"LoRA target has an unknown family: {path}")


def inspect_lora_targets(
    model: Any,
    *,
    rank: int = 16,
    target_regex: str = LORA_TARGET_REGEX,
    expected_counts: Mapping[str, int] = EXPECTED_LORA_MODULE_COUNTS,
) -> tuple[LoraTargetMatch, ...]:
    if rank != 16:
        raise ValueError("generalist-v2 LoRA rank must remain 16")
    pattern = re.compile(target_regex)
    matches: list[LoraTargetMatch] = []
    lookalikes: list[str] = []
    for path, module in model.named_modules():
        suffix = path.rsplit(".", 1)[-1]
        if suffix not in LORA_TARGET_SUFFIXES:
            continue
        if not pattern.fullmatch(path):
            lookalikes.append(path)
            continue
        input_features = getattr(module, "in_features", None)
        output_features = getattr(module, "out_features", None)
        if not isinstance(input_features, int) or not isinstance(output_features, int):
            raise TypeError(f"LoRA target is not a linear projection: {path}")
        matches.append(
            LoraTargetMatch(
                path=path,
                suffix=suffix,
                family=_target_family(path),
                module_class=type(module).__name__,
                input_features=input_features,
                output_features=output_features,
                lora_parameter_count=rank * (input_features + output_features),
            )
        )
    if lookalikes:
        raise RuntimeError(
            "target-like projections exist outside the text-decoder constraint: "
            + ", ".join(lookalikes[:5])
        )
    counts = Counter(item.suffix for item in matches)
    if dict(counts) != dict(expected_counts):
        raise RuntimeError(
            "pinned Qwen3.5 LoRA module counts differ: "
            f"observed={dict(counts)}, expected={dict(expected_counts)}"
        )
    families = {item.family for item in matches}
    if families != {"full_attention", "gated_deltanet", "mlp"}:
        raise RuntimeError(f"generalist-v2 LoRA families are incomplete: {families}")
    forbidden = ("vision", "visual", "embed", "norm", "lm_head")
    contaminated = [
        item.path for item in matches if any(token in item.path for token in forbidden)
    ]
    if contaminated:
        raise RuntimeError(f"forbidden LoRA targets matched: {contaminated[:5]}")
    return tuple(sorted(matches, key=lambda item: item.path))


def lora_target_summary(matches: Sequence[LoraTargetMatch]) -> dict[str, Any]:
    if not matches:
        raise ValueError("cannot summarize no LoRA target matches")
    return {
        "matched_module_count": len(matches),
        "module_counts_by_suffix": dict(
            sorted(Counter(item.suffix for item in matches).items())
        ),
        "module_counts_by_family": dict(
            sorted(Counter(item.family for item in matches).items())
        ),
        "trainable_lora_parameter_count": sum(
            item.lora_parameter_count for item in matches
        ),
        "matched_modules": [item.path for item in matches],
        "projection_shapes_by_suffix": {
            suffix: sorted(
                {
                    (item.input_features, item.output_features)
                    for item in matches
                    if item.suffix == suffix
                }
            )
            for suffix in sorted({item.suffix for item in matches})
        },
        "vision_modules_matched": 0,
        "embedding_modules_matched": 0,
        "normalization_modules_matched": 0,
        "lm_head_modules_matched": 0,
    }


def choose_precision_lane(
    config: GeneralistV2Config, *, device_total_memory_bytes: int
) -> str:
    config.validate()
    if device_total_memory_bytes <= 0:
        raise ValueError("CUDA device memory must be positive")
    preferred = config.precision["preferred"]
    if device_total_memory_bytes >= int(preferred["minimum_vram_bytes"]):
        return str(preferred["lane"])
    return str(config.precision["fallback"]["lane"])


def _require_local_cuda() -> tuple[Any, int, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 requires its isolated training runtime"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("generalist-v2 requires a project-controlled local CUDA GPU")
    device_index = torch.cuda.current_device()
    return torch, device_index, torch.cuda.get_device_properties(device_index)


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _model_source(model_snapshot: Path | None) -> tuple[str, dict[str, Any]]:
    if model_snapshot is None:
        return MODEL_ID, {"revision": MODEL_REVISION, "local_files_only": True}
    snapshot = model_snapshot.resolve()
    if snapshot.name != MODEL_REVISION:
        raise ValueError(
            "local Qwen3.5 snapshot directory must be the pinned revision "
            f"{MODEL_REVISION}"
        )
    required = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise ValueError(f"pinned Qwen3.5 snapshot is incomplete: {missing}")
    return str(snapshot), {"local_files_only": True}


def run_architecture_load_preflight(
    config: GeneralistV2Config,
    output: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    torch, device_index, properties = _require_local_cuda()
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 architecture smoke needs Transformers"
        ) from error
    source, source_kwargs = _model_source(model_snapshot)
    torch.cuda.reset_peak_memory_stats(device_index)
    loaded_config = AutoConfig.from_pretrained(
        source, trust_remote_code=False, **source_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=False, **source_kwargs
    )
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=False,
        **source_kwargs,
    )
    if type(loaded_config).__name__ != "Qwen3_5Config":
        raise RuntimeError("pinned model did not load Qwen3_5Config")
    if type(model).__name__ != config.model["architecture_class"]:
        raise RuntimeError("pinned model did not load the text-only causal LM class")
    if type(model.config).__name__ != "Qwen3_5TextConfig" or (
        getattr(model.config, "model_type", None) != "qwen3_5_text"
    ):
        raise RuntimeError("pinned text model config differs from qwen3_5_text")
    if any("vision" in path or "visual" in path for path, _ in model.named_modules()):
        raise RuntimeError(
            "text-only Qwen3.5 model unexpectedly contains vision modules"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matches = inspect_lora_targets(
        model,
        rank=int(config.lora["r"]),
        target_regex=str(config.lora["target_regex"]),
        expected_counts=config.lora["expected_module_counts"],
    )
    model.to(torch.device("cuda", device_index))
    model.eval()
    prompt = (
        "import Mathlib\n\n/- Complete the proof below.\n"
        "Return only Lean code continuing after `by`; do not use `sorry` or `admit`. -/\n"
        "theorem generalist_v2_architecture_smoke : True := by\n  "
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        raise RuntimeError("pinned tokenizer produced an empty smoke prompt")
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device_index)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=tensor, use_cache=False).logits
    if logits.shape[:2] != tensor.shape:
        raise RuntimeError("Qwen3.5 forward smoke returned an invalid logits shape")
    if not bool(torch.isfinite(logits[:, -1, :]).all().item()):
        raise RuntimeError("Qwen3.5 forward smoke produced non-finite logits")
    torch.cuda.synchronize(device_index)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = lora_target_summary(matches)
    value = {
        "schema_version": ARCHITECTURE_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "config_class": type(loaded_config).__name__,
            "source_model_type": loaded_config.model_type,
            "architecture_class": type(model).__name__,
            "text_config_class": type(model.config).__name__,
            "model_type": model.config.model_type,
            "text_only": True,
            "total_parameter_count": total_parameter_count,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "add_special_tokens": False,
            "chat_template_applied": False,
        },
        "serialization_id": GENERALIST_SERIALIZATION_ID,
        "lora": {
            "target_regex": config.lora["target_regex"],
            "rank": config.lora["r"],
            **summary,
        },
        "forward_smoke": {
            "execution": "project-controlled-local-cuda",
            "input_tokens": len(input_ids),
            "output_shape": list(logits.shape),
            "last_token_logits_finite": True,
            "base_parameters_frozen": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "cuda_device_index": device_index,
            "cuda_device": properties.name,
            "cuda_device_capability": [properties.major, properties.minor],
            "cuda_device_total_memory_bytes": properties.total_memory,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
            "torch_cuda_version": torch.version.cuda,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "packages": _package_versions(
                ("torch", "transformers", "huggingface-hub", "safetensors")
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del logits, tensor, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _validate_trainables(
    model: Any, matches: Sequence[LoraTargetMatch]
) -> tuple[tuple[str, ...], int, int]:
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("generalist-v2 attached no trainable LoRA parameters")
    invalid = [
        name for name in trainable if ".lora_A." not in name and ".lora_B." not in name
    ]
    forbidden = [
        name
        for name in trainable
        if any(
            token in name for token in ("vision", "visual", "embed", "norm", "lm_head")
        )
    ]
    target_suffixes = {match.suffix for match in matches}
    unexpected_target = [
        name
        for name in trainable
        if not any(f".{suffix}." in name for suffix in target_suffixes)
    ]
    if invalid or forbidden or unexpected_target:
        raise RuntimeError(
            "generalist-v2 trainable parameter boundary failed: "
            f"non_lora={invalid[:3]}, forbidden={forbidden[:3]}, "
            f"unexpected_target={unexpected_target[:3]}"
        )
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_count = sum(match.lora_parameter_count for match in matches)
    if trainable_count != expected_count:
        raise RuntimeError(
            "generalist-v2 trainable parameter count differs from matched LoRA shape: "
            f"{trainable_count} != {expected_count}"
        )
    total_count = sum(parameter.numel() for parameter in model.parameters())
    return trainable, trainable_count, total_count


def enable_sequence_chunked_mlp(model: Any, *, chunk_tokens: int) -> int:
    """Checkpoint long token-independent Qwen MLPs in bounded sequence chunks."""
    if chunk_tokens < 1:
        raise ValueError("generalist-v2 MLP sequence chunk must be positive")
    try:
        import torch
        from torch.utils.checkpoint import checkpoint
    except ImportError as error:
        raise RuntimeError("generalist-v2 MLP chunking requires PyTorch") from error
    matched = 0
    for path, module in model.named_modules():
        if not (
            path.endswith(".mlp")
            and "model.layers." in path
            and all(
                hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
            )
        ):
            continue
        original_forward = module.forward

        def chunked_forward(x: Any, _forward: Any = original_forward) -> Any:
            if x.ndim != 3 or x.shape[1] <= chunk_tokens:
                return _forward(x)
            return torch.cat(
                [
                    checkpoint(
                        _forward,
                        x[:, start : start + chunk_tokens, :],
                        use_reentrant=True,
                    )
                    for start in range(0, x.shape[1], chunk_tokens)
                ],
                dim=1,
            )

        module.forward = chunked_forward
        matched += 1
    if matched != 32:
        raise RuntimeError(
            f"generalist-v2 expected to sequence-chunk 32 text MLPs, found {matched}"
        )
    return matched


def inspect_gated_delta_rule_backend() -> dict[str, Any]:
    """Require the pinned local training kernel instead of the slow torch loop."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
        )
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 cannot inspect the DeltaNet backend"
        ) from error
    implementation = inspect.getclosurevars(torch_chunk_gated_delta_rule).nonlocals.get(
        "implementation"
    )
    module = getattr(implementation, "__module__", "")
    name = getattr(implementation, "__name__", "")
    version = importlib.metadata.version("flash-linear-attention")
    if (
        module != "fla.ops.gated_delta_rule.chunk"
        or name != "chunk_gated_delta_rule"
        or version != FLA_VERSION
    ):
        raise RuntimeError(
            "generalist-v2 requires the pinned FLA DeltaNet training kernel: "
            f"module={module}, name={name}, version={version}"
        )
    return {
        "implementation_module": module,
        "implementation_name": name,
        "distribution": "flash-linear-attention",
        "distribution_version": version,
        "upstream_tag": FLA_UPSTREAM_TAG,
        "upstream_revision": FLA_UPSTREAM_REVISION,
        "license": "MIT",
        "execution": "project-controlled-local-cuda",
    }


def load_training_runtime(
    config: GeneralistV2Config,
    *,
    model_snapshot: Path | None = None,
) -> GeneralistTrainingRuntime:
    config.validate()
    torch, device_index, properties = _require_local_cuda()
    try:
        import bitsandbytes as bnb
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 training needs its isolated Transformers/PEFT/TRL runtime"
        ) from error
    lane = choose_precision_lane(
        config, device_total_memory_bytes=properties.total_memory
    )
    source, source_kwargs = _model_source(model_snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=False, **source_kwargs
    )
    tokenizer.padding_side = "right"
    if tokenizer.eos_token_id is None:
        raise RuntimeError("pinned Qwen3.5 tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    common = {
        "dtype": torch.bfloat16,
        "device_map": {"": device_index},
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        **source_kwargs,
    }
    if lane == "bf16-lora":
        model = AutoModelForCausalLM.from_pretrained(source, **common)
        quantized_linear_count = 0
    elif lane == "nf4-qlora":
        fallback = config.precision["fallback"]
        quantization = BitsAndBytesConfig(
            load_in_4bit=bool(fallback["load_in_4bit"]),
            bnb_4bit_quant_type=str(fallback["quantization_type"]),
            bnb_4bit_use_double_quant=bool(fallback["double_quantization"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            source, quantization_config=quantization, **common
        )
        if not bool(getattr(model, "is_loaded_in_4bit", False)):
            raise RuntimeError("generalist-v2 QLoRA base did not load in 4-bit")
        quantized_linear_count = sum(
            isinstance(module, bnb.nn.Linear4bit) for module in model.modules()
        )
        if quantized_linear_count == 0:
            raise RuntimeError("generalist-v2 found no 4-bit linear modules")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        raise RuntimeError(f"unknown generalist-v2 precision lane: {lane}")
    if type(model).__name__ != config.model["architecture_class"]:
        raise RuntimeError(
            "generalist-v2 runtime did not load the text-only model class"
        )
    model.config.use_cache = False
    matches = inspect_lora_targets(
        model,
        rank=int(config.lora["r"]),
        target_regex=str(config.lora["target_regex"]),
        expected_counts=config.lora["expected_module_counts"],
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.lora["r"]),
        lora_alpha=int(config.lora["lora_alpha"]),
        lora_dropout=float(config.lora["lora_dropout"]),
        bias=str(config.lora["bias"]),
        target_modules=str(config.lora["target_regex"]),
        modules_to_save=None,
        revision=MODEL_REVISION,
    )
    model = get_peft_model(model, lora_config)
    model.peft_config["default"].base_model_name_or_path = MODEL_ID
    model.peft_config["default"].revision = MODEL_REVISION
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT
        }
    )
    model.enable_input_require_grads()
    if not bool(getattr(model, "is_gradient_checkpointing", False)):
        raise RuntimeError("generalist-v2 gradient checkpointing did not enable")
    sequence_chunked_mlp_module_count = enable_sequence_chunked_mlp(
        model, chunk_tokens=MLP_SEQUENCE_CHUNK_TOKENS
    )
    gated_delta_rule_backend = inspect_gated_delta_rule_backend()
    trainable, trainable_count, total_count = _validate_trainables(model, matches)
    return GeneralistTrainingRuntime(
        model=model,
        tokenizer=tokenizer,
        lane=lane,
        target_matches=matches,
        trainable_parameter_names=trainable,
        trainable_parameter_count=trainable_count,
        total_parameter_count=total_count,
        quantized_linear_module_count=int(quantized_linear_count),
        sequence_chunked_mlp_module_count=sequence_chunked_mlp_module_count,
        gated_delta_rule_backend=gated_delta_rule_backend,
    )


def run_runtime_preparation_smoke(
    config: GeneralistV2Config,
    output: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    torch, device_index, properties = _require_local_cuda()
    torch.cuda.reset_peak_memory_stats(device_index)
    runtime = load_training_runtime(config, model_snapshot=model_snapshot)
    target_summary = lora_target_summary(runtime.target_matches)
    preferred_minimum = int(config.precision["preferred"]["minimum_vram_bytes"])
    value = {
        "schema_version": RUNTIME_PREPARATION_SCHEMA_VERSION,
        "status": "passed",
        "model": config.model,
        "selected_lane": runtime.lane,
        "lane_resolution": {
            "preferred_lane": config.precision["preferred"]["lane"],
            "preferred_minimum_vram_bytes": preferred_minimum,
            "device_total_memory_bytes": properties.total_memory,
            "fallback_lane": config.precision["fallback"]["lane"],
            "fallback_selected_because_device_below_preferred_minimum": (
                properties.total_memory < preferred_minimum
            ),
        },
        "quantization": (
            config.precision["fallback"]
            if runtime.lane == config.precision["fallback"]["lane"]
            else None
        ),
        "lora": {
            "r": config.lora["r"],
            "lora_alpha": config.lora["lora_alpha"],
            "lora_dropout": config.lora["lora_dropout"],
            "bias": config.lora["bias"],
            "target_regex": config.lora["target_regex"],
            **target_summary,
        },
        "trainable_boundary": {
            "trainable_parameter_tensors": len(runtime.trainable_parameter_names),
            "trainable_parameter_count": runtime.trainable_parameter_count,
            "all_trainables_are_intended_lora_a_or_b": True,
            "vision_trainables": 0,
            "embedding_trainables": 0,
            "normalization_trainables": 0,
            "lm_head_trainables": 0,
        },
        "quantized_linear_module_count": runtime.quantized_linear_module_count,
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "python": platform.python_version(),
            "cuda_device_index": device_index,
            "cuda_device": properties.name,
            "cuda_device_capability": [properties.major, properties.minor],
            "cuda_device_total_memory_bytes": properties.total_memory,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
            "torch_cuda_version": torch.version.cuda,
            "gradient_checkpointing_enabled": bool(
                runtime.model.is_gradient_checkpointing
            ),
            "gradient_checkpointing_use_reentrant": (
                GRADIENT_CHECKPOINTING_USE_REENTRANT
            ),
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "mlp_sequence_chunk_tokens": MLP_SEQUENCE_CHUNK_TOKENS,
            "sequence_chunked_mlp_module_count": (
                runtime.sequence_chunked_mlp_module_count
            ),
            "gated_delta_rule_backend": runtime.gated_delta_rule_backend,
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
        "scope_limit": {
            "dataset_v2_rows_used": 0,
            "forward_backward_update_run": False,
            "production_context_fit_claimed": False,
            "next_required_input": "accepted and merged Dataset-v2 package from issue #56",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    return value


def pad_weighted_target_only_batch(
    features: Sequence[Mapping[str, Any]], *, pad_token_id: int
) -> dict[str, list[Any]]:
    if not features:
        raise ValueError("cannot collate an empty generalist-v2 batch")
    maximum = max(len(feature["input_ids"]) for feature in features)
    batch: dict[str, list[Any]] = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "example_weight": [],
    }
    for feature in features:
        input_ids = [int(item) for item in feature["input_ids"]]
        labels = [int(item) for item in feature["labels"]]
        attention_mask = [int(item) for item in feature["attention_mask"]]
        if not (len(input_ids) == len(labels) == len(attention_mask)):
            raise ValueError("generalist-v2 batch feature lengths differ")
        weight = float(feature["example_weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("generalist-v2 batch has an invalid example weight")
        padding = maximum - len(input_ids)
        batch["input_ids"].append(input_ids + [pad_token_id] * padding)
        batch["labels"].append(labels + [IGNORE_INDEX] * padding)
        batch["attention_mask"].append(attention_mask + [0] * padding)
        batch["example_weight"].append(weight)
    return batch


class WeightedTargetOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("generalist-v2 collator requires PyTorch") from error
        batch = pad_weighted_target_only_batch(features, pad_token_id=self.pad_token_id)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "example_weight": torch.tensor(
                batch["example_weight"], dtype=torch.float32
            ),
        }


def statement_weighted_causal_loss(
    logits: Any,
    labels: Any,
    example_weights: Any,
    *,
    weight_normalizer: float,
) -> Any:
    try:
        import torch
        from torch.nn import functional
    except ImportError as error:
        raise RuntimeError("generalist-v2 weighted loss requires PyTorch") from error
    if not math.isfinite(weight_normalizer) or weight_normalizer <= 0:
        raise ValueError("weighted loss normalizer must be finite and positive")
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    flat_loss = functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(shifted_labels.shape)
    target_mask = shifted_labels.ne(IGNORE_INDEX)
    target_counts = target_mask.sum(dim=1)
    if bool(target_counts.eq(0).any().item()):
        raise RuntimeError("generalist-v2 batch has an example with no target tokens")
    per_example = (flat_loss * target_mask).sum(dim=1) / target_counts
    weights = example_weights.to(per_example.device, dtype=per_example.dtype)
    if weights.ndim != 1 or weights.shape[0] != per_example.shape[0]:
        raise ValueError("generalist-v2 example weights do not align with the batch")
    if not bool(torch.isfinite(weights).all().item()) or bool(
        weights.le(0).any().item()
    ):
        raise ValueError("generalist-v2 example weights are not finite and positive")
    return (per_example * weights).mean() / weight_normalizer


def scale_single_example_causal_loss(
    loss: Any,
    example_weights: Any,
    *,
    weight_normalizer: float,
) -> Any:
    """Scale the native target-token mean loss for the fixed micro-batch of one."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("generalist-v2 weighted loss requires PyTorch") from error
    if not math.isfinite(weight_normalizer) or weight_normalizer <= 0:
        raise ValueError("weighted loss normalizer must be finite and positive")
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("generalist-v2 native causal loss is not a finite scalar")
    weights = example_weights.to(loss.device, dtype=loss.dtype)
    if weights.ndim != 1 or weights.shape[0] != 1:
        raise ValueError("generalist-v2 native weighted loss requires micro-batch one")
    if not bool(torch.isfinite(weights).all().item()) or bool(
        weights.le(0).any().item()
    ):
        raise ValueError("generalist-v2 example weight is not finite and positive")
    return loss * weights[0] / weight_normalizer


def checkpointed_target_only_causal_loss(
    causal_model: Any,
    hidden_states: Any,
    labels: Any,
    *,
    chunk_tokens: int = LM_HEAD_LOSS_CHUNK_TOKENS,
) -> Any:
    """Compute exact target-only causal CE without retaining vocabulary logits."""
    try:
        import torch
        from torch.nn import functional
        from torch.utils.checkpoint import checkpoint
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 checkpointed causal loss requires PyTorch"
        ) from error
    if chunk_tokens < 1:
        raise ValueError("generalist-v2 LM-head loss chunk must be positive")
    if labels.ndim != 2 or hidden_states.ndim != 3:
        raise ValueError("generalist-v2 causal loss received invalid tensor ranks")
    if labels.shape != hidden_states.shape[:2]:
        raise ValueError("generalist-v2 labels and hidden states do not align")
    shifted_labels = labels[:, 1:].contiguous()
    target_mask = shifted_labels.ne(IGNORE_INDEX)
    target_count = int(target_mask.sum().item())
    if target_count < 1:
        raise RuntimeError("generalist-v2 example has no supervised target tokens")
    target_hidden = hidden_states[:, :-1, :][target_mask]
    target_labels = shifted_labels[target_mask]

    def chunk_loss(hidden: Any, targets: Any) -> Any:
        logits = causal_model.lm_head(hidden).float()
        return functional.cross_entropy(logits, targets, reduction="sum")

    loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, target_count, chunk_tokens):
        end = min(start + chunk_tokens, target_count)
        loss_sum = loss_sum + checkpoint(
            chunk_loss,
            target_hidden[start:end],
            target_labels[start:end],
            use_reentrant=True,
        )
    return loss_sum / target_count


def should_offload_activations(sequence_tokens: int) -> bool:
    if sequence_tokens < 1:
        raise ValueError("generalist-v2 sequence length must be positive")
    return (
        ACTIVATION_CPU_OFFLOAD
        and sequence_tokens >= ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS
    )


def should_checkpoint_activations(sequence_tokens: int) -> bool:
    if sequence_tokens < 1:
        raise ValueError("generalist-v2 sequence length must be positive")
    return sequence_tokens >= GRADIENT_CHECKPOINTING_MIN_SEQUENCE_TOKENS


def configure_gradient_checkpointing(model: Any, sequence_tokens: int) -> bool:
    required = should_checkpoint_activations(sequence_tokens)
    enabled = bool(getattr(model, "is_gradient_checkpointing", False))
    if required and not enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT
            }
        )
    elif not required and enabled:
        model.gradient_checkpointing_disable()
    observed = bool(getattr(model, "is_gradient_checkpointing", False))
    if observed != required:
        raise RuntimeError("generalist-v2 could not set sequence-aware checkpointing")
    return observed


def build_weighted_sft_trainer(
    runtime: GeneralistTrainingRuntime,
    examples: Sequence[WeightedTokenizedExample],
    config: GeneralistV2Config,
    output_dir: Path,
    *,
    maximum_sequence_tokens: int,
    save_quarter_checkpoints: bool = True,
    weight_normalizer: float | None = None,
    maximum_optimizer_steps: int | None = None,
) -> Any:
    config.validate()
    if maximum_sequence_tokens not in tuple(config.training["context_choices"]):
        raise ValueError("generalist-v2 trainer received an unsupported context")
    if not examples:
        raise ValueError("generalist-v2 trainer received no examples")
    for example in examples:
        example.validate(int(runtime.tokenizer.eos_token_id), maximum_sequence_tokens)
    membership = [(item.statement_id, item.proof_variant_id) for item in examples]
    expected_order = sorted(
        membership,
        key=lambda item: hashlib.sha256(
            (f"generalist-v2-one-pass-v1\0{0}\0{item[0]}\0{item[1]}").encode()
        ).digest(),
    )
    if membership != expected_order:
        raise ValueError("generalist-v2 trainer rows are not in the frozen hash order")
    try:
        import torch
        from datasets import Dataset
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 trainer dependencies are unavailable"
        ) from error

    resolved_weight_normalizer = (
        fmean(item.example_weight for item in examples)
        if weight_normalizer is None
        else float(weight_normalizer)
    )
    if not math.isfinite(resolved_weight_normalizer) or resolved_weight_normalizer <= 0:
        raise ValueError("generalist-v2 trainer weight normalizer is invalid")
    if maximum_optimizer_steps is not None and maximum_optimizer_steps < 1:
        raise ValueError("generalist-v2 maximum optimizer steps must be positive")
    boundaries: set[int] = set()
    if save_quarter_checkpoints:
        trajectory = one_pass_membership_trajectory(
            membership,
            membership_is_ordered=True,
        )
        boundaries = set(trajectory["checkpoint_optimizer_steps"].values())

    class QuarterCheckpointCallback(TrainerCallback):
        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if int(state.global_step) in boundaries:
                control.should_save = True
            return control

    class StatementWeightedSFTTrainer(SFTTrainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            weights = inputs.pop("example_weight")
            labels = inputs.pop("labels")
            causal_model = (
                model.get_base_model() if hasattr(model, "get_base_model") else model
            )
            sequence_tokens = int(inputs["input_ids"].shape[1])
            configure_gradient_checkpointing(causal_model, sequence_tokens)
            activation_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if should_offload_activations(sequence_tokens)
                else nullcontext()
            )
            with (
                activation_context,
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                outputs = causal_model.model(
                    **inputs,
                    chunk_size=LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
                    use_cache=False,
                )
                base_loss = checkpointed_target_only_causal_loss(
                    causal_model,
                    outputs.last_hidden_state,
                    labels,
                )
            loss = scale_single_example_causal_loss(
                base_loss,
                weights,
                weight_normalizer=resolved_weight_normalizer,
            )
            return (loss, outputs) if return_outputs else loss

    class FiniteOptimizationCallback(TrainerCallback):
        def on_pre_optimizer_step(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            trainable = [
                parameter
                for parameter in kwargs["model"].parameters()
                if parameter.requires_grad
            ]
            if not trainable or any(parameter.grad is None for parameter in trainable):
                raise RuntimeError(
                    "generalist-v2 has missing or non-finite gradients before "
                    f"optimizer step {state.global_step + 1}"
                )
            if any(
                not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in trainable
            ):
                raise RuntimeError(
                    "generalist-v2 has missing or non-finite gradients before "
                    f"optimizer step {state.global_step + 1}"
                )
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Any = None,
            **kwargs: Any,
        ) -> Any:
            if logs is not None:
                for key in ("loss", "grad_norm"):
                    if key in logs and not math.isfinite(float(logs[key])):
                        raise RuntimeError(
                            f"generalist-v2 logged non-finite {key} at "
                            f"optimizer step {state.global_step}"
                        )
            return control

    arguments = SFTConfig(
        output_dir=str(output_dir),
        do_train=True,
        eval_strategy="no",
        per_device_train_batch_size=int(config.training["per_device_micro_batch_size"]),
        gradient_accumulation_steps=int(config.training["gradient_accumulation_steps"]),
        learning_rate=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
        max_grad_norm=float(config.training["maximum_gradient_norm"]),
        num_train_epochs=1.0,
        max_steps=(-1 if maximum_optimizer_steps is None else maximum_optimizer_steps),
        lr_scheduler_type=str(config.training["lr_schedule"]),
        # Transformers 5 accepts a fractional warmup through warmup_steps when
        # the value is below one and resolves it against the final step count.
        warmup_steps=float(config.training["warmup_fraction"]),
        optim=str(config.training["optimizer"]),
        seed=int(config.training["seed"]),
        data_seed=int(config.training["data_seed"]),
        train_sampling_strategy="sequential",
        bf16=True,
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="no",
        save_only_model=False,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT
        },
        max_length=maximum_sequence_tokens,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False,
    )
    dataset = Dataset.from_list([item.to_trainer_row() for item in examples])
    return StatementWeightedSFTTrainer(
        model=runtime.model,
        args=arguments,
        train_dataset=dataset,
        data_collator=WeightedTargetOnlyCollator(int(runtime.tokenizer.pad_token_id)),
        processing_class=runtime.tokenizer,
        callbacks=[
            FiniteOptimizationCallback(),
            *([QuarterCheckpointCallback()] if boundaries else []),
        ],
    )


def resolve_near_maximum_variant(
    records: Sequence[GeneralistProofVariant], binding: Mapping[str, Any]
) -> tuple[GeneralistProofVariant, int, int]:
    if binding.get("schema_version") != DATASET_BINDING_SCHEMA_VERSION:
        raise ValueError("generalist-v2 production preflight needs Dataset-v2 binding")
    try:
        lengths = binding["serialization"]["lengths"]
        maximum = lengths["maximum_variant"]
        context_tokens = int(lengths["selected_context_tokens"])
        expected_tokens = int(maximum["tokens"])
        statement_id = str(maximum["statement_id"])
        proof_variant_id = str(maximum["proof_variant_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Dataset-v2 binding lacks a valid maximum variant") from error
    matches = [
        record
        for record in records
        if record.statement_id == statement_id
        and record.proof_variant_id == proof_variant_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "Dataset-v2 maximum variant does not resolve exactly once in training"
        )
    return matches[0], expected_tokens, context_tokens


def select_overfit64_variants(
    records: Sequence[GeneralistProofVariant],
    train_probe: Mapping[str, Any],
) -> tuple[list[GeneralistProofVariant], dict[str, Any]]:
    expected_strata = (
        "real-generic",
        "real-prime-number-theory",
        "synthetic-generic-composition",
        "synthetic-prime-composition",
    )
    if train_probe.get("id") != "dataset-v2-train-probe":
        raise ValueError("overfit64 needs the canonical Dataset-v2 train probe")
    strata = train_probe.get("strata")
    if not isinstance(strata, Mapping) or tuple(sorted(strata)) != tuple(
        sorted(expected_strata)
    ):
        raise ValueError("Dataset-v2 train probe strata differ")
    by_statement: dict[str, list[GeneralistProofVariant]] = {}
    for record in records:
        by_statement.setdefault(record.statement_id, []).append(record)

    selected_by_stratum: dict[str, list[str]] = {}
    for stratum in expected_strata:
        statement_ids = [str(item) for item in strata[stratum]]
        if len(statement_ids) != 64 or len(set(statement_ids)) != 64:
            raise ValueError(f"Dataset-v2 train probe stratum is invalid: {stratum}")
        missing = set(statement_ids) - set(by_statement)
        if missing:
            raise ValueError(f"Dataset-v2 train probe has non-training rows: {stratum}")
        selected_by_stratum[stratum] = sorted(
            statement_ids,
            key=lambda statement_id: hashlib.sha256(
                (f"generalist-v2-overfit64-v1\0{stratum}\0{statement_id}").encode()
            ).digest(),
        )[:16]

    selected_statement_ids = [
        statement_id
        for stratum in expected_strata
        for statement_id in selected_by_stratum[stratum]
    ]
    if len(selected_statement_ids) != 64 or len(set(selected_statement_ids)) != 64:
        raise ValueError("overfit64 selection does not contain 64 unique statements")
    selected = [
        variant
        for statement_id in selected_statement_ids
        for variant in by_statement[statement_id]
    ]
    ordered = deterministic_training_order(selected)
    selected_variant_ids = {item.proof_variant_id for item in ordered}
    expected_variant_ids = {
        item.proof_variant_id
        for statement_id in selected_statement_ids
        for item in by_statement[statement_id]
    }
    if selected_variant_ids != expected_variant_ids:
        raise RuntimeError("overfit64 omitted a selected statement proof variant")
    statement_digest = hashlib.sha256(
        json.dumps(selected_statement_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    variant_digest = hashlib.sha256(
        json.dumps(
            [item.proof_variant_id for item in ordered], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return ordered, {
        "workload_id": "generalist-v2-overfit64-v1",
        "selection_rule": (
            "16 hash-ordered statements from each canonical Dataset-v2 train-probe "
            "stratum; include every selected statement proof variant"
        ),
        "statement_count": len(selected_statement_ids),
        "proof_variant_count": len(ordered),
        "statements_by_stratum": {
            stratum: len(selected_by_stratum[stratum]) for stratum in expected_strata
        },
        "ordered_statement_ids_sha256": statement_digest,
        "ordered_proof_variant_ids_sha256": variant_digest,
        "all_selected_statement_variants_included": True,
    }


def select_smoke4096_variants(
    records: Sequence[GeneralistProofVariant],
) -> tuple[list[GeneralistProofVariant], dict[str, Any]]:
    by_statement: dict[str, list[GeneralistProofVariant]] = {}
    for record in records:
        by_statement.setdefault(record.statement_id, []).append(record)
    if len(by_statement) < 4096:
        raise ValueError("generalist-v2 smoke needs at least 4,096 statements")
    selected_statement_ids = sorted(
        by_statement,
        key=lambda statement_id: hashlib.sha256(
            f"generalist-v2-smoke4096-v1\0{statement_id}".encode()
        ).digest(),
    )[:4096]
    selected = [
        variant
        for statement_id in selected_statement_ids
        for variant in by_statement[statement_id]
    ]
    ordered = deterministic_training_order(selected)
    selected_source_counts = Counter(
        by_statement[statement_id][0].source_kind
        for statement_id in selected_statement_ids
    )
    if set(selected_source_counts) != {"real", "synthetic"}:
        raise RuntimeError("generalist-v2 smoke did not retain both source kinds")
    expected_variant_ids = {
        item.proof_variant_id
        for statement_id in selected_statement_ids
        for item in by_statement[statement_id]
    }
    if {item.proof_variant_id for item in ordered} != expected_variant_ids:
        raise RuntimeError("generalist-v2 smoke omitted a selected proof variant")
    return ordered, {
        "workload_id": "generalist-v2-smoke4096-v1",
        "selection_rule": (
            "first 4,096 statement IDs under the fixed smoke hash order; include "
            "every selected statement proof variant"
        ),
        "statement_count": len(selected_statement_ids),
        "proof_variant_count": len(ordered),
        "statement_source_counts": dict(sorted(selected_source_counts.items())),
        "ordered_statement_ids_sha256": hashlib.sha256(
            json.dumps(selected_statement_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "ordered_proof_variant_ids_sha256": hashlib.sha256(
            json.dumps(
                [item.proof_variant_id for item in ordered], separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "all_selected_statement_variants_included": True,
    }


def tokenize_weighted_training_selection(
    all_records: Sequence[GeneralistProofVariant],
    selected_records: Sequence[GeneralistProofVariant],
    tokenizer: Any,
    *,
    maximum_sequence_tokens: int,
) -> tuple[list[WeightedTokenizedExample], float, dict[str, Any]]:
    if not selected_records:
        raise ValueError("generalist-v2 selected training workload is empty")
    all_variant_ids = {item.proof_variant_id for item in all_records}
    selected_variant_ids = [item.proof_variant_id for item in selected_records]
    if len(set(selected_variant_ids)) != len(selected_variant_ids):
        raise ValueError("generalist-v2 selected training workload repeats a variant")
    if not set(selected_variant_ids).issubset(all_variant_ids):
        raise ValueError("generalist-v2 selected workload is not a training subset")
    weights = compute_training_weights(all_records)
    normalizer = fmean(weights.variant_weights.values())
    examples = [
        tokenize_generalist_variant(
            record,
            tokenizer,
            example_weight=weights.variant_weights[record.proof_variant_id],
            maximum_sequence_tokens=maximum_sequence_tokens,
        )
        for record in selected_records
    ]
    observed_membership = [item.proof_variant_id for item in examples]
    if observed_membership != selected_variant_ids:
        raise RuntimeError(
            "generalist-v2 tokenization changed selected membership/order"
        )
    selected_weights = [item.example_weight for item in examples]
    return (
        examples,
        normalizer,
        {
            "selected_proof_variant_count": len(examples),
            "selected_proof_variants_unique": len(set(observed_membership)),
            "full_membership_weight_normalizer": normalizer,
            "selected_example_weight": {
                "minimum": min(selected_weights),
                "maximum": max(selected_weights),
                "mean": fmean(selected_weights),
            },
            "serialization": serialization_length_evidence(examples),
            "truncated_or_dropped_variants": 0,
        },
    )


def _parameter_names_sha256(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_q0_training_gate(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema_version") != Q0_EVIDENCE_SCHEMA_VERSION
        or evidence.get("checkpoint_id") != "Q0"
        or evidence.get("model_id") != MODEL_ID
        or evidence.get("model_revision") != MODEL_REVISION
        or evidence.get("candidates_per_task") != 8
        or evidence.get("selection_test_workloads_consulted") is not False
        or evidence.get("riemann_used_for_selection") is not False
    ):
        raise ValueError("generalist-v2 optimizer update requires pinned complete Q0")
    workloads = evidence.get("workloads")
    if not isinstance(workloads, Mapping) or set(workloads) != set(
        Q0_EXPECTED_WORKLOADS
    ):
        raise ValueError("generalist-v2 Q0 evidence has incomplete workloads")
    for workload_id, task_count in Q0_EXPECTED_WORKLOADS.items():
        workload = workloads[workload_id]
        categories = workload.get("category_counts", {})
        if (
            workload.get("task_count") != task_count
            or workload.get("candidate_count") != task_count * 8
            or categories.get("generation_error") != 0
            or categories.get("verifier_error") != 0
            or len(workload.get("verified_counts", ())) != task_count
        ):
            raise ValueError(f"generalist-v2 Q0 workload is incomplete: {workload_id}")
    return {
        "schema_version": evidence["schema_version"],
        "checkpoint_id": evidence["checkpoint_id"],
        "model_revision": evidence["model_revision"],
        "workload_count": len(workloads),
        "evidence_sha256": sha256_file(path),
        "complete_before_optimizer_update": True,
    }


def validate_production_preflight_gate(
    path: Path, q0_gate: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    update = evidence.get("update", {})
    runtime = evidence.get("runtime", {})
    binding_manifest_sha256 = evidence.get("dataset", {}).get("binding_manifest_sha256")
    if (
        evidence.get("schema_version") != PRODUCTION_PREFLIGHT_SCHEMA_VERSION
        or evidence.get("status") != "passed"
        or evidence.get("model", {}).get("model_revision") != MODEL_REVISION
        or evidence.get("selected_lane") not in {"bf16-lora", "nf4-qlora"}
        or evidence.get("q0_gate", {}).get("evidence_sha256")
        != q0_gate.get("evidence_sha256")
        or update.get("loss_finite") is not True
        or update.get("all_trainable_gradients_present") is not True
        or update.get("all_gradients_finite") is not True
        or update.get("adapter_parameter_changed") is not True
        or update.get("frozen_parameter_unchanged") is not True
        or update.get("only_intended_lora_parameters_trainable") is not True
        or runtime.get("headroom_passed") is not True
        or not isinstance(binding_manifest_sha256, str)
        or len(binding_manifest_sha256) != 64
    ):
        raise ValueError(
            "generalist-v2 training requires a passed production preflight"
        )
    return {
        "schema_version": evidence["schema_version"],
        "selected_lane": evidence["selected_lane"],
        "binding_manifest_sha256": binding_manifest_sha256,
        "evidence_sha256": sha256_file(path),
        "passed_before_training": True,
    }


def validate_bounded_training_evidence(
    path: Path,
    workload_id: str,
    q0_gate: Mapping[str, Any],
    production_gate: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    training = evidence.get("training", {})
    reload = evidence.get("reload_generation_evaluator_gate", {})
    if (
        evidence.get("schema_version") != BOUNDED_TRAINING_SCHEMA_VERSION
        or evidence.get("status") != "passed"
        or evidence.get("workload", {}).get("workload_id") != workload_id
        or evidence.get("model", {}).get("model_revision") != MODEL_REVISION
        or evidence.get("q0_gate", {}).get("evidence_sha256")
        != q0_gate.get("evidence_sha256")
        or evidence.get("production_preflight_gate", {}).get("evidence_sha256")
        != production_gate.get("evidence_sha256")
        or training.get("logs", {}).get("covers_every_optimizer_step_exactly_once")
        is not True
        or reload.get("fresh_base_and_adapter_reload") is not True
        or reload.get("evaluator_infrastructure_errors") != 0
    ):
        raise ValueError(f"generalist-v2 bounded gate is incomplete: {workload_id}")
    if workload_id == "generalist-v2-overfit64-v1" and (
        training.get("overfit_curve", {}).get("strong_fit_loss_gate") is not True
        or int(reload.get("exact_target_count", 0)) < 1
        or int(reload.get("lean_verified_count", 0)) < 1
    ):
        raise ValueError("generalist-v2 overfit64 did not strongly fit and regenerate")
    adapter_dir = path.parent / str(evidence.get("adapter", {}).get("path", ""))
    required = ("adapter_config.json", "adapter_model.safetensors")
    if any(not (adapter_dir / name).is_file() for name in required):
        raise ValueError(f"generalist-v2 bounded adapter is missing: {workload_id}")
    return {
        "schema_version": evidence["schema_version"],
        "workload_id": workload_id,
        "selected_lane": evidence["selected_lane"],
        "evidence_sha256": sha256_file(path),
        "adapter_model_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "passed_before_full_training": True,
    }


def summarize_finite_optimizer_logs(
    log_history: Sequence[Mapping[str, Any]], expected_optimizer_steps: int
) -> dict[str, Any]:
    entries = [item for item in log_history if "loss" in item]
    observed_steps = [int(item.get("step", -1)) for item in entries]
    if observed_steps != list(range(1, expected_optimizer_steps + 1)):
        raise RuntimeError(
            "generalist-v2 logs do not cover every optimizer step exactly once"
        )
    losses = [float(item["loss"]) for item in entries]
    try:
        gradient_norms = [float(item["grad_norm"]) for item in entries]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("generalist-v2 logs are missing gradient norms") from error
    if not all(math.isfinite(value) for value in (*losses, *gradient_norms)):
        raise RuntimeError("generalist-v2 logs contain non-finite optimization values")
    return {
        "logged_optimizer_steps": len(entries),
        "covers_every_optimizer_step_exactly_once": True,
        "all_losses_finite": True,
        "all_gradient_norms_finite": True,
        "loss": {
            "minimum": min(losses),
            "maximum": max(losses),
            "mean": fmean(losses),
            "first": losses[0],
            "last": losses[-1],
        },
        "gradient_norm_before_clipping": {
            "minimum": min(gradient_norms),
            "maximum": max(gradient_norms),
            "mean": fmean(gradient_norms),
        },
    }


def summarize_overfit_curve(
    log_history: Sequence[Mapping[str, Any]], *, one_pass_optimizer_steps: int
) -> dict[str, Any]:
    if one_pass_optimizer_steps < 1:
        raise ValueError("overfit64 needs at least one optimizer step per pass")
    entries = [item for item in log_history if "loss" in item]
    if len(entries) < 2 * one_pass_optimizer_steps:
        raise RuntimeError("overfit64 did not complete two comparable passes")
    first = fmean(float(item["loss"]) for item in entries[:one_pass_optimizer_steps])
    last = fmean(float(item["loss"]) for item in entries[-one_pass_optimizer_steps:])
    strongly_fit = last < first and (last <= 0.1 or last <= first * 0.25)
    if not strongly_fit:
        raise RuntimeError(
            "overfit64 did not strongly reduce the comparable pass loss: "
            f"first={first}, last={last}"
        )
    return {
        "first_complete_pass_mean_loss": first,
        "last_complete_pass_mean_loss": last,
        "last_to_first_loss_ratio": last / first,
        "strong_fit_loss_gate": True,
        "criterion": "last < first and (last <= 0.1 or last <= 25% of first)",
    }


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"bounded evaluator root is not a Git checkout: {path}")
    return completed.stdout.strip()


def _validate_bounded_adapter_reload(
    config: GeneralistV2Config,
    adapter_dir: Path,
    probe_records: Sequence[GeneralistProofVariant],
    workload_id: str,
    lean_project_root: Path,
    *,
    model_snapshot: Path | None,
) -> dict[str, Any]:
    """Freshly reload an adapter and exercise generation plus Lean evaluation."""
    synthetic = sorted(
        (item for item in probe_records if item.source_kind == "synthetic"),
        key=lambda item: (
            len(item.completion),
            hashlib.sha256(item.proof_variant_id.encode()).digest(),
        ),
    )
    probe_count = 4 if workload_id == "generalist-v2-overfit64-v1" else 1
    if len(synthetic) < probe_count:
        raise RuntimeError("bounded training selection lacks synthetic reload probes")
    probes = synthetic[:probe_count]
    torch, device_index, properties = _require_local_cuda()
    torch.cuda.reset_peak_memory_stats(device_index)
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 adapter reload needs PEFT and Transformers"
        ) from error

    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    if (
        adapter_config.base_model_name_or_path != MODEL_ID
        or adapter_config.revision != MODEL_REVISION
        or int(adapter_config.r) != int(config.lora["r"])
        or adapter_config.target_modules != str(config.lora["target_regex"])
    ):
        raise RuntimeError("bounded adapter does not preserve its pinned identity")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    if tokenizer.eos_token_id is None:
        raise RuntimeError("reloaded bounded adapter tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    source, source_kwargs = _model_source(model_snapshot)
    lane = choose_precision_lane(
        config, device_total_memory_bytes=int(properties.total_memory)
    )
    common = {
        "dtype": torch.bfloat16,
        "device_map": {"": device_index},
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        **source_kwargs,
    }
    if lane == "nf4-qlora":
        fallback = config.precision["fallback"]
        common["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=bool(fallback["load_in_4bit"]),
            bnb_4bit_quant_type=str(fallback["quantization_type"]),
            bnb_4bit_use_double_quant=bool(fallback["double_quantization"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = AutoModelForCausalLM.from_pretrained(source, **common)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("reloaded bounded adapter unexpectedly remains trainable")
    device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    )

    verifier = LeanVerifier(
        lean_project_root,
        timeout_seconds=float(config.evaluation["verifier_timeout_seconds"]),
    )
    results: list[dict[str, Any]] = []
    for record in probes:
        prompt = render_generalist_prompt(record)
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        inputs = {key: value.to(device) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(config.evaluation["sampling"]["max_new_tokens"]),
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
                use_cache=True,
            )
        generation_seconds = time.perf_counter() - started
        generated_ids = generated[0, prompt_tokens:]
        candidate = normalize_transport(
            tokenizer.decode(generated_ids, skip_special_tokens=True)
        )
        task = TaskRecord(
            id=record.statement_id,
            preamble=record.preamble,
            declaration=record.declaration,
            declaration_name=record.declaration_name,
        )
        oracle_failure = verifier.prime_task(
            task,
            normalize_transport(record.completion),
            timeout_seconds=max(
                120.0, float(config.evaluation["verifier_timeout_seconds"])
            ),
        )
        if oracle_failure is not None:
            raise RuntimeError(
                "bounded reload probe oracle does not verify: "
                f"{record.statement_id} {oracle_failure.category}"
            )
        outcome = verifier.verify(task, candidate)
        if outcome.category == "verifier_error":
            raise RuntimeError(
                "bounded adapter evaluator integration returned verifier_error"
            )
        results.append(
            {
                "statement_id": record.statement_id,
                "proof_variant_id": record.proof_variant_id,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(generated_ids.shape[0]),
                "generation_wall_time_seconds": generation_seconds,
                "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                "exact_target": candidate == normalize_transport(record.completion),
                "lean_category": outcome.category,
            }
        )
    exact = sum(item["exact_target"] for item in results)
    verified = sum(item["lean_category"] == "verified" for item in results)
    if workload_id == "generalist-v2-overfit64-v1" and not (exact and verified):
        raise RuntimeError(
            "overfit64 reloaded generation did not reproduce and verify a target"
        )
    torch.cuda.synchronize(device_index)
    value = {
        "fresh_base_and_adapter_reload": True,
        "adapter_base_model_name_or_path": adapter_config.base_model_name_or_path,
        "adapter_base_revision": adapter_config.revision,
        "adapter_rank": int(adapter_config.r),
        "adapter_merged": False,
        "adapter_trainable_after_reload": False,
        "generation_backend": "transformers-greedy",
        "generation_probe_count": len(results),
        "exact_target_count": exact,
        "lean_verified_count": verified,
        "evaluator_infrastructure_errors": 0,
        "results": results,
        "lean_project": {
            "root_revision": _git_head(lean_project_root),
            "mathlib_revision": _git_head(lean_project_root / ".lake/packages/mathlib"),
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
    }
    del generated, generated_ids, inputs, model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return value


def run_bounded_training_gate(
    config: GeneralistV2Config,
    package_root: Path,
    binding_path: Path,
    q0_evidence_path: Path,
    production_preflight_path: Path,
    workload_id: str,
    output_dir: Path,
    lean_project_root: Path,
    *,
    model_snapshot: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    if workload_id not in {
        "generalist-v2-overfit64-v1",
        "generalist-v2-smoke4096-v1",
    }:
        raise ValueError(f"unknown bounded generalist-v2 workload: {workload_id}")
    if (output_dir / "run.json").exists() or (output_dir / "trainer-state").exists():
        raise ValueError("bounded generalist-v2 training requires a fresh output path")
    q0_gate = validate_q0_training_gate(q0_evidence_path)
    production_gate = validate_production_preflight_gate(
        production_preflight_path, q0_gate
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("schema_version") != DATASET_BINDING_SCHEMA_VERSION:
        raise ValueError("bounded training requires the accepted Dataset-v2 binding")
    if (
        production_gate["binding_manifest_sha256"]
        != binding["dataset"]["manifest_sha256"]
    ):
        raise ValueError("bounded training binding differs from production preflight")
    context_tokens = int(binding["serialization"]["lengths"]["selected_context_tokens"])
    if context_tokens != int(config.training["resolved_context_tokens"]):
        raise ValueError("bounded training context differs from Dataset-v2 binding")

    torch, device_index, properties = _require_local_cuda()
    torch.manual_seed(int(config.training["seed"]))
    torch.cuda.manual_seed_all(int(config.training["seed"]))
    torch.cuda.reset_peak_memory_stats(device_index)
    records = load_bound_training_variants(package_root)
    if workload_id == "generalist-v2-overfit64-v1":
        train_probe = json.loads(
            (package_root / "train-probe.json").read_text(encoding="utf-8")
        )
        selected, workload = select_overfit64_variants(records, train_probe)
        maximum_optimizer_steps: int | None = OVERFIT64_OPTIMIZER_STEPS
    else:
        selected, workload = select_smoke4096_variants(records)
        maximum_optimizer_steps = None

    runtime = load_training_runtime(config, model_snapshot=model_snapshot)
    if runtime.lane != production_gate["selected_lane"]:
        raise RuntimeError("bounded training precision lane differs from preflight")
    examples, weight_normalizer, tokenization = tokenize_weighted_training_selection(
        records,
        selected,
        runtime.tokenizer,
        maximum_sequence_tokens=context_tokens,
    )
    reload_probes = tuple(selected)
    activation_cpu_offload_example_count = sum(
        should_offload_activations(len(item.input_ids)) for item in examples
    )
    gradient_checkpointing_example_count = sum(
        should_checkpoint_activations(len(item.input_ids)) for item in examples
    )
    del records, selected
    gc.collect()
    one_pass_steps = math.ceil(
        len(examples) / int(config.training["gradient_accumulation_steps"])
    )
    expected_optimizer_steps = (
        one_pass_steps if maximum_optimizer_steps is None else maximum_optimizer_steps
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    trainer = build_weighted_sft_trainer(
        runtime,
        examples,
        config,
        output_dir / "trainer-state",
        maximum_sequence_tokens=context_tokens,
        save_quarter_checkpoints=False,
        weight_normalizer=weight_normalizer,
        maximum_optimizer_steps=maximum_optimizer_steps,
    )
    started = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - started
    completed_steps = int(trainer.state.global_step)
    if completed_steps != expected_optimizer_steps:
        raise RuntimeError(
            "bounded generalist-v2 training stopped at the wrong step: "
            f"{completed_steps} != {expected_optimizer_steps}"
        )
    log_summary = summarize_finite_optimizer_logs(
        trainer.state.log_history, expected_optimizer_steps
    )
    overfit_curve = (
        summarize_overfit_curve(
            trainer.state.log_history,
            one_pass_optimizer_steps=one_pass_steps,
        )
        if workload_id == "generalist-v2-overfit64-v1"
        else None
    )
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    runtime.tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    required_adapter_files = ("adapter_config.json", "adapter_model.safetensors")
    missing_adapter_files = [
        name for name in required_adapter_files if not (adapter_dir / name).is_file()
    ]
    if missing_adapter_files or any(adapter_dir.glob("model-*.safetensors")):
        raise RuntimeError(
            "bounded generalist-v2 adapter artifact is invalid: "
            f"missing={missing_adapter_files}"
        )
    torch.cuda.synchronize(device_index)
    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_index))
    trainable_parameter_count = runtime.trainable_parameter_count
    selected_lane = runtime.lane
    sequence_chunked_mlp_module_count = runtime.sequence_chunked_mlp_module_count
    gated_delta_rule_backend = runtime.gated_delta_rule_backend
    del trainer, runtime, examples
    gc.collect()
    torch.cuda.empty_cache()
    reload = _validate_bounded_adapter_reload(
        config,
        adapter_dir,
        reload_probes,
        workload_id,
        lean_project_root,
        model_snapshot=model_snapshot,
    )
    value = {
        "schema_version": BOUNDED_TRAINING_SCHEMA_VERSION,
        "status": "passed",
        "workload": workload,
        "model": config.model,
        "dataset": {
            "package_id": config.dataset["package_id"],
            "binding_manifest_sha256": binding["dataset"]["manifest_sha256"],
        },
        "q0_gate": q0_gate,
        "production_preflight_gate": production_gate,
        "selected_lane": selected_lane,
        "tokenization": tokenization,
        "training": {
            **config.training,
            "train_sampling_strategy": "sequential",
            "full_membership_weight_normalizer": weight_normalizer,
            "one_pass_optimizer_steps": one_pass_steps,
            "configured_maximum_optimizer_steps": maximum_optimizer_steps,
            "completed_optimizer_steps": completed_steps,
            "complete_dataset_passes": (
                1.0
                if maximum_optimizer_steps is None
                else maximum_optimizer_steps / one_pass_steps
            ),
            "logs": log_summary,
            "overfit_curve": overfit_curve,
        },
        "adapter": {
            "path": "adapter",
            "format": "peft-lora",
            "required_files": list(required_adapter_files),
            "merged_base_model_shards_saved": False,
            "trainable_parameter_count": trainable_parameter_count,
        },
        "reload_generation_evaluator_gate": reload,
        "runtime": {
            "execution": "project-controlled-local-cuda",
            "python": platform.python_version(),
            "cuda_device_index": device_index,
            "cuda_device": properties.name,
            "cuda_device_total_memory_bytes": int(properties.total_memory),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "reserved_memory_headroom_bytes": int(properties.total_memory)
            - peak_reserved,
            "training_wall_time_seconds": wall_time,
            "torch_cuda_version": torch.version.cuda,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "gradient_checkpointing_use_reentrant": (
                GRADIENT_CHECKPOINTING_USE_REENTRANT
            ),
            "gradient_checkpointing_min_sequence_tokens": (
                GRADIENT_CHECKPOINTING_MIN_SEQUENCE_TOKENS
            ),
            "gradient_checkpointing_example_count": (
                gradient_checkpointing_example_count
            ),
            "linear_attention_chunk_size": LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
            "full_attention_sdpa_backend": FULL_ATTENTION_SDPA_BACKEND,
            "explicit_compute_loss_bf16_autocast": True,
            "activation_cpu_offload": ACTIVATION_CPU_OFFLOAD,
            "activation_cpu_offload_min_sequence_tokens": (
                ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS
            ),
            "activation_cpu_offload_example_count": (
                activation_cpu_offload_example_count
            ),
            "lm_head_loss_chunk_tokens": LM_HEAD_LOSS_CHUNK_TOKENS,
            "target_only_checkpointed_causal_loss": True,
            "mlp_sequence_chunk_tokens": MLP_SEQUENCE_CHUNK_TOKENS,
            "sequence_chunked_mlp_module_count": (sequence_chunked_mlp_module_count),
            "gated_delta_rule_backend": gated_delta_rule_backend,
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
    (output_dir / "run.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def run_production_preflight(
    config: GeneralistV2Config,
    package_root: Path,
    binding_path: Path,
    q0_evidence_path: Path,
    output: Path,
    *,
    model_snapshot: Path | None = None,
    minimum_headroom_bytes: int = MINIMUM_PRODUCTION_HEADROOM_BYTES,
) -> dict[str, Any]:
    """Run the real near-maximum QLoRA forward/backward/update gate."""
    config.validate()
    if minimum_headroom_bytes < MINIMUM_PRODUCTION_HEADROOM_BYTES:
        raise ValueError("production memory headroom cannot be below 512 MiB")
    q0_gate = validate_q0_training_gate(q0_evidence_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    records = load_bound_training_variants(package_root)
    record, expected_tokens, context_tokens = resolve_near_maximum_variant(
        records, binding
    )
    configured_context = int(config.training["resolved_context_tokens"])
    if context_tokens != configured_context:
        raise ValueError("Dataset-v2 binding and training context differ")
    weights = compute_training_weights(records)
    global_weight_normalizer = fmean(weights.variant_weights.values())

    torch, device_index, properties = _require_local_cuda()
    torch.manual_seed(int(config.training["seed"]))
    torch.cuda.manual_seed_all(int(config.training["seed"]))
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    runtime = load_training_runtime(config, model_snapshot=model_snapshot)
    example = tokenize_generalist_variant(
        record,
        runtime.tokenizer,
        example_weight=weights.variant_weights[record.proof_variant_id],
        maximum_sequence_tokens=context_tokens,
    )
    if len(example.input_ids) != expected_tokens:
        raise RuntimeError(
            "production preflight serialization differs from Dataset-v2 binding: "
            f"{len(example.input_ids)} != {expected_tokens}"
        )
    del records, weights
    gc.collect()

    trainer = build_weighted_sft_trainer(
        runtime,
        [example],
        config,
        output.parent / "trainer-preflight",
        maximum_sequence_tokens=context_tokens,
        save_quarter_checkpoints=False,
        weight_normalizer=global_weight_normalizer,
    )
    trainer.create_optimizer()
    batch = trainer.data_collator([example.to_trainer_row()])
    prompt_masked = not bool(
        batch["labels"][0, : example.prompt_tokens].ne(IGNORE_INDEX).any()
    )
    padding = batch["attention_mask"].eq(0)
    padding_masked = not bool(batch["labels"][padding].ne(IGNORE_INDEX).any())
    eos_supervised = int(batch["labels"][0, -1].item()) == int(
        runtime.tokenizer.eos_token_id
    )
    weight_preserved = math.isclose(
        float(batch["example_weight"][0].item()),
        example.example_weight,
        rel_tol=1e-6,
        abs_tol=1e-8,
    )
    if not (prompt_masked and padding_masked and eos_supervised and weight_preserved):
        raise RuntimeError("generalist-v2 production labels or example weight changed")

    device = next(
        parameter.device
        for parameter in runtime.model.parameters()
        if parameter.device.type == "cuda"
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    trainable_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad
    }
    frozen_name, frozen_parameter = next(
        (name, parameter)
        for name, parameter in runtime.model.named_parameters()
        if not parameter.requires_grad and "base_layer.weight" in name
    )
    frozen_before = frozen_parameter.detach().cpu().clone()

    runtime.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    with (
        trainer.compute_loss_context_manager(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        loss = trainer.compute_loss(runtime.model, batch)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("generalist-v2 production preflight loss is non-finite")
    trainer.accelerator.backward(loss)
    trainable_parameters = [
        parameter for parameter in runtime.model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters or any(
        parameter.grad is None for parameter in trainable_parameters
    ):
        raise RuntimeError(
            "generalist-v2 production gradients are missing or non-finite"
        )
    if any(
        not bool(torch.isfinite(parameter.grad).all().item())
        for parameter in trainable_parameters
    ):
        raise RuntimeError(
            "generalist-v2 production gradients are missing or non-finite"
        )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        float(config.training["maximum_gradient_norm"]),
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError("generalist-v2 production gradient norm is non-finite")
    trainer.optimizer.step()
    torch.cuda.synchronize(device_index)

    changed_trainables = tuple(
        name
        for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad
        and not torch.equal(trainable_before[name], parameter.detach().cpu())
    )
    frozen_unchanged = torch.equal(frozen_before, frozen_parameter.detach().cpu())
    if not changed_trainables or not frozen_unchanged:
        raise RuntimeError(
            "generalist-v2 production update did not remain adapter-only: "
            f"changed_trainables={len(changed_trainables)}, "
            f"frozen_unchanged={frozen_unchanged}"
        )

    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_index))
    total_memory = int(properties.total_memory)
    headroom = total_memory - peak_reserved
    if headroom < minimum_headroom_bytes:
        raise RuntimeError(
            "generalist-v2 production preflight lacks required VRAM headroom: "
            f"{headroom} < {minimum_headroom_bytes}"
        )
    wall_time = time.perf_counter() - started
    value = {
        "schema_version": PRODUCTION_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "model": config.model,
        "dataset": {
            "package_id": config.dataset["package_id"],
            "binding_schema_version": binding["schema_version"],
            "binding_manifest_sha256": binding["dataset"]["manifest_sha256"],
        },
        "q0_gate": q0_gate,
        "selected_lane": runtime.lane,
        "near_maximum_example": {
            "statement_id": example.statement_id,
            "proof_variant_id": example.proof_variant_id,
            "sequence_tokens": len(example.input_ids),
            "prompt_tokens": example.prompt_tokens,
            "completion_tokens_excluding_eos": example.completion_tokens,
            "configured_context_tokens": context_tokens,
            "example_weight": example.example_weight,
            "full_membership_weight_normalizer": global_weight_normalizer,
            "prompt_labels_masked": prompt_masked,
            "padding_labels_masked": padding_masked,
            "terminal_eos_supervised": eos_supervised,
            "example_weight_preserved": weight_preserved,
            "truncated_or_dropped": False,
        },
        "update": {
            "loss": float(loss.detach().item()),
            "loss_finite": True,
            "gradient_norm_before_clipping": float(gradient_norm.detach().item()),
            "all_trainable_gradients_present": True,
            "all_gradients_finite": True,
            "trainable_parameter_count": runtime.trainable_parameter_count,
            "trainable_parameter_name_count": len(runtime.trainable_parameter_names),
            "trainable_parameter_names_sha256": _parameter_names_sha256(
                runtime.trainable_parameter_names
            ),
            "changed_trainable_parameter_count": len(changed_trainables),
            "changed_trainable_parameter_names_sha256": _parameter_names_sha256(
                changed_trainables
            ),
            "adapter_parameter_changed": True,
            "frozen_parameter_checked": frozen_name,
            "frozen_parameter_unchanged": True,
            "only_intended_lora_parameters_trainable": True,
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
            "reserved_memory_headroom_bytes": headroom,
            "minimum_required_headroom_bytes": minimum_headroom_bytes,
            "headroom_passed": True,
            "wall_time_seconds": wall_time,
            "torch_cuda_version": torch.version.cuda,
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "gradient_checkpointing_use_reentrant": (
                GRADIENT_CHECKPOINTING_USE_REENTRANT
            ),
            "gradient_checkpointing_min_sequence_tokens": (
                GRADIENT_CHECKPOINTING_MIN_SEQUENCE_TOKENS
            ),
            "gradient_checkpointing_applied": should_checkpoint_activations(
                len(example.input_ids)
            ),
            "linear_attention_chunk_size": LINEAR_ATTENTION_TRAINING_CHUNK_SIZE,
            "full_attention_sdpa_backend": FULL_ATTENTION_SDPA_BACKEND,
            "explicit_compute_loss_bf16_autocast": True,
            "activation_cpu_offload": ACTIVATION_CPU_OFFLOAD,
            "activation_cpu_offload_min_sequence_tokens": (
                ACTIVATION_CPU_OFFLOAD_MIN_SEQUENCE_TOKENS
            ),
            "activation_cpu_offload_applied": should_offload_activations(
                len(example.input_ids)
            ),
            "lm_head_loss_chunk_tokens": LM_HEAD_LOSS_CHUNK_TOKENS,
            "target_only_checkpointed_causal_loss": True,
            "mlp_sequence_chunk_tokens": MLP_SEQUENCE_CHUNK_TOKENS,
            "sequence_chunked_mlp_module_count": (
                runtime.sequence_chunked_mlp_module_count
            ),
            "gated_delta_rule_backend": runtime.gated_delta_rule_backend,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del batch, trainable_before, frozen_before, trainer, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return value
