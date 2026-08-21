from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import platform
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .generalist_v2 import (
    EXPECTED_LORA_MODULE_COUNTS,
    GENERALIST_SERIALIZATION_ID,
    LORA_TARGET_REGEX,
    LORA_TARGET_SUFFIXES,
    MODEL_ID,
    MODEL_REVISION,
    GeneralistV2Config,
    WeightedTokenizedExample,
    one_pass_membership_trajectory,
)
from .phase3 import IGNORE_INDEX


ARCHITECTURE_PREFLIGHT_SCHEMA_VERSION = "generalist-v2-architecture-preflight-v1"
RUNTIME_PREPARATION_SCHEMA_VERSION = "generalist-v2-runtime-preparation-v1"


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
            raise RuntimeError(f"LoRA target is not a linear projection: {path}")
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
        import torch.nn.functional as functional
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


def build_weighted_sft_trainer(
    runtime: GeneralistTrainingRuntime,
    examples: Sequence[WeightedTokenizedExample],
    config: GeneralistV2Config,
    output_dir: Path,
    *,
    maximum_sequence_tokens: int,
) -> Any:
    config.validate()
    if maximum_sequence_tokens not in tuple(config.training["context_choices"]):
        raise ValueError("generalist-v2 trainer received an unsupported context")
    if not examples:
        raise ValueError("generalist-v2 trainer received no examples")
    for example in examples:
        example.validate(int(runtime.tokenizer.eos_token_id), maximum_sequence_tokens)
    try:
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "generalist-v2 trainer dependencies are unavailable"
        ) from error

    weight_normalizer = fmean(item.example_weight for item in examples)
    trajectory = one_pass_membership_trajectory(
        [(item.statement_id, item.proof_variant_id) for item in examples]
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
            outputs = model(**inputs, use_cache=False)
            loss = statement_weighted_causal_loss(
                outputs.logits,
                inputs["labels"],
                weights,
                weight_normalizer=weight_normalizer,
            )
            return (loss, outputs) if return_outputs else loss

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
        lr_scheduler_type=str(config.training["lr_schedule"]),
        # Transformers 5 accepts a fractional warmup through warmup_steps when
        # the value is below one and resolves it against the final step count.
        warmup_steps=float(config.training["warmup_fraction"]),
        optim=str(config.training["optimizer"]),
        seed=int(config.training["seed"]),
        data_seed=int(config.training["data_seed"]),
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        callbacks=[QuarterCheckpointCallback()],
    )
