from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

COMPAT_SCHEMA_VERSION = "qwen35-vllm-text-wrapper-adapter-v1"
GDN_SPLIT_COMPAT_SCHEMA_VERSION = "qwen35-vllm-text-wrapper-gdn-split-adapter-v2"
SOURCE_TENSOR_PREFIX = "base_model.model.model.layers."
RUNTIME_TENSOR_PREFIX = "base_model.model.model.language_model.layers."
GDN_QKV_SOURCE_FRAGMENT = ".linear_attn.in_proj_qkv."
GDN_QKV_RUNTIME_SUFFIXES = ("in_proj_q", "in_proj_k", "in_proj_v")
GDN_QKV_OUTPUT_SIZES = (2048, 2048, 4096)
VLLM_GDN_PACKED_MAPPING = (
    "in_proj_q",
    "in_proj_k",
    "in_proj_v",
    "in_proj_z",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qwen35_vllm_runtime_tensor_key(source_key: str) -> str:
    """Add the text-wrapper namespace omitted by Qwen3.5 PEFT checkpoints."""

    if not source_key.startswith(SOURCE_TENSOR_PREFIX):
        raise ValueError(f"unexpected Qwen3.5 PEFT tensor name: {source_key}")
    if not source_key.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ValueError(f"unexpected Qwen3.5 LoRA tensor suffix: {source_key}")
    return RUNTIME_TENSOR_PREFIX + source_key.removeprefix(SOURCE_TENSOR_PREFIX)


def qwen35_vllm_runtime_tensor_items(
    source_key: str, tensor: Any, *, split_gdn_qkv: bool
) -> list[tuple[str, Any]]:
    """Map one PEFT tensor to exact vLLM runtime tensors.

    vLLM 0.17 exposes Qwen3.5 GatedDeltaNet's fused ``qkvz`` projection as
    four LoRA slices but declares the checkpoint mapping as only ``qkv`` and
    ``z``. Split the already-packed QKV B matrix at the model's declared
    Q/K/V boundaries and duplicate A so the resulting update is algebraically
    identical. No tensor values are otherwise changed.
    """

    runtime_key = qwen35_vllm_runtime_tensor_key(source_key)
    if not split_gdn_qkv or GDN_QKV_SOURCE_FRAGMENT not in runtime_key:
        return [(runtime_key, tensor)]
    if source_key.endswith(".lora_A.weight"):
        return [
            (
                runtime_key.replace(
                    ".in_proj_qkv.lora_A.weight",
                    f".{suffix}.lora_A.weight",
                ),
                tensor.clone(),
            )
            for suffix in GDN_QKV_RUNTIME_SUFFIXES
        ]
    if source_key.endswith(".lora_B.weight"):
        if tensor.ndim != 2 or int(tensor.shape[0]) != sum(GDN_QKV_OUTPUT_SIZES):
            raise ValueError(
                "Qwen3.5 GatedDeltaNet QKV LoRA B shape differs from "
                f"{GDN_QKV_OUTPUT_SIZES}: {tuple(tensor.shape)}"
            )
        return [
            (
                runtime_key.replace(
                    ".in_proj_qkv.lora_B.weight",
                    f".{suffix}.lora_B.weight",
                ),
                part.contiguous(),
            )
            for suffix, part in zip(
                GDN_QKV_RUNTIME_SUFFIXES,
                tensor.split(GDN_QKV_OUTPUT_SIZES, dim=0),
                strict=True,
            )
        ]
    raise ValueError(f"unexpected Qwen3.5 GatedDeltaNet LoRA tensor: {source_key}")


def patch_qwen35_vllm_gdn_lora_mapping(*, expected_version: str) -> dict[str, Any]:
    """Correct vLLM's Qwen3.5 GDN packed-LoRA children in process memory."""

    import vllm
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    )

    if vllm.__version__ != expected_version:
        raise RuntimeError(
            f"Qwen3.5 GDN LoRA mapping patch requires vLLM {expected_version}, "
            f"observed {vllm.__version__}"
        )
    original = ["in_proj_qkv", "in_proj_z"]
    patched = list(VLLM_GDN_PACKED_MAPPING)
    classes = (Qwen3_5ForCausalLMBase, Qwen3_5ForConditionalGeneration)
    observed: dict[str, list[str]] = {}
    for model_class in classes:
        mapping = dict(model_class.packed_modules_mapping)
        current = list(mapping.get("in_proj_qkvz", []))
        if current not in (original, patched):
            raise RuntimeError(
                "unexpected vLLM Qwen3.5 in_proj_qkvz mapping: "
                f"{model_class.__name__}={current}"
            )
        observed[model_class.__name__] = current
        mapping["in_proj_qkvz"] = patched
        model_class.packed_modules_mapping = mapping
    return {
        "vllm_version": vllm.__version__,
        "original_mapping": original,
        "runtime_mapping": patched,
        "patched_classes": sorted(observed),
        "observed_before_patch": observed,
    }


def _canonical_tensor_key(key: str) -> str:
    if key.startswith(SOURCE_TENSOR_PREFIX):
        return key.removeprefix(SOURCE_TENSOR_PREFIX)
    if key.startswith(RUNTIME_TENSOR_PREFIX):
        return key.removeprefix(RUNTIME_TENSOR_PREFIX)
    raise ValueError(f"tensor is outside the Qwen3.5 text-wrapper mapping: {key}")


def _tensor_payload_sha256(path: Path) -> tuple[str, int]:
    import safetensors
    import torch

    digest = hashlib.sha256()
    count = 0
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys(), key=_canonical_tensor_key):
            tensor = handle.get_tensor(key).contiguous()
            digest.update(_canonical_tensor_key(key).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(-1).view(torch.uint8).numpy().tobytes())
            count += 1
    return digest.hexdigest(), count


def _validate_existing(
    target: Path,
    *,
    source_model: Path,
    source_config_sha256: str,
    source_model_sha256: str,
    split_gdn_qkv: bool,
) -> dict[str, Any]:
    manifest_path = target / "qwen35-vllm-compatibility.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"incomplete Qwen3.5 vLLM adapter exists: {target}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_payload_sha256, source_tensor_count = _tensor_payload_sha256(source_model)
    runtime_payload_sha256, runtime_tensor_count = _tensor_payload_sha256(
        target / "adapter_model.safetensors"
    )
    expected_schema = (
        GDN_SPLIT_COMPAT_SCHEMA_VERSION if split_gdn_qkv else COMPAT_SCHEMA_VERSION
    )
    if (
        value.get("schema_version") != expected_schema
        or value.get("gdn_qkv_split", False) is not split_gdn_qkv
        or value.get("source_adapter_config_sha256") != source_config_sha256
        or value.get("source_adapter_model_sha256") != source_model_sha256
        or _sha256_file(target / "adapter_config.json")
        != value.get("runtime_adapter_config_sha256")
        or _sha256_file(target / "adapter_model.safetensors")
        != value.get("runtime_adapter_model_sha256")
        or source_payload_sha256 != value.get("source_tensor_payload_sha256")
        or runtime_payload_sha256 != value.get("runtime_tensor_payload_sha256")
        or source_tensor_count
        != value.get("source_tensor_count", value.get("tensor_count"))
        or runtime_tensor_count
        != value.get("runtime_tensor_count", value.get("tensor_count"))
        or (
            not split_gdn_qkv
            and (
                source_payload_sha256 != runtime_payload_sha256
                or source_tensor_count != runtime_tensor_count
            )
        )
    ):
        raise ValueError(f"Qwen3.5 vLLM compatibility adapter differs: {target}")
    _validate_runtime_transform(
        source_model,
        target / "adapter_model.safetensors",
        split_gdn_qkv=split_gdn_qkv,
    )
    return {**value, "runtime_adapter_dir": str(target.resolve())}


def _validate_runtime_transform(
    source_model: Path, runtime_model: Path, *, split_gdn_qkv: bool
) -> None:
    import safetensors
    import torch

    with safetensors.safe_open(
        source_model, framework="pt", device="cpu"
    ) as source_handle:
        with safetensors.safe_open(
            runtime_model, framework="pt", device="cpu"
        ) as runtime_handle:
            expected_keys: set[str] = set()
            for source_key in source_handle.keys():  # noqa: SIM118
                expected_items = qwen35_vllm_runtime_tensor_items(
                    source_key,
                    source_handle.get_tensor(source_key),
                    split_gdn_qkv=split_gdn_qkv,
                )
                for runtime_key, expected_tensor in expected_items:
                    expected_keys.add(runtime_key)
                    if runtime_key not in runtime_handle.keys():
                        raise RuntimeError(
                            f"derived vLLM adapter omitted tensor: {runtime_key}"
                        )
                    if not torch.equal(
                        expected_tensor, runtime_handle.get_tensor(runtime_key)
                    ):
                        raise RuntimeError(
                            f"derived vLLM adapter changed tensor: {runtime_key}"
                        )
            if expected_keys != set(runtime_handle.keys()):
                raise RuntimeError("derived vLLM adapter has unexpected tensors")


def prepare_qwen35_vllm_adapter(
    source_dir: Path, *, split_gdn_qkv: bool = False
) -> dict[str, Any]:
    """Create a content-bound adapter with vLLM's actual text-wrapper names.

    Qwen3.5 PEFT saves text-only targets as ``model.layers.*``. The vLLM
    conditional-generation wrapper exposes them as ``language_model.model.layers.*``
    but its current HF mapper only recognizes the original multimodal checkpoint
    prefix. A derived, content-identical adapter closes that namespace gap without
    mutating the trained checkpoint or the installed vLLM wheel.
    """

    source_dir = source_dir.resolve()
    source_config = source_dir / "adapter_config.json"
    source_model = source_dir / "adapter_model.safetensors"
    if not source_config.is_file() or not source_model.is_file():
        raise FileNotFoundError(f"Qwen3.5 source adapter is incomplete: {source_dir}")
    config = json.loads(source_config.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != "Qwen/Qwen3.5-4B-Base":
        raise ValueError("vLLM compatibility mapping is only valid for Qwen3.5-4B-Base")

    source_config_sha256 = _sha256_file(source_config)
    source_model_sha256 = _sha256_file(source_model)
    transform_label = "vllm-017-gdn-split" if split_gdn_qkv else "vllm-text-wrapper"
    target = source_dir.parent / (
        f".{source_dir.name}-{transform_label}-{source_model_sha256[:12]}"
    )
    if target.exists():
        return _validate_existing(
            target,
            source_model=source_model,
            source_config_sha256=source_config_sha256,
            source_model_sha256=source_model_sha256,
            split_gdn_qkv=split_gdn_qkv,
        )

    import safetensors
    from safetensors.torch import save_file

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{source_dir.name}-vllm-text-wrapper-tmp-",
            dir=source_dir.parent,
        )
    )
    tensors = {}
    metadata = None
    with safetensors.safe_open(source_model, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for source_key in handle.keys():  # noqa: SIM118
            for runtime_key, tensor in qwen35_vllm_runtime_tensor_items(
                source_key,
                handle.get_tensor(source_key),
                split_gdn_qkv=split_gdn_qkv,
            ):
                if runtime_key in tensors:
                    raise ValueError(
                        f"Qwen3.5 vLLM tensor mapping collided: {runtime_key}"
                    )
                tensors[runtime_key] = tensor
    shutil.copy2(source_config, temporary / "adapter_config.json")
    save_file(
        tensors,
        temporary / "adapter_model.safetensors",
        metadata=metadata,
    )
    del tensors

    source_payload_sha256, source_tensor_count = _tensor_payload_sha256(source_model)
    runtime_model = temporary / "adapter_model.safetensors"
    runtime_payload_sha256, runtime_tensor_count = _tensor_payload_sha256(runtime_model)
    _validate_runtime_transform(
        source_model, runtime_model, split_gdn_qkv=split_gdn_qkv
    )
    if not split_gdn_qkv and (
        source_tensor_count != runtime_tensor_count
        or source_payload_sha256 != runtime_payload_sha256
    ):
        raise RuntimeError("Qwen3.5 vLLM compatibility transformation changed tensors")
    manifest = {
        "schema_version": (
            GDN_SPLIT_COMPAT_SCHEMA_VERSION
            if split_gdn_qkv
            else COMPAT_SCHEMA_VERSION
        ),
        "mapping_reason": (
            "qwen35-conditional-generation-text-wrapper-and-vllm-017-gdn-qkv-split"
            if split_gdn_qkv
            else "qwen35-conditional-generation-text-wrapper-namespace"
        ),
        "gdn_qkv_split": split_gdn_qkv,
        "gdn_qkv_output_sizes": (
            list(GDN_QKV_OUTPUT_SIZES) if split_gdn_qkv else None
        ),
        "vllm_gdn_packed_mapping": (
            list(VLLM_GDN_PACKED_MAPPING) if split_gdn_qkv else None
        ),
        "source_tensor_prefix": SOURCE_TENSOR_PREFIX,
        "runtime_tensor_prefix": RUNTIME_TENSOR_PREFIX,
        "source_adapter_config_sha256": source_config_sha256,
        "source_adapter_model_sha256": source_model_sha256,
        "runtime_adapter_config_sha256": _sha256_file(
            temporary / "adapter_config.json"
        ),
        "runtime_adapter_model_sha256": _sha256_file(runtime_model),
        "source_tensor_payload_sha256": source_payload_sha256,
        "runtime_tensor_payload_sha256": runtime_payload_sha256,
        "source_tensor_count": source_tensor_count,
        "runtime_tensor_count": runtime_tensor_count,
        "semantic_transform_verified": True,
    }
    (temporary / "qwen35-vllm-compatibility.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        temporary.rename(target)
    except FileExistsError:
        shutil.rmtree(temporary)
        return _validate_existing(
            target,
            source_model=source_model,
            source_config_sha256=source_config_sha256,
            source_model_sha256=source_model_sha256,
            split_gdn_qkv=split_gdn_qkv,
        )
    return {**manifest, "runtime_adapter_dir": str(target.resolve())}
