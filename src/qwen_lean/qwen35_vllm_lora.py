from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

COMPAT_SCHEMA_VERSION = "qwen35-vllm-text-wrapper-adapter-v1"
SOURCE_TENSOR_PREFIX = "base_model.model.model.layers."
RUNTIME_TENSOR_PREFIX = "base_model.model.model.language_model.layers."


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
) -> dict[str, Any]:
    manifest_path = target / "qwen35-vllm-compatibility.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"incomplete Qwen3.5 vLLM adapter exists: {target}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_payload_sha256, source_tensor_count = _tensor_payload_sha256(source_model)
    runtime_payload_sha256, runtime_tensor_count = _tensor_payload_sha256(
        target / "adapter_model.safetensors"
    )
    if (
        value.get("schema_version") != COMPAT_SCHEMA_VERSION
        or value.get("source_adapter_config_sha256") != source_config_sha256
        or value.get("source_adapter_model_sha256") != source_model_sha256
        or _sha256_file(target / "adapter_config.json")
        != value.get("runtime_adapter_config_sha256")
        or _sha256_file(target / "adapter_model.safetensors")
        != value.get("runtime_adapter_model_sha256")
        or source_payload_sha256 != value.get("source_tensor_payload_sha256")
        or runtime_payload_sha256 != value.get("runtime_tensor_payload_sha256")
        or source_payload_sha256 != runtime_payload_sha256
        or source_tensor_count != value.get("tensor_count")
        or source_tensor_count != runtime_tensor_count
    ):
        raise ValueError(f"Qwen3.5 vLLM compatibility adapter differs: {target}")
    return {**value, "runtime_adapter_dir": str(target.resolve())}


def prepare_qwen35_vllm_adapter(source_dir: Path) -> dict[str, Any]:
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
    target = source_dir.parent / (
        f".{source_dir.name}-vllm-text-wrapper-{source_model_sha256[:12]}"
    )
    if target.exists():
        return _validate_existing(
            target,
            source_model=source_model,
            source_config_sha256=source_config_sha256,
            source_model_sha256=source_model_sha256,
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
            runtime_key = qwen35_vllm_runtime_tensor_key(source_key)
            if runtime_key in tensors:
                raise ValueError(f"Qwen3.5 vLLM tensor mapping collided: {runtime_key}")
            tensors[runtime_key] = handle.get_tensor(source_key)
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
    if (
        source_tensor_count != runtime_tensor_count
        or source_payload_sha256 != runtime_payload_sha256
    ):
        raise RuntimeError("Qwen3.5 vLLM compatibility transformation changed tensors")
    manifest = {
        "schema_version": COMPAT_SCHEMA_VERSION,
        "mapping_reason": "qwen35-conditional-generation-text-wrapper-namespace",
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
        "tensor_count": source_tensor_count,
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
        )
    return {**manifest, "runtime_adapter_dir": str(target.resolve())}
