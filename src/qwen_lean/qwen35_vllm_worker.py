from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu_worker import Worker

from .qwen35_vllm_lora import patch_qwen35_vllm_gdn_lora_mapping


class Qwen35Vllm017Worker(Worker):
    """Apply the exact Qwen3.5 GDN LoRA mapping before model construction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        patch_qwen35_vllm_gdn_lora_mapping(expected_version="0.17.0")
        super().__init__(*args, **kwargs)
