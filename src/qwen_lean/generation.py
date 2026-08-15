from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

from .artifacts import write_artifacts
from .evaluator import LEAN_TOOLCHAIN, MATHLIB_REVISION
from .prompt import PROMPT_FORMAT_ID, render_prompt
from .schema import CandidateResult, RunMetadata, TaskRecord
from .verifier import LeanVerifier


MODEL_ID = "Qwen/Qwen3-8B-Base"
DEFAULT_MAX_NEW_TOKENS = 128


def run_model_smoke(
    task: TaskRecord,
    output_dir: Path,
    project_root: Path,
    *,
    task_source: str,
    timeout_seconds: float = 30.0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[RunMetadata, CandidateResult]:
    started = time.perf_counter()
    generation_started = started
    tokenizer_id = MODEL_ID
    runtime: dict[str, Any] = {"python": platform.python_version()}
    candidate_text = ""

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("the Phase 0 model smoke requires a CUDA GPU")

        cuda_device_index = 0
        cuda_properties = torch.cuda.get_device_properties(cuda_device_index)
        runtime.update(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "inference_execution": "local_cuda",
                "cuda_device_index": cuda_device_index,
                "cuda_device": cuda_properties.name,
                "cuda_device_capability": [
                    cuda_properties.major,
                    cuda_properties.minor,
                ],
                "cuda_device_total_memory_bytes": cuda_properties.total_memory,
                "torch_cuda_version": torch.version.cuda,
            }
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer_id = tokenizer.name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": cuda_device_index},
            low_cpu_mem_usage=True,
        )
        if model.device.type != "cuda" or model.device.index != cuda_device_index:
            raise RuntimeError(
                f"model loaded on {model.device}, expected local cuda:{cuda_device_index}"
            )
        prompt = render_prompt(task)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation_ids = generated[0, input_length:]
        candidate_text = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generation_latency = time.perf_counter() - generation_started
    except Exception as error:
        generation_latency = time.perf_counter() - generation_started
        metadata = _model_metadata(
            task_source,
            tokenizer_id,
            timeout_seconds,
            max_new_tokens,
            runtime,
        )
        result = CandidateResult(
            task_id=task.id,
            candidate_id="model-0",
            candidate_index=0,
            candidate_text=candidate_text,
            category="generation_error",
            lean_exit_code=None,
            diagnostics={"stdout": "", "stderr": f"{type(error).__name__}: {error}"},
            generation_latency_seconds=generation_latency,
            verification_latency_seconds=None,
            total_latency_seconds=time.perf_counter() - started,
        )
        write_artifacts(output_dir, metadata, [result])
        return metadata, result

    verifier = LeanVerifier(project_root, timeout_seconds=timeout_seconds)
    outcome = verifier.verify(task, candidate_text)
    metadata = _model_metadata(
        task_source,
        tokenizer_id,
        timeout_seconds,
        max_new_tokens,
        runtime,
    )
    result = CandidateResult(
        task_id=task.id,
        candidate_id="model-0",
        candidate_index=0,
        candidate_text=candidate_text,
        category=outcome.category,
        lean_exit_code=outcome.lean_exit_code,
        diagnostics=outcome.diagnostics,
        generation_latency_seconds=generation_latency,
        verification_latency_seconds=outcome.latency_seconds,
        total_latency_seconds=time.perf_counter() - started,
    )
    write_artifacts(output_dir, metadata, [result])
    return metadata, result


def _model_metadata(
    task_source: str,
    tokenizer_id: str,
    timeout_seconds: float,
    max_new_tokens: int,
    runtime: dict[str, Any],
) -> RunMetadata:
    return RunMetadata(
        candidate_source="model",
        task_source=task_source,
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=LEAN_TOOLCHAIN,
        mathlib_revision=MATHLIB_REVISION,
        verifier_timeout_seconds=timeout_seconds,
        model_id=MODEL_ID,
        tokenizer_id=tokenizer_id,
        generation_settings={
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "stop": "tokenizer_eos_or_token_limit",
            "dtype": "bfloat16",
            "device": "cuda:0",
        },
        runtime=runtime,
    )
