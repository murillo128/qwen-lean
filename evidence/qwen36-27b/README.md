# Qwen3.6-27B 4-bit Ada feasibility

**BLOCKED:** the frozen, fully GPU-resident Stage 0 lane loaded the official `Qwen/Qwen3.6-27B` revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` but failed before a real generation. vLLM reported that model loading took 63.727833 seconds and 17.93 GiB, then reported a CUDA out-of-memory error while requesting another 272 MiB, when only 216.38 MiB remained free.

The attempted lane used online BitsAndBytes FP4 conversion from the pinned official BF16 safetensors with float32 quantized-linear compute, BF16 activations, tensor parallelism 1, a 2,048-token context, four maximum sequences, eager mode, zero CPU weight offload, and zero KV swap space. It ran only on the project NVIDIA RTX 4000 Ada Generation; hosted inference was not used.

Peak observations were 20,425,753,088 CUDA-allocated bytes, 20,661,141,504 CUDA-reserved bytes, and 19,840 MiB device memory used. The attempt produced 0 candidates and 0 generated tokens. No dev16 or full-validation benchmark began, and the frozen configuration was not changed after the failure.

This is a hardware-feasibility result, not a model-quality result. Weights, caches, and bulky runtime logs remain outside Git; `preflight.json` retains the exact package, GPU, quantization, memory, and failure metadata while omitting the machine-local cache path.
