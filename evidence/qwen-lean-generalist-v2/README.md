# qwen-lean generalist v2 — Dataset-v2 binding and preparation

`OBSERVED`: the exact `Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b` loaded through the isolated
Qwen3.5 runtime as the text-only `Qwen3_5ForCausalLM` class. A 49-token raw
whole-proof prefix completed a finite local-CUDA forward pass on the project Ada
GPU. Peak CUDA allocation was 8,547,122,176 bytes.

`OBSERVED`: the pinned text decoder contains 248 intended rank-16 LoRA target
modules: 32 full-attention projections, 120 Gated DeltaNet projections, and 96
MLP projections. Their prospective adapter has 32,464,896 trainable parameters.
The exact module inventory is retained in `pre56-architecture.json`; no vision,
embedding, normalization, or `lm_head` module matched.

`OBSERVED`: the same isolated dependency lock loaded the base through the
automatic NF4 QLoRA fallback on the available 20,989,804,544-byte device,
attached 496 LoRA A/B parameter tensors over all 248 projections, and found no
trainable parameter outside that boundary. Peak allocation for model
quantization plus adapter preparation was 5,662,899,200 bytes. The preferred
BF16-LoRA lane was not selected because this device is below the accepted 48 GiB
minimum.

`ACCEPTED`: these are pre-corpus architecture and runtime-preparation checks.
They used no Dataset-v2 rows and do not claim a production near-maximum-length
forward/backward/update, production context fit, finite training gradients, or
model-quality result. Those gates remain part of the post-binding execution
sequence.

`OBSERVED`: after issue #56 merged, the authoritative Dataset-v2 package bound
without unresolved references or role leakage. `general-train-v2` contains
181,531 statements and 182,812 proof variants. Every optimizer-visible variant
is present exactly once in the declared one-pass membership. The frozen 4x
synthetic multiplier produces 6.8458789% of statement-level training mass; no
domain multiplier is present.

`OBSERVED`: exhaustive serialization with the exact pinned Qwen3.5 tokenizer
selected the 32,768-token context bucket. The maximum full sequence is 19,385
tokens (maximum completion 9,293); no row was truncated, dropped, or packed.
The deterministic fresh-composition views contain 406 validation and 415 test
statements. Their Riemann-filtered subviews contain 100 validation and 104 test
statements with zero training statement or derivation-family overlap.

Commands:

```text
uv sync --project tools/qwen35-generalist --locked
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-architecture-preflight --config config/qwen35-4b-generalist-v2.json --output evidence/qwen-lean-generalist-v2/pre56-architecture.json
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-runtime-preparation-smoke --config config/qwen35-4b-generalist-v2.json --output evidence/qwen-lean-generalist-v2/pre56-runtime-preparation.json
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-bind-dataset --config config/qwen35-4b-generalist-v2.json --package-root data/lean-whole-proof-v2 --view-dir artifacts/qwen-lean-generalist-v2/dataset-binding/views --output evidence/qwen-lean-generalist-v2/dataset-binding.json
```
