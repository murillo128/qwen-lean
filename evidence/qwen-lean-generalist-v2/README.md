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

`ACCEPTED`: the pre-corpus architecture and runtime-preparation checks used no
Dataset-v2 rows and made no production context-fit or model-quality claim. The
post-binding production gate below supplies the required context-fit evidence;
bounded training and evaluation remain separate later gates.

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

`OBSERVED`: the complete pre-update Q0 control used eight candidates per task
and recorded zero generation or verifier infrastructure errors. The pinned base
solved 16/406 fresh-composition validation tasks (pass@1/pass@4/pass@8
0.005234/0.020408/0.039409), 66/244 clean miniF2F validation tasks
(0.066086/0.185831/0.270492), 9/256 Dataset-v2 train-probe tasks
(0.005371/0.019810/0.035156), and 2/100 fresh Riemann validation tasks
(0.002500/0.010000/0.020000). The train probe reproduced no stored target
exactly at Q0. `q0.json` retains the ordered task outcomes and raw-artifact
hashes; test workloads were not consulted and the Riemann result remains
diagnostic-only.

`OBSERVED`: the production preflight serialized the exact maximum Dataset-v2
row to 19,385 tokens and completed one real NF4 QLoRA forward, backward, and
optimizer update on the project-controlled RTX 4000 Ada GPU. All 496 intended
LoRA parameter tensors had present finite gradients, 248 tensors changed on the
first update, and the checked frozen base parameter remained unchanged. The
run used non-reentrant decoder checkpointing, activation CPU offload, 64-token
DeltaNet and target-only LM-head loss chunks, and checkpointed 1,024-token
sequence chunks in all 32 decoder MLPs. Peak CUDA allocation was 18,143,714,816
bytes and peak reservation was 18,440,257,536 bytes, leaving 2,549,547,008
bytes of reserved-memory headroom against the 536,870,912-byte gate.
`production-preflight.json` retains the exact example, parameter inventory,
runtime lock, and measurements.

Commands:

```text
uv sync --project tools/qwen35-generalist --locked
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-architecture-preflight --config config/qwen35-4b-generalist-v2.json --output evidence/qwen-lean-generalist-v2/pre56-architecture.json
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-runtime-preparation-smoke --config config/qwen35-4b-generalist-v2.json --output evidence/qwen-lean-generalist-v2/pre56-runtime-preparation.json
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-bind-dataset --config config/qwen35-4b-generalist-v2.json --package-root data/lean-whole-proof-v2 --view-dir artifacts/qwen-lean-generalist-v2/dataset-binding/views --output evidence/qwen-lean-generalist-v2/dataset-binding.json
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-q0-evidence --config config/qwen35-4b-generalist-v2.json --evaluation-root artifacts/qwen-lean-generalist-v2/q0 --output evidence/qwen-lean-generalist-v2/q0.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-production-preflight --config config/qwen35-4b-generalist-v2.json --package-root data/lean-whole-proof-v2 --binding evidence/qwen-lean-generalist-v2/dataset-binding.json --q0-evidence evidence/qwen-lean-generalist-v2/q0.json --model-snapshot /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b --output evidence/qwen-lean-generalist-v2/production-preflight.json
```
