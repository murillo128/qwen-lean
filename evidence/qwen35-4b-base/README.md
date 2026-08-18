# Qwen3.5-4B-Base foundation assessment

`OBSERVED`: the strict local-GPU assessment completed dev16 and all 244 miniF2F
validation tasks with four raw whole-proof candidates per task and zero unresolved
generation or verifier infrastructure errors.

| Workload | Tasks | Candidates | Solved tasks | pass@1 | pass@4 | Peak GPU MiB | Generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | 16 | 64 | 0 | 0.000000 | 0.000000 | 19576 | 42052 |
| full validation | 244 | 976 | 45 | 0.071721 | 0.184426 | 19226 | 321338 |

`ACCEPTED`: this casting assessment used `Qwen/Qwen3.5-4B-Base` and its
tokenizer at `1001bb4d826a52d1f399e183466143f4da7b741b`, BF16, temperature 0.8, top-p 0.95, no top-k,
a 1,024-token generation limit, and seed 0. Prompts remained exact
`whole-proof-v1` raw continuations with no chat template, extraction, repair, or
Lean feedback. Verifier timeouts remain unsuccessful proof outcomes.

`OBSERVED`: the full run took 1745.89 seconds
(499.09 generation and
1246.80 verification), generated
643.85 tokens/s, and used
38.80 s of total measured run wall time per solved task. Device-level peak
memory was 19226 MiB on
NVIDIA RTX 4000 Ada Generation.

`OBSERVED`: Qwen3.5 support required the isolated vLLM build
`0.27.2rc1.dev203+g41f179b57` at `41f179b57aa8ab6f634f508128ce1f1efadd0eb1`. Text-only
`language_model_only=true` omitted the unused vision encoder while preserving the
language model and raw continuation contract. The pinned Hub snapshot was
resolved to a local path to avoid revision loss inside the worker, and native
top-p sampling was selected because FlashInfer JIT required an unavailable CUDA
toolkit. Raw candidates, model caches, and bulky logs remain local and ignored
by Git.
