# DeepSeek-Prover-V2-7B strict Lean specialist assessment

**OBSERVED:** `deepseek-ai/DeepSeek-Prover-V2-7B` completed all 244 miniF2F validation
tasks and 976 raw candidates. It verified
148 candidates across
81 tasks. pass@1/pass@4 were
0.151639/0.331967. The accepted qwen-lean anchors are `Qwen/Qwen3-8B-Base` 0.012807/0.047717, `Qwen/Qwen3-8B` 0.004098/0.016393, `Qwen/Qwen3.5-4B-Base` 0.071721/0.184426, `reference-sft-v1` 0.039447/0.103162.

The dev16 smoke completed 64 candidates with
pass@1/pass@4 of 0.031250/0.062500. Both runs contain zero unresolved generation or
verifier infrastructure errors. The full run retains
251 `verifier_timeout` outcomes as unsuccessful
proofs. Compute per solved task was 76.20 run-wall seconds.

Concurrent non-assessment Lean work was observed on the shared host during full
verification. Generation timing is unaffected, but verification and total wall
time are descriptive system measurements. The frozen 30-second verifier
timeouts were not retried or reclassified.

**ACCEPTED:** the primary score uses exact `whole-proof-v1` raw continuation,
four candidates per task, temperature 0.8, top-p 0.95, no top-k, 1,024 new
tokens, seed 0, no chat wrapper, no proof extraction, no verifier feedback, and
no repair. It ran in BF16 without quantization using local vLLM
`0.10.2` on NVIDIA RTX 4000 Ada Generation.

The model and tokenizer are pinned to `a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b`. No optional native
prompt/reasoning diagnostic was run. The repository's model license is MIT;
weights, caches, and raw candidates remain outside Git. Compact JSON retains
execution identity, category and finish-reason counts, token/latency summaries,
wall times, throughput, and compute per solved task.

`comparison.json` uses only accepted qwen-lean evidence. No accepted Goedel
measurement currently exists, so the Goedel entry is explicitly unavailable;
no failed or partial sibling-run observation was imported. Historical
`Qwen/Qwen3-8B-Base` and `reference-sft-v1` anchors used eight candidates while
the strict DeepSeek and other named Qwen lanes used four.
