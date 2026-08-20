# DeepSeek-Prover-V2-7B Riemann casting

`OBSERVED`: the frozen `riemann-specialist-validation-v1` assessment completed all 556 validation
tasks with four candidates per task and zero unresolved generation or verifier
infrastructure errors. It verified 319 candidates and
solved 176 tasks: pass@1
0.143435, pass@4
0.316547.

| Deterministic source domain | Tasks | Solved | pass@1 | pass@4 |
| --- | ---: | ---: | ---: | ---: |
| prime-counting-pnt | 0 | 0 | n/a | n/a |
| arithmetic-functions | 0 | 0 | n/a | n/a |
| zeta-analytic-number-theory | 17 | 0 | 0.000000 | 0.000000 |
| prime-arithmetic-divisibility | 19 | 3 | 0.052632 | 0.157895 |
| real-complex-analysis | 57 | 10 | 0.065789 | 0.175439 |
| broad-number-theory | 54 | 4 | 0.018519 | 0.074074 |
| shared-formal-prerequisite | 409 | 159 | 0.180929 | 0.388753 |

`ACCEPTED`: the run reused `deepseek-ai/DeepSeek-Prover-V2-7B` and its tokenizer at
`a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b` exactly as accepted by issue #31:
BF16, no quantization, and no chat template. The common casting contract used raw
`whole-proof-v1` continuations, temperature 0.8, top-p 0.95, no top-k, four
candidates, a 1,024-token limit, and seed 0. No extraction, semantic repair,
Lean-feedback retry, or candidate regeneration was applied.

`ACCEPTED`: For epic #63 this is a specialist-parent comparator for practical training-parent selection; it is not a like-for-like general-foundation result.

`OBSERVED`: generation took 6427.94
seconds, verification took 5396.22
seconds, and the run generated 963476 tokens at
149.89 tokens/s. Per solved task,
generation used 5474.30 generated tokens and 36.52 generation seconds. Device-level peak memory was
18636 MiB on
NVIDIA RTX 4000 Ada Generation.

`OBSERVED`: domain labels use committed source path/declaration metadata only.
Direct graph relevance, relevance distance, component inclusion, and
component-associated prerequisite views remain separate in `full.json` and
`task-outcomes.jsonl`; model outputs never define a category. Protected near and
far holdouts were not loaded. Raw generations, model caches, and bulky logs stay
outside Git.

`OBSERVED`: Paired against the accepted Qwen3.5-4B-Base task vector, both solved 10 tasks, DeepSeek-Prover-V2-7B alone solved 166, and Qwen3.5-4B-Base alone solved 3 (exact two-sided McNemar p=2.15053e-45). Paired against the accepted Qwen3.5-9B-Base task vector, both solved 5 tasks, DeepSeek-Prover-V2-7B alone solved 171, and Qwen3.5-9B-Base alone solved 0 (exact two-sided McNemar p=6.68191e-52).

`OBSERVED`: Concurrent non-assessment Lean 4.32.2 verification was observed on the shared host during full verification. Generation timing is unaffected; verification and total wall time are descriptive and contention may have contributed to frozen 30-second verifier_timeout outcomes, which were not retried or reclassified.
