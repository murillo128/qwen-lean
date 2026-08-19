# Qwen3.5-4B-Base Riemann foundation casting

`OBSERVED`: the frozen `riemann-specialist-validation-v1` assessment completed all 556 validation
tasks with four candidates per task and zero unresolved generation or verifier
infrastructure errors. It verified 13 candidates and
solved 13 tasks: pass@1
0.005845, pass@4
0.023381.

| Deterministic source domain | Tasks | Solved | pass@1 | pass@4 |
| --- | ---: | ---: | ---: | ---: |
| prime-counting-pnt | 0 | 0 | n/a | n/a |
| arithmetic-functions | 0 | 0 | n/a | n/a |
| zeta-analytic-number-theory | 17 | 0 | 0.000000 | 0.000000 |
| prime-arithmetic-divisibility | 19 | 0 | 0.000000 | 0.000000 |
| real-complex-analysis | 57 | 1 | 0.004386 | 0.017544 |
| broad-number-theory | 54 | 1 | 0.004630 | 0.018519 |
| shared-formal-prerequisite | 409 | 11 | 0.006724 | 0.026895 |

`ACCEPTED`: the run reused `Qwen/Qwen3.5-4B-Base` and its tokenizer at
`1001bb4d826a52d1f399e183466143f4da7b741b` exactly as accepted by issue #44: BF16, no quantization, the
text-only lane, and no chat template. The common casting contract used raw
`whole-proof-v1` continuations, temperature 0.8, top-p 0.95, no top-k, four
candidates, a 1,024-token limit, and seed 0. No extraction, semantic repair,
Lean-feedback retry, or candidate regeneration was applied.

`OBSERVED`: generation took 776.02
seconds, verification took 2985.27
seconds, and the run generated 480436 tokens at
619.10 tokens/s. Per solved task,
generation used 36956.62 generated tokens and 59.69 generation seconds. Device-level peak memory was
19030 MiB on
NVIDIA RTX 4000 Ada Generation.

`OBSERVED`: domain labels use committed source path/declaration metadata only.
Direct graph relevance, relevance distance, component inclusion, and
component-associated prerequisite views remain separate in `full.json` and
`task-outcomes.jsonl`; model outputs never define a category. Protected near and
far holdouts were not loaded. Raw generations, model caches, and bulky logs stay
outside Git.
