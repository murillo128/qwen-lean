# Qwen3.5-9B-Base Riemann specialist validation

**OBSERVED:** the immutable `Qwen/Qwen3.5-9B-Base` snapshot `68c46c4b3498877f3ef123c856ecfde50c39f404` completed the full 556-task `riemann-specialist-validation-v1` workload locally on the project RTX 4000 Ada. The run used the accepted unquantized `bf16-text-only-v1` lane and exact `whole-proof-v1` raw continuations; no chat template, extraction, repair, Lean feedback, retry, training, miniF2F rerun, or protected-holdout access occurred.

| Tasks | Candidates | Solved tasks | Verified candidates | pass@1 | pass@4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 556 | 2224 | 5 | 5 | 0.0022482014 | 0.0089928058 |

| Deterministic domain view | Tasks | Solved | pass@1 | pass@4 |
| --- | ---: | ---: | ---: | ---: |
| prime-counting-pnt | 0 | 0 | n/a | n/a |
| arithmetic-functions | 0 | 0 | n/a | n/a |
| zeta-analytic-number-theory | 17 | 0 | 0.0000000000 | 0.0000000000 |
| prime-arithmetic-divisibility | 19 | 0 | 0.0000000000 | 0.0000000000 |
| real-complex-analysis | 57 | 0 | 0.0000000000 | 0.0000000000 |
| broad-number-theory | 54 | 0 | 0.0000000000 | 0.0000000000 |
| shared-formal-prerequisite | 409 | 5 | 0.0030562347 | 0.0122249389 |

**ACCEPTED:** domain and relevance views reuse the first accepted candidate's
`riemann-domain-breakdown-v1` rules exactly (SHA-256
`b5c19a9c6d134c39751f391b88840c506d889cf370fc4931949c3627bc323d2a`). The rules use committed source and graph
metadata only; the legacy multi-label views stored in the generation config were
not used for final classification.

The run generated 683449 tokens in 4029.78 generation seconds and used 19170 MiB peak device memory. Verification recorded 1 unsuccessful timeout proof outcome(s) and 0 unresolved infrastructure errors. `task-outcomes.jsonl` preserves all 556 compact paired outcomes; the final results match the immutable raw checkpoint `156afa6fbdc2fb264540747b68f70483ab52fb6ce1a732331e135d0346f13465` exactly. Raw continuations and model/cache artifacts remain outside Git.
