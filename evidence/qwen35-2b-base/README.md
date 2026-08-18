# Qwen3.5-2B-Base whole-proof assessment

**OBSERVED:** the official `Qwen/Qwen3.5-2B-Base` foundation was evaluated independently under the unchanged `whole-proof-v1` raw-continuation and Lean-verification contract. This is model-assessment evidence, not a training or automatic-promotion result.

| Workload | Tasks | Candidates | Solved tasks | pass@1 | pass@4 | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 gate | 16 | 64 | 0 | 0.000000 | 0.000000 | 0 | 0 |
| full validation | 244 | 976 | 10 | 0.010246 | 0.040984 | 0 | 0 |

**ACCEPTED:** generation used temperature 0.8, top-p 0.95, no top-k, a 1,024 generated-token cap, seed 0, and four candidates per task. No chat template, extraction, repair, Lean feedback, candidate regeneration, or hosted inference was used. `verifier_timeout` remains an unsuccessful proof outcome rather than an infrastructure error.

**OBSERVED:** the full run generated 426154 tokens; finish reasons were `{"eos": 665, "token_limit": 311}` and evaluator categories were `{"empty_candidate": 0, "generation_error": 0, "lean_rejected": 966, "verified": 10, "verifier_error": 0, "verifier_timeout": 0}`. Generation took 327.905 seconds at 1299.625 generated tokens/second; end-to-end run time was 1339.138 seconds. Compute per solved task was 32.791 generation seconds per solved task.

**OBSERVED:** inference executed locally in BF16 with vLLM 0.17.0 on `NVIDIA RTX 4000 Ada Generation`. Peak observed GPU memory was 19688718336 of 21469593600 bytes. The Apache-2.0 model and tokenizer were pinned to `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`; raw candidates, weights, caches, and bulky logs remain outside Git.
