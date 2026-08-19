# OLMo 3 7B whole-proof assessment

**OBSERVED:** the official `allenai/Olmo-3-1025-7B` Base checkpoint was evaluated under the
unchanged `whole-proof-v1` raw-continuation and Lean-verification contract. This
is independent model-assessment evidence, not a training or promotion result.

| Workload | Tasks | Candidates | Solved tasks | pass@1 | pass@4 | Verified candidates | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 smoke | 16 | 64 | 0 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| full validation | 244 | 976 | 0 | 0.000000 | 0.000000 | 0 | 0 | 3 |

**ACCEPTED:** generation used temperature 0.8, top-p 0.95, no top-k, a 1,024
generated-token cap, seed 0, and four candidates per task. No chat template,
proof extraction, repair, Lean feedback, candidate regeneration, or hosted
inference was used.
`verifier_timeout` remains an unsuccessful proof outcome.

**OBSERVED:** the full run generated 610473
tokens; finish reasons were `{"eos": 445, "token_limit": 531}`
and evaluator categories were `{"empty_candidate": 0, "generation_error": 0, "lean_rejected": 973, "verified": 0, "verifier_error": 0, "verifier_timeout": 3}`.
Generation took 4103.676 seconds at
148.762 generated tokens/second;
end-to-end run time was 5879.112 seconds. Compute
per solved task was unavailable because no task was solved.

**OBSERVED:** inference executed locally in BF16 without quantization using vLLM
0.12.0 on `NVIDIA RTX 4000 Ada Generation`. Peak observed
GPU memory was 20017971200 of
20989804544 bytes. The Apache-2.0 model and
tokenizer are pinned to `a81bae42db3975be1671e27b9c9a56da1a9f980f`. The isolated runtime package versions
are retained in the JSON evidence; weights, caches, raw candidates, and bulky
logs remain outside Git.
