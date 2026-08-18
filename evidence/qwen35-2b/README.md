# Qwen3.5-2B efficiency baseline

**ACCEPTED:** the strict lane used the immutable official `Qwen/Qwen3.5-2B` checkpoint and tokenizer at `15852e8c16360a2fea060d615a32b45270f8a8fc` in BF16 on the project RTX 4000 Ada. Inputs were exact `whole-proof-v1` raw continuations with no chat wrapper, proof extraction, verifier feedback, or repair.

**OBSERVED:** dev16 completed 16 tasks and 64 candidates with pass@1 0.000000 and pass@4 0.000000.

**OBSERVED:** the complete miniF2F validation run completed 244 tasks and 976 candidates. It verified 3 candidates across 3 tasks, with pass@1 0.003074 and pass@4 0.012295.

**OBSERVED:** versus retained `reference-sft-v1`, the Qwen3.5-2B pass@1/pass@4 deltas were -0.036373/-0.090867. The strict lane used four candidates per task as mandated here; the retained reference used eight, while the shared prompt, sampling parameters, tasks, and verifier semantics remained aligned.

**OBSERVED:** generation produced 972291 tokens in 663.658 seconds (1465.049 tokens/second, including engine initialization). Peak observed GPU memory was 18372 MiB. Full latency, finish-reason, token, category, and compute-per-solved-task evidence is retained in `full.json`.

**ACCEPTED:** the run completed with zero generation/verifier infrastructure errors. `lean_rejected`, `empty_candidate`, and `verifier_timeout` are unsuccessful proof outcomes; verifier timeouts do not count as infrastructure errors or authorize regeneration.
