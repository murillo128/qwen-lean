# GPT-5.3-Codex Spark one-shot assessment

**ACCEPTED:** all benchmark candidates used fresh nested Codex CLI executions with the explicit `gpt-5.3-codex-spark` model pin, `xhigh` reasoning override, ChatGPT authentication, an isolated empty working directory, a sanitized PATH without Lean/Lake/Elan, disabled tool surfaces, and raw final-message verification without repair.

**OBSERVED:** dev16 completed 16/16 candidates with pass@1 0.000000. The complete miniF2F validation run completed 244/244 candidates with 49 verified proofs and pass@1 0.200820.

**ACCEPTED:** the unchanged Phase 1 miniF2F verifier timeout is 30 seconds. The full run's 8 `verifier_timeout` outcomes are unsuccessful proof attempts, not infrastructure errors; they count in the 244-candidate denominator and do not authorize candidate regeneration or verification retry.

**OBSERVED:** accepted executions contained 0 external-tool events. Child-process accounting records 2 infrastructure retries separately from proof outcomes, and accepted executions contained zero detected non-Spark GPT-5.3, GPT-5.6, model-migration, substitution, or fallback markers. Full JSONL event streams and raw final messages remain in ignored local `artifacts/` storage.

**OBSERVED:** on the same 244-task validation set, accepted Qwen3-8B-Base pass@1 was 0.012807 and `reference-sft-v1` pass@1 was 0.039447. Spark used one isolated xhigh reasoning execution rather than the Phase 1 eight-sample stochastic generation procedure, so this is verifier-aligned but not an identical inference process.
