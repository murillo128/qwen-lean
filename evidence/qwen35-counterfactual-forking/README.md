# Counterfactual forking signal

This directory contains compact evidence for issue #92. Raw suffix token arrays,
combined reasoning trajectories, model caches, and verifier traces remain outside
Git in the restart-safe artifact directory.

**OBSERVED:** `weak_or_uninformative_under_frozen_model_interface_and_budget`.

- Eligible tasks: 30
- Discovery branches: 1260
- Tasks with operational value range at least 0.5: 0
- Confirmed intervals: 0
- Confirmation sign replication: None
- Stable positive / negative intervals: 0 / 0
- Lean-verified branches: 1

Discovery `Delta_op` is descriptive, not causal: adjacent states have different
remaining token budgets. Confirmation compares both sides with the later
prefix's remaining budget and independent seeds. This measurement does not
authorize or select DPO, RLVR, value learning, replay, or planner changes.

## Material limitations

- bounded 30-task candidate-index-0 Mathia-guided T1 sample
- token-quantile checkpoints are not semantic decision boundaries
- operational discovery deltas mix state with consumed token budget
- matched-budget confirmation uses 12 Bernoulli branches per side
- go/no-go thresholds are heuristics rather than statistical theorems
- negative or weak signal is retained without selecting a training method
- execution required restart after 6 failed or interrupted discovery generation segments; the completed restart-safe JSONL is the authoritative sample
- vLLM emitted-text diagnostics differed from tokenizer-decoded authoritative token IDs in 7 persisted branches; parsing and verification used the token-ID reconstruction
- OBSERVED: fixed per-request seeds did not reproduce identical outcomes for at least one interrupted asynchronous vLLM request after process restart
