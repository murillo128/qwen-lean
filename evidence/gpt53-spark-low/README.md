# GPT-5.3-Codex Spark low-reasoning ablation

**ACCEPTED:** all low-arm candidates used fresh nested Codex CLI executions with the explicit `gpt-5.3-codex-spark` model pin, `low` reasoning override, ChatGPT authentication, an isolated empty working directory, a sanitized PATH without Lean/Lake/Elan, disabled tool surfaces, and raw final-message verification without repair.

**OBSERVED:** dev16 completed 16/16 candidates with pass@1 0.000000. The complete miniF2F validation run completed 244/244 candidates with 35 verified proofs and pass@1 0.143443.

**OBSERVED:** the frozen xhigh control verified 49/244 tasks (pass@1 0.200820). Low and xhigh solved 27 tasks in common; 22 were solved only by xhigh, 8 only by low, and 187 by neither. The exact two-sided McNemar p-value is 0.0161248.

**OBSERVED:** reducing reasoning effort materially reduced reported test-time reasoning output: 5,518,772 xhigh tokens versus 1,670,247 low tokens (69.74% reduction). Verified proof success decreased from 49 to 35. Latency, other usage fields, verifier timeouts, retries, and proof-per-token efficiencies are recorded in `compute.json`.

**ACCEPTED:** `lean_rejected`, `empty_candidate`, and `verifier_timeout` remain unsuccessful proof attempts and never authorize candidate regeneration. Accepted executions contained 0 external-tool events and 1 bounded generation-infrastructure retries.

**ACCEPTED:** this is descriptive evidence from one frozen low-versus-xhigh effort ablation. The relative ratio and efficiency values are not compute-independent model-quality claims, and no causal claim is made beyond this exact procedure.
