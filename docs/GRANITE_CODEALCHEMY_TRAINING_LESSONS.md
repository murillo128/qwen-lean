# Granite 4.2 and CodeAlchemy training lessons

## Status

This note records external evidence that is relevant to qwen-lean post-training, planner training, and future interactive Lean proving.

It is **not** a source of accepted project hyperparameters or roadmap ordering. Repository decisions remain in `docs/DECISIONS.md`, sequencing remains in `PLAN.md`, and execution details remain in controlling issues.

Primary sources:

- IBM Granite 4.2 training description, published 2026-08-25: <https://huggingface.co/blog/ibm-granite/granite-4-2>
- IBM Research CodeAlchemy release, published 2026-07-16: <https://research.ibm.com/blog/code-alchemy-for-synthetic-code>

## OBSERVED — Granite 4.2 uses SFT followed by staged verifier-driven RL

IBM reports the following broad post-training pattern for Granite 4.2:

```text
SFT
 -> RLVR
 -> focused skill boosters
 -> agentic RL in real environments (8B/30B)
 -> RLHF
```

The foundational RL stages use GRPO and a mixture of reward types. Crucially for this repository, the RLVR math mixture explicitly includes **formal proving in Lean**, with task-specific verification grounding reward.

IBM reports multiple generations per prompt during RLVR and trains later agentic stages on complete trajectories in real environments. The exact group sizes, learning rates, KL settings, distributed infrastructure, and sequence lengths are specific to Granite's large-scale training environment and must not be copied into qwen-lean without local evidence.

### Project implications

This external result strengthens, but does not settle, several existing qwen-lean hypotheses:

- Lean verification is suitable as a primary optimization signal, not only an evaluation metric.
- SFT can be treated as a bootstrap that teaches the output/interface distribution before verifier-driven optimization decides what behavior actually succeeds.
- A group-relative method is especially natural when multiple candidate proofs or plans can be sampled for the same obligation and compared by exact verifier outcome.
- Reward density and within-group variation must be measured before scaling RL; all-zero groups do not provide useful relative signal.
- Candidate-level outcomes should be retained even when final reporting uses pass@K/solved-within-K.
- Future interactive Lean proving can be treated as a real-environment agent problem: act, observe a proof state or diagnostic, revise, and continue until Lean accepts the theorem or the budget is exhausted.

These points are consistent with the existing Phase 8 verifier-reward branch, Phase 9 proof-state supervision, Phase 11 tactic-level milestone, and `docs/QWEN_LEAN_PLANNER_DIRECTION.md`.

## OBSERVED — specialized SFT with replay is used as a preservation mechanism

For Granite 4.2 30B, IBM reports a second SFT phase focused on agentic coding. Agentic/SWE/coding data is upsampled, approximately 16% of the mixture remains replay data from the original SFT corpus, and the second phase uses a lower learning rate.

IBM describes the intent as increasing exposure to the specialized trajectories without discarding capabilities acquired in the original SFT stage.

### Project implications

This is relevant to the qwen-lean v3 anti-collapse problem and to any later narrow planner/domain specialization.

A future focused SFT stage should consider **replay from broader earlier data as one preservation control**, alongside or against the project's existing KL/distillation-style Base-preservation mechanisms. The useful experiment is not “use 16% replay”; it is whether a bounded replay mixture improves specialization while preserving Base/general Lean search diversity.

The exact replay fraction, learning-rate change, anchor distribution, and interaction with QLoRA are all project-specific and remain experimental.

## OBSERVED — CodeAlchemy pairs generated code with actual execution traces

IBM Research reports that CodeAlchemy contains more than 1.3 million validated code-file/execution-trace pairs generated through sandbox execution.

The motivating distinction is directly relevant to Lean:

```text
static artifact: what valid code/proof looks like
execution trace: what happens as it executes / transforms state
```

For Lean, the mechanically grounded analogue is stronger because proof states and tactic outcomes are part of the proof checker itself.

A future `LeanTrace` view can preserve trajectories such as:

```text
proof state
 -> tactic/action
 -> resulting proof state
```

and later interactive failure-recovery trajectories such as:

```text
proof state
 -> attempted tactic
 -> Lean error/diagnostic
 -> revised tactic
 -> resulting proof state
```

A Lean diagnostic is an environment observation, not a mathematical label. Only successful elaboration/proof checking establishes a valid transition or completed proof.

### Project implications

- Dataset v3's proof-prefix continuation is a useful first step toward process-aware training without requiring raw proof-state conditioning.
- Phase 9 can later evaluate explicit Lean-grounded state/action/state data rather than free-form rationale supervision.
- Phase 11 can use the same environment information online, making recovery from failed tactics part of the policy rather than only an offline dataset feature.
- Stored traces should preserve enough state/action/result information to distinguish syntax failure, elaboration failure, valid progress, and complete verification.

## OBSERVED — synthetic diversity can beat substantially more real data in a specific IBM experiment

IBM reports a CodeAlchemy experiment in which a Granite 4.0 3B model trained on 100B tokens of synthetic code outperformed the same model trained on 600B tokens of real code. IBM also reports that, in its rewriting experiments, data rewritten by a smaller Gemma 4B model produced better downstream results than data rewritten by GPT-OSS 20B, with diversity offered as a likely explanation.

IBM further reports that mixing 5% of the original lower-quality code back into the synthetic mixture improved some benchmarks.

These are specific experimental observations, not universal scaling laws.

### Project implications

For qwen-lean, **verified diversity matters more than mechanically multiplying near-duplicate proofs**.

Useful synthetic augmentation candidates include:

- genuinely different verified proofs for the same theorem;
- different decomposition and lemma routes;
- different proof-form families (`calc`, `apply`, `refine`, rewriting, induction, constructor-style, etc.);
- verified proof-prefix continuation examples at structurally valid boundaries;
- alternative planner outputs that are judged by downstream prover+Lean utility rather than teacher resemblance;
- failure/recovery traces once an interactive Lean environment exists.

Any synthetic variant must remain verifier-centered and contamination-aware. Cosmetic rewrites must not count as diversity, theorem-level optimizer mass must remain controlled, and held-out obligations must not be transformed into training examples simply to increase volume.

This is consistent with Dataset v3's source-form preservation, structural proof fingerprints, theorem-normalized training mass, and clean evaluation isolation.

## OPEN — staged qwen-lean training hypothesis

The combined external evidence suggests a useful future hypothesis for this project:

```text
broad/source-preserving SFT
        |
        v
non-collapsed Lean executor
        |
        v
verifier-reward optimization
        |
        v
optional focused specialization with preservation/replay
        |
        v
interactive Lean environment training/search
```

For the planner, the analogous hypothesis is:

```text
Base planner
   |
optional bounded SFT bootstrap
(plan language/interface only; no source-proof leakage)
   |
   v
planner RLVR / group-relative optimization
   |
   v
reward from frozen qwen-lean + Lean
```

Neither sequence is accepted merely because IBM used a related pattern. Existing independent-branch comparisons should still be completed when they answer an isolatable project question, and new compositions should be introduced under explicit controlling issues.

## What not to copy

Do not treat any of the following Granite/CodeAlchemy details as qwen-lean defaults:

- Granite learning rates;
- group size or generations per prompt;
- KL coefficients;
- context lengths;
- full-parameter/distributed infrastructure assumptions;
- the approximately 16% replay fraction;
- the 5% original-code mixture;
- absolute token counts;
- claims that a smaller synthetic-data generator is always better.

The transferable lesson is the **experimental structure**: bootstrap with supervised data when useful, preserve diversity/capability during specialization, optimize against objective execution/verifier outcomes, and eventually train on real environment trajectories rather than static text alone.
