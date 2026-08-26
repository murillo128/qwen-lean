# qwen-lean-planner direction

## Status

This document records a new architectural direction for a separate learned proof-search planner between Mathia-style conceptual intuition and qwen-lean formal execution.

It is a durable architecture note, not a roadmap phase, execution order, or modification to `PLAN.md`.

Controlling design issue: #80.
Cross-repository conceptual-input design: `murillo128/mathia#52`.

## Motivation

The current conceptual/formal picture has a missing level of abstraction. A high-level intuition may identify the right mechanism while still leaving too much work for a whole-proof prover:

```text
Mathia intuition
      |
      v
   ???
      |
      v
qwen-lean whole-proof search
```

The proposed bridge is a separate planner whose job is to operationalize a mathematical idea into a Lean-aware proof-search strategy while leaving actual proof construction to qwen-lean.

```text
Mathia/oracle intuition
          +
   exact Lean theorem/goal
          |
          v
 qwen-lean-planner
          |
          v
  qwen-lean prover
          |
          v
        Lean
```

This separates:

- conceptual representation;
- proof-search planning;
- formal execution;
- exact verification.

## ACCEPTED — separate model role

`qwen-lean-planner` is not another mode of the qwen-lean prover checkpoint. It is a separately trained and separately identifiable model/checkpoint.

The first experiment should freeze the prover while learning/evaluating the planner so that changes in downstream proof yield can be attributed to the planning layer.

Do not jointly train planner and prover in the first experiment.

## ACCEPTED — Qwen3.5-4B-Base foundation

The initial planner should be a descendant of `Qwen/Qwen3.5-4B-Base`, using the exact immutable revision accepted by the current 4B qwen-lean foundation work when implementation is frozen.

Initialize from the Base model rather than from a qwen-lean whole-proof SFT adapter.

Conceptually:

```text
                Qwen3.5-4B-Base
                       |
              +--------+--------+
              |                 |
              v                 v
          qwen-lean       qwen-lean-planner
       formal execution    search planning
```

The roles may share a foundation family and tokenizer without sharing post-training weights.

## ACCEPTED — no gold planner target as the core training signal

The planner should not be trained primarily by extracting a human/teacher “correct plan” from known proofs.

Known verified proofs may be used upstream to create proof-aware oracle intuitions under the Mathia-side leakage contract, but the proof itself must never be planner-visible.

The planner instead generates candidate plans and learns from their **downstream utility**.

## OPEN — bounded planner SFT bootstrap before verifier-reward optimization

A short SFT bootstrap is compatible with the no-gold-plan principle if its purpose is to teach the planner interface, vocabulary, decomposition style, and plan/proof boundary rather than to define the final notion of a correct plan.

A bootstrap dataset must be generated only from planner-visible inputs such as the theorem plus accepted oracle intuition. It must not expose or mechanically extract the hidden source proof. Candidate bootstrap targets may come from teacher-generated plans, synthetic plans, or verifier-filtered planner samples, but the durable optimization signal remains downstream formal utility.

The intended conceptual sequence, if this bootstrap proves useful, is:

```text
Qwen3.5-4B-Base
       |
       v
bounded planner SFT
learn plan language / interface
       |
       v
planner RLVR / GRPO
optimize plans by downstream Lean utility
```

This is **OPEN**, not an authorization to create a gold tactic-plan corpus from source proofs and not a requirement that every planner experiment use SFT. The planner-variance pre-test and reward-density evidence should determine whether the bootstrap improves the learning problem.

## ACCEPTED — downstream environment is frozen qwen-lean + Lean

For a theorem `T`, accepted oracle intuition `I`, and candidate plan `P`:

```text
(T, I)
  |
  v
planner -> P
           |
           v
    frozen qwen-lean x K
           |
           v
    Lean verification
           |
           v
       reward(P)
```

Only Lean-accepted complete proofs count as successful formal outcomes for the first clean reward channel.

Ordinary qwen-lean failure is not mathematical refutation. Verifier infrastructure errors/timeouts must remain distinguishable from Lean rejection.

## ACCEPTED — preserve more information than binary pass@K during learning

A binary `pass@K` indicator is useful for final operational evaluation but wastes information for planner optimization.

If a fixed plan produces:

```text
0 / K verified proofs
1 / K verified proofs
...
K / K verified proofs
```

that success fraction provides an immediately available plan-level signal.

The exact reward transformation remains OPEN, but the design should preserve candidate-level outcomes so alternatives such as raw success fraction, baseline-relative advantage, or group-relative normalization can be evaluated without regenerating all formal attempts.

Credit assignment must remain attached to the individual sampled plan. If one plan among a group causes a verified proof, that success should not be broadcast as positive reward to unrelated plans in the same group.

## ACCEPTED — usefulness is relative to the prover baseline

The same frozen qwen-lean worker must also run without planner guidance under a matched formal-compute budget.

A theorem that qwen-lean already solves easily should not cause arbitrary plans to be interpreted as good planning.

The planner experiment must distinguish:

```text
no-plan baseline
plan improves success probability
plan is neutral
plan harms success probability
```

The exact baseline-relative reward formula remains OPEN.

## ACCEPTED — proof-aware oracle intuition is the first conceptual condition

Planner training should initially remove uncertainty about Mathia's theorem-only capability.

The conceptual input therefore comes from the Mathia-side **proof-aware oracle-intuition mode**:

```text
known theorem + source proof
        |
        v
proof-aware oracle intuition
        |
    leakage gate
        |
        v
accepted natural-language intuition
        |
        v
qwen-lean-planner
```

The planner never receives the source proof. A planner-visible artifact must already have passed the Mathia leakage gate.

This condition asks a narrow question:

> If the conceptual layer has a good idea, can a learned 4B planner make that idea materially more useful to qwen-lean?

## Planner output level

The planner is intentionally closer to Lean execution than Mathia.

Candidate output may include:

- an ordered attack plan;
- useful reductions and representations;
- intermediate goals/sublemmas;
- decomposition into branches;
- premise or lemma candidates;
- tactic families worth trying;
- fallback routes after a likely failure;
- instructions about what not to expand or normalize prematurely.

The planner is allowed to be Lean-aware.

However it should remain measurably distinct from a second whole-proof prover. The first design must define and test the boundary between a search program/plan and an executable complete Lean proof.

Whether small Lean proposition snippets or partial skeletons are allowed is OPEN.

## Planner-variance pre-test before RL

Before choosing an optimization algorithm, first test whether candidate plans generate a useful reward landscape.

For each fixed theorem and oracle intuition:

```text
               P1 -> qwen-lean x K -> Lean
              /
(T, I) -> planner samples P2 -> qwen-lean x K -> Lean
              \
               ...
```

Measure at least:

- per-plan verified-success fraction;
- within-theorem reward variance;
- overlap in successful proofs/problems across plans;
- uplift/harm versus the matched no-plan baseline;
- proportion of all-zero groups;
- formal-compute cost.

The key diagnostic is whether different plans meaningfully change the prover's search distribution.

If almost every plan has identical reward, especially all zero, the reward channel is not yet suitable for planner RL. Increasing training complexity must not hide that negative result.

## SPECULATIVE — group-relative verifier-reward optimization

If the pre-test shows usable reward variance, a natural first optimization family is RL with verifiable rewards, potentially a group-relative method such as GRPO.

The attraction is structural:

```text
same theorem + same intuition
        |
   several plans
   /    |     \
 low  medium  high downstream reward
        |
        v
increase probability of the more useful plans
```

No teacher plan is required because Lean ultimately validates the behavior induced by each plan.

No production-scale hyperparameter should be copied from unrelated large-model recipes. Group size, KL behavior, learning rate, rollout count, and candidate budgets must be measured for the selected Qwen/QLoRA setup and local verifier throughput.

This method is not frozen by this document. #80 must still compare the simplest viable verifier-reward approach against verifier-filtered self-training or preference-style alternatives.

## Curriculum/reward-density principle

Planner training should not begin only on the hardest research problems.

A useful training pool needs the frozen prover to have enough headroom that some plans can help and some can hurt. Otherwise all plans receive the same zero reward.

The exact theorem-selection policy is OPEN, but it should intentionally include problems around the prover's capability frontier and preserve harder tasks for evaluation/generalization.

Riemann is a later consumer of the architecture, not the domain that should define the first planner reward landscape.

## Solver-specific steering risk

A planner trained against one frozen qwen-lean checkpoint may learn idiosyncratic prompts for that model rather than transferable proof-search planning.

This is still operationally useful, but claims must remain solver-conditional until transfer is tested.

Useful later controls include:

- another qwen-lean checkpoint;
- changed theorem notation/presentation;
- changed planner-to-prover prompt wrapper;
- another formal prover where practical.

A large in-distribution gain with no transfer should be described as checkpoint-specific steering rather than broad mathematical planning capability.

## OBSERVED — Granite 4.2 provides an external precedent for the staged pattern

IBM's Granite 4.2 training description, published 2026-08-25, is directly relevant external evidence: <https://huggingface.co/blog/ibm-granite/granite-4-2>.

The reported recipe uses supervised fine-tuning followed by multi-stage GRPO. Its foundational RLVR mixture explicitly includes **formal proving in Lean**, with task-specific verifiers grounding reward. For the 30B model, IBM also reports a second specialized SFT stage that keeps approximately 16% replay from the original SFT mixture while lowering the learning rate. The 8B and 30B models then add multi-turn agentic RL in real environments where the model acts, observes outcomes, recovers from errors, and receives outcome reward.

This does not prove that the same recipe or hyperparameters will work for qwen-lean-planner, but it strengthens several design hypotheses already present here:

- use SFT, when needed, as a behavioral/interface bootstrap rather than the final correctness oracle;
- use Lean verification as the decisive downstream reward signal;
- preserve per-candidate/group reward information instead of collapsing learning to aggregate pass@K;
- choose a curriculum with enough positive-reward density for group-relative learning;
- consider replay from broader pre-specialization data as an anti-forgetting/anti-collapse control during later focused SFT, without copying IBM's 16% literally;
- treat interactive Lean execution as a future real-environment trajectory problem rather than only a static text-generation task.

The external precedent therefore supports the broad direction `SFT/bootstrap -> verifier-reward optimization -> later environment-interactive training`, while leaving all project-specific implementation choices evidence-driven.

## OPEN — Lean execution traces and failure-recovery trajectories

A future process/agentic training view can preserve more than final proof text. Lean can expose mechanically grounded trajectories such as:

```text
proof state
  -> tactic/action
  -> resulting proof state
```

and, in a later interactive environment:

```text
proof state
  -> attempted tactic
  -> Lean diagnostic / failure
  -> revised tactic
  -> new proof state
```

Successful state transitions remain Lean-grounded; diagnostics are environment observations rather than mathematical labels. These traces may be useful both for process supervision and for later multi-turn RL where the policy learns to recover from failed formal actions.

This is not part of the first static planner experiment. It connects the planner direction to the existing proof-state process-supervision and tactic-level proving milestones, and should be implemented only under their own execution contracts.

## OPEN — replay as a specialization-preservation control

The Granite 4.2 second-stage SFT result motivates testing replay when qwen-lean or qwen-lean-planner undergoes a narrow specialization stage. A focused stage may mix a minority of broad earlier SFT examples or another explicit preservation objective so specialization does not erase useful general behavior.

The exact mechanism is OPEN. Replay should be compared against the project's existing Base-preservation/KL-style controls where relevant; neither the approximately 16% Granite mixture nor its learning rate is a transferable default.

## Future Mathia optimization

If the planner becomes useful under oracle intuition, the same interface provides a stronger future reward for Mathia:

```text
theorem only
    |
    v
  Mathia
    |
    v
frozen planner
    |
    v
frozen qwen-lean
    |
    v
   Lean
    |
    v
reward Mathia intuition
```

This would let Mathia learn which theorem-only conceptual representations actually survive the planning and formal-execution pipeline, rather than optimize for resemblance to a frontier teacher.

This future Mathia training belongs to the Mathia repository and is not part of the first planner experiment.

## Relationship to existing qwen-lean work

This direction does not replace:

- whole-proof qwen-lean training;
- verifier-filtered self-training;
- qwen-lean verifier-reward experiments;
- proof-state process supervision;
- the existing later tactic-level proving/search milestone.

It adds a separate architectural hypothesis: **a planner can mediate between conceptual intuition and formal execution**.

The exact relationship to later interactive tactic-level search remains OPEN and should be resolved from evidence rather than by reordering `PLAN.md` now.

## Relationship to Riemann

The layered research architecture becomes:

```text
Mathia
conceptual reasoning / intuition
        |
        v
qwen-lean-planner
Lean-aware search strategy
        |
        v
qwen-lean
formal proof execution/search
        |
        v
Lean
verification
```

For research claims that originate outside Lean, a separate faithful statement-formalization step may still be needed before this pipeline begins. That is orthogonal to the planner role.

## OPEN design questions

Issue #80 owns the unresolved details, including:

- exact frozen planner Base revision;
- exact frozen qwen-lean worker;
- whether a bounded no-proof-leakage SFT bootstrap is useful before RLVR;
- planner prompt/serialization;
- plan/proof boundary;
- plan count and formal attempts per plan;
- theorem curriculum;
- reward normalization;
- verifier failure semantics;
- matched-seed policy;
- RL versus self-training mechanism;
- replay/preservation controls for any focused specialization stage;
- checkpoint selection;
- transfer requirements.

## Non-goals

This document does not:

- alter `PLAN.md`;
- schedule the planner relative to existing phases;
- give source proofs to the planner;
- make a source-proof-derived gold plan corpus the core planner supervision;
- train Mathia;
- freeze SFT as a mandatory planner bootstrap;
- freeze GRPO as the final algorithm;
- copy Granite 4.2 hyperparameters into the project;
- merge planner and prover weights;
- claim Riemann progress;
- authorize open-conjecture execution.
