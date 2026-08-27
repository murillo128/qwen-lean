# Counterfactual forking as a verifier-driven training primitive

## Status

**SPECULATIVE**

This document records a general training/research primitive inspired by Forking Paths Analysis and the more sample-efficient Forking Fast work from Goodfire.

It does **not** change the accepted training roadmap, choose a new optimizer, authorize an RL run, or require a particular model architecture. It is a reusable way to extract better credit assignment from Lean verification and can be tested inside later qwen-lean, graph-planner, preference-learning, or verifier-reward experiments.

The central observation is simple:

> Lean gives the project an objective terminal verifier. Instead of using only `whole rollout -> PASS/FAIL`, branch from intermediate decisions, resample downstream continuations, and estimate which decisions materially increase or decrease the probability of a Lean-accepted proof.

This turns verification from only an outcome metric into a source of counterfactual training signal.

## Motivation: binary terminal reward wastes information

The strongest project invariant is that final proof success is determined by Lean. A naive verifier-reward loop therefore looks like:

```text
problem
   |
   v
model rollout
   |
   v
Lean
   |
   +--> verified = 1
   `--> rejected = 0
```

That signal is exact, but sparse. If a long reasoning/proof trajectory fails, the terminal `0` does not identify whether:

- an early decomposition was wrong;
- a premise choice was harmful;
- a useful direction was later abandoned;
- the overall idea was good but local Lean execution failed;
- several decisions were neutral and only one fork determined the outcome.

The same problem appears at graph level: if a candidate proof graph ultimately fails, a single terminal reward does not tell us which node or edge improved the downstream search and which one consumed compute without helping.

Counterfactual forking addresses this as a **credit-assignment** problem.

## Core primitive

Given a trajectory with intermediate prefixes or decisions:

```text
P0 -> d1 -> P1 -> d2 -> P2 -> ... -> dn -> Pn -> final outcome
```

choose selected prefixes `Pi` and sample multiple downstream continuations from each one under a frozen model/environment contract:

```text
Pi
├── continuation 1 -> Lean -> PASS/FAIL
├── continuation 2 -> Lean -> PASS/FAIL
├── continuation 3 -> Lean -> PASS/FAIL
└── continuation K -> Lean -> PASS/FAIL
```

Estimate:

```text
V(Pi) = P(Lean-verified completion | prefix Pi, frozen downstream policy)
```

A local change can then be approximated by:

```text
Delta_i = V(Pi) - V(Pi-1)
```

Interpretation:

- large positive `Delta_i`: promising fork / decision increased downstream proof probability;
- large negative `Delta_i`: harmful fork / decision reduced downstream proof probability;
- near-zero `Delta_i`: locally low-impact decision under the measured policy and budget.

This is not a mathematical truth about the decision. It is an empirical value estimate conditional on the exact downstream prover, prompt, sampler, verifier environment, and compute budget used to measure it.

## Prefer semantic decision boundaries over every token

The original forking idea can be applied at arbitrary token positions, but qwen-lean should prefer higher-level boundaries whenever they are observable and stable enough:

- reasoning-step boundaries;
- proof-prefix boundaries;
- completed tactic/action boundaries;
- selected premise/lemma decisions;
- proposed intermediate lemmas;
- graph-node or graph-edge decisions;
- recovery/replan decisions after a failed local attempt.

Token-level branching is still a diagnostic option, but it is not the default design target. Semantic boundaries make the resulting training signal easier to interpret and reduce unnecessary verifier/generation cost.

The method must also remain useful when a model's internal reasoning is not externally exposed. Proof prefixes, explicit controller actions, graph decisions, and other observable system boundaries are sufficient.

## Lean outcome model

Final training success remains binary and verifier-centered:

```text
Lean accepts reconstructed theorem -> success
anything else                     -> not success
```

However, rejected branches should retain diagnostic categories such as:

```text
syntax/parse failure
elaboration/type failure
unknown declaration/premise
failed tactic/action
unsolved goals
resource timeout
infrastructure error
```

These categories are useful for analysis and possibly for separately designed shaped-reward experiments, but they must not silently redefine task success. Infrastructure errors must never be treated as mathematical negative reward.

## Training uses

Counterfactual branch outcomes can support several training families without requiring a teacher proof plan.

### 1. Preference pairs from the same prefix

For the same theorem and same prefix, compare downstream continuations under matched budgets:

```text
prefix P
├── continuation A -> high verified yield
└── continuation B -> low verified yield
```

This creates a natural preference relation:

```text
(P, A) > (P, B)
```

Such pairs can feed DPO-style/preference optimization or a simpler reranker/value model. The preference is grounded in downstream formal utility rather than a human-authored explanation or extracted gold tactic plan.

### 2. Localized verifier-reward / RL advantage

Instead of assigning the same terminal outcome to an entire long rollout, use measured counterfactual value differences to identify decisions that contributed positive or negative downstream advantage.

The exact RL objective remains an experiment-specific choice. A shaped variant must be named and compared against the accepted terminal-verifier baseline rather than silently replacing it.

### 3. Verifier-filtered replay

High-value branches can become additional supervised examples, especially when they recover from prefixes that otherwise have low success probability.

This generalizes ordinary verifier-filtered self-training from:

```text
keep whole successful proof
```

to potentially retaining useful successful continuations and recovery trajectories from meaningful forks.

### 4. Value learning for search

A lightweight value model or score can learn to estimate:

```text
P(verified downstream | current state / graph decision / proof prefix)
```

The value is useful even if it is never used directly as an RL reward. It can rank branches, allocate sampling budget, and help a controller decide where more exploration is worth the cost.

## qwen-lean application

Under the current architecture, qwen-lean owns local Lean reasoning and tactic/proof construction. Counterfactual forking can therefore be applied inside the executor layer without reintroducing a separate tactic planner.

Example:

```text
current goal
   |
   v
qwen-lean local reasoning / proof prefix
   |
   +--> decision A -> downstream samples -> Lean yield
   +--> decision B -> downstream samples -> Lean yield
   `--> decision C -> downstream samples -> Lean yield
```

The training question becomes:

> Which local reasoning/proof decisions make later Lean-verified completion more likely?

This is a better fit for qwen-lean than supervising natural-language tactic plans merely because they resemble a source proof.

Possible observable fork points include:

- choice of induction/decomposition;
- first major proof constructor;
- selected lemma/premise;
- intermediate subgoal formulation;
- proof-prefix continuation;
- recovery action after Lean rejection.

## Graph-planner application

The same primitive applies one level higher to the graph planner, but the unit of credit is a graph decision rather than a local tactic.

```text
target / current proof graph
   |
   +--> edge or node A -> qwen-lean attempts -> Lean yield
   +--> edge or node B -> qwen-lean attempts -> Lean yield
   `--> edge or node C -> qwen-lean attempts -> Lean yield
```

This estimates quantities such as:

```text
P(Lean success | candidate graph edge)
P(Lean success | proposed intermediate node)
P(Lean success | decomposition branch)
```

Over time this can train a graph-level value function for:

- premise/edge ranking;
- intermediate-node selection;
- branch prioritization;
- identifying graph expansions that reduce the unresolved proof frontier;
- allocating downstream qwen-lean attempts where they have the highest expected utility.

This is consistent with `GRAPH_PLANNER_ARCHITECTURE.md`: the planner remains a graph reasoner, while local tactic planning remains inside qwen-lean.

## Adaptive test-time compute

The same measurements may later improve inference efficiency.

A brute-force pass@N policy spends roughly equal sampling effort everywhere. A forking-aware controller can instead spend more compute around high-uncertainty/high-impact decisions:

```text
stable region
   -> continue with little branching

high-impact / uncertain fork
   -> branch several continuations
   -> verify / estimate value
   -> retain promising branches
```

This is a future inference optimization, not a training requirement. The relevant project hypothesis is that owned local inference can gain more from **where** compute is spent, not only from increasing `N` globally.

## Sampling efficiency and prefix caching

Naively resampling many continuations from every possible prefix is expensive. The Goodfire Forking Fast result is relevant because it treats nearby fork-outcome distributions as mostly smooth and concentrates sampling around change points rather than uniformly at every position.

For this project, the general implementation principles are:

- reuse KV/prefix caches whenever multiple branches share a long prefix;
- begin with semantic fork points rather than every token;
- use adaptive resampling where estimated value changes materially or uncertainty is high;
- pool/smooth nearby estimates only when the approximation is validated on Lean outcomes;
- measure actual compute savings locally rather than importing headline speedup claims from another workload.

Prefix-cached branching is especially compatible with local inference because the shared theorem/context/reasoning prefix should not be recomputed for every branch when the runtime can reuse it safely.

## Experimental hygiene

Counterfactual value estimates are solver-conditional. Comparisons must freeze or match at least:

- theorem/task identity;
- visible context and graph state;
- downstream qwen-lean checkpoint;
- tokenizer/prompt/interface;
- candidate budget and seed mapping where practical;
- sampling parameters;
- Lean/mathlib environment;
- verifier timeout/resource contract.

Do not compare a branch under more formal-worker attempts against another branch with less compute and call the difference learned value.

Where a no-decision/no-guidance baseline is meaningful, retain it. Easy theorems that succeed under almost any branch should not be interpreted as strong evidence that one particular decision was useful.

## Leakage boundary

Source proofs and hidden target dependencies must not be exposed to the model/controller merely to manufacture useful fork points on held-out tasks.

Allowed training evidence can include:

- model-generated branches;
- Lean verification outcomes;
- proof prefixes that are legitimately part of the training corpus;
- graph structure allowed by the current split/environment;
- mechanically replayable training-proof trajectories.

For held-out evaluation, the hidden proof path remains hidden. Counterfactual branching must explore from model/system-generated observable state, not from an oracle location extracted from the target proof.

## Minimal discriminating experiment

Before making this a major training dependency, run a bounded experiment designed to answer whether the signal exists.

Suggested shape:

```text
50-100 development-safe Lean tasks
        |
        v
frozen qwen-lean checkpoint
        |
        v
identify a small set of semantic fork points per task
        |
        v
sample matched continuations from each fork
        |
        v
Lean verification
        |
        v
estimate V(prefix) and local Delta
```

Measure at least:

- verified-yield variance between continuations of the same prefix;
- stability of high-positive/high-negative fork rankings under modest resampling;
- whether measured fork value predicts later verified completion on held-out branches;
- fraction of tasks with no informative variance/all-zero reward;
- generation tokens and verifier work required per informative fork;
- savings from prefix caching/adaptive sampling versus a simple uniform baseline.

A useful result would show reproducible fork-level variation that predicts downstream Lean success at a cost low enough to train on. An all-zero or unstable signal is a valid negative result and should block scaling rather than be hidden by increasingly elaborate shaping.

## Relationship to existing training branches

This concept can complement the existing post-training families without collapsing them into one mandatory pipeline:

```text
verifier-filtered self-training
    -> retain successful branches/continuations

preference training
    -> rank continuations from matched prefixes

RLVR / GRPO
    -> use branch-level downstream utility as a separately tested credit signal

proof-state / incremental supervision
    -> use explicit observable proof boundaries as natural fork points

graph-planner training
    -> learn value over nodes/edges/decompositions
```

The first experiment should therefore test the measurement primitive itself. Only after branch values are demonstrably stable and useful should a later issue choose whether DPO, value learning, RLVR, filtered replay, or a combination is the best consumer of that signal.

## Non-goals

This concept does not imply:

- replacing Lean terminal verification with a learned reward model;
- treating partial progress as proof success;
- exposing hidden source proofs during evaluation;
- reviving a separate tactic-planner model;
- assuming every token deserves an independent reward;
- assuming Forking Fast's reported efficiency transfers unchanged to Lean;
- requiring chain-of-thought visibility from every model;
- changing the current roadmap before a bounded signal test exists.

## Durable hypothesis

The reusable hypothesis is:

> Lean verification is not only a terminal pass/fail oracle. By counterfactually branching from intermediate model or graph decisions and measuring downstream verified yield, we can estimate which decisions actually increase proof probability, producing teacherless credit assignment for preference learning, value learning, RL, self-training, and adaptive local search.

The key project advantage is the combination of **strong formal verification**, **controllable local generation**, and **branchable proof/graph state**.

## References

- Goodfire, *Forking Fast: Sample-Efficient Forking Paths Analysis*, arXiv:2608.19611 — https://arxiv.org/abs/2608.19611
- *Forking Paths in Neural Text Generation*, arXiv:2412.07961 — https://arxiv.org/abs/2412.07961
- Goodfire reference implementation — https://github.com/ericb-goodfire/forking-fast
- Repository architecture context — `docs/GRAPH_PLANNER_ARCHITECTURE.md`
- Durable post-training branches and Lean-verifier contract — `docs/DECISIONS.md`, especially D002 and D013
