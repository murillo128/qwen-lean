# Mathia-guided proof-search retrospective conclusions

This document records the project interpretation of the frozen Mathia prompt A/B evidence from #86 and the scoring-excluded retrospective diagnostics from #93. The underlying measurements remain in `evidence/mathia-prompt-ab/`; this document is interpretation and project direction, not a replacement for those artifacts.

Use the repository markers as follows:

- `OBSERVED` for facts directly supported by the frozen evidence;
- `ACCEPTED` for durable project objectives or architectural principles;
- `OPEN` for choices that remain unresolved pending further experiments.

## 1. The objective is frontier capability, not raw solved-theorem count

**ACCEPTED:** future qwen-lean model selection must prioritize expansion of the structural frontier of Lean proofs over maximizing the raw number of easy/direct theorems solved.

A model that loses some direct one-line or near-one-line proofs can still be a better qwen-lean if it materially gains `branching`, `deep`, or otherwise structurally harder proofs. Direct capability remains a guardrail because local tactics and simple transformations are building blocks for harder proofs, but gains on direct tasks must not be allowed to compensate automatically for regressions at the hard frontier.

Aggregate pass@k remains useful, but it is not sufficient as the sole model-selection objective. Future held-out workloads should carry structural metadata before generation so checkpoint selection can inspect frontier movement explicitly.

The exact future checkpoint-selection rule is `OPEN`; do not retroactively rescore historical experiments using a post-hoc difficulty taxonomy.

## 2. Mathia guidance appears to move search toward the desired structural frontier

**OBSERVED:** on the 388 fresh-composition tasks with a pre-existing frozen structural class, the paired Q0 theorem-only versus Arm-B Mathia-guided solved@8 transitions are:

| structural class | Q0 only | B only | net B-only minus Q0-only |
| --- | ---: | ---: | ---: |
| direct | 5 | 4 | -1 |
| branching | 0 | 5 | +5 |
| deep | 1 | 2 | +1 |

The frozen taxonomy has no `multi-step` class, and the 223 matched MiniF2F tasks have no comparable frozen structural label. The comparison is stochastic n=8 and does not establish causal harm or benefit for any individual theorem.

**Interpretation:** this pattern is encouraging for the project objective. The main measurable trade appears not to be “Mathia makes the executor globally better or worse”, but that Mathia changes the proof-search distribution. In the classifiable subset, the net loss is concentrated in `direct`, while the net gain appears in `branching` and `deep`.

This makes the #86 result more promising than its aggregate solved@8 delta alone suggests.

## 3. Mathia should be treated as a search diversifier, not a mandatory replacement for the theorem-only policy

**ACCEPTED:** Mathia guidance should not be assumed to be the unique inference lane for every theorem.

The evidence supports preserving a direct/formal lane while using Mathia to introduce alternative conceptual relationships when they are useful, especially on harder or stalled problems. A simple theorem that already has a short formal route should not be forced through a longer conceptual derivation merely because Mathia can describe one.

The intended system shape remains compatible with:

```text
target
  |
  +-> direct / formal route -> qwen-lean -> Lean
  |
  `-> Mathia concept -> Graph Planner -> qwen-lean -> Lean
```

The controller/search policy may later decide when to use one or both lanes. Exact routing is `OPEN`.

## 4. Provisional direction: a Mathia-conditioned v4 is worth pursuing

**ACCEPTED, provisional on training design:** the #86/#93 evidence is sufficient to justify designing a future qwen-lean v4 that can exploit Mathia guidance.

This is not approval to repeat ordinary imitation SFT. Issue #83 is a negative result: bounded whole-proof + continuation SFT with structural reweighting, lower LR, explicit Base-preservation KL, and anti-collapse gates still produced repeated-template collapse, and no checkpoint was selected.

Therefore the exact v4 learning mechanism remains `OPEN` pending the native-thinking and forking results. Candidate mechanisms may include verifier-driven self-training, RLVR/process reward, proof-state/failure-recovery trajectories, replay/preservation, counterfactual forking, or combinations thereof. The mechanism should be chosen because it preserves and expands useful search behavior, not because it merely fits target proofs.

## 5. Unknown references are not uniformly negative

**OBSERVED:** #93 records 32 unknown-reference occurrences across 23 Arm-B candidates. The retrospective reconstructed 23 first call sites, replayed 22 prefixes to a captured state, and extracted five conservative candidate formal obligations. A scoring-excluded single-node oracle was testable in only two cases: one advanced to a later non-unknown failure and one reached a second unknown; none closed the parent theorem.

The two testable cases do not currently provide strong evidence of useful new mathematical decomposition. One is effectively an unavailable witness, and one follows a name/API family that appears to correspond to existing Lean declarations rather than a novel intermediate theorem.

**ACCEPTED:** an unknown theorem/lemma reference must not be penalized automatically as a hallucination. There are distinct cases:

1. wrong name/namespace for an existing Lean declaration;
2. unsupported guess that an existing premise/lemma should exist;
3. a genuinely new candidate intermediate proposition that could be proved and composed back into the parent theorem.

The third case is potentially positive evidence of decomposition. The stronger future test is:

```text
candidate intermediate node
        -> qwen-lean
        -> Lean verifies node
        -> reinsert verified node into parent route
        -> parent theorem progresses or closes
```

Until such a test succeeds, #93 does not establish spontaneous high-quality lemma invention.

## 6. Graph Planner remains a grounding and graph-reasoning layer, not a tactic planner

**ACCEPTED:** the #93 evidence is consistent with the current Graph Planner architecture.

Failures such as using `sq_sqrt` instead of `Real.sq_sqrt`, guessing unavailable Mathlib declarations, or failing to connect a conceptual relation to the cheapest existing formal route are primarily evidence that exact formal grounding matters. The external planner should operate over Mathlib retrieval/grounding, missing nodes/edges, intermediate obligations, and proof-graph structure; local tactic execution remains qwen-lean's responsibility.

The retrospective does not prove that every observed failure belongs to the Graph Planner. It does strengthen the motivation for a layer that can distinguish:

```text
concept already exists in Mathlib -> retrieve/ground exact node
concept needs a new intermediate fact -> create dynamic proof-graph node
local Lean action is wrong -> qwen-lean repair/search
```

## 7. Stopping and search control are first-class capabilities

**OBSERVED:** among the bounded regression candidates, 48/184 hit the token limit across 17 tasks, and 29 candidate diagnostics contain `No goals to be solved`.

These are not the same failure mode as wrong mathematics or missing knowledge. They indicate that search length, stopping discipline, and interaction with the Lean state can materially affect useful capability.

**OPEN:** a separate prefix-recovery diagnostic should determine how often an exact earlier prefix had already closed the goal before the model continued generating. The result would inform whether interactive state/action execution or stronger final-channel stopping should be part of v4.

## 8. What to wait for before freezing v4

Two active evidence streams can materially change the v4 design:

- **native thinking (#89):** determine whether internal reasoning improves local Lean planning/search without merely moving token explosion into the reasoning channel;
- **counterfactual forking:** determine whether branching from useful intermediate prefixes can preserve promising routes and improve frontier coverage more efficiently than one-shot autoregressive sampling.

**OPEN:** freeze the exact v4 training and inference architecture only after those results are available.

The current working hypothesis is:

> Mathia supplies useful conceptual search diversity; Graph Planner grounds and structures that diversity; qwen-lean must learn robust local execution/search; Lean provides the decisive verification signal. The system should be evaluated by how far it pushes the frontier of difficult verifiable proofs, not by how efficiently it accumulates easy benchmark wins.
