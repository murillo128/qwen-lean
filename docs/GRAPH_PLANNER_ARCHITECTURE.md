# Graph planner architecture

## Status

This document records the architectural refinement of the `qwen-lean-planner` concept.

The planner is **not a tactic planner**. Local tactic selection and Lean proof construction belong inside `qwen-lean`. The planner works at the level of mathematical/formal **graphs and relations**: it connects a target theorem and Mathia-style conceptual ideas to useful existing knowledge and to a dynamically constructed proof graph.

This note refines the role described in `QWEN_LEAN_PLANNER_DIRECTION.md` and should be treated as the current conceptual model when the two differ. It does not alter `PLAN.md` or schedule implementation work.

Mathia itself is unchanged by this decision. This document only specifies how qwen-lean would consume Mathia's existing conceptual output.

## Core separation of roles

```text
Mathia
concepts / mechanisms / conjectures / new relationships
        |
        v
Graph Planner
knowledge-graph search / grounding / proof decomposition
        |
        v
qwen-lean
local tactic reasoning / Lean proof generation
        |
        v
Lean
exact verification
```

The responsibilities are intentionally different:

- **Mathia:** ask what mathematical idea, viewpoint, relation, abstraction, or conjecture could move the problem forward.
- **Graph Planner:** ask how that idea or target can be connected to formal knowledge already available, and what intermediate nodes would make the target reachable.
- **qwen-lean:** ask how to prove one concrete Lean goal with the supplied local context and candidate premises/subgoals.
- **Lean:** decide exact formal validity.
- **Controller:** decide which branch to explore next and maintain the evolving search state.

The former notion of a separate tactic planner is dropped. Tactic planning is part of qwen-lean's local reasoning/search policy.

## Three graph layers

The architecture distinguishes three related but different graphs.

### 1. Mathlib knowledge graph

A static, versioned graph extracted offline from the compiled Lean/Mathlib environment.

Candidate node kinds include:

- theorem / lemma;
- definition;
- inductive / constructor;
- structure / field;
- namespace/module metadata where useful.

Candidate typed edges include:

- proof dependency;
- signature/type dependency;
- definition dependency;
- simp dependency;
- import/module relation;
- structure/field relation;
- reverse `used-by` edges.

Useful derived metadata can include:

- direct and transitive dependency counts;
- graph depth;
- reverse-use frequency / centrality;
- co-occurring premises;
- k-hop neighborhoods;
- theorem family / namespace;
- source position and statement text;
- embeddings or other retrieval features.

The artifact must be tied to an immutable Mathlib revision and extractor/schema version.

Conceptually:

```text
Mathlib @ commit X
      |
      v
kernel/environment graph extraction
      |
      v
versioned graph store
nodes + typed edges + reverse edges + retrieval metadata
```

The full graph is not serialized into every model prompt. It is a local knowledge base from which a bounded relevant neighborhood is retrieved.

### 2. Mathia idea graph

Mathia remains a high-level conceptual reasoner. Its natural-language output can be interpreted as a provisional semantic graph even if it is not emitted in a formal graph DSL.

Example:

```text
symmetry
   |
   v
paired objects
   |
   v
geometric constraint
   |
   v
possible contradiction
```

These are mathematical relations, not Lean tactics and not necessarily statements already present in Mathlib.

Mathia's most valuable role in hard research problems is to propose **new candidate relations or intermediate ideas that do not yet exist in the formal knowledge graph**.

### 3. Dynamic proof graph

For a concrete target, the Graph Planner builds and revises a task-specific graph:

```text
TARGET
├── candidate lemma A
│   ├── known Mathlib lemma C
│   └── candidate lemma D
├── candidate lemma B
└── known Mathlib theorem E
```

Nodes may be:

- already-known Mathlib facts;
- retrieved premises;
- proposed intermediate lemmas;
- decomposed subgoals;
- unresolved mathematical obligations.

Edges express why one node is expected to support another. Node status evolves as qwen-lean and Lean attempt formalization:

```text
unknown -> attempted -> verified
                   \-> failed-to-prove
```

`failed-to-prove` is not mathematical refutation.

## The Graph Planner is a bridge between graphs

The planner's central job is **graph grounding and graph completion**:

```text
Mathia conceptual relationships
             +
        target theorem
             +
retrieved Mathlib neighborhood
             |
             v
       Graph Planner
             |
             v
candidate proof graph / graph edits
```

Typical planner actions are therefore things such as:

- retrieve or rank useful premises;
- connect a conceptual relation to concrete Mathlib notions;
- propose an intermediate lemma;
- strengthen or generalize an intermediate statement so later composition is possible;
- select an induction/decomposition structure at the theorem level;
- split a target into branches;
- replace an unproductive graph route with another route;
- identify the smallest unresolved node after downstream formal attempts.

It should not primarily emit sequences such as `rw`, `apply`, `simp`, `omega`, or other local tactic programs. Those belong to qwen-lean.

## qwen-lean owns local tactic planning

qwen-lean receives bounded tasks rather than an entire research strategy. A useful worker request may contain:

```text
current Lean goal
local context
candidate intermediate lemma
retrieved relevant Mathlib premises
small amount of graph/planner guidance
```

qwen-lean then performs its own internal tactic reasoning and generates/repairs Lean code under verifier feedback.

This keeps the high-level planner from duplicating the prover and allows qwen-lean training on proof states, whole proofs, tactics, failure-recovery trajectories, and verifier reward to improve the local execution layer directly.

## Mathia usage under this architecture

No Mathia objective, representation, or repository contract is changed here.

Mathia continues to produce compact conceptual mathematical reasoning: mechanisms, abstractions, analogies, conjectures, useful viewpoints, possible reductions, and relationships worth investigating.

The new usage is:

```text
Mathia output
      |
      v
interpret as conceptual relations / candidate graph expansions
      |
      v
Graph Planner grounds them against Mathlib and the current proof graph
      |
      v
qwen-lean attempts the resulting concrete Lean nodes
```

For routine theorems, Mathia may not be needed:

```text
target -> Graph Planner -> qwen-lean -> Lean
```

For research-level problems or stalled graph search, Mathia is the exploration layer:

```text
formal graph search stalls
        |
        v
Mathia proposes a new conceptual relationship
        |
        v
Graph Planner tries to ground/expand it
        |
        v
qwen-lean tests formal consequences
```

The feedback to Mathia should be informational rather than a false refutation. For example, the controller can report that an idea reduced the branch to one unresolved mathematical obligation while all surrounding graph nodes were formally verified.

## Offline Mathlib graph preprocessor

A dedicated preprocessing step should materialize the Mathlib graph once per pinned Mathlib revision.

The preferred implementation principle is to inspect Lean's elaborated/kernel environment rather than infer dependencies only from source-text parsing. The exact extractor is an implementation choice; reuse of existing Lean graph/training-data tooling should be evaluated before building a new extractor.

Possible durable output:

```text
mathlib_graph/
  metadata.json
  graph.sqlite
  nodes.ndjson
  edges.ndjson
```

`metadata.json` should bind at least:

- Mathlib commit/revision;
- Lean version/toolchain;
- extractor version;
- graph schema version.

Large generated graph artifacts should follow the repository's existing artifact policy rather than being committed blindly to Git.

## Training opportunity

The graph view creates supervision that is much better aligned with the planner role than tactic imitation.

For existing verified Mathlib theorems we can derive tasks such as:

1. premise prediction;
2. missing-edge prediction;
3. next useful node prediction;
4. dependency-neighborhood reconstruction;
5. intermediate-lemma/decomposition prediction;
6. graph completion conditioned on a conceptual Mathia-style description.

Historical proof dependencies are useful **structural supervision**, but they are not assumed to be the unique or mathematically optimal proof plan. If Mathlib contains a route `T -> A -> B`, the planner is allowed to discover a different valid route `T -> X -> Y`.

This refines the previous `no gold tactic plan` principle:

- do not train the planner to reproduce a source proof or extracted tactic trace;
- proof-derived dependency edges may be used offline to teach graph structure and premise relationships;
- the hidden proof/dependency solution must not be exposed as input for held-out evaluation or inference;
- final utility is still measured downstream by whether the proposed graph helps qwen-lean produce Lean-verified results.

## Inference and leakage boundary

For a new or held-out target, the system may consult the graph of **already available formal knowledge**, but it must not look up the hidden proof dependencies of the target being evaluated.

```text
new target statement
       |
       v
retrieve bounded neighborhood from allowed Mathlib graph
       |
       v
Graph Planner proposes graph extensions / subgoals
       |
       v
qwen-lean attempts nodes
       |
       v
Lean verification updates node status
```

The training/evaluation split must therefore be graph-aware and explicitly prevent target-proof leakage.

## Verifier-reward training becomes graph utility

Downstream reward should attach to candidate graph decisions rather than to prose tactic plans.

Useful signals can include:

- whether a proposed node is formalizable;
- fraction of proposed leaf nodes qwen-lean can verify;
- whether the graph reduces the target to smaller verified components;
- whether a candidate edge/decomposition increases final proof success relative to the no-planner baseline;
- cost/depth/branching penalties where empirically justified;
- final Lean-verified theorem completion.

This gives richer credit assignment than only rewarding complete theorem success. A partially successful graph can expose the exact unresolved obstruction without pretending the theorem is solved.

## Research loop

A long-horizon research branch can therefore look like:

```text
Mathia
propose mathematical relationship
        |
        v
Graph Planner
connect it to formal knowledge and create candidate nodes
        |
        v
qwen-lean workers
attempt local nodes
        |
        v
Lean
verify / reject formal attempts
        |
        v
Controller updates proof graph
        |
        +---- if formal gap: retry/replan locally
        |
        +---- if mathematical gap: return obstruction to Mathia
```

The graph is the shared state between conceptual exploration and formal execution.

## Consequences for the previous planner concept

The following are no longer the primary planner responsibility:

- tactic-family selection as the main output;
- ordered tactic scripts;
- local Lean repair strategies;
- detailed proof-state action planning.

Those belong to qwen-lean and later interactive qwen-lean training/search.

The planner remains allowed to be Lean-aware where needed to ground graph nodes, type intermediate statements, understand available declarations, and communicate precise subgoals. Lean awareness is a means to manipulate the proof graph, not a reason to become a second prover.

## Durable architectural summary

```text
                STATIC KNOWLEDGE
             Mathlib knowledge graph
                      |
                      v
Mathia -------> Graph Planner <------ target theorem
 ideas             |
/new edges          v
               dynamic proof graph
                      |
             concrete local goals
                      v
                  qwen-lean
            internal tactic planning
                      |
                      v
                    Lean
                      |
              verified graph state
                      |
             controller / feedback
```

The central hypothesis is now:

> The missing layer between conceptual mathematics and Lean execution is not primarily a tactic planner. It is a learned graph reasoner that retrieves, relates, decomposes, and extends formal mathematical knowledge, while qwen-lean owns the local tactic-level implementation.