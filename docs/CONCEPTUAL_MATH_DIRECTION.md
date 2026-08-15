# Conceptual mathematics direction

This document records a long-term research idea for `qwen-lean`. It is intentionally exploratory: it is **not** a roadmap phase, implementation contract, accepted architecture, dataset specification, or commitment to a particular training method.

The current project teaches and evaluates formal Lean proof generation. That work is useful in its own right, but it can also provide a formal backend for a more ambitious question:

> Can an LLM learn to reason about mathematics primarily at a higher conceptual level, and use a separate formalization/proving layer to turn the resulting ideas into precise Lean statements and verified proofs?

## Motivation

A working hypothesis behind this direction is that a large amount of formal mathematics is the precise implementation of ideas that can often be expressed much more compactly: cyclic behavior, divisibility, symmetry, invariance, decomposition, change of representation, equivalence, abstraction, generalization, and similar recurring patterns.

From this viewpoint, formal mathematics is indispensable for precision and verification, but it need not be the representation in which all mathematical reasoning happens. A useful analogy is software design versus implementation: the high-level model determines what matters and how the pieces relate; the implementation makes that design exact and executable.

This is a research hypothesis rather than a philosophical claim that mathematics is "only" syntax or that formal proof is unimportant. One of the things the project could eventually test is whether separating conceptual reasoning from formal proof generation actually improves mathematical capability.

## Philosophical context behind the hypothesis

The discussion that motivated this direction started from a deliberately non-Platonic intuition: mathematical objects can be viewed as human-built representations for recurring relationships rather than as entities that must literally exist independently of us.

Very simple ideas can generate large mathematical structures. Counting leads to arithmetic; asking whether one quantity divides another leads to divisibility, remainders, congruence classes, factorization, and number-theoretic structure. Repeated reversible transformations lead naturally to cycles, composition, inverses, symmetry, and group-like structure. The exact historical development is richer than any one lineage, but the recurring pattern is the important part here.

A relationship is formalized, then often **reified** into a new object that can itself be studied. Relations between those objects become further objects and abstractions. Conceptually:

```text
simple relation or pattern
          |
          v
formal definition / representation
          |
          v
new mathematical object
          |
          v
relations between those objects
          |
          v
higher abstraction
          |
          v
repeat
```

This recursive construction can produce a very deep formal tower whose connection to the original intuition becomes difficult to see. Much of pure mathematics can then be viewed, under this working interpretation, as exploring the consequences of representational systems that were introduced because they compressed some useful pattern.

The fact that the same abstraction appears in many domains does not by itself require a Platonic interpretation. A group, graph, vector space, or similar structure can instead be thought of as an interface that intentionally preserves only a particular kind of relationship. It is then unsurprising that the same interface is useful wherever that relationship occurs. What is discovered is not necessarily a pre-existing abstract object, but which abstractions compress phenomena well and what consequences follow once their rules are fixed.

This does **not** mean the consequences are arbitrary. We may choose the representation or axioms, but after doing so we do not freely choose their implications. In that sense there remains a genuine discovery problem inside a constructed formal system.

A related intuition applies to mathematical physics and other mathematical sciences. The conceptual content of an idea can be much smaller than the formal machinery required to state it precisely, calculate with it, and propagate all its consequences. A symmetry, stationary principle, conservation idea, or geometric picture may be easy to describe while its coordinate expressions, tensor manipulations, differential equations, or perturbative calculations are large.

That motivates a distinction between two kinds of complexity:

```text
semantic / conceptual complexity
- what objects or phenomena matter?
- what is the right viewpoint?
- what is invariant?
- which assumptions are essential?
- what should be conjectured?

mechanical / formal complexity
- expand expressions
- change coordinates or representations
- normalize terms
- apply lemmas
- manipulate symbols
- construct a fully checkable derivation
```

Today mathematicians and theoretical scientists often have to master both because the formal machinery has historically been executed largely by humans. The working hypothesis here is that an AI system might separate these roles more strongly: reason mainly in a compact conceptual representation, then delegate the exact formalization and derivation to a lower layer.

The software analogy is useful but intentionally imperfect:

```text
conceptual mathematical idea     software/system design
            |                              |
            v                              v
formal mathematical structure    implementation/IR
            |                              |
            v                              v
Lean proof / derivation           compiled executable form
            |                              |
            v                              v
kernel verification               compiler/tests/runtime checks
```

Under this analogy, large amounts of local theorem search, rewriting, tactic selection, and symbolic manipulation are closer to implementation or compilation than to the highest level of mathematical understanding. Automating that work could let a model spend more of its capacity on choosing representations, recognizing common structure, weakening assumptions, inventing useful abstractions, and proposing conjectures.

There are important limits to this analogy. Mathematics is not merely bureaucratic translation. Formalization can reveal consequences that intuition misses, expose contradictions in an apparently simple idea, or show that a proposed abstraction is insufficient. In physics, mathematics also cannot prove that a model is true of nature; it proves consequences conditional on the model, while empirical evidence decides whether the model describes the world. The formal layer is therefore not disposable: it is both a correctness constraint and a cognitive tool that can feed surprising information back into the conceptual layer.

The project does not need to resolve any philosophy-of-mathematics question. The useful research claim is narrower and empirical:

> Some mathematical capability may improve if an LLM is trained to operate explicitly on compact conceptual descriptions and conjectures before being asked to compile those ideas into exact formal mathematics.

## Desired separation of layers

The tentative mental model is:

```text
mathematical material / problems / examples
                |
                v
    conceptual mathematical reasoning
                |
                v
      concepts, viewpoints, conjectures
                |
                v
          Lean formalizer/prover
                |
                v
           Lean verification
```

The upper layer should not merely predict Lean tactics or perform longer theorem-search traces. Its role would be closer to asking:

- What is this problem really about?
- Which details appear accidental and which structure matters?
- Is there a simpler representation or viewpoint?
- Does an invariant, symmetry, quotient, cycle, decomposition, or analogy explain the behavior?
- Is an assumption stronger than necessary?
- Can several concrete results be compressed into a more general conjecture?
- What conjecture would be worth attempting to formalize next?

The lower layer would then turn those ideas into exact statements and proofs. Lean remains the final authority for the formal claim.

## Representation: close to mathematical natural language

The conceptual layer should remain much closer to concise mathematical natural language than to a new formal DSL.

The goal is not to invent "Lean-lite" or require every early thought to have a precise type. Useful mathematical exploration often contains provisional statements such as:

- "This looks cyclic."
- "The absolute value probably does not matter; only the residue does."
- "Try viewing this as a permutation of the residue classes."
- "Primality may be stronger than necessary; perhaps coprimality is what is actually used."
- "These examples seem to be instances of the same symmetry."

Such statements may be deliberately less precise than Lean while still being substantially more structured and mathematical than unrestricted conversational prose.

A future corpus might therefore use short **mathematical meta-descriptions**: compact explanations of the underlying concepts, useful viewpoint, possible generalization, conjecture, or reason a result should be true. The exact style and schema are deliberately open.

## Possible data-generation direction

One possible experiment is to use Codex or another strong teacher system to generate these meta-descriptions from mathematical source material. Possible sources include:

- Lean/mathlib theorem statements and verified proofs;
- groups of related theorems rather than isolated proof strings;
- worked mathematical material with suitable licensing;
- general mathematical exposition;
- examples and counterexamples generated specifically for an experiment.

The generated description would aim to capture the conceptual content rather than restating the Lean implementation. For example, it might identify the relevant abstraction, explain what information can be ignored, connect the theorem to a recurring pattern, or propose a useful generalization.

This data would initially be synthetic or weakly supervised. A Lean proof can verify the eventual formal statement, but it cannot certify that a natural-language conceptual explanation is insightful or even faithful. That distinction must remain explicit.

## Possible training experiment

A later experiment could train Qwen on this conceptual/meta-description corpus and ask whether doing so improves mathematical behavior beyond training on formal theorem/proof pairs alone.

The exact training method is intentionally unspecified. Possibilities might include adapter-based SFT, broader/full-model fine-tuning, preference training, or combinations with the formal-training branches already explored by the project. The point of this document is not to choose among them now.

A useful comparison could eventually separate models such as:

```text
formal-proof training only

vs.

conceptual/meta-description training

vs.

conceptual training + formal Lean training
```

The interesting outcome is not merely whether the model writes nicer explanations. The stronger hypothesis is that conceptual training could improve abilities such as abstraction, generalization, conjecture formation, representation choice, and ultimately successful formal mathematics.

## AI-feedback training opportunity

This direction also provides a natural place to experiment with **AI feedback** because the conceptual output is not directly machine-verifiable in the way a Lean proof is.

A teacher or judge model could, for example:

- generate multiple conceptual descriptions for the same mathematical material;
- rank descriptions by clarity, faithfulness, generality, or usefulness;
- identify when a description merely paraphrases a proof rather than extracting an idea;
- propose or rank conjectures;
- critique a conjecture after a counterexample or failed formalization;
- create chosen/rejected pairs for preference training.

This could support experiments with preference optimization or other AI-feedback post-training methods. The project should keep the distinction between **AI-judged conceptual quality** and **Lean-verified formal correctness** visible rather than treating them as the same signal.

A particularly interesting future feedback loop is:

```text
conceptual idea / conjecture
          |
          v
AI feedback / refinement
          |
          v
Lean formalization and proof attempt
          |
          v
verified result, counterexample, or failure evidence
          |
          +------> refine the conceptual model
```

A failed proof attempt does not imply that a conjecture is false, so any such loop would need to distinguish prover weakness, formalization problems, and actual counterexamples.

## What would count as evidence

The eventual question is empirical: does adding this layer make the model better at mathematics?

Potential evidence could include improvements in some combination of:

- held-out mathematical problem solving;
- quality and usefulness of conjectures;
- ability to generalize a theorem or weaken unnecessary assumptions;
- ability to recognize common structure across superficially different problems;
- rate at which conceptual outputs can be turned into valid formal statements;
- final Lean proof success when the conceptual layer feeds a formalizer/prover;
- performance compared with an otherwise similar model trained primarily on theorem/proof data.

The exact evaluation protocol is open and should only be designed once this direction becomes an active experiment.

## Relationship to the current project

Nothing here changes the current execution plan. The existing Lean work builds valuable infrastructure and a formal proving backend. Proof-state supervision, self-training, and verifier-reward RL can all be explored without committing to this conceptual architecture.

If the formal backend becomes sufficiently capable, this direction can later use it as the equivalent of an implementation and verification layer beneath higher-level mathematical reasoning.

For now, the durable idea is only:

> Explore whether training an LLM on compact, higher-level mathematical meta-descriptions and conceptual relationships can improve mathematical reasoning, conjecture formation, and eventual formal proof success; use Lean as a downstream formalizer/verifier and use the conceptual layer as a natural testbed for AI-feedback training.
