# qwen-lean plan

This document defines the technical sequence of the project and the gate for moving from one phase to the next. Durable architectural choices belong in `docs/DECISIONS.md`; current roadmap status belongs in epic #1; implementation details for a phase belong in its controlling GitHub issue.

## Project success criterion

The first major project milestone is reached when a fine-tuned Qwen model shows a clear improvement over the unchanged base model on held-out Lean proof generation using the same prompt, generation settings, Lean environment, and evaluation harness.

The primary task signal is verifier success. A proof is successful only if the declared Lean environment accepts the completed theorem. We will report pass@k-style metrics rather than treating training loss as evidence of theorem-proving improvement.

## Phase 0 — Evaluation contract and runtime foundation

**Purpose:** establish the smallest trustworthy end-to-end loop before creating a training dataset or spending meaningful GPU time.

Build the repository/runtime foundation needed to:

- load the chosen base model;
- format a theorem-generation prompt;
- generate candidate proofs;
- insert a candidate into the intended Lean context;
- run Lean and capture success, failure category, diagnostics, latency, and generation metadata;
- preserve compact machine-readable evaluation results.

**Exit gate:** a small fixed set of Lean fixtures can be evaluated end to end and repeated without changing the task contract. Successful and failing candidates are distinguished correctly by Lean.

## Phase 1 — Base-model baseline

**Purpose:** measure what the unchanged base model can already do before training.

Evaluate `Qwen/Qwen3-8B-Base` with the Phase 0 harness on a small development workload and the first external benchmark subset suitable for development.

Capture enough metadata to make later comparisons meaningful: model/tokenizer identity, prompt format, generation settings, Lean environment, benchmark split, and task metrics.

**Exit gate:** baseline pass@k results and failure breakdowns exist in a form that the later fine-tuned runs can use without changing the evaluator semantics.

## Phase 2 — Verified training dataset

**Purpose:** create a clean supervised dataset of real Lean theorem/proof pairs before introducing synthetic data.

Start from verified Lean/mathlib proofs. Extraction should retain the context needed to reconstruct the training example and later reason about contamination. Split data by a structural boundary stronger than random theorem-level sampling so closely related material does not trivially appear on both sides of the comparison.

Produce basic dataset statistics, including example count, token-length distributions, filtering losses, and split sizes.

**Exit gate:** sampled records can be reconstructed and verified in Lean; train/validation/held-out boundaries are documented; obvious train/eval leakage is absent; the data can be loaded by the training pipeline.

## Phase 3 — Training sanity check

**Purpose:** prove that the SFT/QLoRA pipeline is wired correctly before scaling it.

Use a tiny dataset, roughly tens to a hundred examples, and intentionally overfit it. This is a plumbing test, not a useful model.

The run should make it possible to detect errors in target masking, prompt construction, tokenizer/model pairing, adapter configuration, checkpoint loading, and generation from the resulting adapter.

**Exit gate:** the model can strongly fit the tiny training set and the resulting checkpoint can be loaded and evaluated through the same inference and Lean-verification path used for the baseline.

## Phase 4 — SFT smoke experiment

**Purpose:** test realistic training behavior cheaply before the first full run.

Train on a small but non-trivial subset, approximately thousands rather than tens of thousands of examples. Use the same data format and evaluation contract intended for the first full SFT experiment.

Inspect optimization stability, memory use, throughput, checkpointing, and a small held-out evaluation.

**Exit gate:** training is numerically stable, the complete artifact path works, and the evaluator produces directly comparable base-versus-fine-tuned results. Any material configuration problem is resolved before scaling.

## Phase 5 — First full SFT/QLoRA experiment

**Purpose:** answer the first substantive project question: does supervised post-training teach the base model materially more Lean proof generation?

Train the selected Qwen base model with QLoRA on the verified dataset at the scale supported by the Phase 2 corpus and available single-GPU compute. The initial planning target is on the order of tens to low hundreds of thousands of examples, but the exact size is a dataset decision rather than a fixed quota.

**Exit gate:** a usable adapter/checkpoint exists, its training run is understood, and it has been evaluated through the unchanged harness on held-out data and the development benchmark.

## Phase 6 — Comparative evaluation

**Purpose:** determine what changed, not merely whether a training job completed.

Compare the base model and the first full SFT model using the same evaluation contract. Report at least pass@1 and higher-k sampling metrics where useful, plus failure categories and operational measurements that help explain the result.

Use the external benchmark test split only for checkpoints that have already been selected without tuning against that test set.

**Exit gate:** the project can state a supported conclusion about whether SFT improved Lean proof generation, where it improved, and what important limitations remain.

## Phase 7 — Verifier-filtered self-training

**Purpose:** use Lean's objective verifier to create better training data before introducing online reinforcement learning.

Have the best SFT model sample multiple candidate proofs for additional theorem statements. Run candidates through Lean, keep successful examples under explicit filtering rules, and use the verified synthetic examples for a new supervised training iteration.

**Entry condition:** the SFT model must already solve enough examples for verifier filtering to yield useful data.

**Exit gate:** self-training is compared against the best prior SFT checkpoint using the unchanged evaluation contract, with the contribution and limitations of synthetic verified data understood.

## Phase 8 — GRPO / verifier-reward reinforcement learning

**Purpose:** learn from online verifier feedback after the simpler verifier-filtered approach is understood.

Start with a sparse objective tied to valid proof completion and only add shaping signals if the sparse reward is empirically inadequate and the shaping semantics can be justified.

**Entry condition:** generation, verification, reward computation, artifact handling, and comparative evaluation must already be stable enough that RL-specific failures can be distinguished from infrastructure failures.

**Exit gate:** the RL checkpoint is compared against the best supervised/self-trained checkpoint under the same task-level evaluator, with cost and stability considered alongside pass@k.

## Phase 9 — Tactic-level proving and search

**Purpose:** move beyond whole-proof generation if the project benefits from a more capable theorem-proving architecture.

This is a separate milestone. The task changes from generating an entire proof in one completion to repeatedly choosing tactics from a Lean proof state, potentially adding premise retrieval, search, and progress/value estimates.

**Entry condition:** whole-proof experiments have produced enough evidence to justify the additional system complexity.

**Exit gate:** to be defined by a dedicated design issue when this phase becomes active.

Open technical choices are owned by `docs/DECISIONS.md` and should be resolved only by the phase that first needs them.
