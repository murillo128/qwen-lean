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

## Phase 6 — Comparative evaluation and common post-training parent

**Purpose:** determine what changed after SFT and establish the common checkpoint from which later post-training methods will be compared independently.

Compare the base model and the first full SFT model using the same evaluation contract. Report at least pass@1 and higher-k sampling metrics where useful, plus failure categories and operational measurements that help explain the result.

Add a deterministic training-set generation diagnostic for the selected SFT checkpoint. Evaluate a fixed sample of Phase 5 training theorems through the same free-generation and Lean-verification path used for held-out evaluation, and report both exact target-proof reproduction and Lean-accepted proof success. Compare this train-sample result directly with the held-out result so Phase 6 can distinguish memorization from generalization and quantify the train-to-heldout generalization gap. The controlling Phase 6 issue should freeze the exact sample size, selection rule, and candidate budget before observing model outputs.

Select one SFT checkpoint for later post-training without tuning against the external benchmark test split. That checkpoint becomes the common parent and retained control for Phases 7, 8, and 9.

Use the external benchmark test split only for checkpoints that have already been selected without tuning against that test set.

**Exit gate:** the project can state a supported conclusion about whether SFT improved Lean proof generation, where it improved, how training-set proof regeneration differs from held-out generalization, and what important limitations remain; one reference SFT checkpoint is explicitly selected as the common parent/control for the independent post-training branches.

## Parallel next-generation corpus refresh — Dataset v2

**Purpose:** build the training-first verified corpus used by the next selected trainable foundation and by the Riemann/prime specialization path, without rewriting the historical Dataset-v1 experiment.

Dataset v2 starts from the repository-owned Phase 2/Riemann data layer but recomputes future training membership under D016. It should recover useful verified source proofs that the first extractor omitted because of proof syntax, include all eligible prime/number-theory source theorems, preserve declared Lean context/environment, and add Lean-verified synthetic composition examples. Fresh validation/test must use new obligations and new derivation families rather than withholding useful source knowledge.

The corpus refresh is independent of the final foundation choice and requires no model training. The controlling Dataset-v2 issue owns its extraction, composition, contamination, packaging, and preflight details.

**Exit gate:** the authoritative Dataset-v2 training corpus is loadable from a fresh checkout, all eligible prime source knowledge is optimizer-eligible or explicitly blocked for a concrete reason, recoverable term-style proofs are no longer silently lost, synthetic composition passes its freshness/essentiality gates, and clean validation/test workloads are ready for routine pass@8 checkpoint selection. The next foundation-specific training issue may then choose token/context limits and curriculum weights without changing corpus membership silently.

## Phase 7 — Branch A: verifier-filtered self-training

**Purpose:** measure what is gained by turning the SFT model's own Lean-verified successes into additional supervised training data.

Start from the reference SFT checkpoint selected in Phase 6. Sample multiple candidate proofs for additional theorem statements, run every candidate through the unchanged Lean verifier, retain successful examples under explicit filtering rules, and perform a new supervised training iteration on the resulting verified synthetic data.

This branch is independent of Phases 8 and 9. Where practical, use the same theorem-source pool and comparable generation budget intended for the other post-SFT branches so the later comparison is interpretable.

**Entry condition:** the reference SFT model solves enough sampled examples for verifier filtering to produce a useful synthetic corpus.

**Exit gate:** a self-trained Model A exists and has been evaluated against the retained reference SFT checkpoint using the unchanged task contract; synthetic-data yield, cost, and limitations are understood.

## Phase 8 — Branch B: GRPO / verifier-reward reinforcement learning

**Purpose:** independently measure what is gained by training directly from online Lean verifier reward rather than converting successes into a new SFT dataset.

Start from the same reference SFT checkpoint selected in Phase 6, **not** from the Phase 7 self-trained checkpoint or the Phase 9 process-supervised checkpoint. The first GRPO/RLVR experiment uses a binary outcome reward tied to valid proof completion: `1` for a `verified` proof and `0` otherwise.

Generate groups of candidate proofs, verify them through the unchanged Lean path, and use the verifier outcomes as the online training signal. Record reward density and group variation explicitly so sparse-reward failure can be distinguished from implementation failure.

If the binary reward produces too little useful signal, do not silently change the reward function mid-experiment. Preserve that result and design any syntax/elaboration/proof-progress shaping as a distinct named follow-up variant with its own contract and evaluation.

**Entry condition:** generation, verification, reward computation, artifact handling, and comparative evaluation are stable enough that RL-specific failures can be distinguished from infrastructure failures.

**Exit gate:** a GRPO/RLVR Model B exists or the binary-reward experiment has produced a supported sparse-reward conclusion; the result is evaluated against the same retained SFT control without using Model A or Model C as its initialization.

## Phase 9 — Branch C: proof-state process supervision

**Purpose:** measure whether exposing Lean's verified intermediate proof states and tactic transitions teaches useful proof-construction structure beyond whole-proof output supervision alone.

Start from the same reference SFT checkpoint selected in Phase 6. Build a process-supervision corpus from Lean-verified proofs by extracting trajectories of the form `proof state -> tactic/action -> resulting proof state`. The source proofs may come from existing verified training material and, when useful, independently sampled proofs from the common SFT parent, but this branch must not depend on Model A or Model B outputs/checkpoints as required inputs.

Train a distinct Model C with an explicitly designed process-supervision objective or auxiliary supervised mixture. The exact serialization and loss mixture belong to the Phase 9 design issue; they must preserve which intermediate states and actions are mechanically grounded by Lean rather than treating free-form natural-language rationales as verified evidence.

For this first process-supervision experiment, **evaluation remains whole-proof generation under `whole-proof-v1` and the unchanged Lean verifier**. Do not change the evaluation task to interactive tactic selection merely because proof states are used during training. That architectural change remains a later milestone.

Where practical, use theorem sources and training/generation budgets that make comparison with the other independent branches meaningful. Record process-corpus size, trajectory lengths, tactic/state coverage, and any filtering losses needed to interpret the result.

**Entry condition:** the project can reliably extract proof-state/tactic transitions from verified Lean proofs with enough context to replay or validate sampled trajectories.

**Exit gate:** a process-supervised Model C exists and has been evaluated against the same retained SFT control using whole-proof evaluation; the project can state whether process supervision improves proof validity or task success beyond output-only SFT, with data yield, cost, and limitations understood.

## Phase 10 — Independent post-training comparison

**Purpose:** determine what each post-training method contributes before combining them or changing the proving architecture.

Compare at minimum:

- the retained reference SFT checkpoint (control);
- Model A from verifier-filtered self-training;
- Model B from binary-reward GRPO/RLVR, when training produced a usable checkpoint;
- Model C from proof-state process supervision;
- any separately designed shaped-RL variant, if one was needed.

Use the same whole-proof prompt, Lean environment, held-out workloads, generation/evaluation settings, and verifier semantics. Compare task quality and also the cost needed to obtain it: candidate-generation volume, training GPU time, verifier workload, process-data extraction cost, stability, and relevant failure distributions.

For Model C, also inspect whether gains correlate with process-level changes such as higher rates of valid tactic execution or more useful intermediate proof-state transitions, while keeping final model-quality claims anchored to verifier-based whole-proof success.

Do not treat self-training followed by GRPO, process supervision followed by GRPO, or another composition as the default experiment. Only after the independent comparison may a later design test whether methods are complementary in sequence or mixture.

**Exit gate:** the project can state which isolated post-training method improved over the common SFT parent, by how much, at what cost, and whether a combined follow-up or a tactic-level architecture is justified.

## Phase 11 — Tactic-level proving and search

**Purpose:** move beyond whole-proof generation if the project benefits from a more capable theorem-proving architecture.

This is a separate milestone. The task changes from generating an entire proof in one completion to repeatedly choosing tactics from a Lean proof state, potentially adding premise retrieval, search, and progress/value estimates.

Proof-state data learned or extracted in Phase 9 may inform this design, but Phase 11 changes the inference/control loop and must not be treated as merely turning on an already-trained auxiliary objective.

**Entry condition:** whole-proof experiments and the process-supervision branch have produced enough evidence to justify the additional system complexity.

**Exit gate:** to be defined by a dedicated design issue when this phase becomes active.

Open technical choices are owned by `docs/DECISIONS.md` and should be resolved only by the phase that first needs them.