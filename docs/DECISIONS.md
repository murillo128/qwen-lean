# qwen-lean decision log

This document records durable technical decisions that should remain stable across multiple execution issues. Phase-local implementation details belong in the controlling issue; sequencing belongs in `PLAN.md`; current status belongs in epic #1.

Use the repository decision markers exactly: `ACCEPTED`, `OPEN`, `SPECULATIVE`, `REJECTED`, `OBSERVED`, and `BLOCKED`.

## D001 — Initial task: whole-proof generation

**Status:** ACCEPTED

The first theorem-proving task is whole-proof generation: given the theorem statement and the Lean context required by the experiment, the model generates the complete proof body for that theorem.

This is intentionally simpler than an interactive prover that repeatedly chooses tactics from proof states. It gives the project a small end-to-end system with an objective verifier and lets us learn the post-training pipeline before adding retrieval or search.

Tactic-level generation, premise retrieval, and proof search are later milestones, not v1 requirements.

**Exact prompt and proof-target serialization:** ACCEPTED — `whole-proof-v1` uses
plain code completion ending at `<declaration> := by\n  `, and verifier input is
that same prefix followed directly by the raw generated continuation. Only line
endings and trailing transport whitespace may be normalized; no proof extraction
or semantic repair is allowed.

## D002 — Task success is determined by Lean

**Status:** ACCEPTED

A candidate is successful only when the declared Lean environment accepts the reconstructed theorem with that proof. Syntactic resemblance to Lean, parsing alone, or a lower training loss are not proof success.

Evaluation must keep at least these concepts distinct when diagnosing failures:

- generated text/output formatting;
- Lean parsing/elaboration/proof-checking outcome;
- task-level success under the declared benchmark contract.

The primary model-quality metrics will be verifier-based pass@k measurements, starting with pass@1 and adding larger k where sampling multiple candidates is informative.

## D003 — First trainable base model: Qwen3-8B-Base

**Status:** ACCEPTED

Use `Qwen/Qwen3-8B-Base` for the first baseline and supervised fine-tuning cycle.

A base checkpoint is preferred over an instruction-tuned checkpoint for the initial learning experiment because it makes the effect of our own post-training easier to interpret. The choice is not a claim that Qwen3-8B-Base is the strongest or newest available Qwen model; it is the controlled starting point for the first experiment.

A separately post-trained Qwen model or a Lean-specialized model may be evaluated later as a reference, but it does not replace the unchanged base checkpoint as the primary baseline for measuring our training delta.

## D004 — First post-training method: SFT with QLoRA

**Status:** ACCEPTED

The first training method is supervised fine-tuning (SFT) using QLoRA.

SFT trains the model on known input/output examples. LoRA keeps the base model frozen and trains small low-rank adapter matrices; QLoRA combines LoRA with a quantized base model so the experiment fits on much less GPU memory than full-parameter fine-tuning.

The initial intent is 4-bit base-model quantization with trainable LoRA adapters. Full-parameter fine-tuning is not part of the first experiment.

**Exact quantization settings, LoRA rank/alpha, sequence length, optimizer parameters, and training schedule:** OPEN

## D005 — ML and inference stack

**Status:** ACCEPTED

Use the standard Python Hugging Face/PyTorch ecosystem rather than a custom training framework:

- PyTorch for tensor/GPU execution;
- Transformers for Qwen model and tokenizer loading;
- Hugging Face Datasets for dataset loading/processing;
- TRL for supervised and later post-training trainers;
- PEFT for LoRA adapters;
- bitsandbytes for the initial 4-bit QLoRA path;
- `uv` for Python environment and dependency execution;
- vLLM as the intended high-throughput inference runtime once evaluation exceeds tiny smoke tests;
- Lean 4/Lake as the initial proof-verification runtime.

Direct Transformers generation is acceptable for small diagnostics where introducing vLLM adds no value. The evaluator contract must remain the same regardless of inference transport.

**Experiment-tracking backend:** OPEN

## D006 — The repository is operated by agents, not by a human CLI user

**Status:** ACCEPTED

The user steers the project through ChatGPT and Codex; Codex is expected to implement code, modify configuration, and execute experiments on configured compute.

Python is therefore an implementation language, not a prerequisite for the user. The repository should optimize for clear agent execution: ordinary scripts/modules, declarative configuration where useful, machine-readable results, tests, and inspectable commands.

A polished end-user CLI and notebook-first workflow are not project goals. A small CLI or command wrapper may still be added when it reduces ambiguity or makes repeated agent execution safer, but it should not become a separate product surface.

## D007 — Start with real verified Lean data before synthetic scale

**Status:** ACCEPTED

The first supervised corpus should be built primarily from real verified Lean/mathlib theorem/proof pairs. Synthetic proof generation is deliberately postponed until a baseline training and verification loop exists.

The extraction path must preserve enough context to reconstruct examples and reason about split contamination. A maintained Lean extraction/tracing tool such as LeanDojo-v2 is a preferred candidate, but the exact extraction implementation is not yet fixed.

**Extraction implementation and retained context:** OPEN

The Phase 2 design should choose between LeanDojo-v2 and a simpler direct Lean/mathlib extraction path after inspecting what context the whole-proof task actually needs.

## D008 — Dataset splitting must be contamination-aware

**Status:** ACCEPTED

Do not rely on a random theorem-pair split for the main held-out comparison. Closely related lemmas from the same file or namespace can make a random split unrealistically easy.

Use a structural grouping boundary such as file, namespace, source unit, or a stronger contamination-aware rule.

**Exact split unit and deduplication rules:** OPEN

## D009 — miniF2F is the first external benchmark

**Status:** ACCEPTED

Use the Lean version of miniF2F as the first external theorem-proving benchmark because it is compact, interpretable, and widely used enough to make results easier to contextualize.

Use its development/validation portion during iteration. Treat the test split as a held-out benchmark for checkpoints selected without tuning against those test results.

A second, harder/current Lean benchmark should be added after the first evaluation path is working.

**Second external benchmark:** OPEN

## D010 — First training path should fit a single 24 GB NVIDIA GPU

**Status:** ACCEPTED

Design the first Qwen3-8B QLoRA training path to be viable on one 24 GB NVIDIA GPU, such as the RTX 3090 class, using memory-saving techniques when required. A 48 GB GPU is a convenience for larger batches/context and faster iteration, not a baseline requirement.

Training software should remain provider-neutral Linux/NVIDIA code. Vast.ai, OCI, Linode, Hugging Face Jobs, or another provider may be selected for training per experiment based on access, cost, and operational convenience.

Model inference and generation execution policy is a repository-wide invariant owned by `AGENTS.md`; this training decision does not override it.

Claims about fit, memory, throughput, or cost must be based on measured configurations rather than assumed from the GPU model name.

**Training provider and machine flavor per run:** OPEN

## D011 — Artifact ownership

**Status:** ACCEPTED

Keep source code, configuration, small fixtures, compact evaluation evidence, and documentation in GitHub.

Keep large model weights, LoRA checkpoints, datasets, caches, and bulky logs outside Git, normally in Hugging Face Hub or another appropriate artifact store. Check licenses and redistribution terms before publishing derived weights or datasets, and never commit secrets or machine credentials.

## D012 — Practical experiment fidelity, not high-assurance reproducibility

**Status:** ACCEPTED

The project needs enough metadata to understand what was run and make fair comparisons, but it does not target high-assurance or publication-grade reproducibility by default.

For material comparisons, record the model/tokenizer, dataset/split, relevant training or generation configuration, Lean/evaluation contract, hardware details that affect interpretation, and produced metrics. Deterministic seeds, repeated runs, immutable hashes, and extensive provenance are used when they matter to the question being answered, not as universal requirements.

Future-phase details such as verifier-filtered self-training policy, RL reward shaping, and tactic-level search architecture are intentionally deferred until their phase becomes active rather than recorded as premature decisions here.
