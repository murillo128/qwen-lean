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

**First full-cycle configuration:** ACCEPTED — the first full SFT cycle uses a
4-bit NF4 base with double quantization and BF16 compute; LoRA rank 16, alpha 32,
and dropout 0 over `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
and `down_proj`; the 1,024-token, no-packing, no-truncation `mathlib-sft-v1`
serialization; micro-batch 1 with gradient accumulation 8; paged 8-bit AdamW;
a `1e-4` cosine learning rate with a 1/32 warmup fraction; and one full eligible
training epoch. These settings define the first full cycle, not a claim of global
optimality or a requirement for unrelated future experiments.

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

The extraction path preserves enough context to reconstruct examples and reason about split contamination.

**Extraction implementation and retained context:** ACCEPTED — the first verified corpus uses LeanDojo-v2 repository tracing over the pinned mathlib source. Records retain source identity and spans, declaration and raw tactic-proof text, derived whole-proof completions, and resolved premise metadata sufficient to reconstruct the declaration in its original source-file context. Exact tool and source revisions remain phase-local configuration.

## D008 — Dataset splitting must be contamination-aware

**Status:** ACCEPTED

Do not rely on a random theorem-pair split for the main held-out comparison. Closely related lemmas from the same file or namespace can make a random split unrealistically easy.

**Exact split unit and deduplication rules:** ACCEPTED — the first internal split groups by source file and joins files into indivisible connected components when they share a normalized statement fingerprint. Components are assigned deterministically by record count to 90% train, 5% validation, and 5% held-out; held-out is not used for fitting or model selection. Exact normalized statement matches from the pinned miniF2F validation and test declarations are excluded before splitting.

## D009 — miniF2F is the first external benchmark

**Status:** ACCEPTED

Use the Lean version of miniF2F as the first external theorem-proving benchmark because it is compact, interpretable, and widely used enough to make results easier to contextualize.

The first Lean 4 evaluation cycle uses the Google DeepMind miniF2F fork in its own upstream Lean project; phase configuration pins the exact benchmark revision.

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

Future-phase details such as exact synthetic filtering thresholds, shaped-reward semantics, proof-state process-supervision serialization, and tactic-level search architecture are intentionally deferred until the phase that first needs them.

## D013 — Compare post-SFT methods as independent branches first

**Status:** ACCEPTED

Verifier-filtered self-training, verifier-reward reinforcement learning, and proof-state process supervision are initially separate experiments, not a mandatory sequential pipeline.

After the first full SFT cycle is evaluated, select one reference SFT checkpoint without tuning against the held-out benchmark test split. Retain that checkpoint as the control and use it as the **same initialization** for all three post-training branches:

- Branch A: verifier-filtered self-training using Lean-verified synthetic whole proofs as additional supervised targets;
- Branch B: GRPO / reinforcement learning from Lean verifier reward;
- Branch C: process supervision using Lean-grounded intermediate proof states and tactic/action transitions extracted from verified proofs.

The first Branch B experiment uses a binary outcome reward: `1` only when the final reconstructed theorem is `verified` by the accepted Lean evaluator, and `0` otherwise. It must not initialize from Branch A or Branch C for the first comparison.

Branch C must keep mechanically grounded process evidence distinct from free-form explanations. A proof-state trajectory such as `state -> tactic -> resulting state` can be treated as Lean-grounded training data when extracted from a verified/replayable proof. A natural-language rationale may be useful auxiliary data later, but it is not considered verified merely because the associated final proof succeeds.

The first Branch C experiment remains a **whole-proof model at evaluation time**. Using proof states during training does not change the accepted `whole-proof-v1` evaluation contract and does not itself authorize an interactive tactic-level inference loop. Tactic-level proving/search remains a separate later architecture decision.

Where practical, the three branches should use comparable theorem-source pools and data/generation budgets. Their final evaluation must use the same whole-proof prompt, Lean environment, verifier semantics, held-out workloads, and generation settings so differences can be attributed to the post-training method rather than the evaluator.

If binary GRPO produces insufficient positive reward density or useful within-group variation, preserve that as an experimental result. Syntax validity, elaboration success, tactic/proof-state progress, or other reward shaping may then be tested only as a **separate named variant** with an explicit reward contract; do not silently change the reward during the binary experiment. Any partial reward must remain secondary to complete verifier success rather than becoming an alternative objective.

Only after the independent comparison may a later experiment compose methods, for example verifier-filtered self-training followed by GRPO or process supervision followed by GRPO, to test whether their gains are complementary.

**Exact candidate counts, theorem pool, training budgets, GRPO group size, sparse-reward stopping rule, process-state extraction format, Branch C objective/loss mixture, and any shaped-reward formula:** OPEN

## D014 — First-SFT serialization and loss mask

**Status:** ACCEPTED

`mathlib-sft-v1` trains the plain `whole-proof-v1` proof-request suffix without
the source-file preamble: the shared instruction, retained canonical declaration,
and `:= by\n  ` prefix are followed by the exact retained Phase 2 completion and
one tokenizer EOS token. No chat template or other hidden prompt transformation
is applied.

The causal-LM objective supervises only the completion and terminal EOS. Prompt
tokens, padding, and batch-fill tokens have label `-100`; examples are not packed
or silently truncated. Phase-local sanity hyperparameters do not belong to this
durable serialization and masking contract.

## D015 — Reference SFT checkpoint for independent post-training branches

**Status:** ACCEPTED

The logical checkpoint `reference-sft-v1` is the controlled common initialization
and retained SFT control for the independent post-training branches defined by the
current durable roadmap. It consists of:

- base model `Qwen/Qwen3-8B-Base` at revision
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`;
- unmerged PEFT adapter `murillo2000/qwen3-8b-base-lean-sft-qlora` at Hub
  revision `5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5`;
- origin Phase 5 validation-selected step `9962` from
  `phase5-train-full-v1-lora`.

This checkpoint was selected and its identity frozen before Phase 6 training-set
generation and miniF2F-test outputs. Those later evaluation results characterize
the checkpoint but do not retroactively select it. `reference-sft-v1` is the
controlled parent chosen for this project experiment; it is not claimed to be a
globally optimal SFT checkpoint.
