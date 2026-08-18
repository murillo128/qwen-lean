# AGENTS.md

Instructions for ChatGPT, Codex, and other coding agents working in this repository.

## Mission

Develop and evaluate a Lean-focused language-model fine-tuning workflow. Repository documents define durable technical decisions, data/training/evaluation conventions, and active experiments; GitHub issues and pull requests define current work. Chat history is provisional when it conflicts with those sources.

## Load context progressively

For non-trivial work, load this bootstrap context once:

1. `AGENTS.md`;
2. the controlling GitHub issue body.

Then load only the context needed for the active role and phase:

- exact decision IDs, plan sections, validation sections, manifests, or evidence linked by the issue;
- relevant source, tests, training or evaluation scripts, configuration, and model/dataset state;
- the one workflow skill that owns the current action.

Read epic #1 only when selecting the next phase, checking dependencies, or updating global project status. An executor with a controlling issue does not need the epic as routine implementation context.

Do not preload every repository document, every skill, complete prior issue or pull-request histories, or whole result directories. Read a complete document only when the issue makes the whole document authoritative or section-level reading cannot resolve the task.

On session resume, verify branch, `HEAD`, worktree state, and new controlling-issue or PR discussion since the last material handoff. Do not replay unchanged history.

## Source-of-truth hierarchy

1. Tests, evaluation outputs, and captured evidence establish observed behavior.
2. `docs/DECISIONS.md`, when present, establishes accepted architecture and durable technical decisions.
3. `PLAN.md` and linked planning documents, when present, establish technical sequence and exit gates.
4. Dataset, model, training, and evaluation specifications establish experiment conventions.
5. The controlling issue establishes the bounded execution contract for its scope.
6. Pull requests, checks, reviews, experiment outputs, and Git history preserve implementation and evidence.
7. Epic #1 establishes operational roadmap status only.
8. Chat messages are provisional until recorded in an authoritative source.

When sources materially conflict, stop and document the conflict. Do not silently choose one.

Use these decision markers exactly in design notes: `ACCEPTED`, `OPEN`, `SPECULATIVE`, `REJECTED`, `OBSERVED`, and `BLOCKED`. Never present an `OPEN` or `SPECULATIVE` item as decided.

## Role routing and instruction ownership

Load skills lazily by role:

- design authority: `.agents/skills/design-github-issue/SKILL.md`;
- multi-issue orchestration: `.agents/skills/codex-issue-orchestrator/SKILL.md`;
- main executor: `.agents/skills/spec-driven-codex-loop/SKILL.md`;
- Git and GitHub mutation or publication: `.agents/skills/codex-github-operations/SKILL.md`;
- independent checkpoint or final review: `.agents/skills/codex-independent-review/SKILL.md`.

Do not read a role skill merely because it exists. The executor does not need the design or reviewer procedure; the reviewer does not need the executor or GitHub-operations procedure.

`AGENTS.md` owns repository-wide invariants and routing. Each skill owns its procedure. Issues own phase-specific scope, commands, and gates. Avoid copying the same rule into all three places; reference the owning source and record only the phase-specific delta.

Trivial typo-only edits may skip the complete issue workflow unless the user explicitly requests it, but repository safety and source-of-truth rules still apply.

## Model, data, and training constraints

Agents must not:

- silently change the base model, tokenizer, dataset split, prompt format, objective, quantization method, or evaluation protocol when those choices affect comparability;
- claim training or evaluation success from process completion alone;
- treat training loss as sufficient evidence of task improvement;
- mix training and held-out evaluation data without an explicit reason;
- publish model weights, datasets, or derived artifacts without checking their licenses and redistribution terms;
- include secrets, credentials, private datasets, or machine-specific access tokens in committed configuration.

Record material preprocessing and filtering rules when they affect interpretation of results. Exact immutable revisions and hashes are optional unless the current issue specifically needs them.

## Correctness and evaluation requirements

Changes to data preparation, training, inference, or evaluation must preserve a trustworthy comparison against an explicitly identified baseline when a comparison is being claimed.

Hard failures include:

- train/eval leakage or contamination that invalidates reported metrics;
- silent tokenizer or chat-template changes that alter the experiment contract;
- mismatched base-model, adapter, or tokenizer combinations;
- non-finite loss or gradients without explicit handling and diagnosis;
- evaluation scripts that score a different output format or task than the declared benchmark;
- metrics computed from incomplete, stale, or mixed experiment outputs;
- claiming an improvement when the compared runs differ in material ways unrelated to the change being tested.

Deterministic seeds and repeated runs are useful when variance matters, but are not mandatory by default.

For Lean tasks, keep syntactic validity, elaboration or proof-checking success, and task-level correctness distinct. A response that looks like Lean code is not a successful proof unless the declared evaluator accepts it.

## Experiment and artifact discipline

Keep enough experiment metadata to understand what was run and compare results meaningfully. For material experiments, this will usually include:

- base model and tokenizer;
- dataset and split;
- training/evaluation configuration;
- relevant environment or hardware details;
- commands or entry points used;
- produced metrics and known limitations.

Large checkpoints, model weights, datasets, caches, and bulky logs belong outside Git. Commit scripts, configs, small fixtures, and concise evidence that helps inspect or repeat useful experiments.

## Inference execution

All model inference and generation for this project must execute on project-controlled local GPU compute. The current default inference device is the available NVIDIA Ada GPU; a controlling issue may identify another project-controlled local device when needed.

Artifact distribution is separate from inference execution. Hugging Face Hub or another artifact store may be used to download or cache models, tokenizers, adapters, or datasets, but model forward passes and generation must not be delegated to Hugging Face Jobs, Inference API/Endpoints, Spaces compute, or another hosted inference/GPU-job service.

A runner without access to the project GPU must not silently substitute hosted compute. Use the repository's normal handoff/blocker workflow to continue on a runner that can access the local GPU.

This invariant applies to evaluation, smoke tests, baseline generation, and inference used during later training or data-generation workflows. Training compute is governed separately by the controlling phase and its accepted decisions.

## Performance and hardware claims

Measure the quantities relevant to the current issue instead of inferring them from hardware names or theoretical capability. When cost, memory, or throughput matters, record the actual training or inference configuration, precision/quantization, sequence lengths, batch sizes, gradient accumulation, device type, peak memory, throughput, and materially relevant timing data.

Do not compare runs as if they were equivalent when hardware, precision, sequence length, batch construction, kernel/compiler settings, or evaluation workload differ materially.

## External dependencies and upstream projects

Respect the current contribution, licensing, and disclosure rules of any external project or model repository being modified or targeted upstream. Internal repository authorization does not override an upstream project's policies.

Do not modify or vendor third-party code, datasets, model files, or prompts without checking applicable licensing and attribution requirements.

Never force-push or rewrite shared history without explicit user approval.

## Git behavior

- Do not commit generated model weights, adapter checkpoints, large datasets, caches, bulky traces, or benchmark binaries.
- Commit configs, scripts, summarized evidence, and small deterministic fixtures.
- Use explicit paths when staging.
- Avoid unrelated formatting changes.
- Commit messages should describe one intentional outcome.
- Direct commits to the default branch require explicit user instruction; otherwise use a feature branch and draft pull request.
- Codex implementation workflows end at a **ready-for-review** pull request and handoff; Codex must not merge or enable auto-merge.
- Merge requires an explicit user instruction in a later user-facing review interaction after ChatGPT/user review; issue acceptance, CI success, or independent-review `PASS` alone is not merge authorization.

## Current work

Use epic #1 to identify the current phase and controlling issue. Once a controlling issue exists, that issue and its PR are the active execution context; do not encode phase-specific status in this file or duplicate it into repository documents.
