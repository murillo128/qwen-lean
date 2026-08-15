---
name: design-github-issue
description: Define a self-contained execution-ready GitHub issue that resolves material decisions and gives a fresh executor the facts needed to implement safely, then explain the designed mechanism separately to the user in chat.
---

# Design a GitHub Execution Issue

## Responsibility

Use this skill before non-trivial implementation starts, or when execution returns because a material design or validation decision is unresolved.

The design authority owns:

- the observable outcome;
- material architectural and validation decisions;
- the complete phase-specific context needed to execute safely;
- scope, invariants, exclusions, failure semantics, and acceptance criteria;
- risk-based review checkpoints when useful;
- the issue's initial readiness and any design-authority state transition;
- a separate concise explanation to the user of how the designed mechanism works when that helps them learn the project.

It does not implement code, operate branches, publish commits, perform independent review, or grant the executor merge authority.

## Assume a fresh executor

Design the issue for an executor that:

- has no access to the design session's reasoning;
- should not need to reconstruct material facts from prior issues or PR history;
- must be able to distinguish required behavior from examples, observations, alternatives, and future work.

The issue must contain every phase-specific fact, decision, constraint, and acceptance rule required for correct implementation. Links are supporting references, not substitutes for material instructions.

## Keep user teaching outside the issue

The GitHub issue is an execution contract for Codex or another implementation agent. Do **not** add tutorial-style or pedagogical sections solely to teach the user, including a `How it works` section whose purpose is user education.

After designing or publishing the issue, explain the relevant mechanism separately in the assistant's chat response when the work introduces a non-obvious mechanism, model-training technique, evaluation concept, data flow, or system boundary.

For that user-facing explanation:

- explain the end-to-end mechanism or data/control flow, not implementation line by line;
- define specialized terms the first time they are needed, without assuming Python, ML-framework, training, or systems knowledge unnecessarily;
- make clear how the important components interact and why the chosen approach matters;
- mention a material trade-off when it explains a design choice, without enumerating every alternative;
- distinguish concepts from product/library names where confusion is plausible;
- keep it proportional: normally one to four short paragraphs or a compact bullet flow.

The issue itself may still describe technical data flow, semantics, or component interaction when Codex needs that information to implement correctly. Keep such text contractual and implementation-oriented, not pedagogical.

## Load material design context

Start with:

1. `AGENTS.md` when present;
2. the request, roadmap item, or existing controlling issue.

Then inspect only what is needed to settle the phase:

- exact plan and decision sections;
- relevant source seams, APIs, ownership boundaries, state, and tests;
- baseline behavior or prior experiment results that constrain the work;
- required hardware, model, dataset, artifact, dependency, or environment inputs;
- overlapping current work and superseded attempts when their findings constrain the design.

Prefer authoritative current outcomes over complete historical traversal.

## The issue is the executor's complete contract

Depending on the phase, include:

- current limitation and observable goal;
- accepted baseline behavior and defaults that must remain unchanged;
- relevant model, dataset, dependency, or artifact inputs when they affect the result;
- inspected implementation seams and data shapes;
- resolved API or configuration semantics and invalid combinations;
- ordering, failure behavior, and resource constraints where relevant;
- permitted implementation scope and explicit exclusions;
- commands, targets, fixtures, hardware, datasets, and artifacts needed for validation;
- objective acceptance criteria and material review risks;
- prior negative evidence when it prohibits repeating a known-invalid mechanism.

Use precise names, paths, values, examples, and equations where they remove ambiguity.

Do not copy generic Git, publication, review, label, merge, or reporting procedure already owned by skills. Do not duplicate chronological histories, complete logs, routine GitHub metadata, or user-oriented teaching content.

An issue may state observable post-merge completion conditions, but it must not authorize Codex/the executor to merge or enable auto-merge. The repository GitHub workflow owns that boundary: Codex delivers a ready-for-review PR; a later explicit user/ChatGPT review decides whether to merge.

## Readiness

Use exactly one workflow state label:

- `execution-ready`
- `design-required`
- `investigation-required`
- `blocked`
- `in-progress`
- `completed`

At issue publication, set exactly one state label through `codex-github-operations`. The issue body may record **Initial state** for historical context, but the label is authoritative for current state.

## Design method

### 1. Define the observable outcome

State what must become true, why it matters, the current limitation, and the boundary of the requested change.

### 2. Resolve material unknowns

Resolve questions that can change behavior, compatibility, architecture, data handling, model behavior, failure handling, validation, licensing, or deployment strategy.

Use these classifications only when useful:

- `OBSERVED`
- `ACCEPTED`
- `OPEN`
- `SPECULATIVE`
- `REJECTED`
- `BLOCKED`

Do not turn `OPEN` or `SPECULATIVE` items into implementation requirements. Record durable cross-phase architecture in an appropriate repository decision document; keep phase-local choices in the issue.

### 3. Bound implementation without under-specifying it

Define the smallest coherent outcome, permitted subsystem or files, explicit exclusions, and invariants. Include exact files or seams when an executor could otherwise modify the wrong layer.

### 4. Define validation that proves the outcome

Specify material validation concretely:

- repository-native build, test, lint, evaluation, or benchmark targets;
- correctness, repeated-run, failure-path, numerical, data, or performance checks when relevant;
- required environment and external artifacts;
- objective pass/fail criteria;
- technical evidence artifacts when useful.

Use exact commands when arguments or environment are part of what is being proven; otherwise identify the target and required result without freezing replaceable invocation syntax.

### 5. Keep evidence proportional

Capture enough technical evidence to support the decision or comparison being made. This may include model/dataset identity, configuration, commands, results, metrics, artifacts, and limitations.

Do not require elaborate provenance, immutable archives, hashes, or machine-readable manifests unless the issue specifically needs them.

### 6. Add review checkpoints when they reduce risk

Add independent checkpoints only for distinct material risks such as architecture, data integrity, numerical behavior, backend execution, broad refactoring, or decision-driving performance evidence.

A checkpoint defines:

- the covered outcome and target semantics;
- material risks and acceptance criteria;
- evidence to inspect or reproduce;
- what would make progression unsafe.

When the last checkpoint can inspect the complete final diff and all remaining acceptance criteria, it can be declared **final-capable**. A final-capable verdict is a technical gate to the ready-for-review handoff, not merge authorization.

### 7. Define dependency and publication boundaries

When work depends on external repositories, models, datasets, or generated artifacts, specify the identity and update boundary needed for the current task. Do not require bookkeeping commits that add no technical value.

### 8. Define restart semantics

Distinguish:

- local implementation defect: correct a bounded delta;
- design defect: return to `design-required`;
- evidence gap: return to `investigation-required`;
- replaceable tool failure: use another transport or leave a handoff;
- real blocker: no safe practical continuation exists.

Two consecutive review failures for substantially the same validation or bookkeeping mechanism should trigger design review before a third corrective cycle. This never waives a continuing material defect.

### 9. Check overlap

Inspect only plausibly overlapping open issues, PRs, branches, and recent attempts. Link superseded work and summarize its material constraint instead of copying its history.

### 10. Explain the mechanism to the user in chat

After the issue contract is complete, give the user a separate concise explanation of what will be built and how its important pieces interact. This explanation belongs in the assistant response, not in the GitHub issue.

Do not repeat the issue field by field. Focus on concepts the user needs in order to understand the project and the decisions just made.

## Execution-ready check

Before marking the issue `execution-ready`, confirm:

- a fresh executor can implement without design-session reasoning;
- the observable outcome and terminology are unambiguous;
- all material facts and decisions are present;
- linked sources supplement rather than replace the contract;
- scope, invariants, failure behavior, and acceptance are clear;
- required inputs and validation capabilities are identified;
- review checkpoints, if any, match distinct risks;
- dependency and external-evidence boundaries are explicit when applicable;
- no user-oriented tutorial content has been added merely for pedagogy;
- no issue text grants the executor merge or auto-merge authority;
- `execution-ready` is the issue's only state label.

Then ensure the assistant response explains any non-obvious mechanism to the user separately and at an appropriate level.

## Issue structure

```markdown
# <Outcome-oriented title>

## Readiness
**Initial state:** execution-ready | design-required | investigation-required | blocked

## Goal and current limitation
<Observable outcome, why it matters, and current behavior.>

## Baseline and inputs
<Material baseline facts, models, datasets, artifacts, and defaults.>

## Resolved technical contract
<APIs, data flow, failure semantics, bounds, and concrete seams.>

## Scope
### In scope
### Out of scope
### Invariants

## Validation and evidence
<Required targets, cases, environment, artifacts, and objective gates.>

## Checkpoints
<Only distinct material-risk checkpoints; mark the last one final-capable when applicable.>

## Delivery
<PR shape, dependency/publication boundaries, ready-for-review handoff, and observable post-merge completion. Do not authorize executor merge.>
```

Add or split sections when technical completeness requires it.
