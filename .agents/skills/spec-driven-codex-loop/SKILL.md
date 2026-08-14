---
name: spec-driven-codex-loop
description: Execute an approved controlling issue through bounded implementation, repository-native validation, publication, independent review, and concise handoff.
---

# Spec-Driven Codex Loop

## Responsibility

Use this skill for non-trivial implementation under an approved controlling issue. The issue is the complete phase-specific contract; repository documents define durable architecture; branches and PRs preserve implementation; tests and technical manifests preserve evidence.

The executor owns implementation, validation, commits, progression, and handoff. Delegate GitHub mutations to `codex-github-operations` and checkpoint/final review to `codex-independent-review`. The executor may not review its own work independently.

## Context and authority

Load once:

1. `AGENTS.md` when present;
2. the controlling issue;
3. only the exact plan, decision, source, test, build, evaluation, manifest, model, dataset, or external inputs needed by the active outcome.

Assume the design authority may have more context. Do not weaken the issue, reconstruct its intent from history, or choose between materially different implementations when the issue is silent. Return to design instead.

On resume, verify branch, `HEAD`, worktree, the controlling issue's single authoritative state label, and new material comments. Reuse unchanged inspected context.

## Profiles

`STANDARD` is the default: implement one coherent outcome at a time, validate before publication, review only declared material checkpoints, reuse an unchanged final-capable checkpoint as final review, and report only material events.

Use `HIGH_ASSURANCE` only when the issue explicitly requires it and only for the additional risks or evidence it defines.

## Entry gate and workflow state

Before editing, confirm:

- exactly one state label exists;
- it is `execution-ready` or `in-progress`;
- branch and worktree are safe;
- scope, invariants, failure semantics, acceptance, and required inputs are clear;
- no competing branch or PR creates ambiguous ownership.

Before the first implementation edit, use `codex-github-operations` to replace `execution-ready` with `in-progress`. Do not post a comment solely for this transition.

Use label replacements for later transitions:

- missing material design decision: `design-required`;
- evidence needed before design: `investigation-required`;
- genuinely unavailable external capability: `blocked`;
- accepted terminal outcome: `completed` and close the issue.

Add comments only when a material reason, technical finding, contract amendment, exact checkpoint target/verdict, blocker capability, or final handoff must be preserved.

## Execution loop

### 1. Establish the bounded outcome

Confirm intended behavior, permitted subsystem, invariants, required validation/evidence, and next checkpoint. Do not combine unrelated work.

### 2. Implement the smallest coherent delta

- follow the issue and accepted architecture;
- preserve baseline behavior outside scope;
- add tests or evaluation coverage with implementation when required;
- use repository-native integration;
- avoid unrelated cleanup and formatting;
- stop when evidence invalidates the design or acceptance strategy.

Commits should represent reviewable outcomes. Mechanical substeps do not need separate commits.

### 3. Handle dependencies and external inputs deliberately

For submodules, vendored code, external repositories, models, datasets, generated artifacts, or other pinned inputs:

- preserve exact identities required by the issue;
- update pins only at coherent implementation or review boundaries;
- publish any external target that another actor must inspect;
- never present an uncommitted, mutable, or unavailable dependency state as a review target.

### 4. Validate honestly

Prefer repository-native build, test, lint, type-check, evaluation, and benchmark commands. Disposable diagnostics are acceptable during investigation; durable required validation should use the approved project path.

Run required and useful narrower checks. Record material deviations, environmental limits, and checks not run. Never claim an unrun check passed. A local implementation failure is corrected within scope; it is not an external blocker.

For long commands, use an appropriate wait rather than repeated short polling. Silence or a wait timeout is not failure while the process remains active.

### 5. Build and retain evidence proportionally

When required, use machine-readable technical evidence containing technical facts such as:

- implementation and dependency revisions;
- model, dataset, input, and artifact identities and hashes;
- environment and configuration;
- commands and results;
- metrics, gates, outcomes, and limitations.

Do not add issue/PR numbers, branch names, labels, comment IDs, review verdicts, merge commits, or closeout state unless one is a technical input to the tested system. Do not modify a manifest merely to record review or merge.

Keep manifests, bounded summaries, schemas, reproduction tooling, small deterministic fixtures, and checksum indexes in Git. Put large repetitive raw evidence in an authorized immutable checksum-addressed archive. Never publish secrets, restricted data, model weights, or artifacts without distribution rights.

### 6. Publish intentionally

Publish when remote preservation, collaboration, a checkpoint, or PR review requires it. Exact full SHAs belong at review targets, immutable evidence boundaries, recovery handoffs, and pinned dependencies—not routine progress prose.

## Material comments

Under `STANDARD`, comment only when:

- a checkpoint is ready;
- scope or acceptance changes;
- a material failure, blocker, design return, or investigation return needs its cause preserved;
- final handoff is ready.

Use:

```markdown
## <Checkpoint ready | Contract amendment | Design required | Investigation required | Blocked | Complete>

**Delivered or confirmed:** <one to three bullets>
**Validation:** <result or authoritative evidence>
**Material issue:** <none or concise finding>
**Next:** <one bounded action>
```

At a checkpoint, include the exact published target and any exact dependency revisions needed for review. Otherwise omit routine commit metadata.

## Review checkpoints

At a declared checkpoint:

1. publish the exact target;
2. provide scope, material risks, acceptance criteria, and immutable evidence;
3. invoke one fresh independent review;
4. continue only after `PASS` or non-blocking `PASS_WITH_NOTES`.

A final-capable checkpoint is the final PR review only when it covers the complete final diff, final relevant dependency targets, immutable final evidence, and all remaining acceptance criteria. Any later technical change invalidates it; workflow-only changes do not.

Progression:

- `PASS`: continue with `in-progress`;
- `PASS_WITH_NOTES`: continue unless a note violates an exit gate;
- `FAIL`: choose a bounded correction, `design-required`, `investigation-required`, or calibration;
- `BLOCKED`: set `blocked` only when required evidence/review capability has no safe alternative and record the exact missing capability;
- transport failure: use another route or leave a precise handoff; it is not an implementation verdict.

Do not mechanically implement every reviewer suggestion.

## Repeated-review circuit breaker

Under `STANDARD`, after two consecutive failures in substantially the same validation, attestation, parser, documentation-sync, or bookkeeping mechanism, stop compensating patches, preserve valid evidence, and return to design authority before a third cycle unless the defect is materially different. This never waives a continuing technical defect.

## Pull request discipline

Use one PR per controlling issue unless the issue explicitly decomposes delivery. The PR body contains only the controlling issue, delivered behavior, current validation/review state, and material deviations or residual risks. Keep it draft until required implementation and review are complete.

## Handoff

Include only what the next actor cannot derive cheaply:

- issue and current bounded outcome;
- branch or PR;
- last accepted checkpoint;
- material evidence;
- unresolved finding;
- one immediate next action.

Include exact heads only when needed to disambiguate a target or preserve recovery.
