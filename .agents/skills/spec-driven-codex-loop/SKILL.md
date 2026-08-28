---
name: spec-driven-codex-loop
description: Execute an approved controlling issue through bounded implementation, repository-native validation, publication, independent review, and a ready-for-review handoff.
---

# Spec-Driven Codex Loop

## Responsibility

Use this skill for non-trivial implementation under an approved controlling issue. The issue is the complete phase-specific contract; repository documents define durable architecture; branches and PRs preserve implementation; tests and evaluation outputs preserve evidence.

The executor owns implementation, validation, commits, progression through technical review, and handoff. Delegate GitHub mutations to `codex-github-operations` and checkpoint/final review to `codex-independent-review`. The executor may not review its own work independently.

The executor's terminal delivery state is a PR that is **ready for review**, not merged. It must not merge the PR, enable auto-merge, or treat a technical review verdict as merge authorization. Merge acceptance belongs to a later explicit user/ChatGPT review-and-merge interaction.

## Context and authority

Load once:

1. `AGENTS.md` when present;
2. the controlling issue;
3. only the exact plan, decision, source, test, build, evaluation, model, dataset, artifact, or external inputs needed by the active outcome.

Do not weaken the issue, reconstruct its intent from history, or choose between materially different implementations when the issue is silent. Return to design instead.

Do not silently promote exploratory notes, hypotheses, brainstorming, or provisional chat conclusions into requirements. Use them only when the controlling issue or an authoritative repository source explicitly adopts them.

On resume, verify branch, `HEAD`, worktree, the controlling issue's single authoritative state label, and new material comments. Reuse unchanged inspected context.

## Entry gate and workflow state

Before editing, confirm:

- exactly one state label exists;
- it is `execution-ready` or `in-progress`;
- branch and worktree are safe;
- scope, invariants, failure semantics, acceptance, and required inputs are clear;
- no competing branch or PR creates ambiguous ownership.

Before the first implementation edit, use `codex-github-operations` to replace `execution-ready` with `in-progress`. Do not post a comment solely for this transition.

Use label replacements for execution-time returns:

- missing material design decision: `design-required`;
- evidence needed before design: `investigation-required`;
- genuinely unavailable external capability: `blocked`.

`completed` is a post-merge state. The Codex executor must not set `completed` or close the controlling issue as part of implementation delivery. After an explicit user/ChatGPT review accepts and merges the ready PR, the user-facing merge workflow may set `completed` and close the issue after observing the merge.

By default, add comments only when a material reason, technical finding, contract amendment, exact checkpoint target/verdict, blocker capability, or final handoff must be preserved. A calling workflow may explicitly request additional progress-observability comments; when it does, follow that narrow reporting policy without treating progress comments as checkpoints or technical evidence.

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

For submodules, vendored code, external repositories, models, datasets, generated artifacts, or other external inputs:

- preserve the identities required by the issue;
- update inputs only at coherent implementation or review boundaries when identity matters;
- publish any external target that another actor must inspect;
- never present an unavailable dependency state as a review target.

### 4. Validate honestly

Prefer repository-native build, test, lint, type-check, evaluation, and benchmark commands. Disposable diagnostics are acceptable during investigation; durable required validation should use the approved project path.

Run required and useful narrower checks. Record material deviations, environmental limits, and checks not run. Never claim an unrun check passed. A local implementation failure is corrected within scope; it is not an external blocker.

### 5. Retain evidence proportionally

Keep enough technical evidence to support the claim being made. This may include configuration, model/dataset identity, commands, results, metrics, artifacts, and limitations.

Do not add workflow bookkeeping to technical artifacts unless it is itself relevant to the tested system. Large model weights, datasets, caches, and bulky logs belong outside Git.

### 6. Publish intentionally

Publish when remote preservation, collaboration, a checkpoint, or PR review requires it. Exact SHAs are useful for review targets and dependency pins, not routine progress prose.

## Comments and progress observability

By default, comment only when:

- a checkpoint is ready;
- scope or acceptance changes;
- a material failure, blocker, design return, or investigation return needs its cause preserved;
- final handoff is ready.

Use:

```markdown
## <Checkpoint ready | Contract amendment | Design required | Investigation required | Blocked | Ready for review>

**Delivered or confirmed:** <one to three bullets>
**Validation:** <result or evidence>
**Material issue:** <none or concise finding>
**Next:** <one bounded action>
```

At a checkpoint, include the exact published target and any dependency revision needed for review.

When a calling workflow explicitly requests progress observability, concise progress comments are additionally allowed or required at the phase boundaries defined by that caller. Such comments must remain operational rather than evidentiary: they report what is running, what just finished, coarse progress when cheaply available, and what comes next.

Progress-observability comments:

- do not require a published review target;
- do not trigger independent review;
- do not change issue scope, acceptance, or workflow state;
- do not replace normal checkpoint, blocker, design/investigation-return, or final-handoff comments;
- should not reproduce logs or emit per-item/per-step chatter.

## Review checkpoints

At a declared checkpoint:

1. publish the exact target;
2. provide scope, material risks, acceptance criteria, and relevant evidence;
3. invoke one fresh independent review;
4. continue only after `PASS` or non-blocking `PASS_WITH_NOTES`.

A checkpoint may serve as final technical review when it covers the complete final diff and all remaining acceptance criteria. Any later technical change invalidates that verdict; workflow-only changes do not. A final technical verdict never authorizes merge by itself.

Progression:

- `PASS`: continue with `in-progress`; if this was the final-capable review and all implementation work is complete, prepare the PR for user-facing review;
- `PASS_WITH_NOTES`: continue unless a note violates an exit gate; if final-capable and non-blocking, prepare the PR for user-facing review;
- `FAIL`: choose a bounded correction, `design-required`, or `investigation-required`;
- `BLOCKED`: set `blocked` only when required evidence/review capability has no safe alternative;
- transport failure: use another route or leave a precise handoff; it is not an implementation verdict.

Do not mechanically implement every reviewer suggestion.

## Repeated-review circuit breaker

After two consecutive failures in substantially the same validation or bookkeeping mechanism, stop compensating patches and return to design authority before a third cycle unless the defect is materially different. This never waives a continuing technical defect.

## Pull request discipline

Use one PR per controlling issue unless the issue explicitly decomposes delivery. Keep it draft while required implementation, validation, or independent technical review remains incomplete.

When the complete final diff has passed the required validation and final-capable independent review, update the PR description with the final technical state and mark the PR **ready for review**. Then stop execution and hand it off.

The Codex executor must never:

- merge the PR;
- enable auto-merge;
- interpret `PASS` / `PASS_WITH_NOTES`, issue acceptance criteria, or a final-capable checkpoint as merge authorization;
- close the controlling issue or set it to `completed` before an explicit user/ChatGPT review accepts and merges the PR.

## Handoff

Include only what the next actor cannot derive cheaply:

- issue and current bounded outcome;
- ready-for-review PR and exact final reviewed target when useful;
- last accepted checkpoint;
- material evidence;
- unresolved non-blocking note or finding;
- immediate next action: user/ChatGPT review and merge decision.

Do not continue past this handoff unless a later explicit user-facing instruction requests review, correction, or merge.
