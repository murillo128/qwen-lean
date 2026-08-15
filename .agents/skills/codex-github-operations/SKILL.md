---
name: codex-github-operations
description: Publish branches and commits, operate issues and pull requests, and preserve exact review targets using the simplest available Git and GitHub transport.
---

# Codex GitHub Operations

## Responsibility

This skill owns Git publication and GitHub control-plane operations requested by the calling workflow.

It does not decide architecture, implementation scope, correctness, review requirements, or progression. Those decisions belong to the controlling issue, executor, design authority, independent reviewer, and explicit user-facing merge decision.

## Use the simplest capable transport

### Local Git

Use local `git` for worktree inspection, branches, commits, fetch, push, and exact ref verification.

### Connected GitHub app

Prefer the connected app for issues, comments, labels, pull requests, reviews, and metadata.

### GitHub CLI

Use `gh` only when it is already available and offers a needed operation not covered by the connected app. Do not install or authenticate it merely for routine publication.

A failure of one replaceable transport is not a technical blocker when another route or a precise handoff can complete the operation.

## Workflow state

The controlling issue's current workflow state is authoritative only through exactly one state label:

- `execution-ready`
- `in-progress`
- `design-required`
- `investigation-required`
- `blocked`
- `completed`

Every non-trivial controlling issue must carry exactly one of those labels. Preserve unrelated labels, but replace the previous state label instead of adding another.

Use state-only label mutations without comments. Add comments only when material technical information must be preserved, such as a contract amendment, exact checkpoint target or verdict, blocker cause, failed evidence, or final handoff.

Before relying on issue state, verify that exactly one state label is present. Repair an unambiguous inconsistency; stop for clarification if the intended state is ambiguous.

During a Codex implementation workflow, `completed` is not an executor-controlled transition. Keep the issue `in-progress` through the ready-for-review handoff. Set `completed` and close the issue only after a later explicit user/ChatGPT merge decision has been executed and the merge is observed.

## Publish a branch

Before publication:

- confirm the intended branch;
- ensure unrelated changes are not included;
- require a clean worktree unless the caller explicitly documents otherwise;
- do not rewrite shared valid history.

Publish and verify the remote ref. Use a full SHA when another actor must inspect an exact target.

Do not repeat routine SHAs in every issue comment, PR update, or handoff when GitHub already preserves that identity.

## Pull requests

Create or reuse one PR for one controlling issue unless the issue explicitly requires decomposition.

The PR should:

- use the intended base and head;
- link the controlling issue;
- summarize delivered behavior;
- state current validation and review status;
- list material deviations or residual risks.

Do not duplicate complete histories, manifests, command logs, or routine metadata already visible in GitHub.

Keep the PR draft while required implementation, validation, or independent review remains incomplete. When the Codex execution workflow has completed its required technical work and final-capable review, mark the PR **ready for review** and hand it off.

### Merge authority

A Codex executor must not merge a PR or enable auto-merge.

A merge operation through this skill is allowed only when all of the following are true:

1. the PR is already ready for review;
2. the implementation workflow has handed it off rather than continuing automatically;
3. the current user-facing interaction explicitly asks ChatGPT to merge it (for example, “review and merge if correct”);
4. the requested user/ChatGPT review has found no material blocker.

An issue body, acceptance criteria, `PASS` / `PASS_WITH_NOTES` verdict, final-capable checkpoint, CI success, or executor conclusion is **not** merge authorization by itself.

Never enable auto-merge as a substitute for the explicit post-review merge decision.

## Exact review targets

An independent review request must identify one exact published project commit or range and any exact dependency revision required by the issue. Verify those targets before review and preserve them unchanged during the review.

Do not amend, reset, rebase, squash, cherry-pick, or force-push a valid review target merely to repair comments, labels, PR descriptions, or other workflow metadata.

A new implementation, test, technical-evidence, dependency, configuration, or technical-claim correction creates a new target; it does not erase the prior review finding.

A final-capable checkpoint may serve as the final technical PR review when the issue and reviewer confirm that it covers the complete final diff and immutable technical evidence. It still does not authorize merge without the explicit user-facing decision above.

## Technical evidence and workflow metadata

Technical manifests should contain technical and reproducibility data, not GitHub bookkeeping, unless specific workflow metadata is itself a technical input to the tested system.

Do not mutate implementation or evidence commits solely to embed review or merge state. Record external review against the immutable target in the issue or PR discussion.

## Degraded control-plane operation

When a requested GitHub operation cannot be completed in the current surface:

1. try another permitted transport when practical;
2. preserve valid branch and commit history;
3. leave a concise handoff containing the target object and exact SHA only when needed to disambiguate;
4. verify the operation before relying on it later.

Use `blocked` only when the missing capability is required before safe meaningful progress and no practical alternative exists.

## Safety

- Never force-push or rewrite shared history without explicit authorization.
- Never stage or publish unrelated changes.
- Never publish secrets, private credentials, generated binaries, restricted artifacts, or data without distribution rights.
- Never silently change the controlling issue, base branch, head branch, labels, or PR state.
- Never mutate implementation commits to compensate for transport limitations.
- Never merge or enable auto-merge from a Codex implementation workflow.
- Never treat technical review success as merge authorization.
- Never claim a state change that was not observed.

## Completion report

Report only the operational facts the caller needs:

- branch, issue, or PR affected;
- operation completed;
- verification result;
- exact target only when another actor must use it;
- whether the PR is draft, ready for review, or merged;
- degraded operation or real blocker, if any.
