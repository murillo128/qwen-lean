---
name: codex-issue-orchestrator
description: Orchestrate multiple controlling GitHub issues through fresh Codex workers in parallel or sequential order while preserving each issue's normal spec-driven execution workflow.
---

# Codex Issue Orchestrator

## Responsibility

Use this skill when one parent Codex session must execute a known set of controlling GitHub issues through separate worker agents.

This skill owns only:

- issue ordering and dispatch;
- parallel versus sequential scheduling;
- independent versus dependent issue semantics;
- the base ref passed to each worker;
- failure propagation between issues;
- batch-worker progress observability;
- the final orchestration summary.

It does **not** own issue implementation, validation, publication, review, or merge decisions. Every worker executes exactly one controlling issue using `.agents/skills/spec-driven-codex-loop/SKILL.md` and the normal repository workflow.

The orchestrator must not implement an issue itself merely because a worker fails, and it must not redesign an issue from batch-level context.

## Required orchestration inputs

Resolve these before dispatch:

- an explicit ordered list of controlling issues;
- `schedule`: `parallel` or `sequential`;
- `dependency`: `independent` or `dependent`;
- the initial base branch or exact base ref.

Valid combinations are:

| Schedule | Dependency | Meaning |
| --- | --- | --- |
| `parallel` | `independent` | launch independent workers concurrently |
| `sequential` | `independent` | run independent workers one at a time |
| `sequential` | `dependent` | run a stacked dependency chain one issue at a time |

`parallel + dependent` is invalid. Stop and request a coherent orchestration mode instead of inventing dependency behavior.

Do not infer extra work from neighboring issues, labels, or an epic body once the issue list has been resolved.

## Worker contract

For every dispatched issue:

1. start a fresh worker agent;
2. give it exactly one controlling issue and the base ref selected by this skill;
3. require it to load `AGENTS.md`, that controlling issue, and `spec-driven-codex-loop` normally;
4. enable the orchestration progress-observability policy defined below;
5. let the worker own its branch, implementation, validation, evidence, commits, independent review, PR publication, and ready-for-review handoff;
6. wait for the worker outcome according to the selected schedule;
7. record only the outcome needed by the orchestration loop.

Do not carry implementation conclusions, temporary diagnostics, prompt changes, benchmark observations, or uncommitted state from one worker into another unless `dependency=dependent` and the state is preserved in the predecessor's published branch by design.

Each controlling issue still gets its own PR under the normal spec-driven workflow. This skill never merges or enables auto-merge.

## Batch worker observability

Orchestrated execution is intentionally more observable than an ordinary single-issue Codex run. The worker should leave concise progress comments on its **controlling issue** in addition to the normal material checkpoint and final-handoff comments.

This is a progress-reporting request from the calling workflow; it does not change the issue contract, acceptance criteria, review gates, or source-of-truth hierarchy.

Post a progress update when it gives a remote observer useful new state, normally at these boundaries:

- worker started and the execution/base context is established;
- a material preflight, compatibility, setup, or environment phase finishes;
- a long-running training, generation, evaluation, migration, or similar phase starts;
- a long-running phase reaches a useful coarse progress boundary when that progress is cheaply available;
- that long-running phase finishes and the worker moves into validation/evidence/review;
- an unexpected failure, retry, fallback decision, or blocker materially changes what the worker is doing.

For a long-running loop, prefer a few coarse updates such as roughly quarter/half/three-quarter progress over time-based chatter. Do not comment per task, candidate, test case, training step, retry, or log line.

A progress comment should be short and operational, for example:

```markdown
## Progress

**Current:** full benchmark generation started
**Completed:** model/revision pin + dev smoke
**Progress:** 0/244 tasks
**Next:** finish generation, then Lean verification
```

Omit fields that are not useful. Update counts only when the worker already has them cheaply; do not add instrumentation solely to produce issue comments.

Progress comments are observability, not checkpoints:

- they do not require publication of a review target;
- they do not trigger independent review;
- they do not authorize contract changes;
- they do not replace the normal checkpoint, blocker, or ready-for-review comments required by `spec-driven-codex-loop`.

## Independent issues

Independent issues must not inherit unmerged sibling work.

For each worker:

- branch from the designated common base branch/ref, not from another issue's PR branch;
- keep its implementation and PR independent of sibling outcomes;
- do not change its issue contract because an earlier or concurrent model, benchmark, or implementation performed well or poorly.

### Sequential independent

Run one worker, wait for it to terminate, record its outcome, then always continue to the next issue.

A worker ending in any of these states does **not** block later issues:

- ready-for-review;
- blocked;
- design-required;
- investigation-required;
- implementation or validation failure;
- worker/process/transport failure.

The orchestrator records the failure and continues. It does not repair the failed issue itself.

### Parallel independent

Launch one fresh worker per issue concurrently.

Each worker uses the same designated independent base policy and remains isolated from sibling branches. One worker failing or blocking must not cancel unrelated workers.

Wait for all workers to reach a terminal handoff/failure state, then summarize the batch.

## Sequential dependent issues

Use this mode only when each issue intentionally builds on the previous issue's unmerged implementation.

The chain is stacked:

```text
initial base
   |
issue A branch -> PR A
   |
issue B branch -> PR B (base: issue A branch)
   |
issue C branch -> PR C (base: issue B branch)
```

Rules:

- the first worker starts from the declared initial base;
- each later worker starts from the exact published head of the immediately preceding successful issue;
- each later PR targets the predecessor branch so its diff represents only that issue's additional work;
- a fresh worker is still used for every issue;
- do not merge predecessor PRs automatically to advance the chain.

A downstream issue may start only after its predecessor reaches a valid ready-for-review handoff with a published branch/head suitable as the next base.

Any predecessor outcome that does not provide that successful dependency state stops the chain. This includes `blocked`, `design-required`, `investigation-required`, failed implementation/validation/review, worker failure, or an unavailable/ambiguous predecessor branch.

When the chain stops:

- preserve the predecessor's real outcome;
- do not launch downstream workers;
- report the first blocking issue and mark later issues as not executed due to dependency failure.

If an external action changes or merges a branch while the dependent chain is running and the intended next base becomes ambiguous, stop instead of guessing a new stack topology.

## Existing work and resume behavior

Before dispatching an issue, avoid duplicating work that is already terminal for the requested purpose.

- `completed` issues are recorded as already completed unless the caller explicitly requests re-execution;
- an existing ready-for-review handoff may be reused as the issue outcome when no additional execution was requested;
- an in-progress issue with an existing branch/PR should be resumed by its worker through the normal spec-driven workflow rather than creating competing ownership.

If ownership is ambiguous, do not launch a second executor for the same controlling issue.

## Orchestrator failure policy

The parent orchestrator should remain small and durable:

- worker-local technical failures belong to the worker and controlling issue;
- an independent worker failure is recorded, not escalated into a batch abort;
- a dependent worker failure stops only the dependency chain as defined above;
- failure to launch the orchestration mechanism itself is an orchestrator failure and should be reported precisely.

Do not convert a batch orchestration problem into a scientific or implementation decision inside a child issue.

## Completion

After all permitted workers have terminated, produce one compact table with at least:

| Issue | Outcome | PR / branch | Result or blocker |
| --- | --- | --- | --- |

Useful outcomes include:

- `ready-for-review`;
- `already-completed`;
- `blocked`;
- `design-required`;
- `investigation-required`;
- `failed`;
- `not-run-dependency-failure`.

Do not duplicate complete worker logs or PR histories. The issue and PR remain the source of truth for each worker's detailed execution.

The orchestration completes when every independent issue has terminated, or when a dependent chain has either completed or stopped at its first failed dependency. It never implies that any PR is approved for merge.
