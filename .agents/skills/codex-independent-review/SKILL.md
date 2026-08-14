---
name: codex-independent-review
description: Independently review an exact published target against the controlling issue's material risks and acceptance criteria, returning a concise risk-calibrated verdict.
---

# Codex Independent Review

## Responsibility

Use this skill for declared checkpoints and final review when a separate final review is required.

The reviewer owns independent exact-target inspection, proportional validation, materiality, and the verdict. It does not implement fixes, redesign the issue, mutate workflow state, publish commits, or continue execution.

## Trust the issue as the technical contract

Judge the implementation against the controlling issue's explicit contract. Do not replace settled decisions with a new design merely because another approach is possible.

Open linked sources only to inspect the exact implementation, durable decision, dependency, or evidence identified by the issue. Do not require the executor to reconstruct intent from unrelated project history.

## Minimal review packet

A fresh reviewer needs:

1. `AGENTS.md` when present;
2. this skill;
3. the controlling issue or exact checkpoint section;
4. the exact published target or range;
5. immutable technical evidence required by that checkpoint.

Load prior comments or reviews only when an unresolved material finding depends on them.

## Independence

The reviewer must:

- use a fresh context that does not inherit the executor's hidden reasoning;
- inspect exactly the requested target;
- remain read-only;
- judge evidence rather than intent;
- not implement corrections or advance later work.

Independence does not mean maximal hostility. Test declared risks and plausible normal use, not every imaginable malformed input or representational variant unless the issue requires that boundary.

## Profiles

### STANDARD

Review only declared material checkpoints. Stop when the material risk is adequately covered and no unsafe defect remains.

### HIGH_ASSURANCE

Apply additional issue-defined checks only within the explicit architecture, numerical, concurrency, persistence, backend, data, or security boundary. Do not infer this profile from issue size.

## Authority and evidence boundaries

Treat these as authoritative when the issue declares them:

- code and tests at the exact target;
- immutable input and artifact identities;
- technical manifests containing revisions, environment, commands, results, metrics, gates, and limitations;
- reproduced or credibly inspected runtime evidence.

Treat GitHub metadata and workflow bookkeeping as derived state unless they are themselves under test.

When raw evidence is stored in an immutable checksum-addressed external archive, validate its identity, index, relevant samples, reproduction path, and claimed aggregates proportionally.

## Materiality

Return `FAIL` only when a finding:

- violates an explicit invariant or acceptance criterion;
- exposes a plausible normal-path defect;
- makes required technical evidence materially false, incomplete, ambiguous, or non-reproducible;
- introduces unapproved scope, architecture, dependency, format, or behavior;
- makes progression from the reviewed target unsafe.

Use `PASS_WITH_NOTES` for editorial wording, bookkeeping, optional hardening, stale non-authoritative prose, or robustness outside the declared boundary when the technical outcome remains trustworthy.

For every `FAIL`, state:

1. the exact criterion violated;
2. the material consequence;
3. the smallest corrective delta, or why design must reopen.

## Review procedure

### 1. Establish risk and authority

Identify the checkpoint outcome, scope, invariants, acceptance criteria, authoritative evidence, exact target, and explicit threat or failure boundary.

### 2. Inspect the exact target

Check:

- diff and scope compliance;
- implementation and integration;
- credible required evidence;
- plausible correctness, ownership, lifetime, numerical, concurrency, data, backend, or performance failures covered by the checkpoint;
- unexpected dependencies, secrets, restricted artifacts, or behavior;
- safety to proceed.

### 3. Test proportionally

Run issue-defined validation when the environment supports it. Prefer checks capable of falsifying the claimed outcome.

Distinguish commands personally run from committed or external evidence inspected. Never claim an unrun check passed.

### 4. Determine whether the review is final

A checkpoint can serve as final review when the issue declares it final-capable and the exact target includes:

- the complete final diff;
- final dependency revisions relevant to the contract;
- immutable final technical evidence;
- all remaining acceptance criteria and unresolved material findings.

A later change to code, tests, technical evidence, dependencies, configuration, or technical claims invalidates that verdict and requires review of the changed target. Workflow-only metadata changes do not.

### 5. Report briefly

Record only:

- exact target;
- verdict and safety to proceed;
- whether the review serves as final review;
- material findings;
- validation run or evidence inspected;
- smallest required delta or non-blocking notes.

## Reviewer transport

Use any fresh isolated read-only reviewer capable of inspecting the exact target.

If one transport fails, preserve the target and try another permitted route. Do not amend implementation commits, guess a different target, or fall back to executor self-review.

Return `BLOCKED` only when required evidence or independent-review capability remains unavailable after practical alternatives are exhausted.

## Repeated-review circuit breaker

Under `STANDARD`, when two consecutive reviews fail for substantially the same validation, attestation, parser, documentation-sync, or bookkeeping mechanism:

- stop open-ended searches for representational variants;
- use `PASS_WITH_NOTES` when progression is technically safe and the remaining concern is non-material;
- request design review when the validation strategy itself prevents a trustworthy decision.

The circuit breaker never waives a continuing material defect.

## Verdicts

Return exactly one:

- `PASS`
- `PASS_WITH_NOTES`
- `FAIL`
- `BLOCKED`

Transport failure is an attempt result, not a verdict.

## Concise review request

```text
Act as a fresh independent read-only reviewer for <checkpoint> of issue #<issue>.

Review exact target <sha-or-range> against the issue's complete technical contract,
material risks, acceptance criteria, and immutable technical evidence. Inspect only
the context needed to determine whether progression is technically safe.

This checkpoint is <final-capable | not final-capable>. If final-capable, confirm
whether the target contains the complete final diff and all remaining evidence.

Return FAIL only for a concrete material violation, plausible normal-path defect,
untrustworthy required evidence, unapproved scope, or unsafe progression. Treat
workflow metadata, editorial, bookkeeping, and optional hardening concerns as
PASS_WITH_NOTES.

Do not implement fixes or mutate repository or GitHub state. Return exactly PASS,
PASS_WITH_NOTES, FAIL, or BLOCKED.
```

## Concise review comment

```markdown
## <Checkpoint> — PASS | PASS_WITH_NOTES | FAIL | BLOCKED

**Target:** `<SHA/range>`
**Safe to proceed:** yes | no
**Serves as final review:** yes | no

**Material findings:**
- <none or finding with violated criterion and consequence>

**Validation/evidence:**
- <command, manifest, artifact, or external archive identity and result>

**Required delta or notes:**
- <none, smallest correction, design return, missing evidence, or non-blocking note>
```
