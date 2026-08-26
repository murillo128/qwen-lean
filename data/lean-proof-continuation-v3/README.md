# Lean proof-continuation Dataset v3

This directory is the committed reproducibility snapshot for the canonical
`lean-proof-continuation-v3` package built by GitHub issue #82. Dataset v2 and
its official results remain frozen historical evidence.

## Contract

Dataset v3 recovers each real proof's raw `source_expression` against the exact
pinned source checkout. A raw term proof remains a term proof; an existing
`by` proof keeps its original tactic/layout form. Only declaration equation and
`where` forms use a recorded standalone transformation because those source
forms cannot be appended directly after `theorem ... :=`.

The canonical record stores a proof once. Each continuation is a reference to a
conservative Lean-layout boundary plus prefix/continuation/reconstruction
hashes. The optimizer view materializes either:

```text
declaration :=                         -> whole proof
declaration := verified proof prefix   -> remaining proof continuation
```

Every training theorem contributes exact mass `1`. For proofs with safe
boundaries, half is assigned to whole proofs and half to continuations; mass is
split first across structurally unique variants and then across boundaries. If
no safe boundary exists, the full theorem mass remains on the whole proof. The
ratio is view configuration, so a later training issue can change it without
re-extracting source records.

## Files

- `manifest.json` freezes configuration, package hashes, counts, and gates.
- `verification.json` records pinned-source recovery, Lean acceptance,
  reconstruction, mass, leakage, placeholder, and no-drop evidence.
- `diagnostics.json` reports raw-record and optimizer-mass proof-form,
  transformation, continuation, length, variant, source, composition, and
  theorem-contribution diagnostics.
- `validation-membership.jsonl` freezes the clean v3 selection membership.
- `test-membership.jsonl` freezes the sealed v3 final-test membership. It must
  not be used for checkpoint selection or hyperparameter decisions.
- `materialization-fixture.json` demonstrates one whole-proof and incremental
  examples end to end.

The large `records.jsonl.gz` and `optimizer-view.jsonl.gz` artifacts remain
local under `artifacts/dataset-v3/lean-proof-continuation-v3/`. Their exact byte
sizes and SHA-256 hashes are frozen in `manifest.json`; derived rows contain
references rather than redundant proof copies.

## Rebuild

Prepare the exact target checkout declared in `config/dataset-v3.json` at
`artifacts/riemann/sources/PrimeNumberTheoremAnd`, including its pinned mathlib
dependencies, then run:

```bash
uv run python tools/dataset_v3_build.py \
  --target-root artifacts/riemann/sources/PrimeNumberTheoremAnd
```

The build refuses source/hash drift, placeholders, unresolved source lemmas,
cross-split statement/proof/derivation leakage, non-unit theorem mass, unsafe
truncation/drop, or Lean-invalid generated and persisted proofs.

Consumers resolve records with `qwen_lean.dataset_v3.read_records`, resolve the
reference-only optimizer view with `read_view`, and call `materialize_example`.
The materializer raises rather than truncating or silently omitting an example
that exceeds a caller-supplied context limit.
