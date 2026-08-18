# Riemann data/discovery stage evidence

This is a source-bounded census, not a claim of global completeness or progress on proving RH.

## Pinned inputs

- Phase 2 mathlib revision: `81a5d257c8e410db227a6665ed08f64fea08e997`
- Phase 2 records indexed: 88958
- External projects inspected: 4
- Native-verified external Lean records accepted: 22

## Internal graph and corpora

- `core`: 182
- `premise-1`: 139
- `premise-2`: 92
- `user-1`: 27
- `user-2`: 8
- `source-neighborhood`: 52
- `number-theory-control`: 3242

Corpus counts:

- `number-theory-far-holdout-v1`: 136
- `number-theory-wide-v1`: 3446
- `riemann-bubble-v1`: 452
- `riemann-core-v1`: 165
- `riemann-near-holdout-v1`: 356
- `riemann-specialist-validation-v1`: 556

## Atlas

- Entries: 234
- Typed mathematical relationships: 226
- Internal Lean premise/user/source edges are stored separately from mathematical relations.

## Material limitations

- RH itself, GRH, and heuristic criteria are knowledge anchors, not ordinary proof targets.
- PNT+ compiles at the pinned native revision, but modules/declarations containing or depending on `sorry` were excluded from the external corpus.
- Other-prover entries establish formalization presence only; they are not Lean SFT records.
- Literature statements are attributed knowledge records and never treated as verified Lean proofs.
- The census is bounded by the source list in `sources/source-manifest.json` and records unresolved gaps explicitly.
