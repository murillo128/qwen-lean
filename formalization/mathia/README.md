# Mathia WI-011 finite formalization

This directory machine-checks the finite deduction layer controlled by
[`murillo128/mathia#74`](https://github.com/murillo128/mathia/issues/74).
The original qwen-lean issue
[`#101`](https://github.com/murillo128/qwen-lean/issues/101) is a closed, superseded execution
record; this repository and PR remain the child Lean artifact for the Mathia-owned question.

- `WI011_GATE0.md` records the independently reviewed statement, adversarial, prior-art, mathlib,
  and dependency gate, plus its reconciliation with the repaired current Mathia WI-011.
- `WI011TraceEnergy.lean` proves the scalar trace--energy envelope, pressure transfer, and exact
  `m=438` branch/radical arithmetic.
- `WI011FourPointAssembly.lean` proves the four-point coefficient ledger, the parameterized local
  certificate-to-block theorem, exact shifted-window and finite endpoint accounting, and the
  combined `wi011_m438_finite_splice` theorem.

Build with:

```text
lake build MathiaFormalization
```

The source files print the axiom footprints of the principal theorems. They use only
`propext`, `Classical.choice`, and `Quot.sound`, the standard axioms reported by mathlib-backed
finite proofs.

## Attribution and integration boundary

The trace--energy envelope and window-in-frame accounting are prior art from
`tawanerguo-cn/zeta-simple-zeros`, independently rederived at historical revision
`0102fd8915c88fdd7c66231467745c17c0005fe4` of
`trmdy/zeta-simple-zeros-673137`. The local four-point certificate is the external Lean theorem
`four_point_cert` in `teal-sea/zeta-lab`; this repository does not copy or reprove it.

Accordingly, `wi011_m438_finite_splice` takes the local certificate as an explicit hypothesis.
Connecting that theorem to the external zeta stability/explicit-formula bridge requires a
compatible port or downstream import of the external artifact. This repository intentionally
does not add that Lean 4.33-rc2/path-dependent zeta graph to its pinned Lean/mathlib 4.32.0
environment. No theorem here claims the full asymptotic zeta result or its decimal proportion.
