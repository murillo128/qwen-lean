# WI-011 Gate 0: finite trace--energy and four-point assembly audit

Issue: [qwen-lean #101](https://github.com/murillo128/qwen-lean/issues/101)

Status: `OBSERVED` Gate-0 evidence. The executor proposes `PASS_TO_PROOF`, subject to the
blocking independent review required by the issue. This document freezes the proof surfaces and
records the repairs and exclusions found before any Lean proof work.

## 1. Bounded outcome and dependency decision

The implementation will add two focused Lean files:

- `formalization/mathia/WI011TraceEnergy.lean` for the scalar spectral envelope, pressure
  transfer, and the exact `m = 438` instantiation;
- `formalization/mathia/WI011FourPointAssembly.lean` for coefficient accounting, the
  local-certificate-to-block implication, and finite shift/endpoint counts.

No matrix corollary will be attempted. Mathlib exposes Hermitian eigenvalues, trace as a sum of
eigenvalues, and nonnegativity of PSD eigenvalues, but connecting the piecewise profile to a
matrix functional calculus would add machinery that the finite splice neither consumes nor
tests. The scalar `Fin m -> Real` theorem is the truthful boundary.

The external `teal-sea/zeta-lab` package will not become a qwen-lean dependency. Its
`four_point_cert` is a real, sorry-free theorem at the cited surface, but its package is built via
a path dependency on the much larger zeta bridge and uses Lean `v4.33.0-rc2` plus a different
mathlib revision. Qwen-lean is pinned to Lean/mathlib `v4.32.0`. Porting or vendoring that graph
would exceed this issue. The block theorem will instead take a visibly parameterized local
certificate hypothesis. No external certificate theorem is copied or restated as unconditional.

## 2. Frozen Lean theorem surfaces

Names may receive mechanically necessary argument-order adjustments during proof, but their
mathematical content and hypotheses are frozen here.

### 2.1 Scalar trace--energy layer

```lean
namespace Mathia.WI011

def psi (t : Real) : Real :=
  if t <= 2 then (t - 1) ^ 2 else 2 * t - 3

def energy {m : Nat} (lambda : Fin m -> Real) : Real :=
  sum i, (lambda i - 1) ^ 2

def defect {m : Nat} (lambda : Fin m -> Real) : Real :=
  sum i, psi (lambda i)

def phi (m : Nat) (E : Real) : Real :=
  if E <= (m : Real) / (m - 1) then E
  else 2 * Real.sqrt (((m - 1 : Nat) : Real) * E / m) - 1 + E / m

theorem defect_eq_energy_add_large_correction
    {m : Nat} (lambda : Fin m -> Real) :
    let L := Finset.univ.filter (fun i => 2 < lambda i)
    defect lambda = energy lambda
      + 2 * (sum i in L, (lambda i - 1)) - L.card
      - (sum i in L, (lambda i - 1) ^ 2)

theorem phi_monoOn_nonneg {m : Nat} (hm : 2 <= m) :
    MonotoneOn (phi m) (Set.Ici 0)

theorem phi_increment_le {m : Nat} (hm : 2 <= m)
    {x y : Real} (hx : 0 <= x) (hxy : x <= y) :
    phi m y <= phi m x + (y - x)

theorem traceEnergy_envelope {m : Nat} (hm : 2 <= m)
    (lambda : Fin m -> Real)
    (hlambda : forall i, 0 <= lambda i)
    (htrace : (sum i, lambda i) = m) :
    phi m (energy lambda) <= defect lambda

theorem traceEnergy_pressure {m : Nat} (hm : 2 <= m)
    (lambda : Fin m -> Real)
    (hlambda : forall i, 0 <= lambda i)
    (htrace : (sum i, lambda i) = m)
    {A P : Real} (hA : 0 <= A) (hP : 0 <= P)
    (hbudget : A <= energy lambda + P) :
    phi m A <= defect lambda + P
```

The pressure theorem deliberately has no unstated upper bound on `A` or `E`. Its exact domain is
`m >= 2`, nonnegative eigenvalues with trace `m`, `A >= 0`, `P >= 0`, and `E + P >= A`.

### 2.2 Four-point coefficient and block layer

For `q : Nat`, the block has `m = q + 4` points and `q + 1 = m - 3` consecutive
four-point windows. Pair weights are arbitrary and nonnegative; they need not be translation
invariant or arise from a kernel.

```lean
def fourPointPairSpend (w : Nat -> Nat -> Real) (s : Nat) : Real :=
  (2 / 3 : Real) *
      (w s (s + 1) + w (s + 1) (s + 2) + w (s + 2) (s + 3))
    + (w s (s + 2) + w (s + 1) (s + 3))
    + 2 * w s (s + 3)

def blockPairEnergy (q : Nat) (w : Nat -> Nat -> Real) : Real :=
  2 * sum r in Finset.range (q + 3),
        sum i in Finset.range (q + 3 - r), w i (i + r + 1)

theorem blockPairEnergy_eq_pairSum (q : Nat) (w : Nat -> Nat -> Real) :
    blockPairEnergy q w =
      2 * sum i in Finset.range (q + 4),
            sum j in Finset.Ico (i + 1) (q + 4), w i j

theorem fourPointPairSpend_sum_le_blockPairEnergy
    (q : Nat) (w : Nat -> Nat -> Real)
    (hw : forall i j, 0 <= w i j) :
    (sum s in Finset.range (q + 1), fourPointPairSpend w s)
      <= blockPairEnergy q w

theorem localCertificate_to_block
    (q : Nat) (w : Nat -> Nat -> Real) (pressure : Nat -> Real)
    (epsilon : Real)
    (hw : forall i j, 0 <= w i j)
    (hpressure : forall s, s < q + 1 -> 0 <= pressure s)
    (hcert : forall s, s < q + 1 ->
      epsilon <= fourPointPairSpend w s + pressure s) :
    epsilon * (q + 1) <=
      blockPairEnergy q w + sum s in Finset.range (q + 1), pressure s
```

The offset form of `blockPairEnergy` enumerates every pair `0 <= i < j < q + 4` exactly once,
with `r + 1 = j - i`; the equality theorem exposes the conventional `2 * sum (i < j)` form.

### 2.3 Exact finite shift and endpoint layer

```lean
def fourPointContainedAtOffset {m : Nat} (a : Fin m) : Prop :=
  a.val + 3 < m

theorem fourPoint_containing_shift_count {m : Nat} (hm : 4 <= m) :
    ((Finset.univ : Finset (Fin m)).filter fourPointContainedAtOffset).card = m - 3

def threeGapSpan (g : Nat -> Real) (s : Nat) : Real :=
  g s + g (s + 1) + g (s + 2)

theorem threeGapSpan_boundary_identity (q : Nat) (g : Nat -> Real) :
    3 * (sum j in Finset.range (q + 3), g j) =
      (sum s in Finset.range (q + 1), threeGapSpan g s)
        + 2 * g 0 + g 1 + g (q + 1) + 2 * g (q + 2)

def fullBlockOffsets (n q s : Nat) : Finset Nat :=
  (Finset.range (q + 1)).filter
    (fun t => t <= s && s - t + (q + 4) <= n)

theorem fullBlockOffsets_card_of_interior {n q s : Nat}
    (hleft : q <= s) (hright : s + q + 4 <= n) :
    (fullBlockOffsets n q s).card = q + 1

theorem exceptional_fourPoint_starts_card_le {n q : Nat} :
    ((Finset.range (n - 3)).filter
      (fun s => not (q <= s && s + q + 4 <= n))).card <= 2 * q

theorem finite_containment_incidence_with_boundary {n q : Nat} :
    (q + 1) * (n - 3) <=
      (sum s in Finset.range (n - 3), (fullBlockOffsets n q s).card)
        + 2 * q * (q + 1)
```

The first theorem is the exact `m - 3` count across all `m` alignments. The next identity shows
that interior gaps occur in exactly three three-gap spans and that the entire finite loss is the
explicit endpoint expression `2*g 0 + g 1 + g (q+1) + 2*g (q+2)`. The last two statements
isolate the incomplete-first/last-block effect: at most `2q = 2(m-4)` window starts are
exceptional, and at most `2q(q+1)` containment incidences are lost. Thus fixed `m` produces a
literal finite constant before any analytic asymptotic passage.

### 2.4 Exact `m = 438` arithmetic

```lean
def epsilon4 : Real := 231 / 100000
def A438 : Real := 20097 / 20000

theorem A438_eq : epsilon4 * (438 - 3) = A438
theorem A438_gt_branch : (438 : Real) / 437 < A438
theorem phi438_exact :
    phi 438 A438 =
      2 * Real.sqrt (8782389 / 8760000) - 1 + 20097 / 8760000
theorem phi438_interval :
    (1004848 / 1000000 : Real) < phi 438 A438
      /\ phi 438 A438 < 1004849 / 1000000
theorem phi438_lt_two : phi 438 A438 < 2
```

The exact theorem remains primary. The rational interval is a small certified regression, not a
floating-point premise.

## 3. Mathematical rederivation and adversarial findings

### 3.1 Trace--energy identity and cases

Put `x_i = lambda_i - 1`, so `x_i >= -1`, `sum x_i = 0`, and
`E = sum x_i^2`. Let `L = {i | x_i > 1}`, `k = |L|`,
`R = sum_L x_i`, and `Q = sum_L x_i^2`. Since `psi (1+x)` replaces `x^2`
by `2x-1` exactly on `L`, direct sum splitting gives

```text
D = E + 2R - k - Q.
```

If `k = 0`, then `D = E`, and `phi_m(E) <= E`; on the second branch the difference is
`(sqrt((m-1)E/m) - 1)^2`.

If `k = 1`, write the large coordinate as `r`. Cauchy--Schwarz on the other `m-1`
coordinates gives `m*r^2 <= (m-1)E`. With
`s = sqrt((m-1)E/m)`, `1 <= r <= s`, and

```text
D - phi_m(E) = (s - r) * (s + r - 2) >= 0.
```

The cited prior-art sketch handles `k >= 2` only by `D > 2`. That is enough for the WI-011
application because `phi_438(A438) < 2`, but it does not prove the global theorem
`D >= phi_m(E)` when `E` is large. The proof plan is repaired as follows.

For each large coordinate write `x_i = 1 + y_i`, `y_i >= 0`. Concentrating the nonnegative
`y_i` in one coordinate gives

```text
Q <= (R - k + 1)^2 + (k - 1).
```

Set `r = R-k+1`, `S = E-Q`, and
`E' = S + r^2 + (k-1)`. This replaces the `k` large coordinates by `r,1,...,1`:
it preserves `D`, preserves the zero sum, and gives `E <= E'`. Cauchy on the original
complement gives `R^2 <= (m-k)S`; the exact identity

```text
(m-1)R^2 + (m-1)(k-1)(m-k) - (m-k)(R-k+1)^2
  = (k-1)(R+m-k)^2
```

then yields `m*r^2 <= (m-1)E'`. The valid one-large-coordinate argument proves
`D >= phi_m(E')`, and monotonicity gives `phi_m(E') >= phi_m(E)`. This closes every `k` without
an upper range on `E`.

The branch point is continuous: both formulas equal `m/(m-1)`. The second-branch slope is
nonnegative and at most one because its square-root argument is at least one at the branch.
An algebraic proof using `sqrt(c*y)^2 - sqrt(c*x)^2 = c(y-x)` establishes
`phi(y) - phi(x) <= y-x`; no differentiability API is required. Monotonicity plus this
one-sided 1-Lipschitz estimate proves pressure transfer for every `A >= 0`.

### 3.2 Four-point coefficients and block energy

The WI-009 functional assigns coefficients `2/3`, `1`, and `2` to pairs separated by
`r = 1,2,3` gaps. In a consecutive four-point-window sum, those pairs occur at most
`3,2,1` times respectively. Their total spends are therefore at most `2` in every case,
exactly the coefficient of the block energy. Boundary pairs occur fewer times and add slack.
No symmetry, translation invariance, monotonicity, or kernel identity is used; only
nonnegativity of each pair weight is needed.

Summing a parameterized local hypothesis over the `q+1=m-3` windows and applying the spend
bound gives `E + P >= epsilon*(m-3)`. Nonnegative pressure is stated separately so the scalar
pressure-transfer theorem can consume it.

### 3.3 Shift and endpoint counts

A four-point window may start in block positions `0,...,m-4`, exactly `m-3` positions among the
`m` shifted frames. Equivalently, the three internal cuts exclude exactly three alignments.

For a finite list of `q+3=m-1` adjacent gaps, direct summation gives

```text
3 * sum(gaps)
  = sum(all three-gap spans)
    + 2*g_0 + g_1 + g_(m-3) + 2*g_(m-2).
```

This formula remains correct for `m=4`: the two middle endpoint terms name the same gap and
must both occur. It avoids an invalid disjoint-endpoint assumption. Full-block containment can
fail only for the first or last `m-4` window starts, hence at most `2(m-4)` exceptional starts
and at most `2(m-4)(m-3)` lost incidences. These are explicit finite terms, not formal `O(1)` or
`o(N)` assumptions.

### 3.4 Executable adversarial checks

An exact `Fraction` enumeration tested all 130,450 nonnegative rational spectra of total trace
`m` on the quarter grid for `m=2,...,6` (per-size counts below).
The global envelope and the minimal-pressure WI-011 implication held in every case:

| `m` | spectra | `k=0` | `k=1` | `k>=2` |
|---:|---:|---:|---:|---:|
| 2 | 9 | 9 | 0 | 0 |
| 3 | 91 | 61 | 30 | 0 |
| 4 | 969 | 489 | 480 | 0 |
| 5 | 10,626 | 3,951 | 6,525 | 150 |
| 6 | 118,755 | 32,661 | 79,164 | 6,930 |

Direct enumeration of the required small blocks produced:

| `m` | gap-in-span multiplicities | pair multiplicities at `r=1;2;3` | containing shifts |
|---:|---|---|---:|
| 4 | `1,1,1` | `1,1,1 ; 1,1 ; 1` | 1 |
| 5 | `1,2,2,1` | `1,2,2,1 ; 1,2,1 ; 1,1` | 2 |
| 6 | `1,2,3,2,1` | `1,2,3,2,1 ; 1,2,2,1 ; 1,1,1` | 3 |

These checks are falsification aids only. The delivered claims will be Lean theorems.

## 4. Exact constant audit

Exact rational calculation gives

```text
epsilon * (438-3) = (231/100000)*435 = 20097/20000 = A438,
A438 - 438/437 = 22389/8740000 > 0,
(437/438)*A438 = 8782389/8760000 = 2927463/2920000.
```

Thus the radical and second branch in the issue are correct. A high-precision diagnostic gives
`phi_438(A438) = 1.0048483690271541680...`; Lean will instead prove the exact formula and the
rational enclosure `(1.004848, 1.004849)`.

## 5. Prior-art and existing-formalization audit

The sources were inspected at these public revisions:

| source | inspected revision | classification |
|---|---|---|
| `murillo128/mathia` WI-009/WI-011 | `ccf26f8956083aba8be9c3dbfff0b9c3dd2722da` | WI-011 is the candidate splice under test; it explicitly disclaims novelty for the envelope and window accounting. |
| `tawanerguo-cn/zeta-simple-zeros` | `45149f6d403059a71be73c5e3f884cee7cd62b20` | MIT prior art for the fixed-`A` trace--energy implication and shifted pressure assembly; no Lean source is present. Its current paper explicitly says no global minimizer claim is needed. |
| `trmdy/zeta-simple-zeros-673137` | historical `0102fd8915c88fdd7c66231467745c17c0005fe4` | MIT independent rederivation in `docs/refined-deduction.md`, explicitly crediting tawanerguo; not Lean-formalized. Later HEAD `1610b97...` archives newer operating points, so the historical revision is the exact evidence target. |
| `teal-sea/zeta-lab` | `c02ad1a56ce18d99c326d87e9318d064621d3fea` | MIT, sorry-free external `four_point_cert` and zeta stability bridge. Its four-point theorem has exactly the WI-009 coefficients and target `2310/10^6`; it does not contain the WI-011 `m=438` splice or exact constant. |

GitHub code searches for `1.004848369027154`, `8782389/8760000`, `Phi_438`, and the combined
`m=438`/`four_point_cert` target found no public Lean implementation. The tawanerguo and trmdy
trees contain no Lean files. Zeta-lab's generic bridge has related older block and shift
machinery, but the checked four-point package depends by path on that large bridge, is not a
self-contained compatible import, and is pinned to Lean `v4.33.0-rc2`. Therefore the missing,
compatible delta is the finite scalar/combinatorial splice specified above.

Attribution in code comments and evidence will say:

- trace--energy envelope and window-in-frame accounting: prior art;
- four-point certificate: external Lean-checked input, parameterized here rather than imported;
- exact `m=438` combination: Mathia splice being formalized;
- no claim of an end-to-end zeta theorem or of the decimal `0.672852563956...` in qwen-lean.

## 6. Mathlib v4.32.0 reuse inventory

The pinned local mathlib source supplies the required building blocks:

| need | declarations / import area |
|---|---|
| finite sums and ranges | `Finset.sum_range_succ`, `Finset.sum_bij`, `Finset.card_range`, `Finset.card_Ico`; `Mathlib.Algebra.BigOperators.Group.Finset.Basic`, `Mathlib.Data.Finset.Interval` |
| filtered finite sets | `Finset.card_filter_add_card_filter_not`, `Finset.filter_subset`, sum/filter decomposition; `Mathlib.Data.Finset.Card` |
| Cauchy--Schwarz | `Finset.sq_sum_le_card_mul_sum_sq`; `Mathlib.Algebra.Order.Chebyshev` |
| concentration of nonnegative squares | `Finset.sum_sq_le_sq_sum_of_nonneg`; `Mathlib.Algebra.Order.BigOperators.Ring.Finset` |
| square roots | `Real.sqrt_nonneg`, `Real.sq_sqrt`, `Real.sqrt_le_sqrt`, `Real.sqrt_monotone`; `Mathlib.Analysis.Real.Sqrt` / `Mathlib.Analysis.SpecialFunctions.Sqrt` |
| piecewise/order algebra | `MonotoneOn`, `Set.Ici`, `by_cases`, `split_ifs`; core order and `Mathlib.Order` imports |
| exact arithmetic | `norm_num`, `positivity`, `linarith`, `nlinarith`, `ring_nf`, `field_simp`, `omega`; available through focused Mathlib tactic imports (or `import Mathlib` if import minimization becomes brittle) |
| optional matrix API audited, not selected | `Matrix.IsHermitian.eigenvalues`, `Matrix.trace_eq_sum_eigenvalues`, `Matrix.PosSemidef.eigenvalues_nonneg`; `Mathlib.Analysis.Matrix.Spectrum`, `Mathlib.Analysis.Matrix.PosDef` |

No custom generic linear-algebra framework is warranted. The combinatorial file will use
`Finset.range` and induction because the objects are ordered consecutive windows; graph or matrix
abstractions would obscure the exact endpoint arithmetic.

## 7. Gate verdict

Proposed verdict: `PASS_TO_PROOF`.

The adversarial audit found and repaired an incompleteness in the published global-envelope proof
sketch, froze the missing pressure domain, preserved repeated endpoint terms at `m=4`, and found no
counterexample, coefficient mismatch, branch error, arithmetic error, compatible existing Lean
implementation, or justified external dependency. Proof work may begin only if the independent
Gate-0 reviewer accepts these surfaces and the compression repair.
