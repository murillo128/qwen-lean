# WP-014 Schiffer two-point gate

## Verdict

`PASS_TO_PROOF`, subject to the blocking independent review required by issue
[#100](https://github.com/murillo128/qwen-lean/issues/100).

The proposed statement is faithful to Mathia WP-014, its singular cases can be
made explicit, the determinant reduction is exact, and no exact or stronger
compatible Lean formalization was found.  The proof should use an elementary
fifth-order sine bound built from existing mathlib results rather than
formalizing the Mittag--Leffler expansion.

## Sources and fixed mathematical statement

The authoritative inputs were inspected at Mathia commit
`ccf26f8956083aba8be9c3dbfff0b9c3dd2722da`:

- [WP-014](https://github.com/murillo128/mathia/blob/ccf26f8956083aba8be9c3dbfff0b9c3dd2722da/research/weil_positivity/findings/WP-014-exact-schiffer-kernel-is-not-positive-definite.md),
  the controlling derivation;
- [PF-085](https://github.com/murillo128/mathia/blob/ccf26f8956083aba8be9c3dbfff0b9c3dd2722da/research/prime_flute/findings/PF-085-grunsky-schiffer-completion-is-trace-class-and-misses-quarter-threshold.md),
  the upstream specialized kernel formula and diagonal value.

The Lean source will define the tail domain as a subtype and keep the
off-diagonal domain as a proof argument.  The full kernel will be an explicit
piecewise definition: the displayed continuous-extension value on the diagonal
and the specialized formula off the diagonal.  Thus no division-by-zero
default is used to encode a singular case.

The declarations are frozen to the following shape (names may gain local
helper lemmas without changing these public statements):

```lean
noncomputable section

namespace Mathia.WP014

abbrev TailPoint := {x : ℝ // 2 < x}

def delta (x y : TailPoint) : ℝ :=
  Real.pi * (1 / (x : ℝ) - 1 / (y : ℝ))

def offDiagonalKernel (x y : TailPoint) (_hxy : x ≠ y) : ℝ :=
  Real.pi ^ 2 / ((x : ℝ) ^ 2 * (y : ℝ) ^ 2) *
    (1 / Real.sin (delta x y) ^ 2 - 1 / delta x y ^ 2)

def kernel (x y : TailPoint) : ℝ :=
  if hxy : x = y then Real.pi ^ 2 / (3 * (x : ℝ) ^ 4)
  else offDiagonalKernel x y hxy

def twoPointGram (x y : TailPoint) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![kernel x x, kernel x y; kernel y x, kernel y y]

theorem abs_delta_mem (x y : TailPoint) (hxy : x ≠ y) :
    0 < |delta x y| ∧ |delta x y| < Real.pi / 2

theorem schifferScalar_gt_one_third {t : ℝ}
    (ht₀ : 0 < |t|) (htπ : |t| < Real.pi / 2) :
    1 / Real.sin t ^ 2 - 1 / t ^ 2 > 1 / 3

theorem det_twoPointGram (x y : TailPoint) (hxy : x ≠ y) :
    (twoPointGram x y).det =
      Real.pi ^ 4 / ((x : ℝ) ^ 4 * (y : ℝ) ^ 4) *
        (1 / 9 -
          (1 / Real.sin (delta x y) ^ 2 - 1 / delta x y ^ 2) ^ 2)

theorem det_twoPointGram_neg (x y : TailPoint) (hxy : x ≠ y) :
    (twoPointGram x y).det < 0

theorem twoPointGram_not_posSemidef (x y : TailPoint) (hxy : x ≠ y) :
    ¬ (twoPointGram x y).PosSemidef

end Mathia.WP014
```

The core scalar expression is deliberately `1 / Real.sin t ^ 2`; no cosecant
API is introduced.  The diagonal is accepted as the continuous-extension
definition supplied by PF-085.  Deriving that limit or the original
`V'(x)V'(y)/(V(y)-V(x))²` formula is not required for the two-point
obstruction and would dominate this bounded issue.

## Independent reconstruction

For distinct tail points, put

```text
d = pi * (1/x - 1/y)
f = 1 / sin(d)^2 - 1 / d^2.
```

Because `x,y > 2`, both reciprocals lie strictly between `0` and `1/2`.
Their difference therefore lies in `(-1/2, 1/2)`.  Distinct positive `x,y`
have distinct reciprocals, so multiplication by nonzero `pi` gives
`0 < |d| < pi/2`.  This also implies `sin d ≠ 0` (reduce to `|d|` and use
positivity of sine on `(0, pi)`).  All denominators in the off-diagonal formula
are consequently nonzero.

The two diagonal entries are `pi^2/(3*x^4)` and `pi^2/(3*y^4)`.  Both
off-diagonal entries are `pi^2*f/(x^2*y^2)`, since replacing `d` by `-d`
does not change either square.  Direct multiplication gives

```text
det = pi^4/(x^4*y^4) * (1/9 - f^2).
```

Thus `f > 1/3` implies `det < 0`, because the prefactor is strictly positive.
This establishes the required theorem and, through mathlib's generic theorem
that a positive-semidefinite complex/real matrix has nonnegative determinant,
the non-PSD corollary.

### Elementary scalar route

For `0 < t < pi/2`, define

```text
q(t) = t - t^3/6 + t^5/120.
```

Mathlib already proves `t - t^3/6 < sin t`.  Differentiating
`1 - t^2/2 + t^4/24 - cos t` reduces its strict positivity to that result;
differentiating `q(t) - sin t` then gives the first polynomial.  Two strict
monotonicity arguments from zero yield

```text
0 < sin t < q(t).
```

Set `u=t^2`.  From `t < pi/2` and `pi < 4`, one has `0 < u < 4`.  Exact ring
normalization gives

```text
(3 + u) * q(t)^2 - 3*u
  = u^3 / 14400 * (u^3 - 37*u^2 + 520*u - 2880).
```

The last polynomial is negative on `0 < u < 4`: write its first two terms as
`u^2*(u-37) ≤ 0`, while `520*u - 2880 < -800`.  Hence
`(3+t^2)*sin(t)^2 < 3*t^2`.  Clearing the now-explicit positive denominators
is exactly

```text
1 / sin(t)^2 - 1 / t^2 > 1/3.
```

The negative case follows by applying the positive result to `-t` and using
`Real.sin_neg`.  This proves the requested absolute-value statement without
series, limits, numerical certificates, or new analytic infrastructure.

## Adversarial checks

1. **Swap/sign symmetry.** Swapping `x,y` sends `d` to `-d`; both squared
   denominators and the scalar are unchanged.  The Gram matrix is symmetric.
2. **Near the diagonal.** As `x → y`, `d → 0` and `f(d) → 1/3` from
   above.  The determinant tends to zero from below but remains strictly
   negative for every distinct pair.  The explicit diagonal branch is
   therefore consistent with the source.
3. **Tail boundary.** In the joint extreme `x → 2+`, `y → ∞`,
   `|d| → pi/2` from below and `f(d) → 1 - 4/pi^2 > 1/3`.  For fixed
   `x>2` and `y → ∞`, the bracket stays negative while the positive
   `y^-4` prefactor sends the determinant to zero from below.
4. **Denominators.** Tail membership gives `x,y>0`; distinctness gives
   `d≠0`; `0<|d|<pi/2` gives `sin d≠0`.  The fifth-order proof clears
   denominators only after strict positivity is established.
5. **Determinant algebra.** Independent symbolic expansion produced
   `-pi^4*(3*f-1)*(3*f+1)/(9*x^4*y^4)`, identical to the displayed formula.
6. **Full scalar interval.** The polynomial argument uses only
   `0<t<pi/2` (relaxed to `t<2` for the terminal polynomial), so it covers the
   full open interval rather than a neighborhood of zero.
7. **Overall sign.** Negating every kernel entry multiplies a `2x2`
   determinant by `(-1)^2`; symbolic subtraction of the two determinants is
   zero.  The obstruction is unchanged.
8. **PSD semantics.** `Matrix.PosSemidef` includes Hermitian symmetry in its
   definition.  The corollary does not assume symmetry separately: a
   hypothetical `PosSemidef` proof would imply `0 ≤ det`, contradicting the
   proved strict negative determinant.  Symmetry is nevertheless true by item
   1.

Non-authoritative 80-decimal sanity checks sampled `t=10^-12`, `0.1`,
`pi/4`, and `pi/2-10^-30`; the observed margins `f(t)-1/3` were respectively
approximately `6.67e-26`, `6.68e-4`, `4.55e-2`, and `2.61e-1`.  Endpoint,
near-diagonal, and widely separated point pairs all produced negative
determinants.  These checks were used only to try to falsify the statement.

## Prior-art audit

Searches were performed on 2026-08-29 across GitHub code, indexed web/arXiv
results, and the pinned mathlib tree.  Exact-phrase and formula searches
included:

- `"pi cot(pi/x)"`, `"π cot(π/x)"`, and `"Schiffer kernel" cot`;
- `"csc^2 t" "1/t^2"`, `"csc² t" "1/t²"`, and the same concepts
  restricted to Lean;
- `Schiffer language:Lean` and a proposed scalar-theorem name.

The GitHub code API returned no Lean Schiffer result, no specialized
`pi*cot(pi/x)` result, and no exact scalar inequality.  General web searches
likewise found no exact or stronger formalization.  The only located source of
the specialization is the authoritative Mathia derivation above, which is
prose rather than a formal proof.

The boundary is therefore:

- generic Grunsky/Schiffer operator inequalities and Fredholm/Schatten
  constructions are prior art; for example
  [Takhtajan--Teo](https://arxiv.org/abs/math/0406408) treats Grunsky operators,
  Hilbert--Schmidt behavior, and associated Fredholm determinants;
- generic `2x2` determinant algebra and PSD determinant nonnegativity already
  exist in mathlib and will be reused;
- the exact `V(x)=pi*cot(pi/x)` two-point determinant obstruction was not found
  outside Mathia WP-014;
- the exact scalar inequality was not found as an existing compatible Lean
  theorem.

No new dependency or copied external proof is justified.

## Pinned mathlib reuse inventory

The audit used mathlib `v4.32.0` at commit
`81a5d257c8e410db227a6665ed08f64fea08e997`.  The focused source should import:

```lean
import Mathlib.Analysis.Matrix.PosDef
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Tactic
```

Exact reusable declarations and mechanisms:

- `Real.sin_gt_sub_cube` and `Real.sin_pos_of_pos_of_lt_pi` for the scalar
  proof and sine denominator;
- `Real.sin_neg`, `Real.pi_pos`, and `Real.pi_lt_four` for symmetry, signs,
  and the terminal interval bound;
- `one_div_lt_one_div_of_lt`, `inv_inj`, and `abs_lt` for the delta domain;
- `strictMonoOn_of_deriv_pos`, `fun_prop`, and the standard derivative simp
  lemmas for the fifth-order sine bound;
- `field_simp`, `ring`, `ring_nf`, `linarith`, and `nlinarith` for exact field
  and polynomial normalization;
- `Matrix.det_fin_two` for the determinant convention (row/column order
  `0,1` and formula `a*d-b*c`);
- `Matrix.PosSemidef.det_nonneg` for the non-PSD corollary.

`Real.cot` and `Real.cot_eq_cos_div_sin` exist, but introducing cotangent or a
new cosecant layer would only obscure the already-specialized formula.  No
reusable theorem for the fifth-order sine upper bound, the exact Schiffer
scalar inequality, or the specialized kernel was found.  No Mittag--Leffler
or infinite-series machinery is needed.
