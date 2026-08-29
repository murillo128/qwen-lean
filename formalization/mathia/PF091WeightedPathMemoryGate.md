# PF-091 weighted-path memory: statement and reuse gate

Date: 2026-08-29
Toolchain: Lean 4.32.0, mathlib `v4.32.0` (`81a5d257c8e410db227a6665ed08f64fea08e997`)

This is the blocking Gate-0 artifact for qwen-lean issue #102. It audits only the
finite weighted-path algebra in Mathia PF-080, PF-081, and PF-091. Burger's
surface theorem, collar/PDE errors, prime realization, and the factor
`1 / (2 * π^2)` are outside the proposed Lean declarations.

## Gate recommendation

`PASS_TO_PROOF`, subject to fresh independent review of this committed artifact.
No proof implementation may start before that review passes.

The generic electrical-network facts are standard and are not claimed as Mathia
novelty. The project-specific value of the proposed file is a small, checked
specialization that exposes the ordered stronger-side resistance correction.

## Frozen declaration surfaces

The proof file will use zero-based `Fin` indices. Thus `n` edges mean `n + 1`
vertices; Lean edge `e : Fin n` is mathematical edge `m = e.val + 1`.
Names may receive harmless namespace qualification, but these statement surfaces
are frozen:

```lean
def weightedPath3 (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ
def muMinus (a b : ℝ) : ℝ
def muPlus (a b : ℝ) : ℝ
def IsEigenvalue (M : Matrix (Fin 3) (Fin 3) ℝ) (mu : ℝ) : Prop

theorem weightedPath3_characteristic_factor (a b lambda : ℝ) :
    (lambda • (1 : Matrix (Fin 3) (Fin 3) ℝ) - weightedPath3 a b).det =
      lambda * (lambda ^ 2 - 2 * (a + b) * lambda + 3 * a * b)

theorem muMinus_isEigenvalue (ha : 0 < a) (hb : 0 < b) :
    IsEigenvalue (weightedPath3 a b) (muMinus a b)
theorem muPlus_isEigenvalue (ha : 0 < a) (hb : 0 < b) :
    IsEigenvalue (weightedPath3 a b) (muPlus a b)
theorem muMinus_pos (ha : 0 < a) (hb : 0 < b) : 0 < muMinus a b
theorem muMinus_lt_muPlus (ha : 0 < a) (hb : 0 < b) :
    muMinus a b < muPlus a b

def h (r : ℝ) : ℝ := 1 + r - Real.sqrt (1 - r + r ^ 2)
theorem h_quadratic_limit :
    Tendsto (fun r => (h r - (3 / 2) * r) / r ^ 2)
      (𝓝[>] (0 : ℝ)) (𝓝 (-3 / 8 : ℝ))
theorem muMinus_scaled_limit (a : ℝ) (ha : 0 < a) :
    Tendsto
      (fun b => (muMinus a b - (3 / 2) * b) / (b ^ 2 / a))
      (𝓝[>] (0 : ℝ)) (𝓝 (-3 / 8 : ℝ))

def feshbachU : Fin 3 → ℝ
def feshbachPsi : Fin 3 → ℝ
def weakEdgeMatrix : Matrix (Fin 3) (Fin 3) ℝ
theorem feshbachU_norm_sq : feshbachU ⬝ᵥ feshbachU = 1
theorem feshbachPsi_norm_sq : feshbachPsi ⬝ᵥ feshbachPsi = 1
theorem feshbachU_orthogonal_psi : feshbachU ⬝ᵥ feshbachPsi = 0
theorem feshbach_overlap_sq :
    |feshbachU ⬝ᵥ (weakEdgeMatrix *ᵥ feshbachPsi)| ^ 2 = 3 / 4

def pathCurrent (n : ℕ) (e : Fin n) : ℝ
def pathPotential {n : ℕ} (w : Fin n → ℝ) : Fin (n + 1) → ℝ
def pathEnergy {n : ℕ} (w : Fin n → ℝ) (x : Fin (n + 1) → ℝ) : ℝ
def pathAverage {n : ℕ} (x : Fin (n + 1) → ℝ) : ℝ

theorem pathPotential_edge_increment (hpos : ∀ e, 0 < w e) (e : Fin n) :
    pathPotential w e.succ - pathPotential w e.castSucc = pathCurrent n e / w e
theorem pathPotential_weak_equation (hpos : ∀ e, 0 < w e)
    (y : Fin (n + 1) → ℝ) :
    (∑ e, w e *
      (pathPotential w e.succ - pathPotential w e.castSucc) *
      (y e.succ - y e.castSucc)) =
      y (Fin.last n) - pathAverage y
theorem pathResistance_eq (hpos : ∀ e, 0 < w e) :
    pathPotential w (Fin.last n) - pathAverage (pathPotential w) =
      (1 / (n + 1 : ℝ) ^ 2) *
        ∑ e, (e.val + 1 : ℝ) ^ 2 / w e
theorem pathEnergy_eq_resistance (hpos : ∀ e, 0 < w e) :
    pathEnergy w (pathPotential w) =
      pathPotential w (Fin.last n) - pathAverage (pathPotential w)
```

Small `n = 1, 2, 3` corollaries will check respectively

```text
1/(4*w₁)
(1/w₁ + 4/w₂)/9
(1/w₁ + 4/w₂ + 9/w₃)/16.
```

The `n = 1` corollary will also check
`-(3/2) * b^2 * (1/(4*a)) = -(3/8) * b^2/a`. The optional
hierarchy-dominance theorem is deliberately omitted: it is not needed for the
exact finite mechanism, and stating it faithfully would add an asymptotic family
of weights rather than a modest corollary.

## Independent algebra and adversarial checks

### Three-vertex spectrum

Expanding `det(lambda * I - G)` independently gives

```text
lambda * (lambda^2 - 2*(a+b)*lambda + 3*a*b).
```

The quadratic discriminant is

```text
4 * ((a+b)^2 - 3*a*b) = 4 * (a^2 - a*b + b^2),
```

so the two roots are exactly `a+b +/- sqrt(a^2-a*b+b^2)`.
For `a,b > 0`, the radicand is `(a-b)^2 + a*b > 0`, while
`a^2-a*b+b^2 < (a+b)^2` because the difference is `3*a*b`.
Consequently both roots are positive and the minus root is strictly smaller.
No hypothesis `a > b` is used.

Degenerate and symmetric checks:

- `b = 0`, `a > 0`: eigenvalues `0, 0, 2*a`;
- `a = 0`, `b > 0`: eigenvalues `0, 0, 2*b`;
- `a = b > 0`: eigenvalues `0, a, 3*a`;
- scaling by `c > 0` multiplies the matrix and all three eigenvalues by `c`.

These checks also fix the units: the second term `b^2/a` has the same units as
an edge weight/eigenvalue.

### The `-3/8` coefficient without Taylor machinery

Let `s = sqrt(1-r+r^2)`. Near zero on the right, all denominators below are
positive. Two exact rationalizations give

```text
h(r) = 3*r / (1+r+s)
1-r-s = -r / (1-r+s).
```

For `r != 0`, substitution therefore yields

```text
(h(r) - (3/2)*r) / r^2
  = -3 / (2 * (1+r+s) * (1-r+s)).
```

Continuity of `sqrt` and ordinary field limits send the right-hand side to
`-3/(2*2*2) = -3/8`. This fixes the sign without a bespoke `BigO` or a
uniform-remainder claim. For `a > 0`,

```text
muMinus(a,b) = a * h(b/a),
a * (b/a)^2 = b^2/a,
```

which gives the scaled theorem.

### Feshbach normalization and orientation

With real Euclidean dot product,

```text
u   = (1,-1,0)/sqrt(2)
psi = (1,1,-2)/sqrt(6)
B₂  = [[0,0,0],[0,1,-1],[0,-1,1]].
```

Direct multiplication gives `u.u = 1`, `psi.psi = 1`, `u.psi = 0`, and

```text
B₂*psi = (0,3,-3)/sqrt(6),
u.(B₂*psi) = -3/sqrt(12),
|u.(B₂*psi)|^2 = 9/12 = 3/4.
```

Reversing the edge orientation does not change `B₂ = d*d^T`; changing the sign
of either normalized vector changes only the unsquared overlap. PF-081 contains
one prose occurrence of `nu` where the defined vector is `u`; the issue body has
already repaired that typo, and Lean will use `feshbachU` consistently.

### Weighted-path current, factor, and zero mode

Put `j = n + 1`. The zero-mean endpoint source is

```text
q = e_last - (1/j)*1.
```

Orient edges from vertex `m-1` to `m` and define
`I_m = w_m * (x_m-x_(m-1))`. Kirchhoff conservation forces
`I_m = m/j` for `m = 1,...,j-1`: the vertex-zero equation is
`-I_1 = -1/j`, every interior equation is `I_m-I_(m+1) = -1/j`,
and the endpoint equation is `I_(j-1) = (j-1)/j`.

For conductance `w_m > 0`, choose a potential with
`x_m-x_(m-1) = I_m/w_m`. Discrete summation by parts gives the weak equation

```text
sum_m w_m*(x_m-x_(m-1))*(y_m-y_(m-1))
  = y_last - average(y).
```

Taking `y=x` identifies endpoint-vs-average voltage drop with Dirichlet energy,
and substitution gives

```text
x_last - average(x)
  = sum_m I_m^2/w_m
  = (1/j^2) * sum_m m^2/w_m.
```

This formulation makes orientation harmless (both current and potential drop
change sign) and makes the gauge explicit (adding a constant to `x` changes
neither side).

The literal Mathia notation is valid under the Moore--Penrose convention, but
the invariant source must be exposed. If `L^+ * 1 = 0` and
`q = e_last-(1/j)*1`, symmetry gives

```text
e_last^T L^+ e_last = q^T L^+ q.
```

The right side is the just-proved endpoint-vs-average voltage/energy. Lean will
prove that elementary weak equation and energy identity, rather than introduce
a new pseudoinverse theory merely to restate the left side.

The factor `1/j^2` is therefore confirmed. For `j=2`, `R=1/(4*a)`, and the
Feshbach prefactor `-(j+1)/j = -3/2` produces exactly `-3*b^2/(8*a)`.

The PF-091 dominance sentence is safe only with enough hierarchy hypotheses.
The single ratio `w_(j-1)/w_(j-2) -> 0` does not by itself control every earlier
term, whereas PF-091 assumes the full fixed adjacent hierarchy. This gate avoids
silently weakening that condition by omitting the optional dominance theorem.

## Prior-art classification

- The weighted-Laplacian pseudoinverse characterization and electrical
  interpretation are standard; for example, Ghosh--Boyd--Saberi define weighted
  effective resistance in exactly that general setting in
  [*Minimizing Effective Resistance of a Graph*](https://web.stanford.edu/~boyd/papers/eff_res.html).
- The path formula here is the series-network/current specialization of that
  standard theory. No novelty is claimed for it, for the quadratic formula, for
  square-root asymptotics, or for Schur/Feshbach elimination.
- Targeted web searches for the exact three-vertex formula, the endpoint-vs-average
  weighted-path source, and Lean/Isabelle/Coq effective-resistance formalizations
  did not locate an exact or stronger compatible formal source.
- Aksoy--Rashid--Hasan--Tahar's 2026
  [Isabelle/HOL network-matrix formalization](https://arxiv.org/abs/2603.25682)
  includes weighted Laplacians, Kron reduction, and power dissipation, but does
  not state this endpoint-vs-average path identity and supplies no reusable Lean
  declaration.
- A source audit of pinned mathlib found no Moore--Penrose inverse or weighted
  effective-resistance definition. `Matrix.NonsingularInverse` explicitly says
  pseudoinverses are not considered, and `SimpleGraph.lapMatrix` is the unweighted
  `degree - adjacency` matrix.
- `Matrix.SchurComplement` supplies invertible block-matrix identities, not a
  reduced-resolvent perturbation theorem that subsumes the requested coefficient.
- No prior formal source for the specific Prime-Flute upstream-memory specialization
  was found. The relevant public provenance remains Mathia
  [PF-080](https://github.com/murillo128/mathia/blob/main/research/prime_flute/findings/PF-080-exact-collar-galerkin-mass-reveals-interscale-memory.md),
  [PF-081](https://github.com/murillo128/mathia/blob/main/research/prime_flute/findings/PF-081-two-scale-feshbach-isolates-the-upstream-memory-coefficient.md),
  and [PF-091](https://github.com/murillo128/mathia/blob/main/research/prime_flute/findings/PF-091-graded-multiscale-burger-window-resolves-an-upstream-memory-ladder.md).

Classification: `PARTIAL_REUSE_PASS` is not needed because no existing generic
formal theorem subsumes a target. The intended result is `PASS_TO_PROOF` with
reuse of low-level mathlib declarations and a small elementary path proof.

## Pinned mathlib reuse inventory

Exact reusable modules/declarations observed against `v4.32.0`:

- matrices/vectors: `Mathlib.Data.Matrix.Mul` (`Matrix.mulVec`, `dotProduct`) and
  `Mathlib.LinearAlgebra.Matrix.Notation` (`![...]`);
- determinants/spectral kernel: `Mathlib.LinearAlgebra.Matrix.Determinant.Basic`
  (`Matrix.det_fin_three`) and `Mathlib.LinearAlgebra.Matrix.ToLinearEquiv`
  (`Matrix.exists_mulVec_eq_zero_iff`);
- characteristic polynomials if useful for presentation:
  `Mathlib.LinearAlgebra.Matrix.Charpoly.Basic` (`Matrix.charpoly`);
- square roots and limits: `Mathlib.Analysis.Real.Sqrt` (`Real.sq_sqrt`,
  `Real.sqrt_sq_eq_abs`, `Real.continuous_sqrt`) plus `Filter.Tendsto`,
  `Continuous.tendsto`, and `nhdsWithin`/right-neighborhood notation;
- asymptotics if later useful: `Mathlib.Analysis.Asymptotics.Defs`
  (`Asymptotics.IsLittleO`), although the frozen target uses `Tendsto`;
- finite sums/indexing: `Fin.sum_univ_succ`, `Fin.last`, `Fin.castSucc`, and
  `Fin.succ`;
- available but unsuitable as the main representation:
  `Mathlib.Combinatorics.SimpleGraph.LapMatrix` (`SimpleGraph.lapMatrix`,
  `lapMatrix_toLinearMap₂'`) because it is unweighted;
- available but not a substitute for Feshbach perturbation:
  `Mathlib.LinearAlgebra.Matrix.SchurComplement`;
- automation: `ring`, `ring_nf`, `field_simp`, `norm_num`, `positivity`,
  `linarith`, and `nlinarith` from `Mathlib.Tactic`.

The probe importing `Mathlib` confirmed these declaration names under the pinned
toolchain. The proof file will narrow imports where practical.

## Proof-phase validation contract

The final checkpoint must compile the focused Lean file under the pinned
toolchain, print axioms for all principal results, contain no `sorry`, `admit`,
or new axioms, pass the `j=2,3,4` regression corollaries and the `j=2`
Feshbach-bookkeeping check, run repository-native lightweight checks, and pass
`git diff --check`. The resulting finite theorem must not claim any surface/PDE
promotion.

## Proof realization and Mathia mapping

`PF091WeightedPathMemory.lean` realizes the frozen surfaces as follows:

- `weightedPath3_characteristic_factor`, `muMinus_isEigenvalue`, and
  `muPlus_isEigenvalue` formalize the PF-081 three-vertex spectrum;
- `h_quadratic_limit` and `muMinus_scaled_limit` make the upstream `-3/8`
  coefficient and the scale `b^2/a` explicit;
- `feshbach_overlap_sq` proves the normalized PF-081 coupling `3/4`;
- `pathPotential_weak_equation`, `pathEnergy_eq_resistance`, and
  `pathResistance_eq` formalize the PF-080/PF-091 finite resistance mechanism
  with the centered source and gauge exposed;
- the `j=2,3,4` corollaries check indexing, while
  `twoScale_upstream_bookkeeping` recovers `-3*b^2/(8*a)` at `j=2`.

These are finite graph statements only. Promotion to the hyperbolic-surface
coefficient in PF-091 remains outside issue #102 and outside the Lean file.
