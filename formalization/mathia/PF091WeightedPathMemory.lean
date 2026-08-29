import Mathlib

/-!
# PF-091 finite weighted-path memory

This file formalizes only the finite-dimensional mechanism audited in
`PF091WeightedPathMemoryGate.md`. It makes no claim about hyperbolic surfaces,
Burger's approximation, collar PDE estimates, or prime-pattern realization.
-/

namespace Mathia.PF091

open Filter Matrix
open scoped BigOperators Matrix Topology

noncomputable section

/-! ## The exact three-vertex weighted path -/

/-- The weighted Laplacian of the three-vertex path with edge weights `a` and `b`. -/
def weightedPath3 (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a, -a, 0], ![-a, a + b, -b], ![0, -b, b]]

/-- The smaller algebraic root of the nonzero spectral quadratic. -/
def muMinus (a b : ℝ) : ℝ := a + b - Real.sqrt (a ^ 2 - a * b + b ^ 2)

/-- The larger algebraic root of the nonzero spectral quadratic. -/
def muPlus (a b : ℝ) : ℝ := a + b + Real.sqrt (a ^ 2 - a * b + b ^ 2)

/-- Elementary matrix-vector formulation of a real eigenvalue. -/
def IsEigenvalue (M : Matrix (Fin 3) (Fin 3) ℝ) (mu : ℝ) : Prop :=
  ∃ v : Fin 3 → ℝ, v ≠ 0 ∧ M *ᵥ v = mu • v

/-- Exact characteristic factorization of the three-vertex weighted path. -/
theorem weightedPath3_characteristic_factor (a b lambda : ℝ) :
    (lambda • (1 : Matrix (Fin 3) (Fin 3) ℝ) - weightedPath3 a b).det =
      lambda * (lambda ^ 2 - 2 * (a + b) * lambda + 3 * a * b) := by
  simp [weightedPath3, Matrix.det_fin_three]
  ring

private theorem radicand_pos {a b : ℝ} (ha : 0 < a) :
    0 < a ^ 2 - a * b + b ^ 2 := by
  nlinarith [sq_nonneg (a - b)]

private theorem root_isEigenvalue {a b mu : ℝ}
    (hroot : mu ^ 2 - 2 * (a + b) * mu + 3 * a * b = 0) :
    IsEigenvalue (weightedPath3 a b) mu := by
  have hdet : (mu • (1 : Matrix (Fin 3) (Fin 3) ℝ) - weightedPath3 a b).det = 0 := by
    rw [weightedPath3_characteristic_factor, hroot, mul_zero]
  obtain ⟨v, hv, hkernel⟩ := Matrix.exists_mulVec_eq_zero_iff.mpr hdet
  refine ⟨v, hv, ?_⟩
  have hsub : mu • v - weightedPath3 a b *ᵥ v = 0 := by
    simpa [Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec] using hkernel
  exact (sub_eq_zero.mp hsub).symm

/-- The smaller displayed value is an eigenvalue for positive weights. -/
theorem muMinus_isEigenvalue {a b : ℝ} (ha : 0 < a) (_hb : 0 < b) :
    IsEigenvalue (weightedPath3 a b) (muMinus a b) := by
  apply root_isEigenvalue
  have hs := Real.sq_sqrt (radicand_pos (b := b) ha).le
  simp only [muMinus]
  nlinarith

/-- The larger displayed value is an eigenvalue for positive weights. -/
theorem muPlus_isEigenvalue {a b : ℝ} (ha : 0 < a) (_hb : 0 < b) :
    IsEigenvalue (weightedPath3 a b) (muPlus a b) := by
  apply root_isEigenvalue
  have hs := Real.sq_sqrt (radicand_pos (b := b) ha).le
  simp only [muPlus]
  nlinarith

/-- The constant vector supplies the zero eigenvalue. -/
theorem zero_isEigenvalue (a b : ℝ) : IsEigenvalue (weightedPath3 a b) 0 := by
  refine ⟨![1, 1, 1], by simp, ?_⟩
  ext i
  fin_cases i <;>
    norm_num [weightedPath3, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- The smaller displayed eigenvalue is positive for positive edge weights. -/
theorem muMinus_pos {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : 0 < muMinus a b := by
  have hr := radicand_pos (b := b) ha
  have hs := Real.sq_sqrt hr.le
  have hs0 := Real.sqrt_nonneg (a ^ 2 - a * b + b ^ 2)
  have hab : 0 < a + b := add_pos ha hb
  have hlt : a ^ 2 - a * b + b ^ 2 < (a + b) ^ 2 := by nlinarith
  simp only [muMinus]
  nlinarith

/-- The minus branch is strictly below the plus branch for positive weights. -/
theorem muMinus_lt_muPlus {a b : ℝ} (ha : 0 < a) (_hb : 0 < b) :
    muMinus a b < muPlus a b := by
  have hs : 0 < Real.sqrt (a ^ 2 - a * b + b ^ 2) :=
    Real.sqrt_pos.2 (radicand_pos (b := b) ha)
  simp only [muMinus, muPlus]
  linarith

/-! ## The dimensionless `-3/8` coefficient -/

/-- Dimensionless smaller-root profile. -/
def h (r : ℝ) : ℝ := 1 + r - Real.sqrt (1 - r + r ^ 2)

private theorem h_quotient_eq (r : ℝ) (hr : 0 < r) :
    (h r - (3 / 2) * r) / r ^ 2 =
      -3 / (2 * (1 + r + Real.sqrt (1 - r + r ^ 2)) *
        (1 - r + Real.sqrt (1 - r + r ^ 2))) := by
  have hrad : 0 < 1 - r + r ^ 2 := by nlinarith [sq_nonneg (r - 1 / 2)]
  have hs := Real.sq_sqrt hrad.le
  have hs0 := Real.sqrt_nonneg (1 - r + r ^ 2)
  have hd₁ : 0 < 1 + r + Real.sqrt (1 - r + r ^ 2) := by positivity
  have hd₂ : 0 < 1 - r + Real.sqrt (1 - r + r ^ 2) := by
    by_cases hle : r ≤ 1
    · nlinarith [Real.sqrt_pos.2 hrad]
    · nlinarith [sq_nonneg (Real.sqrt (1 - r + r ^ 2) + (r - 1))]
  simp only [h]
  field_simp [ne_of_gt hr, ne_of_gt hd₁, ne_of_gt hd₂]
  nlinarith

/-- The right-hand quadratic coefficient of the dimensionless smaller root is `-3/8`. -/
theorem h_quadratic_limit :
    Tendsto (fun r => (h r - (3 / 2) * r) / r ^ 2)
      (𝓝[>] (0 : ℝ)) (𝓝 (-3 / 8 : ℝ)) := by
  let F : ℝ → ℝ := fun r =>
    -3 / (2 * (1 + r + Real.sqrt (1 - r + r ^ 2)) *
      (1 - r + Real.sqrt (1 - r + r ^ 2)))
  have hF : ContinuousAt F 0 := by
    dsimp [F]
    fun_prop (disch := norm_num)
  have ht : Tendsto F (𝓝[>] (0 : ℝ)) (𝓝 (-3 / 8 : ℝ)) := by
    have ht0 : Tendsto F (𝓝[Set.Ioi (0 : ℝ)] 0) (𝓝 (F 0)) :=
      hF.continuousWithinAt
    convert ht0 using 1
    norm_num [F]
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with r hr
  exact (h_quotient_eq r hr).symm

/-- Scaling identity `muMinus(a,b) = a*h(b/a)` for `a > 0`. -/
theorem muMinus_eq_scaled (a b : ℝ) (ha : 0 < a) : muMinus a b = a * h (b / a) := by
  have ha0 : a ≠ 0 := ne_of_gt ha
  have hrad : a ^ 2 - a * b + b ^ 2 = a ^ 2 * (1 - b / a + (b / a) ^ 2) := by
    field_simp
  rw [muMinus, h, hrad, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq ha.le]
  field_simp

/-- The scaled coefficient explicitly normalizes by `b^2/a`. -/
theorem muMinus_scaled_limit (a : ℝ) (ha : 0 < a) :
    Tendsto
      (fun b => (muMinus a b - (3 / 2) * b) / (b ^ 2 / a))
      (𝓝[>] (0 : ℝ)) (𝓝 (-3 / 8 : ℝ)) := by
  have hscale : Tendsto (fun b : ℝ => b / a) (𝓝[>] (0 : ℝ)) (𝓝[>] (0 : ℝ)) := by
    apply tendsto_nhdsWithin_iff.mpr
    constructor
    · have hc : Tendsto (fun b : ℝ => b / a) (𝓝[Set.Ioi (0 : ℝ)] 0)
          (𝓝 ((0 : ℝ) / a)) := (continuousAt_id.div_const a).continuousWithinAt
      simpa using hc
    · filter_upwards [self_mem_nhdsWithin] with b hb
      change 0 < b at hb
      exact div_pos hb ha
  have ht := h_quadratic_limit.comp hscale
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with b hb
  change 0 < b at hb
  simp only [Function.comp_apply]
  rw [muMinus_eq_scaled a b ha]
  rw [div_pow]
  field_simp [ne_of_gt ha, ne_of_gt hb]

/-! ## The normalized Feshbach overlap -/

/-- Normalized stronger internal mode `(1,-1,0)/sqrt(2)`. -/
def feshbachU : Fin 3 → ℝ := ![1 / Real.sqrt 2, -1 / Real.sqrt 2, 0]

/-- Normalized weak-neck mode `(1,1,-2)/sqrt(6)`. -/
def feshbachPsi : Fin 3 → ℝ :=
  ![1 / Real.sqrt 6, 1 / Real.sqrt 6, -2 / Real.sqrt 6]

/-- Rank-one matrix of the second edge, independent of its orientation. -/
def weakEdgeMatrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 1, -1], ![0, -1, 1]]

/-- The stronger mode is normalized in the real Euclidean dot product. -/
theorem feshbachU_norm_sq : feshbachU ⬝ᵥ feshbachU = 1 := by
  have h2 : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  have hs2 := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  simp [feshbachU, dotProduct, Fin.sum_univ_succ]
  field_simp [h2]
  nlinarith

/-- The weak-neck mode is normalized in the real Euclidean dot product. -/
theorem feshbachPsi_norm_sq : feshbachPsi ⬝ᵥ feshbachPsi = 1 := by
  have h6 : Real.sqrt 6 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  have hs6 := Real.sq_sqrt (show (0 : ℝ) ≤ 6 by norm_num)
  simp [feshbachPsi, dotProduct, Fin.sum_univ_succ]
  field_simp [h6]
  nlinarith

/-- The two normalized graph modes are orthogonal. -/
theorem feshbachU_orthogonal_psi : feshbachU ⬝ᵥ feshbachPsi = 0 := by
  have h2 : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  have h6 : Real.sqrt 6 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  simp [feshbachU, feshbachPsi, dotProduct, Fin.sum_univ_succ]
  field_simp [h2, h6]
  ring

/-- Exact squared Feshbach coupling coefficient. -/
theorem feshbach_overlap_sq :
    |feshbachU ⬝ᵥ (weakEdgeMatrix *ᵥ feshbachPsi)| ^ 2 = 3 / 4 := by
  have h2 : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  have h6 : Real.sqrt 6 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  have hs2 := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  have hs6 := Real.sq_sqrt (show (0 : ℝ) ≤ 6 by norm_num)
  rw [sq_abs]
  simp [feshbachU, feshbachPsi, weakEdgeMatrix, Matrix.mulVec, dotProduct,
    Fin.sum_univ_succ]
  field_simp [h2, h6]
  nlinarith

/-! ## Gauge-explicit weighted-path resistance -/

/-- Edge current for the endpoint-minus-average source on a path with `n + 1` vertices. -/
def pathCurrent (n : ℕ) (e : Fin n) : ℝ := (e.val + 1 : ℝ) / (n + 1 : ℝ)

/-- Potential with zero value at the left endpoint and prescribed current increments. -/
def pathPotential {n : ℕ} (w : Fin n → ℝ) : Fin (n + 1) → ℝ :=
  Fin.partialSum fun e => pathCurrent n e / w e

/-- Weighted path Dirichlet energy. -/
def pathEnergy {n : ℕ} (w : Fin n → ℝ) (x : Fin (n + 1) → ℝ) : ℝ :=
  ∑ e, w e * (x e.succ - x e.castSucc) ^ 2

/-- Arithmetic mean of a vector on the `n + 1` path vertices. -/
def pathAverage {n : ℕ} (x : Fin (n + 1) → ℝ) : ℝ :=
  (∑ i, x i) / (n + 1 : ℝ)

/-- Every potential increment is the corresponding current divided by conductance. -/
theorem pathPotential_edge_increment {n : ℕ} {w : Fin n → ℝ} (e : Fin n) :
    pathPotential w e.succ - pathPotential w e.castSucc = pathCurrent n e / w e := by
  rw [pathPotential, Fin.partialSum_succ]
  ring

private theorem pathCurrent_sum_by_parts (n : ℕ) (y : Fin (n + 1) → ℝ) :
    (∑ e, pathCurrent n e * (y e.succ - y e.castSucc)) =
      y (Fin.last n) - pathAverage y := by
  have hj : (n + 1 : ℝ) ≠ 0 := by positivity
  have hleft :
      (∑ e : Fin n, ((e.val + 1 : ℝ) / (n + 1 : ℝ)) *
        (y e.succ - y e.castSucc)) =
        (∑ e : Fin n, (e.val + 1 : ℝ) * (y e.succ - y e.castSucc)) /
          (n + 1 : ℝ) := by
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro e _
    ring
  have hsucc :
      (∑ e : Fin n, (e.val + 1 : ℝ) * y e.succ) =
        ∑ i : Fin (n + 1), (i.val : ℝ) * y i := by
    rw [Fin.sum_univ_succ]
    simp
  have hcast :
      (∑ i : Fin (n + 1), (i.val + 1 : ℝ) * y i) =
        (∑ e : Fin n, (e.val + 1 : ℝ) * y e.castSucc) +
          (n + 1 : ℝ) * y (Fin.last n) := by
    rw [Fin.sum_univ_castSucc]
    simp
  have hplus :
      (∑ i : Fin (n + 1), (i.val + 1 : ℝ) * y i) =
        (∑ i : Fin (n + 1), (i.val : ℝ) * y i) + ∑ i, y i := by
    simp_rw [add_mul]
    rw [Finset.sum_add_distrib]
    simp
  have hnum :
      (∑ e : Fin n, (e.val + 1 : ℝ) * (y e.succ - y e.castSucc)) =
        (n + 1 : ℝ) * y (Fin.last n) - ∑ i, y i := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, hsucc]
    linarith
  simp only [pathCurrent, pathAverage]
  rw [hleft, hnum]
  field_simp [hj]

/-- Weak Laplace equation for the zero-mean endpoint source. -/
theorem pathPotential_weak_equation {n : ℕ} {w : Fin n → ℝ}
    (hpos : ∀ e, 0 < w e) (y : Fin (n + 1) → ℝ) :
    (∑ e, w e *
      (pathPotential w e.succ - pathPotential w e.castSucc) *
      (y e.succ - y e.castSucc)) =
      y (Fin.last n) - pathAverage y := by
  rw [← pathCurrent_sum_by_parts n y]
  apply Finset.sum_congr rfl
  intro e _
  rw [pathPotential_edge_increment]
  field_simp [ne_of_gt (hpos e)]

/-- Dirichlet energy equals the endpoint potential minus its vertex average. -/
theorem pathEnergy_eq_resistance {n : ℕ} {w : Fin n → ℝ}
    (hpos : ∀ e, 0 < w e) :
    pathEnergy w (pathPotential w) =
      pathPotential w (Fin.last n) - pathAverage (pathPotential w) := by
  simpa [pathEnergy, pow_two, mul_assoc] using
    pathPotential_weak_equation hpos (pathPotential w)

/-- Exact effective-resistance identity for the endpoint-minus-average source. -/
theorem pathResistance_eq {n : ℕ} {w : Fin n → ℝ}
    (hpos : ∀ e, 0 < w e) :
    pathPotential w (Fin.last n) - pathAverage (pathPotential w) =
      (1 / (n + 1 : ℝ) ^ 2) * ∑ e, (e.val + 1 : ℝ) ^ 2 / w e := by
  rw [← pathEnergy_eq_resistance hpos]
  simp only [pathEnergy, pathPotential_edge_increment]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro e _
  have hj : (n + 1 : ℝ) ≠ 0 := by positivity
  have hw : w e ≠ 0 := ne_of_gt (hpos e)
  simp only [pathCurrent]
  field_simp [hj, hw]

/-! ### Deterministic indexing regressions -/

/-- The two-vertex (`j=2`) specialization is `1/(4*w₁)`. -/
theorem pathResistance_two_vertices (w₁ : ℝ) (hw₁ : 0 < w₁) :
    pathPotential ![w₁] (Fin.last 1) - pathAverage (pathPotential ![w₁]) =
      1 / (4 * w₁) := by
  have hpos : ∀ e : Fin 1, 0 < ![w₁] e := by
    intro e
    fin_cases e
    exact hw₁
  rw [pathResistance_eq hpos]
  norm_num [Fin.sum_univ_succ]
  field_simp [ne_of_gt hw₁]

/-- The three-vertex (`j=3`) specialization has coefficients `1,4` over `9`. -/
theorem pathResistance_three_vertices (w₁ w₂ : ℝ) (hw₁ : 0 < w₁) (hw₂ : 0 < w₂) :
    pathPotential ![w₁, w₂] (Fin.last 2) - pathAverage (pathPotential ![w₁, w₂]) =
      (1 / w₁ + 4 / w₂) / 9 := by
  have hpos : ∀ e : Fin 2, 0 < ![w₁, w₂] e := by
    intro e
    fin_cases e
    · exact hw₁
    · exact hw₂
  rw [pathResistance_eq hpos]
  norm_num [Fin.sum_univ_succ]
  field_simp [ne_of_gt hw₁, ne_of_gt hw₂]

/-- The four-vertex (`j=4`) specialization has coefficients `1,4,9` over `16`. -/
theorem pathResistance_four_vertices (w₁ w₂ w₃ : ℝ)
    (hw₁ : 0 < w₁) (hw₂ : 0 < w₂) (hw₃ : 0 < w₃) :
    pathPotential ![w₁, w₂, w₃] (Fin.last 3) -
        pathAverage (pathPotential ![w₁, w₂, w₃]) =
      (1 / w₁ + 4 / w₂ + 9 / w₃) / 16 := by
  have hpos : ∀ e : Fin 3, 0 < ![w₁, w₂, w₃] e := by
    intro e
    fin_cases e
    · exact hw₁
    · exact hw₂
    · exact hw₃
  rw [pathResistance_eq hpos]
  norm_num [Fin.sum_univ_succ]
  field_simp [ne_of_gt hw₁, ne_of_gt hw₂, ne_of_gt hw₃]
  ring

/-- The `j=2` resistance and Feshbach prefactor reproduce `-3*b^2/(8*a)`. -/
theorem twoScale_upstream_bookkeeping (a b : ℝ) (ha : 0 < a) :
    -(3 / 2 : ℝ) * b ^ 2 * (1 / (4 * a)) = -(3 / 8 : ℝ) * (b ^ 2 / a) := by
  field_simp [ne_of_gt ha]
  ring

end

end Mathia.PF091

#print axioms Mathia.PF091.weightedPath3_characteristic_factor
#print axioms Mathia.PF091.muMinus_isEigenvalue
#print axioms Mathia.PF091.muPlus_isEigenvalue
#print axioms Mathia.PF091.h_quadratic_limit
#print axioms Mathia.PF091.muMinus_scaled_limit
#print axioms Mathia.PF091.feshbach_overlap_sq
#print axioms Mathia.PF091.pathPotential_weak_equation
#print axioms Mathia.PF091.pathResistance_eq
#print axioms Mathia.PF091.pathEnergy_eq_resistance
