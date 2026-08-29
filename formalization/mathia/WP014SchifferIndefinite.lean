import Mathlib.Analysis.Matrix.PosDef
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Tactic

/-!
# The exact two-point obstruction for the Mathia WP-014 Schiffer kernel

For the tail domain `x > 2`, Mathia WP-014 specializes the Schiffer kernel of
`V(x) = π cot (π / x)` to an elementary trigonometric expression.  This file
proves that every Gram matrix on two distinct tail points has negative
determinant, and hence is not positive semidefinite.

The diagonal branch in `kernel` is the continuous-extension value from WP-014.
The singular off-diagonal formula is exposed only with a proof that its two
points are distinct.
-/

noncomputable section

open Set

namespace Mathia.WP014

/-- The real tail domain on which the specialized Schiffer formula is used. -/
abbrev TailPoint := {x : ℝ // 2 < x}

/-- The trigonometric displacement in the specialized Schiffer kernel. -/
def delta (x y : TailPoint) : ℝ :=
  Real.pi * (1 / (x : ℝ) - 1 / (y : ℝ))

/-- The specialized kernel away from the diagonal.

The proof argument makes the singular domain explicit; it is not used in the
closed formula itself.
-/
def offDiagonalKernel (x y : TailPoint) (_hxy : x ≠ y) : ℝ :=
  Real.pi ^ 2 / ((x : ℝ) ^ 2 * (y : ℝ) ^ 2) *
    (1 / Real.sin (delta x y) ^ 2 - 1 / delta x y ^ 2)

/-- The specialized Schiffer kernel, with the WP-014 continuous-extension
value on the diagonal and the exact formula off the diagonal. -/
def kernel (x y : TailPoint) : ℝ :=
  if hxy : x = y then Real.pi ^ 2 / (3 * (x : ℝ) ^ 4)
  else offDiagonalKernel x y hxy

/-- The Gram matrix of the kernel on two tail points. -/
def twoPointGram (x y : TailPoint) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![kernel x x, kernel x y; kernel y x, kernel y y]

private theorem cos_lt_quartic {t : ℝ} (ht : 0 < t) :
    Real.cos t < 1 - t ^ 2 / 2 + t ^ 4 / 24 := by
  let f (u : ℝ) : ℝ := 1 - u ^ 2 / 2 + u ^ 4 / 24 - Real.cos u
  have hderiv (u : ℝ) :
      deriv f u = -u + u ^ 3 / 6 + Real.sin u := by
    simp (disch := fun_prop) [f]
    ring
  have hmono : StrictMonoOn f (Ici 0) := by
    apply strictMonoOn_of_deriv_pos (convex_Ici 0) (by fun_prop)
    grind [Real.sin_gt_sub_cube, interior_Ici]
  have h0 : f 0 < f t := hmono (by simp) ht.le ht
  simpa [f] using h0

private theorem sin_lt_quintic {t : ℝ} (ht : 0 < t) :
    Real.sin t < t - t ^ 3 / 6 + t ^ 5 / 120 := by
  let f (u : ℝ) : ℝ := u - u ^ 3 / 6 + u ^ 5 / 120 - Real.sin u
  have hderiv (u : ℝ) :
      deriv f u = 1 - u ^ 2 / 2 + u ^ 4 / 24 - Real.cos u := by
    simp (disch := fun_prop) [f]
    ring
  have hmono : StrictMonoOn f (Ici 0) := by
    apply strictMonoOn_of_deriv_pos (convex_Ici 0) (by fun_prop)
    grind [cos_lt_quartic, interior_Ici]
  have h0 : f 0 < f t := hmono (by simp) ht.le ht
  simpa [f] using h0

/-- Distinct tail points produce a nonzero displacement strictly inside the
open half-period where the scalar inequality applies. -/
theorem abs_delta_mem (x y : TailPoint) (hxy : x ≠ y) :
    0 < |delta x y| ∧ |delta x y| < Real.pi / 2 := by
  have hx₀ : 0 < (x : ℝ) := lt_trans (by norm_num) x.property
  have hy₀ : 0 < (y : ℝ) := lt_trans (by norm_num) y.property
  have hix₀ : 0 < 1 / (x : ℝ) := one_div_pos.mpr hx₀
  have hiy₀ : 0 < 1 / (y : ℝ) := one_div_pos.mpr hy₀
  have hix : 1 / (x : ℝ) < 1 / 2 :=
    one_div_lt_one_div_of_lt (by norm_num) x.property
  have hiy : 1 / (y : ℝ) < 1 / 2 :=
    one_div_lt_one_div_of_lt (by norm_num) y.property
  have hdiff : |1 / (x : ℝ) - 1 / (y : ℝ)| < 1 / 2 := by
    rw [abs_lt]
    constructor <;> linarith
  have hdiff_ne : 1 / (x : ℝ) - 1 / (y : ℝ) ≠ 0 := by
    intro h
    have hinv : (x : ℝ)⁻¹ = (y : ℝ)⁻¹ := by
      simpa [one_div] using sub_eq_zero.mp h
    apply hxy
    exact Subtype.ext (inv_inj.mp hinv)
  rw [delta, abs_mul, abs_of_pos Real.pi_pos]
  constructor
  · exact mul_pos Real.pi_pos (abs_pos.mpr hdiff_ne)
  · calc
      Real.pi * |1 / (x : ℝ) - 1 / (y : ℝ)| < Real.pi * (1 / 2) :=
        mul_lt_mul_of_pos_left hdiff Real.pi_pos
      _ = Real.pi / 2 := by ring

private theorem schifferScalar_gt_one_third_of_pos {t : ℝ}
    (ht₀ : 0 < t) (htπ : t < Real.pi / 2) :
    1 / Real.sin t ^ 2 - 1 / t ^ 2 > 1 / 3 := by
  let q : ℝ := t - t ^ 3 / 6 + t ^ 5 / 120
  let u : ℝ := t ^ 2
  have hsin₀ : 0 < Real.sin t :=
    Real.sin_pos_of_pos_of_lt_pi ht₀ (by nlinarith [Real.pi_pos])
  have hsinq : Real.sin t < q := by
    simpa [q] using sin_lt_quintic ht₀
  have hq₀ : 0 < q := hsin₀.trans hsinq
  have hu₀ : 0 < u := by
    simp only [u]
    positivity
  have ht_two : t < 2 := by nlinarith [Real.pi_lt_four]
  have hu_four : u < 4 := by
    have hsq := mul_self_lt_mul_self ht₀.le ht_two
    norm_num at hsq
    simpa [u, pow_two] using hsq
  have hfirst : u ^ 2 * (u - 37) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos (sq_nonneg u) (by linarith)
  have hpoly : u ^ 3 - 37 * u ^ 2 + 520 * u - 2880 < 0 := by
    calc
      u ^ 3 - 37 * u ^ 2 + 520 * u - 2880 =
          u ^ 2 * (u - 37) + (520 * u - 2880) := by ring
      _ < 0 := by nlinarith
  have hfactor₀ : 0 < u ^ 3 / 14400 := by positivity
  have hq_bound : (3 + u) * q ^ 2 < 3 * u := by
    have hfactor :
        (3 + u) * q ^ 2 - 3 * u =
          u ^ 3 / 14400 * (u ^ 3 - 37 * u ^ 2 + 520 * u - 2880) := by
      simp only [q, u]
      ring
    have hneg := mul_neg_of_pos_of_neg hfactor₀ hpoly
    nlinarith
  have hsq : Real.sin t ^ 2 < q ^ 2 :=
    (sq_lt_sq₀ hsin₀.le hq₀.le).2 hsinq
  have hmain : (3 + u) * Real.sin t ^ 2 < 3 * u :=
    (mul_lt_mul_of_pos_left hsq (by positivity)).trans hq_bound
  have hsin_sq₀ : 0 < Real.sin t ^ 2 := sq_pos_of_pos hsin₀
  have ht_sq₀ : 0 < t ^ 2 := sq_pos_of_pos ht₀
  have hid :
      1 / Real.sin t ^ 2 - 1 / t ^ 2 =
        (t ^ 2 - Real.sin t ^ 2) / (Real.sin t ^ 2 * t ^ 2) := by
    field_simp
  rw [hid]
  apply (lt_div_iff₀ (mul_pos hsin_sq₀ ht_sq₀)).2
  simp only [u] at hmain
  nlinarith

/-- The decisive WP-014 scalar inequality, stated on the full symmetric open
half-period and with the nonzero condition explicit. -/
theorem schifferScalar_gt_one_third {t : ℝ}
    (ht₀ : 0 < |t|) (htπ : |t| < Real.pi / 2) :
    1 / Real.sin t ^ 2 - 1 / t ^ 2 > 1 / 3 := by
  rcases lt_or_gt_of_ne (abs_pos.mp ht₀) with ht | ht
  · have h := schifferScalar_gt_one_third_of_pos (t := -t) (neg_pos.mpr ht)
      (by simpa [abs_of_neg ht] using htπ)
    simpa [Real.sin_neg] using h
  · exact schifferScalar_gt_one_third_of_pos ht
      (by simpa [abs_of_pos ht] using htπ)

private theorem delta_swap (x y : TailPoint) : delta y x = -delta x y := by
  simp only [delta]
  ring

private theorem offDiagonalKernel_swap (x y : TailPoint) (hxy : x ≠ y) :
    offDiagonalKernel y x hxy.symm = offDiagonalKernel x y hxy := by
  unfold offDiagonalKernel
  rw [delta_swap x y, Real.sin_neg]
  ring

private theorem determinant_factor (p x y f : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
    (p ^ 2 / (3 * x ^ 4)) * (p ^ 2 / (3 * y ^ 4)) -
        (p ^ 2 / (x ^ 2 * y ^ 2) * f) ^ 2 =
      p ^ 4 / (x ^ 4 * y ^ 4) * (1 / 9 - f ^ 2) := by
  field_simp [hx, hy]
  all_goals ring

/-- Exact determinant reduction for the two-point Schiffer Gram matrix. -/
theorem det_twoPointGram (x y : TailPoint) (hxy : x ≠ y) :
    (twoPointGram x y).det =
      Real.pi ^ 4 / ((x : ℝ) ^ 4 * (y : ℝ) ^ 4) *
        (1 / 9 -
          (1 / Real.sin (delta x y) ^ 2 - 1 / delta x y ^ 2) ^ 2) := by
  rw [Matrix.det_fin_two]
  change kernel x x * kernel y y - kernel x y * kernel y x = _
  have hxx : kernel x x = Real.pi ^ 2 / (3 * (x : ℝ) ^ 4) := by
    simp [kernel]
  have hyy : kernel y y = Real.pi ^ 2 / (3 * (y : ℝ) ^ 4) := by
    simp [kernel]
  have hxy_kernel : kernel x y = offDiagonalKernel x y hxy := by
    simp [kernel, hxy]
  have hyx_kernel : kernel y x = offDiagonalKernel y x hxy.symm := by
    simp [kernel, hxy.symm]
  rw [hxx, hyy, hxy_kernel, hyx_kernel]
  rw [offDiagonalKernel_swap x y hxy]
  unfold offDiagonalKernel
  convert determinant_factor Real.pi (x : ℝ) (y : ℝ)
    (1 / Real.sin (delta x y) ^ 2 - 1 / delta x y ^ 2)
    (ne_of_gt (lt_trans (by norm_num) x.property))
    (ne_of_gt (lt_trans (by norm_num) y.property)) using 1
  all_goals ring

/-- Every pair of distinct tail points has strictly negative Gram determinant. -/
theorem det_twoPointGram_neg (x y : TailPoint) (hxy : x ≠ y) :
    (twoPointGram x y).det < 0 := by
  rw [det_twoPointGram x y hxy]
  let f : ℝ := 1 / Real.sin (delta x y) ^ 2 - 1 / delta x y ^ 2
  have hf : 1 / 3 < f := by
    simpa only [f] using schifferScalar_gt_one_third (abs_delta_mem x y hxy).1
      (abs_delta_mem x y hxy).2
  have hf₀ : 0 ≤ f := le_trans (by norm_num) hf.le
  have hsq : (1 / 3 : ℝ) ^ 2 < f ^ 2 :=
    (sq_lt_sq₀ (by norm_num) hf₀).2 hf
  have hbracket : (1 / 9 : ℝ) - f ^ 2 < 0 := by
    norm_num at hsq ⊢
    linarith
  have hcoefficient :
      0 < Real.pi ^ 4 / ((x : ℝ) ^ 4 * (y : ℝ) ^ 4) := by
    exact div_pos (pow_pos Real.pi_pos 4)
      (mul_pos (pow_pos (lt_trans (by norm_num) x.property) 4)
        (pow_pos (lt_trans (by norm_num) y.property) 4))
  exact mul_neg_of_pos_of_neg hcoefficient hbracket

/-- Negative determinant rules out mathlib's positive-semidefinite matrix
predicate without reimplementing generic PSD theory. -/
theorem twoPointGram_not_posSemidef (x y : TailPoint) (hxy : x ≠ y) :
    ¬ (twoPointGram x y).PosSemidef := by
  intro hpsd
  exact (not_lt_of_ge hpsd.det_nonneg) (det_twoPointGram_neg x y hxy)

end Mathia.WP014
