import Mathlib

/-!
# Mathia WI-011: the finite trace--energy envelope

This file formalizes the prior-art scalar envelope used by Mathia WI-011 and the exact
`m = 438` splice arithmetic.  It intentionally starts from a finite family of nonnegative
eigenvalues with prescribed trace; it does not introduce a matrix-functional-calculus layer.
-/

noncomputable section

open scoped BigOperators

namespace Mathia.WI011

/-- The scalar spectral-defect profile from the stability bridge. -/
def psi (t : ℝ) : ℝ :=
  if t ≤ 2 then (t - 1) ^ 2 else 2 * t - 3

/-- Squared distance of the eigenvalue family from the unit spectrum. -/
def energy {m : ℕ} (lambda : Fin m → ℝ) : ℝ :=
  ∑ i, (lambda i - 1) ^ 2

/-- Sum of the scalar spectral defects. -/
def defect {m : ℕ} (lambda : Fin m → ℝ) : ℝ :=
  ∑ i, psi (lambda i)

/-- Centered eigenvalue coordinates. -/
def centered {m : ℕ} (lambda : Fin m → ℝ) (i : Fin m) : ℝ :=
  lambda i - 1

/-- Coordinates on which `psi` uses its linear branch. -/
def largeSet {m : ℕ} (lambda : Fin m → ℝ) : Finset (Fin m) :=
  Finset.univ.filter fun i => 1 < centered lambda i

/-- The sharp finite trace--energy lower envelope. -/
def phi (m : ℕ) (E : ℝ) : ℝ :=
  if E ≤ (m : ℝ) / ((m : ℝ) - 1) then E
  else 2 * Real.sqrt ((((m : ℝ) - 1) / (m : ℝ)) * E) - 1 + E / (m : ℝ)

private lemma cast_m_pos {m : ℕ} (hm : 2 ≤ m) : (0 : ℝ) < m := by
  exact_mod_cast (lt_of_lt_of_le (by decide : 0 < 2) hm)

private lemma cast_m_sub_one_pos {m : ℕ} (hm : 2 ≤ m) : (0 : ℝ) < (m : ℝ) - 1 := by
  have hm' : (2 : ℝ) ≤ m := by exact_mod_cast hm
  linarith

private lemma envelopeCoefficient_pos {m : ℕ} (hm : 2 ≤ m) :
    0 < ((m : ℝ) - 1) / (m : ℝ) := by
  positivity [cast_m_pos hm, cast_m_sub_one_pos hm]

private lemma coefficient_add_recip {m : ℕ} (hm : 2 ≤ m) :
    ((m : ℝ) - 1) / (m : ℝ) + 1 / (m : ℝ) = 1 := by
  field_simp [ne_of_gt (cast_m_pos hm)]
  ring

private lemma coefficient_mul_threshold {m : ℕ} (hm : 2 ≤ m) :
    ((m : ℝ) - 1) / (m : ℝ) * ((m : ℝ) / ((m : ℝ) - 1)) = 1 := by
  field_simp [ne_of_gt (cast_m_pos hm), ne_of_gt (cast_m_sub_one_pos hm)]

private lemma threshold_nonneg {m : ℕ} (hm : 2 ≤ m) :
    0 ≤ (m : ℝ) / ((m : ℝ) - 1) := by
  positivity [cast_m_pos hm, cast_m_sub_one_pos hm]

private def phiTail (m : ℕ) (E : ℝ) : ℝ :=
  2 * Real.sqrt ((((m : ℝ) - 1) / (m : ℝ)) * E) - 1 + E / (m : ℝ)

private lemma phiTail_at_threshold {m : ℕ} (hm : 2 ≤ m) :
    phiTail m ((m : ℝ) / ((m : ℝ) - 1)) = (m : ℝ) / ((m : ℝ) - 1) := by
  have hc := coefficient_mul_threshold hm
  rw [phiTail, hc, Real.sqrt_one]
  have hsum := coefficient_add_recip hm
  have hm0 := cast_m_pos hm
  have hm1 := cast_m_sub_one_pos hm
  field_simp [ne_of_gt hm0, ne_of_gt hm1] at hsum ⊢
  ring

private lemma phiTail_mono {m : ℕ} (hm : 2 ≤ m) {x y : ℝ} (hxy : x ≤ y) :
    phiTail m x ≤ phiTail m y := by
  unfold phiTail
  have hc := (envelopeCoefficient_pos hm).le
  have hsqrt :
      Real.sqrt ((((m : ℝ) - 1) / (m : ℝ)) * x) ≤
        Real.sqrt ((((m : ℝ) - 1) / (m : ℝ)) * y) :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hxy hc)
  have hm0 := cast_m_pos hm
  have hdiv : x / (m : ℝ) ≤ y / (m : ℝ) := div_le_div_of_nonneg_right hxy hm0.le
  linarith

private lemma phiTail_increment_le {m : ℕ} (hm : 2 ≤ m) {x y : ℝ}
    (hx : (m : ℝ) / ((m : ℝ) - 1) ≤ x) (hxy : x ≤ y) :
    phiTail m y ≤ phiTail m x + (y - x) := by
  let c : ℝ := ((m : ℝ) - 1) / (m : ℝ)
  let sx : ℝ := Real.sqrt (c * x)
  let sy : ℝ := Real.sqrt (c * y)
  have hc : 0 < c := by simpa [c] using envelopeCoefficient_pos hm
  have hx0 : 0 ≤ x := (threshold_nonneg hm).trans hx
  have hy0 : 0 ≤ y := hx0.trans hxy
  have hcx0 : 0 ≤ c * x := mul_nonneg hc.le hx0
  have hcy0 : 0 ≤ c * y := mul_nonneg hc.le hy0
  have hsx_sq : sx ^ 2 = c * x := by
    simpa [sx] using Real.sq_sqrt hcx0
  have hsy_sq : sy ^ 2 = c * y := by
    simpa [sy] using Real.sq_sqrt hcy0
  have hsxy : sx ≤ sy := by
    simpa [sx, sy] using Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hxy hc.le)
  have hsx1 : 1 ≤ sx := by
    rw [← Real.sqrt_one]
    apply Real.sqrt_le_sqrt
    calc
      1 = c * ((m : ℝ) / ((m : ℝ) - 1)) := by
        simpa [c] using (coefficient_mul_threshold hm).symm
      _ ≤ c * x := mul_le_mul_of_nonneg_left hx hc.le
  have hfactor : 0 ≤ (sy - sx) * (sy + sx - 2) :=
    mul_nonneg (sub_nonneg.mpr hsxy) (by linarith)
  have hroot_increment : 2 * (sy - sx) ≤ c * (y - x) := by
    nlinarith [hfactor]
  have hcoeff : c + (m : ℝ)⁻¹ = 1 := by
    simpa [c, div_eq_mul_inv] using coefficient_add_recip hm
  have hm0 := cast_m_pos hm
  change 2 * sy - 1 + y / (m : ℝ) ≤
    (2 * sx - 1 + x / (m : ℝ)) + (y - x)
  rw [div_eq_mul_inv, div_eq_mul_inv]
  nlinarith

theorem phi_monoOn_nonneg {m : ℕ} (hm : 2 ≤ m) :
    MonotoneOn (phi m) (Set.Ici 0) := by
  intro x hx y hy hxy
  by_cases hxBranch : x ≤ (m : ℝ) / ((m : ℝ) - 1)
  · by_cases hyBranch : y ≤ (m : ℝ) / ((m : ℝ) - 1)
    · simp [phi, hxBranch, hyBranch, hxy]
    · have hthreshold_y : (m : ℝ) / ((m : ℝ) - 1) ≤ y := le_of_not_ge hyBranch
      have htail := phiTail_mono hm hthreshold_y
      simp only [phi, hxBranch, hyBranch, ↓reduceIte]
      change x ≤ phiTail m y
      rw [phiTail_at_threshold hm] at htail
      exact hxBranch.trans htail
  · have hxTail : (m : ℝ) / ((m : ℝ) - 1) < x := lt_of_not_ge hxBranch
    have hyBranch : ¬y ≤ (m : ℝ) / ((m : ℝ) - 1) := not_le.mpr (hxTail.trans_le hxy)
    simp only [phi, hxBranch, hyBranch, ↓reduceIte]
    change phiTail m x ≤ phiTail m y
    exact phiTail_mono hm hxy

theorem phi_increment_le {m : ℕ} (hm : 2 ≤ m) {x y : ℝ}
    (_hx : 0 ≤ x) (hxy : x ≤ y) :
    phi m y ≤ phi m x + (y - x) := by
  by_cases hxBranch : x ≤ (m : ℝ) / ((m : ℝ) - 1)
  · by_cases hyBranch : y ≤ (m : ℝ) / ((m : ℝ) - 1)
    · simp [phi, hxBranch, hyBranch]
    · have hthreshold_y : (m : ℝ) / ((m : ℝ) - 1) ≤ y := le_of_not_ge hyBranch
      have htail := phiTail_increment_le hm (le_refl _) hthreshold_y
      simp only [phi, hxBranch, hyBranch, ↓reduceIte]
      change phiTail m y ≤ x + (y - x)
      rw [phiTail_at_threshold hm] at htail
      linarith
  · have hxTail : (m : ℝ) / ((m : ℝ) - 1) ≤ x := le_of_not_ge hxBranch
    have hyBranch : ¬y ≤ (m : ℝ) / ((m : ℝ) - 1) :=
      not_le.mpr ((lt_of_not_ge hxBranch).trans_le hxy)
    simp only [phi, hxBranch, hyBranch, ↓reduceIte]
    change phiTail m y ≤ phiTail m x + (y - x)
    exact phiTail_increment_le hm hxTail hxy

private lemma phi_le_self {m : ℕ} (hm : 2 ≤ m) {E : ℝ} (hE : 0 ≤ E) :
    phi m E ≤ E := by
  by_cases hBranch : E ≤ (m : ℝ) / ((m : ℝ) - 1)
  · simp [phi, hBranch]
  · rw [phi, if_neg hBranch]
    let c : ℝ := ((m : ℝ) - 1) / (m : ℝ)
    let s : ℝ := Real.sqrt (c * E)
    have hc : 0 ≤ c := by simpa [c] using (envelopeCoefficient_pos hm).le
    have hs_sq : s ^ 2 = c * E := by
      simpa [s] using Real.sq_sqrt (mul_nonneg hc hE)
    have hcoeff : c + (m : ℝ)⁻¹ = 1 := by
      simpa [c, div_eq_mul_inv] using coefficient_add_recip hm
    have hs_nonneg : 0 ≤ s := by positivity
    change 2 * s - 1 + E / (m : ℝ) ≤ E
    rw [div_eq_mul_inv]
    nlinarith [sq_nonneg (s - 1)]

private lemma phi_le_of_one_large {m : ℕ} (hm : 2 ≤ m) {E D r : ℝ}
    (hE : 0 ≤ E) (hr : 1 ≤ r)
    (hcauchy : (m : ℝ) * r ^ 2 ≤ ((m : ℝ) - 1) * E)
    (hD : D = E + 2 * r - 1 - r ^ 2) :
    phi m E ≤ D := by
  have hm0 := cast_m_pos hm
  have hm1 := cast_m_sub_one_pos hm
  have hr_sq : 1 ≤ r ^ 2 := by nlinarith [sq_nonneg (r - 1)]
  have hthreshold : (m : ℝ) / ((m : ℝ) - 1) ≤ E := by
    rw [div_le_iff₀ hm1]
    nlinarith
  by_cases hBranch : E ≤ (m : ℝ) / ((m : ℝ) - 1)
  · have hEeq := le_antisymm hBranch hthreshold
    have hr_eq : r = 1 := by
      have hcross : ((m : ℝ) - 1) * E = (m : ℝ) := by
        rw [hEeq]
        field_simp [ne_of_gt hm1]
      nlinarith [sq_nonneg (r - 1)]
    rw [phi, if_pos hBranch, hD, hr_eq]
    ring_nf
    exact le_rfl
  · let c : ℝ := ((m : ℝ) - 1) / (m : ℝ)
    let s : ℝ := Real.sqrt (c * E)
    have hc : 0 ≤ c := by simpa [c] using (envelopeCoefficient_pos hm).le
    have hs_sq : s ^ 2 = c * E := by
      simpa [s] using Real.sq_sqrt (mul_nonneg hc hE)
    have hs_nonneg : 0 ≤ s := by positivity
    have hrsq : r ^ 2 ≤ s ^ 2 := by
      rw [hs_sq]
      dsimp [c]
      rw [div_mul_eq_mul_div, le_div_iff₀ hm0]
      simpa [mul_comm] using hcauchy
    have hrs : r ≤ s := by nlinarith
    have hfactor : 0 ≤ (s - r) * (s + r - 2) :=
      mul_nonneg (sub_nonneg.mpr hrs) (by linarith)
    have hcoeff : c + (m : ℝ)⁻¹ = 1 := by
      simpa [c, div_eq_mul_inv] using coefficient_add_recip hm
    rw [phi, if_neg hBranch, hD]
    change 2 * s - 1 + E / (m : ℝ) ≤ E + 2 * r - 1 - r ^ 2
    rw [div_eq_mul_inv]
    nlinarith

theorem energy_nonneg {m : ℕ} (lambda : Fin m → ℝ) : 0 ≤ energy lambda := by
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- Exact correction made by the linear branch on the large coordinates. -/
theorem defect_eq_energy_add_large_correction {m : ℕ} (lambda : Fin m → ℝ) :
    defect lambda = energy lambda
      + 2 * (∑ i ∈ largeSet lambda, centered lambda i)
      - (largeSet lambda).card
      - ∑ i ∈ largeSet lambda, centered lambda i ^ 2 := by
  classical
  have hpoint (i : Fin m) :
      psi (lambda i) = centered lambda i ^ 2
        + if i ∈ largeSet lambda then
            2 * centered lambda i - 1 - centered lambda i ^ 2
          else 0 := by
    by_cases hi : 1 < centered lambda i
    · have hi' : 1 < lambda i - 1 := by simpa [centered] using hi
      have hlambda : 2 < lambda i := by linarith
      simp [psi, centered, largeSet, hi', not_le.mpr hlambda]
      ring
    · have hi' : ¬1 < lambda i - 1 := by simpa [centered] using hi
      have hlambda : lambda i ≤ 2 := by linarith
      simp [psi, centered, largeSet, hi', hlambda]
  unfold defect energy
  simp_rw [hpoint]
  rw [Finset.sum_add_distrib]
  rw [← Finset.sum_filter]
  simp only [Finset.filter_mem_eq_inter, Finset.univ_inter]
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp only [centered, Finset.sum_const, nsmul_eq_mul]
  norm_num
  ring

private lemma sum_centered_eq_zero {m : ℕ} (lambda : Fin m → ℝ)
    (htrace : (∑ i, lambda i) = (m : ℝ)) :
    (∑ i, centered lambda i) = 0 := by
  unfold centered
  rw [Finset.sum_sub_distrib]
  simp [htrace]

/-- The global scalar trace--energy envelope.  The `k ≥ 2` case is compressed to one large
coordinate plus coordinates at the branch point, avoiding any hidden upper range on `E`. -/
theorem traceEnergy_envelope {m : ℕ} (hm : 2 ≤ m)
    (lambda : Fin m → ℝ) (_hlambda : ∀ i, 0 ≤ lambda i)
    (htrace : (∑ i, lambda i) = (m : ℝ)) :
    phi m (energy lambda) ≤ defect lambda := by
  classical
  set L : Finset (Fin m) := largeSet lambda with hLdef
  have hidentity := defect_eq_energy_add_large_correction lambda
  rw [← hLdef] at hidentity
  by_cases hLempty : L = ∅
  · rw [hLempty] at hidentity
    simp at hidentity
    rw [hidentity]
    exact phi_le_self hm (energy_nonneg lambda)
  · let K : ℝ := L.card
    let R : ℝ := ∑ i ∈ L, centered lambda i
    let Q : ℝ := ∑ i ∈ L, centered lambda i ^ 2
    let S : ℝ := ∑ i ∈ Lᶜ, centered lambda i ^ 2
    let r : ℝ := R - K + 1
    let E' : ℝ := S + r ^ 2 + (K - 1)
    have hLnonempty : L.Nonempty := Finset.nonempty_iff_ne_empty.mpr hLempty
    have hK1 : 1 ≤ K := by
      dsimp [K]
      exact_mod_cast (Finset.card_pos.mpr hLnonempty)
    have hLcard_le : L.card ≤ m := by
      simpa using Finset.card_le_card (Finset.subset_univ L)
    have hKle : K ≤ (m : ℝ) := by
      dsimp [K]
      exact_mod_cast hLcard_le
    have hRge : K ≤ R := by
      calc
        K = ∑ _i ∈ L, (1 : ℝ) := by simp [K]
        _ ≤ ∑ i ∈ L, centered lambda i := by
          apply Finset.sum_le_sum
          intro i hi
          have hi' : i ∈ largeSet lambda := by simpa [hLdef] using hi
          exact le_of_lt (Finset.mem_filter.mp hi').2
        _ = R := rfl
    have hr : 1 ≤ r := by dsimp [r]; linarith
    have hxsum : (∑ i, centered lambda i) = 0 := sum_centered_eq_zero lambda htrace
    have hCsum : (∑ i ∈ Lᶜ, centered lambda i) = -R := by
      have hsplit := Finset.sum_compl_add_sum L (centered lambda)
      rw [hxsum] at hsplit
      dsimp [R]
      linarith
    have hEsplit : energy lambda = S + Q := by
      have hsplit := Finset.sum_compl_add_sum L (fun i => centered lambda i ^ 2)
      unfold energy
      change (∑ i, centered lambda i ^ 2) = S + Q
      dsimp [S, Q]
      linarith
    have hCcard : ((Lᶜ).card : ℝ) = (m : ℝ) - K := by
      rw [Finset.card_compl]
      simp only [Fintype.card_fin]
      rw [Nat.cast_sub hLcard_le]
    have hCS0 := sq_sum_le_card_mul_sum_sq (s := Lᶜ) (f := centered lambda)
    have hCS : R ^ 2 ≤ ((m : ℝ) - K) * S := by
      rw [hCsum, hCcard] at hCS0
      simpa [S] using hCS0
    have hS : 0 ≤ S := by
      exact Finset.sum_nonneg fun _ _ => sq_nonneg _
    have hKdiff : 0 ≤ (m : ℝ) - K := sub_nonneg.mpr hKle
    have hR1 : 1 ≤ R := hK1.trans hRge
    have hKdiff_pos : 0 < (m : ℝ) - K := by
      by_contra hnot
      have hzero : (m : ℝ) - K = 0 := le_antisymm (not_lt.mp hnot) hKdiff
      rw [hzero, zero_mul] at hCS
      nlinarith [sq_nonneg R]
    have hy_nonneg : ∀ i ∈ L, 0 ≤ centered lambda i - 1 := by
      intro i hi
      have hi' : i ∈ largeSet lambda := by simpa [hLdef] using hi
      linarith [(Finset.mem_filter.mp hi').2]
    have hsum_y : (∑ i ∈ L, (centered lambda i - 1)) = R - K := by
      rw [Finset.sum_sub_distrib]
      simp [R, K]
    have hy_squares := Finset.sum_sq_le_sq_sum_of_nonneg hy_nonneg
    rw [hsum_y] at hy_squares
    have hQexpand :
        Q = (∑ i ∈ L, (centered lambda i - 1) ^ 2) + 2 * (R - K) + K := by
      dsimp [Q]
      calc
        (∑ i ∈ L, centered lambda i ^ 2) =
            ∑ i ∈ L, ((centered lambda i - 1) ^ 2
              + 2 * (centered lambda i - 1) + 1) := by
                apply Finset.sum_congr rfl
                intro i hi
                ring
        _ = (∑ i ∈ L, (centered lambda i - 1) ^ 2)
              + 2 * (R - K) + K := by
                rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
                  hsum_y]
                simp [K]
    have hQupper : Q ≤ r ^ 2 + (K - 1) := by
      dsimp [r]
      nlinarith
    have hE_nonneg : 0 ≤ energy lambda := energy_nonneg lambda
    have hE'_nonneg : 0 ≤ E' := by
      dsimp [E']
      nlinarith [sq_nonneg r]
    have hE_le : energy lambda ≤ E' := by
      rw [hEsplit]
      dsimp [E']
      linarith
    have hDcompressed : defect lambda = E' + 2 * r - 1 - r ^ 2 := by
      rw [hidentity, hEsplit]
      dsimp [E', r, R, Q, S, K]
      ring
    have hscaledCS :
        ((m : ℝ) - 1) * R ^ 2 ≤
          ((m : ℝ) - 1) * (((m : ℝ) - K) * S) :=
      mul_le_mul_of_nonneg_left hCS (by linarith [cast_m_sub_one_pos hm])
    have hcompressionFactor :
        0 ≤ (K - 1) * (R + (m : ℝ) - K) ^ 2 :=
      mul_nonneg (sub_nonneg.mpr hK1) (sq_nonneg _)
    have hmultiplied :
        ((m : ℝ) - K) * r ^ 2 ≤
          ((m : ℝ) - K) * (((m : ℝ) - 1) * (S + K - 1)) := by
      dsimp [r]
      nlinarith
    have hvirtualBase : r ^ 2 ≤ ((m : ℝ) - 1) * (S + K - 1) := by
      exact (mul_le_mul_iff_of_pos_left hKdiff_pos).mp hmultiplied
    have hvirtualCauchy :
        (m : ℝ) * r ^ 2 ≤ ((m : ℝ) - 1) * E' := by
      dsimp [E']
      nlinarith
    have hphiE' : phi m E' ≤ defect lambda :=
      phi_le_of_one_large hm hE'_nonneg hr hvirtualCauchy hDcompressed
    have hmono : phi m (energy lambda) ≤ phi m E' :=
      phi_monoOn_nonneg hm hE_nonneg hE'_nonneg hE_le
    exact hmono.trans hphiE'

/-- Pressure transfer in the exact domain needed by WI-011. -/
theorem traceEnergy_pressure {m : ℕ} (hm : 2 ≤ m)
    (lambda : Fin m → ℝ) (hlambda : ∀ i, 0 ≤ lambda i)
    (htrace : (∑ i, lambda i) = (m : ℝ))
    {A P : ℝ} (hA : 0 ≤ A) (hP : 0 ≤ P)
    (hbudget : A ≤ energy lambda + P) :
    phi m A ≤ defect lambda + P := by
  by_cases hAE : A ≤ energy lambda
  · exact (phi_monoOn_nonneg hm hA (energy_nonneg lambda) hAE).trans
      ((traceEnergy_envelope hm lambda hlambda htrace).trans
        (le_add_of_nonneg_right hP))
  · have hEA : energy lambda ≤ A := le_of_not_ge hAE
    have hinc := phi_increment_le hm (energy_nonneg lambda) hEA
    have henv := traceEnergy_envelope hm lambda hlambda htrace
    have hgap : A - energy lambda ≤ P := by linarith
    linarith

/-- The externally Lean-checked four-point certificate's exact target, used here only as data. -/
def epsilon4 : ℝ := 231 / 100000

/-- The WI-011 block target at `m = 438`. -/
def A438 : ℝ := 20097 / 20000

theorem A438_eq : epsilon4 * ((438 : ℝ) - 3) = A438 := by
  norm_num [epsilon4, A438]

theorem A438_gt_branch : (438 : ℝ) / 437 < A438 := by
  norm_num [A438]

theorem phi438_exact :
    phi 438 A438 =
      2 * Real.sqrt (8782389 / 8760000) - 1 + 20097 / 8760000 := by
  norm_num [phi, A438]

theorem phi438_interval :
    (1004848 / 1000000 : ℝ) < phi 438 A438 ∧
      phi 438 A438 < 1004849 / 1000000 := by
  rw [phi438_exact]
  let s : ℝ := Real.sqrt (8782389 / 8760000)
  have hs_nonneg : 0 ≤ s := by positivity
  have hs_lower : (100127692 / 100000000 : ℝ) < s := by
    have hsq : (100127692 / 100000000 : ℝ) ^ 2 < 8782389 / 8760000 := by
      norm_num
    have h := Real.sqrt_lt_sqrt (sq_nonneg (100127692 / 100000000 : ℝ)) hsq
    rw [Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 100127692 / 100000000)] at h
    exact h
  have hs_upper : s < (10012774 / 10000000 : ℝ) := by
    have hsq : (8782389 / 8760000 : ℝ) < (10012774 / 10000000 : ℝ) ^ 2 := by
      norm_num
    have h := Real.sqrt_lt_sqrt (by norm_num : (0 : ℝ) ≤ 8782389 / 8760000) hsq
    rw [Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 10012774 / 10000000)] at h
    exact h
  change (1004848 / 1000000 : ℝ) < 2 * s - 1 + 20097 / 8760000 ∧
    2 * s - 1 + 20097 / 8760000 < 1004849 / 1000000
  constructor <;> nlinarith

theorem phi438_lt_two : phi 438 A438 < 2 := by
  linarith [phi438_interval.2]

#print axioms defect_eq_energy_add_large_correction
#print axioms traceEnergy_envelope
#print axioms traceEnergy_pressure
#print axioms phi438_exact
#print axioms phi438_interval

end Mathia.WI011
