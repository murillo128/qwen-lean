import Mathlib
import WI011TraceEnergy

/-!
# Mathia WI-011: four-point windows and finite shifted-block accounting

All pair weights are arbitrary and nonnegative.  In particular, none of the finite theorems
assumes translation invariance or a Montgomery--Taylor kernel.  The local four-point certificate
is an explicit hypothesis, not an imported assumption-free theorem.
-/

noncomputable section

open scoped BigOperators

namespace Mathia.WI011

/-- The non-pressure part of the WI-009 four-point functional at window start `s`. -/
def fourPointPairSpend (w : ℕ → ℕ → ℝ) (s : ℕ) : ℝ :=
  (2 / 3 : ℝ) *
      (w s (s + 1) + w (s + 1) (s + 2) + w (s + 2) (s + 3))
    + (w s (s + 2) + w (s + 1) (s + 3))
    + 2 * w s (s + 3)

/-- Pair energy in a block of `q+4` points, enumerated by positive pair separation. -/
def blockPairEnergy (q : ℕ) (w : ℕ → ℕ → ℝ) : ℝ :=
  2 * ∑ r ∈ Finset.range (q + 3),
    ∑ i ∈ Finset.range (q + 3 - r), w i (i + r + 1)

/-- The contribution at one positive pair separation `r+1`. -/
def offsetPairEnergy (q : ℕ) (w : ℕ → ℕ → ℝ) (r : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (q + 3 - r), w i (i + r + 1)

/-- A three-gap span beginning at gap `s`. -/
def threeGapSpan (g : ℕ → ℝ) (s : ℕ) : ℝ :=
  g s + g (s + 1) + g (s + 2)

/-- Exact endpoint accounting for the sum of all three-gap spans in a block.  At `q=0`
(four points), the two middle endpoint terms intentionally name the same gap. -/
theorem threeGapSpan_boundary_identity (q : ℕ) (g : ℕ → ℝ) :
    3 * (∑ j ∈ Finset.range (q + 3), g j) =
      (∑ s ∈ Finset.range (q + 1), threeGapSpan g s)
        + 2 * g 0 + g 1 + g (q + 1) + 2 * g (q + 2) := by
  induction q with
  | zero =>
      norm_num [threeGapSpan, Finset.sum_range_succ]
      ring
  | succ q ih =>
      have hleft : (∑ j ∈ Finset.range (q + 1 + 3), g j) =
          (∑ j ∈ Finset.range (q + 3), g j) + g (q + 3) := by
        rw [show q + 1 + 3 = (q + 3) + 1 by omega, Finset.sum_range_succ]
      have hwindows : (∑ s ∈ Finset.range (q + 1 + 1), threeGapSpan g s) =
          (∑ s ∈ Finset.range (q + 1), threeGapSpan g s) + threeGapSpan g (q + 1) := by
        rw [Finset.sum_range_succ]
      have hnew : threeGapSpan g (q + 1) = g (q + 1) + g (q + 2) + g (q + 3) := by
        simp [threeGapSpan, Nat.add_assoc]
      rw [hleft, hwindows]
      rw [hnew]
      norm_num [Nat.add_assoc] at *
      linarith [ih]

private theorem twoTerm_boundary_identity (q : ℕ) (g : ℕ → ℝ) :
    2 * (∑ j ∈ Finset.range (q + 2), g j) =
      (∑ s ∈ Finset.range (q + 1), (g s + g (s + 1))) + g 0 + g (q + 1) := by
  induction q with
  | zero =>
      norm_num [Finset.sum_range_succ]
      ring
  | succ q ih =>
      have hleft : (∑ j ∈ Finset.range (q + 1 + 2), g j) =
          (∑ j ∈ Finset.range (q + 2), g j) + g (q + 2) := by
        rw [show q + 1 + 2 = (q + 2) + 1 by omega, Finset.sum_range_succ]
      have hwindows : (∑ s ∈ Finset.range (q + 1 + 1), (g s + g (s + 1))) =
          (∑ s ∈ Finset.range (q + 1), (g s + g (s + 1)))
            + (g (q + 1) + g (q + 2)) := by
        rw [Finset.sum_range_succ]
      rw [hleft, hwindows]
      norm_num [Nat.add_assoc] at *
      linarith [ih]

/-- Summed four-point pair spend consumes no more than the available block pair energy. -/
theorem fourPointPairSpend_sum_le_blockPairEnergy
    (q : ℕ) (w : ℕ → ℕ → ℝ) (hw : ∀ i j, 0 ≤ w i j) :
    (∑ s ∈ Finset.range (q + 1), fourPointPairSpend w s) ≤ blockPairEnergy q w := by
  let a : ℕ → ℝ := fun i => w i (i + 1)
  let b : ℕ → ℝ := fun i => w i (i + 2)
  let c : ℕ → ℝ := fun i => w i (i + 3)
  have ha : ∀ i, 0 ≤ a i := fun i => hw i (i + 1)
  have hb : ∀ i, 0 ≤ b i := fun i => hw i (i + 2)
  have hc : ∀ i, 0 ≤ c i := fun i => hw i (i + 3)
  have haIdentity := threeGapSpan_boundary_identity q a
  have haLoss : 0 ≤ 2 * a 0 + a 1 + a (q + 1) + 2 * a (q + 2) := by
    have := ha 0
    have := ha 1
    have := ha (q + 1)
    have := ha (q + 2)
    linarith
  have haBound :
      (∑ s ∈ Finset.range (q + 1), threeGapSpan a s) ≤
        3 * ∑ i ∈ Finset.range (q + 3), a i := by
    linarith
  have hbIdentity := twoTerm_boundary_identity q b
  have hbLoss : 0 ≤ b 0 + b (q + 1) := add_nonneg (hb 0) (hb (q + 1))
  have hbBound :
      (∑ s ∈ Finset.range (q + 1), (b s + b (s + 1))) ≤
        2 * ∑ i ∈ Finset.range (q + 2), b i := by
    linarith
  have hspend :
      (∑ s ∈ Finset.range (q + 1), fourPointPairSpend w s) =
        (2 / 3 : ℝ) * (∑ s ∈ Finset.range (q + 1), threeGapSpan a s)
          + (∑ s ∈ Finset.range (q + 1), (b s + b (s + 1)))
          + 2 * (∑ s ∈ Finset.range (q + 1), c s) := by
    simp only [fourPointPairSpend, threeGapSpan, a, b, c]
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum]
  have hspendNear :
      (∑ s ∈ Finset.range (q + 1), fourPointPairSpend w s) ≤
        2 * ((∑ i ∈ Finset.range (q + 3), a i)
          + (∑ i ∈ Finset.range (q + 2), b i)
          + (∑ i ∈ Finset.range (q + 1), c i)) := by
    rw [hspend]
    nlinarith
  have hoffset_nonneg : ∀ r, 0 ≤ offsetPairEnergy q w r := by
    intro r
    exact Finset.sum_nonneg fun i _ => hw i (i + r + 1)
  have hoffsetSplit :
      (∑ r ∈ Finset.range (q + 3), offsetPairEnergy q w r) =
        offsetPairEnergy q w 0 + offsetPairEnergy q w 1 + offsetPairEnergy q w 2
          + ∑ r ∈ Finset.range q, offsetPairEnergy q w (3 + r) := by
    rw [show q + 3 = 3 + q by omega, Finset.sum_range_add]
    norm_num [Finset.sum_range_succ]
  have hoffset0 : offsetPairEnergy q w 0 = ∑ i ∈ Finset.range (q + 3), a i := by
    simp [offsetPairEnergy, a]
  have hoffset1 : offsetPairEnergy q w 1 = ∑ i ∈ Finset.range (q + 2), b i := by
    have hsub : q + 3 - 1 = q + 2 := by omega
    simp [offsetPairEnergy, b, hsub, Nat.add_assoc]
  have hoffset2 : offsetPairEnergy q w 2 = ∑ i ∈ Finset.range (q + 1), c i := by
    have hsub : q + 3 - 2 = q + 1 := by omega
    simp [offsetPairEnergy, c, hsub, Nat.add_assoc]
  have hnearBlock :
      2 * ((∑ i ∈ Finset.range (q + 3), a i)
          + (∑ i ∈ Finset.range (q + 2), b i)
          + (∑ i ∈ Finset.range (q + 1), c i)) ≤ blockPairEnergy q w := by
    have hfar : 0 ≤ ∑ r ∈ Finset.range q, offsetPairEnergy q w (3 + r) :=
      Finset.sum_nonneg fun r _ => hoffset_nonneg (3 + r)
    rw [blockPairEnergy]
    change 2 * ((∑ i ∈ Finset.range (q + 3), a i)
          + (∑ i ∈ Finset.range (q + 2), b i)
          + (∑ i ∈ Finset.range (q + 1), c i)) ≤
      2 * ∑ r ∈ Finset.range (q + 3), offsetPairEnergy q w r
    rw [hoffsetSplit]
    rw [hoffset0, hoffset1, hoffset2]
    linarith
  exact hspendNear.trans hnearBlock

/-- Summing any parameterized local four-point certificate gives the WI-011 block budget. -/
theorem localCertificate_to_block
    (q : ℕ) (w : ℕ → ℕ → ℝ) (pressure : ℕ → ℝ) (epsilon : ℝ)
    (hw : ∀ i j, 0 ≤ w i j)
    (_hpressure : ∀ s, s < q + 1 → 0 ≤ pressure s)
    (hcert : ∀ s, s < q + 1 →
      epsilon ≤ fourPointPairSpend w s + pressure s) :
    epsilon * (q + 1) ≤
      blockPairEnergy q w + ∑ s ∈ Finset.range (q + 1), pressure s := by
  have hsum := Finset.sum_le_sum fun s hs => hcert s (Finset.mem_range.mp hs)
  have hspend := fourPointPairSpend_sum_le_blockPairEnergy q w hw
  rw [Finset.sum_add_distrib] at hsum
  simp only [Finset.sum_const, nsmul_eq_mul] at hsum
  norm_num at hsum
  nlinarith

/-- The generic finite splice: a local four-point certificate plus the exact pair-energy
identification feeds the scalar spectral pressure theorem. -/
theorem localCertificate_to_spectralDefect
    (q : ℕ) (w : ℕ → ℕ → ℝ) (pressure : ℕ → ℝ) (epsilon : ℝ)
    (lambda : Fin (q + 4) → ℝ)
    (hlambda : ∀ i, 0 ≤ lambda i)
    (htrace : (∑ i, lambda i) = (q + 4 : ℕ))
    (henergy : energy lambda = blockPairEnergy q w)
    (hepsilon : 0 ≤ epsilon)
    (hw : ∀ i j, 0 ≤ w i j)
    (hpressure : ∀ s, s < q + 1 → 0 ≤ pressure s)
    (hcert : ∀ s, s < q + 1 →
      epsilon ≤ fourPointPairSpend w s + pressure s) :
    phi (q + 4) (epsilon * (q + 1)) ≤
      defect lambda + ∑ s ∈ Finset.range (q + 1), pressure s := by
  have hblock := localCertificate_to_block q w pressure epsilon hw hpressure hcert
  have hP : 0 ≤ ∑ s ∈ Finset.range (q + 1), pressure s :=
    Finset.sum_nonneg fun s hs => hpressure s (Finset.mem_range.mp hs)
  have hA : 0 ≤ epsilon * ((q + 1 : ℕ) : ℝ) := by positivity
  have hbudget : epsilon * ((q + 1 : ℕ) : ℝ) ≤
      energy lambda + ∑ s ∈ Finset.range (q + 1), pressure s := by
    rw [henergy]
    simpa [Nat.cast_add, Nat.cast_one] using hblock
  have hresult := traceEnergy_pressure (by omega : 2 ≤ q + 4) lambda hlambda htrace
    hA hP hbudget
  simpa [Nat.cast_add, Nat.cast_one] using hresult

/-- The exact `m=438`, `epsilon=231/100000` finite WI-011 splice.  The local certificate
remains a parameterized hypothesis; no external zeta theorem is imported. -/
theorem wi011_m438_finite_splice
    (w : ℕ → ℕ → ℝ) (pressure : ℕ → ℝ) (lambda : Fin 438 → ℝ)
    (hlambda : ∀ i, 0 ≤ lambda i)
    (htrace : (∑ i, lambda i) = (438 : ℝ))
    (henergy : energy lambda = blockPairEnergy 434 w)
    (hw : ∀ i j, 0 ≤ w i j)
    (hpressure : ∀ s, s < 435 → 0 ≤ pressure s)
    (hcert : ∀ s, s < 435 →
      epsilon4 ≤ fourPointPairSpend w s + pressure s) :
    phi 438 A438 ≤ defect lambda + ∑ s ∈ Finset.range 435, pressure s := by
  have h := localCertificate_to_spectralDefect 434 w pressure epsilon4 lambda hlambda
    (by exact_mod_cast htrace) henergy (by norm_num [epsilon4]) hw hpressure hcert
  norm_num [epsilon4, A438] at h ⊢
  exact h

private def offsetPairSum (n : ℕ) (w : ℕ → ℕ → ℝ) : ℝ :=
  ∑ r ∈ Finset.range n, ∑ i ∈ Finset.range (n - r), w i (i + r + 1)

private def orderedPairSum (n : ℕ) (w : ℕ → ℕ → ℝ) : ℝ :=
  ∑ i ∈ Finset.range (n + 1), ∑ j ∈ Finset.Ico (i + 1) (n + 1), w i j

private theorem offsetPairSum_succ (n : ℕ) (w : ℕ → ℕ → ℝ) :
    offsetPairSum (n + 1) w = offsetPairSum n w
      + ∑ i ∈ Finset.range (n + 1), w i (n + 1) := by
  have hinner : ∀ r ∈ Finset.range n,
      (∑ i ∈ Finset.range (n + 1 - r), w i (i + r + 1)) =
        (∑ i ∈ Finset.range (n - r), w i (i + r + 1)) + w (n - r) (n + 1) := by
    intro r hr
    have hrn := Finset.mem_range.mp hr
    have hcount : n + 1 - r = (n - r) + 1 := by omega
    rw [hcount, Finset.sum_range_succ]
    congr 1
    congr 1
    omega
  have hmain :
      (∑ r ∈ Finset.range n,
          ∑ i ∈ Finset.range (n + 1 - r), w i (i + r + 1)) =
        (∑ r ∈ Finset.range n,
          ∑ i ∈ Finset.range (n - r), w i (i + r + 1))
          + ∑ r ∈ Finset.range n, w (n - r) (n + 1) := by
    calc
      _ = ∑ r ∈ Finset.range n,
          ((∑ i ∈ Finset.range (n - r), w i (i + r + 1)) + w (n - r) (n + 1)) := by
            apply Finset.sum_congr rfl
            exact hinner
      _ = _ := Finset.sum_add_distrib
  have hextra :
      (∑ r ∈ Finset.range n, w (n - r) (n + 1)) + w 0 (n + 1) =
        ∑ i ∈ Finset.range (n + 1), w i (n + 1) := by
    have hreflect := Finset.sum_range_reflect (fun i => w i (n + 1)) (n + 1)
    rw [Finset.sum_range_succ] at hreflect
    simpa using hreflect
  unfold offsetPairSum
  rw [Finset.sum_range_succ, hmain]
  have hlast : (∑ i ∈ Finset.range (n + 1 - n), w i (i + n + 1)) = w 0 (n + 1) := by
    norm_num
  rw [hlast]
  linarith

private theorem orderedPairSum_succ (n : ℕ) (w : ℕ → ℕ → ℝ) :
    orderedPairSum (n + 1) w = orderedPairSum n w
      + ∑ i ∈ Finset.range (n + 1), w i (n + 1) := by
  have hinner : ∀ i ∈ Finset.range (n + 1),
      (∑ j ∈ Finset.Ico (i + 1) (n + 2), w i j) =
        (∑ j ∈ Finset.Ico (i + 1) (n + 1), w i j) + w i (n + 1) := by
    intro i hi
    have hi' := Finset.mem_range.mp hi
    exact Finset.sum_Ico_succ_top (by omega) (w i)
  have hmain :
      (∑ i ∈ Finset.range (n + 1), ∑ j ∈ Finset.Ico (i + 1) (n + 2), w i j) =
        (∑ i ∈ Finset.range (n + 1), ∑ j ∈ Finset.Ico (i + 1) (n + 1), w i j)
          + ∑ i ∈ Finset.range (n + 1), w i (n + 1) := by
    calc
      _ = ∑ i ∈ Finset.range (n + 1),
          ((∑ j ∈ Finset.Ico (i + 1) (n + 1), w i j) + w i (n + 1)) := by
            apply Finset.sum_congr rfl
            exact hinner
      _ = _ := Finset.sum_add_distrib
  unfold orderedPairSum
  rw [show n + 1 + 1 = (n + 1) + 1 by omega, Finset.sum_range_succ, hmain]
  simp

private theorem offsetPairSum_eq_orderedPairSum (n : ℕ) (w : ℕ → ℕ → ℝ) :
    offsetPairSum n w = orderedPairSum n w := by
  induction n with
  | zero => simp [offsetPairSum, orderedPairSum]
  | succ n ih =>
      rw [offsetPairSum_succ, orderedPairSum_succ, ih]

/-- The offset enumeration used by `blockPairEnergy` is exactly the conventional sum over
all pairs `0 ≤ i < j < q+4`. -/
theorem blockPairEnergy_eq_pairSum (q : ℕ) (w : ℕ → ℕ → ℝ) :
    blockPairEnergy q w =
      2 * ∑ i ∈ Finset.range (q + 4),
        ∑ j ∈ Finset.Ico (i + 1) (q + 4), w i j := by
  change 2 * offsetPairSum (q + 3) w = 2 * orderedPairSum (q + 3) w
  rw [offsetPairSum_eq_orderedPairSum]

/-- Natural-number block offsets at which a four-point window does not cross a boundary. -/
def fourPointContainingOffsets (m : ℕ) : Finset ℕ :=
  (Finset.range m).filter fun a => a + 3 < m

/-- A fixed four-point window is inside a block in exactly `m-3` of the `m` alignments. -/
theorem fourPoint_containing_shift_count {m : ℕ} (hm : 4 ≤ m) :
    (fourPointContainingOffsets m).card = m - 3 := by
  have hset : fourPointContainingOffsets m = Finset.range (m - 3) := by
    ext a
    simp only [fourPointContainingOffsets, Finset.mem_filter, Finset.mem_range]
    omega
  rw [hset, Finset.card_range]

/-- Candidate within-block positions for a four-point window starting at `s`, retaining only
alignments whose whole `q+4` point block lies inside the finite frame of `n` points. -/
def fullBlockOffsets (n q s : ℕ) : Finset ℕ :=
  (Finset.range (q + 1)).filter fun t => t ≤ s ∧ s - t + (q + 4) ≤ n

/-- Every one of the `q+1=m-3` alignments is present for an interior window. -/
theorem fullBlockOffsets_card_of_interior {n q s : ℕ}
    (hleft : q ≤ s) (hright : s + q + 4 ≤ n) :
    (fullBlockOffsets n q s).card = q + 1 := by
  have hset : fullBlockOffsets n q s = Finset.range (q + 1) := by
    apply Finset.filter_eq_self.mpr
    intro t ht
    simp only [Finset.mem_range] at ht
    constructor <;> omega
  rw [hset, Finset.card_range]

/-- Window starts at which a containing aligned block may be incomplete. -/
def exceptionalFourPointStarts (n q : ℕ) : Finset ℕ :=
  (Finset.range (n - 3)).filter fun s => ¬(q ≤ s ∧ s + q + 4 ≤ n)

/-- For a frame at least one block long, only the first and last `q=m-4` window starts can be
exceptional. -/
private theorem exceptional_fourPoint_starts_card_le_of_frame {n q : ℕ} (hn : q + 4 ≤ n) :
    (exceptionalFourPointStarts n q).card ≤ 2 * q := by
  let left : Finset ℕ := Finset.range q
  let right : Finset ℕ := Finset.Ico (n - q - 3) (n - 3)
  have hsubset : exceptionalFourPointStarts n q ⊆ left ∪ right := by
    intro s hs
    simp only [exceptionalFourPointStarts, Finset.mem_filter, Finset.mem_range,
      not_and_or, not_le] at hs
    simp only [Finset.mem_union, left, right, Finset.mem_range, Finset.mem_Ico]
    rcases hs.2 with hsq | hright
    · exact Or.inl hsq
    · right
      constructor <;> omega
  calc
    (exceptionalFourPointStarts n q).card ≤ (left ∪ right).card :=
      Finset.card_le_card hsubset
    _ ≤ left.card + right.card := Finset.card_union_le left right
    _ ≤ 2 * q := by
      simp only [left, right, Finset.card_range, Nat.card_Ico]
      omega

theorem exceptional_fourPoint_starts_card_le {n q : ℕ} :
    (exceptionalFourPointStarts n q).card ≤ 2 * q := by
  by_cases hn : q + 4 ≤ n
  · exact exceptional_fourPoint_starts_card_le_of_frame hn
  · calc
      (exceptionalFourPointStarts n q).card ≤ (Finset.range (n - 3)).card :=
        Finset.card_le_card (Finset.filter_subset _ _)
      _ = n - 3 := Finset.card_range _
      _ ≤ 2 * q := by omega

/-- The finite frame loses at most `2q(q+1)=2(m-4)(m-3)` containing-block incidences. -/
theorem finite_containment_incidence_with_boundary {n q : ℕ} :
    (q + 1) * (n - 3) ≤
      (∑ s ∈ Finset.range (n - 3), (fullBlockOffsets n q s).card)
        + 2 * q * (q + 1) := by
  let starts : Finset ℕ := Finset.range (n - 3)
  let exc : Finset ℕ := exceptionalFourPointStarts n q
  let good : Finset ℕ := starts \ exc
  have hexc_subset : exc ⊆ starts := by
    intro s hs
    exact (Finset.mem_filter.mp hs).1
  have hgood_subset : good ⊆ starts := Finset.sdiff_subset
  have hgood_card : good.card + exc.card = starts.card := by
    have h := Finset.card_sdiff_add_card_inter starts exc
    rw [Finset.inter_eq_right.mpr hexc_subset] at h
    exact h
  have hgood_offsets : ∀ s ∈ good, (fullBlockOffsets n q s).card = q + 1 := by
    intro s hs
    have hs' := Finset.mem_sdiff.mp hs
    have hsStart := Finset.mem_range.mp hs'.1
    have hsNot := hs'.2
    have hinterior : q ≤ s ∧ s + q + 4 ≤ n := by
      simp only [exc, exceptionalFourPointStarts, Finset.mem_filter, Finset.mem_range,
        hsStart, true_and] at hsNot
      exact not_not.mp hsNot
    exact fullBlockOffsets_card_of_interior hinterior.1 hinterior.2
  have hsum_good :
      (∑ s ∈ good, (fullBlockOffsets n q s).card) = good.card * (q + 1) := by
    calc
      _ = ∑ _s ∈ good, (q + 1) := by
        apply Finset.sum_congr rfl
        exact hgood_offsets
      _ = _ := by simp
  have hsum_le :
      (∑ s ∈ good, (fullBlockOffsets n q s).card) ≤
        ∑ s ∈ starts, (fullBlockOffsets n q s).card :=
    Finset.sum_le_sum_of_subset_of_nonneg hgood_subset (fun _ _ _ => Nat.zero_le _)
  have hexc_card : exc.card ≤ 2 * q := by
    exact exceptional_fourPoint_starts_card_le
  calc
    (q + 1) * (n - 3) = starts.card * (q + 1) := by simp [starts, mul_comm]
    _ = good.card * (q + 1) + exc.card * (q + 1) := by
      rw [← hgood_card, Nat.add_mul]
    _ ≤ (∑ s ∈ starts, (fullBlockOffsets n q s).card) + exc.card * (q + 1) := by
      rw [← hsum_good]
      exact Nat.add_le_add_right hsum_le _
    _ ≤ (∑ s ∈ starts, (fullBlockOffsets n q s).card) + 2 * q * (q + 1) := by
      exact Nat.add_le_add_left (Nat.mul_le_mul_right (q + 1) hexc_card) _
    _ = _ := by rfl

/-- Direct finite regression aid for pair/window multiplicities. -/
def pairWindowMultiplicity (m i r : ℕ) : ℕ :=
  ((Finset.range (m - 3)).filter fun s => s ≤ i ∧ i + r ≤ s + 3).card

/-- Direct finite regression aid for gap/span multiplicities. -/
def gapSpanMultiplicity (m j : ℕ) : ℕ :=
  ((Finset.range (m - 3)).filter fun s => s ≤ j ∧ j < s + 3).card

example : pairWindowMultiplicity 4 0 1 = 1 ∧ pairWindowMultiplicity 4 0 2 = 1 ∧
    pairWindowMultiplicity 4 0 3 = 1 := by native_decide

example : pairWindowMultiplicity 5 1 1 = 2 ∧ pairWindowMultiplicity 5 1 2 = 2 ∧
    pairWindowMultiplicity 5 1 3 = 1 := by native_decide

example : pairWindowMultiplicity 6 2 1 = 3 ∧ pairWindowMultiplicity 6 1 2 = 2 ∧
    pairWindowMultiplicity 6 1 3 = 1 := by native_decide

example : gapSpanMultiplicity 4 1 = 1 ∧ gapSpanMultiplicity 5 1 = 2 ∧
    gapSpanMultiplicity 6 2 = 3 := by native_decide

#print axioms blockPairEnergy_eq_pairSum
#print axioms fourPointPairSpend_sum_le_blockPairEnergy
#print axioms localCertificate_to_block
#print axioms localCertificate_to_spectralDefect
#print axioms wi011_m438_finite_splice
#print axioms threeGapSpan_boundary_identity
#print axioms finite_containment_incidence_with_boundary

end Mathia.WI011
