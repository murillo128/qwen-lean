# Cross-model Lean capability study

This document preserves the conclusions of the bounded cross-model study tracked by GitHub epic #27. It is an analysis record, not a new training contract. The authoritative per-run evidence remains under `evidence/` and in the merged assessment PRs linked from #27.

## Question

The study asked which current open-weight checkpoints are useful reference points or training parents for qwen-lean under one verifier-aligned task: generate a raw Lean whole-proof continuation and let Lean decide success.

The study intentionally separated three questions:

1. **Trainable foundation quality** — which Base/pre-trained model is the strongest starting point for our own Lean post-training?
2. **Generic post-training compatibility** — does ordinary instruction/reasoning post-training help or hurt the strict raw Lean continuation interface?
3. **Specialist ceiling** — how much stronger can a purpose-built Lean prover be under the same one-shot continuation contract?

Spark was included as a separate test-time-reasoning reference, not as a compute-matched local model.

## Shared casting contract

Most new local assessments used the frozen miniF2F validation workload:

- 244 tasks;
- 4 candidates/task = 976 candidates;
- `whole-proof-v1` raw continuation;
- no chat wrapper for the strict lane;
- temperature 0.8, top-p 0.95, no top-k;
- max 1,024 generated tokens;
- seed 0;
- no proof extraction, semantic repair, verifier feedback, or candidate regeneration;
- Lean verification is the success oracle;
- `verifier_timeout` is an unsuccessful proof outcome, not an infrastructure success.

The historical `Qwen3-8B-Base` and `reference-sft-v1` anchors used 8 candidates/task. Their accepted pass@1/pass@4 estimators are reused, but finite-sampling uncertainty differs from the exact 4-candidate casting runs.

## Accepted results

| Model / procedure | Role | pass@1 | pass@4 / measured coverage | Solved tasks | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `Qwen/Qwen3-8B-Base` | historical Base | 0.01281 | 0.04772 | historical n=8 | accepted anchor |
| `Qwen/Qwen3-8B` | generic post-trained | 0.00410 | 0.01639 | 4/244 | accepted |
| `Qwen/Qwen3.5-2B-Base` | trainable Base | 0.01025 | 0.04098 | 10/244 | accepted |
| `Qwen/Qwen3.5-2B` | generic post-trained | 0.00307 | 0.01230 | 3/244 | accepted |
| `Qwen/Qwen3.5-4B-Base` | trainable Base | **0.07172** | **0.18443** | **45/244** | accepted |
| `Qwen/Qwen3.5-4B` | generic post-trained | 0.02357 | 0.08197 | 20/244 | accepted |
| `Qwen/Qwen3.5-9B-Base` | trainable Base | 0.03996 | 0.11885 | 29/244 | accepted |
| `Qwen/Qwen3.5-9B` | generic post-trained | 0.01639 | 0.06148 | 15/244 | accepted |
| `allenai/Olmo-3-1025-7B` | trainable Base | 0.00000 | 0.00000 | 0/244 | accepted |
| `mistralai/Ministral-3-8B-Base-2512` | trainable Base | 0.00000 | 0.00000 | 0/244 | accepted |
| `google/gemma-3-4b-pt` | trainable pre-trained | — | — | — | blocked by gated-access/license acknowledgement |
| `Qwen/Qwen3.6-27B` | larger released checkpoint | — | — | — | single-Ada 4-bit Stage-0 OOM before generation |
| `Goedel-LM/Goedel-Prover-V2-8B` | Lean specialist | 0.02049 | 0.07787 | 19/244 | accepted |
| `deepseek-ai/DeepSeek-Prover-V2-7B` | Lean specialist | **0.15164** | **0.33197** | **81/244** | accepted |
| `reference-sft-v1` | qwen-lean SFT control | 0.03945 | 0.10316 | historical n=8 | accepted anchor |
| Spark `low` | test-time reasoning | — | 0.14344 measured coverage | 35/244 | accepted reference |
| Spark `xhigh` | test-time reasoning | — | 0.20082 measured coverage | 49/244 | accepted reference |

Do not compare Spark's coverage as if it were pass@4 from the local models: Spark uses one reasoning execution per theorem and materially different hidden test-time compute.

## Main conclusions

### 1. Qwen3.5-4B-Base is the strongest measured general/pre-trained training parent

`Qwen3.5-4B-Base` is the clear winner among the accepted trainable Base/pre-trained models under the strict whole-proof casting lane. It solved 45/244 tasks with pass@1 0.07172 and pass@4 0.18443.

The result is strongly non-monotonic in nominal parameter count. `Qwen3.5-9B-Base` solved only 29/244 with pass@1 0.03996 and pass@4 0.11885. The 4B model is therefore about 1.79x higher at pass@1 and 1.55x higher at pass@4 despite being smaller.

This is not evidence that 4B is globally more capable than 9B. It is evidence that its learned distribution is substantially better aligned with direct Lean whole-proof continuation.

### 2. The 4B-vs-9B gap is not just an aggregate-score curiosity

A later recovery audit over the retained local 244x4 candidate artifacts produced a paired task comparison:

- solved by both: 14;
- 4B-only: 31;
- 9B-only: 15;
- solved by neither: 184;
- Jaccard similarity: 0.23333;
- exact two-sided McNemar p-value: 0.025896.

The 46 discordant tasks therefore favored 4B by 31 to 15. This supports treating the observed 4B advantage as a real property of this benchmark run rather than merely comparing two noisy aggregate pass@4 values.

The same recovery audit showed a family-level pattern. Using miniF2F task-name families as a coarse taxonomy:

| Family | Total | 4B solved | 9B solved |
| --- | ---: | ---: | ---: |
| `algebra_*` | 18 | 8 (44.4%) | 5 (27.8%) |
| `mathd_algebra_*` | 70 | 20 (28.6%) | 13 (18.6%) |
| `mathd_numbertheory_*` | 60 | 13 (21.7%) | 6 (10.0%) |
| AMC | 45 | 3 (6.7%) | 4 (8.9%) |
| `induction_*` | 8 | 1 | 1 |
| AIME | 15 | 0 | 0 |
| IMO | 20 | 0 | 0 |
| `numbertheory_*` | 8 | 0 | 0 |

The advantage is concentrated in algebra and `mathd_numbertheory`, not in harder competition families. This is useful motivation for the separate Riemann-specific casting epic #63, because miniF2F does not directly measure the complex-analysis/zeta/PNT machinery relevant to that branch.

**Provenance note:** the paired/task-family numbers above came from a local recovery audit of retained raw assessment artifacts. The raw candidate files were not committed to Git; authoritative repository evidence for the accepted aggregate scores remains the compact evidence in the merged PRs.

### 3. Generic post-training is consistently mismatched to the raw continuation lane

Across both Qwen generations and all comparable sizes, the generic post-trained sibling is materially worse than its Base checkpoint under `whole-proof-v1`:

- Qwen3 8B post-trained retains about 32% of Base pass@1 and 34% of Base pass@4;
- Qwen3.5 2B retains 30% / 30%;
- Qwen3.5 4B retains about 33% / 44%;
- Qwen3.5 9B retains about 41% / 52%.

This does not imply that generic post-trained models are generally weaker. It says that their output distribution is poorly matched to a raw `by` continuation with no chat wrapper or response extraction. For our own Lean specialization, a Base/pre-trained parent is therefore the cleaner and empirically stronger starting point.

### 4. Model family matters at least as much as model size

OLMo 3 7B and Ministral 3 8B Base both completed the strict BF16 workload on the project Ada and solved 0/244 tasks. Ministral generated all 976 candidates to the 1,024-token limit; OLMo produced no verified candidate either.

These results separate hardware feasibility from interface/task suitability. A model may fit and run correctly yet be a poor raw Lean continuation model before specialization.

Gemma 3 4B remains unmeasured because the authenticated project account had not acknowledged the gated Gemma terms. This is a documented external-access blocker rather than a quality result.

### 5. DeepSeek-Prover-V2-7B sets a much higher specialist reference point

DeepSeek-Prover-V2-7B is the strongest accepted local direct-generation endpoint in this study:

- 81/244 solved;
- pass@1 0.15164;
- pass@4 0.33197;
- 148 verified candidates.

It is about 2.11x higher at pass@1 and 1.80x higher at pass@4 than Qwen3.5-4B-Base. It also exceeds the measured solved coverage of both Spark reasoning lanes, although those procedures are not compute-matched.

DeepSeek is not a clean Base-foundation comparison: it is already a heavily Lean-specialized model. The right interpretation is that **specialized post-training has a large remaining payoff**. It is therefore valuable both as a specialist benchmark and as a plausible parent for an additional domain specialization such as Riemann, provided that experiment is kept distinct from training a general foundation from scratch.

The DeepSeek run retained 251 verifier timeouts under documented concurrent host load. Those timeouts remain accepted failures; the reported score is not retrospectively corrected.

### 6. A specialist label is not enough

Goedel-Prover-V2-8B solved 19/244 with pass@1 0.02049 and pass@4 0.07787. Under this strict raw lane it is far below DeepSeek and below `reference-sft-v1` on the accepted pass estimators.

This does not contradict Goedel's intended/native prover behavior: the study deliberately disabled native prompt wrapping and verifier-guided self-correction so that the strict result remained comparable to qwen-lean's direct continuation contract.

### 7. The released Qwen3.6-27B is outside the single-Ada study envelope

The frozen fully GPU-resident 4-bit lane loaded about 17.93 GiB of model state but ran out of memory during engine initialization before the first real generation on the 20 GB RTX 4000 Ada. It therefore has no accepted quality score in this study.

This is a hardware/configuration blocker only. It does not say anything about Qwen3.6-27B quality on larger hardware or under a separately designed quantization experiment.

## Practical model-selection implications

For subsequent qwen-lean experiments, the study supports the following role separation:

- **Preferred general/pre-trained foundation:** `Qwen/Qwen3.5-4B-Base`.
- **General larger runner-up:** `Qwen/Qwen3.5-9B-Base`, retained mainly where domain-specific capacity could matter enough to justify the extra compute.
- **Strong specialist endpoint / specialist-parent candidate:** `deepseek-ai/DeepSeek-Prover-V2-7B`.
- **Historical controlled SFT reference:** `reference-sft-v1` remains the parent/control for the already-defined historical post-SFT branches; this study does not rewrite that experiment.
- **Not competitive under the measured raw casting lane:** OLMo 3 7B Base and Ministral 3 8B Base.
- **Unmeasured:** Gemma 3 4B PT (access blocker) and Qwen3.6-27B quality (single-Ada hardware blocker).

The Riemann branch should not infer its final parent solely from miniF2F. Epic #63 exists to compare credible parents on the frozen Riemann specialist workload, where analytic number theory, zeta/PNT machinery, and complex analysis are more relevant. DeepSeek should be treated there, if evaluated, as a **specialist parent** rather than silently mixed with clean Base-foundation comparisons.

## Evaluation lessons for future post-training and Mathia work

The study established a useful longitudinal evaluation pattern:

- preserve task-level solved outcomes, not only aggregate pass@k;
- report pass@1 separately from solved-within-k coverage;
- use paired wins/losses and McNemar tests when comparing checkpoints on the same task set;
- keep compute/procedure comparisons separate when one system uses hidden reasoning or adaptive search;
- retain domain/family breakdowns where they answer the research question;
- compare future SFT/RL/Riemann checkpoints against their exact parent to measure the post-training delta;
- for Mathia, hold the formal prover and proof-search budget fixed when measuring the incremental value of conceptual planning.

Dataset v2 changes the future training/evaluation data boundary, so the original miniF2F casting set should remain historical evidence rather than becoming the only optimization target for future checkpoints.

## Closure boundary

Epic #27 can be considered complete once the Gemma child is recorded as a documented access blocker: every other retained assessment has an accepted result or an accepted hardware blocker. New models should be assessed in a new bounded issue/epic rather than silently extending this historical study indefinitely.

The separate Riemann foundation-casting epic #63 remains open until its domain-specific candidate assessments and paired synthesis are complete.