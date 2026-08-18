# SFT-2 ablation evidence

The immutable `reference-sft-v1` adapter was continued without merging or stacking a second LoRA. A fresh optimizer and fresh 312-step cosine warmup were started for this stage. All 79,696 exact Phase 5 training members were consumed once in 9,962 staged updates; Q1/Q2/Q3 remained diagnostic and the primary endpoint was fixed at Q4 before any SFT-2 result was observed.

**OBSERVED:** full Phase 5 validation trajectory (staged step: target-token metrics) was 0: CE 1.108314, accuracy 0.707521, 2491: CE 1.171958, accuracy 0.697083, 4981: CE 1.124802, accuracy 0.706603, 7472: CE 1.074400, accuracy 0.715115, 9962: CE 1.060242, accuracy 0.717755. Peak CUDA reserved memory was 12.12 GiB. All logged losses and gradients were finite, the reference parent hashes remained unchanged, and the Q4 adapter reloaded in a fresh process.

**OBSERVED:** SFT-2 minus `reference-sft-v1` Lean pass@1/pass@4 deltas were 0.014648/0.021484 on train512 and 0.000488/-0.005859 on heldout512. Exact-target train512 deltas were 0.009766/0.021484. miniF2F validation pass@1/pass@4/pass@8 deltas were 0.006660/0.007670/0.008197. `comparison.json` retains deterministic paired bootstrap intervals and separate learning, memorization, saturation, and regression signals; no opaque combined score is used.

**OBSERVED:** all three generation workloads finished with zero infrastructure errors and zero unresolved verifier timeouts. Train512 retried one stored timed-out candidate twice under the unchanged 300-second contract; miniF2F validation retried five stored timed-out candidates once under the unchanged 30-second contract. The repeated timeouts were recorded as bounded Lean rejections without regenerating candidates or changing their pass/fail contribution. Heldout512 required no retry. The retry policy and counts remain explicit in `comparison.json`.

**ACCEPTED:** the bounded ablation and all integrity gates completed. Quality was not an execution gate. D015 and `reference-sft-v1` remain unchanged, miniF2F test was not evaluated, and SFT-2 is not automatically promoted or published as the reference parent.
