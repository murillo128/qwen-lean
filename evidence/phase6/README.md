# Phase 6 evidence

The fixed `reference-sft-v1` candidate is the Phase 5 validation-selected step-9962 unmerged PEFT adapter at immutable Hub revision `5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5` on the pinned Qwen base. Its identity was frozen before Phase 6 train generation and the first miniF2F test evaluation; Q1/Q2/Q3 were not eligible alternatives, and no Phase 6 metric changed the candidate.

**OBSERVED:** on `phase6-train512-v1`, base Lean pass@1/pass@4 were 0.001465/0.005859, while SFT produced 0.032715/0.087891. Exact-target pass@1/pass@4 were 0.000000/0.000000 for base and 0.016113/0.042969 for SFT. SFT produced 34 verified non-exact candidates, which are alternative valid proofs rather than retained-target reproductions.

**OBSERVED:** accepted internal heldout base/SFT pass@1 were 0.001953/0.016602; miniF2F validation base/SFT pass@1 were 0.012807/0.039447; first-use miniF2F test base/SFT pass@1 were 0.015369/0.045082. `comparison.json` reports the fixed train/heldout gap formulas, while the workload files retain deterministic 10,000-resample seed-0 task-bootstrap intervals.

**OBSERVED:** mean generated length changed from 414.33 to 70.94 tokens on the train diagnostic and from 258.62 to 301.29 on miniF2F test. Train token-limit finishes changed from 641 to 24; test token-limit finishes changed from 384 to 333. These operational shifts are reported separately from verifier success.

**ACCEPTED:** all new runs completed their fixed candidate counts with zero generation/verifier infrastructure errors and zero unresolved timeouts. `reference-sft-v1` is the controlled common parent and retained SFT control for the independent post-training branches; it is not claimed to be globally optimal SFT.

Raw continuations, model caches, adapter weights, source datasets, and bulky candidate artifacts remain under ignored `artifacts/phase6/` or their accepted Phase 5 locations.
