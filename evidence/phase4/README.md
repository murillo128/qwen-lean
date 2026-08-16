# Phase 4 evidence

`workloads.json` records every ordered record ID and eligibility count without committing tokenized dataset rows. The remaining files retain the compact production preflight, full-state two-process training trajectory, selected-adapter reload, heldout comparison, and Phase 1-comparable miniF2F result. Every post-selection artifact is bound to the training-selected adapter by optimizer step, logical identity, canonical training-relative path, and the SHA-256 of the raw training artifact. Checkpoints, adapter weights, raw generations, and detailed candidates remain under ignored `artifacts/phase4/`.

**OBSERVED:** the fixed 4,096-example QLoRA trajectory stopped at optimizer step 256 and resumed in a fresh process to step 512 with optimizer, scheduler, RNG, and derived data position preserved. Validation target-token cross-entropy moved from 1.865807 at step 0 through 1.506306, 1.461882, 1.437467, and 1.430901; validation-only selection chose step 512. Peak CUDA reserved memory was 12.12 GiB, below the 24 GiB design ceiling.

**OBSERVED:** on `phase4-heldout64-v1`, unchanged base pass@1/pass@4 were 0.007812/0.031250; the selected adapter produced 0.007812/0.031250. Both runs completed all 64 tasks and 256 candidates with zero infrastructure errors and zero unresolved verifier timeouts. Adapter miniF2F dev16 pass@1/pass@4/pass@8 were 0.000000/0.000000/0.000000 over all 128 candidates under the exact Phase 1 contract.

**ACCEPTED:** Phase 4's smoke/data/comparison integrity gates are satisfied. This is evidence that the fixed infrastructure and configuration are safe to consider for Phase 5 design; it is not a full-corpus SFT result or a requirement that smoke quality improve over the base model.
