# Phase 5 evidence

`workloads.json` retains full-corpus counts, all over-length exclusions, hashes of the ordered full train/validation memberships, and the explicit ordered 512 heldout IDs. The remaining files retain compact production preflight, two-process training, selected-adapter reload, heldout comparison, and full miniF2F validation evidence. Checkpoints, adapter weights, tokenized workload rows, raw generations, model caches, and bulky logs remain under ignored `artifacts/phase5/`.

**OBSERVED:** `79696` eligible training records were consumed exactly once in 9962 optimizer updates, with 8 record(s) in the final partial update and no duplicate fill. The production process stopped at Q2 step 4981 and resumed in a fresh process with optimizer, scheduler, RNG, and data position preserved. Full `4426`-record validation target-token cross-entropy moved from 1.832871 at step 0 through 1.262099, 1.171152, 1.120195, 1.108314; validation-only selection chose step 9962. Peak CUDA reserved memory was 12.12 GiB, below the 24 GiB design ceiling.

**OBSERVED:** per-step logging covers all 9962 optimizer updates exactly once. Every recorded loss and pre-clipping gradient norm is finite; loss ranged from 0.519600 to 2.427500, and gradient norm ranged from 0.425408 to 8.503636. End-to-end training throughput including boundary validation was 2.036 examples/s.

**OBSERVED:** on `phase5-heldout512-v1`, unchanged base pass@1/pass@4 were 0.001953/0.007812; the selected adapter produced 0.016602/0.048828. Both runs completed 512 tasks and 2,048 candidates with zero infrastructure errors and zero unresolved verifier timeouts. The selected adapter's full miniF2F validation pass@1/pass@4/pass@8 were 0.039447/0.103162/0.143443 over all 244 tasks and 1,952 candidates beside the accepted unchanged-base Phase 1 evidence.

**ACCEPTED:** Phase 5's full-corpus execution and comparison-integrity gates are satisfied. Semantic improvement or regression remains an observed result for Phase 6 analysis; it did not influence checkpoint selection.
