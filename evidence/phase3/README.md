# Phase 3 evidence

`preflight.json`, `training.json`, and `adapter-reload.json` record the successful real-GPU QLoRA plumbing and resumable-training checks. `memorization-checkpoints.json` records every amended stopping boundary; `memorization.json` retains the compact final-boundary result. `semantic-verification.json` and `minif2f-smoke.json` record the two superseding final gates without copying raw candidates.

**OBSERVED:** all six 100-step checkpoints passed teacher-forced eligibility. Fresh BF16 vLLM exact matches were 100→22/64, 200→34/64, 300→44/64, 400→43/64, 500→44/64, 600→49/64, with zero generation infrastructure errors. Step 600 produced 49/64 exact and 49/64 Lean-accepted continuations (48 were both; one additional non-exact continuation was accepted). All 64 were attempted with zero verifier infrastructure errors and zero timeouts.

**ACCEPTED:** step 600 passes the superseding 48/64 exact and semantic gates. The unchanged miniF2F dev16 adapter smoke completed 16/16 candidates with zero infrastructure errors and zero verifier timeouts; 0/16 verified proofs is permitted for this plumbing smoke. Adapter weights, full trainer checkpoints, and detailed candidate outputs remain under ignored `artifacts/`.
