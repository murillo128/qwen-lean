# Phase 3 evidence

`preflight.json`, `training.json`, and `adapter-reload.json` record the successful real-GPU QLoRA plumbing and resumable-training checks. `memorization-checkpoints.json` records every amended stopping boundary; `memorization.json` retains the compact final-boundary result.

**OBSERVED:** all six 100-step checkpoints passed teacher-forced eligibility. Fresh BF16 vLLM exact matches were 100→22/64, 200→34/64, 300→44/64, 400→43/64, 500→44/64, 600→49/64, with zero generation infrastructure errors.

**BLOCKED:** no checkpoint through the fixed 600-step maximum reached the required 56/64 vLLM gate. The downstream miniF2F adapter smoke was not run because memorization is its prerequisite. Adapter weights, full trainer checkpoints, and detailed candidate outputs remain under ignored `artifacts/`.
