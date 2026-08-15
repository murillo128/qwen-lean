# Phase 3 evidence

`preflight.json`, `training.json`, and `adapter-reload.json` record the successful real-GPU QLoRA plumbing checks. `memorization.json` records the failed accepted vLLM free-generation gate without copying bulky per-example outputs. `diagnosis.json` separates the 4-bit training-runtime result from the unchanged BF16 Phase 1 inference-base result.

**OBSERVED:** the first qualifying teacher-forced checkpoint occurred at optimizer step 100, but exact free generation reached only 49/64 on the NF4 training runtime and 27/64 on both BF16 Transformers and vLLM.

**BLOCKED:** the required 56/64 vLLM gate is unmet. The downstream miniF2F adapter smoke was not run because the controlling issue makes memorization its prerequisite. Adapter weights, the materialized workload, trainer state, and detailed candidate outputs remain under ignored `artifacts/`.
