# Qwen3.5-4B strict Lean casting assessment

**OBSERVED:** the post-trained `Qwen/Qwen3.5-4B` strict raw-continuation lane completed
all 244 miniF2F validation tasks and 976 candidates. It verified
23 candidates across 20
tasks. pass@1/pass@4 were 0.023566/0.081967; the accepted `reference-sft-v1`
values were 0.039447/0.103162. The strict scores are 59.74%/79.46% of those
reference values.

The dev16 gate completed 64 candidates with pass@1/pass@4 of
0.000000/0.000000. Its exact generated candidates were retained and reverified after
the original parallel run exposed and tests fixed a shared preamble-probe
synchronization defect. The accepted dev evidence has zero timeouts or
infrastructure errors. The real one-task BF16 compatibility preflight peaked at
17.50 GiB device memory; the full run peaked at 17.82
GiB on NVIDIA RTX 4000 Ada Generation.

**ACCEPTED:** the primary score uses exact `whole-proof-v1` raw continuation,
four candidates per task, temperature 0.8, top-p 0.95, no top-k, 1,024 new
tokens, seed 0, no chat template, no proof extraction, no verifier feedback,
and no repair. It ran in BF16 without quantization using vLLM
`0.23.0` on local project-controlled GPU compute. The supported
vLLM PyTorch-native sampler was frozen because FlashInfer 0.6.12's JIT headers
were incompatible with the available packaged CUDA compiler; this changes only
the implementation of the same temperature/top-p sampling contract.

The cold-start environment probe may use 120 seconds to load the pinned Lean
module graph; every generated candidate retains the unchanged 30-second
verifier timeout. The full run retained 1 `verifier_timeout`
as an unsuccessful proof outcome and recorded zero infrastructure errors.

The model and tokenizer are pinned to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The official model is
Apache-2.0; no weights or raw candidate corpus are committed. Compact JSON here
retains package/hardware identity, memory, counts, error and finish-reason
distributions, token totals, timing, throughput, and compute-per-solved-task.
Raw candidates and model caches remain outside Git.

`comparison.json` records the finite-budget caveat: this issue mandates four
candidates per task, while accepted reference evidence used eight. Both use the
same pass@k estimator and unchanged Lean verifier semantics.
