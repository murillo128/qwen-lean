# Qwen3.5-9B strict miniF2F casting assessment

**OBSERVED:** `reference_sft_higher_at_pass1_and_pass4`. Qwen3.5-9B strict pass@1/pass@4 were 0.016393/0.061475; `reference-sft-v1` pass@1/pass@4 were 0.039447/0.103162. The accepted result uses the BF16 lane.

| Workload | Tasks | Candidates | pass@1 | pass@4 | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | 16 | 64 | 0.000000 | 0.000000 | 0 | 0 |
| full validation | 244 | 976 | 0.016393 | 0.061475 | 0 | 10 |

The strict lane preserves the raw `whole-proof-v1` continuation prompt with no chat template, proof extraction, or Lean-guided retry. Raw candidates, model weights, caches, and bulky logs remain outside Git; the JSON evidence records exact identities, precision, packages, GPU, counts, finish reasons, token lengths, latency, and compute summaries.

Execution limitations: Wall-time and candidate-timeout observations include periods of external non-batch Lean CPU and I/O contention on the shared host; all 10 candidate timeouts remain unsuccessful under the frozen protocol. The accepted full verification reuses the exact generated candidates after serializing the shared verifier preamble probe; model generation was not rerun.
