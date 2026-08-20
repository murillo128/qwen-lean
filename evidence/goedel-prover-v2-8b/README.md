# Goedel-Prover-V2-8B strict miniF2F assessment

**OBSERVED:** strict pass@1/pass@4 were 0.020492/0.077869; `reference-sft-v1` measured 0.039447/0.103162, and the Qwen3-8B Base anchor measured 0.012807/0.047717.

| Workload | Tasks | Candidates | pass@1 | pass@4 | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | 16 | 64 | 0.000000 | 0.000000 | 0 | 0 |
| full validation | 244 | 976 | 0.020492 | 0.077869 | 0 | 0 |

The strict lane uses the raw `whole-proof-v1` continuation with no chat wrapper, proof extraction, verifier-guided retry, or self-correction. Raw candidates, model weights, caches, and bulky logs remain outside Git. The optional native-prover diagnostic was not run and is not mixed into these scores. Verification used one worker to avoid timeout distortion from a concurrent shared-host Dataset v2 build; verifier semantics were unchanged, but verification and total wall times are host-load-dependent. Peak GPU memory was not measured because the optional NVML monitor package was unavailable.
