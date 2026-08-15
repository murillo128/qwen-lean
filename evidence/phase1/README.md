# Phase 1 unchanged-base miniF2F baseline

`OBSERVED`: the accepted baseline is a complete local-GPU evaluation of the
unchanged `Qwen/Qwen3-8B-Base` model against the pinned miniF2F validation
workload. This is validation evidence for the evaluation pipeline and base
model, not evidence of a fine-tuning improvement.

## Results

| Workload | Tasks | Candidates | Tasks with a verified candidate | pass@1 | pass@4 | pass@8 | Infrastructure errors | Verifier timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 gate | 16 | 128 | 0 (0.00%) | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| full validation | 244 | 1,952 | 21 (8.61%) | 0.012807 | 0.047717 | 0.086066 | 0 | 0 |

The full workload produced 25 verified candidates and 1,927 Lean rejections.
Generation ended at EOS for 1,508 candidates and at the 1,024-token limit for
444 candidates. The complete run took 3,097.13 seconds: 1,235.19 seconds for
generation and 1,861.94 seconds for corrected verification.

## Execution identity

- Model and tokenizer: `Qwen/Qwen3-8B-Base` at
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Benchmark: `google-deepmind/miniF2F` at
  `f0a20e14c1eeccd859d51bb4c2b3ee487889c303`, using only the 244 primary
  declarations in `MiniF2F/Valid.lean` and excluding `.variants.` declarations.
- Lean environment: Lean `v4.27.0`, mathlib
  `a3a10db0e9d66acbebf76c5e6a135066525ac900`, and formal_conjectures
  `f564f8d7a6d28bfeea4deeb0c12aa81348da6d73`.
- Inference: local NVIDIA RTX 4000 Ada Generation, vLLM `0.10.2`, BF16,
  tensor parallelism 1, no quantization.
- Sampling: eight candidates per task, temperature `0.8`, top-p `0.95`, no
  top-k, 1,024 generated-token limit, and seed `0`.

## Verifier correction

Lean 4.27 can print a promoted `hasSorry` diagnostic containing `: error:` while
returning process exit code 0. The production verifier therefore rejects any
Lean error diagnostic as well as nonzero process exits. Environment validation
records that the known `ring` proof is accepted and `sorry` is rejected under
this rule.

The accepted summaries reverify every stored model continuation with the
corrected production verifier. Candidate generation was not rerun or edited:
the SHA-256 digest of the ordered generation-identity projection
(`task_id`, `candidate_id`, `candidate_index`, `candidate_text`, generation
latency, generated-token count, and finish reason) is
`1af6e925fa5ea55924bdc9c5015e2942b9e94d1c3133774164af83026cc4ac41`
for both the original and accepted full-run records.

The JSON files in this directory are the compact accepted evidence. Raw model
continuations, external benchmark checkouts, model caches, and bulky logs remain
local and ignored by Git.
