# Q0-pass / Arm-B-fail regression retrospective

**OBSERVED, scoring-excluded:** This diagnostic reuses frozen #78 Q0 and #86 Arm-B candidates. It does not regenerate, repair for scoring, or modify any official classification or metric.

The exact regression set contains 23 tasks: 17 MiniF2F and 6 fresh composition.

| primary classification | tasks | fraction |
| --- | ---: | ---: |
| FORMAT_ONLY | 3 | 13.0% |
| CONTENT_OR_SEARCH | 20 | 87.0% |
| INCONCLUSIVE | 0 | 0.0% |

Mechanically recoverable: 3 tasks and 4 unique candidate proofs (4 verified transform variants).

## Method and provenance

The regression membership is reconstructed from the frozen manifest and all official Arm-B verification records. Only the six declared superficial wrapper removals (and deterministic compositions of at most four removals) are tested. A task is `FORMAT_ONLY` only when Lean accepts at least one transformed candidate in the exact workload environment.

- execution manifest SHA-256: `a878484aed23f127755e85dc03e077aabc1a2b8b8ecd8ca76fe3803a16e15cba`
- #86 verification result-set SHA-256: `c8a1d69db810aa2f9b4468c443b80b330cc7517aea046657ac53e166616cc4c2`
- Q0 compact evidence SHA-256: `76f0d7da0f3e4b8d00c0afd035604991f08afc32173eed3a79cd06dca15f06f5`
- Q0 recovery archive SHA-256: `aeac05f215c9882456a712de341593a19a7a7253da7e4cebff64e015301d9182`
- verifier environment-set SHA-256: `b4bf6652242cbb9109949b2df52c79804c4c11497cf17c1c7138bc6bc69c1c00`

## Per-task evidence

| workload | task | Q0 verified | B finish eos/token limit | variants | classification | tags |
| --- | --- | ---: | ---: | ---: | --- | --- |
| minif2f-valid-clean-v2 | `mathd_algebra_116` | 1 | 5/3 | 2 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `mathd_numbertheory_149` | 3 | 6/2 | 2 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `mathd_algebra_462` | 1 | 8/0 | 4 | FORMAT_ONLY | — |
| minif2f-valid-clean-v2 | `mathd_numbertheory_33` | 2 | 8/0 | 0 | CONTENT_OR_SEARCH | hallucinated_lemma_or_api, different_proof_family_from_q0 |
| minif2f-valid-clean-v2 | `mathd_numbertheory_188` | 2 | 7/1 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `mathd_numbertheory_45` | 1 | 7/1 | 3 | CONTENT_OR_SEARCH | incomplete_or_token_limited |
| minif2f-valid-clean-v2 | `mathd_algebra_245` | 1 | 8/0 | 1 | CONTENT_OR_SEARCH | hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `aime_1984_p15` | 1 | 7/1 | 3 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `mathd_numbertheory_102` | 2 | 8/0 | 3 | FORMAT_ONLY | — |
| minif2f-valid-clean-v2 | `mathd_algebra_55` | 3 | 8/0 | 2 | CONTENT_OR_SEARCH | cannot_determine |
| minif2f-valid-clean-v2 | `mathd_numbertheory_284` | 1 | 7/1 | 1 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `mathd_algebra_234` | 2 | 7/1 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api, lost_simple_q0_strategy |
| minif2f-valid-clean-v2 | `mathd_algebra_493` | 1 | 7/1 | 1 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `mathd_numbertheory_403` | 1 | 4/4 | 1 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| minif2f-valid-clean-v2 | `induction_sum_odd` | 1 | 6/2 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, guidance_followed_but_formalization_failed |
| minif2f-valid-clean-v2 | `mathd_algebra_206` | 1 | 7/1 | 3 | FORMAT_ONLY | — |
| minif2f-valid-clean-v2 | `amc12a_2016_p3` | 1 | 8/0 | 1 | CONTENT_OR_SEARCH | cannot_determine |
| fresh-composition-valid-v2 | `66a407e8ba866f6356cf064b5be0dcca8f0bcaa4fa7cf9ab44ad5e1a00807105` | 1 | 3/5 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, lost_simple_q0_strategy |
| fresh-composition-valid-v2 | `8bb1136d99be78c7e97bd883b56df9e6bcda0b302a1d1b1f7420b67489ad2cba` | 1 | 1/7 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| fresh-composition-valid-v2 | `9920d4f31cbebb14a7c51aec86da6209186dfa2fc05b6d19496aeb9432530f3b` | 1 | 2/6 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| fresh-composition-valid-v2 | `b55a5364a22998e67db4ebbb3caf976203dd1b852f58e0ae09cef552471b1feb` | 1 | 5/3 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| fresh-composition-valid-v2 | `d4217004bec55ff93b35a3174d56eabb37ba727ed6f5f936546d2cbae769a9ac` | 1 | 4/4 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api |
| fresh-composition-valid-v2 | `d4938ddb0414e0fcf87c08d685820aca623ca4ced7a8e1d1bef51d9c4da8b294` | 1 | 3/5 | 0 | CONTENT_OR_SEARCH | incomplete_or_token_limited, hallucinated_lemma_or_api, different_proof_family_from_q0 |

## Training questions

1. Confirmed FORMAT_ONLY failures account for 3/23 tasks (13.0%).
2. After removing confirmed format-only cases, 20/23 tasks (87.0%) remain CONTENT_OR_SEARCH; causal harm from guidance is not established by stochastic n=8 samples.
3. Prioritize a combination: output-protocol SFT addresses the confirmed serialization failures, while the larger residual motivates proof-search and theorem+guidance conditioning work. This retrospective cannot choose between guidance gating and planner training.
4. Observable short-Q0/B tactic-family divergences: `mathd_algebra_234`, `66a407e8ba866f6356cf064b5be0dcca8f0bcaa4fa7cf9ab44ad5e1a00807105`. These are observable short-Q0/B tactic-family divergences only; the stochastic comparison cannot establish that Mathia guidance caused them.
5. Cases where an explicit intuition tactic family appears in B but formalization still fails: `induction_sum_odd`. In these tasks B visibly attempts a tactic family named by the frozen intuition, but all eight candidates still fail Lean.

The Q0-pass/B-fail selection compares two stochastic n=8 samples from different prompt-conditioned distributions. FORMAT_ONLY is directly confirmed by Lean; CONTENT_OR_SEARCH establishes an observable distribution change after strict wrapper transforms, not causal harm from the intuition.

## Permanent raw evidence boundary

`q0-b-regressions/raw-b-candidates.jsonl` contains exactly 184 untouched Arm-B continuations. `q0-b-regressions/q0-verified-candidates.jsonl` contains all 31 authoritative verified Q0 continuations for the same tasks. All transformed candidates are separate in `q0-b-regressions/transformed-b-candidates.jsonl` and reference their source raw SHA-256.

A fresh checkout can audit the committed subset with `uv run pytest -q tests/test_mathia_prompt_ab_regressions.py`. Recomputing the Lean diagnostic additionally requires the frozen #86 artifact root, both frozen Lean projects, and the hash-matching Q0 recovery archive; the complete CLI is available through `python -m qwen_lean.mathia_prompt_ab_regressions --help`.
