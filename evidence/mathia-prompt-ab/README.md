# Qwen3.5-4B-Base Mathia prompt A/B

**OBSERVED:** This artifact reports issue #86 without regenerating Q0. Arm B produced a statistically clear paired solved@8 advantage over Arm A; adopt the frozen explicit proof-task wording as the default qwen-lean intuition interface.

| workload | arm | solved@8 | verified candidates | pass@1 | pass@4 | pass@8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| minif2f-valid-clean-v2 | A | 39/223 | 61/1784 | 0.034193 | 0.109353 | 0.174888 |
| minif2f-valid-clean-v2 | B | 61/223 | 113/1784 | 0.063341 | 0.180397 | 0.273543 |
| fresh-composition-valid-v2 | A | 2/388 | 2/3104 | 0.000644 | 0.002577 | 0.005155 |
| fresh-composition-valid-v2 | B | 19/388 | 21/3104 | 0.006765 | 0.025957 | 0.048969 |
| combined | A | 41/611 | 63/4888 | 0.012889 | 0.041548 | 0.067103 |
| combined | B | 80/611 | 134/4888 | 0.027414 | 0.082324 | 0.130933 |

## Paired combined result

A-only/B-only/both/neither solved@8: 17/56/24/514. Exact two-sided McNemar p=5.26868e-06.

## Against the unchanged Q0 reference

| interface | solved@8 | pass@1 | pass@4 | pass@8 | Q0 fail -> pass | Q0 pass -> fail | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q0 | 73/611 | 0.026391 | 0.077718 | 0.119476 | — | — | — |
| A | 41/611 | 0.012889 | 0.041548 | 0.067103 | 15 | 47 | 5.77837e-05 |
| B | 80/611 | 0.027414 | 0.082324 | 0.130933 | 30 | 23 | 0.410103 |

Q0 is the unchanged authoritative Dataset-v2 Base evidence from issue #78 restricted to the same 611 tasks. Raw generations and Lean outcomes remain in the bound outside-Git artifact root; the committed JSON binds their atomic shard/result hashes and restart history.

**OBSERVED:** Arm A regressed significantly against Q0, while Arm B was statistically indistinguishable from Q0. The result supports the explicit instruction over raw intuition context, but does not establish that frozen intuition improves on theorem-only Q0.

## Scoring-excluded format diagnostic

`format-contamination-diagnostic.json` records a bounded mechanical wrapper check requested during execution. Transformed variants are diagnostic only and do not modify the raw-continuation classifications or any official metric above.

No model was trained, no Q0 candidate was regenerated, and this result does not automatically change the training contract.
