# Qwen3.5-4B thinking-budget scaling continuation

**OBSERVED:** the revised canonical parsed-output gate passed and all 48 frozen B4/B8/B16 candidates were generated and verified under both the strict parsed and deployed normalized interfaces.

| Arm | Non-empty final | Strict verified | Deployed verified | Wrapper changed | Raw tokens |
|---|---:|---:|---:|---:|---:|
| B4 | 16/16 | 1/16 | 1/16 | 0/16 | 102819 |
| B8 | 16/16 | 1/16 | 1/16 | 0/16 | 152987 |
| B16 | 16/16 | 1/16 | 1/16 | 0/16 | 230603 |

**OBSERVED conclusion:** `budget_control_works_but_longer_thinking_only_adds_cost`.

This remains an operational preflight, not a powered pass@k study. It does not alter the historical Stage 1/Stage 2a targets or authorize a larger experiment. Raw artifacts remain Git-ignored.
