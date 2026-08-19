# Ministral 3 8B Base strict miniF2F assessment

The immutable `mistralai/Ministral-3-8B-Base-2512` snapshot `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef` was evaluated locally on the project RTX 4000 Ada under the unchanged raw `whole-proof-v1` contract. No chat template, image input, proof extraction, repair, Lean feedback, or retry was applied.

The accepted precision lane was `bf16-text-only-v1`. The BF16 feasibility attempt reported status `passed` and memory failure `False`.

| Workload | Tasks | Candidates | Solved tasks | Verified candidates | pass@1 | pass@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | 16 | 64 | 0 | 0 | 0.0000000000 | 0.0000000000 |
| full validation | 244 | 976 | 0 | 0 | 0.0000000000 | 0.0000000000 |

The full run generated 999424 tokens in 3438.71 generation seconds. It retained 0 verifier-timeout proof outcomes and 0 unresolved infrastructure errors. Raw candidates and model/cache artifacts remain outside Git.
