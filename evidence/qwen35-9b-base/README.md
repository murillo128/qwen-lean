# Qwen3.5-9B-Base strict miniF2F assessment

The immutable `Qwen/Qwen3.5-9B-Base` snapshot `68c46c4b3498877f3ef123c856ecfde50c39f404` was evaluated locally on the project RTX 4000 Ada under the unchanged raw `whole-proof-v1` contract. No chat template, proof extraction, repair, Lean feedback, or retry was applied.

The accepted precision lane was `bf16-text-only-v1`. The BF16 feasibility attempt reported status `passed` and memory failure `False`.

| Workload | Tasks | Candidates | Solved tasks | Verified candidates | pass@1 | pass@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev16 | 16 | 64 | 0 | 0 | 0.0000000000 | 0.0000000000 |
| full validation | 244 | 976 | 29 | 39 | 0.0399590164 | 0.1188524590 |

The full run generated 260516 tokens in 1603.89 generation seconds. It retained 0 verifier-timeout proof outcomes and 0 unresolved infrastructure errors. Raw candidates and model/cache artifacts remain outside Git.
