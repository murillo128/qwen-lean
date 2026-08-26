# qwen-lean generalist v3 Stage-0 feasibility

`OBSERVED`: the exact merged Dataset-v3 package binds successfully and contains
178,448 training theorems and 317,554 optimizer examples. Under the pinned
Qwen3.5 tokenizer, 317,265 examples fit within 4,096 tokens, but the longest
whole-proof example contains 157,034 tokens. The smallest declared supported
context that preserves every example without truncation or dropping is 262,144.

`BLOCKED`: the required no-update near-maximum forward/backward preflight failed
on the project RTX 4000 Ada before the first decoder layer completed. The
Gated DeltaNet `in_proj_qkv` LoRA path requested one 4.79 GiB allocation with
only 4.69 GiB free. No optimizer was created, no adapter update occurred, and
no Base or trained-model canary generation was started.

This is a contract-feasibility failure, not a model-quality result. Resolving it
requires design authority to choose and freeze one materially different path:

- revise the Dataset-v3 optimizer membership/serialization under an explicit
  no-leakage policy rather than silently dropping or truncating the six examples
  above 65,536 tokens;
- authorize and measure a larger project-controlled local GPU lane; or
- specify a validated long-sequence execution mechanism that bounds the
  Gated DeltaNet/LoRA projection without changing the objective.

During an aggregate preflight diagnostic, code read role/boundary/form metadata
for the frozen test records. It did not generate, verify, or inspect per-task
test outcomes and did not influence any hyperparameter or checkpoint decision.
An amended contract should explicitly state whether this metadata-only access
affects the seal; this repository does not silently claim it never occurred.
