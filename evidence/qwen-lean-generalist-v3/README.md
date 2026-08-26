# qwen-lean generalist v3 Stage-0 feasibility

`OBSERVED`: the exact merged Dataset-v3 package binds successfully and contains
178,448 training theorems and 317,554 optimizer examples. Under the pinned
Qwen3.5 tokenizer, 317,265 examples fit within 4,096 tokens, but the longest
whole-proof example contains 157,034 tokens. The smallest declared supported
context that preserves every example without truncation or dropping is 262,144.

The original all-example contract was returned to design because its required
no-update near-maximum forward/backward preflight failed
on the project RTX 4000 Ada before the first decoder layer completed. The
Gated DeltaNet `in_proj_qkv` LoRA path requested one 4.79 GiB allocation with
only 4.69 GiB free. No optimizer was created, no adapter update occurred, and
no Base or trained-model canary generation was started.

Issue #83 then authorized an explicit training-only 65,536-token execution view.
`OBSERVED`: exact enumeration under the pinned tokenizer quarantines the declared
six examples, all from two `deep` theorems. Both theorems have no remaining
eligible examples and are explicitly excluded. The resulting view contains
317,548 examples from 178,446 theorems; Dataset v3 itself is unchanged. The
longest remaining eligible example has 60,624 tokens. Compact exact counts and
hashes are in `stage0-feasibility-65k.json`; the full local execution-view
manifest remains under the ignored artifact tree.

`BLOCKED`: the amended no-update preflight also failed on the project RTX 4000
Ada, this time in decoder layer 0 inside the FLA Gated DeltaNet chunk output
allocation. It requested 948 MiB with about 588 MiB free. Forward did not
complete, so backward did not start. No optimizer was created, no update
occurred, and canary generation did not start.

This remains a contract-feasibility failure, not a model-quality result. The
amendment explicitly requires a design return if the 65,536-token bound fails;
execution therefore did not silently lower the cap, switch GPU, change LoRA
targets, or add a new sequence mechanism.

The earlier design alternatives were:

- revise the Dataset-v3 optimizer membership/serialization under an explicit
  no-leakage policy rather than silently dropping or truncating the six examples
  above 65,536 tokens;
- authorize and measure a larger project-controlled local GPU lane; or
- specify a validated long-sequence execution mechanism that bounds the
  Gated DeltaNet/LoRA projection without changing the objective.

During the original aggregate preflight diagnostic, code read role/boundary/form metadata
for the frozen test records. It did not generate, verify, or inspect per-task
test outcomes and did not influence any hyperparameter or checkpoint decision.
The feasibility amendment classified this metadata-only access as seal-safe.
No model generation, verification, per-task test outcome inspection, or
test-driven selection occurred, so the sealed-test quality contract remains
intact.
