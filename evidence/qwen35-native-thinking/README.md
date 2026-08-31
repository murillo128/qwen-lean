# Qwen3.5-4B native thinking A/B

**OBSERVED:** both frozen native-chat arms completed all 611 Mathia-guided tasks
and 2,444 candidates per arm. T0 solved 6 tasks
within four candidates; T1 solved 5. Combined
pass@1/pass@4 were 0.004092/0.009820
for T0 and 0.002455/0.008183
for T1. The paired solved@4 delta (T1 minus T0) was
-0.001637; exact two-sided McNemar
p=1.

**ACCEPTED:** model/tokenizer revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Mathia freeze
`frontier_assisted_intuition_corpus_26b3030e14ab4df8694b9c14ab30a297214f33f7a7ebdb8686ccd2ffae37849f`, user-visible prompt bytes, temperature 0.6, top-p 0.95,
top-k 20, four candidates, seed mapping, 4,096-token output budget, BF16,
native Qwen chat template, and Lean verifier semantics were matched. The only
intended variable was `chat_template_kwargs.enable_thinking`. The vLLM `qwen3`
parser separated native reasoning from final content; only exact final-channel
bytes were submitted to Lean, without extraction, sanitization, repair, or
verifier-driven retry.

**OBSERVED:** the deterministic interpretation category is
`t1_hurts_with_token_or_format_behavior`. T1 hit the shared token limit before
usable final content on 2434/2444 candidates
(99.59%);
reasoning-budget exhaustion, rather than final-channel contamination, was the
dominant observed interface failure. Thinking is not compute matched: cost and
token totals for each arm are retained in `results.json`. This result does not
change the external-planner design or authorize a training architecture.
