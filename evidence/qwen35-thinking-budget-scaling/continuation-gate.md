# Qwen3.5-4B thinking-budget continuation gate

**OBSERVED:** `passed` under the pinned `qwen3` canonical parsed-output contract and `lean-wrapper-normalization-v1`.

The 32/64/128-token bounded probes retain raw token IDs, replay the parser deterministically, preserve exact parsed/normalized bytes and hashes, and treat raw-suffix identity only as a diagnostic.

Scientific generation is authorized only when every continuation gate check passes. Historical Stage 1 and Stage 2a artifacts remain unchanged.
