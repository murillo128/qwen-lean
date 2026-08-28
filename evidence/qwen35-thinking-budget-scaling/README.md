# Qwen3.5-4B thinking-budget runtime gate

**OBSERVED:** the exact Stage 1 runtime exposes native `thinking_token_budget` with the frozen `qwen3` parser.

All three probes bounded reasoning and continued into a non-empty final channel. The B32 probe emitted a duplicate reasoning-end marker; the pinned `qwen3` parser absorbed it but returned final content with one additional leading newline, so the final was not an exact raw suffix. This violates the Stage 2a parser-integrity gate.

**OBSERVED conclusion:** `budget_control_runtime_or_parser_not_usable`.

The outcome-bearing 48-generation B4/B8/B16 scaling probe was not started. The durable probe JSONL was not regenerated or rewritten, and the independently reviewed Stage 1 target remains unchanged. A runtime upgrade or custom two-stage forcing mechanism requires an explicit bounded amendment before further scientific generation.
