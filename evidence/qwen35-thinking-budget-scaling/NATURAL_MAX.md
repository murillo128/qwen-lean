# Qwen3.5-4B BNAT-MAX natural-thinking continuation

**OBSERVED:** the exact local machine accepted `max_model_len=262144` under the frozen BF16 runtime and completed all 16 paired BNAT-MAX candidates without a reasoning-token budget.

BNAT-MAX produced non-empty parsed finals for 3/16 candidates and Lean verified 1/16.

Within the immutable 11-candidate B16-forced subset, 11 consumed the available context without final content.

**OBSERVED conclusion:** `natural_thinking_consumes_available_context`.

The complete 16/16 durable candidate records contain 141773.61 seconds of per-candidate generation latency, yielding 24.151 generated tokens per candidate-latency second.

Segment-level wall-clock and peak-VRAM telemetry are incomplete: the retained segment covers 6/16 candidates after the restart. Its observed wall time is 65037.69 seconds and its observed peak is 19665649664 bytes; neither is claimed as complete-run telemetry.

This is a 16-candidate capability check, not evidence of statistical superiority. Historical Stage 1 and B4/B8/B16 artifacts remain unchanged, and no larger program is authorized.
