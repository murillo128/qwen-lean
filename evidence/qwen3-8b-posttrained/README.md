# Qwen3-8B official post-trained strict Lean casting assessment

**OBSERVED:** `Qwen/Qwen3-8B` completed all 244 miniF2F validation
tasks and 976 raw candidates. It verified
4 candidates across
4 tasks. pass@1/pass@4 were
0.004098/0.016393, compared with 0.012807/0.047717 for the accepted unchanged
`Qwen/Qwen3-8B-Base` anchor and 0.039447/0.103162 for `reference-sft-v1`.

The dev16 preflight completed 64 candidates with
pass@1/pass@4 of 0.000000/0.000000. Both accepted runs contain zero unresolved
generation/verifier infrastructure errors. The full run retained
0 `verifier_timeout` outcomes as unsuccessful
proofs. Compute per solved task was 868.56 run-wall seconds.

The dev16 continuations were generated once and then reverified unchanged after
the original run exposed a shared post-generation preamble-probe timeout. Its
generation-identity digest is `e498d2724a64b8021024d724468c69ff04304f49ae6c330eef38754f37492f2e`; accepted
reverification produced 64 Lean rejections and zero task-level timeouts.

**ACCEPTED:** the score uses exact `whole-proof-v1` raw continuation, four
candidates per task, temperature 0.8, top-p 0.95, no top-k, 1,024 new tokens,
seed 0, no chat template, no proof extraction, no verifier feedback, and no
repair. It ran in BF16 without quantization using local vLLM
`0.10.2` on NVIDIA RTX 4000 Ada Generation. FlashInfer was
not installed, so vLLM automatically used its PyTorch-native sampling fallback;
the frozen temperature/top-p/no-top-k contract was unchanged.

The model and tokenizer are pinned to `b968826d9c46dd6066d109eabc6255188de91218`. No optional native/chat
diagnostic was run. The official model is Apache-2.0; weights, caches, and raw
candidates remain outside Git. Compact JSON retains execution identity, counts,
category and finish-reason breakdowns, token/latency summaries, wall times,
throughput, and compute per solved task.

`comparison.json` records the finite-budget caveat: this issue mandates four
candidates per task, while both accepted anchors used eight. The anchors were
read from accepted repository evidence and were not rerun.
