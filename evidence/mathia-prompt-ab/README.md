# Qwen3.5-4B-Base Mathia prompt A/B

This directory is the compact, repository-owned evidence for issue #86. The
current checkpoint freezes the complete prompt/candidate manifest before any
model generation.

## Prompt-freeze gate

**ACCEPTED:** `execution-manifest.json` binds 611 tasks in authoritative Q0
order: 223 `minif2f-valid-clean-v2` tasks and 388
`fresh-composition-valid-v2` tasks. It defines 8 stable candidate identities
for each task in each arm (4,888 per arm; 9,776 total).

- execution-manifest SHA-256:
  `a878484aed23f127755e85dc03e077aabc1a2b8b8ecd8ca76fe3803a16e15cba`
- generation-config SHA-256:
  `2169b897acbff7a5882780643b56130d19258d18ca4794b68a3db28768a693ff`
- ordered 611-task ID SHA-256:
  `ffee81b7463a8f43de102c9b5ef7a4e8d0fc0cf4a461a33cfe757b119011a3d4`
- frozen instruction SHA-256:
  `2d3d5a28792bf8c0f61740d7dd427a8d251e65093fc0052af0c14839beaac691`

The materializer revalidated the Mathia freeze, all accepted intuition hashes,
the source-projected theorem/context bytes, Dataset-v2 package hashes, and the
unregenerated Q0 evidence from issue #78. Arm A and B use identical task order,
intuition bytes, public context, and declaration bytes. Arm B differs only by
the exact frozen task-instruction bytes in issue #86. Oracle/source proofs,
Q0/Q4/DeepSeek results, and final-test information are not prompt inputs.

## Interruption safety

Generation finalizes one atomic eight-candidate task shard at a time and never
reschedules a valid completed shard. Verification finalizes one result per
candidate, keyed by candidate identity, raw-generation hash, and the frozen
Lean environment selected by workload. MiniF2F uses
`miniF2F@f0a20e14c1eeccd859d51bb4c2b3ee487889c303`; fresh composition uses
`PrimeNumberTheoremAnd@7715064f690d0689f30889846f4e2c5e7ec0c47e`.
Both commands require explicit `--resume`, inventory durable work first, and
schedule only missing identities. The `pause` command creates a marker consumed
between bounded chunks; `unpause` clears it before resumption.

The local inference gate accepts the project RTX 4070 Ti by its Ada compute
capability 8.9. The exact locked runtime and both workload-specific verifier
environments have passed pre-inference identity and representative-preamble
checks.

The vLLM allocation target is 0.89 rather than Q0's 0.95 because the current
WSL display reservation leaves 10.78/11.99 GiB free before engine startup. This
changes only the local cache-allocation ceiling: BF16 weights, 32,768-token
context, request seeds, sampling, prompt bytes, and candidate budget remain
unchanged. The adjustment was made with zero durable candidates after the 0.95
engine preflight stopped before model loading. At 0.89 the exact engine then
initialized successfully under a pre-set `PAUSE` marker with a 47,489-token KV
cache and zero `llm.generate` calls, model outputs, or durable candidates. vLLM's
normal internal dummy/profile forward during engine initialization remained
local and is not a scientific candidate.

Raw generations and verifier diagnostics remain outside Git under `artifacts/`.
The final compact evidence will bind every shard/result hash and process-session
history. No candidate has been generated at this checkpoint.
