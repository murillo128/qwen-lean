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
  `45cb6b5ad1faabebe5ae3e4e32bc6ceae99e895d5c687380620d09f391147fca`
- generation-config SHA-256:
  `cd9e58cff999923143a91fe0d790750a036d0a37a51fb85095f1588c6f26b496`
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
candidate, keyed by candidate identity, raw-generation hash, and frozen Lean
environment. Both commands require explicit `--resume`, inventory durable
work first, and schedule only missing identities. The `pause` command creates a
marker consumed between bounded chunks; `unpause` clears it before resumption.

Raw generations and verifier diagnostics remain outside Git under `artifacts/`.
The final compact evidence will bind every shard/result hash and process-session
history. No candidate has been generated at this checkpoint.
