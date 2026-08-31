# qwen-lean generalist v3 execution evidence

`OBSERVED`: the final issue-authorized 16,384-token Ada execution view and
unchanged Base validation canary both pass. Compact evidence is in
`stage0-16k.json` and `base-validation-canary.json`; bulky manifests, cached
full-vocabulary logits, streams, and candidate-level outputs remain under the
ignored `artifacts/qwen-lean-generalist-v3/` tree.

The final independent tokenizer census and execution-view enumeration agree on
exactly 51 quarantined examples (34 continuation, 17 whole), all from 17
`deep` theorems. Each affected theorem has no remaining optimizer-eligible row
and is explicitly excluded. The execution view retains 317,503 examples from
178,431 theorems without mutating Dataset v3, truncating a row, or silently
dropping evidence. Its longest eligible example is 15,617 tokens.

The clean-GPU gate observed no competing compute process and 97.76% free VRAM
before model load. The exact NF4 QLoRA/BF16 forward, finite target-only loss,
backward, and all-LoRA-gradient checks passed on the longest retained example.
Peak CUDA allocated memory was 15,415,834,624 bytes, leaving 5,573,969,920
bytes of allocated-memory headroom, above the required 1 GiB. No optimizer was
created or updated.

The unchanged pinned Base was then evaluated on all 48 validation compositions
under both whole and incremental interfaces (`96 × 8 = 768` candidates), with
the native 262,144-token model context and no truncation. Ten interface tasks
exceed the 16K training-only ceiling; the longest validation input contains
242,631 tokens and was evaluated intact. Base solved 0/48 whole and 0/48
incremental tasks; all 768 results were ordinary candidate outcomes (748 Lean
rejections and 20 empty candidates), with no generation or verifier
infrastructure errors. Despite zero verified proofs, Base retained high output
diversity: 322 whole and 381 incremental normalized complete-output templates.
This is the frozen step-0 comparator, not a successful model-quality result.

The 64,000-microbatch deterministic training stream now has stable canonical
and gzip byte hashes across repeated complete writes. The frozen 512-anchor
manifest contains no validation/test theorem, and the exact full-vocabulary
Base next-token logits are cached before any optimizer update.

The mandatory PEFT-to-vLLM parity gate also passed before optimizer creation.
HF forward checks changed on all four deterministic probes. The vLLM 0.17.0
worker independently verified all 496 source tensors and the exact semantic
Q/K/V split used to derive 592 runtime tensors: 248 trained PEFT modules map to
296 runtime adapter children and exactly 152 fused runtime wrappers, with no
missing or unexpected module. The adapter changed generated text on two probes;
all four probes changed either text or first-token probability. This sentinel
establishes transport activity and completeness only, not model quality.

## Bounded SFT trajectory and stop decision

`OBSERVED`: all four issue-authorized bounded configurations completed exactly
500 optimizer updates on the same frozen stream, with retained evaluations at
steps 100, 250, and 500. No optimizer update beyond step 500 occurred. The
machine-readable Base-plus-12 trajectory is in `bounded-trajectory.json`; the
validation and optimizer trajectories are plotted in
`bounded-validation-trajectory.svg` and `bounded-training-trajectory.svg`.
The complete deterministic plot package also includes separate whole and
incremental coverage, verified density, EOS, median/p75/p90 length, diversity,
Base-retention, first-construct, structural-capability, and anchor-drift SVGs.
The aggregate JSON retains the 637 denominator-bearing validation metric rows,
all 2,000 raw per-step objective rows, and every normalized template's theorem,
occurrence, and verification counts needed to regenerate and audit those views.

| Configuration | LR | Base KL weight | Step 100 solved / verified | Step 250 solved / verified | Step 500 solved / verified | Step 500 mean anchor KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 diagnostic | 3e-5 | 0.0 | 6 / 6 | 4 / 5 | 6 / 8 | 4.9639 |
| C1 | 3e-5 | 0.1 | 2 / 2 | 1 / 1 | 0 / 0 | 0.4561 |
| C2 | 1e-5 | 0.1 | 2 / 2 | 0 / 0 | 0 / 0 | 0.4871 |
| C3 rescue | 1e-5 | 0.3 | 0 / 0 | 2 / 2 | 0 / 0 | 0.3780 |

Every solved and verified count is measured over the same complete 96-task,
768-candidate validation canary. The hard gate evaluates every normalized
template, rather than only the most frequent template. Every retained
checkpoint failed the hard repeated-template gate in the whole-proof lane and
is therefore ineligible,
including the early checkpoints with nonzero Lean verification. C0 was always a
diagnostic-only arm; its verified candidates coincided with much larger direct
Base drift. C1, C2, and C3 show that the cached-logit preservation term
substantially constrained mean anchor KL, but lower direct drift did not
preserve healthy search behavior or produce an eligible checkpoint.

`OBSERVED`: the frozen positive step-500 gate failed for C1 and C2. The single
authorized C3 rescue also failed. The issue stop rule is therefore active:
SFT is stopped, no checkpoint is frozen, no 1k/2k/4k/8k continuation or broader
rescue sweep is authorized, and the sealed Dataset-v3 test remains untouched.
The negative result is not evidence that the preservation mechanism was
inactive; it is evidence that these bounded SFT configurations did not combine
verified Lean learning with non-collapsed generation.

The next experiment belongs in a separate design scope. The controlling issue
calls for verifier-driven training rather than a larger imitation-only retry;
its latest evidence note also identifies a bounded broad-data replay comparison
as a preservation control worth testing. Neither follow-up is part of this run,
and no replay percentage or RL objective is accepted here.

## Historical feasibility returns

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

Issue #83 next authorized a 32,768-token training execution view. `OBSERVED`:
exact enumeration quarantines the declared 18 examples, all from six `deep`
theorems. Each affected theorem loses all three optimizer rows and is explicitly
excluded with its full removed mass recorded. The resulting view contains
317,536 examples from 178,442 theorems, and its longest eligible example has
31,212 tokens.

`BLOCKED`: the exact 31,212-token no-update preflight still failed before the
first decoder layer completed, in the same FLA Gated DeltaNet chunk-output
allocation. It requested 488 MiB with about 342 MiB free. Forward did not
complete; backward and the required 1 GiB allocated-headroom check were not
reached. No optimizer was created and no model update occurred. Exact compact
evidence is in `stage0-feasibility-32k.json`.

These are historical contract-feasibility failures, not model-quality results.
Each controlling amendment required a design return; execution did not silently
lower the cap, switch GPU, change LoRA targets, or add a sequence mechanism.

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
