# qwen-lean generalist v2 — Dataset-v2 binding and preparation

`OBSERVED`: the exact `Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b` loaded through the isolated
Qwen3.5 runtime as the text-only `Qwen3_5ForCausalLM` class. A 49-token raw
whole-proof prefix completed a finite local-CUDA forward pass on the project Ada
GPU. Peak CUDA allocation was 8,547,122,176 bytes.

`OBSERVED`: the pinned text decoder contains 248 intended rank-16 LoRA target
modules: 32 full-attention projections, 120 Gated DeltaNet projections, and 96
MLP projections. Their prospective adapter has 32,464,896 trainable parameters.
The exact module inventory is retained in `pre56-architecture.json`; no vision,
embedding, normalization, or `lm_head` module matched.

`OBSERVED`: the same isolated dependency lock loaded the base through the
automatic NF4 QLoRA fallback on the available 20,989,804,544-byte device,
attached 496 LoRA A/B parameter tensors over all 248 projections, and found no
trainable parameter outside that boundary. Peak allocation for model
quantization plus adapter preparation was 5,662,899,200 bytes. The preferred
BF16-LoRA lane was not selected because this device is below the accepted 48 GiB
minimum.

`ACCEPTED`: the pre-corpus architecture and runtime-preparation checks used no
Dataset-v2 rows and made no production context-fit or model-quality claim. The
post-binding production gate below supplies the required context-fit evidence;
bounded training and evaluation remain separate later gates.

`OBSERVED`: after issue #56 merged, the authoritative Dataset-v2 package bound
without unresolved references or role leakage. `general-train-v2` contains
181,531 statements and 182,812 proof variants. Every optimizer-visible variant
is present exactly once in the declared one-pass membership. The frozen 4x
synthetic multiplier produces 6.8458789% of statement-level training mass; no
domain multiplier is present.

`OBSERVED`: exhaustive serialization with the exact pinned Qwen3.5 tokenizer
selected the 32,768-token context bucket. The maximum full sequence is 19,385
tokens (maximum completion 9,293); no row was truncated, dropped, or packed.
The deterministic fresh-composition views contain 406 validation and 415 test
statements. Their Riemann-filtered subviews contain 100 validation and 104 test
statements with zero training statement or derivation-family overlap.

`OBSERVED`: the complete pre-update Q0 control used eight candidates per task
and recorded zero generation or verifier infrastructure errors. The pinned base
solved 16/406 fresh-composition validation tasks (pass@1/pass@4/pass@8
0.005234/0.020408/0.039409), 66/244 clean miniF2F validation tasks
(0.066086/0.185831/0.270492), 9/256 Dataset-v2 train-probe tasks
(0.005371/0.019810/0.035156), and 2/100 fresh Riemann validation tasks
(0.002500/0.010000/0.020000). The train probe reproduced no stored target
exactly at Q0. `q0.json` retains the ordered task outcomes and raw-artifact
hashes; test workloads were not consulted and the Riemann result remains
diagnostic-only.

`OBSERVED`: the production preflight serialized the exact maximum Dataset-v2
row to 19,385 tokens and completed one real NF4 QLoRA forward, backward, and
optimizer update on the project-controlled RTX 4000 Ada GPU. All 496 intended
LoRA parameter tensors had present finite gradients, 248 tensors changed on the
first update, and the checked frozen base parameter remained unchanged. The
run used non-reentrant decoder checkpointing from 1,024 tokens, 32-token
DeltaNet chunks, 64-token target-only LM-head loss chunks, and checkpointed
1,024-token sequence chunks in all 32 decoder MLPs. Shorter training examples
remain on the faster non-checkpointed path. An explicit BF16 autocast and
PyTorch FlashAttention backend constraint prevent a silent dense FP32 SDPA
fallback.
The 24 DeltaNet layers used the locally executed FLA 0.5.2 training kernel,
pinned to upstream tag commit `9c8e42e762fce087c27b673af4922795d9edb85e`
under its MIT license. Its output and gradients were checked against the
Transformers torch reference before use. Peak CUDA allocation was
12,071,390,720 bytes and peak reservation was 12,744,392,704 bytes, leaving
8,245,411,840 bytes of reserved-memory headroom against the 536,870,912-byte gate.
`production-preflight.json` retains the exact example, parameter inventory,
runtime lock, and measurements.

`OBSERVED`: overfit64 strongly fit the deterministic 64-statement probe over
its frozen 600-step sanity trajectory. Mean loss fell from 39.918397 on the
first complete pass to 0.000578 on the last (ratio 0.00001448). A fresh frozen
adapter reload reproduced all four generation probes exactly; all four were
Lean-verified with zero evaluator infrastructure errors.

`OBSERVED`: the first realistic-smoke attempt exposed a host-memory failure,
not a CUDA failure. The kernel killed the process at optimizer step 186 while
CPU-offloaded activations occupied about 14 GiB of shared memory on a host whose
RAM-backed `/tmp` already contained 11.7 GiB of unrelated project state. A
fresh exact 19,385-token maximum-row forward/backward/update with CPU activation
offload disabled passed with finite gradients and an adapter-only update. It
reserved 18,427,674,624 of 20,989,804,544 CUDA bytes, leaving 2,562,129,920
bytes of headroom. Bounded and full training therefore disable CPU activation
offload; `no-offload-production-preflight.json` records the superseding memory
path check without rewriting the truthful gate hash used by the completed
overfit run.

`OBSERVED`: the fresh realistic smoke retry completed one deterministic pass
over 4,096 statements and all 4,128 selected proof variants (516 optimizer
steps). All logged losses and gradient norms were finite; mean loss was
11.126338 and the first/last losses were 19.439495/6.114424. Training reserved
17,593,008,128 of 20,989,804,544 CUDA bytes, leaving 3,396,796,416 bytes of
headroom, and completed in 1,839.964 seconds without CPU activation offload. A
fresh frozen adapter reload generated and submitted the required probe to Lean
with zero evaluator infrastructure errors. The generated proof was rejected,
which is recorded as model behavior rather than an operational gate failure.
The repository-native bounded-training validator accepted `smoke4096.json`;
the run evidence SHA-256 is
`4c3ab699c3ea656835b2b3e5f875b9cec3fe35b0afc31ac5bd69e27c84940a73`
and the retained adapter SHA-256 is
`eec2cc20dacb7ca211d6f59f4ad4c1b2c88f3d8a6b0dc720f397749f58c8e87f`.

`OBSERVED`: the full NF4 QLoRA run completed exactly one pass over all 182,812
optimizer-visible proof variants in 22,852 optimizer steps. All logged losses
and gradient norms were finite; mean loss was 7.148310 and the first/last
losses were 16.811586/2.858912. The run finished in 81,315.080 seconds on the
project Ada GPU without CPU activation offload, reserved at most 18,874,368,000
of 20,989,804,544 CUDA bytes, and retained 2,115,436,544 bytes of headroom.
Complete adapter-only checkpoints with optimizer/scheduler/RNG state were
written at Q1/Q2/Q3/Q4 steps 5,713/11,426/17,139/22,852. `full-training.json`
binds their adapter and trainer-state hashes; the complete outside-Git source
run has SHA-256
`c6175c15f8419c30807d8b4fec3cec338a3e41966ebd6beb96c205046c1f3798`.
This is operational training evidence, not a checkpoint-quality claim; Q1-Q4
were subsequently evaluated under the frozen validation procedure described
below.

`OBSERVED`: the original Q1/Q2 and partial-Q3 vLLM screening was invalid for
model quality because Qwen3.5 PEFT names under `model.layers.*` did not resolve
against vLLM's `language_model.model.layers.*` wrappers. The artifacts remain
preserved as diagnostic-only evidence. The corrected evaluator maps all 496
adapter tensors / 248 PEFT modules into 152 fused or unfused runtime modules
with no omission. Its independent HF/PEFT and vLLM known-positive arms both
reproduced overfit64 at 4/4 exact, and the fixed Q2 smoke changed all 16 output
or forward signals. `lora-inference-parity-corrected.json` records the passed
gate; no invalid pre-fix score was used for selection.

`OBSERVED`: corrected n=8 screening completed all required Q1-Q4 lanes with
zero infrastructure errors and selected Q4 by the frozen validation-only rule.
Q4 solved 74/244 clean miniF2F validation tasks (pass@1/pass@4/pass@8
0.163422/0.256089/0.303279) and 60/406 fresh-composition validation tasks
(0.066810/0.117417/0.147783). Its train probe solved 79/256 (pass@8
0.308594). Tests, domain diagnostics, and train-probe outcomes were not
selection inputs; Q3 was the strongest runner-up.

`OBSERVED`: the frozen Q4 winner then completed the amendment's validation-only
n=64 measurement without early stopping. Across 41,600 retained and verified
candidates, clean miniF2F reached 107/244 solved and pass@64 0.438525, while
fresh composition reached 94/406 and pass@64 0.231527. Base, runner-up, and test
n=64 lanes were not run. The compact artifact binds raw-candidate SHA-256 values
`f10b9cc31d664b2a94aac57bbf7c954c63c0f4998b42e263ccc69ded10ac2e78`
and `4a44b689e481ef7289cacdb03293c689ef01b749908e5d24e3099dcd21627a40`;
the raw gzip records remain outside Git for the required post-hoc capability
analysis.

`ACCEPTED`: issue amendment `#5409570320` removes every remaining Riemann lane
from the completion contract. Evidence produced before that amendment remains
immutable diagnostic history, but it is not extended, regenerated, included in
the final general comparison, or treated as a release gate. The remaining final
assessment is restricted to the n=8 clean miniF2F and fresh-composition test
workloads; no additional n=64 lane is run.

`ACCEPTED`: the later issue amendment `#5415045961` stops the disproportionately
expensive DeepSeek fresh-composition test and removes it as a completion gate.
The process stopped after the log observed 208/3320 completed candidates. The
pinned blocking vLLM call had not returned, so it had materialized no candidate
JSONL to retain or score. `deepseek-fresh-incomplete.json` binds the exact
outside-Git operational log SHA-256 and marks the lane `INCOMPLETE / DIAGNOSTIC
ONLY / NOT FOR MODEL-QUALITY COMPARISON`; no pass@k or extrapolation is present.
All #78 GPU processes, including the older stale process, were terminated and
the Ada was released before CPU analysis and review.

`OBSERVED`: on clean miniF2F test, Base/Q4/DeepSeek solved 87/76/95 of 244 tasks
within eight samples. Their pass@1/pass@4/pass@8 values were respectively
0.105533/0.269848/0.356557, 0.158299/0.267857/0.311475, and
0.154713/0.319555/0.389344. Q4 improved pass@1 over Base, but its pass@8 delta
was -0.045082 with paired-bootstrap 95% interval [-0.094262, 0.004098]. Q4 was
below DeepSeek at pass@8 by -0.077869, interval [-0.147541, -0.008197]. On the
fresh-composition final test, Base solved 6/415 while Q4 solved 0/415; the
pass@8 delta was -0.014458, interval [-0.026506, -0.004819], with exact McNemar
p=0.03125. These final-only results do not retroactively change the frozen
validation selection and no retuning was performed.

`ACCEPTED`: issue amendment `#5415834670` supersedes the prior final technical
review and adds one bounded Q4-specific inference-integrity gate. No benchmark,
training, DeepSeek, or Riemann lane was regenerated. Twelve prior Q4-positive
fresh-validation tasks were frozen deterministically as the four highest-density
tasks in each of the `direct`, `branching`, and `deep` classes under the existing
n=64 evidence.

`OBSERVED`: the exact Q4 adapter hash
`2398be7ac95db85d646bce66762abcea96487a93b7d92508ddcc274914ef470e`
passed the new greedy HF/PEFT↔vLLM canary. HF loaded all 496 tensors with zero
missing, unexpected, or value-mismatched payloads; Q4 changed the next-token
forward signal on 12/12 probes. The corrected vLLM transform preserved the
source payload exactly across all 496 tensors and loaded all 248 PEFT modules
into 152 runtime modules with zero omissions or unexpected modules. Base was
Lean-rejected on 12/12 probes in both backends, while Q4 reproduced the exact
known-valid output and Lean-verified on 12/12 in each backend; HF Q4 and vLLM Q4
matched exactly on all 12. `q4-inference-canary.json` retains every full output,
hash, forward/logprob signal, load audit, and Lean result.

`OBSERVED`: the deterministic 100-candidate diagnosis used only the already
generated Q4 fresh-test failures. All 3,320 stored candidates ended by EOS;
none hit the token limit, lengths were 13–184 tokens (mean 45.57), 85.45% were
at most 64 tokens, and 3,309/3,320 began with the same `exact ⟨…⟩` constructor
mode. After constant-name normalization, the three dominant templates account
for 1,158, 982, and 647 candidates across 145, 136, and 135 unrelated tasks.
The sample's primary categories were 65 structurally wrong templates, 23
unknown/mismatched lemmas, 8 type/elaboration errors, 3 incomplete proofs, and
1 other rejection. Combined with the passed Q4 canary, this classifies the
extreme final result as a genuine narrow-template/reduced-exploration
generalization failure rather than residual adapter or inference pathology.

`OBSERVED`: Stage 9 partitions miniF2F validation into 48 robust, 47
search-sensitive, 12 lottery, and 137 dead-zone tasks; fresh composition has
39/42/13/312 respectively. Verified-proof duplication is high (0.8529 miniF2F,
0.9440 fresh), so the evidence supports both search/diversity opportunity and a
large executor/knowledge dead zone. The strongest sufficiently sized groups
include miniF2F `mathd_numbertheory` and fresh direct/complex/integer tasks;
weak groups include IMO/AIME and existential/iff shapes on miniF2F, plus deep,
category-theory, negation/iff, and geometry groups on fresh composition. On the
accepted same-task miniF2F validation comparator, Q4 and DeepSeek both solved
55 tasks, Q4 alone solved 19, DeepSeek alone solved 78, and neither solved 92.
The analysis authorizes no extra training and recommends independently sourced
skill-matched data rather than recycling failed validation theorems.

Commands:

```text
uv sync --project tools/qwen35-generalist --locked
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-architecture-preflight --config config/qwen35-4b-generalist-v2.json --output evidence/qwen-lean-generalist-v2/pre56-architecture.json
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-runtime-preparation-smoke --config config/qwen35-4b-generalist-v2.json --output evidence/qwen-lean-generalist-v2/pre56-runtime-preparation.json
uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-bind-dataset --config config/qwen35-4b-generalist-v2.json --package-root data/lean-whole-proof-v2 --view-dir artifacts/qwen-lean-generalist-v2/dataset-binding/views --output evidence/qwen-lean-generalist-v2/dataset-binding.json
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-q0-evidence --config config/qwen35-4b-generalist-v2.json --evaluation-root artifacts/qwen-lean-generalist-v2/q0 --output evidence/qwen-lean-generalist-v2/q0.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-production-preflight --config config/qwen35-4b-generalist-v2.json --package-root data/lean-whole-proof-v2 --binding evidence/qwen-lean-generalist-v2/dataset-binding.json --q0-evidence evidence/qwen-lean-generalist-v2/q0.json --model-snapshot /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b --output evidence/qwen-lean-generalist-v2/production-preflight.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 uv run --project tools/qwen35-generalist --locked qwen-lean generalist-v2-train --config config/qwen35-4b-generalist-v2.json --package-root data/lean-whole-proof-v2 --binding evidence/qwen-lean-generalist-v2/dataset-binding.json --q0-evidence evidence/qwen-lean-generalist-v2/q0.json --production-preflight evidence/qwen-lean-generalist-v2/production-preflight.json --overfit-run artifacts/qwen-lean-generalist-v2/overfit64/run.json --smoke-run artifacts/qwen-lean-generalist-v2/smoke4096/run.json --model-snapshot /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b --output-dir artifacts/qwen-lean-generalist-v2/full-training
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-final-evidence --config config/qwen35-4b-generalist-v2.json --q0-evidence evidence/qwen-lean-generalist-v2/q0.json --selection evidence/qwen-lean-generalist-v2/checkpoint-selection.json --final-root /dev/shm/qwen-lean-generalist-v2-final-assessment --deepseek-root /dev/shm/qwen-lean-generalist-v2-deepseek-recovery --deepseek-fresh-incomplete evidence/qwen-lean-generalist-v2/deepseek-fresh-incomplete.json --package-root data/lean-whole-proof-v2 --view-dir artifacts/qwen-lean-generalist-v2/dataset-binding/views --output evidence/qwen-lean-generalist-v2/final-assessment.json
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-refinement-evidence --extended evidence/qwen-lean-generalist-v2/extended-validation.json --final evidence/qwen-lean-generalist-v2/final-assessment.json --q0-evidence evidence/qwen-lean-generalist-v2/q0.json --selection evidence/qwen-lean-generalist-v2/checkpoint-selection.json --deepseek-root /dev/shm/qwen-lean-generalist-v2-deepseek-recovery --extended-root /dev/shm/qwen-lean-generalist-v2-extended-validation --package-root data/lean-whole-proof-v2 --view-dir artifacts/qwen-lean-generalist-v2/dataset-binding/views --output evidence/qwen-lean-generalist-v2/refinement-conclusions.json
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-q4-canary-probes --config config/qwen35-4b-generalist-v2.json --package-root data/lean-whole-proof-v2 --view-dir artifacts/qwen-lean-generalist-v2/dataset-binding/views --extended-evidence evidence/qwen-lean-generalist-v2/extended-validation.json --extended-results /dev/shm/qwen-lean-generalist-v2-extended-validation/Q4/fresh-composition-valid-v2/results.jsonl --extended-generation-metadata /dev/shm/qwen-lean-generalist-v2-extended-validation/Q4/fresh-composition-valid-v2/generation-metadata.json --q4-adapter-dir artifacts/qwen-lean-generalist-v2/full-training/trainer-state/checkpoint-22852 --output /dev/shm/qwen-lean-generalist-v2-q4-canary/probes.json
HF_HUB_OFFLINE=1 PYTHONPATH=src uv run --project tools/qwen35-generalist --locked python -m qwen_lean generalist-v2-q4-canary-hf --config config/qwen35-4b-generalist-v2.json --probes /dev/shm/qwen-lean-generalist-v2-q4-canary/probes.json --q4-adapter-dir artifacts/qwen-lean-generalist-v2/full-training/trainer-state/checkpoint-22852 --model-snapshot /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b --output /dev/shm/qwen-lean-generalist-v2-q4-canary/hf-runtime.json
HF_HUB_OFFLINE=1 PYTHONPATH=src <pinned-vllm-python> -m qwen_lean generalist-v2-q4-canary-vllm --config config/qwen35-4b-generalist-v2.json --base-evaluation-config config/qwen35-4b-base-assessment.json --probes /dev/shm/qwen-lean-generalist-v2-q4-canary/probes.json --q4-adapter-dir artifacts/qwen-lean-generalist-v2/full-training/trainer-state/checkpoint-22852 --output /dev/shm/qwen-lean-generalist-v2-q4-canary/vllm-runtime.json
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-q4-canary-evidence --config config/qwen35-4b-generalist-v2.json --probes /dev/shm/qwen-lean-generalist-v2-q4-canary/probes.json --hf-runtime /dev/shm/qwen-lean-generalist-v2-q4-canary/hf-runtime.json --vllm-runtime /dev/shm/qwen-lean-generalist-v2-q4-canary/vllm-runtime.json --general-lean-project-root <PrimeNumberTheoremAnd-project-root> --raw-verification-output /dev/shm/qwen-lean-generalist-v2-q4-canary/raw-verification.json --output evidence/qwen-lean-generalist-v2/q4-inference-canary.json --workers 8
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-q4-fresh-test-failure-diagnosis --final-evidence evidence/qwen-lean-generalist-v2/final-assessment.json --generations /dev/shm/qwen-lean-generalist-v2-final-assessment/selected/fresh-composition-test-v2/generations.jsonl --results /dev/shm/qwen-lean-generalist-v2-final-assessment/selected/fresh-composition-test-v2/results.jsonl --generation-metadata /dev/shm/qwen-lean-generalist-v2-final-assessment/selected/fresh-composition-test-v2/generation-metadata.json --output evidence/qwen-lean-generalist-v2/q4-fresh-test-failure-diagnosis.json
PYTHONPATH=src .venv/bin/python -m qwen_lean generalist-v2-release-evidence --config config/qwen35-4b-generalist-v2.json --binding evidence/qwen-lean-generalist-v2/dataset-binding.json --training evidence/qwen-lean-generalist-v2/full-training.json --selection evidence/qwen-lean-generalist-v2/checkpoint-selection.json --extended evidence/qwen-lean-generalist-v2/extended-validation.json --final evidence/qwen-lean-generalist-v2/final-assessment.json --refinement evidence/qwen-lean-generalist-v2/refinement-conclusions.json --deepseek-preflight evidence/qwen-lean-generalist-v2/deepseek-final-preflight.json --lora-parity evidence/qwen-lean-generalist-v2/lora-inference-parity-corrected.json --q4-canary evidence/qwen-lean-generalist-v2/q4-inference-canary.json --q4-failure-diagnosis evidence/qwen-lean-generalist-v2/q4-fresh-test-failure-diagnosis.json --output evidence/qwen-lean-generalist-v2/release.json
```
