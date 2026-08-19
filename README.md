# qwen-lean

`qwen-lean` is a learning-oriented project for building an end-to-end LLM post-training workflow around Lean 4 theorem proving.

The first concrete experiment is deliberately narrow: take an open-weight Qwen base model, fine-tune it to generate complete Lean proofs, and measure improvement with Lean itself as the verifier. The project will start with supervised fine-tuning (SFT) using QLoRA, then add verifier-filtered self-training and only later consider reinforcement learning or tactic-level proof search.

## Project question

Can post-training make a relatively small general-purpose base model materially better at generating valid Lean proofs, and can we measure that improvement with a trustworthy, inexpensive evaluation loop?

A generated proof counts as successful only when the declared Lean environment accepts it. Training loss, code-like syntax, or a plausible-looking proof are not task success.

## How the project is operated

This repository is designed for agent-driven execution.

- The user steers goals, trade-offs, and learning through ChatGPT and Codex.
- Codex implements repository changes and runs experiments on configured compute.
- Python is an implementation language for the ML tooling, not a required user interface.
- Scripts and configuration should be easy for agents to inspect, change, execute, and compare; a polished human-facing CLI is not a project goal.

The intended high-level flow is:

```text
Lean/mathlib data
      |
      v
training dataset ---> Qwen base model ---> SFT / QLoRA ---> fine-tuned model
                                                        |
                                                        v
                                                    inference
                                                        |
                                                        v
                                                 candidate proofs
                                                        |
                                                        v
                                                  Lean verifier
                                                        |
                                                        v
                                                   pass@k metrics
```

## Where project information lives

Information has one primary home to avoid documentation drift:

- [`PLAN.md`](PLAN.md) defines the technical sequence and exit gates.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) records durable accepted, open, rejected, and observed technical decisions and their rationale.
- [`docs/CONCEPTUAL_MATH_DIRECTION.md`](docs/CONCEPTUAL_MATH_DIRECTION.md) preserves an exploratory long-term research direction, including its philosophical motivation, for higher-level conceptual mathematical training before Lean formalization; it is not an accepted decision or scheduled roadmap phase.
- [Epic #1](/murillo128/qwen-lean/issues/1) tracks current roadmap status and links active execution issues; it does not duplicate the technical plan.
- Individual GitHub issues are bounded execution contracts for one phase or change.
- [`AGENTS.md`](AGENTS.md) and `.agents/skills/` define how ChatGPT, Codex, and reviewers should operate in the repository.

Chat discussion is exploratory until a durable decision is recorded in `docs/DECISIONS.md` or a phase contract is recorded in an issue.

## Initial technical direction

The first cycle uses whole-proof generation with `Qwen/Qwen3-8B-Base`, Lean 4/mathlib verification, QLoRA-based SFT, and a Python ML stack built around PyTorch, Transformers, TRL, PEFT, and Hugging Face Datasets. vLLM is the intended batch inference runtime once evaluation moves beyond tiny smoke tests.

See the decision log and plan for which prompt, dataset, training, and later-stage
choices are accepted versus still open.

## Phase 0 evaluator

The Phase 0 runtime implements the `whole-proof-v1` code-completion contract and
checks every candidate in an isolated process using Lean 4/mathlib `v4.32.0`.
Lean's `hasSorry` diagnostic is promoted to an error so `sorry` and `admit`
cannot be reported as verified proofs without rejecting unrelated warnings.

After installing `uv` and `elan`, prepare the pinned Lean dependency and run the
normal test suite:

```bash
lake update
lake exe cache get
uv sync --frozen
uv run pytest
```

Evaluate the deterministic fixture set and write `run.json` plus
`results.jsonl`:

```bash
uv run qwen-lean fixture --output-dir artifacts/fixture
```

On the project-controlled local Ada GPU, run the separate direct-Transformers
smoke path. Hugging Face Hub may supply the model artifacts, but generation
executes in this local CUDA process:

```bash
uv sync --frozen --extra model
uv run --extra model qwen-lean model-smoke --output-dir artifacts/model-smoke
```

The smoke is a plumbing check, not a model-quality baseline. Its continuation is
sent unmodified, aside from transport whitespace normalization, through the same
Lean verifier used by fixture candidates.

## Phase 1 miniF2F baseline

Phase 1 evaluates the pinned Google DeepMind miniF2F validation workload in the
benchmark's own Lean 4.27.0 Lake environment. The committed configuration fixes
the 244-task primary manifest, deterministic dev16 workload, Qwen model revision,
eight-sample profile, and local-vLLM engine settings. Derived benchmark identifiers
retain the upstream Apache-2.0 attribution in the manifest and configuration.

Prepare the external benchmark outside this Git checkout and its pinned Lake cache:

```bash
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/miniF2F.git /tmp/qwen-lean-minif2f
git -C /tmp/qwen-lean-minif2f fetch --depth=1 origin f0a20e14c1eeccd859d51bb4c2b3ee487889c303
git -C /tmp/qwen-lean-minif2f checkout --detach f0a20e14c1eeccd859d51bb4c2b3ee487889c303
(cd /tmp/qwen-lean-minif2f && lake exe cache get)
(cd /tmp/qwen-lean-minif2f && lake build MiniF2F.ProblemImports)
uv run qwen-lean minif2f-validate --benchmark-root /tmp/qwen-lean-minif2f
```

On the project Ada GPU, install the frozen baseline/model extras, run dev16 first,
then run the complete validation baseline:

```bash
uv sync --frozen --extra baseline --extra model
uv run --extra baseline --extra model qwen-lean phase1-baseline \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-dev16-v1 \
  --output-dir artifacts/phase1-dev16
uv run --extra baseline --extra model qwen-lean phase1-baseline \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-v1 \
  --output-dir artifacts/phase1-full
```

Each model run writes versioned `run.json`, raw `results.jsonl`, and `summary.json`.
Only compact accepted baseline evidence belongs under `evidence/`; raw continuations
and external benchmark/model caches remain local and ignored.

## Qwen3.5-2B post-trained strict casting assessment

The Qwen3.5 assessment keeps its newer vLLM stack isolated from the accepted
Phase 1 environment. Prepare the locked runtime, run the real BF16 preflight,
then run dev16 before the frozen 244-task, four-candidate validation workload:

```bash
uv sync --frozen --project tools/qwen35-assessment
uv run --frozen --project tools/qwen35-assessment qwen-lean qwen35-2b-preflight \
  --benchmark-root /tmp/qwen-lean-minif2f
uv run --frozen --project tools/qwen35-assessment qwen-lean qwen35-2b-assess \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-dev16-v1 \
  --output-dir artifacts/qwen35-2b/dev16
uv run --frozen --project tools/qwen35-assessment qwen-lean qwen35-2b-assess \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-v1 \
  --output-dir artifacts/qwen35-2b/full
uv run --frozen --project tools/qwen35-assessment qwen-lean qwen35-2b-evidence
```

Raw candidates, model files, and caches remain outside Git. The evidence command
retains only compact aggregate results and a hash binding them to the local raw
candidate manifest.

### Qwen3.5-2B-Base assessment

The independent Qwen3.5 casting assessment uses an isolated locked vLLM 0.17.0
runtime because the original Phase 1 stack predates the Qwen3.5 architecture.
It keeps the same pinned miniF2F checkout, raw `whole-proof-v1` continuation,
and Lean verifier while disabling image and video inputs at the engine boundary.
Run the compatibility/memory preflight, dev16 gate, and full 244-by-4 assessment
in that order on the project Ada GPU:

```bash
uv sync --frozen --project tools/qwen35-eval
PYTHONPATH=src uv run --frozen --project tools/qwen35-eval python -m qwen_lean qwen35-preflight
PYTHONPATH=src uv run --frozen --project tools/qwen35-eval python -m qwen_lean qwen35-assess \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-dev16-v1 \
  --output-dir artifacts/qwen35-2b-base/dev16
PYTHONPATH=src uv run --frozen --project tools/qwen35-eval python -m qwen_lean qwen35-assess \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-v1 \
  --output-dir artifacts/qwen35-2b-base/full
PYTHONPATH=src uv run --frozen --project tools/qwen35-eval python -m qwen_lean qwen35-evidence
```

The compact evidence records exact revisions and runtime versions, peak GPU
memory, pass@1/pass@4, all evaluator categories and finish reasons, generated
tokens, timing, throughput, and compute per solved task when defined. Raw
candidates, weights, caches, and logs remain ignored under `artifacts/`.

## OLMo 3 7B Base strict casting assessment

OLMo 3 requires a newer inference stack than the Phase 1 baseline. Its locked
vLLM 0.12.0 / Transformers 4.57.3 runtime remains isolated while reusing the
same pinned miniF2F checkout, raw `whole-proof-v1` continuation, and Lean
verifier. Run the BF16 compatibility/memory preflight, dev16 smoke, complete
244-by-4 assessment, and compact-evidence writer in order on the project Ada GPU:

```bash
uv sync --frozen --project tools/olmo3-assessment
uv run --frozen --project tools/olmo3-assessment qwen-lean olmo3-preflight
uv run --frozen --project tools/olmo3-assessment qwen-lean olmo3-assess \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-dev16-v1 \
  --output-dir artifacts/olmo3-7b/dev16
uv run --frozen --project tools/olmo3-assessment qwen-lean olmo3-assess \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --workload minif2f-valid-v1 \
  --output-dir artifacts/olmo3-7b/full
uv run --frozen --project tools/olmo3-assessment qwen-lean olmo3-evidence
```

Only compact evidence is committed. Weights, caches, raw candidates, and bulky
logs remain outside Git under the ignored `artifacts/` namespace.
## Goedel-Prover-V2-8B strict casting assessment

Issue #30 evaluates the immutable Goedel-Prover-V2-8B checkpoint under the same
raw `whole-proof-v1` miniF2F validation contract. Download the pinned snapshot,
validate it and the local Lean/GPU environment, then run dev16 before the full
244-task workload:

```bash
hf download Goedel-LM/Goedel-Prover-V2-8B \
  --revision dfd02e6271a58375dfbf3ece0175277cf6b6a89a

uv run --extra baseline qwen-lean goedel-preflight \
  --benchmark-root /path/to/built/pinned/miniF2F \
  --model-snapshot /path/to/huggingface/snapshots/dfd02e6271a58375dfbf3ece0175277cf6b6a89a

uv run --extra baseline qwen-lean goedel-assess \
  --benchmark-root /path/to/built/pinned/miniF2F \
  --workload minif2f-valid-dev16-v1 \
  --output-dir artifacts/goedel-prover-v2/dev16

uv run --extra baseline qwen-lean goedel-assess \
  --benchmark-root /path/to/built/pinned/miniF2F \
  --workload minif2f-valid-v1 \
  --output-dir artifacts/goedel-prover-v2/full

uv run qwen-lean goedel-evidence
```

The assessed lane uses four sampled candidates per task at temperature 0.8,
top-p 0.95, no top-k, and 1,024 new tokens. It does not apply the model-card chat
prompt, extract a proof from prose, retry generation, or expose Lean feedback to
the model. The optional native-prover/self-correction diagnostic is separate and
is not part of the strict score.

## Phase 2 verified mathlib corpus

Phase 2 uses LeanDojo-v2 at the revision pinned in
`config/phase2-mathlib.json` to trace the matching mathlib v4.32.0 source. Its
broad dependencies live in the separate `tools/phase2-extractor` uv project and
cannot change the evaluator/model environment. The pipeline retains original
source identity and spans, screens exact miniF2F statement overlap, creates
file/duplicate-component 90/5/5 splits, and records Qwen tokenizer lengths
without imposing a training cutoff.

With the pinned miniF2F checkout prepared as above, run the deterministic pilot
before the full corpus:

```bash
uv run --frozen --project tools/phase2-extractor \
  python tools/phase2_extract.py \
  --mini-root /tmp/qwen-lean-minif2f \
  --output-dir artifacts/phase2/pilot \
  --pilot --verify --loader-smoke
```

Then build and verify the full local corpus and write only compact evidence into
Git:

```bash
uv run --frozen --project tools/phase2-extractor \
  python tools/phase2_extract.py \
  --mini-root /tmp/qwen-lean-minif2f \
  --output-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --verify --loader-smoke \
  --evidence-dir evidence/phase2
```

The generated `train.jsonl`, `validation.jsonl`, `heldout.jsonl`, trace, source
checkout, and tokenizer cache stay under ignored `artifacts/`. A later training
phase can load the local files without LeanDojo or source parsing:

```bash
uv run --frozen --extra phase2 qwen-lean phase2-loader-smoke \
  --artifact-dir artifacts/phase2/mathlib-whole-proof-v1
```

Phase 2 does not publish the corpus or train a model.

## Phase 3 QLoRA sanity path

Phase 3 materializes `phase3-overfit64-v1` from the Phase 2 `train` split using
the pinned Qwen tokenizer, then trains the target-only `mathlib-sft-v1`
serialization. The training extra pins TRL, PEFT, and bitsandbytes alongside the
unchanged model stack. Workload data, trainer state, and adapter weights remain
under ignored `artifacts/`.

Materialize the fixed workload and run the real-GPU preflight before training:

```bash
uv run --frozen --extra training qwen-lean phase3-materialize \
  --artifact-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --output artifacts/phase3/workload.json
uv run --frozen --extra training qwen-lean phase3-preflight \
  --workload artifacts/phase3/workload.json \
  --output artifacts/phase3/preflight.json
uv run --frozen --extra training qwen-lean phase3-train \
  --workload artifacts/phase3/workload.json \
  --output-dir artifacts/phase3/training-amended \
  --target-step 100
```

Each eligible 100-step checkpoint must pass the required local-vLLM memorization
gate in a fresh process. If it fails, resume the same optimizer trajectory exactly
one boundary at a time; do not warm-start an adapter with a reset optimizer:

```bash
uv run --frozen --extra baseline qwen-lean phase3-memorization \
  --workload artifacts/phase3/workload.json \
  --adapter-dir artifacts/phase3/training-amended/trainer-state/checkpoint-100 \
  --optimizer-step 100 \
  --output artifacts/phase3/memorization-amended/step-100.json
uv run --frozen --extra training qwen-lean phase3-train \
  --workload artifacts/phase3/workload.json \
  --output-dir artifacts/phase3/training-amended \
  --target-step 200 \
  --resume-from-checkpoint \
    artifacts/phase3/training-amended/trainer-state/checkpoint-100
```

The superseding Phase 3 gate accepts the existing step-600 checkpoint when its
full-set teacher-forced CE is at most `0.05`, target-token accuracy is at least
`99.5%`, and fresh BF16 vLLM produces at least 48/64 exact continuations with no
generation infrastructure errors. Verify those same raw continuations in their
pinned original mathlib source contexts before running the miniF2F smoke:

```bash
uv run --frozen --extra training qwen-lean phase3-adapter-reload \
  --workload artifacts/phase3/workload.json \
  --adapter-dir artifacts/phase3/training-amended/trainer-state/checkpoint-600 \
  --output artifacts/phase3/adapter-reload-amended.json
uv run --frozen qwen-lean phase3-semantic-verify \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --mathlib-root .lake/packages/mathlib \
  --memorization artifacts/phase3/memorization-amended/step-600.json \
  --training artifacts/phase3/training-amended/run.json \
  --output artifacts/phase3/semantic-verification-step-600.json
uv run --frozen --extra baseline qwen-lean phase3-adapter-smoke \
  --benchmark-root /tmp/qwen-lean-minif2f \
  --adapter-dir artifacts/phase3/training-amended/trainer-state/checkpoint-600 \
  --output-dir artifacts/phase3/minif2f-smoke-step-600
```

The semantic gate attempts all 64 candidates and requires at least 48 Lean
acceptances with zero verifier infrastructure errors and unresolved timeouts. It
uses only transport normalization and inserts each raw continuation directly
after `by\n  `; it does not extract or repair generated text. The miniF2F smoke is
a plumbing gate, so zero verified proofs is allowed when all 16 candidates are
generated and checked without infrastructure failures.

Compact evidence can be generated after the required local artifacts exist:

```bash
uv run qwen-lean phase3-evidence \
  --artifact-dir artifacts/phase3 \
  --evidence-dir evidence/phase3
```

## Phase 4 realistic smoke experiment

Phase 4 deterministically selects 4,096 training, 512 validation, and 64
heldout examples from the Phase 2 corpus. It trains the fixed 512-step QLoRA
trajectory in two processes, with a mandatory full-state stop and resume at
step 256:

```bash
uv run --frozen --extra training qwen-lean phase4-materialize \
  --artifact-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --output artifacts/phase4/workloads.json
uv run --frozen --extra training qwen-lean phase4-preflight \
  --workload artifacts/phase4/workloads.json \
  --output artifacts/phase4/preflight.json
uv run --frozen --extra training qwen-lean phase4-train \
  --workload artifacts/phase4/workloads.json \
  --output-dir artifacts/phase4/training
uv run --frozen --extra training qwen-lean phase4-train \
  --workload artifacts/phase4/workloads.json \
  --output-dir artifacts/phase4/training \
  --resume-from-checkpoint \
    artifacts/phase4/training/trainer-state/checkpoint-256
```

Selection uses only validation target-token cross-entropy at steps 128, 256,
384, and 512. Reload the selected standard PEFT checkpoint in a fresh process,
then evaluate the base model and selected adapter on identical heldout requests:

```bash
uv run --frozen --extra training qwen-lean phase4-adapter-reload \
  --workload artifacts/phase4/workloads.json \
  --training artifacts/phase4/training/run.json \
  --adapter-dir artifacts/phase4/training/trainer-state/checkpoint-512 \
  --output artifacts/phase4/adapter-reload.json
uv run --frozen --extra baseline qwen-lean phase4-heldout \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --mathlib-root artifacts/phase2/leandojo-trace/mathlib4 \
  --workload artifacts/phase4/workloads.json \
  --training artifacts/phase4/training/run.json \
  --mode base --output-dir artifacts/phase4/heldout/base
uv run --frozen --extra baseline qwen-lean phase4-heldout \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --mathlib-root artifacts/phase2/leandojo-trace/mathlib4 \
  --workload artifacts/phase4/workloads.json \
  --training artifacts/phase4/training/run.json \
  --mode adapter \
  --adapter-dir artifacts/phase4/training/trainer-state/checkpoint-512 \
  --output-dir artifacts/phase4/heldout/adapter
uv run --frozen qwen-lean phase4-heldout-compare \
  --training artifacts/phase4/training/run.json \
  --base-dir artifacts/phase4/heldout/base \
  --adapter-dir artifacts/phase4/heldout/adapter \
  --output artifacts/phase4/heldout-comparison.json
```

Finally, evaluate the selected adapter in the built, pinned Phase 1 miniF2F
environment and write compact review evidence. Raw candidates, checkpoints,
weights, and workload rows remain under ignored `artifacts/`:

```bash
uv run --frozen --extra baseline qwen-lean phase4-minif2f \
  --benchmark-root /path/to/built/pinned/miniF2F \
  --training artifacts/phase4/training/run.json \
  --adapter-dir artifacts/phase4/training/trainer-state/checkpoint-512 \
  --output-dir artifacts/phase4/minif2f
uv run --frozen qwen-lean phase4-evidence \
  --artifact-dir artifacts/phase4 \
  --evidence-dir evidence/phase4
```

## Phase 5 full-corpus SFT experiment

Phase 5 retains every Phase 2 train and validation record whose complete
`mathlib-sft-v1` serialization, including EOS, fits 1,024 tokens. It derives the
one-pass optimizer count, quarter checkpoints, midpoint stop, and 1/32 warmup
from the exact eligible training count. Materialize the workloads and run the
production preflight before starting the two-process trajectory:

The validation-selected step-9962 adapter is published as the standard,
unmerged PEFT LoRA repository
[`murillo2000/qwen3-8b-base-lean-sft-qlora`](https://huggingface.co/murillo2000/qwen3-8b-base-lean-sft-qlora)
at Hub revision
[`5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5`](https://huggingface.co/murillo2000/qwen3-8b-base-lean-sft-qlora/commit/5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5).
It requires `Qwen/Qwen3-8B-Base` revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; neither the base weights nor the
Phase 5 trainer state are stored in this Git repository.

```bash
uv run --frozen --extra training qwen-lean phase5-materialize \
  --artifact-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --output artifacts/phase5/workloads.json
uv run --frozen --extra training qwen-lean phase5-preflight \
  --workload artifacts/phase5/workloads.json \
  --output artifacts/phase5/preflight.json
uv run --frozen --extra training qwen-lean phase5-train \
  --workload artifacts/phase5/workloads.json \
  --output-dir artifacts/phase5/training
uv run --frozen --extra training qwen-lean phase5-train \
  --workload artifacts/phase5/workloads.json \
  --output-dir artifacts/phase5/training \
  --resume-from-checkpoint \
    artifacts/phase5/training/trainer-state/checkpoint-<Q2-step>
```

After validation-only selection, substitute the selected checkpoint reported in
`artifacts/phase5/training/run.json`, reload it, and run both heldout roles on the
same deterministic 512-task workload:

```bash
uv run --frozen --extra training qwen-lean phase5-adapter-reload \
  --workload artifacts/phase5/workloads.json \
  --training artifacts/phase5/training/run.json \
  --adapter-dir artifacts/phase5/training/trainer-state/checkpoint-<selected-step> \
  --output artifacts/phase5/adapter-reload.json
uv run --frozen --extra baseline qwen-lean phase5-heldout \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --mathlib-root artifacts/phase2/leandojo-trace/mathlib4 \
  --workload artifacts/phase5/workloads.json \
  --training artifacts/phase5/training/run.json \
  --mode base --output-dir artifacts/phase5/heldout/base
uv run --frozen --extra baseline qwen-lean phase5-heldout \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --mathlib-root artifacts/phase2/leandojo-trace/mathlib4 \
  --workload artifacts/phase5/workloads.json \
  --training artifacts/phase5/training/run.json \
  --mode adapter \
  --adapter-dir artifacts/phase5/training/trainer-state/checkpoint-<selected-step> \
  --output-dir artifacts/phase5/heldout/adapter
uv run --frozen qwen-lean phase5-heldout-compare \
  --training artifacts/phase5/training/run.json \
  --base-dir artifacts/phase5/heldout/base \
  --adapter-dir artifacts/phase5/heldout/adapter \
  --output artifacts/phase5/heldout-comparison.json
```

Finally, run the selected adapter on all 244 miniF2F validation tasks under the
unchanged Phase 1 contract and write compact evidence. Raw tokenized workloads,
checkpoints, weights, candidates, and logs remain ignored under `artifacts/`:

```bash
uv run --frozen --extra baseline qwen-lean phase5-minif2f \
  --benchmark-root /tmp/qwen-lean-minif2f-f0a20e14 \
  --training artifacts/phase5/training/run.json \
  --adapter-dir artifacts/phase5/training/trainer-state/checkpoint-<selected-step> \
  --output-dir artifacts/phase5/minif2f
uv run --frozen qwen-lean phase5-evidence \
  --artifact-dir artifacts/phase5 \
  --evidence-dir evidence/phase5
```

## Phase 6 comparative evaluation

Phase 6 performs no training. It first binds `reference-sft-v1` to the immutable
published step-9962 adapter, reconstructs the exact Phase 5 training membership,
and freezes the deterministic 512-task diagnostic before any new generation:

```bash
uv run --frozen --extra training qwen-lean phase6-freeze \
  --adapter-dir /path/to/hub/snapshot/5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5 \
  --output artifacts/phase6/candidate.json
uv run --frozen --extra training qwen-lean phase6-materialize \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --output artifacts/phase6/train-workload.json
uv run --frozen qwen-lean phase6-checkpoint-a-evidence \
  --candidate artifacts/phase6/candidate.json \
  --train-workload artifacts/phase6/train-workload.json \
  --benchmark-root /path/to/built/pinned/miniF2F \
  --evidence-dir evidence/phase6
```

After Checkpoint A passes independent review, run both local-GPU train-diagnostic
arms and both complete miniF2F-test arms. The base commands still require the
frozen adapter directory so they can validate the pre-generation candidate
manifest; they do not enable that adapter.

```bash
uv run --frozen --extra baseline qwen-lean phase6-train \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --mathlib-root artifacts/phase2/leandojo-trace/mathlib4 \
  --workload artifacts/phase6/train-workload.json \
  --candidate artifacts/phase6/candidate.json \
  --adapter-dir /path/to/hub/snapshot/5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5 \
  --mode base --output-dir artifacts/phase6/train/base
uv run --frozen --extra baseline qwen-lean phase6-train \
  --dataset-dir artifacts/phase2/mathlib-whole-proof-v1 \
  --mathlib-root artifacts/phase2/leandojo-trace/mathlib4 \
  --workload artifacts/phase6/train-workload.json \
  --candidate artifacts/phase6/candidate.json \
  --adapter-dir /path/to/hub/snapshot/5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5 \
  --mode adapter --output-dir artifacts/phase6/train/adapter
uv run --frozen --extra baseline qwen-lean phase6-minif2f-test \
  --benchmark-root /path/to/built/pinned/miniF2F \
  --candidate artifacts/phase6/candidate.json \
  --adapter-dir /path/to/hub/snapshot/5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5 \
  --mode base --output-dir artifacts/phase6/minif2f-test/base
uv run --frozen --extra baseline qwen-lean phase6-minif2f-test \
  --benchmark-root /path/to/built/pinned/miniF2F \
  --candidate artifacts/phase6/candidate.json \
  --adapter-dir /path/to/hub/snapshot/5a5fadc8ecfd46b31c7c6c2f3b8c00f1bcea6af5 \
  --mode adapter --output-dir artifacts/phase6/minif2f-test/adapter
uv run --frozen qwen-lean phase6-evidence --evidence-dir evidence/phase6
```

Raw continuations and bulky run artifacts remain ignored under
`artifacts/phase6/`. Compact evidence keeps exact-target and Lean success
separate and includes deterministic seed-0 task-bootstrap intervals.

## Status

The roadmap epic owns the current phase and links its controlling execution issue.
