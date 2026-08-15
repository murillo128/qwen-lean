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

## Status

The roadmap epic owns the current phase and links its controlling execution issue.
