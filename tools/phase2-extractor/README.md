# Phase 2 extraction environment

This subproject isolates LeanDojo-v2 and its broad dependency graph from the
qwen-lean evaluator/model environment. Its direct dependency is the exact
LeanDojo-v2 Git revision required by Phase 2. The root project is imported only
as local extraction/splitting code; LeanDojo's trainer, prover, database, and
hosted inference paths are not used.

The trace runner uses LeanDojo-v2's repository tracer, then reads the resulting
AST files one at a time through `TracedFile`/`TracedTheorem`. This avoids the
public wrapper's optional all-files-in-memory materialization without changing
the traced source, theorem spans, proof extraction, or premise resolution.

Run `uv lock --project tools/phase2-extractor` after an intentional dependency
change. Normal corpus builds use `uv run --frozen --project
tools/phase2-extractor python tools/phase2_extract.py ...` as documented in the
root README.
