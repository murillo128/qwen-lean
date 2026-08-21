from __future__ import annotations

from pathlib import Path

from test_generalist_v2_dataset import _record

from qwen_lean.generalist_v2_dataset import generalist_variants
from qwen_lean.generalist_v2_evaluation import (
    _source_position_verification_task,
    _synthetic_task,
)


def test_synthetic_evaluation_uses_persisted_import_context() -> None:
    record = _record()
    task = _synthetic_task(record)

    assert task.id == record.statement_id
    assert task.preamble == "import Mathlib"
    assert task.declaration == record.canonical_declaration
    assert generalist_variants(record)[0].completion == "trivial"


def test_source_position_verifier_preserves_exact_source_prefix(
    tmp_path: Path,
) -> None:
    source = (
        "import Mathlib\n\n"
        "namespace Fixture\n\n"
        "variable (localContext : True)\n\n"
        "theorem fixture : True := by\n"
        "  exact localContext\n"
    )
    source_path = tmp_path / ".lake/packages/mathlib/Fixture.lean"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    value = _record().to_dict()
    value["provenance"] = "real-mathlib"
    value["derivation_family_fingerprint"] = None
    value["generator_family"] = None
    value["structural_class"] = None
    value["normalized_proof_dag"] = None
    value["source_lemma_ids"] = []
    value["environment"]["repository"] = (
        "https://github.com/leanprover-community/mathlib4"
    )
    value["environment"]["revision"] = "revision"
    value["environment"]["file_path"] = "Fixture.lean"
    value["environment"]["source_span"] = {
        "start": {"line": 7, "column": 1},
        "end": {"line": 8, "column": 21},
    }
    record = type(_record()).from_dict(value)

    task = _source_position_verification_task(record, tmp_path)

    assert task.preamble.endswith("variable (localContext : True)")
    assert "theorem fixture" not in task.preamble
    assert task.declaration == "theorem fixture : True"
