from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import CandidateResult, RunMetadata


def write_artifacts(
    output_dir: Path, metadata: RunMetadata, results: Iterable[CandidateResult]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def read_artifacts(output_dir: Path) -> tuple[RunMetadata, list[CandidateResult]]:
    metadata = RunMetadata.from_dict(
        json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    )
    results = [
        CandidateResult.from_dict(json.loads(line))
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return metadata, results
