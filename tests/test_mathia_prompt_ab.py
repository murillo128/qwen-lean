from __future__ import annotations

import json
from pathlib import Path

import pytest

import qwen_lean.mathia_prompt_ab as prompt_ab
from qwen_lean.mathia_prompt_ab import (
    BoundTask,
    PromptABConfig,
    _exact_two_sided_mcnemar,
    _generation_shard_path,
    _sha256_text,
    _write_once_json,
    candidate_identity,
    inventory_generations,
    render_arm_prompt,
)
from qwen_lean.schema import TaskRecord
from qwen_lean.verifier import VerificationOutcome


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config/mathia-prompt-ab.json"
MANIFEST_PATH = (
    REPOSITORY_ROOT / "evidence/mathia-prompt-ab/execution-manifest.json"
)
RESULTS_PATH = REPOSITORY_ROOT / "evidence/mathia-prompt-ab/results.json"
RESULTS_README_PATH = REPOSITORY_ROOT / "evidence/mathia-prompt-ab/README.md"
FORMAT_DIAGNOSTIC_PATH = (
    REPOSITORY_ROOT
    / "evidence/mathia-prompt-ab/format-contamination-diagnostic.json"
)


def _bound_task() -> BoundTask:
    return BoundTask(
        ordinal=0,
        workload_id="minif2f-valid-clean-v2",
        task=TaskRecord(
            id="example",
            preamble="import Mathlib",
            declaration="theorem example : True",
            declaration_name="example",
        ),
        intuition_id="intuition-example",
        intuition_text="Use the canonical inhabitant of True.",
        intuition_sha256=_sha256_text("Use the canonical inhabitant of True."),
        model_visible_theorem_sha256="theorem-hash",
        q0_verified_candidate_count=0,
        metadata={},
    )


def test_prompt_templates_are_exact_and_differ_only_by_frozen_instruction() -> None:
    config = PromptABConfig.load(CONFIG_PATH)
    bound = _bound_task()
    arm_a = render_arm_prompt(config, bound, "A")
    arm_b = render_arm_prompt(config, bound, "B")
    assert arm_a == (
        "import Mathlib\n\n"
        "/- Mathematical intuition:\n"
        "Use the canonical inhabitant of True.\n"
        "-/\n\n"
        "theorem example : True := by\n  "
    )
    instruction = (
        "Complete the Lean proof below.\n"
        "Use the mathematical intuition as high-level guidance for the proof.\n"
        "Return only Lean code continuing after `by`.\n"
        "Do not use `sorry` or `admit`."
    )
    assert arm_b == arm_a.replace(
        "/- Mathematical intuition:\n",
        f"/- {instruction}\n\nMathematical intuition:\n",
        1,
    )


def test_candidate_identity_is_stable_and_binds_every_scientific_field() -> None:
    fields = {
        "arm_id": "A",
        "workload_id": "minif2f-valid-clean-v2",
        "task_id": "example",
        "prompt_sha256": "prompt",
        "candidate_index": 0,
        "sampling_seed": 0,
        "model_revision": "revision",
        "generation_config_sha256": "generation",
    }
    first = candidate_identity(**fields)
    assert candidate_identity(**fields) == first
    for key, replacement in {
        "arm_id": "B",
        "workload_id": "fresh-composition-valid-v2",
        "task_id": "other",
        "prompt_sha256": "other-prompt",
        "candidate_index": 1,
        "sampling_seed": 1,
        "model_revision": "other-revision",
        "generation_config_sha256": "other-generation",
    }.items():
        assert candidate_identity(**{**fields, key: replacement}) != first


def test_config_rejects_prompt_wording_drift(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["arms"]["B"]["comment_prefix"] = value["arms"]["B"][
        "comment_prefix"
    ].replace("high-level guidance", "guidance")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="arm B prompt wording differs"):
        PromptABConfig.load(path)


def test_cuda_gate_accepts_project_rtx_4070_ti_by_ada_capability() -> None:
    prompt_ab._validate_cuda_device_identity(
        "NVIDIA GeForce RTX 4070 Ti", 8, 9, [8, 9]
    )
    with pytest.raises(RuntimeError, match="compute capability 8.6"):
        prompt_ab._validate_cuda_device_identity(
            "NVIDIA GeForce RTX 3090", 8, 6, [8, 9]
        )


def test_verifier_environments_are_bound_per_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PromptABConfig.load(CONFIG_PATH)
    roots = {
        workload_id: tmp_path / workload_id
        for workload_id in prompt_ab.WORKLOAD_IDS
    }
    expected_revisions: dict[Path, str] = {}
    for workload_id, root in roots.items():
        contract = config.verifier["environments"][workload_id]
        root.mkdir()
        (root / "lean-toolchain").write_text(
            contract["lean_toolchain"] + "\n", encoding="utf-8"
        )
        (root / "lake-manifest.json").write_text(
            json.dumps(
                {
                    "packages": [
                        {"name": "mathlib", "rev": contract["mathlib_revision"]}
                    ]
                }
            ),
            encoding="utf-8",
        )
        mathlib_root = root / ".lake/packages/mathlib"
        mathlib_root.mkdir(parents=True)
        expected_revisions[root.resolve()] = contract["project_revision"]
        expected_revisions[mathlib_root.resolve()] = contract["mathlib_revision"]

    def fake_git_output(root: Path, *args: str) -> str:
        assert args == ("rev-parse", "HEAD")
        return expected_revisions[root.resolve()]

    monkeypatch.setattr(prompt_ab, "_git_output", fake_git_output)
    bundle = prompt_ab.verifier_environment_identities(config, roots)
    assert set(bundle["environments"]) == set(prompt_ab.WORKLOAD_IDS)
    assert (
        bundle["environments"]["minif2f-valid-clean-v2"]["project_revision"]
        == "f0a20e14c1eeccd859d51bb4c2b3ee487889c303"
    )
    assert (
        bundle["environments"]["fresh-composition-valid-v2"][
            "project_revision"
        ]
        == "7715064f690d0689f30889846f4e2c5e7ec0c47e"
    )
    hashes = bundle["environment_sha256_by_workload"]
    assert hashes["minif2f-valid-clean-v2"] != hashes[
        "fresh-composition-valid-v2"
    ]
    with pytest.raises(ValueError, match="must bind both"):
        prompt_ab.verifier_environment_identities(
            config, {"minif2f-valid-clean-v2": roots["minif2f-valid-clean-v2"]}
        )


def test_verification_result_rejects_workload_environment_mismatch() -> None:
    generation = {
        "candidate_id": "candidate",
        "arm_id": "A",
        "workload_id": "minif2f-valid-clean-v2",
        "task_id": "example",
        "candidate_index": 0,
        "raw_continuation_sha256": "raw",
    }
    value = {
        "schema_version": prompt_ab.VERIFICATION_RESULT_SCHEMA_VERSION,
        "manifest_sha256": "manifest",
        **generation,
        "verifier_environment_sha256": "fresh-environment",
        "category": "verified",
        "diagnostics": {},
    }
    with pytest.raises(ValueError, match="durable verification result differs"):
        prompt_ab._validate_verification_result(
            value,
            manifest_sha256="manifest",
            environment_sha256="minif2f-environment",
            generation=generation,
        )


def test_verifier_environment_probe_uses_q0_timeout_and_deduplicates() -> None:
    calls: dict[str, list[tuple[str, float]]] = {
        workload_id: [] for workload_id in prompt_ab.WORKLOAD_IDS
    }

    class FakeVerifier:
        def __init__(self, workload_id: str) -> None:
            self.workload_id = workload_id

        def prime_preamble(self, preamble: str, *, timeout_seconds: float):
            calls[self.workload_id].append((preamble, timeout_seconds))
            return None

    tasks = {
        0: _bound_task().task,
        1: TaskRecord(
            id="fresh-example",
            preamble="import PrimeNumberTheoremAnd",
            declaration="theorem fresh_example : True",
            declaration_name="fresh_example",
        ),
    }
    generations = [
        {"task_ordinal": 0, "workload_id": "minif2f-valid-clean-v2"},
        {"task_ordinal": 0, "workload_id": "minif2f-valid-clean-v2"},
        {"task_ordinal": 1, "workload_id": "fresh-composition-valid-v2"},
    ]
    evidence = prompt_ab._prime_verifier_environments(
        {
            workload_id: FakeVerifier(workload_id)
            for workload_id in prompt_ab.WORKLOAD_IDS
        },
        tasks,
        generations,
    )
    assert calls == {
        "minif2f-valid-clean-v2": [
            ("import Mathlib", prompt_ab.VERIFIER_ENVIRONMENT_PROBE_TIMEOUT_SECONDS)
        ],
        "fresh-composition-valid-v2": [
            (
                "import PrimeNumberTheoremAnd",
                prompt_ab.VERIFIER_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
            )
        ],
    }
    assert evidence["minif2f-valid-clean-v2"]["probe_count"] == 1
    assert evidence["fresh-composition-valid-v2"]["probe_count"] == 1


def test_verifier_environment_probe_fails_before_candidate_verification() -> None:
    class FailedProbeVerifier:
        def prime_preamble(self, preamble: str, *, timeout_seconds: float):
            return VerificationOutcome(
                category="verifier_timeout",
                lean_exit_code=None,
                diagnostics={"stdout": "", "stderr": "cold probe timed out"},
                latency_seconds=timeout_seconds,
            )

    with pytest.raises(RuntimeError, match="environment probe failed"):
        prompt_ab._prime_verifier_environments(
            {
                workload_id: FailedProbeVerifier()
                for workload_id in prompt_ab.WORKLOAD_IDS
            },
            {0: _bound_task().task},
            [{"task_ordinal": 0, "workload_id": "minif2f-valid-clean-v2"}],
        )


def test_atomic_write_once_rejects_nonidentical_replacement(tmp_path: Path) -> None:
    path = tmp_path / "immutable.json"
    _write_once_json(path, {"value": 1})
    _write_once_json(path, {"value": 1})
    with pytest.raises(ValueError, match="immutable artifact differs"):
        _write_once_json(path, {"value": 2})


def test_generation_inventory_skips_one_complete_atomic_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prompt_ab, "EXPECTED_TASKS", 1)
    manifest_sha256 = "manifest"
    slots = {
        arm_id: [
            f"{arm_id}-{index}"
            for index in range(8)
        ]
        for arm_id in ("A", "B")
    }
    task = {
        "ordinal": 0,
        "workload_id": "minif2f-valid-clean-v2",
        "task_id": "example",
        "prompt_sha256": {"A": "prompt-a", "B": "prompt-b"},
        "candidate_slots": slots,
    }
    manifest = {"tasks": [task]}
    candidates = [
        {
            "candidate_id": f"A-{index}",
            "candidate_index": index,
            "sampling_seed": 0,
            "raw_continuation": "trivial",
            "raw_continuation_sha256": _sha256_text("trivial"),
            "token_count": 1,
            "finish_reason": "eos",
            "generation_latency_seconds": 0.1,
            "generation_error": None,
        }
        for index in range(8)
    ]
    shard = {
        "schema_version": prompt_ab.GENERATION_SHARD_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "arm_id": "A",
        "workload_id": task["workload_id"],
        "task_id": task["task_id"],
        "task_ordinal": 0,
        "prompt_sha256": "prompt-a",
        "candidates": candidates,
    }
    _write_once_json(_generation_shard_path(tmp_path, "A", task), shard)
    inventory = inventory_generations(manifest, tmp_path, manifest_sha256)
    assert inventory["completed_candidate_count"] == 8
    assert inventory["completed_tasks_by_arm"] == {"A": [0], "B": []}
    assert inventory["generation_failure_count"] == 0
    assert set(inventory["candidates_by_id"]) == {f"A-{index}" for index in range(8)}


def test_exact_mcnemar_handles_ties_and_two_sided_tail() -> None:
    assert _exact_two_sided_mcnemar(0, 0) == 1.0
    assert _exact_two_sided_mcnemar(4, 0) == pytest.approx(0.125)
    assert _exact_two_sided_mcnemar(2, 2) == 1.0


def test_committed_execution_manifest_has_every_unique_candidate_slot() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == prompt_ab.MANIFEST_SCHEMA_VERSION
    assert manifest["task_count"] == 611
    assert manifest["task_counts"] == {
        "minif2f-valid-clean-v2": 223,
        "fresh-composition-valid-v2": 388,
    }
    assert manifest["prompt_integrity_gate"]["passed"] is True
    candidate_ids = [
        candidate_id
        for task in manifest["tasks"]
        for arm_id in ("A", "B")
        for candidate_id in task["candidate_slots"][arm_id]
    ]
    assert len(candidate_ids) == 9_776
    assert len(set(candidate_ids)) == 9_776


def test_committed_results_readme_is_reproducible_and_reports_boundaries() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    readme = RESULTS_README_PATH.read_text(encoding="utf-8")
    diagnostic = json.loads(FORMAT_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))

    assert readme == prompt_ab.render_results_readme(results)
    assert "## Against the unchanged Q0 reference" in readme
    assert "does not establish that frozen intuition improves" in readme
    assert "## Scoring-excluded format diagnostic" in readme
    assert diagnostic["decision_marker"] == "OBSERVED"
    assert diagnostic["arms"]["B"][
        "raw_rejections_with_verified_mechanical_variant"
    ] == 22
    assert diagnostic["scoring_boundary"].startswith(
        "Mechanical wrapper variants are diagnostic only."
    )
