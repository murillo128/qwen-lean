import json
from collections import Counter
from pathlib import Path

import pytest

from qwen_lean.artifacts import write_artifacts
from qwen_lean.baseline import GeneratedCandidate
from qwen_lean.metrics import summarize_results
from qwen_lean.phase2_schema import MathlibProofRecord
from qwen_lean.prompt import PROMPT_FORMAT_ID, render_proof_request
from qwen_lean.riemann_assessment import (
    FROZEN_DOMAIN_CONFIG_SHA256,
    GENERATION_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA_VERSION,
    WORKLOAD_ID,
    RiemannAssessmentConfig,
    _generation_settings,
    _relevance_views,
    _sha256_file,
    _task_ids_digest,
    _token_statistics,
    _verify_generated,
    load_domain_config,
    load_specialist_tasks,
    materialize_task_metadata,
    write_compact_evidence,
)
from qwen_lean.schema import (
    PHASE1_RESULT_SCHEMA_VERSION,
    CandidateResult,
    RunMetadata,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/qwen35-9b-riemann-assessment.json"
DOMAIN_CONFIG = ROOT / "config/riemann-domain-breakdown.json"


def test_config_freezes_issue_65_identity_and_sampling() -> None:
    config = RiemannAssessmentConfig.load(CONFIG)

    assert config.workload["corpus_id"] == "riemann-specialist-validation-v1"
    assert config.workload["expected_task_count"] == 556
    assert config.model["model_revision"] == (
        "68c46c4b3498877f3ef123c856ecfde50c39f404"
    )
    assert config.lane["lane_id"] == "bf16-text-only-v1"
    assert config.lane["quantization"] is None
    assert config.sampling["candidates_per_task"] == 4
    assert config.sampling["temperature"] == 0.8
    assert config.sampling["top_p"] == 0.95
    assert config.sampling["top_k"] == -1
    assert config.sampling["max_new_tokens"] == 1024
    assert config.sampling["seed"] == 0


def test_config_rejects_contract_drift(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["sampling"]["temperature"] = 0.7
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="sampling differs"):
        RiemannAssessmentConfig.load(changed)


def test_committed_specialist_workload_and_frozen_domain_views_are_complete() -> None:
    config = RiemannAssessmentConfig.load(CONFIG)
    tasks = load_specialist_tasks(config, ROOT)
    domains = load_domain_config(DOMAIN_CONFIG)
    metadata = materialize_task_metadata(tasks, domains)

    assert len(tasks) == 556
    assert len({item.record.id for item in tasks}) == 556
    assert all(item.record.split == "validation" for item in tasks)
    assert all(
        item.record.source_revision == config.workload["source_revision"]
        for item in tasks
    )
    assert _sha256_file(DOMAIN_CONFIG) == FROZEN_DOMAIN_CONFIG_SHA256
    assert _task_ids_digest([item.record.id for item in tasks]) == (
        "746cc51878ba7099a01d00f2ffd3d02965b216856766816cbe62d8af98887f4b"
    )
    assert Counter(item["primary_domain"] for item in metadata) == {
        "shared-formal-prerequisite": 409,
        "real-complex-analysis": 57,
        "broad-number-theory": 54,
        "prime-arithmetic-divisibility": 19,
        "zeta-analytic-number-theory": 17,
    }
    assert Counter(item["direct_relevance_class"] for item in metadata) == {
        None: 471,
        "number-theory-control": 61,
        "premise-1": 10,
        "core": 6,
        "premise-2": 4,
        "source-neighborhood": 3,
        "user-1": 1,
    }
    assert Counter(item["component_inclusion_basis"] for item in metadata) == {
        "component-contains-riemann-bubble": 485,
        "component-contains-number-theory-control": 61,
        "component-contains-core": 10,
    }
    relevance = _relevance_views(tasks)
    assert len(relevance["riemann-bubble"]) == 24
    assert len(relevance["number-theory-control"]) == 61
    assert len(relevance["component-context"]) == 471


def test_generation_settings_forbid_output_adaptation() -> None:
    config = RiemannAssessmentConfig.load(CONFIG)

    settings = _generation_settings(config)

    assert settings["chat_template"] is None
    assert settings["proof_extraction"] is None
    assert settings["semantic_repair"] is None
    assert settings["lean_feedback"] is None
    assert settings["candidate_regeneration"] is None


def test_compact_evidence_links_generation_and_rejects_summary_drift(
    tmp_path: Path,
) -> None:
    config = RiemannAssessmentConfig.load(CONFIG)
    tasks = load_specialist_tasks(config, ROOT)
    preflight_path = tmp_path / "preflight.json"
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "model": config.model,
        "model_snapshot": "/uncommitted/model-cache",
        "accepted_lane": "bf16-text-only-v1",
        "prompt_format_id": PROMPT_FORMAT_ID,
        "chat_template": None,
        "prompt_transformation": None,
        "bf16_attempt": {
            "status": "passed",
            "candidate_count": 4,
            "lane": config.lane,
        },
        "workload": {
            **config.workload,
            "loaded_task_count": 556,
        },
        "runtime": {
            "cuda_device": config.runtime["expected_cuda_device_name"],
            "torch": config.runtime["torch_version"],
            "transformers": config.runtime["transformers_version"],
            "bitsandbytes": config.runtime["bitsandbytes_version"],
            "vllm": config.runtime["inference_engine_version"],
        },
        "verifier_environment": {
            "mathlib_revision": config.workload["source_revision"],
            "lean_toolchain": config.workload["lean_toolchain"],
            "candidate_reconstruction": (
                "phase2-source-proof-span-substitution-v1"
            ),
            "has_sorry_disabled": True,
        },
        "verifier_controls": {
            "original_proof_status": "accepted",
            "controlled_invalid_status": "rejected",
            "timeout_seconds": 30.0,
        },
    }
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    generated_rows = []
    results = []
    for task_offset, task in enumerate(tasks):
        for candidate_index in range(4):
            text = f"candidate-{task.record.id[:8]}-{candidate_index}"
            tokens = candidate_index + 1
            generated_rows.append(
                {
                    "task_id": task.record.id,
                    "candidate_index": candidate_index,
                    "text": text,
                    "token_count": tokens,
                    "finish_reason": "eos",
                    "generation_latency_seconds": 0.1,
                }
            )
            verified = task_offset == 0 and candidate_index == 0
            results.append(
                CandidateResult(
                    task_id=task.record.id,
                    candidate_id=f"model-{candidate_index}",
                    candidate_index=candidate_index,
                    candidate_text=text,
                    category="verified" if verified else "lean_rejected",
                    lean_exit_code=0 if verified else 1,
                    diagnostics={"stdout": "", "stderr": ""},
                    generation_latency_seconds=0.1,
                    verification_latency_seconds=0.01,
                    total_latency_seconds=0.11,
                    generated_token_count=tokens,
                    finish_reason="eos",
                )
            )
    generation_path = generation_dir / "generations.jsonl"
    generation_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in generated_rows),
        encoding="utf-8",
    )
    generation = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "assessment_id": config.value["assessment_id"],
        "config_sha256": config.digest(),
        "preflight_sha256": _sha256_file(preflight_path),
        "model": config.model,
        "accepted_lane": "bf16-text-only-v1",
        "workload_id": WORKLOAD_ID,
        "task_count": 556,
        "candidate_count": 2224,
        "generation_settings": _generation_settings(config),
        "runtime": {
            "engine_load_time_seconds": 1.0,
            "generation_wall_time_seconds": 10.0,
        },
        "generations_sha256": _sha256_file(generation_path),
    }
    (generation_dir / "generation.json").write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    runtime = {
        "config_sha256": config.digest(),
        "inference_execution": "local_cuda",
        "cuda_device": config.runtime["expected_cuda_device_name"],
        "torch": config.runtime["torch_version"],
        "transformers": config.runtime["transformers_version"],
        "bitsandbytes": config.runtime["bitsandbytes_version"],
        "engine_load_time_seconds": 1.0,
        "generation_wall_time_seconds": 10.0,
        "verification_wall_time_seconds": 5.0,
        "peak_device_memory_used_mib": 123,
    }
    metadata = RunMetadata(
        schema_version=PHASE1_RESULT_SCHEMA_VERSION,
        candidate_source="model",
        task_source=f"{WORKLOAD_ID}@{config.workload['membership_sha256']}",
        prompt_format_id=PROMPT_FORMAT_ID,
        lean_toolchain=config.workload["lean_toolchain"],
        mathlib_revision=config.workload["source_revision"],
        verifier_timeout_seconds=30.0,
        model_id=config.model["model_id"],
        tokenizer_id=config.model["tokenizer_id"],
        model_revision=config.model["model_revision"],
        tokenizer_revision=config.model["tokenizer_revision"],
        workload_id=WORKLOAD_ID,
        benchmark_split="validation",
        benchmark_repository=config.workload["source_repository"],
        benchmark_revision=config.workload["source_revision"],
        verifier_environment=preflight["verifier_environment"],
        candidates_per_task=4,
        inference_engine=config.runtime["inference_engine"],
        inference_engine_version=config.runtime["inference_engine_version"],
        generation_settings=_generation_settings(config),
        runtime=runtime,
    )
    summary = summarize_results(
        results,
        expected_task_ids=[task.record.id for task in tasks],
        candidates_per_task=4,
    )
    summary.update(
        {
            "workload_id": WORKLOAD_ID,
            "generated_tokens": _token_statistics(results),
            "engine_load_time_seconds": 1.0,
            "generation_wall_time_seconds": 10.0,
            "verification_wall_time_seconds": 5.0,
            "run_wall_time_seconds": 16.0,
            "throughput": {
                "generated_tokens_per_second": 556.0,
                "candidates_per_second": 222.4,
            },
            "compute_per_solved_task": {
                "generated_tokens": 5560.0,
                "generation_gpu_seconds": 10.0,
            },
        }
    )
    artifact_dir = tmp_path / "full"
    write_artifacts(artifact_dir, metadata, results, summary=summary)

    evidence_dir = tmp_path / "evidence"
    payload = write_compact_evidence(
        config,
        ROOT,
        DOMAIN_CONFIG,
        preflight_path,
        generation_dir,
        artifact_dir,
        evidence_dir,
    )

    assert payload["full"]["generation_checkpoint"]["results_match_generation"]
    assert payload["full"]["task_outcomes"]["rows"] == 556
    assert payload["full"]["domain_config"]["sha256"] == (
        FROZEN_DOMAIN_CONFIG_SHA256
    )

    summary["throughput"]["candidates_per_second"] = 1.0
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="throughput differs"):
        write_compact_evidence(
            config,
            ROOT,
            DOMAIN_CONFIG,
            preflight_path,
            generation_dir,
            artifact_dir,
            tmp_path / "tampered-evidence",
        )


def test_verification_substitutes_exact_raw_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "theorem demo : True := by\n  trivial\n"
    record = MathlibProofRecord.from_dict(
        {
            "schema_version": "mathlib-whole-proof-v1",
            "id": "task",
            "source_repository": "fixture",
            "source_revision": "fixture",
            "file_path": "Demo.lean",
            "declaration_name": "demo",
            "declaration_kind": "theorem",
            "source_span": {
                "start": {"line": 1, "column": 1},
                "end": {"line": 2, "column": 10},
            },
            "declaration_span": {
                "start": {"line": 1, "column": 1},
                "end": {"line": 1, "column": 24},
            },
            "proof_span": {
                "start": {"line": 1, "column": 24},
                "end": {"line": 2, "column": 10},
            },
            "declaration": "theorem demo : True",
            "proof": "by\n  trivial",
            "completion": "trivial",
            "premises": [],
            "file_group": "Demo.lean",
            "component_id": "component",
            "split": "validation",
            "statement_fingerprint": "fixture",
            "token_lengths": {
                "declaration": 1,
                "proof": 1,
                "completion": 1,
                "declaration_and_proof": 2,
                "declaration_and_completion": 2,
            },
        }
    )
    generated = GeneratedCandidate(
        task=load_task(record),
        candidate_index=0,
        text="exact True.intro",
        token_count=3,
        finish_reason="eos",
        generation_latency_seconds=0.2,
    )
    captured: dict[str, str] = {}

    class Check:
        status = "accepted"
        exit_code = 0
        diagnostic = ""
        latency_seconds = 0.1

    def fake_run(source_value: str, *_: object, **__: object) -> Check:
        captured["source"] = source_value
        return Check()

    monkeypatch.setattr("qwen_lean.riemann_assessment.run_lean_source", fake_run)
    result = _verify_generated(
        generated,
        record,
        source,
        ROOT,
        ROOT,
        30.0,
    )

    assert result.category == "verified"
    assert captured["source"] == "theorem demo : True := by\n  exact True.intro\n"
    assert render_proof_request(record.declaration).endswith(":= by\n  ")


def load_task(record: MathlibProofRecord):
    from qwen_lean.schema import TaskRecord

    return TaskRecord(record.id, "", record.declaration, record.declaration_name)
