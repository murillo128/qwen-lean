from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_lean.generalist_v2 import GeneralistV2Config
from qwen_lean.generalist_v2_dataset import sha256_file
from qwen_lean.generalist_v2_release import (
    compact_generalist_v2_release_evidence,
)
from qwen_lean.generalist_v2_q4_canary import (
    Q4_ADAPTER_MODEL_SHA256,
    Q4_CANARY_REQUIRED_GATES,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _release_inputs(tmp_path: Path) -> tuple[GeneralistV2Config, dict[str, Path]]:
    config = GeneralistV2Config.load(ROOT / "config/qwen35-4b-generalist-v2.json")
    manifest_sha256 = config.dataset["binding"]["manifest_sha256"]
    adapter_sha256 = Q4_ADAPTER_MODEL_SHA256
    workload = {
        "pass_at_k": {"pass@1": 0.1, "pass@4": 0.2, "pass@8": 0.3},
        "tasks_with_verified_candidate": 3,
    }
    paths = {
        "binding": _write(
            tmp_path / "binding.json",
            {
                "schema_version": "generalist-v2-dataset-binding-v1",
                "artifact_id": "qwen-lean-generalist-v2",
                "dataset": {
                    "manifest_sha256": manifest_sha256,
                    "files": {"records.jsonl.gz": {"sha256": "records-sha256"}},
                    "general_train": {"sha256": "general-train-sha256"},
                },
                "resolved": {
                    "general_train": {
                        "statements": 181531,
                        "proof_variants": 182812,
                    }
                },
                "serialization": {
                    "id": "lean-sft-v2",
                    "lengths": {"selected_context_tokens": 32768},
                    "completion_and_one_eos_supervised": True,
                    "prompt_supervised": False,
                    "packing": False,
                    "truncation": False,
                },
                "trajectory": {"optimizer_visible_variants": 182812},
                "weights": {"resolved_synthetic_mass_fraction": 0.068},
            },
        ),
        "training": _write(
            tmp_path / "training.json",
            {
                "schema_version": "generalist-v2-full-training-v1",
                "status": "passed",
                "model": config.model,
                "selected_lane": "nf4-qlora",
                "adapter": {
                    "format": "peft-lora",
                    "merged": False,
                    "base_model_shards_saved": False,
                    "trainable_parameter_count": 1,
                },
                "checkpoints": {
                    "Q4": {
                        "optimizer_step": 22852,
                        "adapter_config_sha256": "adapter-config-sha256",
                        "adapter_model_sha256": adapter_sha256,
                    }
                },
                "dataset": {"binding_manifest_sha256": manifest_sha256},
                "training": {
                    "trajectory": {"optimizer_visible_variants": 182812},
                    "completed_optimizer_steps": 22852,
                    "exactly_one_complete_pass": True,
                },
                "source_run_sha256": "run-sha256",
            },
        ),
        "selection": _write(
            tmp_path / "selection.json",
            {
                "schema_version": "generalist-v2-checkpoint-selection-v1",
                "status": "frozen",
                "q0_evidence_sha256": "q0-evidence-sha256",
                "selection": {
                    "selected_checkpoint": "Q4",
                    "rule": "validation-only",
                    "screening_ranking": ["Q2", "Q1", "Q3", "Q4"],
                },
                "selected_checkpoint": {
                    "adapter_model_sha256": adapter_sha256,
                    "optimizer_step": 22852,
                },
                "checkpoints": {
                    "Q4": {
                        "workloads": {
                            "minif2f-valid-clean-v2": workload,
                            "fresh-composition-valid-v2": workload,
                        }
                    }
                },
            },
        ),
        "extended": _write(
            tmp_path / "extended.json",
            {
                "schema_version": "generalist-v2-extended-validation-v1",
                "status": "selected-checkpoint-extended-validation-complete",
                "screening_selected_checkpoint": "Q4",
                "evaluated_checkpoint": {
                    "adapter_model_sha256": adapter_sha256,
                    "workloads": {"minif2f-valid-clean-v2": workload},
                },
            },
        ),
        "final": _write(
            tmp_path / "final.json",
            {
                "schema_version": "generalist-v2-final-assessment-v1",
                "status": "complete",
                "selected_checkpoint": "Q4",
                "selected_adapter_model_sha256": adapter_sha256,
                "workloads": {
                    "minif2f-test-clean-v2": {
                        "task_count": 1,
                        "models": {
                            label: {
                                **workload,
                                "category_counts": {"verified": 1},
                            }
                            for label in ("base", "selected", "deepseek")
                        },
                        "deepseek_gap_closed_fraction": 0.5,
                    }
                },
            },
        ),
        "deepseek_preflight": _write(
            tmp_path / "deepseek-preflight.json",
            {
                "schema_version": "generalist-v2-deepseek-final-preflight-v1",
                "status": "passed",
                "selected_checkpoint_frozen": "Q4",
            },
        ),
        "lora_parity": _write(
            tmp_path / "lora-parity.json",
            {
                "schema_version": "generalist-v2-lora-parity-evidence-v1",
                "gate_id": "qwen35-vllm-lora-parity-v1",
                "status": "passed",
                "model": {
                    "model_id": "Qwen/Qwen3.5-4B-Base",
                    "model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                },
                "vllm": {
                    "version": "0.27.2rc1.dev203+g41f179b57",
                    "source_revision": "41f179b57aa8ab6f634f508128ce1f1efadd0eb1",
                },
                "target_regex": config.lora["target_regex"],
                "requirements": {
                    "prior_evaluator_invalidated": True,
                    "static_overfit64_complete": True,
                    "static_q2_complete": True,
                    "hf_known_positive_reproduced": True,
                    "vllm_overfit_adapter_effect": True,
                    "q2_hf_forward_effect": True,
                    "q2_vllm_inference_effect": True,
                    "all_expected_outputs_present": True,
                    "zero_verifier_infrastructure_errors": True,
                },
                "adapters": {
                    "Q2": {"adapter_model_sha256": adapter_sha256}
                },
            },
        ),
    }
    paths["refinement"] = _write(
        tmp_path / "refinement.json",
        {
            "schema_version": "generalist-v2-refinement-conclusions-v1",
            "status": "complete",
            "selected_checkpoint": "Q4",
            "selected_adapter_model_sha256": adapter_sha256,
            "extended_validation_sha256": sha256_file(paths["extended"]),
            "final_assessment_sha256": sha256_file(paths["final"]),
            "selection_evidence_sha256": sha256_file(paths["selection"]),
            "q0_evidence_sha256": "q0-evidence-sha256",
            "analysis_scope": {
                "validation_only": True,
                "test_workloads_inspected_for_refinement": False,
            },
            "workloads": {
                "minif2f-valid-clean-v2": {
                    "exact_c_i_distribution": {"0": 1},
                    "partitions": {
                        "robust": {"task_count": 0},
                        "search-sensitive": {"task_count": 0},
                        "lottery": {"task_count": 0},
                        "dead-zone": {"task_count": 1},
                    },
                    "paired_n8_overlap": {
                        "q0_vs_q4": {"counts": {"solved_by_neither": 1}},
                        "q4_vs_deepseek": {"counts": {"solved_by_neither": 1}},
                        "compute_contract": "same n=8 contract",
                    },
                }
            },
            "conclusions": {"actionable_future_work": []},
        },
    )
    paths["q4_canary"] = _write(
        tmp_path / "q4-canary.json",
        {
            "schema_version": "generalist-v2-q4-canary-evidence-v1",
            "gate_id": "qwen35-q4-final-inference-canary-v1",
            "status": "passed",
            "classification": "PASS",
            "adapter_model_sha256": adapter_sha256,
            "requirements": {
                name: True for name in sorted(Q4_CANARY_REQUIRED_GATES)
            },
            "probe_selection": {"probe_count": 12},
            "functional_parity": {"hf_q4_vs_vllm_q4": {}},
            "arm_summaries": {"hf_q4": {}, "vllm_q4": {}},
        },
    )
    paths["q4_failure_diagnosis"] = _write(
        tmp_path / "q4-failure-diagnosis.json",
        {
            "schema_version": (
                "generalist-v2-q4-fresh-test-failure-diagnosis-v1"
            ),
            "status": "complete",
            "checkpoint_id": "Q4",
            "adapter_model_sha256": adapter_sha256,
            "method": {"benchmark_regenerated": False, "sample_count": 100},
            "full_candidate_diagnostics": {"candidate_count": 3320},
            "sample": {
                "primary_category_counts": {"unknown_or_mismatched_lemma": 100},
                "overlapping_signal_counts": {
                    "unknown_or_mismatched_lemma": 100
                },
            },
            "interpretation": {
                "classification": "narrow-template failure pattern"
            },
            "artifacts": {
                "final_assessment_sha256": sha256_file(paths["final"])
            },
        },
    )
    return config, paths


def test_release_evidence_binds_selected_adapter_and_inputs(tmp_path: Path) -> None:
    config, paths = _release_inputs(tmp_path)
    output = tmp_path / "release.json"

    evidence = compact_generalist_v2_release_evidence(
        config,
        paths["binding"],
        paths["training"],
        paths["selection"],
        paths["extended"],
        paths["final"],
        paths["refinement"],
        paths["deepseek_preflight"],
        paths["lora_parity"],
        paths["q4_canary"],
        paths["q4_failure_diagnosis"],
        output,
    )

    assert evidence["status"] == "ready-for-review"
    assert evidence["adapter"]["selected_checkpoint"] == "Q4"
    assert evidence["adapter"]["adapter_model_sha256"] == Q4_ADAPTER_MODEL_SHA256
    assert evidence["screening"]["test_or_riemann_used_for_selection"] is False
    assert evidence["scope_amendment"]["riemann_evidence_completion_gate"] is False
    assert evidence["refinement_conclusions"]["analysis_scope"][
        "validation_only"
    ] is True
    assert evidence["lora_inference_parity_gate"]["status"] == "passed"
    assert evidence["q4_final_inference_canary"]["status"] == "passed"
    assert evidence["q4_fresh_test_failure_diagnosis"]["method"][
        "benchmark_regenerated"
    ] is False
    assert evidence["evidence_sha256"]["full_training"] == sha256_file(
        paths["training"]
    )
    assert json.loads(output.read_text(encoding="utf-8")) == evidence


def test_release_evidence_rejects_mismatched_adapter(tmp_path: Path) -> None:
    config, paths = _release_inputs(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    selection["selected_checkpoint"]["adapter_model_sha256"] = "different"
    _write(paths["selection"], selection)

    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        compact_generalist_v2_release_evidence(
            config,
            paths["binding"],
            paths["training"],
            paths["selection"],
            paths["extended"],
            paths["final"],
            paths["refinement"],
            paths["deepseek_preflight"],
            paths["lora_parity"],
            paths["q4_canary"],
            paths["q4_failure_diagnosis"],
            tmp_path / "release.json",
        )
