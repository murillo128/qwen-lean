from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import qwen_lean.counterfactual_forking_assessment as counterfactual
from qwen_lean.counterfactual_forking_assessment import (
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    FORK_GENERATION_SCHEMA,
    FORK_VERIFICATION_SCHEMA,
    HANDOFF_COMMIT,
    HANDOFF_MANIFEST_SHA256,
    CounterfactualForkingConfig,
    ForkRequest,
    ParentTrajectory,
    _confirmation_results,
    _fork_generation_record,
    _request,
    _run_async_fork_generation,
    _validate_fork_generation_record,
    _verify_fork_generation_record,
    confirmation_requests,
    discovery_prefix_values,
    discovery_requests,
    fork_generation_config_sha256,
    fork_states,
    load_fork_generation_records,
    load_fork_verification_records,
    select_confirmation_intervals,
    validate_handoff_records,
)
from qwen_lean.native_thinking_assessment import (
    GENERATION_RECORD_SCHEMA,
    MathiaTask,
)
from qwen_lean.verifier import VerificationOutcome

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/qwen35-counterfactual-forking.json"


def test_config_freezes_parent_forks_budgets_and_independent_seeds() -> None:
    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    states = fork_states(4096)

    assert [(state.label, state.prefix_len) for state in states] == [
        ("P0", 0),
        ("P15", 614),
        ("P30", 1228),
        ("P45", 1843),
        ("P60", 2457),
        ("P75", 3072),
        ("P90", 3686),
    ]
    assert [4096 - state.prefix_len for state in states] == [
        4096,
        3482,
        2868,
        2253,
        1639,
        1024,
        410,
    ]
    assert tuple(config.discovery["seeds"]) == DISCOVERY_SEEDS
    assert tuple(config.confirmation["seeds"]) == CONFIRMATION_SEEDS
    assert set(DISCOVERY_SEEDS).isdisjoint(CONFIRMATION_SEEDS)
    assert config.handoff["release_transport"]["release_tag"] == (
        "issue-92-counterfactual-parent-handoff"
    )
    assert config.execution["gpu_memory_utilization"] == 0.89
    assert config.native.engine["gpu_memory_utilization"] == 0.9
    assert len(fork_generation_config_sha256(config)) == 64


def test_async_generation_returns_every_persisted_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSamplingParams:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class FakeTokensPrompt:
        def __init__(self, *, prompt_token_ids: list[int]) -> None:
            self.prompt_token_ids = prompt_token_ids

    class FakeAsyncLLM:
        @classmethod
        def from_engine_args(cls, _args: object) -> FakeAsyncLLM:
            return cls()

        async def generate(self, *_args: object, **_kwargs: object):
            yield SimpleNamespace(finished=True, outputs=[object()])

        def shutdown(self) -> None:
            pass

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_inputs = types.ModuleType("vllm.inputs")
    fake_inputs.TokensPrompt = FakeTokensPrompt
    fake_v1 = types.ModuleType("vllm.v1")
    fake_engine = types.ModuleType("vllm.v1.engine")
    fake_async_llm = types.ModuleType("vllm.v1.engine.async_llm")
    fake_async_llm.AsyncLLM = FakeAsyncLLM
    for name, module in (
        ("vllm", fake_vllm),
        ("vllm.inputs", fake_inputs),
        ("vllm.v1", fake_v1),
        ("vllm.v1.engine", fake_engine),
        ("vllm.v1.engine.async_llm", fake_async_llm),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    parent = _parent()
    state = parent.states[-1]
    requests = [
        _request(
            config,
            phase="discovery",
            parent=parent,
            state=state,
            seed=seed,
            max_tokens=4096 - state.prefix_len,
        )
        for seed in (100, 101)
    ]
    monkeypatch.setattr(counterfactual, "_engine_args", lambda *_args: object())
    monkeypatch.setattr(
        counterfactual,
        "_fork_generation_record",
        lambda _config, _tokenizer, request, _output, **_kwargs: {
            "branch_id": request.branch_id
        },
    )

    records = asyncio.run(
        _run_async_fork_generation(
            config,
            tokenizer=None,
            requests=requests,
            generation_path=tmp_path / "generations.jsonl",
            snapshot_path=tmp_path,
        )
    )

    assert [record["branch_id"] for record in records] == [
        request.branch_id for request in requests
    ]
    assert len((tmp_path / "generations.jsonl").read_text().splitlines()) == 2


def test_generation_record_builder_returns_the_validated_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def decode(self, _token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert not skip_special_tokens
            return "suffix"

    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    parent = _parent()
    state = parent.states[-1]
    request = _request(
        config,
        phase="discovery",
        parent=parent,
        state=state,
        seed=100,
        max_tokens=4096 - state.prefix_len,
    )
    monkeypatch.setattr(
        counterfactual,
        "_parse_combined_response",
        lambda *_args, **_kwargs: {
            "raw_text": "combined",
            "reasoning_content": "combined",
            "reasoning_token_count": state.prefix_len + 2,
            "final_content": None,
            "final_token_count": 0,
            "final_content_is_exact_raw_suffix": True,
            "final_content_parity": "no_final_content",
            "terminal_token_ids": [],
            "terminal_text": "",
        },
    )

    record = _fork_generation_record(
        config,
        FakeTokenizer(),
        request,
        SimpleNamespace(token_ids=[1, 2], text="suffix", finish_reason="length"),
        latency_seconds=1.0,
    )

    assert record["branch_id"] == request.branch_id
    assert record["schema_version"] == FORK_GENERATION_SCHEMA
    assert record["suffix_text_vllm_parity"] == "exact"


def test_generation_record_allows_only_token_limit_incomplete_unicode_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteTokenizer:
        def decode(self, _token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert not skip_special_tokens
            return "suffix\ufffd"

    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    parent = _parent()
    state = parent.states[-1]
    request = _request(
        config,
        phase="discovery",
        parent=parent,
        state=state,
        seed=100,
        max_tokens=4096 - state.prefix_len,
    )
    monkeypatch.setattr(
        counterfactual,
        "_parse_combined_response",
        lambda *_args, **_kwargs: {
            "raw_text": "combined\ufffd",
            "reasoning_content": "combined\ufffd",
            "reasoning_token_count": state.prefix_len + 1,
            "final_content": None,
            "final_token_count": 0,
            "final_content_is_exact_raw_suffix": True,
            "final_content_parity": "no_final_content",
            "terminal_token_ids": [],
            "terminal_text": "",
        },
    )

    record = _fork_generation_record(
        config,
        IncompleteTokenizer(),
        request,
        SimpleNamespace(token_ids=[1], text="suffix", finish_reason="length"),
        latency_seconds=1.0,
    )

    assert record["suffix_response_text"] == "suffix\ufffd"
    assert record["suffix_text_vllm_parity"] == (
        "token_limit_trailing_incomplete_unicode"
    )
    assert record["vllm_emitted_text_length"] == len("suffix")

    mismatch = _fork_generation_record(
        config,
        IncompleteTokenizer(),
        request,
        SimpleNamespace(token_ids=[1], text="sufXix", finish_reason="length"),
        latency_seconds=1.0,
    )
    assert mismatch["suffix_text_vllm_parity"] == (
        "diagnostic_mismatch_token_ids_authoritative"
    )
    assert mismatch["vllm_emitted_text"] == "sufXix"
    assert mismatch["vllm_text_first_mismatch_index"] == 3

    non_length = _fork_generation_record(
        config,
        IncompleteTokenizer(),
        request,
        SimpleNamespace(token_ids=[1], text="suffix", finish_reason="stop"),
        latency_seconds=1.0,
    )
    assert non_length["suffix_text_vllm_parity"] == (
        "diagnostic_mismatch_token_ids_authoritative"
    )
    tampered = dict(mismatch)
    tampered["vllm_text_first_mismatch_index"] = 0
    with pytest.raises(ValueError, match="mismatch position changed"):
        _validate_fork_generation_record(tampered, request)


def test_generation_restart_preserves_and_drops_only_a_torn_tail(
    tmp_path: Path,
) -> None:
    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    parent = _parent()
    state = parent.states[-1]
    requests = [
        _request(
            config,
            phase="discovery",
            parent=parent,
            state=state,
            seed=seed,
            max_tokens=4096 - state.prefix_len,
        )
        for seed in (100, 101)
    ]
    complete_payload = json.dumps(
        _fork_record(requests[0]), sort_keys=True, separators=(",", ":")
    ).encode()
    interrupted_tail = b'{"branch_id":"interrupted'
    path = tmp_path / "generations.jsonl"
    path.write_bytes(complete_payload + b"\n" + interrupted_tail)

    records = load_fork_generation_records(path, requests)

    assert [record["branch_id"] for record in records] == [requests[0].branch_id]
    assert path.read_bytes() == complete_payload + b"\n"
    sidecars = list(tmp_path.glob("generations.jsonl.interrupted-tail-*.bin"))
    assert len(sidecars) == 1
    assert sidecars[0].read_bytes() == interrupted_tail


def test_handoff_integrity_uses_exact_line_and_allows_later_appends(
    tmp_path: Path,
) -> None:
    record = _parent_record("candidate-a", [10, 20, 30])
    raw_line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    expected = {
        key: record[key]
        for key in (
            "candidate_id",
            "workload",
            "task_id",
            "candidate_index",
            "seed",
            "prompt_sha256",
            "rendered_prompt_sha256",
            "raw_response_sha256",
            "raw_response_token_count",
            "reasoning_token_count",
            "final_token_count",
            "finish_reason",
        )
    }
    expected.update(
        {
            "raw_generation_jsonl_line_number": 2,
            "raw_generation_record_sha256": hashlib.sha256(
                raw_line.encode()
            ).hexdigest(),
            "raw_response_token_ids_sha256": _json_hash([10, 20, 30]),
        }
    )
    manifest = {
        "source": {"generation_config_sha256": "generation"},
        "candidates": [expected],
    }
    artifact = tmp_path / "generations.jsonl"
    artifact.write_text(
        '{"unrelated":true}\n' + raw_line + '\n{"later_append":true}\n',
        encoding="utf-8",
    )

    located = validate_handoff_records(manifest, artifact)

    assert located == {"candidate-a": record}

    artifact.write_text(
        '{"unrelated":true}\n' + raw_line.replace("reason", "changed") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="record bytes changed"):
        validate_handoff_records(manifest, artifact)


def test_compact_release_transport_validates_package_and_frozen_record_order(
    tmp_path: Path,
) -> None:
    record = _parent_record("candidate-a", [10, 20, 30])
    raw_line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    expected = {
        key: record[key]
        for key in (
            "candidate_id",
            "workload",
            "task_id",
            "candidate_index",
            "seed",
            "prompt_sha256",
            "rendered_prompt_sha256",
            "raw_response_sha256",
            "raw_response_token_count",
            "reasoning_token_count",
            "final_token_count",
            "finish_reason",
        )
    }
    expected.update(
        {
            "raw_generation_jsonl_line_number": 2445,
            "raw_generation_record_sha256": hashlib.sha256(
                raw_line.encode()
            ).hexdigest(),
            "raw_response_token_ids_sha256": _json_hash([10, 20, 30]),
        }
    )
    manifest = {
        "source": {"generation_config_sha256": "generation"},
        "candidates": [expected],
    }
    package_dir = tmp_path / "extracted"
    package_dir.mkdir()
    generations = package_dir / "generations.jsonl"
    generations.write_text(raw_line + "\n", encoding="utf-8")
    packaged_manifest = package_dir / "counterfactual-forking-handoff.json"
    packaged_manifest.write_bytes(
        (
            ROOT / "evidence/qwen35-native-thinking/counterfactual-forking-handoff.json"
        ).read_bytes()
    )
    generations_sha256 = _file_hash(generations)
    metadata = {
        "schema_version": "test-package-v1",
        "producer_issue": 89,
        "consumer_issue": 92,
        "handoff_commit": HANDOFF_COMMIT,
        "record_count": 1,
        "task_count": 1,
        "candidate_count_per_task": 1,
        "extraction_order": "frozen handoff manifest candidate order",
        "source_generation_artifact": (
            "artifacts/qwen35-native-thinking/full/generations.jsonl"
        ),
        "referenced_jsonl_line_range": {"minimum": 2445, "maximum": 2445},
        "integrity": {
            "status": "PASS",
            "byte_preserved_original_jsonl_lines": True,
            "candidates_regenerated": False,
            "records_reconstructed_from_parsed_json": False,
            "reasoning_text_retokenized": False,
            "fail_closed": True,
            "validated_hashes": [
                "raw_generation_record_sha256",
                "raw_response_sha256",
                "raw_response_token_ids_sha256",
            ],
        },
        "payload_files": [
            {
                "filename": "generations.jsonl",
                "sha256": generations_sha256,
                "size_bytes": generations.stat().st_size,
            },
            {
                "filename": "counterfactual-forking-handoff.json",
                "sha256": HANDOFF_MANIFEST_SHA256,
                "size_bytes": packaged_manifest.stat().st_size,
            },
        ],
    }
    metadata_path = package_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    metadata_sha256 = _file_hash(metadata_path)
    sums_path = package_dir / "SHA256SUMS"
    sums_path.write_text(
        "\n".join(
            (
                f"{HANDOFF_MANIFEST_SHA256}  counterfactual-forking-handoff.json",
                f"{generations_sha256}  generations.jsonl",
                f"{metadata_sha256}  metadata.json",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    package = tmp_path / "parent.tar.zst"
    package.write_bytes(b"immutable package bytes")
    release_transport = {
        "release_tag": "test-release",
        "release_url": "https://github.com/example/repo/releases/tag/test-release",
        "asset_name": package.name,
        "asset_size_bytes": package.stat().st_size,
        "asset_sha256": _file_hash(package),
        "package_metadata_schema": "test-package-v1",
        "package_metadata_sha256": metadata_sha256,
        "sha256sums_sha256": _file_hash(sums_path),
        "compact_generations_sha256": generations_sha256,
        "record_count": 1,
        "extraction_order": "frozen handoff manifest candidate order",
    }

    located = validate_handoff_records(
        manifest,
        generations,
        release_package_path=package,
        release_transport=release_transport,
    )

    assert located == {"candidate-a": record}
    package.write_bytes(b"tampered package bytes")
    with pytest.raises(ValueError, match="asset size|asset SHA-256"):
        validate_handoff_records(
            manifest,
            generations,
            release_package_path=package,
            release_transport=release_transport,
        )


def test_discovery_requests_use_exact_prefix_tokens_and_restart_binding() -> None:
    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    parent = _parent()

    requests = discovery_requests(config, [parent])

    assert len(requests) == 7 * 6
    p45 = next(
        request
        for request in requests
        if request.state.label == "P45" and request.seed == 103
    )
    assert p45.max_tokens == 4096 - p45.state.prefix_len
    assert p45.fork_prompt_token_ids == (
        parent.rendered_prompt_token_ids
        + parent.raw_response_token_ids[: p45.state.prefix_len]
    )
    record = _fork_record(p45)
    _validate_fork_generation_record(record, p45)

    tampered = dict(record)
    tampered["parent_raw_generation_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding changed"):
        _validate_fork_generation_record(tampered, p45)

    tampered = dict(record)
    tampered["fork_generation_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity field changed"):
        _validate_fork_generation_record(tampered, p45)


def test_parser_final_bytes_allow_only_the_pinned_terminal_eos() -> None:
    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    request = discovery_requests(config, [_parent()])[0]
    record = _fork_record(request)
    final = "by exact trivial"
    terminal = counterfactual.QWEN35_EOS_TOKEN_TEXT
    suffix_ids = [1, counterfactual.QWEN35_EOS_TOKEN_ID]
    suffix_text = final + terminal
    combined_text = "reasoning</think>" + suffix_text
    record.update(
        {
            "suffix_response_text": suffix_text,
            "suffix_response_sha256": hashlib.sha256(suffix_text.encode()).hexdigest(),
            "suffix_response_token_ids": suffix_ids,
            "suffix_response_token_ids_sha256": _json_hash(suffix_ids),
            "suffix_response_token_count": len(suffix_ids),
            "combined_response_text": combined_text,
            "combined_response_sha256": hashlib.sha256(
                combined_text.encode()
            ).hexdigest(),
            "combined_response_token_count": (
                request.state.prefix_len + len(suffix_ids)
            ),
            "final_content": final,
            "final_content_sha256": hashlib.sha256(final.encode()).hexdigest(),
            "final_token_count": len(suffix_ids),
            "final_production_status": "nonempty",
            "parser_final_content_is_exact_raw_suffix": False,
            "parser_final_content_parity": "exact_before_terminal_eos",
            "parser_terminal_token_ids": [counterfactual.QWEN35_EOS_TOKEN_ID],
            "parser_terminal_text": terminal,
        }
    )

    _validate_fork_generation_record(record, request)

    tampered = dict(record)
    tampered["parser_terminal_text"] = "<|endoftext|>"
    with pytest.raises(ValueError, match="parser changed final bytes"):
        _validate_fork_generation_record(tampered, request)


def test_confirmation_selection_threshold_and_matched_budget() -> None:
    config = CounterfactualForkingConfig.load(CONFIG_PATH)
    parent = _parent()
    values = []
    v_values = [0.0, 2 / 6, 0.0, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
    for state, value in zip(parent.states, v_values, strict=True):
        values.append(
            {
                "workload": parent.task.workload,
                "task_id": parent.task.task_id,
                "parent_candidate_id": parent.handoff["candidate_id"],
                "fork_state": state.label,
                "fork_fraction": state.fraction,
                "fork_prefix_len": state.prefix_len,
                "V_op": value,
                "F": 1.0,
            }
        )

    selected = select_confirmation_intervals(values, task_order=[parent.task.task_id])

    assert [row["Delta_op"] for row in selected] == [2 / 6, -2 / 6]
    plan = {
        "schema_version": "qwen35-counterfactual-confirmation-plan-v1",
        "selected_intervals": selected,
    }
    requests = confirmation_requests(config, [parent], plan)
    assert len(requests) == 2 * 2 * 12
    for interval in selected:
        interval_requests = [
            request
            for request in requests
            if request.interval_id == interval["interval_id"]
        ]
        assert {request.max_tokens for request in interval_requests} == {
            4096 - int(interval["after_prefix_len"])
        }
        assert {request.seed for request in interval_requests} == set(
            CONFIRMATION_SEEDS
        )
        assert {request.interval_side for request in interval_requests} == {
            "before",
            "after",
        }


def test_discovery_values_and_confirmation_classification() -> None:
    rows = []
    for state_index, state in enumerate(fork_states(4096)):
        for seed_index, seed in enumerate(DISCOVERY_SEEDS):
            verified = state.label == "P15" and seed_index < 2
            rows.append(
                _analysis_row(
                    state=state.label,
                    fraction=state.fraction,
                    prefix_len=state.prefix_len,
                    seed=seed,
                    verified=verified,
                    interval_id=None,
                    interval_side=None,
                )
            )
    values = discovery_prefix_values(rows)
    p15 = next(row for row in values if row["fork_state"] == "P15")
    assert p15["V_op"] == 2 / 6
    assert p15["F"] == 1.0

    interval = {
        "interval_id": "task-a:P0->P15",
        "workload": "minif2f-valid-clean-v2",
        "task_id": "task-a",
        "parent_candidate_id": "parent-a",
        "before_state": "P0",
        "before_fraction": 0.0,
        "before_prefix_len": 0,
        "after_state": "P15",
        "after_fraction": 0.15,
        "after_prefix_len": 614,
        "matched_remaining_budget": 3482,
        "Delta_op": 2 / 6,
        "interval_index": 1,
        "selection_index": 0,
        "selection_rule": "test",
    }
    confirmation_rows = []
    for side in ("before", "after"):
        for index, seed in enumerate(CONFIRMATION_SEEDS):
            confirmation_rows.append(
                _analysis_row(
                    state="P0" if side == "before" else "P15",
                    fraction=0.0 if side == "before" else 0.15,
                    prefix_len=0 if side == "before" else 614,
                    seed=seed,
                    verified=side == "after" and index < 2,
                    interval_id=interval["interval_id"],
                    interval_side=side,
                    max_tokens=3482,
                )
            )
    result = _confirmation_results(
        {"selected_intervals": [interval]}, confirmation_rows
    )[0]
    assert result["Delta_mb"] == 2 / 12
    assert result["classification"] == "stable_positive"
    assert result["discovery_sign_replicated"] is True


def test_verification_is_final_channel_only_and_retries_infrastructure(
    tmp_path: Path,
) -> None:
    task = _task()
    final = "```lean\nexact True.intro\n```  "
    generation = {
        "branch_id": "branch-a",
        "phase": "discovery",
        "workload": task.workload,
        "task_id": task.task_id,
        "fork_state": "P15",
        "fork_prefix_len": 10,
        "branch_seed": 100,
        "fork_generation_config_sha256": "config",
        "final_content": final,
        "final_content_sha256": hashlib.sha256(final.encode()).hexdigest(),
    }
    verifier = _CapturingVerifier()

    record = _verify_fork_generation_record(
        generation,
        task,
        verifier,
        attempt_index=0,  # type: ignore[arg-type]
    )

    assert verifier.source == (
        f"{task.preamble}\n\n{task.declaration} := by\n  {final}\n"
    )
    assert record["final_content_submitted_without_repair"] is True

    request = _request(
        CounterfactualForkingConfig.load(CONFIG_PATH),
        phase="discovery",
        parent=_parent(),
        state=_parent().states[1],
        seed=100,
        max_tokens=3482,
    )
    path = tmp_path / "verifications.jsonl"
    base = {
        "schema_version": FORK_VERIFICATION_SCHEMA,
        "branch_id": request.branch_id,
        "phase": request.phase,
        "workload": request.parent.task.workload,
        "task_id": request.parent.task.task_id,
        "fork_state": request.state.label,
        "fork_prefix_len": request.state.prefix_len,
        "branch_seed": request.seed,
        "fork_generation_config_sha256": request.generation_config_sha256,
        "final_content_sha256": None,
        "final_content_submitted_without_repair": True,
        "lean_success": False,
        "lean_exit_code": None,
        "diagnostics": {},
        "verification_latency_seconds": 0.0,
    }
    path.write_text(
        "\n".join(
            json.dumps({**base, "attempt_index": index, "category": category})
            for index, category in enumerate(("verifier_timeout", "empty_candidate"))
        )
        + "\n",
        encoding="utf-8",
    )
    attempts = load_fork_verification_records(path, [request])
    assert [row["attempt_index"] for row in attempts] == [0, 1]
    assert attempts[-1]["category"] == "empty_candidate"


def _task() -> MathiaTask:
    return MathiaTask(
        task_id="task-a",
        workload="minif2f-valid-clean-v2",
        preamble="import Mathlib",
        declaration="theorem task_a : True",
        declaration_name="task_a",
        intuition="Use truth introduction.",
        intuition_sha256="intuition",
        theorem_sha256="theorem",
    )


def _parent() -> ParentTrajectory:
    task = _task()
    raw_ids = tuple(range(4096))
    return ParentTrajectory(
        ordinal=0,
        task=task,
        handoff={
            "candidate_id": "parent-a",
            "raw_response_sha256": "raw",
            "raw_response_token_ids_sha256": _json_hash(list(raw_ids)),
        },
        record={"finish_reason": "token_limit", "final_content": None},
        record_sha256="record",
        raw_response_token_ids=raw_ids,
        rendered_prompt_token_ids=(8000, 8001),
        rendered_prompt_sha256="prompt",
        states=fork_states(4096),
        parser_parity={"status": "passed"},
    )


def _parent_record(candidate_id: str, token_ids: list[int]) -> dict[str, object]:
    raw_text = "reason"
    return {
        "schema_version": GENERATION_RECORD_SCHEMA,
        "candidate_id": candidate_id,
        "arm": "t1",
        "enable_thinking": True,
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "generation_config_sha256": "generation",
        "workload": "minif2f-valid-clean-v2",
        "task_id": "task-a",
        "candidate_index": 0,
        "seed": 0,
        "prompt_sha256": "prompt",
        "rendered_prompt_sha256": "rendered",
        "rendered_prompt_token_count": 2,
        "raw_response_text": raw_text,
        "raw_response_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "raw_response_token_ids": token_ids,
        "raw_response_token_count": len(token_ids),
        "reasoning_content": raw_text,
        "reasoning_content_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "reasoning_token_count": len(token_ids),
        "final_content": None,
        "final_content_sha256": None,
        "final_token_count": 0,
        "finish_reason": "token_limit",
    }


def _fork_record(request: ForkRequest) -> dict[str, object]:
    suffix_ids = [1, 2]
    suffix_text = "suffix"
    combined_text = "combined"
    parent = request.parent
    prefix = list(parent.raw_response_token_ids[: request.state.prefix_len])
    return {
        "schema_version": FORK_GENERATION_SCHEMA,
        "branch_id": request.branch_id,
        **request.identity_payload(),
        "parent_raw_generation_record_sha256": parent.record_sha256,
        "parent_raw_response_sha256": parent.handoff["raw_response_sha256"],
        "parent_raw_response_token_ids_sha256": parent.handoff[
            "raw_response_token_ids_sha256"
        ],
        "rendered_prompt_sha256": parent.rendered_prompt_sha256,
        "rendered_prompt_token_count": len(parent.rendered_prompt_token_ids),
        "reasoning_prefix_token_ids_sha256": _json_hash(prefix),
        "fork_prompt_token_ids_sha256": _json_hash(list(request.fork_prompt_token_ids)),
        "fork_prompt_token_count": len(request.fork_prompt_token_ids),
        "inference_input_kind": "prompt_token_ids",
        "suffix_response_text": suffix_text,
        "suffix_response_sha256": hashlib.sha256(suffix_text.encode()).hexdigest(),
        "suffix_response_token_ids": suffix_ids,
        "suffix_response_token_ids_sha256": _json_hash(suffix_ids),
        "suffix_response_token_count": len(suffix_ids),
        "combined_response_text": combined_text,
        "combined_response_sha256": hashlib.sha256(combined_text.encode()).hexdigest(),
        "combined_response_token_count": len(prefix) + len(suffix_ids),
        "reasoning_content": combined_text,
        "reasoning_content_sha256": hashlib.sha256(combined_text.encode()).hexdigest(),
        "reasoning_token_count": len(prefix) + len(suffix_ids),
        "final_content": None,
        "final_content_sha256": None,
        "final_token_count": 0,
        "final_production_status": "empty",
        "parser_final_content_is_exact_raw_suffix": True,
        "finish_reason": "token_limit",
        "raw_finish_reason": "length",
        "generation_latency_seconds": 1.0,
        "request_id": request.branch_id,
    }


def _analysis_row(
    *,
    state: str,
    fraction: float,
    prefix_len: int,
    seed: int,
    verified: bool,
    interval_id: str | None,
    interval_side: str | None,
    max_tokens: int | None = None,
) -> dict[str, object]:
    return {
        "branch_id": f"{interval_id}-{interval_side}-{state}-{seed}",
        "workload": "minif2f-valid-clean-v2",
        "task_id": "task-a",
        "parent_candidate_id": "parent-a",
        "fork_state": state,
        "fork_fraction": fraction,
        "fork_prefix_len": prefix_len,
        "branch_seed": seed,
        "max_tokens": 4096 - prefix_len if max_tokens is None else max_tokens,
        "interval_id": interval_id,
        "interval_side": interval_side,
        "final_production_status": "nonempty",
        "verification": {"category": "verified" if verified else "lean_rejected"},
    }


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _CapturingVerifier:
    source: str | None = None

    def _run_source(self, source: str) -> VerificationOutcome:
        self.source = source
        return VerificationOutcome(
            category="lean_rejected",
            lean_exit_code=1,
            diagnostics={"stdout": "", "stderr": "expected"},
            latency_seconds=0.01,
        )
