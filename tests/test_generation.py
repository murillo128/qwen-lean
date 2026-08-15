from qwen_lean.generation import MODEL_ID, _model_metadata


def test_model_metadata_uses_supplied_task_source() -> None:
    metadata = _model_metadata(
        "custom-fixture-v1",
        MODEL_ID,
        30.0,
        128,
        {"python": "test"},
    )

    assert metadata.task_source == "custom-fixture-v1"
