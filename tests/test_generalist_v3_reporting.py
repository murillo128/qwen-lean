from qwen_lean.generalist_v3_reporting import (
    _checkpoint_training_summary,
    _lane_summary,
    _rolling_mean,
    _svg_chart,
)


def test_checkpoint_training_summary_reports_finite_trailing_health() -> None:
    rows = [
        {
            "optimizer_step": step,
            "learning_rate": 1e-5,
            "sft_loss_mean": float(step),
            "preservation_kl": float(step) / 10,
            "weighted_preservation_kl": float(step) / 100,
            "gradient_norm_before_clipping": float(step) / 2,
        }
        for step in range(1, 31)
    ]
    summary = _checkpoint_training_summary(rows, 30)
    assert summary["window_optimizer_steps"] == 25
    assert summary["sft_loss"] == 30.0
    assert summary["sft_loss_trailing_25_mean"] == 18.0
    assert summary["maximum_gradient_norm_through_checkpoint"] == 15.0
    assert summary["all_logged_objective_and_gradient_values_finite"] is True


def test_lane_summary_includes_eos_lengths_and_proof_constructs() -> None:
    lane = {
        "task_count": 2,
        "candidate_count": 8,
        "pass_at_1": 0.125,
        "pass_at_4": 0.5,
        "pass_at_8": 1.0,
        "solved_at_8": 1,
        "verified_candidates": 1,
        "verified_density": 0.125,
        "normalized_template_diversity": 0.75,
        "finish_reason_counts": {"eos": 6, "token_limit": 2},
        "generated_tokens": {"median": 42.0, "le_64_fraction": 0.75},
        "first_construct_counts": {"exact": 3, "constructor": 2, "term": 3},
        "dominant_template": {
            "sha256": "abc",
            "occurrences": 2,
            "theorem_count": 1,
            "verified_occurrences": 1,
        },
    }
    summary = _lane_summary(lane)
    assert summary["eos_fraction"] == 0.75
    assert summary["generated_tokens"]["median"] == 42.0
    assert summary["first_construct_counts"]["constructor"] == 2


def test_reporting_svg_and_rolling_mean_are_deterministic() -> None:
    assert _rolling_mean([1.0, 3.0, 5.0], 2) == [1.0, 2.0, 4.0]
    svg = _svg_chart(
        title="trajectory",
        panels=(("coverage", "solved", (("C1", ((0, 0), (100, 2))),)),),
    )
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "trajectory" in svg
    assert "coverage" in svg
    assert "#2563eb" in svg
