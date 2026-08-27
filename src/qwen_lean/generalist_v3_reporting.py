from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from .dataset_v2 import sha256_file
from .generalist_v3 import GeneralistV3Config


CONFIGURATION_IDS = ("C0", "C1", "C2", "C3")
CHECKPOINT_STEPS = (100, 250, 500)
COLORS = {
    "C0": "#6b7280",
    "C1": "#2563eb",
    "C2": "#ea580c",
    "C3": "#16a34a",
}
MAJOR_CONSTRUCTS = (
    "exact",
    "constructor",
    "apply",
    "refine",
    "rw",
    "simp",
    "intro",
    "have",
    "other",
)
STRUCTURAL_GROUPS = ("direct", "branching", "deep")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"non-finite generalist-v3 trajectory value: {label}")
    return observed


def _read_training_log(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.strip()]
    if len(rows) != len(lines):
        raise ValueError("generalist-v3 training log contains blank rows")
    return rows, lines


def _checkpoint_training_summary(
    rows: Sequence[Mapping[str, Any]], step: int, *, window: int = 25
) -> dict[str, Any]:
    current = rows[step - 1]
    recent = rows[max(0, step - window) : step]
    fields = (
        "sft_loss_mean",
        "preservation_kl",
        "weighted_preservation_kl",
        "gradient_norm_before_clipping",
    )
    for row in rows[:step]:
        for field in fields:
            _finite_number(row[field], f"step {row['optimizer_step']} {field}")
    return {
        "learning_rate": _finite_number(current["learning_rate"], "learning_rate"),
        "sft_loss": _finite_number(current["sft_loss_mean"], "sft_loss"),
        "sft_loss_trailing_25_mean": fmean(
            _finite_number(row["sft_loss_mean"], "sft_loss") for row in recent
        ),
        "preservation_kl": _finite_number(
            current["preservation_kl"], "preservation_kl"
        ),
        "preservation_kl_trailing_25_mean": fmean(
            _finite_number(row["preservation_kl"], "preservation_kl")
            for row in recent
        ),
        "weighted_preservation_kl": _finite_number(
            current["weighted_preservation_kl"], "weighted_preservation_kl"
        ),
        "gradient_norm_before_clipping": _finite_number(
            current["gradient_norm_before_clipping"], "gradient_norm"
        ),
        "gradient_norm_trailing_25_mean": fmean(
            _finite_number(
                row["gradient_norm_before_clipping"], "gradient_norm"
            )
            for row in recent
        ),
        "maximum_gradient_norm_through_checkpoint": max(
            _finite_number(row["gradient_norm_before_clipping"], "gradient_norm")
            for row in rows[:step]
        ),
        "all_logged_objective_and_gradient_values_finite": True,
        "window_optimizer_steps": len(recent),
    }


def _lane_summary(lane: Mapping[str, Any]) -> dict[str, Any]:
    candidates = int(lane["candidate_count"])
    finish_reasons = dict(lane["finish_reason_counts"])
    return {
        "task_count": int(lane["task_count"]),
        "candidate_count": candidates,
        "pass_at_1": float(lane["pass_at_1"]),
        "pass_at_4": float(lane["pass_at_4"]),
        "pass_at_8": float(lane["pass_at_8"]),
        "solved_at_8": int(lane["solved_at_8"]),
        "verified_candidates": int(lane["verified_candidates"]),
        "verified_density": float(lane["verified_density"]),
        "normalized_template_diversity": float(
            lane["normalized_template_diversity"]
        ),
        "unique_normalized_templates": int(lane["unique_normalized_templates"]),
        "finish_reason_counts": finish_reasons,
        "eos_fraction": finish_reasons.get("eos", 0) / candidates,
        "generated_tokens": dict(lane["generated_tokens"]),
        "first_construct_counts": dict(lane["first_construct_counts"]),
        "template_statistics": [dict(item) for item in lane["template_statistics"]],
        "dominant_template": dict(lane["dominant_template"]),
    }


def _validation_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    summary = evidence["summary"]
    return {
        "combined": dict(summary["combined"]),
        "whole": _lane_summary(summary["whole"]),
        "incremental": _lane_summary(summary["incremental"]),
    }


def _compact_anchor_drift(drift: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_count": int(drift["anchor_count"]),
        "all_finite": drift["all_finite"] is True,
        "mean_anchor_kl": float(drift["mean_anchor_kl"]),
        "median_anchor_kl": float(drift["median_anchor_kl"]),
        "p95_anchor_kl": float(drift["p95_anchor_kl"]),
        "maximum_anchor_kl": float(drift["maximum_anchor_kl"]),
        "base_logits_sha256": drift["base_logits_sha256"],
        "checkpoint_adapter_model_sha256": drift[
            "checkpoint_adapter_model_sha256"
        ],
    }


def _rolling_mean(values: Sequence[float], window: int) -> list[float]:
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def _major_construct_counts(lane: Mapping[str, Any]) -> dict[str, int]:
    result = {construct: 0 for construct in MAJOR_CONSTRUCTS}
    for construct, count in lane["first_construct_counts"].items():
        if construct in {"simp", "simp_all", "simp_rw", "simpa"}:
            category = "simp"
        elif construct in {"intro", "intros"}:
            category = "intro"
        elif construct in result and construct != "other":
            category = construct
        else:
            category = "other"
        result[category] += int(count)
    return result


def _trajectory_metric_rows(
    base_point: Mapping[str, Any], points: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        point: Mapping[str, Any],
        interface: str,
        structural_group: str,
        metric: str,
        value: float | int,
        *,
        numerator: float | int | None = None,
        denominator: float | int | None = None,
    ) -> None:
        row = {
            "config": point.get("configuration_id", "Base"),
            "optimizer_step": int(point["optimizer_step"]),
            "interface": interface,
            "structural_group": structural_group,
            "metric": metric,
            "value": value,
        }
        if numerator is not None:
            row["numerator"] = numerator
        if denominator is not None:
            row["denominator"] = denominator
        rows.append(row)

    for point in (base_point, *points):
        validation = point["validation"]
        for interface in ("whole", "incremental"):
            lane = validation[interface]
            task_count = int(lane["task_count"])
            candidate_count = int(lane["candidate_count"])
            add(
                point,
                interface,
                "aggregate",
                "solved_at_8",
                int(lane["solved_at_8"]),
                numerator=int(lane["solved_at_8"]),
                denominator=task_count,
            )
            add(
                point,
                interface,
                "aggregate",
                "verified_candidate_density",
                float(lane["verified_density"]),
                numerator=int(lane["verified_candidates"]),
                denominator=candidate_count,
            )
            eos = int(lane["finish_reason_counts"].get("eos", 0))
            add(
                point,
                interface,
                "aggregate",
                "eos_finish_fraction",
                eos / candidate_count,
                numerator=eos,
                denominator=candidate_count,
            )
            for percentile in ("median", "p75", "p90"):
                add(
                    point,
                    interface,
                    "aggregate",
                    f"generated_tokens_{percentile}",
                    float(lane["generated_tokens"][percentile]),
                    denominator=candidate_count,
                )
            add(
                point,
                interface,
                "aggregate",
                "normalized_template_diversity",
                float(lane["normalized_template_diversity"]),
                numerator=int(lane["unique_normalized_templates"]),
                denominator=candidate_count,
            )
            for construct, count in _major_construct_counts(lane).items():
                add(
                    point,
                    interface,
                    "aggregate",
                    f"first_construct_fraction_{construct}",
                    count / candidate_count,
                    numerator=count,
                    denominator=candidate_count,
                )

        combined = validation["combined"]
        add(
            point,
            "combined",
            "aggregate",
            "solved_at_8",
            int(combined["solved_at_8"]),
            numerator=int(combined["solved_at_8"]),
            denominator=int(combined["interface_task_count"]),
        )
        add(
            point,
            "combined",
            "aggregate",
            "verified_candidate_density",
            float(combined["verified_density"]),
            numerator=int(combined["verified_candidates"]),
            denominator=int(combined["candidate_count"]),
        )
        add(
            point,
            "combined",
            "aggregate",
            "normalized_template_diversity",
            float(combined["normalized_template_diversity"]),
            denominator=int(combined["candidate_count"]),
        )
        base_solved = int(base_point["validation"]["combined"]["solved_at_8"])
        retained = (
            base_solved
            if point.get("configuration_id") is None
            else int(point["base_comparison"]["retained_base_solved_interface_tasks"])
        )
        add(
            point,
            "combined",
            "aggregate",
            "base_solved_retention_count",
            retained,
            numerator=retained,
            denominator=base_solved,
        )
        add(
            point,
            "combined",
            "aggregate",
            "mean_anchor_kl",
            float(point["anchor_drift"]["mean_anchor_kl"]),
            denominator=int(point["anchor_drift"]["anchor_count"]),
        )

        for structural_group in STRUCTURAL_GROUPS:
            structural = point["structural_summary"][structural_group]
            for interface in ("whole", "incremental"):
                lane = structural[interface]
                add(
                    point,
                    interface,
                    structural_group,
                    "solved_at_8",
                    int(lane["solved_at_8"]),
                    numerator=int(lane["solved_at_8"]),
                    denominator=int(lane["task_count"]),
                )
                add(
                    point,
                    interface,
                    structural_group,
                    "verified_candidate_density",
                    float(lane["verified_density"]),
                    numerator=int(lane["verified_candidates"]),
                    denominator=int(lane["candidate_count"]),
                )
    return rows


def _svg_chart(
    *,
    title: str,
    panels: Sequence[tuple[str, str, Sequence[tuple[str, Sequence[tuple[float, float]]]]]],
) -> str:
    width = 1000
    panel_height = 245
    height = 85 + panel_height * len(panels)
    left, right = 80.0, 965.0
    plot_width = right - left
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#111827}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2.5}.point{stroke:#fff;stroke-width:1}</style>',
        f'<text x="{left}" y="38" font-size="22" font-weight="700">{title}</text>',
    ]
    for panel_index, (panel_title, y_label, series) in enumerate(panels):
        top = 70.0 + panel_index * panel_height
        bottom = top + 170.0
        all_points = [point for _, points in series for point in points]
        x_values = [point[0] for point in all_points] or [0.0, 1.0]
        y_values = [point[1] for point in all_points] or [0.0, 1.0]
        x_min, x_max = min(x_values), max(x_values)
        y_min = min(0.0, min(y_values))
        y_max = max(y_values)
        if y_max <= y_min:
            y_max = y_min + 1.0
        y_max *= 1.08

        def x_coord(value: float) -> float:
            return left + (value - x_min) / max(1.0, x_max - x_min) * plot_width

        def y_coord(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        elements.append(
            f'<text x="{left}" y="{top - 12}" font-size="16" font-weight="700">{panel_title}</text>'
        )
        for tick in range(5):
            fraction = tick / 4
            y = bottom - fraction * (bottom - top)
            value = y_min + fraction * (y_max - y_min)
            elements.extend(
                [
                    f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>',
                    f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="11" text-anchor="end">{value:.4g}</text>',
                ]
            )
        elements.extend(
            [
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                f'<text x="{left - 58}" y="{(top + bottom) / 2}" font-size="11" text-anchor="middle" transform="rotate(-90 {left - 58} {(top + bottom) / 2})">{y_label}</text>',
            ]
        )
        for x_tick in (0, 100, 250, 500):
            if x_tick < x_min or x_tick > x_max:
                continue
            x = x_coord(float(x_tick))
            elements.append(
                f'<text x="{x:.2f}" y="{bottom + 20}" font-size="11" text-anchor="middle">{x_tick}</text>'
            )
        legend_x = right - 90 * len(series)
        for series_index, (identifier, points) in enumerate(series):
            color = COLORS[identifier]
            coords = " ".join(
                f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}" for x, y in points
            )
            elements.append(
                f'<polyline class="line" stroke="{color}" points="{coords}"/>'
            )
            for x, y in points:
                elements.append(
                    f'<circle class="point" fill="{color}" cx="{x_coord(float(x)):.2f}" cy="{y_coord(float(y)):.2f}" r="3.5"/>'
                )
            lx = legend_x + series_index * 90
            elements.extend(
                [
                    f'<line x1="{lx}" y1="{top - 18}" x2="{lx + 22}" y2="{top - 18}" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{lx + 28}" y="{top - 14}" font-size="11">{identifier}</text>',
                ]
            )
        elements.append(
            f'<text x="{(left + right) / 2}" y="{bottom + 38}" font-size="11" text-anchor="middle">optimizer step (Base = 0)</text>'
        )
    elements.append("</svg>\n")
    return "\n".join(elements)


def compact_bounded_trajectory_evidence(
    config: GeneralistV3Config,
    base_evidence_path: Path,
    validation_root: Path,
    training_root: Path,
    output_path: Path,
    validation_plot_path: Path,
    training_plot_path: Path,
) -> dict[str, Any]:
    base = _read_json(base_evidence_path)
    if (
        base.get("schema_version") != "generalist-v3-base-canary-evidence-v1"
        or base.get("status") != "passed"
        or base.get("protocol", {}).get("sealed_test_accessed") is not False
    ):
        raise ValueError("generalist-v3 trajectory requires the passed sealed Base canary")

    base_point = {
        "checkpoint_id": "Base",
        "optimizer_step": 0,
        "validation": _validation_summary(base),
        "structural_summary": base["structural_summary"],
        "anchor_drift": {
            "anchor_count": 512,
            "all_finite": True,
            "mean_anchor_kl": 0.0,
            "median_anchor_kl": 0.0,
            "p95_anchor_kl": 0.0,
            "maximum_anchor_kl": 0.0,
        },
        "compact_evidence_sha256": sha256_file(base_evidence_path),
    }
    points: list[dict[str, Any]] = []
    training_trajectories: dict[str, list[dict[str, Any]]] = {}
    shared_stream_identity: tuple[str, str] | None = None
    training_bindings: dict[str, Any] = {}

    for identifier in CONFIGURATION_IDS:
        run_dir = training_root / identifier
        training_path = run_dir / "training.json"
        log_path = run_dir / "training-log.jsonl"
        training = _read_json(training_path)
        rows, log_lines = _read_training_log(log_path)
        expected_configuration = config.training["configurations"][identifier]
        if (
            training.get("schema_version") != "generalist-v3-bounded-training-v1"
            or training.get("status") != "passed"
            or training.get("configuration_id") != identifier
            or training.get("optimizer_updates") != 500
            or training.get("optimizer_steps") != 500
            or training.get("sealed_test_accessed") is not False
            or training.get("configuration") != expected_configuration
            or len(rows) != 500
            or [int(row.get("optimizer_step", -1)) for row in rows]
            != list(range(1, 501))
            or training.get("training_log_sha256") != sha256_file(log_path)
        ):
            raise ValueError(f"generalist-v3 bounded training binding differs for {identifier}")
        stream_identity = (
            training["gates"]["training_stream_sha256"],
            training["gates"]["training_stream_manifest_sha256"],
        )
        if shared_stream_identity is None:
            shared_stream_identity = stream_identity
        elif stream_identity != shared_stream_identity:
            raise ValueError("generalist-v3 configurations used different training streams")
        training_bindings[identifier] = {
            "training_sha256": sha256_file(training_path),
            "training_log_sha256": sha256_file(log_path),
            "runtime": dict(training["runtime"]),
        }
        training_trajectories[identifier] = [
            {
                "optimizer_step": int(row["optimizer_step"]),
                "learning_rate": _finite_number(row["learning_rate"], "learning_rate"),
                "sft_loss": _finite_number(row["sft_loss_mean"], "sft_loss"),
                "preservation_kl": _finite_number(
                    row["preservation_kl"], "preservation_kl"
                ),
                "weighted_preservation_kl": _finite_number(
                    row["weighted_preservation_kl"], "weighted_preservation_kl"
                ),
                "gradient_norm_before_clipping": _finite_number(
                    row["gradient_norm_before_clipping"], "gradient_norm"
                ),
            }
            for row in rows
        ]

        for step in CHECKPOINT_STEPS:
            checkpoint = training["checkpoints"].get(str(step))
            evidence_path = validation_root / identifier / f"step-{step}.json"
            evidence = _read_json(evidence_path)
            prefix_sha256 = _sha256_bytes(b"".join(log_lines[:step]))
            if (
                checkpoint is None
                or checkpoint.get("schema_version")
                != "generalist-v3-training-checkpoint-v1"
                or checkpoint.get("training_log_sha256") != prefix_sha256
                or checkpoint.get("sealed_test_accessed") is not False
                or evidence.get("schema_version")
                != "generalist-v3-checkpoint-canary-evidence-v1"
                or evidence.get("status") != "passed"
                or evidence.get("configuration_id") != identifier
                or evidence.get("optimizer_step") != step
                or evidence.get("protocol", {}).get("sealed_test_accessed") is not False
                or evidence.get("adapter", {}).get("identity", {}).get(
                    "adapter_model_sha256"
                )
                != checkpoint.get("adapter_model_sha256")
            ):
                raise ValueError(
                    f"generalist-v3 checkpoint trajectory binding differs for {identifier}-{step}"
                )
            points.append(
                {
                    "checkpoint_id": f"{identifier}-{step}",
                    "configuration_id": identifier,
                    "configuration": dict(expected_configuration),
                    "optimizer_step": step,
                    "stream_rows_consumed": int(checkpoint["stream_rows_consumed"]),
                    "adapter_model_sha256": checkpoint["adapter_model_sha256"],
                    "training": _checkpoint_training_summary(rows, step),
                    "validation": _validation_summary(evidence),
                    "structural_summary": evidence["structural_summary"],
                    "base_comparison": {
                        key: evidence["base_comparison"][key]
                        for key in (
                            "base_solved_interface_tasks",
                            "retained_base_solved_interface_tasks",
                            "new_solved_interface_tasks",
                            "lost_base_solved_task_ids",
                            "mean_task_template_jaccard",
                        )
                    },
                    "anchor_drift": _compact_anchor_drift(evidence["anchor_drift"]),
                    "collapse_gates": dict(evidence["collapse_gates"]),
                    "positive_500_step_gate": evidence["positive_500_step_gate"],
                    "compact_evidence_sha256": sha256_file(evidence_path),
                }
            )

    by_checkpoint = {point["checkpoint_id"]: point for point in points}
    if any(
        by_checkpoint[f"{identifier}-500"]["positive_500_step_gate"] is not False
        for identifier in ("C1", "C2", "C3")
    ):
        raise ValueError("generalist-v3 trajectory does not satisfy the bounded SFT stop rule")
    if any(
        point["collapse_gates"]["eligible"] is not False
        for point in points
        if point["optimizer_step"] == 500 and point["configuration_id"] in ("C1", "C2", "C3")
    ):
        raise ValueError("generalist-v3 failed step-500 gate evidence differs")

    metric_rows = _trajectory_metric_rows(base_point, points)

    def metric_series(
        metric: str, interface: str, structural_group: str = "aggregate"
    ) -> list[tuple[str, list[tuple[float, float]]]]:
        base_values = [
            (float(row["optimizer_step"]), float(row["value"]))
            for row in metric_rows
            if row["config"] == "Base"
            and row["interface"] == interface
            and row["structural_group"] == structural_group
            and row["metric"] == metric
        ]
        if len(base_values) != 1:
            raise ValueError(
                f"generalist-v3 trajectory Base metric is incomplete: "
                f"{interface}/{structural_group}/{metric}"
            )
        series = []
        for identifier in CONFIGURATION_IDS:
            values = [
                (float(row["optimizer_step"]), float(row["value"]))
                for row in metric_rows
                if row["config"] == identifier
                and row["interface"] == interface
                and row["structural_group"] == structural_group
                and row["metric"] == metric
            ]
            series.append((identifier, sorted([*base_values, *values])))
        return series

    plot_root = output_path.parent
    plot_svgs: dict[str, tuple[Path, str]] = {
        "validation": (
            validation_plot_path,
            _svg_chart(
                title="Qwen-Lean v3 bounded validation overview",
                panels=(
                    (
                        "Combined Lean coverage",
                        "solved@8",
                        metric_series("solved_at_8", "combined"),
                    ),
                    (
                        "Combined Lean-valid density",
                        "verified density",
                        metric_series("verified_candidate_density", "combined"),
                    ),
                    (
                        "Combined output diversity",
                        "normalized diversity",
                        metric_series("normalized_template_diversity", "combined"),
                    ),
                    (
                        "Direct drift from Base",
                        "mean anchor KL",
                        metric_series("mean_anchor_kl", "combined"),
                    ),
                ),
            ),
        ),
        "coverage": (
            plot_root / "bounded-coverage-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 solved@8 trajectory by interface",
                panels=tuple(
                    (
                        f"{interface} solved@8",
                        "solved tasks",
                        metric_series("solved_at_8", interface),
                    )
                    for interface in ("whole", "incremental")
                ),
            ),
        ),
        "verified_density": (
            plot_root / "bounded-density-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 verified-candidate density by interface",
                panels=tuple(
                    (
                        f"{interface} verified density",
                        "verified density",
                        metric_series("verified_candidate_density", interface),
                    )
                    for interface in ("whole", "incremental")
                ),
            ),
        ),
        "base_retention": (
            plot_root / "bounded-base-retention-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 Base-solved retention (Base solved 0/96)",
                panels=(
                    (
                        "Retained Base-solved interface tasks",
                        "retained tasks",
                        metric_series("base_solved_retention_count", "combined"),
                    ),
                ),
            ),
        ),
        "eos": (
            plot_root / "bounded-eos-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 EOS finish fraction by interface",
                panels=tuple(
                    (
                        f"{interface} EOS fraction",
                        "EOS fraction",
                        metric_series("eos_finish_fraction", interface),
                    )
                    for interface in ("whole", "incremental")
                ),
            ),
        ),
        "length": (
            plot_root / "bounded-length-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 generated-token trajectory by interface",
                panels=tuple(
                    (
                        f"{interface} generated tokens {percentile}",
                        "generated tokens",
                        metric_series(f"generated_tokens_{percentile}", interface),
                    )
                    for interface in ("whole", "incremental")
                    for percentile in ("median", "p75", "p90")
                ),
            ),
        ),
        "diversity": (
            plot_root / "bounded-diversity-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 normalized-template diversity by interface",
                panels=tuple(
                    (
                        f"{interface} normalized diversity",
                        "normalized diversity",
                        metric_series("normalized_template_diversity", interface),
                    )
                    for interface in ("whole", "incremental")
                ),
            ),
        ),
        "first_construct": (
            plot_root / "bounded-first-construct-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 first-proof-construct fractions",
                panels=tuple(
                    (
                        f"{interface}: {construct}",
                        "candidate fraction",
                        metric_series(
                            f"first_construct_fraction_{construct}", interface
                        ),
                    )
                    for interface in ("whole", "incremental")
                    for construct in MAJOR_CONSTRUCTS
                ),
            ),
        ),
        "structural": (
            plot_root / "bounded-structural-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 structural capability trajectory",
                panels=tuple(
                    (
                        f"{structural_group} {interface} {label}",
                        y_label,
                        metric_series(metric, interface, structural_group),
                    )
                    for structural_group in STRUCTURAL_GROUPS
                    for interface in ("whole", "incremental")
                    for metric, label, y_label in (
                        ("solved_at_8", "solved@8", "solved tasks"),
                        (
                            "verified_candidate_density",
                            "verified density",
                            "verified density",
                        ),
                    )
                ),
            ),
        ),
        "drift": (
            plot_root / "bounded-drift-trajectory.svg",
            _svg_chart(
                title="Qwen-Lean v3 Base-preservation drift",
                panels=(
                    (
                        "Mean anchor KL",
                        "mean anchor KL",
                        metric_series("mean_anchor_kl", "combined"),
                    ),
                ),
            ),
        ),
        "training": (
            training_plot_path,
            _svg_chart(
                title="Qwen-Lean v3 raw per-step optimizer trajectory",
                panels=tuple(
                    (
                        title,
                        label,
                        [
                            (
                                identifier,
                                [
                                    (
                                        float(row["optimizer_step"]),
                                        float(row[field]),
                                    )
                                    for row in training_trajectories[identifier]
                                ],
                            )
                            for identifier in CONFIGURATION_IDS
                        ],
                    )
                    for title, label, field in (
                        ("Target-only SFT loss", "loss", "sft_loss"),
                        (
                            "Unweighted Base-preservation KL",
                            "KL",
                            "preservation_kl",
                        ),
                        (
                            "Weighted Base-preservation KL",
                            "weighted KL",
                            "weighted_preservation_kl",
                        ),
                        (
                            "Gradient norm before clipping",
                            "gradient norm",
                            "gradient_norm_before_clipping",
                        ),
                    )
                ),
            ),
        ),
    }
    for path, svg in plot_svgs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")
    plot_records = {
        name: {"path": path.name, "sha256": sha256_file(path)}
        for name, (path, _) in plot_svgs.items()
    }

    evidence = {
        "schema_version": "generalist-v3-bounded-trajectory-evidence-v1",
        "status": "passed",
        "marker": "OBSERVED",
        "outcome": {
            "sft_status": "stopped",
            "selected_checkpoint": None,
            "checkpoint_frozen": False,
            "reason": "C1 and C2 failed the positive 500-step gate; the single authorized C3 rescue also failed",
            "continuation_optimizer_updates_authorized": False,
            "sealed_test_evaluation_authorized": False,
        },
        "base": base_point,
        "checkpoints": points,
        "validation_metric_rows": metric_rows,
        "training_trajectories": training_trajectories,
        "training_bindings": training_bindings,
        "shared_training_stream": {
            "training_stream_sha256": shared_stream_identity[0],
            "training_stream_manifest_sha256": shared_stream_identity[1],
            "all_configurations_matched": True,
        },
        "plots": plot_records,
        "protocol": {
            "base_and_all_retained_checkpoints_included": True,
            "configurations": list(CONFIGURATION_IDS),
            "retained_optimizer_steps": list(CHECKPOINT_STEPS),
            "maximum_optimizer_updates": 500,
            "optimizer_updates_beyond_500": 0,
            "all_validation_compositions_evaluated": True,
            "structural_groups_available": list(STRUCTURAL_GROUPS),
            "raw_training_objective_rows_included": True,
            "raw_retained_checkpoint_values_included": True,
            "truncation": False,
            "silent_exclusions": 0,
            "sealed_test_accessed": False,
        },
    }
    _write_json(output_path, evidence)
    return evidence
