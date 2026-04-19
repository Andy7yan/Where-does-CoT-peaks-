"""Stage 1 plotting for the v8 figure set."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_phase2.coarse_analysis import DIFFICULTY_ORDER


DEFAULT_NORMALIZED_BINS = 6
OVERALL_ANALYSIS_DIRNAME = "analysis"
LEGACY_OVERALL_ANALYSIS_DIRNAME = "analysis_phase1"
PQ_ANALYSIS_DIRNAME = "pq_analysis"


def run_stage1_plotting(
    *,
    overall_run_dir: str | Path,
    pq_run_dir: str | Path,
    output_dir: str | Path,
    representative_questions: Sequence[str] | None = None,
    max_heatmap_questions: int = 5,
    normalized_bins: int = DEFAULT_NORMALIZED_BINS,
) -> dict[str, Any]:
    """Render the v8 static plots for overall and PQ pipelines."""

    overall_run_path = Path(overall_run_dir)
    pq_run_path = Path(pq_run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    overall_analysis_dir = ensure_overall_analysis_views(overall_run_path)
    pq_analysis_dir = pq_run_path / PQ_ANALYSIS_DIRNAME
    if not pq_analysis_dir.exists():
        raise FileNotFoundError(f"Missing pq_analysis directory: {pq_analysis_dir}")

    t1a_rows = _read_csv(overall_analysis_dir / "t1a_overview.csv")
    pq_step_rows = _read_csv(pq_analysis_dir / "t1b_step_surface.csv")
    pq_ratio_rows = _read_csv(pq_analysis_dir / "t1c_kstar_ratio.csv")
    pq_lstar_rows = _read_csv(pq_analysis_dir / "t2b_lstar_difficulty.csv")

    selected_questions = (
        list(representative_questions)
        if representative_questions
        else select_representative_questions(
            pq_step_rows,
            max_questions=max_heatmap_questions,
        )
    )

    generated_paths: list[str] = []
    generated_paths.extend(
        plot_t1a_overview(
            rows=t1a_rows,
            output_dir=output_path / "t1a",
        )
    )
    generated_paths.extend(
        plot_t1b_heatmaps(
            rows=pq_step_rows,
            output_dir=output_path / "t1b",
            question_ids=selected_questions,
        )
    )
    generated_paths.extend(
        plot_t1b_normalized_heatmaps(
            rows=pq_step_rows,
            output_dir=output_path / "t1b_norm",
            question_ids=selected_questions,
            normalized_bins=normalized_bins,
        )
    )
    generated_paths.append(
        plot_t1c_scatter(
            rows=pq_ratio_rows,
            output_path=output_path / "t1c_kstar_ratio_vs_difficulty.png",
        )
    )
    generated_paths.append(
        plot_t2b_scatter(
            rows=pq_lstar_rows,
            output_path=output_path / "t2b_lstar_vs_difficulty.png",
        )
    )

    return {
        "overall_analysis_dir": str(overall_analysis_dir),
        "pq_analysis_dir": str(pq_analysis_dir),
        "output_dir": str(output_path),
        "representative_questions": selected_questions,
        "generated_files": generated_paths,
    }


def ensure_overall_analysis_views(run_dir: str | Path) -> Path:
    """Ensure the overall pipeline exposes v8 analysis view CSVs."""

    run_path = Path(run_dir)
    analysis_dir = run_path / OVERALL_ANALYSIS_DIRNAME
    required = (
        analysis_dir / "t1a_overview.csv",
        analysis_dir / "t1b_step_surface.csv",
        analysis_dir / "bin_status.csv",
        analysis_dir / "failure_stats.csv",
        analysis_dir / "S_calibration.json",
    )
    if all(path.exists() for path in required):
        return analysis_dir

    legacy_dir = run_path / LEGACY_OVERALL_ANALYSIS_DIRNAME
    if not legacy_dir.exists():
        raise FileNotFoundError(
            f"Missing overall analysis source directory. Checked '{analysis_dir}' and '{legacy_dir}'."
        )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    t1a_rows = build_t1a_overview_rows_from_legacy(legacy_dir)
    t1b_rows = build_t1b_step_surface_rows_from_legacy(legacy_dir)
    legacy_bin_status = build_v8_bin_status_from_legacy(legacy_dir)
    legacy_failure_rows = build_v8_failure_stats_from_legacy(legacy_dir)

    _write_csv(
        analysis_dir / "t1a_overview.csv",
        rows=t1a_rows,
        fieldnames=[
            "difficulty",
            "L",
            "accuracy",
            "accuracy_se",
            "k_star",
            "mean_tas",
            "tas_se",
            "l_star",
            "bin_status",
            "n_traces",
            "n_clean",
        ],
    )
    _write_csv(
        analysis_dir / "t1b_step_surface.csv",
        rows=t1b_rows,
        fieldnames=[
            "scope",
            "pipeline",
            "L",
            "step",
            "mean_nldd",
            "nldd_se",
            "mean_tas_t",
            "tas_t_se",
            "n_clean",
            "bin_status",
        ],
    )
    _write_csv(
        analysis_dir / "bin_status.csv",
        rows=legacy_bin_status,
        fieldnames=[
            "scope",
            "pipeline",
            "L",
            "n_total_traces",
            "n_correct",
            "n_tier1",
            "n_tier2",
            "n_failed",
            "n_retained",
            "bin_status",
        ],
    )
    _write_csv(
        analysis_dir / "failure_stats.csv",
        rows=legacy_failure_rows,
        fieldnames=["scope", "pipeline", "key", "count"],
    )
    legacy_s_path = legacy_dir / "S_calibration.json"
    if legacy_s_path.exists():
        payload = json.loads(legacy_s_path.read_text(encoding="utf-8"))
        payload["pipeline"] = "v6"
        (analysis_dir / "S_calibration.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return analysis_dir


def build_t1a_overview_rows_from_legacy(legacy_dir: Path) -> list[dict[str, Any]]:
    """Build the unified t1a overview view from legacy overall outputs."""

    accuracy_rows = _read_csv(legacy_dir / "accuracy_by_length.csv")
    kstar_rows = {
        (str(row["difficulty"]), int(row["length"])): row
        for row in _read_csv(legacy_dir / "k_star_by_L.csv")
    }
    lstar_by_difficulty = {
        str(row["difficulty"]): int(row["L_star"])
        for row in _read_csv(legacy_dir / "L_star.csv")
    }
    bin_status_by_key = {
        (str(row["difficulty"]), int(row["length"])): row
        for row in _read_csv(legacy_dir / "bin_status.csv")
    }
    tas_rows = _load_jsonl(legacy_dir / "tas_per_trace.jsonl")
    tas_grouped: dict[tuple[str, int], list[float]] = {}
    for row in tas_rows:
        tas_grouped.setdefault(
            (str(row["difficulty"]), int(row["length"])),
            [],
        ).append(float(row["tas_value"]))

    overview_rows: list[dict[str, Any]] = []
    for row in accuracy_rows:
        difficulty = str(row["difficulty"])
        length = int(row["length"])
        status_row = bin_status_by_key.get((difficulty, length), {})
        tas_values = tas_grouped.get((difficulty, length), [])
        kstar_row = kstar_rows.get((difficulty, length))
        bin_status = str(status_row.get("status", "insufficient"))
        overview_rows.append(
            {
                "difficulty": difficulty,
                "L": length,
                "accuracy": float(row["mean_accuracy"]),
                "accuracy_se": float(row["se_accuracy"]),
                "k_star": int(kstar_row["k_star"]) if kstar_row and bin_status == "ok" else None,
                "mean_tas": sum(tas_values) / len(tas_values) if tas_values else None,
                "tas_se": _standard_error(tas_values) if tas_values else None,
                "l_star": lstar_by_difficulty.get(difficulty) == length,
                "bin_status": bin_status,
                "n_traces": int(row["n"]),
                "n_clean": int(status_row.get("selected_samples", 0)),
            }
        )
    return overview_rows


def build_t1b_step_surface_rows_from_legacy(legacy_dir: Path) -> list[dict[str, Any]]:
    """Build the unified step surface rows for the overall pipeline."""

    nldd_rows = {
        (str(row["difficulty"]), int(row["length"]), int(row["k"])): row
        for row in _read_csv(legacy_dir / "nldd_surface.csv")
    }
    tas_rows = {
        (str(row["difficulty"]), int(row["length"]), int(row["step_index"])): row
        for row in _read_csv(legacy_dir / "tas_curve.csv")
    }
    bin_status_by_key = {
        (str(row["difficulty"]), int(row["length"])): row
        for row in _read_csv(legacy_dir / "bin_status.csv")
    }

    keys = set((difficulty, length) for difficulty, length, _ in nldd_rows) | set(
        (difficulty, length) for difficulty, length, _ in tas_rows
    )
    surface_rows: list[dict[str, Any]] = []
    for difficulty, length in sorted(
        keys,
        key=lambda item: (DIFFICULTY_ORDER.index(item[0]), item[1]),
    ):
        status_row = bin_status_by_key.get((difficulty, length), {})
        for step in range(1, length + 1):
            nldd_row = nldd_rows.get((difficulty, length, step))
            tas_row = tas_rows.get((difficulty, length, step))
            surface_rows.append(
                {
                    "scope": difficulty,
                    "pipeline": "v6",
                    "L": length,
                    "step": step,
                    "mean_nldd": float(nldd_row["mean_nldd"]) if nldd_row else None,
                    "nldd_se": float(nldd_row["se_nldd"]) if nldd_row else None,
                    "mean_tas_t": float(tas_row["mean_tas"]) if tas_row else None,
                    "tas_t_se": float(tas_row["se_tas"]) if tas_row else None,
                    "n_clean": int(status_row.get("selected_samples", 0)),
                    "bin_status": str(status_row.get("status", "insufficient")),
                }
            )
    return surface_rows


def build_v8_bin_status_from_legacy(legacy_dir: Path) -> list[dict[str, Any]]:
    """Convert legacy overall bin_status rows to the v8 schema."""

    rows = _read_csv(legacy_dir / "bin_status.csv")
    return [
        {
            "scope": str(row["difficulty"]),
            "pipeline": "v6",
            "L": int(row["length"]),
            "n_total_traces": int(row["trace_total"]),
            "n_correct": int(row["eligible_clean_traces"]),
            "n_tier1": int(row["tier1_samples"]),
            "n_tier2": int(row["tier2_samples"]),
            "n_failed": max(
                int(row["eligible_clean_traces"]) - int(row["selected_samples"]),
                0,
            ),
            "n_retained": int(row["selected_samples"]),
            "bin_status": str(row["status"]),
        }
        for row in rows
    ]


def build_v8_failure_stats_from_legacy(legacy_dir: Path) -> list[dict[str, Any]]:
    """Convert the compact legacy failure table to a lightweight v8-compatible view."""

    rows = _read_csv(legacy_dir / "failure_stats.csv")
    return [
        {
            "scope": "overall",
            "pipeline": "v6",
            "key": str(row["key"]),
            "count": int(row["count"]),
        }
        for row in rows
    ]


def select_representative_questions(
    rows: Sequence[dict[str, Any]],
    *,
    max_questions: int,
) -> list[str]:
    """Pick representative PQ questions by ok-bin coverage, then by scope id."""

    coverage: dict[str, int] = {}
    for row in rows:
        if str(row.get("pipeline")) != "pq":
            continue
        if str(row.get("bin_status")) != "ok":
            continue
        scope = str(row["scope"])
        coverage[scope] = coverage.get(scope, 0) + 1
    ordered = sorted(
        coverage,
        key=lambda scope: (-coverage[scope], scope),
    )
    return ordered[: max(max_questions, 0)]


def plot_t1a_overview(
    *,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Render the three overall T1-A multi-axis figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame["L"] = pd.to_numeric(frame["L"], errors="coerce")
    frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
    frame["mean_tas"] = pd.to_numeric(frame["mean_tas"], errors="coerce")
    frame["k_star"] = pd.to_numeric(frame["k_star"], errors="coerce")
    frame["l_star"] = frame["l_star"].astype(str).str.lower() == "true"
    frame = frame[frame["bin_status"] == "ok"].copy()
    if frame.empty:
        return []

    sns.set_theme(style="whitegrid")
    generated: list[str] = []
    for difficulty in DIFFICULTY_ORDER:
        subset = frame[frame["difficulty"] == difficulty].sort_values("L")
        if subset.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        ax1.plot(subset["L"], subset["accuracy"], marker="o", color="#1f6feb", label="accuracy")
        ax1.plot(subset["L"], subset["mean_tas"], marker="s", color="#ff7f0e", label="mean_tas")
        ax2.plot(subset["L"], subset["k_star"], marker="^", color="#2ca02c", label="k_star")
        ax2.plot(subset["L"], subset["L"], linestyle="--", color="#666666", label="y=x")

        lstar_rows = subset[subset["l_star"]]
        if not lstar_rows.empty:
            ax1.axvline(
                float(lstar_rows.iloc[0]["L"]),
                linestyle=":",
                color="#aa3377",
                linewidth=1.5,
            )

        ax1.set_title(f"T1-A Overview: {difficulty}")
        ax1.set_xlabel("L")
        ax1.set_ylabel("accuracy / mean_tas")
        ax2.set_ylabel("k_star")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

        fig.tight_layout()
        output_path = output_dir / f"t1a_{difficulty}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        generated.append(str(output_path))
    return generated


def plot_t1b_heatmaps(
    *,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    question_ids: Sequence[str],
) -> list[str]:
    """Render side-by-side NLDD/TAS heatmaps for selected PQ questions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame = frame[(frame["pipeline"] == "pq") & (frame["scope"].isin(question_ids))].copy()
    if frame.empty:
        return []
    frame["L"] = pd.to_numeric(frame["L"], errors="coerce")
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["mean_nldd"] = pd.to_numeric(frame["mean_nldd"], errors="coerce")
    frame["mean_tas_t"] = pd.to_numeric(frame["mean_tas_t"], errors="coerce")

    generated: list[str] = []
    for question_id in question_ids:
        subset = frame[frame["scope"] == question_id].copy()
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        _render_heatmap(
            axes[0],
            subset,
            value_column="mean_nldd",
            title=f"{question_id} NLDD",
            cbar_label="mean_nldd",
        )
        _render_heatmap(
            axes[1],
            subset,
            value_column="mean_tas_t",
            title=f"{question_id} TAS_t",
            cbar_label="mean_tas_t",
        )
        output_path = output_dir / f"t1b_{question_id}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        generated.append(str(output_path))
    return generated


def plot_t1b_normalized_heatmaps(
    *,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    question_ids: Sequence[str],
    normalized_bins: int,
) -> list[str]:
    """Render normalized x-axis heatmaps over step/L bins."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame = frame[(frame["pipeline"] == "pq") & (frame["scope"].isin(question_ids))].copy()
    if frame.empty:
        return []
    frame["L"] = pd.to_numeric(frame["L"], errors="coerce")
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["mean_nldd"] = pd.to_numeric(frame["mean_nldd"], errors="coerce")
    frame["mean_tas_t"] = pd.to_numeric(frame["mean_tas_t"], errors="coerce")

    frame["ratio_bin"] = frame.apply(
        lambda row: min(
            max(int(math.ceil((float(row["step"]) / float(row["L"])) * normalized_bins)), 1),
            normalized_bins,
        ),
        axis=1,
    )

    generated: list[str] = []
    for question_id in question_ids:
        subset = frame[frame["scope"] == question_id].copy()
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        _render_normalized_heatmap(
            axes[0],
            subset,
            value_column="mean_nldd",
            normalized_bins=normalized_bins,
            title=f"{question_id} NLDD (normalized)",
            cbar_label="mean_nldd",
        )
        _render_normalized_heatmap(
            axes[1],
            subset,
            value_column="mean_tas_t",
            normalized_bins=normalized_bins,
            title=f"{question_id} TAS_t (normalized)",
            cbar_label="mean_tas_t",
        )
        output_path = output_dir / f"t1b_norm_{question_id}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        generated.append(str(output_path))
    return generated


def plot_t1c_scatter(
    *,
    rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> str:
    """Render the k*/L vs difficulty scatter plot."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("t1c_kstar_ratio.csv is empty; cannot render T1-C.")
    frame["difficulty_score"] = pd.to_numeric(frame["difficulty_score"], errors="coerce")
    frame["k_star_ratio"] = pd.to_numeric(frame["k_star_ratio"], errors="coerce")
    frame["L"] = pd.to_numeric(frame["L"], errors="coerce")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=frame,
        x="difficulty_score",
        y="k_star_ratio",
        hue="L",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("T1-C: k*/L vs difficulty")
    ax.set_xlabel("difficulty_score")
    ax.set_ylabel("k_star_ratio")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_t2b_scatter(
    *,
    rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> str:
    """Render the L* vs difficulty scatter plot with Pearson stats."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("t2b_lstar_difficulty.csv is empty; cannot render T2-B.")
    frame["difficulty_score"] = pd.to_numeric(frame["difficulty_score"], errors="coerce")
    frame["l_star_A"] = pd.to_numeric(frame["l_star_A"], errors="coerce")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=frame,
        x="difficulty_score",
        y="l_star_A",
        ax=ax,
        color="#1f6feb",
    )
    r_value, p_value = _pearson_with_normal_pvalue(
        frame["difficulty_score"].astype(float).tolist(),
        frame["l_star_A"].astype(float).tolist(),
    )
    ax.set_title("T2-B: L* vs difficulty")
    ax.set_xlabel("difficulty_score")
    ax.set_ylabel("l_star_A")
    ax.text(
        0.02,
        0.98,
        f"Pearson r = {r_value:.3f}\np-value = {p_value:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _render_heatmap(
    ax: Any,
    rows: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    cbar_label: str,
) -> None:
    pivot = rows.pivot_table(
        index="L",
        columns="step",
        values=value_column,
        aggfunc="mean",
    ).sort_index()
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": cbar_label},
        mask=pivot.isna(),
    )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("L")


def _render_normalized_heatmap(
    ax: Any,
    rows: pd.DataFrame,
    *,
    value_column: str,
    normalized_bins: int,
    title: str,
    cbar_label: str,
) -> None:
    pivot = rows.pivot_table(
        index="L",
        columns="ratio_bin",
        values=value_column,
        aggfunc="mean",
    ).sort_index()
    for bin_index in range(1, normalized_bins + 1):
        if bin_index not in pivot.columns:
            pivot[bin_index] = math.nan
    pivot = pivot[sorted(pivot.columns)]
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="magma",
        cbar_kws={"label": cbar_label},
        mask=pivot.isna(),
    )
    ax.set_title(title)
    ax.set_xlabel("step / L bin")
    ax.set_ylabel("L")


def _pearson_with_normal_pvalue(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("Pearson inputs must have equal length.")
    if len(left) < 2:
        return 0.0, 1.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denom_left = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    denom_right = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if denom_left <= 0.0 or denom_right <= 0.0:
        return 0.0, 1.0
    r_value = numerator / (denom_left * denom_right)
    r_value = max(min(r_value, 1.0), -1.0)
    if len(left) <= 2 or abs(r_value) >= 1.0:
        return r_value, 0.0
    t_value = abs(r_value) * math.sqrt((len(left) - 2) / max(1.0e-12, 1.0 - (r_value**2)))
    # Normal approximation keeps plotting self-contained without adding scipy.
    p_value = 2.0 * (1.0 - NormalDist().cdf(t_value))
    return r_value, p_value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_csv(path: Path, *, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _standard_error(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


__all__ = [
    "build_t1a_overview_rows_from_legacy",
    "build_t1b_step_surface_rows_from_legacy",
    "ensure_overall_analysis_views",
    "plot_t1a_overview",
    "plot_t1b_heatmaps",
    "plot_t1b_normalized_heatmaps",
    "plot_t1c_scatter",
    "plot_t2b_scatter",
    "run_stage1_plotting",
    "select_representative_questions",
]
