"""Post-horizon deep dive and outlier analysis on PQ Phase 1 artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize


PNG_DPI = 300
EPSILON = 1e-6
PAPER_PALETTE = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#6B7280",
    "dark": "#111827",
}
SEARCH_DIR_PRIORITY = ("pq_analysis", "analysis", "analysis_phase1")
FILE_PRIORITY = ("t1b_step_surface.csv", "t1c_kstar_ratio.csv", "t2b_lstar_difficulty.csv")


def _apply_paper_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.25)
    plt.rcParams.update(
        {
            "figure.dpi": PNG_DPI,
            "savefig.dpi": PNG_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#D1D5DB",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _polish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def _difficulty_norm(values: pd.Series) -> Normalize:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    return Normalize(vmin=vmin, vmax=vmax)


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _print_saved(path: Path) -> None:
    print(f"saved: {path}")


def _discover_csvs(search_root: Path) -> dict[str, Path | None]:
    matches: dict[str, list[Path]] = {name: [] for name in FILE_PRIORITY}
    for name in FILE_PRIORITY:
        for candidate in search_root.rglob(name):
            if candidate.is_file() and candidate.stat().st_size > 0:
                matches[name].append(candidate)

    def sort_key(path: Path) -> tuple[int, int, int, str]:
        text = str(path).replace("\\", "/")
        dir_rank = min((idx for idx, token in enumerate(SEARCH_DIR_PRIORITY) if f"/{token}/" in f"/{text}/"), default=99)
        papery_rank = 0 if "papery-pq" in text else 1
        path_rank = 0 if "path-pq" in text else 1
        return (dir_rank, papery_rank, path_rank, text)

    return {
        name: (sorted(paths, key=sort_key)[0] if paths else None)
        for name, paths in matches.items()
    }


def _load_csv(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        _warn(f"Could not locate {label}.")
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        _warn(f"Failed reading {label} from {path}: {exc}")
        return None
    if frame.empty:
        _warn(f"{label} at {path} is empty.")
        return None
    return frame


def _normalize_t1b(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    result = frame.copy()
    if "scope" in result.columns and "question_id" not in result.columns:
        result = result.rename(columns={"scope": "question_id"})
    if "question_id" not in result.columns:
        _warn("t1b_step_surface.csv is missing question_id/scope.")
        return None
    if "pipeline" in result.columns:
        result = result[result["pipeline"].astype(str) == "pq"].copy()
    required = {"question_id", "L", "step", "mean_nldd", "nldd_se", "mean_tas_t", "tas_t_se", "n_clean", "bin_status"}
    missing = required - set(result.columns)
    if missing:
        _warn(f"t1b_step_surface.csv is missing columns: {sorted(missing)}")
        return None
    result["question_id"] = result["question_id"].astype(str)
    for col in ("L", "step", "mean_nldd", "nldd_se", "mean_tas_t", "tas_t_se", "n_clean"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["bin_status"] = result["bin_status"].astype(str)
    return result


def _normalize_t1c(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    required = {"question_id", "difficulty_score", "L", "k_star", "k_star_ratio", "n_clean"}
    missing = required - set(frame.columns)
    if missing:
        _warn(f"t1c_kstar_ratio.csv is missing columns: {sorted(missing)}")
        return None
    result = frame.copy()
    result["question_id"] = result["question_id"].astype(str)
    for col in ("difficulty_score", "L", "k_star", "k_star_ratio", "n_clean"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _normalize_t2b(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    result = frame.copy()
    if "l_star_C" not in result.columns and "l_star_S" in result.columns:
        result = result.rename(columns={"l_star_S": "l_star_C"})
    required = {"question_id", "difficulty_score", "l_star_A", "l_star_C", "l_star_consistent"}
    missing = required - set(result.columns)
    if missing:
        _warn(f"t2b_lstar_difficulty.csv is missing columns: {sorted(missing)}")
        return None
    result["question_id"] = result["question_id"].astype(str)
    for col in ("difficulty_score", "l_star_A", "l_star_C"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for col in columns:
        if col not in output.columns:
            output[col] = np.nan
    output = output.loc[:, columns]
    output.to_csv(path, index=False)
    _print_saved(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_saved(path)


def _difficulty_groups(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    groups = pd.Series(index=values.index, dtype="object")
    if valid.empty:
        groups[:] = "unknown"
        return groups
    quantiles = valid.quantile([0.0, 1 / 3, 2 / 3, 1.0]).to_numpy(dtype=float)
    edges = np.unique(quantiles)
    if len(edges) < 4:
        min_val = float(valid.min())
        max_val = float(valid.max())
        if math.isclose(min_val, max_val, abs_tol=EPSILON):
            groups.loc[valid.index] = "medium"
            groups = groups.fillna("unknown")
            return groups
        edges = np.linspace(min_val, max_val, 4)
    groups.loc[values.notna()] = pd.cut(
        values[values.notna()],
        bins=edges,
        labels=["low", "medium", "high"],
        include_lowest=True,
        duplicates="drop",
    ).astype(str)
    groups = groups.fillna("unknown")
    return groups


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if abs(denominator) < EPSILON:
        return math.nan
    return float(numerator / denominator)


def analyze_post_horizon_nldd(t1b: pd.DataFrame | None, t1c: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "question_id",
        "L",
        "k_star",
        "pre_kstar_mean_nldd",
        "post_kstar_mean_nldd",
        "post_kstar_sign",
        "nldd_drop_ratio",
        "difficulty_score",
        "n_post_steps",
    ]
    if t1b is None or t1c is None:
        _warn("Skipping post-horizon NLDD analysis because t1b/t1c is unavailable.")
        return pd.DataFrame(columns=columns)

    ok_rows = t1b[t1b["bin_status"] == "ok"].copy()
    if ok_rows.empty:
        _warn("No ok rows available for post-horizon NLDD analysis.")
        return pd.DataFrame(columns=columns)

    lookup = t1c.set_index(["question_id", "L"])[["k_star", "difficulty_score"]]
    rows: list[dict[str, Any]] = []
    for (question_id, length), group in ok_rows.groupby(["question_id", "L"], sort=False):
        if (question_id, length) not in lookup.index:
            continue
        meta = lookup.loc[(question_id, length)]
        k_star = float(meta["k_star"])
        difficulty_score = float(meta["difficulty_score"])
        group = group.sort_values("step")
        pre_values = group.loc[(group["step"] >= 2) & (group["step"] <= k_star), "mean_nldd"].dropna().to_numpy(dtype=float)
        post_values = group.loc[group["step"] > k_star, "mean_nldd"].dropna().to_numpy(dtype=float)
        pre_mean = float(np.mean(pre_values)) if len(pre_values) else math.nan
        post_mean = float(np.mean(post_values)) if len(post_values) else math.nan
        if math.isnan(post_mean):
            post_sign = np.nan
        elif post_mean > 5:
            post_sign = "positive"
        elif post_mean < -5:
            post_sign = "negative"
        else:
            post_sign = "near_zero"
        rows.append(
            {
                "question_id": question_id,
                "L": float(length),
                "k_star": k_star,
                "pre_kstar_mean_nldd": pre_mean,
                "post_kstar_mean_nldd": post_mean,
                "post_kstar_sign": post_sign,
                "nldd_drop_ratio": _safe_ratio(post_mean, pre_mean),
                "difficulty_score": difficulty_score,
                "n_post_steps": int(len(post_values)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def analyze_post_horizon_tas_slope(t1b: pd.DataFrame | None, t1c: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "question_id",
        "L",
        "k_star",
        "pre_kstar_tas_slope",
        "post_kstar_tas_slope",
        "slope_change_ratio",
        "post_slope_category",
        "difficulty_score",
    ]
    if t1b is None or t1c is None:
        _warn("Skipping post-horizon TAS slope analysis because t1b/t1c is unavailable.")
        return pd.DataFrame(columns=columns)

    ok_rows = t1b[t1b["bin_status"] == "ok"].copy()
    if ok_rows.empty:
        _warn("No ok rows available for post-horizon TAS slope analysis.")
        return pd.DataFrame(columns=columns)

    lookup = t1c.set_index(["question_id", "L"])[["k_star", "difficulty_score"]]
    rows: list[dict[str, Any]] = []
    for (question_id, length), group in ok_rows.groupby(["question_id", "L"], sort=False):
        if (question_id, length) not in lookup.index:
            continue
        meta = lookup.loc[(question_id, length)]
        k_star = float(meta["k_star"])
        difficulty_score = float(meta["difficulty_score"])
        group = group.sort_values("step")
        tas_rows = group.loc[group["mean_tas_t"].notna(), ["step", "mean_tas_t"]].copy()
        tas_rows["delta_tas_t"] = tas_rows["mean_tas_t"].diff()
        pre_slopes = tas_rows.loc[(tas_rows["step"] <= k_star) & tas_rows["delta_tas_t"].notna(), "delta_tas_t"].to_numpy(dtype=float)
        post_slopes = tas_rows.loc[(tas_rows["step"] > k_star) & tas_rows["delta_tas_t"].notna(), "delta_tas_t"].to_numpy(dtype=float)
        pre_mean = float(np.mean(pre_slopes)) if len(pre_slopes) else math.nan
        post_mean = float(np.mean(post_slopes)) if len(post_slopes) else math.nan

        if math.isnan(post_mean):
            category = np.nan
        elif abs(post_mean) < 0.005:
            category = "frozen"
        elif post_mean > 0:
            category = "reversing"
        elif math.isnan(pre_mean):
            category = "decelerating"
        elif abs(post_mean) < abs(pre_mean):
            category = "decelerating"
        else:
            category = "accelerating"

        rows.append(
            {
                "question_id": question_id,
                "L": float(length),
                "k_star": k_star,
                "pre_kstar_tas_slope": pre_mean,
                "post_kstar_tas_slope": post_mean,
                "slope_change_ratio": _safe_ratio(post_mean, pre_mean),
                "post_slope_category": category,
                "difficulty_score": difficulty_score,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def analyze_2x2_cross(post_nldd: pd.DataFrame, post_tas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_row_columns = [
        "question_id",
        "L",
        "difficulty_score",
        "post_kstar_sign",
        "post_slope_category",
        "nldd_class",
        "tas_class",
    ]
    summary_columns = ["nldd_class", "tas_class", "count", "percentage"]
    if post_nldd.empty or post_tas.empty:
        _warn("Skipping 2x2 cross-tab because one upstream analysis is empty.")
        return pd.DataFrame(columns=per_row_columns), pd.DataFrame(columns=summary_columns)

    joined = post_nldd.merge(
        post_tas[["question_id", "L", "post_slope_category"]],
        on=["question_id", "L"],
        how="inner",
    )
    if joined.empty:
        _warn("2x2 cross-tab join produced zero matches.")
        return pd.DataFrame(columns=per_row_columns), pd.DataFrame(columns=summary_columns)

    joined["nldd_class"] = np.where(joined["post_kstar_sign"] == "negative", "negative", "neutral_or_positive")
    joined["tas_class"] = np.where(joined["post_slope_category"] == "frozen", "frozen", "still_moving")
    per_row = joined.loc[:, per_row_columns].copy()
    summary = (
        per_row.groupby(["nldd_class", "tas_class"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    total = int(summary["count"].sum())
    summary["percentage"] = (summary["count"] / total * 100.0) if total else np.nan
    return per_row, summary.loc[:, summary_columns]


def analyze_outlier_kstar_eq1(t1c: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["question_id", "L", "k_star", "difficulty_score", "n_clean"]
    if t1c is None:
        _warn("Skipping k*/L=1 analysis because t1c is unavailable.")
        return pd.DataFrame(columns=columns), {"skipped": True}
    outliers = t1c.loc[t1c["k_star_ratio"] > 0.99, columns].copy()
    total_bins = int(len(t1c))
    count = int(len(outliers))
    l_values = outliers["L"].dropna().to_numpy(dtype=float)
    diff_values = outliers["difficulty_score"].dropna().to_numpy(dtype=float)
    summary = {
        "skipped": False,
        "total_count": count,
        "total_bins": total_bins,
        "percentage_of_all_bins": (count / total_bins * 100.0) if total_bins else 0.0,
        "L_distribution": {
            "mean": float(np.mean(l_values)) if len(l_values) else math.nan,
            "median": float(np.median(l_values)) if len(l_values) else math.nan,
            "min": float(np.min(l_values)) if len(l_values) else math.nan,
            "max": float(np.max(l_values)) if len(l_values) else math.nan,
            "histogram_bins": {},
        },
        "share_L_le_4": float(np.mean(l_values <= 4) * 100.0) if len(l_values) else math.nan,
        "share_L_ge_5": float(np.mean(l_values >= 5) * 100.0) if len(l_values) else math.nan,
        "share_L_ge_6": float(np.mean(l_values >= 6) * 100.0) if len(l_values) else math.nan,
        "difficulty_score_distribution": {
            "mean": float(np.mean(diff_values)) if len(diff_values) else math.nan,
            "median": float(np.median(diff_values)) if len(diff_values) else math.nan,
            "min": float(np.min(diff_values)) if len(diff_values) else math.nan,
            "max": float(np.max(diff_values)) if len(diff_values) else math.nan,
        },
    }
    if len(l_values):
        unique, counts = np.unique(l_values.astype(int), return_counts=True)
        summary["L_distribution"]["histogram_bins"] = {str(int(key)): int(value) for key, value in zip(unique, counts)}
    return outliers, summary


def analyze_outlier_extreme_nldd(t1b: pd.DataFrame | None, t2b: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["question_id", "nldd_peak_value", "peak_L", "peak_step", "difficulty_score"]
    if t1b is None:
        _warn("Skipping extreme NLDD outlier analysis because t1b is unavailable.")
        return pd.DataFrame(columns=columns), {"skipped": True}

    usable = t1b.loc[t1b["mean_nldd"].notna(), ["question_id", "L", "step", "mean_nldd"]].copy()
    if usable.empty:
        _warn("No valid mean_nldd rows available for extreme NLDD outlier analysis.")
        return pd.DataFrame(columns=columns), {"skipped": False, "threshold": math.nan, "count": 0}

    idx = usable.groupby("question_id")["mean_nldd"].idxmax()
    peaks = usable.loc[idx].rename(columns={"L": "peak_L", "step": "peak_step", "mean_nldd": "nldd_peak_value"}).reset_index(drop=True)
    if t2b is not None:
        peaks = peaks.merge(t2b[["question_id", "difficulty_score"]], on="question_id", how="left")
    else:
        peaks["difficulty_score"] = np.nan

    peak_values = peaks["nldd_peak_value"].to_numpy(dtype=float)
    threshold = float(np.mean(peak_values) + 3.0 * np.std(peak_values, ddof=0)) if len(peak_values) else math.nan
    outliers = peaks.loc[peaks["nldd_peak_value"] > threshold, columns].copy() if not math.isnan(threshold) else peaks.iloc[0:0][columns].copy()
    dataset_median_difficulty = float(peaks["difficulty_score"].median()) if peaks["difficulty_score"].notna().any() else math.nan
    diff_values = outliers["difficulty_score"].dropna().to_numpy(dtype=float)
    summary = {
        "skipped": False,
        "threshold": threshold,
        "count": int(len(outliers)),
        "dataset_mean_difficulty": float(peaks["difficulty_score"].mean()) if peaks["difficulty_score"].notna().any() else math.nan,
        "dataset_median_difficulty": dataset_median_difficulty,
        "outlier_mean_difficulty": float(np.mean(diff_values)) if len(diff_values) else math.nan,
        "outlier_median_difficulty": float(np.median(diff_values)) if len(diff_values) else math.nan,
        "share_outliers_below_dataset_median_difficulty": (
            float(np.mean(diff_values < dataset_median_difficulty) * 100.0)
            if len(diff_values) and not math.isnan(dataset_median_difficulty)
            else math.nan
        ),
        "peak_step_distribution": {},
        "share_peak_step_eq_2": float(np.mean(outliers["peak_step"] == 2) * 100.0) if len(outliers) else math.nan,
    }
    if len(outliers):
        counts = outliers["peak_step"].value_counts().sort_index()
        summary["peak_step_distribution"] = {str(int(idx)): int(value) for idx, value in counts.items()}
    return outliers, summary


def analyze_outlier_low_kstar_ratio(t1c: pd.DataFrame | None, t2b: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["question_id", "L", "k_star", "k_star_ratio", "difficulty_score", "n_clean", "l_star_A", "is_longer_than_lstar"]
    if t1c is None:
        _warn("Skipping low k*/L outlier analysis because t1c is unavailable.")
        return pd.DataFrame(columns=columns), {"skipped": True}
    outliers = t1c.loc[t1c["k_star_ratio"] < 0.4, ["question_id", "L", "k_star", "k_star_ratio", "difficulty_score", "n_clean"]].copy()
    if t2b is not None:
        outliers = outliers.merge(t2b[["question_id", "l_star_A"]], on="question_id", how="left")
        outliers["is_longer_than_lstar"] = outliers["L"] > outliers["l_star_A"]
    else:
        outliers["l_star_A"] = np.nan
        outliers["is_longer_than_lstar"] = np.nan
    l_values = outliers["L"].dropna().to_numpy(dtype=float)
    diff_values = outliers["difficulty_score"].dropna().to_numpy(dtype=float)
    longer = outliers["is_longer_than_lstar"].dropna()
    summary = {
        "skipped": False,
        "count": int(len(outliers)),
        "L_distribution": {
            "mean": float(np.mean(l_values)) if len(l_values) else math.nan,
            "median": float(np.median(l_values)) if len(l_values) else math.nan,
            "min": float(np.min(l_values)) if len(l_values) else math.nan,
            "max": float(np.max(l_values)) if len(l_values) else math.nan,
        },
        "difficulty_score_distribution": {
            "mean": float(np.mean(diff_values)) if len(diff_values) else math.nan,
            "median": float(np.median(diff_values)) if len(diff_values) else math.nan,
            "min": float(np.min(diff_values)) if len(diff_values) else math.nan,
            "max": float(np.max(diff_values)) if len(diff_values) else math.nan,
        },
        "share_with_L_gt_lstar_A": float(np.mean(longer.astype(bool)) * 100.0) if len(longer) else math.nan,
        "count_with_L_gt_lstar_A": int(np.sum(longer.astype(bool))) if len(longer) else 0,
    }
    return outliers.loc[:, columns], summary


def plot_fig_e(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or frame["post_kstar_sign"].dropna().empty:
        _warn("Skipping Figure E because source data is empty.")
        return False
    working = frame.dropna(subset=["post_kstar_sign", "difficulty_score"]).copy()
    if working.empty:
        _warn("Skipping Figure E because no rows have both sign and difficulty.")
        return False
    working["difficulty_group"] = _difficulty_groups(working["difficulty_score"])
    order = ["positive", "near_zero", "negative"]
    counts = (
        working.groupby(["post_kstar_sign", "difficulty_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#9ecae1", "#fdae6b", "#74c476"])
    ax.set_xlabel("post_kstar_sign")
    ax.set_ylabel("count")
    ax.set_title("Figure E. Post-horizon NLDD sign distribution")
    ax.legend(title="difficulty")
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(path)
    return True


def plot_fig_f(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or frame["post_slope_category"].dropna().empty:
        _warn("Skipping Figure F because source data is empty.")
        return False
    working = frame.dropna(subset=["post_slope_category", "difficulty_score"]).copy()
    if working.empty:
        _warn("Skipping Figure F because no rows have both slope category and difficulty.")
        return False
    working["difficulty_group"] = _difficulty_groups(working["difficulty_score"])
    order = ["frozen", "decelerating", "accelerating", "reversing"]
    counts = (
        working.groupby(["post_slope_category", "difficulty_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#9ecae1", "#fdae6b", "#74c476"])
    ax.set_xlabel("post_slope_category")
    ax.set_ylabel("count")
    ax.set_title("Figure F. Post-horizon TAS slope categories")
    ax.legend(title="difficulty")
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(path)
    return True


def plot_fig_g(summary: pd.DataFrame, path: Path) -> bool:
    _apply_paper_style()
    if summary.empty:
        _warn("Skipping Figure G because 2x2 summary is empty.")
        return False
    row_order = ["neutral_or_positive", "negative"]
    col_order = ["frozen", "still_moving"]
    pivot_counts = summary.pivot(index="nldd_class", columns="tas_class", values="count").reindex(index=row_order, columns=col_order, fill_value=0)
    pivot_pct = summary.pivot(index="nldd_class", columns="tas_class", values="percentage").reindex(index=row_order, columns=col_order, fill_value=0.0)
    pct_text = pivot_pct.map(lambda x: f"{x:.1f}%")
    annotations = pivot_counts.astype(int).astype(str) + "\n" + pct_text
    labeled_counts = pivot_counts.copy()
    labeled_counts.index = [r"$\bar{D}_{\mathrm{post}}\geq 0$", r"$\bar{D}_{\mathrm{post}}<0$"]
    labeled_counts.columns = [r"$|\Delta T_{\mathrm{post}}|<\epsilon$", r"$|\Delta T_{\mathrm{post}}|\geq\epsilon$"]
    annotations.index = labeled_counts.index
    annotations.columns = labeled_counts.columns
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sns.heatmap(
        labeled_counts,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        annot_kws={"fontsize": 15, "fontweight": "semibold"},
        ax=ax,
    )
    ax.set_xlabel(r"$\Delta T_{\mathrm{post}}$")
    ax.set_ylabel(r"$\bar{D}_{\mathrm{post}}$")
    ax.tick_params(axis="x", labelrotation=0, labelsize=14)
    ax.tick_params(axis="y", labelrotation=0, labelsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(path)
    return True


def plot_fig_h(frame: pd.DataFrame, summary: dict[str, Any], path: Path) -> bool:
    if frame.empty:
        _warn("Skipping Figure H because k*/L=1 outlier data is empty.")
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(frame["L"].min() - 0.5, frame["L"].max() + 1.5, 1.0)
    ax.hist(frame["L"], bins=bins, color="#6baed6", edgecolor="white")
    ax.axvline(4, color="firebrick", linestyle="--", linewidth=1.3)
    share = summary.get("share_L_le_4", math.nan)
    ax.text(0.98, 0.95, f"% with L <= 4: {share:.1f}%", transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("L")
    ax.set_ylabel("count")
    ax.set_title("Figure H. L distribution for k*/L = 1.0 outliers")
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(path)
    return True


def plot_fig_i(frame: pd.DataFrame, path: Path) -> bool:
    _apply_paper_style()
    working = frame.dropna(subset=["L", "nldd_drop_ratio", "difficulty_score"]).copy()
    if working.empty:
        _warn("Skipping Figure I because NLDD drop ratio data is empty.")
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("viridis")
    norm = _difficulty_norm(working["difficulty_score"])
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, size=len(working))
    scatter = ax.scatter(
        working["L"].to_numpy(dtype=float) + jitter,
        working["nldd_drop_ratio"],
        c=working["difficulty_score"],
        cmap=cmap,
        norm=norm,
        alpha=0.30,
        s=16,
        linewidths=0,
        rasterized=True,
    )
    by_l = (
        working.groupby("L", as_index=False)["nldd_drop_ratio"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), count="count")
        .sort_values("L")
    )
    x_l = by_l["L"].to_numpy(dtype=float)
    median = by_l["median"].to_numpy(dtype=float)
    q25 = by_l["q25"].to_numpy(dtype=float)
    q75 = by_l["q75"].to_numpy(dtype=float)
    ax.fill_between(
        x_l,
        q25,
        q75,
        color=PAPER_PALETTE["blue"],
        alpha=0.16,
        linewidth=0,
        label=r"$Q_{25-75}(r_L)$",
    )
    ax.plot(x_l, median, color=PAPER_PALETTE["blue"], linewidth=2.0, marker="o", markersize=3.8, label=r"$\mathrm{median}(r_L \mid L)$")
    ax.axhline(1.0, color=PAPER_PALETTE["gray"], linestyle=":", linewidth=1.1, label=r"$r_L=1$")
    ax.axhline(0.0, color=PAPER_PALETTE["red"], linestyle="--", linewidth=1.2, label=r"$r_L=0$")
    ax.set_xlabel(r"Trace length $L$")
    ax.set_ylabel(r"Post/pre NLDD ratio")
    y_min = float(np.nanquantile(working["nldd_drop_ratio"], 0.01))
    y_max = float(np.nanquantile(working["nldd_drop_ratio"], 0.99))
    pad = max(0.08, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    _polish_axes(ax)
    point_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#6B7280",
        markeredgewidth=0,
        alpha=0.45,
        markersize=5,
        label=r"$(q,L)$ bins",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([point_handle] + handles, [r"$(q,L)$ bins"] + labels, loc="lower right", borderpad=0.6)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015, fraction=0.04)
    cbar.set_label(r"$d(q)$")
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(path)
    return True


def plot_fig_j(frame: pd.DataFrame, path: Path) -> bool:
    _apply_paper_style()
    working = frame.dropna(subset=["pre_kstar_tas_slope", "post_kstar_tas_slope", "difficulty_score"]).copy()
    if working.empty:
        _warn("Skipping Figure J because TAS slope source data is empty.")
        return False
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    cmap = plt.get_cmap("viridis")
    norm = _difficulty_norm(working["difficulty_score"])
    scatter = ax.scatter(
        working["pre_kstar_tas_slope"],
        working["post_kstar_tas_slope"],
        c=working["difficulty_score"],
        cmap=cmap,
        norm=norm,
        alpha=0.38,
        s=18,
        linewidths=0,
        rasterized=True,
    )
    combined = np.concatenate(
        [
            working["pre_kstar_tas_slope"].to_numpy(dtype=float),
            working["post_kstar_tas_slope"].to_numpy(dtype=float),
        ]
    )
    lower = float(np.nanmin(combined))
    upper = float(np.nanmax(combined))
    pad = max(0.02, 0.08 * (upper - lower))
    lower -= pad
    upper = max(upper + pad, 0.02)
    ax.fill_between(
        [lower, upper],
        [lower, upper],
        [upper, upper],
        color=PAPER_PALETTE["green"],
        alpha=0.10,
        linewidth=0,
        label=r"$\Delta T_{\mathrm{post}}>\Delta T_{\mathrm{pre}}$",
    )
    ax.plot([lower, upper], [lower, upper], color=PAPER_PALETTE["dark"], linestyle="--", linewidth=1.2, label=r"$y=x$")
    ax.axhline(0.0, color=PAPER_PALETTE["red"], linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color=PAPER_PALETTE["red"], linestyle=":", linewidth=1.0, label=r"$0$")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Pre-$k^\star$ TAS slope")
    ax.set_ylabel(r"Post-$k^\star$ TAS slope")
    _polish_axes(ax)
    point_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#6B7280",
        markeredgewidth=0,
        alpha=0.45,
        markersize=5,
        label=r"$(q,L)$ bins",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([point_handle] + handles, [r"$(q,L)$ bins"] + labels, loc="lower right", borderpad=0.6)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(r"$d(q)$")
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", default=".", help="Project root used to scan for PQ CSVs.")
    parser.add_argument("--source-dir", default="exploratory/deep_dive", help="Directory for analysis CSV/JSON outputs.")
    parser.add_argument("--figure-dir", default="exploratory/deep_dive", help="Directory for PNG outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_root = Path(args.search_root)
    source_dir = Path(args.source_dir)
    figure_dir = Path(args.figure_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_csvs(search_root)
    for name, path in discovered.items():
        print(f"resolved {name}: {path}")

    t1b = _normalize_t1b(_load_csv(discovered["t1b_step_surface.csv"], "t1b_step_surface.csv"))
    t1c = _normalize_t1c(_load_csv(discovered["t1c_kstar_ratio.csv"], "t1c_kstar_ratio.csv"))
    t2b = _normalize_t2b(_load_csv(discovered["t2b_lstar_difficulty.csv"], "t2b_lstar_difficulty.csv"))

    analysis_status: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    post_nldd = analyze_post_horizon_nldd(t1b, t1c)
    _write_csv(
        post_nldd,
        source_dir / "post_horizon_nldd.csv",
        ["question_id", "L", "k_star", "pre_kstar_mean_nldd", "post_kstar_mean_nldd", "post_kstar_sign", "nldd_drop_ratio", "difficulty_score", "n_post_steps"],
    )
    analysis_status["post_horizon_nldd"] = "ok" if not post_nldd.empty else "empty_or_skipped"
    row_counts["post_horizon_nldd.csv"] = int(len(post_nldd))

    post_tas = analyze_post_horizon_tas_slope(t1b, t1c)
    _write_csv(
        post_tas,
        source_dir / "post_horizon_tas_slope.csv",
        ["question_id", "L", "k_star", "pre_kstar_tas_slope", "post_kstar_tas_slope", "slope_change_ratio", "post_slope_category", "difficulty_score"],
    )
    analysis_status["post_horizon_tas_slope"] = "ok" if not post_tas.empty else "empty_or_skipped"
    row_counts["post_horizon_tas_slope.csv"] = int(len(post_tas))

    cross_rows, cross_summary = analyze_2x2_cross(post_nldd, post_tas)
    _write_csv(
        cross_rows,
        source_dir / "post_horizon_2x2.csv",
        ["question_id", "L", "difficulty_score", "post_kstar_sign", "post_slope_category", "nldd_class", "tas_class"],
    )
    _write_csv(
        cross_summary,
        source_dir / "post_horizon_2x2_summary.csv",
        ["nldd_class", "tas_class", "count", "percentage"],
    )
    analysis_status["post_horizon_2x2"] = "ok" if not cross_rows.empty else "empty_or_skipped"
    row_counts["post_horizon_2x2.csv"] = int(len(cross_rows))
    row_counts["post_horizon_2x2_summary.csv"] = int(len(cross_summary))

    outlier_eq1, outlier_eq1_summary = analyze_outlier_kstar_eq1(t1c)
    _write_csv(outlier_eq1, source_dir / "outlier_kstar_eq_1.csv", ["question_id", "L", "k_star", "difficulty_score", "n_clean"])
    _write_json(outlier_eq1_summary, source_dir / "outlier_kstar_eq_1_summary.json")
    analysis_status["outlier_kstar_eq_1"] = "ok" if not outlier_eq1.empty else "empty_or_skipped"
    row_counts["outlier_kstar_eq_1.csv"] = int(len(outlier_eq1))

    extreme_nldd, extreme_nldd_summary = analyze_outlier_extreme_nldd(t1b, t2b)
    _write_csv(extreme_nldd, source_dir / "outlier_extreme_nldd.csv", ["question_id", "nldd_peak_value", "peak_L", "peak_step", "difficulty_score"])
    _write_json(extreme_nldd_summary, source_dir / "outlier_extreme_nldd_summary.json")
    analysis_status["outlier_extreme_nldd"] = "ok" if not extreme_nldd.empty else "empty_or_skipped"
    row_counts["outlier_extreme_nldd.csv"] = int(len(extreme_nldd))

    low_kstar, low_kstar_summary = analyze_outlier_low_kstar_ratio(t1c, t2b)
    _write_csv(
        low_kstar,
        source_dir / "outlier_low_kstar_ratio.csv",
        ["question_id", "L", "k_star", "k_star_ratio", "difficulty_score", "n_clean", "l_star_A", "is_longer_than_lstar"],
    )
    _write_json(low_kstar_summary, source_dir / "outlier_low_kstar_ratio_summary.json")
    analysis_status["outlier_low_kstar_ratio"] = "ok" if not low_kstar.empty else "empty_or_skipped"
    row_counts["outlier_low_kstar_ratio.csv"] = int(len(low_kstar))

    figure_status = {
        "fig_e_post_horizon_nldd_sign.png": plot_fig_e(post_nldd, figure_dir / "fig_e_post_horizon_nldd_sign.png"),
        "fig_f_post_horizon_tas_slope.png": plot_fig_f(post_tas, figure_dir / "fig_f_post_horizon_tas_slope.png"),
        "fig_g_post_horizon_2x2.png": plot_fig_g(cross_summary, figure_dir / "fig_g_post_horizon_2x2.png"),
        "fig_h_kstar_eq1_L_dist.png": plot_fig_h(outlier_eq1, outlier_eq1_summary, figure_dir / "fig_h_kstar_eq1_L_dist.png"),
        "fig_i_nldd_drop_ratio_vs_L.png": plot_fig_i(post_nldd, figure_dir / "fig_i_nldd_drop_ratio_vs_L.png"),
        "fig_j_tas_slope_pre_vs_post.png": plot_fig_j(post_tas, figure_dir / "fig_j_tas_slope_pre_vs_post.png"),
    }

    summary_payload = {
        "search_root": str(search_root),
        "source_dir": str(source_dir),
        "figure_dir": str(figure_dir),
        "resolved_inputs": {name: (str(path) if path else None) for name, path in discovered.items()},
        "analysis_status": analysis_status,
        "row_counts": row_counts,
        "figure_status": figure_status,
    }
    _write_json(summary_payload, source_dir / "deep_dive_summary.json")

    print("analysis summary:")
    for name, status in analysis_status.items():
        print(f"  {name}: {status}")
    print("row counts:")
    for name, count in row_counts.items():
        print(f"  {name}: {count}")
    print("figure status:")
    for name, ok in figure_status.items():
        print(f"  {name}: {'ok' if ok else 'skipped'}")


if __name__ == "__main__":
    main()
