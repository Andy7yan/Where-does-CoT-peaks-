"""Render targeted follow-up plots from PQ Analysis Phase 1 artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats


PNG_DPI = 150
FILE_PRIORITY = (
    "t1c_kstar_ratio.csv",
    "t2b_lstar_difficulty.csv",
    "t1b_step_surface.csv",
)
SEARCH_DIR_PRIORITY = ("pq_analysis", "analysis", "analysis_phase1")


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _discover_csvs(search_root: Path) -> dict[str, Path | None]:
    matches: dict[str, list[Path]] = {name: [] for name in FILE_PRIORITY}
    for name in FILE_PRIORITY:
        for candidate in search_root.rglob(name):
            if candidate.is_file() and candidate.stat().st_size > 0:
                matches[name].append(candidate)

    def sort_key(path: Path) -> tuple[int, str]:
        text = str(path).replace("\\", "/")
        dir_rank = min((idx for idx, token in enumerate(SEARCH_DIR_PRIORITY) if f"/{token}/" in f"/{text}/"), default=99)
        papery_rank = 0 if "papery-pq" in text else 1
        path_rank = 0 if "path-pq" in text else 1
        return (dir_rank, papery_rank, path_rank, text)

    resolved: dict[str, Path | None] = {}
    for name, paths in matches.items():
        resolved[name] = sorted(paths, key=sort_key)[0] if paths else None
    return resolved


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
    required = {"question_id", "L", "step", "mean_nldd", "mean_tas_t", "n_clean", "bin_status"}
    missing = required - set(result.columns)
    if missing:
        _warn(f"t1b_step_surface.csv is missing columns: {sorted(missing)}")
        return None
    result["question_id"] = result["question_id"].astype(str)
    for col in ("L", "step", "mean_nldd", "mean_tas_t", "n_clean"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["bin_status"] = result["bin_status"].astype(str)
    return result


def _save_source(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _print_saved(path: Path) -> None:
    print(f"saved: {path}")


def _box_strip_plot(frame: pd.DataFrame, output_path: Path) -> None:
    ordered = frame.sort_values(["difficulty_score", "question_id", "L"]).copy()
    ordered["question_id"] = pd.Categorical(ordered["question_id"], categories=ordered["question_id"].drop_duplicates(), ordered=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(ordered["difficulty_score"].min()), vmax=float(ordered["difficulty_score"].max()))
    point_colors = cmap(norm(ordered["difficulty_score"].to_numpy(dtype=float)))
    x_positions = ordered["question_id"].cat.codes.to_numpy(dtype=float)
    jitter = np.random.default_rng(42).uniform(-0.22, 0.22, size=len(ordered))

    fig, ax = plt.subplots(figsize=(max(18, len(ordered["question_id"].cat.categories) * 0.12), 7))
    sns.boxplot(
        data=ordered,
        x="question_id",
        y="k_star_ratio",
        color="#d9d9d9",
        width=0.6,
        fliersize=0,
        linewidth=0.8,
        ax=ax,
    )
    ax.scatter(
        x_positions + jitter,
        ordered["k_star_ratio"],
        c=point_colors,
        s=14,
        alpha=0.8,
        linewidths=0,
    )
    global_median = float(ordered["k_star_ratio"].median())
    ax.axhline(global_median, color="firebrick", linestyle="--", linewidth=1.2, label=f"Global median = {global_median:.3f}")
    ax.set_xlabel("Questions ordered by difficulty")
    ax.set_ylabel("k*/L")
    ax.set_title("Figure A. k*/L stability by question")
    ax.set_xticks([])
    ax.legend(loc="upper right")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("difficulty_score")

    fig.tight_layout()
    fig.savefig(output_path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(output_path)


def _tas_vs_l(frame: pd.DataFrame, output_path: Path) -> None:
    working = frame.copy()
    working["question_id"] = working["question_id"].astype(str)
    final_rows = working[(working["bin_status"] == "ok") & (working["step"] == working["L"]) & working["mean_tas_t"].notna()].copy()
    if final_rows.empty:
        _warn("No valid rows available for Figure B.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(final_rows["difficulty_score"].min()), vmax=float(final_rows["difficulty_score"].max()))
    sc = ax.scatter(
        final_rows["L"],
        final_rows["mean_tas_t"],
        c=final_rows["difficulty_score"],
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=24,
        linewidths=0,
    )
    if final_rows["L"].nunique() >= 3:
        x = final_rows["L"].to_numpy(dtype=float)
        y = final_rows["mean_tas_t"].to_numpy(dtype=float)
        order = np.argsort(x)
        coefficients = np.polyfit(x, y, deg=2)
        x_grid = np.linspace(float(x.min()), float(x.max()), 200)
        y_grid = np.polyval(coefficients, x_grid)
        ax.plot(x_grid, y_grid, color="black", linewidth=1.5, label="Quadratic trend")
        ax.legend(loc="best")
    ax.set_xlabel("L")
    ax.set_ylabel("tas_final")
    ax.set_title("Figure B. TAS decay vs L")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("difficulty_score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(output_path)


def _compute_nldd_features(step_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = step_frame[(step_frame["bin_status"] == "ok") & step_frame["mean_nldd"].notna()].groupby(["question_id", "L"], sort=False)
    for (question_id, length), group in grouped:
        values = group.loc[group["step"] >= 2, "mean_nldd"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        total = float(np.sum(values))
        early_values = group.loc[(group["step"] >= 2) & (group["step"] <= (float(length) / 2.0)), "mean_nldd"].to_numpy(dtype=float)
        early_mass = (float(np.sum(early_values)) / total) if not math.isclose(total, 0.0) else math.nan
        skewness = float(stats.skew(values, bias=False, nan_policy="omit")) if len(values) >= 3 else math.nan
        rows.append(
            {
                "question_id": str(question_id),
                "L": float(length),
                "nldd_peak_value": float(np.max(values)),
                "nldd_early_mass": early_mass,
                "nldd_skewness": skewness,
            }
        )
    return pd.DataFrame(rows)


def _pairgrid_scatter(x: pd.Series, y: pd.Series, color_values: np.ndarray, cmap: Any, norm: Normalize, **kwargs: Any) -> None:
    ax = plt.gca()
    valid = ~(pd.isna(x) | pd.isna(y) | np.isnan(color_values))
    ax.scatter(
        np.asarray(x)[valid],
        np.asarray(y)[valid],
        c=color_values[valid],
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.65,
        linewidths=0,
    )


def _pairplot(frame: pd.DataFrame, output_path: Path) -> None:
    feature_cols = ["nldd_peak_value", "nldd_early_mass", "nldd_skewness"]
    clean = frame.dropna(subset=feature_cols + ["difficulty_score"]).copy()
    if clean.empty:
        _warn("No valid rows available for Figure C.")
        return
    norm = Normalize(vmin=float(clean["difficulty_score"].min()), vmax=float(clean["difficulty_score"].max()))
    cmap = plt.get_cmap("viridis")
    color_values = clean["difficulty_score"].to_numpy(dtype=float)
    grid = sns.PairGrid(clean[feature_cols + ["difficulty_score"]], vars=feature_cols, height=2.7, diag_sharey=False)
    grid.map_diag(sns.kdeplot, fill=True, color="#4c72b0")
    grid.map_offdiag(_pairgrid_scatter, color_values=color_values, cmap=cmap, norm=norm)
    grid.figure.suptitle("Figure C. NLDD shape feature pairplot", y=1.02)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    grid.figure.subplots_adjust(top=0.95, right=0.88)
    colorbar_ax = grid.figure.add_axes([0.91, 0.18, 0.02, 0.64])
    grid.figure.colorbar(sm, cax=colorbar_ax, label="difficulty_score")
    grid.figure.savefig(output_path, dpi=PNG_DPI)
    plt.close(grid.figure)
    _print_saved(output_path)


def _violin_plot(frame: pd.DataFrame, output_path: Path) -> None:
    working = frame.copy()
    working["delta"] = working["L"] - working["l_star_A"]
    working["group"] = np.select(
        [working["delta"] < 0, working["delta"] == 0, working["delta"] > 0],
        ["L < L*", "L ≈ L*", "L > L*"],
        default="Unknown",
    )
    working = working[working["group"] != "Unknown"].copy()
    if working.empty:
        _warn("No valid rows available for Figure D.")
        return
    order = ["L < L*", "L ≈ L*", "L > L*"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=working,
        x="group",
        y="k_star_ratio",
        hue="group",
        order=order,
        hue_order=order,
        inner=None,
        cut=0,
        palette="Set2",
        legend=False,
        ax=ax,
    )
    sns.stripplot(data=working, x="group", y="k_star_ratio", order=order, jitter=0.18, size=3, alpha=0.45, color="black", ax=ax)
    stats_by_group = working.groupby("group")["k_star_ratio"].agg(["median", "count"]).reindex(order)
    y_max = float(working["k_star_ratio"].max())
    for idx, group in enumerate(order):
        if pd.isna(stats_by_group.loc[group, "median"]):
            continue
        median = float(stats_by_group.loc[group, "median"])
        count = int(stats_by_group.loc[group, "count"])
        ax.text(idx, y_max + 0.03, f"median={median:.3f}\nn={count}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Distance from L*")
    ax.set_ylabel("k*/L")
    ax.set_title("Figure D. k*/L by distance to L*")
    ax.set_ylim(top=min(1.12, y_max + 0.14))
    fig.tight_layout()
    fig.savefig(output_path, dpi=PNG_DPI)
    plt.close(fig)
    _print_saved(output_path)


def build_targeted_plots(*, search_root: Path, source_dir: Path, figure_dir: Path) -> dict[str, Any]:
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_csvs(search_root)
    for name, path in discovered.items():
        print(f"resolved {name}: {path}")

    t1c = _normalize_t1c(_load_csv(discovered["t1c_kstar_ratio.csv"], "t1c_kstar_ratio.csv"))
    t2b = _normalize_t2b(_load_csv(discovered["t2b_lstar_difficulty.csv"], "t2b_lstar_difficulty.csv"))
    t1b = _normalize_t1b(_load_csv(discovered["t1b_step_surface.csv"], "t1b_step_surface.csv"))

    summary: dict[str, Any] = {
        "search_root": str(search_root),
        "source_dir": str(source_dir),
        "figure_dir": str(figure_dir),
        "resolved": {name: (str(path) if path else None) for name, path in discovered.items()},
        "generated": [],
    }

    joined_kstar: pd.DataFrame | None = None
    if t1c is not None and t2b is not None:
        joined_kstar = t1c.merge(
            t2b[["question_id", "difficulty_score", "l_star_A", "l_star_C", "l_star_consistent"]],
            on=["question_id", "difficulty_score"],
            how="left",
        )
        _save_source(joined_kstar, source_dir / "fig_a_d_joined_kstar_lstar.csv")
        _box_strip_plot(joined_kstar, figure_dir / "fig_a_kstar_ratio_stability.png")
        summary["generated"].append("fig_a_kstar_ratio_stability.png")

        _violin_plot(joined_kstar, figure_dir / "fig_d_kstar_ratio_by_lstar_distance.png")
        _save_source(joined_kstar.assign(delta=joined_kstar["L"] - joined_kstar["l_star_A"]), source_dir / "fig_d_grouped_source.csv")
        summary["generated"].append("fig_d_kstar_ratio_by_lstar_distance.png")
    else:
        _warn("Skipping Figures A and D because joined k*/L data is unavailable.")

    if t1b is not None and t2b is not None:
        t1b_with_difficulty = t1b.merge(
            t2b[["question_id", "difficulty_score", "l_star_A", "l_star_C", "l_star_consistent"]],
            on="question_id",
            how="left",
        )
        final_tas = t1b_with_difficulty[
            (t1b_with_difficulty["bin_status"] == "ok")
            & (t1b_with_difficulty["step"] == t1b_with_difficulty["L"])
            & t1b_with_difficulty["mean_tas_t"].notna()
        ].copy()
        _save_source(final_tas, source_dir / "fig_b_tas_vs_L_source.csv")
        _tas_vs_l(t1b_with_difficulty, figure_dir / "fig_b_tas_vs_L.png")
        summary["generated"].append("fig_b_tas_vs_L.png")

        nldd_features = _compute_nldd_features(t1b_with_difficulty).merge(
            t2b[["question_id", "difficulty_score"]],
            on="question_id",
            how="left",
        )
        _save_source(nldd_features, source_dir / "fig_c_nldd_shape_features.csv")
        _pairplot(nldd_features, figure_dir / "fig_c_nldd_shape_pairplot.png")
        summary["generated"].append("fig_c_nldd_shape_pairplot.png")
    else:
        _warn("Skipping Figures B and C because t1b/t2b data is unavailable.")

    summary_path = source_dir / "targeted_plots_summary.txt"
    lines = [
        f"search_root: {summary['search_root']}",
        f"source_dir: {summary['source_dir']}",
        f"figure_dir: {summary['figure_dir']}",
    ]
    lines.extend(f"{name}: {path}" for name, path in summary["resolved"].items())
    lines.extend(f"generated: {name}" for name in summary["generated"])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", default=".", help="Project root used to scan for pq_analysis/analysis_phase1/analysis CSVs.")
    parser.add_argument("--source-dir", default="exploratory/targeted", help="Directory for source CSV outputs.")
    parser.add_argument("--figure-dir", default="exploratory/targeted", help="Directory for PNG outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_targeted_plots(
        search_root=Path(args.search_root),
        source_dir=Path(args.source_dir),
        figure_dir=Path(args.figure_dir),
    )


if __name__ == "__main__":
    main()
