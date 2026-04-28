"""Plot only the data needed to explain the high k*/L rebound."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "eq1": "#d55e00",
    "high": "#cc79a7",
    "mid": "#0072b2",
    "low": "#009e73",
    "neutral": "#6b7280",
}


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved: {path}")


def plot_ratio_by_l(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(frame["L"], frame["eq1_rate"], color=COLORS["eq1"], alpha=0.75, label="k*/L = 1 rate")
    ax1.plot(
        frame["L"],
        frame["high_ge_0.85_rate"],
        color=COLORS["high"],
        linewidth=2.0,
        marker="o",
        label="k*/L >= 0.85 rate",
    )
    ax1.set_xlabel("Trace length L")
    ax1.set_ylabel("Share of bins")
    ax1.set_ylim(0, max(0.55, float(frame["high_ge_0.85_rate"].max()) * 1.2))
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(
        frame["L"],
        frame["median_k_star_ratio"],
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label="median k*/L",
    )
    ax2.set_ylabel("Median k*/L")
    ax2.set_ylim(0.45, 1.02)
    ax2.legend(loc="center right")

    ax1.set_title("High k*/L is concentrated at short L, with a small long-L tail")
    save(fig, output_dir / "rebound_ratio_by_L.png")


def plot_group_contrast(frame: pd.DataFrame, output_dir: Path) -> None:
    order = ["eq1", "high_lt1_0.85_1", "mid_0.4_0.85", "low_lt0.4"]
    labels = ["=1", "0.85-1", "0.4-0.85", "<0.4"]
    working = frame.set_index("ratio_group").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    x = range(len(working))
    axes[0].bar(x, working["median_L"], color=[COLORS["eq1"], COLORS["high"], COLORS["mid"], COLORS["low"]])
    axes[0].set_xticks(list(x), labels, rotation=0)
    axes[0].set_ylabel("Median L")
    axes[0].set_title("Length")

    axes[1].bar(x, working["share_L_le_4"], color=COLORS["eq1"], alpha=0.8, label="L <= 4")
    axes[1].bar(
        x,
        working["share_L_ge_6"],
        bottom=working["share_L_le_4"],
        color=COLORS["high"],
        alpha=0.75,
        label="L >= 6",
    )
    axes[1].set_xticks(list(x), labels, rotation=0)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Share")
    axes[1].set_title("Short vs long bins")
    axes[1].legend(loc="upper center", fontsize=8)

    axes[2].bar(x, working["median_difficulty_score"], color=COLORS["neutral"], alpha=0.8)
    axes[2].set_xticks(list(x), labels, rotation=0)
    axes[2].set_ylabel("Median difficulty score")
    axes[2].set_title("Difficulty")

    fig.suptitle("=1 bins look short/easy; high-but-not-1 bins are the long tail")
    save(fig, output_dir / "rebound_ratio_group_contrast.png")


def plot_feature_correlations(frame: pd.DataFrame, output_dir: Path) -> None:
    display = frame.copy()
    display["label"] = display["feature"].replace(
        {
            "L_minus_l_star_A": "L - L*",
            "boundary_lag": "L - k*",
            "n_post_steps": "post-k* steps",
            "difficulty_score": "difficulty",
            "l_star_A": "L*",
        }
    )
    display = display.sort_values("spearman_rho")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [COLORS["eq1"] if abs(value) >= 0.5 else COLORS["neutral"] for value in display["spearman_rho"]]
    ax.barh(display["label"], display["spearman_rho"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="#111827", linewidth=0.9)
    ax.set_xlabel("Spearman rho with k*/L")
    ax.set_title("Boundary lag dominates; difficulty is not the main driver")
    save(fig, output_dir / "rebound_feature_correlations.png")


def plot_l_histograms(group_frame: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, row in group_frame.iterrows():
        hist = ast.literal_eval(row["L_histogram_json"])
        for length, count in hist.items():
            rows.append({"ratio_group": row["ratio_group"], "L": int(length), "count": int(count)})
    hist_frame = pd.DataFrame(rows)
    groups = [
        ("eq1", "=1", COLORS["eq1"]),
        ("high_lt1_0.85_1", "0.85-1", COLORS["high"]),
        ("mid_0.4_0.85", "0.4-0.85", COLORS["mid"]),
        ("low_lt0.4", "<0.4", COLORS["low"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, (group, label, color) in zip(axes.ravel(), groups):
        subset = hist_frame[hist_frame["ratio_group"] == group]
        ax.bar(subset["L"], subset["count"], color=color, alpha=0.82)
        ax.set_title(label)
        ax.set_ylabel("bin count")
    for ax in axes[-1]:
        ax.set_xlabel("Trace length L")
    fig.suptitle("Length histograms by k*/L group")
    save(fig, output_dir / "rebound_length_histograms.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="results/outputs-0426/sources/rebound")
    parser.add_argument("--figure-dir", default="results/outputs-0426/local-figures/rebound")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    figure_dir = Path(args.figure_dir)
    ratio_by_l = pd.read_csv(source_dir / "ratio_by_L.csv")
    group_contrast = pd.read_csv(source_dir / "ratio_group_contrast.csv")
    correlations = pd.read_csv(source_dir / "kstar_ratio_feature_correlations.csv")

    plot_ratio_by_l(ratio_by_l, figure_dir)
    plot_group_contrast(group_contrast, figure_dir)
    plot_feature_correlations(correlations, figure_dir)
    plot_l_histograms(group_contrast, figure_dir)


if __name__ == "__main__":
    main()
