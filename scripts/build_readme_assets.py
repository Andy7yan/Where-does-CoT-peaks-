from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets" / "readme"


def _style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def build_t1a_overview() -> None:
    files = {
        "Easy": ROOT / "results" / "outputs-0422" / "sources" / "t1a" / "t1a_easy.csv",
        "Medium": ROOT / "results" / "outputs-0422" / "sources" / "t1a" / "t1a_medium.csv",
        "Hard": ROOT / "results" / "outputs-0422" / "sources" / "t1a" / "t1a_hard.csv",
    }
    colors = {"accuracy": "#1f77b4", "tas": "#2ca02c", "k_star": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for ax, (label, path) in zip(axes, files.items()):
        frame = pd.read_csv(path)
        x = frame["L"].to_numpy()

        ax.plot(x, frame["accuracy"], color=colors["accuracy"], marker="o", label="Accuracy")
        ax.fill_between(
            x,
            frame["accuracy"] - frame["accuracy_se"],
            frame["accuracy"] + frame["accuracy_se"],
            color=colors["accuracy"],
            alpha=0.12,
        )

        ax.plot(
            x,
            frame["mean_tas"],
            color=colors["tas"],
            marker="^",
            linestyle="--",
            label="TAS",
        )
        ax.fill_between(
            x,
            frame["mean_tas"] - frame["tas_se"],
            frame["mean_tas"] + frame["tas_se"],
            color=colors["tas"],
            alpha=0.12,
        )

        ax.set_title(label, weight="bold")
        ax.set_xlabel("Trace length L")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)

        twin = ax.twinx()
        twin.plot(x, frame["k_star"], color=colors["k_star"], marker="s", label="k*")
        twin.plot(x, x, color="#888888", linestyle=":", linewidth=1.2, label="k*=L")
        twin.set_ylim(0, max(x) + 0.5)

        if ax is axes[0]:
            ax.set_ylabel("Accuracy / TAS")
        else:
            ax.set_ylabel("")
        if ax is axes[-1]:
            twin.set_ylabel("k*")
        else:
            twin.set_ylabel("")

    legend_handles = [
        plt.Line2D([0], [0], color=colors["accuracy"], marker="o", label="Accuracy"),
        plt.Line2D([0], [0], color=colors["tas"], marker="^", linestyle="--", label="TAS"),
        plt.Line2D([0], [0], color=colors["k_star"], marker="s", label="k*"),
        plt.Line2D([0], [0], color="#888888", linestyle=":", label="k*=L"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("T1-A. Accuracy, TAS, and k* across trace length", y=1.11, fontsize=13, weight="bold")
    fig.savefig(ASSET_DIR / "t1-a-overview.png", bbox_inches="tight")
    plt.close(fig)


def build_t1c_ratio() -> None:
    frame = pd.read_csv(ROOT / "results" / "outputs-0422" / "sources" / "t1c.csv")
    x = frame["L"].to_numpy(dtype=float)
    y = frame["k_star_ratio"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.axhspan(0.70, 0.85, color="#d9e4ef", alpha=0.7, label="NLDD paper band (0.70-0.85)")
    ax.scatter(x, y, s=18, alpha=0.18, color="#333333", label="Per-(q, L) bin")

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#111111", linewidth=2, label="OLS fit")

    ax.set_title("Pooled k*/L vs L", weight="bold")
    ax.set_xlabel("Trace length L")
    ax.set_ylabel("k*/L")
    ax.set_ylim(0.2, 1.02)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.98,
        0.04,
        f"k*/L = {slope:.3f}L + {intercept:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd"},
    )
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(ASSET_DIR / "t1-pooled-kstar-ratio-vs-l.png", bbox_inches="tight")
    plt.close(fig)


def _pivot_surface(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = frame.pivot(index="L", columns="step", values=value_col).sort_index()
    max_step = int(frame["step"].max())
    max_l = int(frame["L"].max())
    pivot = pivot.reindex(index=range(int(frame["L"].min()), max_l + 1), columns=range(1, max_step + 1))
    return pivot


def build_t1b_overall() -> None:
    frame = pd.read_csv(ROOT / "results" / "outputs-0422" / "sources" / "t1b" / "t1b_overall.csv")
    frame = frame[frame["bin_status"] == "ok"].copy()

    scopes = ["easy", "medium", "hard"]
    nldd_max = np.nanpercentile(frame["mean_nldd"], 95)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12), constrained_layout=True)

    for row, scope in enumerate(scopes):
        subset = frame[frame["scope"] == scope]
        nldd = _pivot_surface(subset, "mean_nldd")
        tas = _pivot_surface(subset, "mean_tas_t")

        sns.heatmap(
            nldd,
            ax=axes[row, 0],
            cmap="YlOrRd",
            vmin=0,
            vmax=nldd_max,
            mask=nldd.isna(),
            cbar=row == 0,
            cbar_kws={"shrink": 0.8, "label": "mean NLDD"} if row == 0 else None,
        )
        sns.heatmap(
            tas,
            ax=axes[row, 1],
            cmap="viridis",
            vmin=0,
            vmax=1,
            mask=tas.isna(),
            cbar=row == 0,
            cbar_kws={"shrink": 0.8, "label": "mean TAS"} if row == 0 else None,
        )

        axes[row, 0].set_title(f"{scope.title()} NLDD")
        axes[row, 1].set_title(f"{scope.title()} TAS")
        axes[row, 0].set_ylabel("L")
        axes[row, 1].set_ylabel("")
        axes[row, 0].set_xlabel("Step")
        axes[row, 1].set_xlabel("Step")

    fig.suptitle("T1-B overall heatmap. NLDD diagonal structure and TAS decay", y=1.02, fontsize=13, weight="bold")
    fig.savefig(ASSET_DIR / "t1-b-overall-heatmap.png", bbox_inches="tight")
    plt.close(fig)


def copy_existing_assets() -> None:
    copies = {
        "figure-g-post-horizon-2x2.png": ROOT / "results" / "outputs-0423" / "local-figures" / "deep_dive" / "fig_g_post_horizon_2x2.png",
        "figure-j-tas-slope-pre-vs-post.png": ROOT / "results" / "outputs-0423" / "local-figures" / "deep_dive" / "fig_j_tas_slope_pre_vs_post.png",
        "appendix-figure-e-post-horizon-nldd-sign.png": ROOT / "results" / "outputs-0423" / "local-figures" / "deep_dive" / "fig_e_post_horizon_nldd_sign.png",
        "appendix-figure-f-post-horizon-tas-slope.png": ROOT / "results" / "outputs-0423" / "local-figures" / "deep_dive" / "fig_f_post_horizon_tas_slope.png",
        "appendix-figure-h-kstar-eq1-l-dist.png": ROOT / "results" / "outputs-0423" / "local-figures" / "deep_dive" / "fig_h_kstar_eq1_L_dist.png",
        "appendix-figure-i-nldd-drop-ratio-vs-l.png": ROOT / "results" / "outputs-0423" / "local-figures" / "deep_dive" / "fig_i_nldd_drop_ratio_vs_L.png",
        "appendix-t1d-exemplar-0440.png": ROOT / "results" / "legacy" / "outputs-0421" / "local-figures" / "t1b" / "t1b_gsm8k_platinum_0440.png",
        "appendix-t1d-exemplar-0509.png": ROOT / "results" / "legacy" / "outputs-0421" / "local-figures" / "t1b" / "t1b_gsm8k_platinum_0509.png",
        "appendix-t1d-exemplar-0765.png": ROOT / "results" / "legacy" / "outputs-0421" / "local-figures" / "t1b" / "t1b_gsm8k_platinum_0765.png",
        "appendix-t1d-exemplar-0967.png": ROOT / "results" / "legacy" / "outputs-0421" / "local-figures" / "t1b" / "t1b_gsm8k_platinum_0967.png",
        "appendix-t1d-exemplar-0996.png": ROOT / "results" / "legacy" / "outputs-0421" / "local-figures" / "t1b" / "t1b_gsm8k_platinum_0996.png",
    }

    for name, src in copies.items():
        shutil.copy2(src, ASSET_DIR / name)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    build_t1a_overview()
    build_t1c_ratio()
    build_t1b_overall()
    copy_existing_assets()


if __name__ == "__main__":
    main()
