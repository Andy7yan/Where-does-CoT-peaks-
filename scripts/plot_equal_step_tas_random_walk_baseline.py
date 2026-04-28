"""Plot a high-dimensional equal-step random-walk TAS baseline.

This is a fallback null shape for TAS when empirical hidden-state step norms
are unavailable. It compares observed final TAS against an isotropic random
walk with unit step lengths in hidden_dim dimensions. For speed, the endpoint
norm uses the standard high-dimensional Gaussian approximation:

    ||sum_t u_t|| / L ~= sqrt(chi2(hidden_dim) / hidden_dim) / sqrt(L).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_rows = read_fig_rev1_actual_rows(Path(args.fig_rev1_source_csv))
    baseline_rows = simulate_equal_step_baseline(
        actual_rows=actual_rows,
        hidden_dim=args.hidden_dim,
        n_sim=args.n_sim,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    actual_fit = fit_log_linear(
        lengths=[row["L"] for row in baseline_rows],
        values=[row["actual_mean_tas"] for row in baseline_rows],
    )
    random_walk_fit = fit_log_linear(
        lengths=[row["L"] for row in baseline_rows],
        values=[row["rw_mean_tas"] for row in baseline_rows],
    )
    for row, actual_hat, random_walk_hat in zip(
        baseline_rows,
        actual_fit["fitted"],
        random_walk_fit["fitted"],
    ):
        row["actual_log_fit"] = actual_hat
        row["rw_log_fit"] = random_walk_hat

    csv_path = output_dir / "tas_equal_step_random_walk_baseline.csv"
    png_path = output_dir / "tas_equal_step_random_walk_baseline.png"
    stats_path = output_dir / "tas_equal_step_random_walk_baseline_stats.json"

    write_csv(csv_path, baseline_rows)
    plot_rows(
        png_path,
        baseline_rows,
        hidden_dim=args.hidden_dim,
        actual_fit=actual_fit,
        random_walk_fit=random_walk_fit,
    )
    write_stats(
        stats_path,
        {
            "baseline": "equal_step_isotropic_random_walk",
            "hidden_dim": args.hidden_dim,
            "n_sim": args.n_sim,
            "seed": args.seed,
            "fits": {
                "actual_final_tas_a_plus_b_log_L": {
                    "a": actual_fit["a"],
                    "b": actual_fit["b"],
                    "mse": actual_fit["mse"],
                    "r2": actual_fit["r2"],
                },
                "random_walk_a_plus_b_log_L": {
                    "a": random_walk_fit["a"],
                    "b": random_walk_fit["b"],
                    "mse": random_walk_fit["mse"],
                    "r2": random_walk_fit["r2"],
                },
            },
            "actual_source": {
                "csv": str(Path(args.fig_rev1_source_csv)),
                "definition": (
                    "Same aggregation as outputs-0426 fig_rev_1_tas_decay_vs_L: "
                    "final TAS rows from fig_b_tas_vs_L_source.csv, grouped by L, "
                    "with mean_tas_t averaged over (question_id, L) bins."
                ),
            },
            "outputs": {
                "csv": str(csv_path),
                "png": str(png_path),
                "stats": str(stats_path),
            },
        "note": (
                "Fallback null shape only. It does not use empirical step norms; "
                "the random-walk expectation uses the high-dimensional chi-square endpoint approximation; "
                "the strict distribution-matched baseline remains unavailable "
                "without hidden-state step-norm data."
            ),
        },
    )

    print(f"saved_csv: {csv_path}")
    print(f"saved_png: {png_path}")
    print(f"saved_stats: {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig-rev1-source-csv",
        default="results/outputs-0423/sources/targeted/fig_b_tas_vs_L_source.csv",
        help=(
            "Final-TAS source used to build the 0426 Fig Rev 1 TAS decay plot. "
            "Expected columns include L and mean_tas_t."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/outputs-0427",
        help="Directory for baseline outputs.",
    )
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--n-sim", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def read_fig_rev1_actual_rows(path: Path) -> list[dict[str, float]]:
    by_l: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("mean_tas_t") or row.get("mean_tas")
            if value in (None, "", "nan"):
                continue
            length = int(float(row["L"]))
            by_l.setdefault(length, []).append(float(value))
    rows: list[dict[str, float]] = []
    for length in sorted(by_l):
        values = np.asarray(by_l[length], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        se = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
        rows.append(
            {
                "L": float(length),
                "actual_mean_tas": float(values.mean()),
                "actual_se": se,
                "ci_low": float(values.mean() - 1.96 * se),
                "ci_high": float(values.mean() + 1.96 * se),
                "n_bins": float(values.size),
            }
        )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def simulate_equal_step_baseline(
    *,
    actual_rows: list[dict[str, float]],
    hidden_dim: int,
    n_sim: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for actual in actual_rows:
        length = int(actual["L"])
        values = np.empty(n_sim, dtype=np.float64)
        cursor = 0
        while cursor < n_sim:
            current = min(batch_size, n_sim - cursor)
            endpoint_norm = np.sqrt(rng.chisquare(df=hidden_dim, size=current) / hidden_dim)
            values[cursor : cursor + current] = endpoint_norm / math.sqrt(length)
            cursor += current

        baseline_mean = float(values.mean())
        rows.append(
            {
                "L": length,
                "actual_mean_tas": actual["actual_mean_tas"],
                "actual_se": actual["actual_se"],
                "rw_mean_tas": baseline_mean,
                "rw_p2_5": float(np.quantile(values, 0.025)),
                "rw_p97_5": float(np.quantile(values, 0.975)),
                "rw_theory_1_over_sqrt_L": 1.0 / math.sqrt(length),
                "actual_minus_rw": actual["actual_mean_tas"] - baseline_mean,
                "actual_ci_low": actual["ci_low"],
                "actual_ci_high": actual["ci_high"],
                "n_bins": int(actual["n_bins"]),
            }
        )
    return rows


def fit_log_linear(*, lengths: list[float], values: list[float]) -> dict[str, Any]:
    x_raw = np.asarray(lengths, dtype=float)
    y = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones_like(x_raw), np.log(x_raw)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    mse = float(np.mean(residuals**2))
    centered = y - float(y.mean())
    sst = float(np.sum(centered**2))
    sse = float(np.sum(residuals**2))
    r2 = 1.0 - sse / sst if sst > 0.0 else float("nan")
    return {
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "mse": mse,
        "r2": float(r2),
        "fitted": [float(value) for value in fitted],
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = [
        "L",
        "actual_mean_tas",
        "actual_se",
        "rw_mean_tas",
        "rw_p2_5",
        "rw_p97_5",
        "rw_theory_1_over_sqrt_L",
        "actual_minus_rw",
        "actual_ci_low",
        "actual_ci_high",
        "actual_log_fit",
        "rw_log_fit",
        "n_bins",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(
    path: Path,
    rows: list[dict[str, float]],
    *,
    hidden_dim: int,
    actual_fit: dict[str, Any],
    random_walk_fit: dict[str, Any],
) -> None:
    lengths = np.asarray([row["L"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_mean_tas"] for row in rows], dtype=float)
    actual_low = np.asarray([row["actual_ci_low"] for row in rows], dtype=float)
    actual_high = np.asarray([row["actual_ci_high"] for row in rows], dtype=float)
    rw_mean = np.asarray([row["rw_mean_tas"] for row in rows], dtype=float)
    actual_log_fit = np.asarray([row["actual_log_fit"] for row in rows], dtype=float)
    rw_log_fit = np.asarray([row["rw_log_fit"] for row in rows], dtype=float)

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.fill_between(lengths, actual_low, actual_high, color="#4C78A8", alpha=0.18, linewidth=0)
    ax.plot(
        lengths,
        actual,
        color="#4C78A8",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Actual final TAS from Fig Rev 1 source",
    )
    ax.plot(
        lengths,
        actual_log_fit,
        color="#1F4E79",
        linestyle=":",
        linewidth=2.0,
        label=rf"Actual $a+b\log L$ fit, MSE={actual_fit['mse']:.2e}",
    )
    ax.plot(
        lengths,
        rw_mean,
        color="#F58518",
        linestyle="--",
        linewidth=2.2,
        label=rf"Random-walk baseline: equal-step isotropic, $d={hidden_dim}$",
    )
    ax.plot(
        lengths,
        rw_log_fit,
        color="#B75D00",
        linestyle="-.",
        linewidth=2.0,
        label=rf"Random-walk $a+b\log L$ fit",
    )
    ax.set_xlabel("Trace length L")
    ax.set_ylabel("Final TAS")
    ax.set_title("Actual Final TAS vs High-D Random Walk Baseline")
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(False, axis="x")
    ax.set_ylim(0.0, min(1.05, max(1.0, float(np.nanmax(rw_mean)) + 0.05)))
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_stats(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
