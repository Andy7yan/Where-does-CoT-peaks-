"""Build the TAS null-hypothesis diagnostic from retained clean traces.

The strict isotropic baseline needs empirical hidden-state step norms. If a
step-norm JSONL is available, this script uses it directly. Otherwise it can
optionally recompute step norms from the configured model backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_phase1.nldd_shared import _flatten_numeric_values


CSV_COLUMNS = [
    "L",
    "actual_mean_tas",
    "actual_se",
    "isotropic_mean_tas",
    "isotropic_lower_95",
    "isotropic_upper_95",
    "inv_sqrt_fit",
    "log_linear_fit",
    "n_traces",
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    actual_by_l = load_actual_final_tas(Path(args.tas_curve_per_trace))
    if not actual_by_l:
        raise ValueError(f"No final TAS rows found in {args.tas_curve_per_trace}")

    step_norm_path = Path(args.step_norms_path) if args.step_norms_path else output_dir / "step_norms_per_trace.jsonl"
    metadata: dict[str, Any] = {
        "seed": args.seed,
        "n_sim": args.n_sim,
        "tas_curve_per_trace": str(Path(args.tas_curve_per_trace)),
        "step_norms_path": str(step_norm_path),
        "strict_isotropic_baseline": False,
        "isotropic_status": "missing_step_norms",
    }

    if not step_norm_path.exists() and args.recompute_step_norms:
        recompute_step_norms(
            run_dir=Path(args.run_dir),
            config_path=Path(args.config),
            output_path=step_norm_path,
            max_samples=args.max_samples,
        )

    step_norm_pools: dict[int, list[float]] = {}
    hidden_dim: int | None = None
    if step_norm_path.exists():
        step_norm_pools, hidden_dim = load_step_norm_pools(step_norm_path)
        metadata["hidden_dim"] = hidden_dim
        metadata["strict_isotropic_baseline"] = bool(step_norm_pools and hidden_dim)
        metadata["isotropic_status"] = "ok" if metadata["strict_isotropic_baseline"] else "empty_step_norms"

    actual_summary = summarize_actual(actual_by_l)
    isotropic_summary = (
        simulate_isotropic(step_norm_pools, hidden_dim, n_sim=args.n_sim, rng=rng)
        if step_norm_pools and hidden_dim
        else {}
    )
    inv_fit, inv_stats = fit_inv_sqrt(actual_summary)
    log_fit, log_stats = fit_log_linear(actual_summary)

    rows = build_output_rows(
        actual_summary=actual_summary,
        isotropic_summary=isotropic_summary,
        inv_fit=inv_fit,
        log_fit=log_fit,
    )
    csv_path = output_dir / "tas_diagnostic.csv"
    write_csv(csv_path, rows)

    png_path = output_dir / "tas_diagnostic.png"
    plot_diagnostic(
        rows=rows,
        png_path=png_path,
        has_isotropic=bool(isotropic_summary),
    )

    stats_payload = {
        **metadata,
        "n_lengths": len(rows),
        "n_traces": int(sum(row["n_traces"] for row in rows)),
        "inv_sqrt_fit": inv_stats,
        "log_linear_fit": log_stats,
        "outputs": {
            "csv": str(csv_path),
            "png": str(png_path),
            "stats": str(output_dir / "tas_diagnostic_stats.json"),
            "report": str(output_dir / "diagnostic_report.md"),
        },
    }
    (output_dir / "tas_diagnostic_stats.json").write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "diagnostic_report.md", stats_payload)

    print(f"saved_csv: {csv_path}")
    print(f"saved_png: {png_path}")
    print(f"saved_stats: {output_dir / 'tas_diagnostic_stats.json'}")
    print(f"isotropic_status: {stats_payload['isotropic_status']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="results/papery-pq",
        help="Run directory containing retained clean traces.",
    )
    parser.add_argument(
        "--tas-curve-per-trace",
        default="results/papery-pq/pq_analysis/tas_curve_per_trace.jsonl",
        help="Existing per-trace TAS curve JSONL.",
    )
    parser.add_argument(
        "--config",
        default="configs/stage1_per_question.yaml",
        help="Config to use when recomputing hidden-state step norms.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/output-0427",
        help="Output directory for tas_diagnostic.csv/png.",
    )
    parser.add_argument(
        "--step-norms-path",
        default=None,
        help="Optional JSONL with per-trace hidden-state step_norms.",
    )
    parser.add_argument(
        "--recompute-step-norms",
        action="store_true",
        help="Recompute hidden-state step norms via the configured local model backend.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for recomputing step norms, intended only for smoke checks.",
    )
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_actual_final_tas(path: Path) -> dict[int, list[float]]:
    values: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            length = int(row["length"])
            if int(row["step_index"]) != length:
                continue
            value = row.get("tas_value")
            if value is None:
                continue
            values.setdefault(length, []).append(float(value))
    return values


def summarize_actual(actual_by_l: dict[int, list[float]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for length in sorted(actual_by_l):
        values = np.asarray(actual_by_l[length], dtype=float)
        se = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
        rows.append(
            {
                "L": float(length),
                "actual_mean_tas": float(values.mean()),
                "actual_se": se,
                "n_traces": float(values.size),
            }
        )
    return rows


def recompute_step_norms(
    *,
    run_dir: Path,
    config_path: Path,
    output_path: Path,
    max_samples: int | None,
) -> None:
    from src.analysis_phase1.backend import load_analysis_backend
    from src.analysis_phase1.io import load_analysis_samples
    from src.analysis_phase1.pq_io import load_per_question_samples
    from src.common.settings import ExperimentConfig

    config = ExperimentConfig.from_yaml(str(config_path))
    backend = load_analysis_backend(config)
    trace_trajectory_fn = backend["trace_trajectory_fn"]

    if (run_dir / "per_question").exists():
        samples = load_per_question_samples(run_dir)
    else:
        samples = load_analysis_samples(run_dir)
    if max_samples is not None:
        samples = samples[: max(0, int(max_samples))]
    if not samples:
        raise ValueError(f"No retained clean samples found under {run_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            vectors = trace_trajectory_fn(sample.question_text, sample.clean_steps)
            flattened = [np.asarray(_flatten_numeric_values(vector), dtype=float) for vector in vectors]
            if len(flattened) <= 1:
                continue
            hidden_dim = int(flattened[0].shape[0])
            step_norms = [
                float(np.linalg.norm(flattened[index + 1] - flattened[index]))
                for index in range(len(flattened) - 1)
            ]
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "source_trace_id": sample.source_trace_id,
                        "question_id": sample.question_id,
                        "difficulty": sample.difficulty,
                        "length": sample.length,
                        "hidden_dim": hidden_dim,
                        "step_norms": step_norms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_step_norm_pools(path: Path) -> tuple[dict[int, list[float]], int | None]:
    pools: dict[int, list[float]] = {}
    hidden_dims: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            length = int(row.get("length", row.get("L")))
            norms = [float(value) for value in row.get("step_norms", [])]
            if not norms:
                continue
            pools.setdefault(length, []).extend(norm for norm in norms if norm > 0.0)
            if row.get("hidden_dim") is not None:
                hidden_dims.add(int(row["hidden_dim"]))
    if len(hidden_dims) > 1:
        raise ValueError(f"Step-norm file contains multiple hidden dims: {sorted(hidden_dims)}")
    hidden_dim = next(iter(hidden_dims)) if hidden_dims else None
    pools = {length: values for length, values in pools.items() if values}
    return pools, hidden_dim


def simulate_isotropic(
    pools: dict[int, list[float]],
    hidden_dim: int,
    *,
    n_sim: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for length in sorted(pools):
        pool = np.asarray(pools[length], dtype=float)
        if pool.size == 0:
            continue
        values = np.empty(n_sim, dtype=float)
        for sim_index in range(n_sim):
            step_norms = rng.choice(pool, size=length, replace=True)
            directions = rng.normal(size=(length, hidden_dim))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            displacement = np.linalg.norm((step_norms[:, None] * directions).sum(axis=0))
            path_length = float(step_norms.sum())
            values[sim_index] = displacement / path_length if path_length > 0.0 else np.nan
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        results[length] = {
            "isotropic_mean_tas": float(finite.mean()),
            "isotropic_lower_95": float(np.quantile(finite, 0.025)),
            "isotropic_upper_95": float(np.quantile(finite, 0.975)),
        }
    return results


def fit_inv_sqrt(actual_summary: list[dict[str, float]]) -> tuple[dict[int, float], dict[str, float]]:
    lengths = np.asarray([row["L"] for row in actual_summary], dtype=float)
    y = np.asarray([row["actual_mean_tas"] for row in actual_summary], dtype=float)
    x = 1.0 / np.sqrt(lengths)
    a = float(np.dot(x, y) / np.dot(x, x))
    y_hat = a * x
    stats = model_stats(y, y_hat, num_params=1)
    stats["a"] = a
    return {int(length): float(value) for length, value in zip(lengths, y_hat)}, stats


def fit_log_linear(actual_summary: list[dict[str, float]]) -> tuple[dict[int, float], dict[str, float]]:
    lengths = np.asarray([row["L"] for row in actual_summary], dtype=float)
    y = np.asarray([row["actual_mean_tas"] for row in actual_summary], dtype=float)
    design = np.column_stack([np.ones_like(lengths), np.log(lengths)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coefficients
    stats = model_stats(y, y_hat, num_params=2)
    stats["a"] = float(coefficients[0])
    stats["b"] = float(coefficients[1])
    return {int(length): float(value) for length, value in zip(lengths, y_hat)}, stats


def model_stats(y: np.ndarray, y_hat: np.ndarray, *, num_params: int) -> dict[str, float]:
    residuals = y - y_hat
    sse = float(np.dot(residuals, residuals))
    centered = y - float(y.mean())
    sst = float(np.dot(centered, centered))
    r2 = 1.0 - sse / sst if sst > 0.0 else float("nan")
    n = int(y.size)
    aic = n * math.log(max(sse / n, 1.0e-300)) + 2 * num_params
    return {"r2": float(r2), "aic": float(aic), "sse": sse, "n": float(n)}


def build_output_rows(
    *,
    actual_summary: list[dict[str, float]],
    isotropic_summary: dict[int, dict[str, float]],
    inv_fit: dict[int, float],
    log_fit: dict[int, float],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for actual in actual_summary:
        length = int(actual["L"])
        iso = isotropic_summary.get(length, {})
        rows.append(
            {
                "L": length,
                "actual_mean_tas": actual["actual_mean_tas"],
                "actual_se": actual["actual_se"],
                "isotropic_mean_tas": iso.get("isotropic_mean_tas", float("nan")),
                "isotropic_lower_95": iso.get("isotropic_lower_95", float("nan")),
                "isotropic_upper_95": iso.get("isotropic_upper_95", float("nan")),
                "inv_sqrt_fit": inv_fit[length],
                "log_linear_fit": log_fit[length],
                "n_traces": int(actual["n_traces"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_diagnostic(
    *,
    rows: list[dict[str, float]],
    png_path: Path,
    has_isotropic: bool,
) -> None:
    lengths = np.asarray([row["L"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_mean_tas"] for row in rows], dtype=float)
    actual_se = np.asarray([row["actual_se"] for row in rows], dtype=float)
    inv_fit = np.asarray([row["inv_sqrt_fit"] for row in rows], dtype=float)
    log_fit = np.asarray([row["log_linear_fit"] for row in rows], dtype=float)

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(lengths, actual - actual_se, actual + actual_se, color="#4C78A8", alpha=0.18, linewidth=0)
    ax.plot(lengths, actual, color="#4C78A8", linewidth=2.0, marker="o", markersize=4, label="Actual TAS mean +/- SE")

    if has_isotropic:
        iso_mean = np.asarray([row["isotropic_mean_tas"] for row in rows], dtype=float)
        iso_low = np.asarray([row["isotropic_lower_95"] for row in rows], dtype=float)
        iso_high = np.asarray([row["isotropic_upper_95"] for row in rows], dtype=float)
        valid = np.isfinite(iso_mean) & np.isfinite(iso_low) & np.isfinite(iso_high)
        ax.fill_between(lengths[valid], iso_low[valid], iso_high[valid], color="#9CA3AF", alpha=0.22, linewidth=0)
        ax.plot(lengths[valid], iso_mean[valid], color="#6B7280", linestyle="--", linewidth=2.0, label="Isotropic baseline mean +/- 95% CI")

    ax.plot(lengths, inv_fit, color="#E45756", linewidth=1.8, label="a / sqrt(L) fit")
    ax.plot(lengths, log_fit, color="#54A24B", linestyle="--", linewidth=1.8, label="a + b log(L) fit")
    ax.set_xlabel("L")
    ax.set_ylabel("TAS")
    ax.set_title("TAS Decay: Actual vs Isotropic Random Walk Baseline")
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(False, axis="x")
    y_top = max(1.0, float(np.nanmax(actual + actual_se)) + 0.05)
    ax.set_ylim(0.0, min(1.05, y_top))
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def write_report(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# TAS Null-Hypothesis Diagnostic",
        "",
        f"- CSV: `{stats['outputs']['csv']}`",
        f"- PNG: `{stats['outputs']['png']}`",
        f"- Trace count: {stats['n_traces']}",
        f"- Length bins: {stats['n_lengths']}",
        f"- Isotropic baseline status: `{stats['isotropic_status']}`",
        "",
        "## Fits",
        "",
        f"- `a / sqrt(L)`: R2={stats['inv_sqrt_fit']['r2']:.6g}, AIC={stats['inv_sqrt_fit']['aic']:.6g}, a={stats['inv_sqrt_fit']['a']:.6g}",
        f"- `a + b log(L)`: R2={stats['log_linear_fit']['r2']:.6g}, AIC={stats['log_linear_fit']['aic']:.6g}, a={stats['log_linear_fit']['a']:.6g}, b={stats['log_linear_fit']['b']:.6g}",
    ]
    if not stats["strict_isotropic_baseline"]:
        lines.extend(
            [
                "",
                "## Isotropic Baseline",
                "",
                "Strict distribution-matched isotropic simulation was not run because no hidden-state step-norm JSONL was available.",
                "Re-run with `--recompute-step-norms` in an environment that has the configured local model cache and enough compute, or pass `--step-norms-path`.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
