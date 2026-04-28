"""Extract step norms and run the strict isotropic TAS baseline.

This script intentionally refuses to infer step norms from prefix TAS alone.
It first checks the expected JSONL fields, then falls back to hidden-state
``clean.npz`` dumps if they exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


STEP_NORM_FIELDS = ("clean_step_lengths", "step_norms")
CUMULATIVE_FIELDS = ("clean_cumulative_disp", "cumulative_disp")
DEFAULT_HIDDEN_DIM = 4096


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tas_path = Path(args.tas_curve_per_trace)
    schema_rows = inspect_jsonl_schema(tas_path, n=3)
    print_schema(schema_rows)
    write_json(
        output_dir / "tas_curve_schema_sample.json",
        {
            "path": str(tas_path),
            "sample_rows": schema_rows,
        },
    )

    step_norm_path = output_dir / "step_norms_per_trace.jsonl"
    source_result = extract_step_norms_from_jsonl(
        tas_path=tas_path,
        output_path=step_norm_path,
    )

    if source_result["status"] != "completed":
        npz_root = Path(args.hidden_npz_root)
        npz_result = extract_step_norms_from_npz(
            npz_root=npz_root,
            output_path=step_norm_path,
        )
        source_result = {
            "jsonl": source_result,
            "npz": npz_result,
        }
        if npz_result["status"] != "completed":
            discovery_path = output_dir / "step_norm_source_discovery.json"
            write_json(
                discovery_path,
                {
                    "status": "blocked_missing_step_norm_source",
                    "tas_curve_per_trace": str(tas_path),
                    "hidden_npz_root": str(npz_root),
                    "schema_sample_path": str(output_dir / "tas_curve_schema_sample.json"),
                    "source_result": source_result,
                    "message": (
                        "No step_norms/clean_step_lengths/cumulative displacement fields were present "
                        "in tas_curve_per_trace.jsonl, and no usable clean.npz hidden-state dumps were found."
                    ),
                },
            )
            print(f"blocked_missing_step_norm_source: {discovery_path}")
            raise SystemExit(2)

    actual_rows = read_actual_diagnostic(Path(args.actual_diagnostic_csv))
    stats = read_json(Path(args.diagnostic_stats_json))
    hidden_dim = resolve_hidden_dim(
        step_norm_path=step_norm_path,
        explicit_dim=args.hidden_dim,
        fallback_dim=DEFAULT_HIDDEN_DIM,
    )
    iso_rows = simulate_isotropic_baseline(
        step_norm_path=step_norm_path,
        actual_rows=actual_rows,
        hidden_dim=hidden_dim,
        n_sim=args.n_sim,
        seed=args.seed,
        batch_size=args.sim_batch_size,
    )

    iso_csv = output_dir / "isotropic_baseline.csv"
    write_isotropic_csv(iso_csv, iso_rows)
    comparison_png = output_dir / "tas_isotropic_comparison.png"
    plot_comparison(
        rows=iso_rows,
        stats=stats,
        output_path=comparison_png,
    )
    update_stats(
        stats_path=Path(args.diagnostic_stats_json),
        stats=stats,
        iso_rows=iso_rows,
        hidden_dim=hidden_dim,
        n_sim=args.n_sim,
        iso_csv=iso_csv,
        comparison_png=comparison_png,
        step_norm_path=step_norm_path,
    )
    update_report(
        report_path=Path(args.diagnostic_report),
        stats=stats,
        iso_rows=iso_rows,
        hidden_dim=hidden_dim,
        n_sim=args.n_sim,
        seed=args.seed,
    )

    print(f"saved_step_norms: {step_norm_path}")
    print(f"saved_isotropic_csv: {iso_csv}")
    print(f"saved_comparison_png: {comparison_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tas-curve-per-trace",
        default="results/papery-pq/pq_analysis/tas_curve_per_trace.jsonl",
    )
    parser.add_argument(
        "--actual-diagnostic-csv",
        default="results/output-0427/tas_diagnostic.csv",
    )
    parser.add_argument(
        "--diagnostic-stats-json",
        default="results/output-0427/tas_diagnostic_stats.json",
    )
    parser.add_argument(
        "--diagnostic-report",
        default="results/output-0427/diagnostic_report.md",
    )
    parser.add_argument(
        "--hidden-npz-root",
        default="results/papery-pq/nldd/pq",
    )
    parser.add_argument("--output-dir", default="results/output-0427")
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=1000,
        help="Simulation batch size per L; reduce if memory is tight.",
    )
    return parser.parse_args()


def inspect_jsonl_schema(path: Path, *, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "line": index,
                    "fields": sorted(row.keys()),
                    "values": row,
                }
            )
            if len(rows) >= n:
                break
    if not rows:
        raise ValueError(f"No JSON rows found in {path}")
    return rows


def print_schema(rows: list[dict[str, Any]]) -> None:
    for sample in rows:
        print(f"ROW {sample['line']}")
        print(f"fields={sample['fields']}")
        print(json.dumps(sample["values"], ensure_ascii=False))


def extract_step_norms_from_jsonl(*, tas_path: Path, output_path: Path) -> dict[str, Any]:
    first_rows = inspect_jsonl_schema(tas_path, n=3)
    fields = set().union(*(set(row["fields"]) for row in first_rows))
    step_field = next((field for field in STEP_NORM_FIELDS if field in fields), None)
    cumulative_field = next((field for field in CUMULATIVE_FIELDS if field in fields), None)
    if step_field is None and cumulative_field is None:
        return {
            "status": "missing_step_level_fields",
            "available_fields": sorted(fields),
            "required_any": [*STEP_NORM_FIELDS, *CUMULATIVE_FIELDS],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    with tas_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            is_correct = row.get("is_correct", True)
            if not bool(is_correct):
                skipped += 1
                continue
            length = resolve_length(row)
            if length is None or length < 3:
                skipped += 1
                continue
            if step_field is not None:
                norms = [float(value) for value in row.get(step_field, [])]
            else:
                cumulative = [float(value) for value in row.get(cumulative_field, [])]
                norms = diff_cumulative(cumulative)
            if len(norms) != length:
                raise ValueError(
                    f"Step norm length mismatch for row trace={row.get('trace_id') or row.get('source_trace_id')}: "
                    f"L={length}, len(step_norms)={len(norms)}"
                )
            if not norms or any((not math.isfinite(value)) or value < 0.0 for value in norms):
                skipped += 1
                continue
            sink.write(
                json.dumps(
                    {
                        "question_id": str(row.get("question_id", "")),
                        "trace_id": str(row.get("trace_id") or row.get("source_trace_id") or row.get("sample_id", "")),
                        "L": length,
                        "step_norms": norms,
                        "hidden_dim": row.get("hidden_dim"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return {
        "status": "completed" if count else "empty_after_filter",
        "source": str(tas_path),
        "output": str(output_path),
        "n_traces": count,
        "skipped": skipped,
        "field": step_field or cumulative_field,
    }


def extract_step_norms_from_npz(*, npz_root: Path, output_path: Path) -> dict[str, Any]:
    if not npz_root.exists():
        return {
            "status": "missing_npz_root",
            "root": str(npz_root),
        }
    clean_npzs = sorted(npz_root.rglob("clean.npz"))
    if not clean_npzs:
        return {
            "status": "no_clean_npz_files",
            "root": str(npz_root),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    samples: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as sink:
        for npz_path in clean_npzs:
            with np.load(npz_path) as payload:
                keys = list(payload.keys())
                if len(samples) < 3:
                    samples.append({"path": str(npz_path), "keys": keys})
                if "hidden_states" not in payload or "step_boundary_indices" not in payload:
                    continue
                hidden_states = np.asarray(payload["hidden_states"])
                boundaries = np.asarray(payload["step_boundary_indices"], dtype=int).reshape(-1)
                if hidden_states.ndim > 2:
                    hidden_states = hidden_states[-1]
                if hidden_states.ndim != 2 or boundaries.size < 4:
                    continue
                points = hidden_states[boundaries]
                step_vectors = np.diff(points, axis=0)
                norms = np.linalg.norm(step_vectors, axis=1).astype(float).tolist()
                length = len(norms)
                if length < 3:
                    continue
                sink.write(
                    json.dumps(
                        {
                            "question_id": infer_question_id(npz_path),
                            "trace_id": infer_trace_id(npz_path),
                            "L": length,
                            "step_norms": norms,
                            "hidden_dim": int(points.shape[-1]),
                            "source_npz": str(npz_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1

    return {
        "status": "completed" if count else "no_usable_npz_payloads",
        "root": str(npz_root),
        "n_npz": len(clean_npzs),
        "n_traces": count,
        "sample_npz": samples,
        "output": str(output_path),
    }


def resolve_length(row: dict[str, Any]) -> int | None:
    for key in ("L", "actual_num_steps", "length"):
        if row.get(key) is not None:
            return int(row[key])
    norms = row.get("step_norms") or row.get("clean_step_lengths")
    if norms is not None:
        return len(norms)
    return None


def diff_cumulative(cumulative: list[float]) -> list[float]:
    if not cumulative:
        return []
    return [cumulative[0], *[cumulative[index] - cumulative[index - 1] for index in range(1, len(cumulative))]]


def read_actual_diagnostic(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            length = int(row["L"])
            rows[length] = {
                "actual_mean_tas": float(row["actual_mean_tas"]),
                "actual_se": float(row["actual_se"]),
                "inv_sqrt_fit": float(row["inv_sqrt_fit"]),
                "log_linear_fit": float(row["log_linear_fit"]),
                "n_traces": int(row["n_traces"]),
            }
    return rows


def resolve_hidden_dim(*, step_norm_path: Path, explicit_dim: int | None, fallback_dim: int) -> int:
    if explicit_dim is not None:
        return int(explicit_dim)
    dims: set[int] = set()
    with step_norm_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("hidden_dim") is not None:
                dims.add(int(row["hidden_dim"]))
    if len(dims) > 1:
        raise ValueError(f"Multiple hidden dims found in step norms: {sorted(dims)}")
    if dims:
        return next(iter(dims))
    return fallback_dim


def simulate_isotropic_baseline(
    *,
    step_norm_path: Path,
    actual_rows: dict[int, dict[str, float]],
    hidden_dim: int,
    n_sim: int,
    seed: int,
    batch_size: int,
) -> list[dict[str, float]]:
    pools: dict[int, list[float]] = {}
    trace_counts: dict[int, int] = {}
    with step_norm_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            length = int(row.get("L", row.get("length")))
            if length < 3:
                continue
            norms = [float(value) for value in row.get("step_norms", [])]
            if len(norms) != length:
                raise ValueError(f"L={length} but len(step_norms)={len(norms)} in {step_norm_path}")
            pools.setdefault(length, []).extend(norms)
            trace_counts[length] = trace_counts.get(length, 0) + 1

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for length in sorted(pools):
        if length not in actual_rows:
            continue
        pool = np.asarray(pools[length], dtype=np.float64)
        values = np.empty(n_sim, dtype=np.float64)
        cursor = 0
        while cursor < n_sim:
            current = min(batch_size, n_sim - cursor)
            sampled = rng.choice(pool, size=(current, length), replace=True)
            directions = rng.normal(size=(current, length, hidden_dim))
            directions /= np.linalg.norm(directions, axis=2, keepdims=True)
            displacement = np.linalg.norm((sampled[:, :, None] * directions).sum(axis=1), axis=1)
            path_length = sampled.sum(axis=1)
            values[cursor : cursor + current] = displacement / path_length
            cursor += current
        actual = actual_rows[length]
        rows.append(
            {
                "L": length,
                "actual_mean_tas": actual["actual_mean_tas"],
                "actual_se": actual["actual_se"],
                "iso_mean_tas": float(values.mean()),
                "iso_std": float(values.std(ddof=1)),
                "iso_p2_5": float(np.quantile(values, 0.025)),
                "iso_p97_5": float(np.quantile(values, 0.975)),
                "n_traces": int(trace_counts[length]),
                "n_pool": int(pool.size),
                "inv_sqrt_fit": actual["inv_sqrt_fit"],
                "log_linear_fit": actual["log_linear_fit"],
            }
        )
    if not rows:
        raise ValueError("No overlapping L bins between step norms and actual diagnostic rows.")
    return rows


def write_isotropic_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = [
        "L",
        "actual_mean_tas",
        "actual_se",
        "iso_mean_tas",
        "iso_std",
        "iso_p2_5",
        "iso_p97_5",
        "n_traces",
        "n_pool",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot_comparison(*, rows: list[dict[str, float]], stats: dict[str, Any], output_path: Path) -> None:
    lengths = np.asarray([row["L"] for row in rows], dtype=float)
    actual = np.asarray([row["actual_mean_tas"] for row in rows], dtype=float)
    actual_se = np.asarray([row["actual_se"] for row in rows], dtype=float)
    iso_mean = np.asarray([row["iso_mean_tas"] for row in rows], dtype=float)
    iso_low = np.asarray([row["iso_p2_5"] for row in rows], dtype=float)
    iso_high = np.asarray([row["iso_p97_5"] for row in rows], dtype=float)
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
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.fill_between(lengths, actual - actual_se, actual + actual_se, color="#4C78A8", alpha=0.18, linewidth=0)
    ax.plot(lengths, actual, color="#4C78A8", linewidth=2.0, marker="o", markersize=4, label="Actual TAS mean +/- SE")
    ax.fill_between(lengths, iso_low, iso_high, color="#F58518", alpha=0.18, linewidth=0, label="Isotropic 95% CI")
    ax.plot(lengths, iso_mean, color="#F58518", linestyle="--", linewidth=2.0, label="Isotropic mean")
    ax.plot(lengths, inv_fit, color="#E45756", linewidth=1.8, label="a / sqrt(L) fit")
    ax.plot(lengths, log_fit, color="#54A24B", linestyle="--", linewidth=1.8, label="a + b log(L) fit")
    ax.set_xlabel("L")
    ax.set_ylabel("TAS")
    ax.set_title("TAS Decay: Actual vs Distribution-Matched Isotropic Baseline")
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(False, axis="x")
    ax.set_ylim(0.0, min(1.05, max(1.0, float(np.nanmax(actual + actual_se)) + 0.05)))
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def update_stats(
    *,
    stats_path: Path,
    stats: dict[str, Any],
    iso_rows: list[dict[str, float]],
    hidden_dim: int,
    n_sim: int,
    iso_csv: Path,
    comparison_png: Path,
    step_norm_path: Path,
) -> None:
    mean_delta_pct = float(
        np.mean(
            [
                100.0 * (row["actual_mean_tas"] - row["iso_mean_tas"]) / row["actual_mean_tas"]
                for row in iso_rows
                if row["actual_mean_tas"] != 0.0
            ]
        )
    )
    stats.update(
        {
            "strict_isotropic_baseline": True,
            "isotropic_status": "completed",
            "isotropic_D": hidden_dim,
            "isotropic_n_sim": n_sim,
            "isotropic_mean_delta_pct": mean_delta_pct,
            "isotropic_csv": str(iso_csv),
            "isotropic_comparison_png": str(comparison_png),
            "step_norms_per_trace": str(step_norm_path),
        }
    )
    write_json(stats_path, stats)


def update_report(
    *,
    report_path: Path,
    stats: dict[str, Any],
    iso_rows: list[dict[str, float]],
    hidden_dim: int,
    n_sim: int,
    seed: int,
) -> None:
    del stats
    mean_delta_pct = float(
        np.mean(
            [
                100.0 * (row["actual_mean_tas"] - row["iso_mean_tas"]) / row["actual_mean_tas"]
                for row in iso_rows
                if row["actual_mean_tas"] != 0.0
            ]
        )
    )
    direction = "above" if mean_delta_pct > 0 else "below"
    if abs(mean_delta_pct) < 1.0:
        interpretation = "Actual TAS is close to the isotropic baseline; most decay is consistent with path geometry."
    elif mean_delta_pct > 0:
        interpretation = "Actual trajectories retain more endpoint displacement than isotropic direction-randomized walks."
    else:
        interpretation = "Actual trajectories are more direction-cancelling than the isotropic baseline."

    section_lines = [
        "## Isotropic Baseline",
        "",
        f"Distribution-matched simulation: N_sim={n_sim}, D={hidden_dim}, seed={seed}",
        f"Trace count used: {sum(int(row['n_traces']) for row in iso_rows)}",
        f"L range: {int(min(row['L'] for row in iso_rows))} to {int(max(row['L'] for row in iso_rows))}",
        "",
        "Key comparison (mean TAS):",
        "",
        "| L | Actual | Isotropic | Delta (Actual - Iso) | Delta/Actual (%) |",
        "|---|--------|-----------|----------------------|------------------|",
    ]
    for row in iso_rows:
        delta = row["actual_mean_tas"] - row["iso_mean_tas"]
        pct = 100.0 * delta / row["actual_mean_tas"] if row["actual_mean_tas"] else float("nan")
        lines.append(
            f"| {int(row['L'])} | {row['actual_mean_tas']:.4f} | {row['iso_mean_tas']:.4f} | {delta:.4f} | {pct:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Overall: actual TAS is {direction} isotropic baseline by {abs(mean_delta_pct):.1f}% on average.",
            f"Interpretation: {interpretation}",
        ]
    )
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8").rstrip().splitlines()
        try:
            start = existing.index("## Isotropic Baseline")
            lines = existing[:start]
        except ValueError:
            lines = existing
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(section_lines)
    else:
        lines = ["# TAS Null-Hypothesis Diagnostic", "", *section_lines]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_question_id(path: Path) -> str:
    parts = path.parts
    for part in reversed(parts):
        if part.startswith("gsm8k_"):
            return part
    return ""


def infer_trace_id(path: Path) -> str:
    if len(path.parts) >= 2:
        return path.parts[-2]
    return path.stem


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
