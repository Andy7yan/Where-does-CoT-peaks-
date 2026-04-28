"""Contrast high k*/L bins against ordinary bins to diagnose the ratio rebound."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


EPSILON = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def to_int(value: Any) -> int | None:
    value = to_float(value)
    if value is None:
        return None
    return int(value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(values: list[float]) -> dict[str, float | int | None]:
    values = [value for value in values if value is not None and not math.isnan(value)]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = rank
        cursor = end
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 3 or len(y_values) < 3:
        return None
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_denominator == 0.0 or y_denominator == 0.0:
        return None
    return numerator / (x_denominator * y_denominator)


def spearman(x_values: list[float], y_values: list[float]) -> float | None:
    return pearson(rankdata(x_values), rankdata(y_values))


def group_for_ratio(ratio: float) -> str:
    if abs(ratio - 1.0) <= EPSILON:
        return "eq1"
    if ratio >= 0.85:
        return "high_lt1_0.85_1"
    if ratio >= 0.4:
        return "mid_0.4_0.85"
    return "low_lt0.4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/papery-pq")
    parser.add_argument("--deep-dive-dir", default="results/outputs-0426/sources/deep_dive")
    parser.add_argument("--output-dir", default="results/outputs-0426/sources/rebound")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    t1c = read_csv(run_dir / "pq_analysis" / "t1c_kstar_ratio.csv")
    t2b = read_csv(run_dir / "pq_analysis" / "t2b_lstar_difficulty.csv")
    post_nldd = read_csv(Path(args.deep_dive_dir) / "post_horizon_nldd.csv")

    lstar = {
        row["question_id"]: to_int(row.get("l_star_A"))
        for row in t2b
    }
    post_steps = {
        (row["question_id"], to_int(row.get("L"))): to_int(row.get("n_post_steps"))
        for row in post_nldd
    }

    enriched: list[dict[str, Any]] = []
    for row in t1c:
        ratio = to_float(row.get("k_star_ratio"))
        length = to_int(row.get("L"))
        k_star = to_int(row.get("k_star"))
        if ratio is None or length is None or k_star is None:
            continue
        l_star_A = lstar.get(row["question_id"])
        enriched.append(
            {
                "question_id": row["question_id"],
                "L": length,
                "k_star": k_star,
                "k_star_ratio": ratio,
                "ratio_group": group_for_ratio(ratio),
                "difficulty_score": to_float(row.get("difficulty_score")),
                "n_clean": to_int(row.get("n_clean")),
                "l_star_A": l_star_A,
                "L_minus_l_star_A": (length - l_star_A) if isinstance(l_star_A, int) else None,
                "boundary_lag": length - k_star,
                "n_post_steps": post_steps.get((row["question_id"], length)),
            }
        )

    group_order = ["eq1", "high_lt1_0.85_1", "mid_0.4_0.85", "low_lt0.4"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["ratio_group"]].append(row)

    group_rows: list[dict[str, Any]] = []
    for group in group_order:
        rows = grouped[group]
        lengths = [float(row["L"]) for row in rows]
        diffs = [float(row["difficulty_score"]) for row in rows if row["difficulty_score"] is not None]
        lminus = [float(row["L_minus_l_star_A"]) for row in rows if row["L_minus_l_star_A"] is not None]
        post = [float(row["n_post_steps"]) for row in rows if row["n_post_steps"] is not None]
        l_hist = Counter(int(value) for value in lengths)
        group_rows.append(
            {
                "ratio_group": group,
                "bin_count": len(rows),
                "question_count": len({row["question_id"] for row in rows}),
                "mean_L": summarize(lengths)["mean"],
                "median_L": summarize(lengths)["median"],
                "share_L_le_4": mean([value <= 4 for value in lengths]) if lengths else None,
                "share_L_ge_6": mean([value >= 6 for value in lengths]) if lengths else None,
                "mean_difficulty_score": summarize(diffs)["mean"],
                "median_difficulty_score": summarize(diffs)["median"],
                "mean_L_minus_l_star_A": summarize(lminus)["mean"],
                "median_L_minus_l_star_A": summarize(lminus)["median"],
                "share_L_after_lstar": mean([value > 0 for value in lminus]) if lminus else None,
                "mean_n_post_steps": summarize(post)["mean"],
                "median_n_post_steps": summarize(post)["median"],
                "L_histogram_json": json.dumps(dict(sorted(l_hist.items())), sort_keys=True),
            }
        )

    totals_by_l: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        totals_by_l[int(row["L"])].append(row)
    by_l_rows: list[dict[str, Any]] = []
    for length in sorted(totals_by_l):
        rows = totals_by_l[length]
        eq1_count = sum(1 for row in rows if row["ratio_group"] == "eq1")
        high_count = sum(1 for row in rows if row["ratio_group"] in {"eq1", "high_lt1_0.85_1"})
        ratios = [float(row["k_star_ratio"]) for row in rows]
        by_l_rows.append(
            {
                "L": length,
                "bin_count": len(rows),
                "eq1_count": eq1_count,
                "eq1_rate": eq1_count / len(rows),
                "high_ge_0.85_count": high_count,
                "high_ge_0.85_rate": high_count / len(rows),
                "mean_k_star_ratio": mean(ratios),
                "median_k_star_ratio": median(ratios),
            }
        )

    correlation_features = [
        "L",
        "difficulty_score",
        "n_clean",
        "l_star_A",
        "L_minus_l_star_A",
        "boundary_lag",
        "n_post_steps",
    ]
    corr_rows: list[dict[str, Any]] = []
    for feature in correlation_features:
        pairs = [
            (float(row["k_star_ratio"]), float(row[feature]))
            for row in enriched
            if row.get(feature) is not None
        ]
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]
        corr_rows.append(
            {
                "feature": feature,
                "n_complete": len(pairs),
                "spearman_rho": spearman(x_values, y_values),
                "pearson_r": pearson(x_values, y_values),
            }
        )

    write_csv(
        output_dir / "ratio_group_contrast.csv",
        group_rows,
        [
            "ratio_group",
            "bin_count",
            "question_count",
            "mean_L",
            "median_L",
            "share_L_le_4",
            "share_L_ge_6",
            "mean_difficulty_score",
            "median_difficulty_score",
            "mean_L_minus_l_star_A",
            "median_L_minus_l_star_A",
            "share_L_after_lstar",
            "mean_n_post_steps",
            "median_n_post_steps",
            "L_histogram_json",
        ],
    )
    write_csv(
        output_dir / "ratio_by_L.csv",
        by_l_rows,
        [
            "L",
            "bin_count",
            "eq1_count",
            "eq1_rate",
            "high_ge_0.85_count",
            "high_ge_0.85_rate",
            "mean_k_star_ratio",
            "median_k_star_ratio",
        ],
    )
    write_csv(
        output_dir / "kstar_ratio_feature_correlations.csv",
        corr_rows,
        ["feature", "n_complete", "spearman_rho", "pearson_r"],
    )
    write_json(
        output_dir / "rebound_summary.json",
        {
            "total_bins": len(enriched),
            "ratio_group_counts": {group: len(grouped[group]) for group in group_order},
            "main_diagnostic": {
                "boundary_lag_definition": "L - k_star; eq1 has boundary_lag=0 by construction",
                "discrete_resolution_note": "Small L has coarse possible k*/L values, so the top bin is easier to hit and has no post-k* evidence.",
            },
        },
    )

    print(f"group_contrast: {output_dir / 'ratio_group_contrast.csv'}")
    print(f"ratio_by_L: {output_dir / 'ratio_by_L.csv'}")


if __name__ == "__main__":
    main()
