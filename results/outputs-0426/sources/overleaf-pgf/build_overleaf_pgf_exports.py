from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = ROOT / "results" / "outputs-0426" / "overleaf-pgf"
SOURCE_DIR = OUT_ROOT / "source-0426"

PQ_ROOT = ROOT / "results" / "papery-pq"
PQ_ANALYSIS = PQ_ROOT / "pq_analysis"
OVERALL_ANALYSIS = ROOT / "results" / "papery-overall" / "analysis"
OLD_TARGETED = ROOT / "results" / "outputs-0423" / "sources" / "targeted"
OLD_DEEP_DIVE = ROOT / "results" / "outputs-0423" / "sources" / "deep_dive"


@dataclass
class Defaults:
    lcurve_questions: list[str]
    profile_a_question: str
    profile_a_length: int
    profile_b_question: str
    profile_b_length: int


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = frame.loc[:, columns]
    frame.to_csv(path, index=False, na_rep="nan")


def fmt_num(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def fmt_p(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def label_difficulty(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    return "hard" if float(score) >= 0.5 else "medium"


def ensure_difficulty(frame: pd.DataFrame, qmeta: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "difficulty" in result.columns:
        result = result.drop(columns=["difficulty"])
    if "difficulty_score" in result.columns:
        result = result.drop(columns=["difficulty_score"])
    return result.merge(qmeta[["question_id", "difficulty", "difficulty_score"]], on="question_id", how="left")


def hist_by_difficulty(frame: pd.DataFrame, value_col: str, all_bins: list[int] | None = None) -> pd.DataFrame:
    work = frame.dropna(subset=[value_col]).copy()
    work[value_col] = work[value_col].astype(int)
    bins = all_bins or list(range(int(work[value_col].min()), int(work[value_col].max()) + 1))
    rows: list[dict[str, int]] = []
    for value in bins:
        sub = work[work[value_col] == value]
        rows.append(
            {
                "bin": int(value),
                "medium": int((sub["difficulty"] == "medium").sum()),
                "hard": int((sub["difficulty"] == "hard").sum()),
            }
        )
    return pd.DataFrame(rows)


def export_lcurves(
    qmeta: pd.DataFrame, bin_status: pd.DataFrame, lstar: pd.DataFrame, report: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    exported: list[str] = []
    skipped: list[str] = []
    lstar_json_fallbacks: list[str] = []
    status = bin_status[bin_status["pipeline"] == "pq"].copy()
    lstar_lookup = lstar.set_index("question_id")
    qmeta_lookup = qmeta.set_index("question_id")

    for question_dir in sorted((PQ_ROOT / "per_question").iterdir()):
        if not question_dir.is_dir():
            continue
        question_id = question_dir.name
        lcurve_path = question_dir / "l_curve.csv"
        lstar_json_path = question_dir / "l_star.json"
        if not lcurve_path.exists():
            skipped.append(question_id)
            continue
        if question_id in lstar_lookup.index:
            meta = lstar_lookup.loc[question_id].to_dict()
        elif lstar_json_path.exists():
            meta = json.loads(lstar_json_path.read_text(encoding="utf-8"))
            lstar_json_fallbacks.append(question_id)
        else:
            skipped.append(question_id)
            continue

        lcurve = pd.read_csv(lcurve_path).sort_values("L").copy()
        q_status = status[status["scope"] == question_id][["L", "bin_status"]].copy()
        data = lcurve.merge(q_status, on="L", how="left")
        data["bin_status"] = data["bin_status"].fillna("missing")
        data["sufficient"] = (data["bin_status"] == "ok").astype(int)
        data["accuracy_smooth"] = data["accuracy"].rolling(window=3, center=True, min_periods=1).mean()
        qrow = qmeta_lookup.loc[question_id] if question_id in qmeta_lookup.index else None
        data["se"] = data["accuracy_se"]
        data["n_traces"] = data["n"]
        l_star_a = meta.get("l_star_A")
        l_star_s = meta.get("l_star_S")
        data["l_star_A"] = -1 if pd.isna(l_star_a) else int(l_star_a)
        data["l_star_S"] = -1 if pd.isna(l_star_s) else int(l_star_s)
        data["difficulty"] = qrow["difficulty"] if qrow is not None else "unknown"
        data["difficulty_score"] = float(meta.get("difficulty_score", qrow["difficulty_score"] if qrow is not None else math.nan))
        out = data[
            [
                "L",
                "accuracy",
                "se",
                "n_traces",
                "sufficient",
                "l_star_A",
                "l_star_S",
                "accuracy_smooth",
                "difficulty",
                "difficulty_score",
            ]
        ]
        write_csv(SOURCE_DIR / f"fig_new_1_lcurve_{question_id}.csv", out)
        exported.append(question_id)

    rows: list[dict[str, Any]] = []
    for question_id in exported:
        csv_path = SOURCE_DIR / f"fig_new_1_lcurve_{question_id}.csv"
        data = pd.read_csv(csv_path)
        sufficient = data[data["sufficient"] == 1].sort_values("L")
        if len(sufficient) < 5:
            continue
        endpoints = max(float(sufficient.iloc[0]["accuracy"]), float(sufficient.iloc[-1]["accuracy"]))
        shape_score = float(sufficient["accuracy"].max()) - endpoints
        rows.append(
            {
                "question_id": question_id,
                "difficulty": str(data.iloc[0]["difficulty"]),
                "difficulty_score": float(data.iloc[0]["difficulty_score"]),
                "n_bins": int(len(sufficient)),
                "shape_score": shape_score,
                "l_star_A": int(data.iloc[0]["l_star_A"]),
            }
        )
    candidates = pd.DataFrame(rows)
    defaults: list[str] = []
    for difficulty in ("medium", "hard"):
        subset = candidates[candidates["difficulty"] == difficulty].sort_values(
            ["shape_score", "n_bins", "question_id"], ascending=[False, False, True]
        )
        defaults.extend(subset.head(2)["question_id"].tolist())
    if len(defaults) < 4:
        fallback = candidates.sort_values(["shape_score", "n_bins"], ascending=False)["question_id"].tolist()
        defaults.extend([qid for qid in fallback if qid not in defaults][: 4 - len(defaults)])

    report["fig_new_1"] = {
        "exported_lcurve_csv": len(exported),
        "skipped_questions": skipped,
        "lstar_json_fallbacks": lstar_json_fallbacks,
        "default_questions": defaults[:4],
    }
    return candidates, defaults[:4]


def export_pairing(
    t1c: pd.DataFrame, lstar: pd.DataFrame, qmeta: pd.DataFrame, report: dict[str, Any]
) -> pd.DataFrame:
    t1c_lstar = t1c.rename(columns={"L": "l_star_A", "k_star": "k_star_at_lstar"})[
        ["question_id", "l_star_A", "k_star_at_lstar", "n_clean"]
    ].copy()
    pair = lstar[["question_id", "difficulty_score", "l_star_A", "l_star_S", "l_star_consistent"]].merge(
        t1c_lstar, on=["question_id", "l_star_A"], how="inner"
    )
    pair = ensure_difficulty(pair, qmeta)
    pair["l_star_A"] = pair["l_star_A"].astype(int)
    pair["k_star_at_lstar"] = pair["k_star_at_lstar"].astype(int)
    pair["n_clean"] = pair["n_clean"].astype(int)
    pair = pair[["question_id", "difficulty", "l_star_A", "k_star_at_lstar", "n_clean", "difficulty_score"]].sort_values(
        ["difficulty", "question_id"]
    )

    missing = sorted(set(lstar["question_id"]) - set(pair["question_id"]))
    write_csv(SOURCE_DIR / "fig_new_2_pairing.csv", pair)
    for difficulty in ("medium", "hard"):
        write_csv(
            SOURCE_DIR / f"fig_new_2_pairing_{difficulty}.csv",
            pair[pair["difficulty"] == difficulty],
            ["question_id", "difficulty", "l_star_A", "k_star_at_lstar", "n_clean"],
        )

    min_bin = int(min(pair["l_star_A"].min(), pair["k_star_at_lstar"].min()))
    max_bin = int(max(pair["l_star_A"].max(), pair["k_star_at_lstar"].max()))
    bins = list(range(min_bin, max_bin + 1))
    write_csv(SOURCE_DIR / "fig_new_2_hist_lstar.csv", hist_by_difficulty(pair, "l_star_A", bins))
    write_csv(SOURCE_DIR / "fig_new_2_hist_kstar.csv", hist_by_difficulty(pair.rename(columns={"k_star_at_lstar": "k_bin"}), "k_bin", bins))

    if len(pair) >= 2 and pair["l_star_A"].nunique() > 1 and pair["k_star_at_lstar"].nunique() > 1:
        r, p = stats.pearsonr(pair["l_star_A"].astype(float), pair["k_star_at_lstar"].astype(float))
    else:
        r, p = math.nan, math.nan
    stats_frame = pd.DataFrame(
        [
            {
                "pearson_r": fmt_num(r, 3),
                "p_value": fmt_p(p),
                "n_questions": int(len(pair)),
                "axis_min": min_bin - 0.5,
                "axis_max": max_bin + 0.5,
                "ratio_slope": 0.64,
            }
        ]
    )
    write_csv(SOURCE_DIR / "fig_new_2_stats.csv", stats_frame)
    report["fig_new_2"] = {
        "paired_questions": int(len(pair)),
        "missing_lstar_kstar_questions": missing,
        "pearson_r": fmt_num(r, 3),
        "p_value": fmt_p(p),
    }
    return pair


def export_profiles(
    t1b: pd.DataFrame, t1c: pd.DataFrame, qmeta: pd.DataFrame, report: dict[str, Any]
) -> tuple[str, int, str, int]:
    t1c_meta = ensure_difficulty(t1c, qmeta)
    exported = 0
    missing: list[str] = []
    candidates: list[dict[str, Any]] = []

    for row in t1c_meta.itertuples(index=False):
        question_id = str(row.question_id)
        length = int(row.L)
        profile = t1b[(t1b["question_id"] == question_id) & (t1b["L"] == length)].sort_values("step").copy()
        if profile.empty:
            missing.append(f"{question_id}:L{length}")
            continue
        profile["k_star"] = int(row.k_star)
        out = profile[["step", "mean_nldd", "nldd_se", "mean_tas_t", "tas_t_se", "k_star"]].copy()
        write_csv(SOURCE_DIR / f"fig_new_3_profile_{question_id}_L{length}.csv", out)
        exported += 1

        nldd = profile[(profile["step"] >= 2) & profile["mean_nldd"].notna()].sort_values("step")
        if len(nldd) >= 3:
            peak = float(nldd["mean_nldd"].max())
            final = float(nldd.iloc[-1]["mean_nldd"])
            candidates.append(
                {
                    "question_id": question_id,
                    "L": length,
                    "difficulty": row.difficulty,
                    "k_star": int(row.k_star),
                    "n_clean": int(row.n_clean),
                    "peak": peak,
                    "final": final,
                    "drop": peak - final,
                }
            )

    cand = pd.DataFrame(candidates)
    defaults: list[dict[str, Any]] = []
    for difficulty in ("medium", "hard"):
        bounded = cand[
            (cand["difficulty"] == difficulty)
            & cand["L"].between(5, 12)
            & (cand["n_clean"] >= 5)
            & cand["peak"].between(250, 700)
            & (cand["drop"] > 150)
        ].sort_values(["drop", "peak"], ascending=False)
        if bounded.empty:
            bounded = cand[(cand["difficulty"] == difficulty) & (cand["n_clean"] >= 5)].sort_values("drop", ascending=False)
        defaults.append(bounded.iloc[0].to_dict())

    write_csv(SOURCE_DIR / "fig_new_3_defaults.csv", pd.DataFrame(defaults))
    report["fig_new_3"] = {
        "exported_profile_csv": exported,
        "missing_profiles": missing,
        "default_profiles": [{"question_id": d["question_id"], "L": int(d["L"])} for d in defaults],
    }
    return (
        str(defaults[0]["question_id"]),
        int(defaults[0]["L"]),
        str(defaults[1]["question_id"]),
        int(defaults[1]["L"]),
    )


def export_kstar_eq_l_cases(t1b: pd.DataFrame, t1c: pd.DataFrame, qmeta: pd.DataFrame, report: dict[str, Any]) -> tuple[float, float, float]:
    t1c_meta = ensure_difficulty(t1c, qmeta)
    cases_raw = t1c_meta[np.isclose(t1c_meta["k_star_ratio"], 1.0)].copy()
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in cases_raw.itertuples(index=False):
        question_id = str(row.question_id)
        length = int(row.L)
        profile = t1b[(t1b["question_id"] == question_id) & (t1b["L"] == length)]
        final = profile[profile["step"] == length]["mean_nldd"].dropna()
        penult = profile[profile["step"] == length - 1]["mean_nldd"].dropna()
        if final.empty or penult.empty:
            skipped.append(f"{question_id}:L{length}")
            continue
        rows.append(
            {
                "question_id": question_id,
                "difficulty": row.difficulty,
                "L": length,
                "mean_nldd_final": float(final.iloc[0]),
                "mean_nldd_penultimate": float(penult.iloc[0]),
            }
        )
    cases = pd.DataFrame(rows).sort_values(["difficulty", "L", "question_id"])
    write_csv(SOURCE_DIR / "fig_new_4_kstar_eq_l_cases.csv", cases)
    for difficulty in ("medium", "hard"):
        write_csv(
            SOURCE_DIR / f"fig_new_4_cases_{difficulty}.csv",
            cases[cases["difficulty"] == difficulty],
            ["question_id", "difficulty", "L", "mean_nldd_final", "mean_nldd_penultimate"],
        )
    bins = list(range(int(cases["L"].min()), int(cases["L"].max()) + 1)) if not cases.empty else []
    hist_frame = hist_by_difficulty(cases, "L", bins) if bins else pd.DataFrame(columns=["bin", "medium", "hard"])
    write_csv(SOURCE_DIR / "fig_new_4_L_hist.csv", hist_frame)
    global_median_l = float(t1c["L"].median())
    pct_above = float((cases["mean_nldd_final"] > cases["mean_nldd_penultimate"]).mean() * 100.0) if not cases.empty else math.nan
    min_axis = float(np.nanmin([cases["mean_nldd_final"].min(), cases["mean_nldd_penultimate"].min()])) if not cases.empty else 0.0
    max_axis = float(np.nanmax([cases["mean_nldd_final"].max(), cases["mean_nldd_penultimate"].max()])) if not cases.empty else 1.0
    pad = max(25.0, 0.08 * (max_axis - min_axis))
    diag_min = math.floor((min_axis - pad) / 50.0) * 50.0
    diag_max = math.ceil((max_axis + pad) / 50.0) * 50.0
    stats_frame = pd.DataFrame(
        [
            {
                "n_cases": int(len(cases)),
                "median_L_cases": fmt_num(float(cases["L"].median()) if not cases.empty else math.nan, 2),
                "global_median_L": fmt_num(global_median_l, 2),
                "pct_above_diagonal": fmt_num(pct_above, 1),
                "diag_min": diag_min,
                "diag_max": diag_max,
            }
        ]
    )
    write_csv(SOURCE_DIR / "fig_new_4_stats.csv", stats_frame)
    report["fig_new_4"] = {
        "kstar_eq_l_cases": int(len(cases)),
        "skipped_cases_missing_nldd": skipped,
        "pct_above_diagonal": fmt_num(pct_above, 1),
    }
    hist_ymax = float(math.ceil(float((hist_frame["medium"] + hist_frame["hard"]).max()) * 1.15)) if not hist_frame.empty else 1.0
    return diag_min, diag_max, hist_ymax


def export_tas_decay(qmeta: pd.DataFrame, report: dict[str, Any]) -> None:
    source = pd.read_csv(OLD_TARGETED / "fig_b_tas_vs_L_source.csv")
    data = ensure_difficulty(source, qmeta)
    out = data.rename(columns={"mean_tas_t": "mean_tas", "tas_t_se": "tas_se"})[
        ["question_id", "difficulty", "L", "mean_tas", "tas_se"]
    ].copy()
    out["L"] = out["L"].astype(int)
    write_csv(SOURCE_DIR / "fig_rev_1_tas_decay.csv", out)
    for difficulty in ("medium", "hard"):
        write_csv(SOURCE_DIR / f"fig_rev_1_tas_decay_{difficulty}.csv", out[out["difficulty"] == difficulty])

    trend = (
        out.groupby("L", as_index=False)
        .agg(mean_tas=("mean_tas", "mean"), sd=("mean_tas", "std"), n=("mean_tas", "size"))
        .sort_values("L")
    )
    trend["se"] = trend["sd"].fillna(0.0) / np.sqrt(trend["n"])
    trend["ci_low"] = trend["mean_tas"] - 1.96 * trend["se"]
    trend["ci_high"] = trend["mean_tas"] + 1.96 * trend["se"]
    write_csv(SOURCE_DIR / "fig_rev_1_tas_decay_trend.csv", trend[["L", "mean_tas", "se", "ci_low", "ci_high", "n"]])

    fit_data = out.dropna(subset=["L", "mean_tas"])
    x = np.log(fit_data["L"].to_numpy(dtype=float))
    y = fit_data["mean_tas"].to_numpy(dtype=float)
    a, b = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]
    y_hat = a + b * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    x_grid = np.linspace(float(fit_data["L"].min()), float(fit_data["L"].max()), 200)
    fit = pd.DataFrame({"L": x_grid, "fit_tas": a + b * np.log(x_grid)})
    write_csv(SOURCE_DIR / "fig_rev_1_tas_decay_fit.csv", fit)
    write_csv(
        SOURCE_DIR / "fig_rev_1_tas_decay_stats.csv",
        pd.DataFrame([{"a": fmt_num(a, 3), "b": fmt_num(b, 3), "r2": fmt_num(r2, 3), "n_bins": int(len(out))}]),
    )
    report["fig_rev_1"] = {"rows": int(len(out)), "fit_a": fmt_num(a, 3), "fit_b": fmt_num(b, 3), "r2": fmt_num(r2, 3)}


def export_violin(t1c: pd.DataFrame, lstar: pd.DataFrame, report: dict[str, Any]) -> str:
    joined = t1c.merge(lstar[["question_id", "l_star_A"]], on="question_id", how="left").copy()
    joined["l_star_group"] = np.select(
        [joined["L"] < joined["l_star_A"], joined["L"] == joined["l_star_A"], joined["L"] > joined["l_star_A"]],
        ["below", "at", "above"],
        default="unknown",
    )
    main = joined[["question_id", "L", "k_star", "k_star_ratio", "l_star_group"]].copy()
    write_csv(SOURCE_DIR / "fig_rev_2_kstar_ratio_by_lstar_distance.csv", main)

    group_order = [("below", 1), ("at", 2), ("above", 3)]
    rng = np.random.default_rng(42)
    strip_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for group, xpos in group_order:
        values_raw = joined.loc[joined["l_star_group"] == group, "k_star_ratio"].dropna().to_numpy(dtype=float)
        values = np.clip(values_raw, 0.25, 0.95)
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        for value_raw, value, dx in zip(values_raw, values, jitter):
            strip_rows.append(
                {
                    "l_star_group": group,
                    "group_x": xpos,
                    "x_jitter": xpos + float(dx),
                    "k_star_ratio": value_raw,
                    "k_star_ratio_clipped": value,
                    "clipped": int(value_raw > 0.95),
                }
            )
        if len(values) == 0:
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        low = max(float(values.min()), float(q1 - 1.5 * iqr))
        high = min(float(values.max()), float(q3 + 1.5 * iqr))
        box_rows.append(
            {
                "l_star_group": group,
                "group_x": xpos,
                "q1": q1,
                "median": median,
                "q3": q3,
                "whisker_low": low,
                "whisker_high": high,
                "n": int(len(values)),
                "median_label": fmt_num(float(median), 3),
            }
        )

        grid = np.linspace(0.25, 0.95, 120)
        if len(values) >= 2 and float(np.std(values)) > 1e-8:
            density = gaussian_kde(values)(grid)
        else:
            density = np.ones_like(grid)
        width = 0.34 * density / float(np.max(density))
        left = pd.DataFrame({"x": xpos - width, "y": grid})
        right = pd.DataFrame({"x": xpos + width[::-1], "y": grid[::-1]})
        outline = pd.concat([left, right], ignore_index=True)
        write_csv(SOURCE_DIR / f"fig_rev_2_violin_{group}.csv", outline)

    strip = pd.DataFrame(strip_rows)
    boxes = pd.DataFrame(box_rows)
    write_csv(SOURCE_DIR / "fig_rev_2_strip.csv", strip)
    write_csv(SOURCE_DIR / "fig_rev_2_box_stats.csv", boxes)
    global_median = float(joined["k_star_ratio"].median())
    n_eq1 = int(np.isclose(joined["k_star_ratio"], 1.0).sum())
    write_csv(
        SOURCE_DIR / "fig_rev_2_stats.csv",
        pd.DataFrame(
            [
                {
                    "global_median": fmt_num(global_median, 3),
                    "n_clipped_eq1": n_eq1,
                    "n_axis_clipped": int((joined["k_star_ratio"] > 0.95).sum()),
                }
            ]
        ),
    )
    report["fig_rev_2"] = {
        "rows": int(len(main)),
        "global_median": fmt_num(global_median, 3),
        "n_clipped_eq1": n_eq1,
    }
    return make_box_draw_commands(boxes)


def make_box_draw_commands(boxes: pd.DataFrame) -> str:
    lines: list[str] = []
    for row in boxes.itertuples(index=False):
        x = float(row.group_x)
        lines.extend(
            [
                rf"\draw[black, line width=0.35pt] (axis cs:{x:.3f},{row.whisker_low:.6f}) -- (axis cs:{x:.3f},{row.q1:.6f});",
                rf"\draw[black, line width=0.35pt] (axis cs:{x:.3f},{row.q3:.6f}) -- (axis cs:{x:.3f},{row.whisker_high:.6f});",
                rf"\draw[black, fill=white, fill opacity=0.85, line width=0.45pt] (axis cs:{x-0.09:.3f},{row.q1:.6f}) rectangle (axis cs:{x+0.09:.3f},{row.q3:.6f});",
                rf"\draw[accent-red, line width=0.9pt] (axis cs:{x-0.11:.3f},{row.median:.6f}) -- (axis cs:{x+0.11:.3f},{row.median:.6f});",
            ]
        )
    return "\n  ".join(lines)


def export_nldd_drop(qmeta: pd.DataFrame, report: dict[str, Any]) -> None:
    source = pd.read_csv(OLD_DEEP_DIVE / "post_horizon_nldd.csv")
    out = ensure_difficulty(source, qmeta)[["question_id", "difficulty", "L", "nldd_drop_ratio"]].copy()
    out["L"] = out["L"].astype(int)
    write_csv(SOURCE_DIR / "fig_rev_3_nldd_drop_ratio.csv", out)
    for difficulty in ("medium", "hard"):
        write_csv(SOURCE_DIR / f"fig_rev_3_nldd_drop_ratio_{difficulty}.csv", out[out["difficulty"] == difficulty])
    trend = (
        out.groupby("L", as_index=False)
        .agg(
            median=("nldd_drop_ratio", "median"),
            q1=("nldd_drop_ratio", lambda s: s.quantile(0.25)),
            q3=("nldd_drop_ratio", lambda s: s.quantile(0.75)),
            n=("nldd_drop_ratio", "size"),
        )
        .sort_values("L")
    )
    write_csv(SOURCE_DIR / "fig_rev_3_nldd_drop_ratio_trend.csv", trend)
    report["fig_rev_3"] = {"rows": int(len(out))}


def export_tas_slopes(qmeta: pd.DataFrame, report: dict[str, Any]) -> None:
    source = pd.read_csv(OLD_DEEP_DIVE / "post_horizon_tas_slope.csv")
    out = ensure_difficulty(source, qmeta)[
        ["question_id", "difficulty", "L", "pre_kstar_tas_slope", "post_kstar_tas_slope"]
    ].copy()
    out["L"] = out["L"].astype(int)
    write_csv(SOURCE_DIR / "fig_rev_4_tas_slope_pre_post.csv", out)
    for difficulty in ("medium", "hard"):
        write_csv(SOURCE_DIR / f"fig_rev_4_tas_slope_pre_post_{difficulty}.csv", out[out["difficulty"] == difficulty])
    valid = out.dropna(subset=["pre_kstar_tas_slope", "post_kstar_tas_slope"]).copy()
    post_weaker = (valid["post_kstar_tas_slope"].abs() < valid["pre_kstar_tas_slope"].abs()).mean() * 100.0
    write_csv(
        SOURCE_DIR / "fig_rev_4_tas_slope_pre_post_stats.csv",
        pd.DataFrame([{"n_bins": int(len(valid)), "post_weaker_pct": fmt_num(float(post_weaker), 1)}]),
    )
    report["fig_rev_4"] = {"rows": int(len(valid)), "post_weaker_pct": fmt_num(float(post_weaker), 1)}


def export_t1a(report: dict[str, Any]) -> None:
    overview = pd.read_csv(OVERALL_ANALYSIS / "t1a_overview.csv")
    stats_rows: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard"):
        group = overview[overview["difficulty"] == difficulty].copy()
        ok = group[group["bin_status"] == "ok"]
        if ok.empty:
            report.setdefault("fig_rev_5_skipped", []).append(difficulty)
            continue
        max_ok = int(ok["L"].max())
        visible = group[(group["L"] >= 3) & (group["L"] <= max_ok)].copy().sort_values("L")
        export = visible[["L", "accuracy", "accuracy_se", "mean_tas", "tas_se", "k_star", "bin_status"]].copy()
        write_csv(SOURCE_DIR / f"fig_rev_5_t1a_{difficulty}.csv", export)
        ok_visible = visible[visible["bin_status"] == "ok"].sort_values(["accuracy", "L"], ascending=[False, True])
        l_star = int(ok_visible.iloc[0]["L"])
        stats_rows.append(
            {
                "difficulty": difficulty,
                "l_star": l_star,
                "xmax": max_ok + 0.5,
                "right_ymax": max_ok + 0.5,
            }
        )
    write_csv(SOURCE_DIR / "fig_rev_5_t1a_stats.csv", pd.DataFrame(stats_rows))
    report["fig_rev_5"] = {"difficulty_panels": [row["difficulty"] for row in stats_rows]}


def standalone_open() -> str:
    return r"""\ifdefined\PeakCoTMainDocument
\else
\documentclass[border=3pt]{standalone}
\input{peakcot_figure_preamble.tex}
\begin{document}
\fi
"""


def standalone_close() -> str:
    return r"""\ifdefined\PeakCoTMainDocument
\else
\end{document}
\fi
"""


def write_preamble() -> None:
    write_text(
        OUT_ROOT / "peakcot_figure_preamble.tex",
        r"""% Shared preamble for peak-CoT pgfplots figures.
% For inclusion in a main document, load this once in the main preamble and define:
% \newcommand{\PeakCoTMainDocument}{}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{fillbetween,groupplots,statistics}
\usetikzlibrary{calc,positioning}

\definecolor{medium-teal}{HTML}{2B8C8C}
\definecolor{hard-orange}{HTML}{E07B39}
\definecolor{accent-red}{HTML}{C0392B}
\definecolor{fit-red}{HTML}{E74C3C}
\definecolor{band-blue}{HTML}{3498DB}
\definecolor{tas-green}{HTML}{2E8B57}
\definecolor{axis-gray}{HTML}{666666}

\pgfplotsset{
  peakcot base/.style={
    tick label style={font=\footnotesize},
    label style={font=\small},
    title style={font=\small},
    legend style={font=\footnotesize, draw=none, fill=none},
    axis line style={axis-gray},
    tick style={axis-gray},
    ymajorgrids=true,
    xmajorgrids=false,
    grid style={gray!20, line width=0.25pt},
    clip=false,
  }
}
""",
    )


def write_fig_new_1(default_questions: list[str]) -> None:
    qa, qb, qc, qd = default_questions
    write_text(
        OUT_ROOT / "fig_new_1_per_question_lcurves.tex",
        rf"""{standalone_open()}
% Edit these question ids to switch panels. Matching CSVs live in \DataDir.
\newcommand{{\DataDir}}{{source-0426}}
\newcommand{{\QuestionA}}{{\detokenize{{{qa}}}}}
\newcommand{{\QuestionB}}{{\detokenize{{{qb}}}}}
\newcommand{{\QuestionC}}{{\detokenize{{{qc}}}}}
\newcommand{{\QuestionD}}{{\detokenize{{{qd}}}}}

\newcommand{{\LcurvePanel}}[4]{{%
  \pgfplotstableread[col sep=comma]{{\DataDir/fig_new_1_lcurve_#1.csv}}\LcurveData
  \pgfplotstablegetelem{{0}}{{l_star_A}}\of{{\LcurveData}}\edef\ThisLStarA{{\pgfplotsretval}}
  \pgfplotstablegetelem{{0}}{{l_star_S}}\of{{\LcurveData}}\edef\ThisLStarS{{\pgfplotsretval}}
  \pgfplotstablegetelem{{0}}{{difficulty_score}}\of{{\LcurveData}}\edef\ThisDifficulty{{\pgfplotsretval}}
  \nextgroupplot[
    peakcot base,
    width=0.45\textwidth,
    height=0.285\textwidth,
    xmin=2.5,
    xmax=17.5,
    ymin=0,
    ymax=1.02,
    xtick distance=2,
    ytick={{0,0.25,0.5,0.75,1}},
    title={{Q\texttt{{#1}}, $d(q)=\ThisDifficulty$}},
    #4
  ]
    \addplot[draw=none, name path=accHi#2, forget plot]
      table[x=L, y expr={{\thisrow{{sufficient}}==1 ? \thisrow{{accuracy}}+\thisrow{{se}} : nan}}, col sep=comma] {{\DataDir/fig_new_1_lcurve_#1.csv}};
    \addplot[draw=none, name path=accLo#2, forget plot]
      table[x=L, y expr={{\thisrow{{sufficient}}==1 ? \thisrow{{accuracy}}-\thisrow{{se}} : nan}}, col sep=comma] {{\DataDir/fig_new_1_lcurve_#1.csv}};
    \addplot[band-blue, fill opacity=0.14, draw=none, forget plot] fill between[of=accHi#2 and accLo#2];
    \addplot[gray!60, dashed, thin, unbounded coords=jump]
      table[x=L, y=accuracy_smooth, col sep=comma] {{\DataDir/fig_new_1_lcurve_#1.csv}};
    \addplot[band-blue!75!black, thick, mark=*, mark size=1.4pt, unbounded coords=jump,
      restrict expr to domain={{\thisrow{{sufficient}}}}{{1:1}}]
      table[x=L, y=accuracy, col sep=comma] {{\DataDir/fig_new_1_lcurve_#1.csv}};
    \addplot[band-blue!75!black, only marks, mark=o, mark size=1.7pt, mark options={{solid, fill=white}},
      restrict expr to domain={{\thisrow{{sufficient}}}}{{0:0}}]
      table[x=L, y=accuracy, col sep=comma] {{\DataDir/fig_new_1_lcurve_#1.csv}};
    \ifdim\ThisLStarA pt>0pt
      \addplot[accent-red, dashed, line width=0.8pt] coordinates {{(\ThisLStarA,0) (\ThisLStarA,1)}};
      \node[font=\scriptsize, text=accent-red, anchor=south] at (axis cs:\ThisLStarA,0.94) {{$L^*$}};
      \ifnum\ThisLStarA=\ThisLStarS\relax\else
        \ifdim\ThisLStarS pt>0pt
          \addplot[hard-orange, dotted, line width=0.9pt] coordinates {{(\ThisLStarS,0) (\ThisLStarS,1)}};
        \fi
      \fi
    \fi
}}

\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 2, horizontal sep=0.075\textwidth, vertical sep=0.06\textwidth}},
]
\LcurvePanel{{\QuestionA}}{{A}}{{}}{{ylabel={{Accuracy}}}}
\LcurvePanel{{\QuestionB}}{{B}}{{}}{{yticklabels={{}}, ylabel={{}}}}
\LcurvePanel{{\QuestionC}}{{C}}{{}}{{xlabel={{Reasoning length $L$}}, ylabel={{Accuracy}}}}
\LcurvePanel{{\QuestionD}}{{D}}{{}}{{xlabel={{Reasoning length $L$}}, yticklabels={{}}, ylabel={{}}}}
\end{{groupplot}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_new_2() -> None:
    write_text(
        OUT_ROOT / "fig_new_2_lstar_kstar_pairing.tex",
        rf"""{standalone_open()}
\newcommand{{\DataDir}}{{source-0426}}
\pgfplotstableread[col sep=comma]{{\DataDir/fig_new_2_stats.csv}}\PairStats
\pgfplotstablegetelem{{0}}{{pearson_r}}\of{{\PairStats}}\edef\PearsonR{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{p_value}}\of{{\PairStats}}\edef\PearsonP{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{n_questions}}\of{{\PairStats}}\edef\PairN{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{axis_min}}\of{{\PairStats}}\edef\PairMin{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{axis_max}}\of{{\PairStats}}\edef\PairMax{{\pgfplotsretval}}

\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  name=main,
  width=0.68\textwidth,
  height=0.68\textwidth,
  xmin=\PairMin,
  xmax=\PairMax,
  ymin=\PairMin,
  ymax=\PairMax,
  axis equal image,
  xlabel={{$L^*$ (behavioral optimal length)}},
  ylabel={{$k^*(L^*)$ (mechanistic horizon at $L^*$)}},
  legend pos=south east,
]
  \addplot[black, dashed, domain=\PairMin:\PairMax, samples=2] {{x}};
  \addlegendentry{{$k^*=L^*$}}
  \addplot[fit-red, thin, domain=\PairMin:\PairMax, samples=2] {{0.64*x}};
  \addlegendentry{{$k^*/L=0.64$}}
  \addplot+[only marks, mark=*, mark size=2.0pt, draw=medium-teal!80!black, fill=medium-teal, opacity=0.62]
    table[x=l_star_A, y=k_star_at_lstar, col sep=comma] {{\DataDir/fig_new_2_pairing_medium.csv}};
  \addlegendentry{{Medium}}
  \addplot+[only marks, mark=triangle*, mark size=2.4pt, draw=hard-orange!80!black, fill=hard-orange, opacity=0.62]
    table[x=l_star_A, y=k_star_at_lstar, col sep=comma] {{\DataDir/fig_new_2_pairing_hard.csv}};
  \addlegendentry{{Hard}}
  \node[anchor=north west, align=left, font=\footnotesize, fill=white, fill opacity=0.82, text opacity=1, rounded corners=1pt]
    at (axis description cs:0.04,0.96) {{$r=\PearsonR$\\$p=\PearsonP$\\$n=\PairN$ questions}};
\end{{axis}}

\begin{{axis}}[
  at={{(main.north west)}},
  anchor=south west,
  width=0.68\textwidth,
  height=0.13\textwidth,
  xmin=\PairMin,
  xmax=\PairMax,
  ybar stacked,
  bar width=4pt,
  axis x line=none,
  axis y line*=left,
  ytick=\empty,
  xtick=\empty,
  enlargelimits=false,
  clip=false,
]
  \addplot+[draw=none, fill=medium-teal, fill opacity=0.65] table[x=bin, y=medium, col sep=comma] {{\DataDir/fig_new_2_hist_lstar.csv}};
  \addplot+[draw=none, fill=hard-orange, fill opacity=0.65] table[x=bin, y=hard, col sep=comma] {{\DataDir/fig_new_2_hist_lstar.csv}};
\end{{axis}}

\begin{{axis}}[
  at={{(main.south east)}},
  anchor=south west,
  width=0.13\textwidth,
  height=0.68\textwidth,
  ymin=\PairMin,
  ymax=\PairMax,
  xbar stacked,
  bar width=4pt,
  axis y line=none,
  axis x line*=bottom,
  xtick=\empty,
  ytick=\empty,
  enlargelimits=false,
  clip=false,
]
  \addplot+[draw=none, fill=medium-teal, fill opacity=0.65] table[y=bin, x=medium, col sep=comma] {{\DataDir/fig_new_2_hist_kstar.csv}};
  \addplot+[draw=none, fill=hard-orange, fill opacity=0.65] table[y=bin, x=hard, col sep=comma] {{\DataDir/fig_new_2_hist_kstar.csv}};
\end{{axis}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_new_3(q_a: str, l_a: int, q_b: str, l_b: int) -> None:
    write_text(
        OUT_ROOT / "fig_new_3_nldd_tas_step_profiles.tex",
        rf"""{standalone_open()}
% Edit question ids and lengths to switch representative panels.
\newcommand{{\DataDir}}{{source-0426}}
\newcommand{{\QuestionA}}{{\detokenize{{{q_a}}}}}
\newcommand{{\LengthA}}{{{l_a}}}
\newcommand{{\QuestionB}}{{\detokenize{{{q_b}}}}}
\newcommand{{\LengthB}}{{{l_b}}}

\begin{{tikzpicture}}
\pgfplotstableread[col sep=comma]{{\DataDir/fig_new_3_profile_\QuestionA_L\LengthA.csv}}\ProfileA
\pgfplotstablegetelem{{0}}{{k_star}}\of{{\ProfileA}}\edef\KStarA{{\pgfplotsretval}}
\begin{{axis}}[
  peakcot base,
  name=panelA,
  width=0.46\textwidth,
  height=0.34\textwidth,
  xmin=0.8,
  xmax=\LengthA+0.2,
  ymin=-100,
  ymax=750,
  xlabel={{Step position $k$}},
  ylabel={{Mean NLDD($k$)}},
  title={{Q\texttt{{\QuestionA}}, $L=\LengthA$}},
  unbounded coords=jump,
]
  \addplot[draw=none, name path=nlddHiA, forget plot] table[x=step, y expr=\thisrow{{mean_nldd}}+\thisrow{{nldd_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionA_L\LengthA.csv}};
  \addplot[draw=none, name path=nlddLoA, forget plot] table[x=step, y expr=\thisrow{{mean_nldd}}-\thisrow{{nldd_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionA_L\LengthA.csv}};
  \addplot[band-blue, fill opacity=0.14, draw=none, forget plot] fill between[of=nlddHiA and nlddLoA];
  \addplot[band-blue!75!black, thick, mark=*, mark size=1.4pt] table[x=step, y=mean_nldd, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionA_L\LengthA.csv}};
  \addplot[accent-red, dashed] coordinates {{(\KStarA,-100) (\KStarA,750)}};
  \node[font=\scriptsize, text=accent-red, anchor=south] at (axis cs:\KStarA,680) {{$k^*$}};
\end{{axis}}
\begin{{axis}}[
  at={{(panelA.south west)}},
  anchor=south west,
  width=0.46\textwidth,
  height=0.34\textwidth,
  xmin=0.8,
  xmax=\LengthA+0.2,
  ymin=0,
  ymax=1.05,
  axis y line*=right,
  axis x line=none,
  ylabel={{$\mathrm{{TAS}}_t$}},
  yticklabel style={{text=fit-red}},
  ylabel style={{text=fit-red}},
  unbounded coords=jump,
]
  \addplot[draw=none, name path=tasHiA, forget plot] table[x=step, y expr=\thisrow{{mean_tas_t}}+\thisrow{{tas_t_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionA_L\LengthA.csv}};
  \addplot[draw=none, name path=tasLoA, forget plot] table[x=step, y expr=\thisrow{{mean_tas_t}}-\thisrow{{tas_t_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionA_L\LengthA.csv}};
  \addplot[fit-red, fill opacity=0.10, draw=none, forget plot] fill between[of=tasHiA and tasLoA];
  \addplot[fit-red, dashed, thick, mark=square, mark options={{fill=white}}, mark size=1.5pt] table[x=step, y=mean_tas_t, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionA_L\LengthA.csv}};
\end{{axis}}

\pgfplotstableread[col sep=comma]{{\DataDir/fig_new_3_profile_\QuestionB_L\LengthB.csv}}\ProfileB
\pgfplotstablegetelem{{0}}{{k_star}}\of{{\ProfileB}}\edef\KStarB{{\pgfplotsretval}}
\begin{{axis}}[
  peakcot base,
  name=panelB,
  at={{(panelA.south east)}},
  anchor=south west,
  xshift=0.09\textwidth,
  width=0.46\textwidth,
  height=0.34\textwidth,
  xmin=0.8,
  xmax=\LengthB+0.2,
  ymin=-100,
  ymax=750,
  xlabel={{Step position $k$}},
  yticklabels={{}},
  title={{Q\texttt{{\QuestionB}}, $L=\LengthB$}},
  unbounded coords=jump,
]
  \addplot[draw=none, name path=nlddHiB, forget plot] table[x=step, y expr=\thisrow{{mean_nldd}}+\thisrow{{nldd_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionB_L\LengthB.csv}};
  \addplot[draw=none, name path=nlddLoB, forget plot] table[x=step, y expr=\thisrow{{mean_nldd}}-\thisrow{{nldd_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionB_L\LengthB.csv}};
  \addplot[band-blue, fill opacity=0.14, draw=none, forget plot] fill between[of=nlddHiB and nlddLoB];
  \addplot[band-blue!75!black, thick, mark=*, mark size=1.4pt] table[x=step, y=mean_nldd, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionB_L\LengthB.csv}};
  \addplot[accent-red, dashed] coordinates {{(\KStarB,-100) (\KStarB,750)}};
  \node[font=\scriptsize, text=accent-red, anchor=south] at (axis cs:\KStarB,680) {{$k^*$}};
\end{{axis}}
\begin{{axis}}[
  at={{(panelB.south west)}},
  anchor=south west,
  width=0.46\textwidth,
  height=0.34\textwidth,
  xmin=0.8,
  xmax=\LengthB+0.2,
  ymin=0,
  ymax=1.05,
  axis y line*=right,
  axis x line=none,
  ylabel={{$\mathrm{{TAS}}_t$}},
  yticklabel style={{text=fit-red}},
  ylabel style={{text=fit-red}},
  unbounded coords=jump,
]
  \addplot[draw=none, name path=tasHiB, forget plot] table[x=step, y expr=\thisrow{{mean_tas_t}}+\thisrow{{tas_t_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionB_L\LengthB.csv}};
  \addplot[draw=none, name path=tasLoB, forget plot] table[x=step, y expr=\thisrow{{mean_tas_t}}-\thisrow{{tas_t_se}}, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionB_L\LengthB.csv}};
  \addplot[fit-red, fill opacity=0.10, draw=none, forget plot] fill between[of=tasHiB and tasLoB];
  \addplot[fit-red, dashed, thick, mark=square, mark options={{fill=white}}, mark size=1.5pt] table[x=step, y=mean_tas_t, col sep=comma] {{\DataDir/fig_new_3_profile_\QuestionB_L\LengthB.csv}};
\end{{axis}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_new_4(diag_min: float, diag_max: float, hist_ymax: float) -> None:
    write_text(
        OUT_ROOT / "fig_new_4_kstar_eq_l_cases.tex",
        rf"""{standalone_open()}
\newcommand{{\DataDir}}{{source-0426}}
\pgfplotstableread[col sep=comma]{{\DataDir/fig_new_4_stats.csv}}\CaseStats
\pgfplotstablegetelem{{0}}{{n_cases}}\of{{\CaseStats}}\edef\CaseN{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{median_L_cases}}\of{{\CaseStats}}\edef\CaseMedianL{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{global_median_L}}\of{{\CaseStats}}\edef\GlobalMedianL{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{pct_above_diagonal}}\of{{\CaseStats}}\edef\PctAbove{{\pgfplotsretval}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep=0.10\textwidth}},
  width=0.43\textwidth,
  height=0.34\textwidth,
]
\nextgroupplot[
  peakcot base,
  ybar stacked,
  bar width=5pt,
  xlabel={{Trace length $L$}},
  ylabel={{Count of $(q,L)$ bins}},
  ymin=0,
  ymax={hist_ymax:.3f},
  legend style={{at={{(0.02,0.98)}}, anchor=north west}},
]
  \addplot+[draw=none, fill=medium-teal, fill opacity=0.75] table[x=bin, y=medium, col sep=comma] {{\DataDir/fig_new_4_L_hist.csv}};
  \addlegendentry{{Medium}}
  \addplot+[draw=none, fill=hard-orange, fill opacity=0.75] table[x=bin, y=hard, col sep=comma] {{\DataDir/fig_new_4_L_hist.csv}};
  \addlegendentry{{Hard}}
  \addplot[accent-red, dashed] coordinates {{(\GlobalMedianL,0) (\GlobalMedianL,{hist_ymax:.3f})}};
  \node[anchor=north east, align=right, font=\footnotesize, fill=white, fill opacity=0.82, text opacity=1]
    at (axis description cs:0.98,0.95) {{$n=\CaseN$ bins\\median $L=\CaseMedianL$}};

\nextgroupplot[
  peakcot base,
  xmin={diag_min:.3f},
  xmax={diag_max:.3f},
  ymin={diag_min:.3f},
  ymax={diag_max:.3f},
  xlabel={{Mean NLDD($L-1$)}},
  ylabel={{Mean NLDD($L$)}},
  legend pos=south east,
]
  \addplot[black, dashed, domain={diag_min:.3f}:{diag_max:.3f}, samples=2] {{x}};
  \addlegendentry{{$y=x$}}
  \addplot+[only marks, mark=*, mark size=1.7pt, draw=medium-teal!80!black, fill=medium-teal, opacity=0.65]
    table[x=mean_nldd_penultimate, y=mean_nldd_final, col sep=comma] {{\DataDir/fig_new_4_cases_medium.csv}};
  \addlegendentry{{Medium}}
  \addplot+[only marks, mark=triangle*, mark size=2.1pt, draw=hard-orange!80!black, fill=hard-orange, opacity=0.65]
    table[x=mean_nldd_penultimate, y=mean_nldd_final, col sep=comma] {{\DataDir/fig_new_4_cases_hard.csv}};
  \addlegendentry{{Hard}}
  \node[anchor=north west, font=\footnotesize, fill=white, fill opacity=0.82, text opacity=1]
    at (axis description cs:0.04,0.96) {{above diagonal = \PctAbove\%}};
\end{{groupplot}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_rev_1() -> None:
    write_text(
        OUT_ROOT / "fig_rev_1_tas_decay_vs_L.tex",
        rf"""{standalone_open()}
\newcommand{{\DataDir}}{{source-0426}}
\pgfplotstableread[col sep=comma]{{\DataDir/fig_rev_1_tas_decay_stats.csv}}\TasStats
\pgfplotstablegetelem{{0}}{{a}}\of{{\TasStats}}\edef\FitA{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{b}}\of{{\TasStats}}\edef\FitB{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{r2}}\of{{\TasStats}}\edef\FitRsq{{\pgfplotsretval}}
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  width=0.65\textwidth,
  height=0.49\textwidth,
  xlabel={{Trace length $L$}},
  ylabel={{Final TAS, $\mathrm{{TAS}}(L)$}},
  ymin=0,
  ymax=0.72,
  legend style={{at={{(0.02,0.98)}}, anchor=north west}},
]
  \addplot+[only marks, mark=*, mark size=1.15pt, draw=none, fill=medium-teal, opacity=0.40]
    table[x=L, y=mean_tas, col sep=comma] {{\DataDir/fig_rev_1_tas_decay_medium.csv}};
  \addlegendentry{{Medium $(q,L)$ bins}}
  \addplot+[only marks, mark=*, mark size=1.15pt, draw=none, fill=hard-orange, opacity=0.40]
    table[x=L, y=mean_tas, col sep=comma] {{\DataDir/fig_rev_1_tas_decay_hard.csv}};
  \addlegendentry{{Hard $(q,L)$ bins}}
  \addplot[draw=none, name path=tasCiHi, forget plot] table[x=L, y=ci_high, col sep=comma] {{\DataDir/fig_rev_1_tas_decay_trend.csv}};
  \addplot[draw=none, name path=tasCiLo, forget plot] table[x=L, y=ci_low, col sep=comma] {{\DataDir/fig_rev_1_tas_decay_trend.csv}};
  \addplot[band-blue, fill opacity=0.14, draw=none, forget plot] fill between[of=tasCiHi and tasCiLo];
  \addplot[band-blue!70!black, thick, mark=*, mark size=1.5pt]
    table[x=L, y=mean_tas, col sep=comma] {{\DataDir/fig_rev_1_tas_decay_trend.csv}};
  \addlegendentry{{$\mathbb{{E}}[\mathrm{{TAS}}\mid L]$}}
  \addplot[fit-red, thick] table[x=L, y=fit_tas, col sep=comma] {{\DataDir/fig_rev_1_tas_decay_fit.csv}};
  \addlegendentry{{$a+b\log L$ fit}}
  \node[anchor=south east, align=left, font=\footnotesize, fill=white, fill opacity=0.86, text opacity=1]
    at (axis description cs:0.98,0.05) {{$\mathrm{{TAS}}=\FitA+\FitB\log L$\\$R^2=\FitRsq$}};
\end{{axis}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_rev_2(box_draw_commands: str) -> None:
    write_text(
        OUT_ROOT / "fig_rev_2_kstar_ratio_violin_by_lstar_distance.tex",
        rf"""{standalone_open()}
\newcommand{{\DataDir}}{{source-0426}}
\pgfplotstableread[col sep=comma]{{\DataDir/fig_rev_2_stats.csv}}\ViolinStats
\pgfplotstablegetelem{{0}}{{global_median}}\of{{\ViolinStats}}\edef\GlobalMedianRatio{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{n_clipped_eq1}}\of{{\ViolinStats}}\edef\ClippedN{{\pgfplotsretval}}
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  width=0.60\textwidth,
  height=0.80\textwidth,
  xmin=0.45,
  xmax=3.55,
  ymin=0.25,
  ymax=0.95,
  xtick={{1,2,3}},
  xticklabels={{$L<L^*$,$L=L^*$,$L>L^*$}},
  ylabel={{Relative horizon $k^*/L$}},
]
  \addplot[draw=band-blue!70!black, fill=band-blue, fill opacity=0.15, line width=0.45pt]
    table[x=x, y=y, col sep=comma] {{\DataDir/fig_rev_2_violin_below.csv}} \closedcycle;
  \addplot[draw=band-blue!70!black, fill=band-blue, fill opacity=0.15, line width=0.45pt]
    table[x=x, y=y, col sep=comma] {{\DataDir/fig_rev_2_violin_at.csv}} \closedcycle;
  \addplot[draw=band-blue!70!black, fill=band-blue, fill opacity=0.15, line width=0.45pt]
    table[x=x, y=y, col sep=comma] {{\DataDir/fig_rev_2_violin_above.csv}} \closedcycle;
  \addplot+[only marks, mark=*, mark size=0.75pt, draw=none, fill=gray!45, opacity=0.45]
    table[x=x_jitter, y=k_star_ratio_clipped, col sep=comma] {{\DataDir/fig_rev_2_strip.csv}};
  \addplot[gray!65, dashed, line width=0.8pt] coordinates {{(0.48,\GlobalMedianRatio) (3.52,\GlobalMedianRatio)}};
  {box_draw_commands}
  \node[anchor=north east, align=right, font=\footnotesize, fill=white, fill opacity=0.86, text opacity=1]
    at (axis description cs:0.98,0.98) {{$\ClippedN$ points clipped at $k^*/L=1.0$\\global median = \GlobalMedianRatio}};
\end{{axis}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_rev_3() -> None:
    write_text(
        OUT_ROOT / "fig_rev_3_nldd_drop_ratio_vs_L.tex",
        rf"""{standalone_open()}
\newcommand{{\DataDir}}{{source-0426}}
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  width=0.65\textwidth,
  height=0.45\textwidth,
  xlabel={{Trace length $L$}},
  ylabel={{Post/pre NLDD ratio}},
  ymin=-0.10,
  ymax=1.05,
  legend style={{at={{(0.02,0.98)}}, anchor=north west}},
]
  \addplot[gray!60, dotted, line width=0.8pt] coordinates {{(0,1) (30,1)}};
  \addlegendentry{{No post/pre change}}
  \addplot[accent-red, dashed, line width=0.8pt] coordinates {{(0,0) (30,0)}};
  \addlegendentry{{Zero post-horizon NLDD}}
  \addplot+[only marks, mark=*, mark size=1.1pt, draw=none, fill=medium-teal, opacity=0.40]
    table[x=L, y=nldd_drop_ratio, col sep=comma] {{\DataDir/fig_rev_3_nldd_drop_ratio_medium.csv}};
  \addlegendentry{{Medium}}
  \addplot+[only marks, mark=*, mark size=1.1pt, draw=none, fill=hard-orange, opacity=0.40]
    table[x=L, y=nldd_drop_ratio, col sep=comma] {{\DataDir/fig_rev_3_nldd_drop_ratio_hard.csv}};
  \addlegendentry{{Hard}}
  \addplot[draw=none, name path=dropQ3, forget plot] table[x=L, y=q3, col sep=comma] {{\DataDir/fig_rev_3_nldd_drop_ratio_trend.csv}};
  \addplot[draw=none, name path=dropQ1, forget plot] table[x=L, y=q1, col sep=comma] {{\DataDir/fig_rev_3_nldd_drop_ratio_trend.csv}};
  \addplot[band-blue, fill opacity=0.14, draw=none, forget plot] fill between[of=dropQ3 and dropQ1];
  \addplot[band-blue!70!black, thick, mark=*, mark size=1.4pt]
    table[x=L, y=median, col sep=comma] {{\DataDir/fig_rev_3_nldd_drop_ratio_trend.csv}};
  \addlegendentry{{Median by $L$}}
\end{{axis}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_rev_4() -> None:
    write_text(
        OUT_ROOT / "fig_rev_4_tas_slope_pre_post_kstar.tex",
        rf"""{standalone_open()}
\newcommand{{\DataDir}}{{source-0426}}
\pgfplotstableread[col sep=comma]{{\DataDir/fig_rev_4_tas_slope_pre_post_stats.csv}}\SlopeStats
\pgfplotstablegetelem{{0}}{{n_bins}}\of{{\SlopeStats}}\edef\SlopeN{{\pgfplotsretval}}
\pgfplotstablegetelem{{0}}{{post_weaker_pct}}\of{{\SlopeStats}}\edef\PostWeakerPct{{\pgfplotsretval}}
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  width=0.55\textwidth,
  height=0.55\textwidth,
  xmin=-0.55,
  xmax=0.05,
  ymin=-0.55,
  ymax=0.05,
  axis equal image,
  xlabel={{Pre-$k^*$ TAS slope}},
  ylabel={{Post-$k^*$ TAS slope}},
  legend pos=south east,
]
  \addplot[draw=none, fill=tas-green, fill opacity=0.10] coordinates {{(-0.55,-0.55) (-0.55,0.05) (0.05,0.05)}} \closedcycle;
  \addplot[black, dashed, domain=-0.55:0.05, samples=2] {{x}};
  \addlegendentry{{Equal slope}}
  \addplot[accent-red, dotted] coordinates {{(0,-0.55) (0,0.05)}};
  \addplot[accent-red, dotted] coordinates {{(-0.55,0) (0.05,0)}};
  \addplot+[only marks, mark=*, mark size=1.25pt, draw=none, fill=medium-teal, opacity=0.50]
    table[x=pre_kstar_tas_slope, y=post_kstar_tas_slope, col sep=comma] {{\DataDir/fig_rev_4_tas_slope_pre_post_medium.csv}};
  \addlegendentry{{Medium}}
  \addplot+[only marks, mark=*, mark size=1.25pt, draw=none, fill=hard-orange, opacity=0.50]
    table[x=pre_kstar_tas_slope, y=post_kstar_tas_slope, col sep=comma] {{\DataDir/fig_rev_4_tas_slope_pre_post_hard.csv}};
  \addlegendentry{{Hard}}
  \node[anchor=north west, align=left, font=\footnotesize, fill=white, fill opacity=0.86, text opacity=1]
    at (axis description cs:0.04,0.96) {{$n=\SlopeN$\\post weaker: \PostWeakerPct\%}};
\end{{axis}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_fig_rev_5() -> None:
    stats = pd.read_csv(SOURCE_DIR / "fig_rev_5_t1a_stats.csv").set_index("difficulty")
    e = stats.loc["easy"]
    m = stats.loc["medium"]
    h = stats.loc["hard"]
    write_text(
        OUT_ROOT / "fig_rev_5_t1a_triple_metric.tex",
        rf"""{standalone_open()}
\newcommand{{\DataDir}}{{source-0426}}
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  name=easyL,
  width=0.29\textwidth,
  height=0.30\textwidth,
  xmin=2.5,
  xmax={float(e.xmax):.3f},
  ymin=0,
  ymax=1.05,
  xlabel={{Reasoning length $L$}},
  ylabel={{Accuracy / TAS}},
  title={{Easy}},
  legend columns=4,
  legend style={{at={{(1.78,-0.28)}}, anchor=north, column sep=6pt}},
]
  \addplot[draw=none, name path=accHiE, forget plot] table[x=L, y expr=\thisrow{{accuracy}}+\thisrow{{accuracy_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_easy.csv}};
  \addplot[draw=none, name path=accLoE, forget plot] table[x=L, y expr=\thisrow{{accuracy}}-\thisrow{{accuracy_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_easy.csv}};
  \addplot[band-blue, fill opacity=0.12, draw=none, forget plot] fill between[of=accHiE and accLoE];
  \addplot[band-blue!75!black, thick, mark=*, mark size=1.2pt] table[x=L, y=accuracy, col sep=comma] {{\DataDir/fig_rev_5_t1a_easy.csv}};
  \addlegendentry{{Accuracy}}
  \addplot[draw=none, name path=tasHiE, forget plot] table[x=L, y expr=\thisrow{{mean_tas}}+\thisrow{{tas_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_easy.csv}};
  \addplot[draw=none, name path=tasLoE, forget plot] table[x=L, y expr=\thisrow{{mean_tas}}-\thisrow{{tas_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_easy.csv}};
  \addplot[tas-green, fill opacity=0.12, draw=none, forget plot] fill between[of=tasHiE and tasLoE];
  \addplot[tas-green, thick, dashed, mark=triangle*, mark size=1.3pt] table[x=L, y=mean_tas, col sep=comma] {{\DataDir/fig_rev_5_t1a_easy.csv}};
  \addlegendentry{{TAS}}
  \addplot[accent-red, dashed] coordinates {{({int(e.l_star)},0) ({int(e.l_star)},1.05)}};
  \addlegendimage{{hard-orange, thick, mark=square*, mark size=1.2pt}}
  \addlegendentry{{$k^*(L)$}}
  \addlegendimage{{gray!65, dashed}}
  \addlegendentry{{$k^*=L$}}
\end{{axis}}
\begin{{axis}}[
  at={{(easyL.south west)}}, anchor=south west,
  width=0.29\textwidth,
  height=0.30\textwidth,
  xmin=2.5,
  xmax={float(e.xmax):.3f},
  ymin=0,
  ymax={float(e.right_ymax):.3f},
  axis y line*=right,
  axis x line=none,
  yticklabel=\empty,
]
  \addplot[gray!65, dashed, domain=0:{float(e.xmax):.3f}, samples=2] {{x}};
  \addplot[hard-orange, thick, mark=square*, mark size=1.2pt, unbounded coords=jump] table[x=L, y=k_star, col sep=comma] {{\DataDir/fig_rev_5_t1a_easy.csv}};
\end{{axis}}

\begin{{axis}}[
  peakcot base,
  name=mediumL,
  at={{(easyL.south east)}}, anchor=south west, xshift=0.055\textwidth,
  width=0.29\textwidth,
  height=0.30\textwidth,
  xmin=2.5,
  xmax={float(m.xmax):.3f},
  ymin=0,
  ymax=1.05,
  xlabel={{Reasoning length $L$}},
  yticklabels={{}},
  title={{Medium}},
]
  \addplot[draw=none, name path=accHiM, forget plot] table[x=L, y expr=\thisrow{{accuracy}}+\thisrow{{accuracy_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_medium.csv}};
  \addplot[draw=none, name path=accLoM, forget plot] table[x=L, y expr=\thisrow{{accuracy}}-\thisrow{{accuracy_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_medium.csv}};
  \addplot[band-blue, fill opacity=0.12, draw=none, forget plot] fill between[of=accHiM and accLoM];
  \addplot[band-blue!75!black, thick, mark=*, mark size=1.2pt] table[x=L, y=accuracy, col sep=comma] {{\DataDir/fig_rev_5_t1a_medium.csv}};
  \addplot[draw=none, name path=tasHiM, forget plot] table[x=L, y expr=\thisrow{{mean_tas}}+\thisrow{{tas_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_medium.csv}};
  \addplot[draw=none, name path=tasLoM, forget plot] table[x=L, y expr=\thisrow{{mean_tas}}-\thisrow{{tas_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_medium.csv}};
  \addplot[tas-green, fill opacity=0.12, draw=none, forget plot] fill between[of=tasHiM and tasLoM];
  \addplot[tas-green, thick, dashed, mark=triangle*, mark size=1.3pt] table[x=L, y=mean_tas, col sep=comma] {{\DataDir/fig_rev_5_t1a_medium.csv}};
  \addplot[accent-red, dashed] coordinates {{({int(m.l_star)},0) ({int(m.l_star)},1.05)}};
\end{{axis}}
\begin{{axis}}[
  at={{(mediumL.south west)}}, anchor=south west,
  width=0.29\textwidth,
  height=0.30\textwidth,
  xmin=2.5,
  xmax={float(m.xmax):.3f},
  ymin=0,
  ymax={float(m.right_ymax):.3f},
  axis y line*=right,
  axis x line=none,
  yticklabel=\empty,
]
  \addplot[gray!65, dashed, domain=0:{float(m.xmax):.3f}, samples=2] {{x}};
  \addplot[hard-orange, thick, mark=square*, mark size=1.2pt, unbounded coords=jump] table[x=L, y=k_star, col sep=comma] {{\DataDir/fig_rev_5_t1a_medium.csv}};
\end{{axis}}

\begin{{axis}}[
  peakcot base,
  name=hardL,
  at={{(mediumL.south east)}}, anchor=south west, xshift=0.055\textwidth,
  width=0.29\textwidth,
  height=0.30\textwidth,
  xmin=2.5,
  xmax={float(h.xmax):.3f},
  ymin=0,
  ymax=1.05,
  xlabel={{Reasoning length $L$}},
  yticklabels={{}},
  title={{Hard}},
]
  \addplot[draw=none, name path=accHiH, forget plot] table[x=L, y expr=\thisrow{{accuracy}}+\thisrow{{accuracy_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_hard.csv}};
  \addplot[draw=none, name path=accLoH, forget plot] table[x=L, y expr=\thisrow{{accuracy}}-\thisrow{{accuracy_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_hard.csv}};
  \addplot[band-blue, fill opacity=0.12, draw=none, forget plot] fill between[of=accHiH and accLoH];
  \addplot[band-blue!75!black, thick, mark=*, mark size=1.2pt] table[x=L, y=accuracy, col sep=comma] {{\DataDir/fig_rev_5_t1a_hard.csv}};
  \addplot[draw=none, name path=tasHiH, forget plot] table[x=L, y expr=\thisrow{{mean_tas}}+\thisrow{{tas_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_hard.csv}};
  \addplot[draw=none, name path=tasLoH, forget plot] table[x=L, y expr=\thisrow{{mean_tas}}-\thisrow{{tas_se}}, col sep=comma] {{\DataDir/fig_rev_5_t1a_hard.csv}};
  \addplot[tas-green, fill opacity=0.12, draw=none, forget plot] fill between[of=tasHiH and tasLoH];
  \addplot[tas-green, thick, dashed, mark=triangle*, mark size=1.3pt] table[x=L, y=mean_tas, col sep=comma] {{\DataDir/fig_rev_5_t1a_hard.csv}};
  \addplot[accent-red, dashed] coordinates {{({int(h.l_star)},0) ({int(h.l_star)},1.05)}};
\end{{axis}}
\begin{{axis}}[
  at={{(hardL.south west)}}, anchor=south west,
  width=0.29\textwidth,
  height=0.30\textwidth,
  xmin=2.5,
  xmax={float(h.xmax):.3f},
  ymin=0,
  ymax={float(h.right_ymax):.3f},
  axis y line*=right,
  axis x line=none,
  ylabel={{$k^*(L)$}},
]
  \addplot[gray!65, dashed, domain=0:{float(h.xmax):.3f}, samples=2] {{x}};
  \addplot[hard-orange, thick, mark=square*, mark size=1.2pt, unbounded coords=jump] table[x=L, y=k_star, col sep=comma] {{\DataDir/fig_rev_5_t1a_hard.csv}};
\end{{axis}}
\end{{tikzpicture}}
{standalone_close()}
""",
    )


def write_all_tex(defaults: Defaults, diag_min: float, diag_max: float, hist_ymax: float, box_draw_commands: str) -> None:
    write_preamble()
    write_fig_new_1(defaults.lcurve_questions)
    write_fig_new_2()
    write_fig_new_3(defaults.profile_a_question, defaults.profile_a_length, defaults.profile_b_question, defaults.profile_b_length)
    write_fig_new_4(diag_min, diag_max, hist_ymax)
    write_fig_rev_1()
    write_fig_rev_2(box_draw_commands)
    write_fig_rev_3()
    write_fig_rev_4()
    write_fig_rev_5()


def validate_outputs(defaults: Defaults, report: dict[str, Any]) -> None:
    expected_tex = [
        "peakcot_figure_preamble.tex",
        "fig_new_1_per_question_lcurves.tex",
        "fig_new_2_lstar_kstar_pairing.tex",
        "fig_new_3_nldd_tas_step_profiles.tex",
        "fig_new_4_kstar_eq_l_cases.tex",
        "fig_rev_1_tas_decay_vs_L.tex",
        "fig_rev_2_kstar_ratio_violin_by_lstar_distance.tex",
        "fig_rev_3_nldd_drop_ratio_vs_L.tex",
        "fig_rev_4_tas_slope_pre_post_kstar.tex",
        "fig_rev_5_t1a_triple_metric.tex",
    ]
    missing_tex = [name for name in expected_tex if not (OUT_ROOT / name).exists()]
    fixed_csv = [
        "fig_new_2_pairing.csv",
        "fig_new_2_pairing_medium.csv",
        "fig_new_2_pairing_hard.csv",
        "fig_new_2_hist_lstar.csv",
        "fig_new_2_hist_kstar.csv",
        "fig_new_2_stats.csv",
        "fig_new_4_kstar_eq_l_cases.csv",
        "fig_new_4_cases_medium.csv",
        "fig_new_4_cases_hard.csv",
        "fig_new_4_L_hist.csv",
        "fig_new_4_stats.csv",
        "fig_rev_1_tas_decay.csv",
        "fig_rev_1_tas_decay_medium.csv",
        "fig_rev_1_tas_decay_hard.csv",
        "fig_rev_1_tas_decay_trend.csv",
        "fig_rev_1_tas_decay_fit.csv",
        "fig_rev_1_tas_decay_stats.csv",
        "fig_rev_2_kstar_ratio_by_lstar_distance.csv",
        "fig_rev_2_strip.csv",
        "fig_rev_2_box_stats.csv",
        "fig_rev_2_violin_below.csv",
        "fig_rev_2_violin_at.csv",
        "fig_rev_2_violin_above.csv",
        "fig_rev_2_stats.csv",
        "fig_rev_3_nldd_drop_ratio.csv",
        "fig_rev_3_nldd_drop_ratio_medium.csv",
        "fig_rev_3_nldd_drop_ratio_hard.csv",
        "fig_rev_3_nldd_drop_ratio_trend.csv",
        "fig_rev_4_tas_slope_pre_post.csv",
        "fig_rev_4_tas_slope_pre_post_medium.csv",
        "fig_rev_4_tas_slope_pre_post_hard.csv",
        "fig_rev_4_tas_slope_pre_post_stats.csv",
        "fig_rev_5_t1a_easy.csv",
        "fig_rev_5_t1a_medium.csv",
        "fig_rev_5_t1a_hard.csv",
        "fig_rev_5_t1a_stats.csv",
        f"fig_new_3_profile_{defaults.profile_a_question}_L{defaults.profile_a_length}.csv",
        f"fig_new_3_profile_{defaults.profile_b_question}_L{defaults.profile_b_length}.csv",
    ]
    fixed_csv.extend([f"fig_new_1_lcurve_{qid}.csv" for qid in defaults.lcurve_questions])
    missing_csv = [name for name in fixed_csv if not (SOURCE_DIR / name).exists()]

    expected_headers = {
        f"fig_new_1_lcurve_{defaults.lcurve_questions[0]}.csv": [
            "L",
            "accuracy",
            "se",
            "n_traces",
            "sufficient",
            "l_star_A",
            "l_star_S",
            "accuracy_smooth",
            "difficulty",
            "difficulty_score",
        ],
        f"fig_new_3_profile_{defaults.profile_a_question}_L{defaults.profile_a_length}.csv": [
            "step",
            "mean_nldd",
            "nldd_se",
            "mean_tas_t",
            "tas_t_se",
            "k_star",
        ],
        "fig_new_4_kstar_eq_l_cases.csv": [
            "question_id",
            "difficulty",
            "L",
            "mean_nldd_final",
            "mean_nldd_penultimate",
        ],
        "fig_rev_1_tas_decay.csv": ["question_id", "difficulty", "L", "mean_tas", "tas_se"],
        "fig_rev_2_kstar_ratio_by_lstar_distance.csv": ["question_id", "L", "k_star", "k_star_ratio", "l_star_group"],
        "fig_rev_3_nldd_drop_ratio.csv": ["question_id", "difficulty", "L", "nldd_drop_ratio"],
        "fig_rev_4_tas_slope_pre_post.csv": [
            "question_id",
            "difficulty",
            "L",
            "pre_kstar_tas_slope",
            "post_kstar_tas_slope",
        ],
        "fig_rev_5_t1a_easy.csv": ["L", "accuracy", "accuracy_se", "mean_tas", "tas_se", "k_star", "bin_status"],
    }
    header_mismatches: dict[str, Any] = {}
    for name, header in expected_headers.items():
        path = SOURCE_DIR / name
        if not path.exists():
            continue
        actual = pd.read_csv(path, nrows=0).columns.tolist()
        if actual != header:
            header_mismatches[name] = {"expected": header, "actual": actual}

    tex_bad_paths: dict[str, list[str]] = {}
    source_ref_pattern = re.compile(r"(?:results/|results\\|[A-Z]:\\[^\\\n]+\\)")
    for tex_name in expected_tex:
        path = OUT_ROOT / tex_name
        if not path.exists() or tex_name == "peakcot_figure_preamble.tex":
            continue
        text = path.read_text(encoding="utf-8")
        bad = source_ref_pattern.findall(text)
        if bad:
            tex_bad_paths[tex_name] = bad

    png_files = [str(path.relative_to(OUT_ROOT)) for path in OUT_ROOT.rglob("*.png")]
    report["validation"] = {
        "missing_tex": missing_tex,
        "missing_required_csv": missing_csv,
        "header_mismatches": header_mismatches,
        "tex_local_path_hits": tex_bad_paths,
        "png_files": png_files,
        "status": "ok" if not (missing_tex or missing_csv or header_mismatches or tex_bad_paths or png_files) else "check_report",
    }


def write_reports(report: dict[str, Any], defaults: Defaults) -> None:
    report["output_root"] = str(OUT_ROOT.relative_to(ROOT))
    report["source_dir"] = str(SOURCE_DIR.relative_to(ROOT))
    report["defaults"] = {
        "fig_new_1_questions": defaults.lcurve_questions,
        "fig_new_3_panel_A": {"question_id": defaults.profile_a_question, "L": defaults.profile_a_length},
        "fig_new_3_panel_B": {"question_id": defaults.profile_b_question, "L": defaults.profile_b_length},
    }
    csv_count = sum(1 for _ in SOURCE_DIR.glob("*.csv"))
    tex_count = sum(1 for _ in OUT_ROOT.glob("*.tex"))
    report["file_counts"] = {"csv": csv_count, "tex": tex_count}
    write_text(OUT_ROOT / "export_manifest.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Overleaf PGFPlots Export Report",
        "",
        f"- Output root: `{report['output_root']}`",
        f"- CSV directory: `{report['source_dir']}`",
        f"- TeX files: {tex_count}",
        f"- CSV files: {csv_count}",
        "- No PNG/local figure rendering was generated.",
        "- LaTeX was not compiled locally.",
        "",
        "## Defaults",
        "",
        f"- Fig-NEW-1 questions: {', '.join(defaults.lcurve_questions)}",
        f"- Fig-NEW-3 panel A: {defaults.profile_a_question}, L={defaults.profile_a_length}",
        f"- Fig-NEW-3 panel B: {defaults.profile_b_question}, L={defaults.profile_b_length}",
        "",
        "## Export Counts",
        "",
        f"- Fig-NEW-1 L-curve CSVs: {report.get('fig_new_1', {}).get('exported_lcurve_csv')}",
        f"- Fig-NEW-3 step-profile CSVs: {report.get('fig_new_3', {}).get('exported_profile_csv')}",
        f"- Fig-NEW-2 paired questions: {report.get('fig_new_2', {}).get('paired_questions')}",
        f"- Fig-NEW-4 k*/L=1 cases: {report.get('fig_new_4', {}).get('kstar_eq_l_cases')}",
        "",
        "## Validation",
        "",
        f"- Status: `{report['validation']['status']}`",
        f"- Missing TeX: {report['validation']['missing_tex']}",
        f"- Missing required CSV: {report['validation']['missing_required_csv']}",
        f"- Header mismatches: {report['validation']['header_mismatches']}",
        f"- TeX local path hits: {report['validation']['tex_local_path_hits']}",
        f"- PNG files: {report['validation']['png_files']}",
    ]
    missing_pair = report.get("fig_new_2", {}).get("missing_lstar_kstar_questions", [])
    lstar_fallbacks = report.get("fig_new_1", {}).get("lstar_json_fallbacks", [])
    if missing_pair:
        lines.extend(["", "## Skipped / Missing", "", f"- Questions missing k*(L*) pairing: {len(missing_pair)}"])
    elif lstar_fallbacks:
        lines.extend(["", "## Skipped / Missing", ""])
    if lstar_fallbacks:
        lines.append(
            "- L-curve questions using per-question `l_star.json` fallback with missing L* encoded as `-1`: "
            + ", ".join(lstar_fallbacks)
        )
    skipped_profiles = report.get("fig_new_3", {}).get("missing_profiles", [])
    if skipped_profiles:
        lines.append(f"- Missing step profiles: {len(skipped_profiles)}")
    skipped_cases = report.get("fig_new_4", {}).get("skipped_cases_missing_nldd", [])
    if skipped_cases:
        lines.append(f"- k*/L=1 cases missing final/penultimate NLDD: {len(skipped_cases)}")
    write_text(OUT_ROOT / "export_report.md", "\n".join(lines) + "\n")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    qmeta = read_jsonl(PQ_ROOT / "question_metadata.jsonl")
    qmeta = qmeta[["question_id", "difficulty", "difficulty_score"]].copy()
    qmeta["difficulty"] = qmeta["difficulty"].fillna(qmeta["difficulty_score"].apply(label_difficulty))

    bin_status = pd.read_csv(PQ_ANALYSIS / "bin_status.csv")
    t1b = pd.read_csv(PQ_ANALYSIS / "t1b_step_surface.csv")
    t1c = pd.read_csv(PQ_ANALYSIS / "t1c_kstar_ratio.csv")
    lstar = pd.read_csv(PQ_ANALYSIS / "t2b_lstar_difficulty.csv")

    _, lcurve_defaults = export_lcurves(qmeta, bin_status, lstar, report)
    export_pairing(t1c, lstar, qmeta, report)
    profile_a_q, profile_a_l, profile_b_q, profile_b_l = export_profiles(t1b, t1c, qmeta, report)
    diag_min, diag_max, hist_ymax = export_kstar_eq_l_cases(t1b, t1c, qmeta, report)
    export_tas_decay(qmeta, report)
    box_draw_commands = export_violin(t1c, lstar, report)
    export_nldd_drop(qmeta, report)
    export_tas_slopes(qmeta, report)
    export_t1a(report)

    defaults = Defaults(
        lcurve_questions=lcurve_defaults,
        profile_a_question=profile_a_q,
        profile_a_length=profile_a_l,
        profile_b_question=profile_b_q,
        profile_b_length=profile_b_l,
    )
    write_all_tex(defaults, diag_min, diag_max, hist_ymax, box_draw_commands)
    validate_outputs(defaults, report)
    write_reports(report, defaults)
    print(f"Wrote Overleaf PGFPlots export to {OUT_ROOT}")


if __name__ == "__main__":
    main()
