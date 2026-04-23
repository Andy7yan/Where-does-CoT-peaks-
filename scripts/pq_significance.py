"""Per-question k*/L vs accuracy significance analysis and source-data export."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning

try:
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover - optional dependency
    smf = None


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def _to_int(value: Any) -> int | None:
    if value in (None, "", "nan", "None", "null"):
        return None
    return int(value)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "nan", "None", "null"):
        return None
    return float(value)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_difficulty(score: float | None) -> str:
    # The repo's canonical semantics are difficulty_score = 1 - acc_pq, so larger is harder.
    if score is None or math.isnan(score):
        return "medium"
    return "hard" if score > 0.5 else "medium"


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    if values.size == 1:
        val = float(values[0])
        return (val, val)
    draws = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True)
    medians = np.median(draws, axis=1)
    return (float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)))


def safe_wilcoxon_greater(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    try:
        result = stats.wilcoxon(values, alternative="greater", zero_method="wilcox", method="auto")
        return (float(result.statistic), float(result.pvalue))
    except ValueError:
        return (math.nan, math.nan)


def safe_ttest_greater(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return (math.nan, math.nan)
    try:
        result = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
        return (float(result.statistic), float(result.pvalue))
    except TypeError:
        result = stats.ttest_1samp(values, popmean=0.0)
        statistic = float(result.statistic)
        p_two_sided = float(result.pvalue)
        if math.isnan(statistic) or math.isnan(p_two_sided):
            return (math.nan, math.nan)
        if statistic > 0:
            return (statistic, p_two_sided / 2.0)
        return (statistic, 1.0 - (p_two_sided / 2.0))


def pooled_ols_stats(frame: pd.DataFrame) -> tuple[float, float, float]:
    if len(frame) < 2:
        return (math.nan, math.nan, math.nan)
    x = frame["accuracy"].to_numpy(dtype=float)
    y = frame["k_star_ratio"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = (slope * x) + intercept
    residual_ss = float(np.sum((y - fitted) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    r_squared = math.nan if total_ss == 0.0 else 1.0 - (residual_ss / total_ss)
    return (float(slope), float(intercept), float(r_squared))


def summarize_value_groups(frame: pd.DataFrame, value_col: str, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in ("all", "medium", "hard"):
        if group == "all":
            subset = frame
        else:
            subset = frame[frame["difficulty"] == group]
        values = subset[value_col].dropna().to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_median_ci(values, rng)
        rows.append(
            {
                "group": group,
                "n": int(values.size),
                f"mean_{value_col}": float(np.mean(values)) if values.size else math.nan,
                f"median_{value_col}": float(np.median(values)) if values.size else math.nan,
                "pct_positive": float(np.mean(values > 0.0) * 100.0) if values.size else math.nan,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
    return pd.DataFrame(rows)


def build_scatter_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in ("all", "medium", "hard"):
        subset = frame if group == "all" else frame[frame["difficulty"] == group]
        if len(subset) >= 2:
            spearman = stats.spearmanr(subset["accuracy"], subset["k_star_ratio"], nan_policy="omit")
            pooled_rho = float(spearman.statistic)
            pooled_p = float(spearman.pvalue)
            slope, intercept, r_squared = pooled_ols_stats(subset)
        else:
            pooled_rho = math.nan
            pooled_p = math.nan
            slope = math.nan
            intercept = math.nan
            r_squared = math.nan
        rows.append(
            {
                "group": group,
                "n_obs": int(len(subset)),
                "n_questions": int(subset["question_id"].nunique()),
                "pooled_spearman_rho": pooled_rho,
                "pooled_spearman_p": pooled_p,
                "ols_slope": slope,
                "ols_intercept": intercept,
                "ols_r_squared": r_squared,
            }
        )
    return pd.DataFrame(rows)


def load_lstar_metadata(pq_run_dir: Path) -> pd.DataFrame:
    lstar_path = pq_run_dir / "pq_analysis" / "t2b_lstar_difficulty.csv"
    if not lstar_path.exists():
        raise FileNotFoundError(f"Missing required file: {lstar_path}")
    lstar_df = _load_csv(lstar_path)
    lstar_df["question_id"] = lstar_df["question_id"].astype(str)
    lstar_df["difficulty_score"] = pd.to_numeric(lstar_df["difficulty_score"], errors="coerce")
    for col in ("l_star_A", "l_star_S"):
        lstar_df[col] = pd.to_numeric(lstar_df[col], errors="coerce").astype("Int64")

    difficulty_by_question: dict[str, str] = {}
    per_question_root = pq_run_dir / "per_question"
    if per_question_root.exists():
        for question_dir in per_question_root.iterdir():
            if not question_dir.is_dir():
                continue
            payload_path = question_dir / "l_star.json"
            if not payload_path.exists():
                continue
            payload = _read_json(payload_path)
            difficulty = str(payload.get("difficulty", "")).strip()
            if difficulty:
                difficulty_by_question[str(payload.get("question_id", question_dir.name))] = difficulty

    lstar_df["difficulty"] = lstar_df["question_id"].map(difficulty_by_question)
    inferred = lstar_df["difficulty"].isna() | (lstar_df["difficulty"].astype(str).str.len() == 0)
    lstar_df.loc[inferred, "difficulty"] = lstar_df.loc[inferred, "difficulty_score"].map(infer_difficulty)
    return lstar_df


def load_accuracy_from_per_question(pq_run_dir: Path) -> pd.DataFrame | None:
    per_question_root = pq_run_dir / "per_question"
    if not per_question_root.exists():
        return None
    rows: list[dict[str, Any]] = []
    for question_dir in sorted(path for path in per_question_root.iterdir() if path.is_dir()):
        lcurve_path = question_dir / "l_curve.csv"
        if not lcurve_path.exists():
            continue
        frame = _load_csv(lcurve_path)
        if frame.empty:
            continue
        frame["question_id"] = question_dir.name
        frame["L"] = pd.to_numeric(frame["L"], errors="coerce").astype("Int64")
        frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
        frame["n_traces"] = pd.to_numeric(frame.get("n"), errors="coerce").astype("Int64")
        rows.extend(frame[["question_id", "L", "accuracy", "n_traces"]].to_dict(orient="records"))
    if not rows:
        return None
    return pd.DataFrame(rows)


def reconstruct_accuracy_from_traces(traces_path: Path) -> pd.DataFrame:
    if not traces_path.exists():
        raise FileNotFoundError(f"Missing traces file: {traces_path}")
    records: list[dict[str, Any]] = []
    with traces_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            length = _to_int(payload.get("actual_num_steps"))
            question_id = str(payload.get("question_id", ""))
            if not question_id or length is None:
                continue
            records.append(
                {
                    "question_id": question_id,
                    "L": length,
                    "is_correct": bool(payload.get("is_correct", False)),
                }
            )
    if not records:
        raise ValueError(f"No usable trace records found in {traces_path}")
    frame = pd.DataFrame(records)
    grouped = (
        frame.groupby(["question_id", "L"], as_index=False)
        .agg(accuracy=("is_correct", "mean"), n_traces=("is_correct", "size"))
        .sort_values(["question_id", "L"])
    )
    return grouped


def load_accuracy_table(pq_run_dir: Path, traces_path: Path | None) -> tuple[pd.DataFrame, str]:
    pq_analysis_dir = pq_run_dir / "pq_analysis"
    candidate_paths = [
        pq_analysis_dir / "t1a_overview.csv",
        pq_analysis_dir / "accuracy_by_length.csv",
    ]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        frame = _load_csv(candidate)
        if {"question_id", "L", "accuracy"} <= set(frame.columns):
            frame = frame.copy()
            if "bin_status" in frame.columns:
                frame = frame[frame["bin_status"].astype(str) == "ok"]
            n_column = "n_traces" if "n_traces" in frame.columns else "n" if "n" in frame.columns else None
            frame["question_id"] = frame["question_id"].astype(str)
            frame["L"] = pd.to_numeric(frame["L"], errors="coerce").astype("Int64")
            frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
            frame["n_traces"] = pd.to_numeric(frame[n_column], errors="coerce").astype("Int64") if n_column else pd.Series(pd.NA, index=frame.index, dtype="Int64")
            return (frame[["question_id", "L", "accuracy", "n_traces"]].dropna(subset=["question_id", "L", "accuracy"]), str(candidate))

    per_question_frame = load_accuracy_from_per_question(pq_run_dir)
    if per_question_frame is not None:
        return (per_question_frame, str(pq_run_dir / "per_question" / "*" / "l_curve.csv"))

    effective_traces_path = traces_path or (pq_run_dir / "traces.jsonl")
    return (reconstruct_accuracy_from_traces(effective_traces_path), str(effective_traces_path))


def load_kstar_table(pq_run_dir: Path) -> pd.DataFrame:
    kstar_path = pq_run_dir / "pq_analysis" / "t1c_kstar_ratio.csv"
    if not kstar_path.exists():
        raise FileNotFoundError(f"Missing required file: {kstar_path}")
    frame = _load_csv(kstar_path)
    frame["question_id"] = frame["question_id"].astype(str)
    frame["L"] = pd.to_numeric(frame["L"], errors="coerce").astype("Int64")
    for col in ("difficulty_score", "k_star_ratio", "n_clean"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "k_star" in frame.columns:
        frame["k_star"] = pd.to_numeric(frame["k_star"], errors="coerce").astype("Int64")
    return frame


@dataclass
class Analysis1Result:
    detail_df: pd.DataFrame
    plot_df: pd.DataFrame
    stats_df: pd.DataFrame
    eligible_questions: int
    analysed_questions: int
    insufficient_questions: int
    undefined_questions: int
    wilcoxon_stat: float
    wilcoxon_p: float
    ttest_stat: float
    ttest_p: float


def run_analysis1(scatter_df: pd.DataFrame, min_points: int, rng: np.random.Generator) -> Analysis1Result:
    detail_rows: list[dict[str, Any]] = []
    insufficient_questions = 0
    undefined_questions = 0

    for question_id, group in scatter_df.groupby("question_id", sort=True):
        group = group.sort_values("L")
        difficulty = str(group["difficulty"].iloc[0])
        n_points = int(len(group))
        if n_points < min_points:
            insufficient_questions += 1
            detail_rows.append(
                {
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "n_points": n_points,
                    "rho": math.nan,
                    "p_value": math.nan,
                    "analysis1_status": "too_few_points",
                }
            )
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            spearman = stats.spearmanr(group["accuracy"], group["k_star_ratio"], nan_policy="omit")
        rho = float(spearman.statistic)
        p_value = float(spearman.pvalue)
        if math.isnan(rho):
            undefined_questions += 1
            detail_rows.append(
                {
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "n_points": n_points,
                    "rho": math.nan,
                    "p_value": math.nan,
                    "analysis1_status": "undefined_spearman",
                }
            )
            continue

        detail_rows.append(
            {
                "question_id": question_id,
                "difficulty": difficulty,
                "n_points": n_points,
                "rho": rho,
                "p_value": p_value,
                "analysis1_status": "ok",
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    plot_df = detail_df[detail_df["analysis1_status"] == "ok"][["question_id", "difficulty", "rho", "p_value", "n_points"]].copy()
    stats_df = summarize_value_groups(plot_df, "rho", rng).rename(
        columns={"mean_rho": "mean_rho", "median_rho": "median_rho"}
    )
    rho_values = plot_df["rho"].dropna().to_numpy(dtype=float)
    wilcoxon_stat, wilcoxon_p = safe_wilcoxon_greater(rho_values)
    ttest_stat, ttest_p = safe_ttest_greater(rho_values)
    return Analysis1Result(
        detail_df=detail_df,
        plot_df=plot_df,
        stats_df=stats_df,
        eligible_questions=int((detail_df["n_points"] >= min_points).sum()),
        analysed_questions=int(len(plot_df)),
        insufficient_questions=insufficient_questions,
        undefined_questions=undefined_questions,
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_p=wilcoxon_p,
        ttest_stat=ttest_stat,
        ttest_p=ttest_p,
    )


@dataclass
class Analysis2Result:
    detail_df: pd.DataFrame
    plot_df: pd.DataFrame
    stats_df: pd.DataFrame
    missing_lstar_questions: int
    no_pre_questions: int
    no_post_questions: int
    wilcoxon_stat: float
    wilcoxon_p: float


def run_analysis2(kstar_df: pd.DataFrame, lstar_df: pd.DataFrame, rng: np.random.Generator) -> Analysis2Result:
    merged = kstar_df.merge(
        lstar_df[["question_id", "difficulty", "difficulty_score", "l_star_A"]],
        on="question_id",
        how="left",
        suffixes=("", "_meta"),
    )
    merged = merged[merged["k_star_ratio"].notna() & merged["L"].notna()]

    detail_rows: list[dict[str, Any]] = []
    missing_lstar_questions = 0
    no_pre_questions = 0
    no_post_questions = 0

    for question_id, group in merged.groupby("question_id", sort=True):
        group = group.sort_values("L")
        difficulty = str(group["difficulty"].iloc[0])
        l_star = group["l_star_A"].iloc[0]
        if pd.isna(l_star):
            missing_lstar_questions += 1
            detail_rows.append(
                {
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "l_star_A": pd.NA,
                    "mean_kstar_ratio_pre": math.nan,
                    "mean_kstar_ratio_post": math.nan,
                    "delta": math.nan,
                    "n_pre": 0,
                    "n_post": 0,
                    "analysis2_status": "missing_l_star",
                }
            )
            continue

        pre = group[group["L"] <= l_star]
        post = group[group["L"] > l_star]
        if pre.empty:
            no_pre_questions += 1
            detail_rows.append(
                {
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "l_star_A": int(l_star),
                    "mean_kstar_ratio_pre": math.nan,
                    "mean_kstar_ratio_post": math.nan,
                    "delta": math.nan,
                    "n_pre": 0,
                    "n_post": int(len(post)),
                    "analysis2_status": "no_pre",
                }
            )
            continue
        if post.empty:
            no_post_questions += 1
            detail_rows.append(
                {
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "l_star_A": int(l_star),
                    "mean_kstar_ratio_pre": float(pre["k_star_ratio"].mean()),
                    "mean_kstar_ratio_post": math.nan,
                    "delta": math.nan,
                    "n_pre": int(len(pre)),
                    "n_post": 0,
                    "analysis2_status": "no_post",
                }
            )
            continue

        pre_mean = float(pre["k_star_ratio"].mean())
        post_mean = float(post["k_star_ratio"].mean())
        detail_rows.append(
            {
                "question_id": question_id,
                "difficulty": difficulty,
                "l_star_A": int(l_star),
                "mean_kstar_ratio_pre": pre_mean,
                "mean_kstar_ratio_post": post_mean,
                "delta": pre_mean - post_mean,
                "n_pre": int(len(pre)),
                "n_post": int(len(post)),
                "analysis2_status": "ok",
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    plot_df = detail_df[detail_df["analysis2_status"] == "ok"][
        [
            "question_id",
            "difficulty",
            "l_star_A",
            "mean_kstar_ratio_pre",
            "mean_kstar_ratio_post",
            "delta",
            "n_pre",
            "n_post",
        ]
    ].copy()
    stats_df = summarize_value_groups(plot_df, "delta", rng).rename(
        columns={"mean_delta": "mean_delta", "median_delta": "median_delta"}
    )
    delta_values = plot_df["delta"].dropna().to_numpy(dtype=float)
    wilcoxon_stat, wilcoxon_p = safe_wilcoxon_greater(delta_values)
    return Analysis2Result(
        detail_df=detail_df,
        plot_df=plot_df,
        stats_df=stats_df,
        missing_lstar_questions=missing_lstar_questions,
        no_pre_questions=no_pre_questions,
        no_post_questions=no_post_questions,
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_p=wilcoxon_p,
    )


def fit_mixedlm(formula: str, data: pd.DataFrame) -> tuple[Any | None, str | None]:
    if smf is None:
        return (None, "Skipped: statsmodels not available")
    last_error: Exception | None = None
    for method in ("lbfgs", "powell", "bfgs", "cg", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, data=data, groups=data["question_id"])
                result = model.fit(reml=False, method=method, disp=False)
            return (result, None)
        except Exception as exc:  # pragma: no cover - depends on runtime data
            last_error = exc
    return (None, f"Failed: {last_error}")


def run_analysis3(scatter_df: pd.DataFrame) -> dict[str, Any]:
    if smf is None:
        return {
            "base_status": "Skipped: statsmodels not available",
            "interaction_status": "Skipped: statsmodels not available",
            "base_result": None,
            "interaction_result": None,
            "interaction_coef": math.nan,
            "interaction_p": math.nan,
            "n_obs": int(len(scatter_df)),
            "n_groups": int(scatter_df["question_id"].nunique()),
        }

    model_df = scatter_df.copy()
    model_df["difficulty"] = pd.Categorical(model_df["difficulty"], categories=["medium", "hard"])

    base_result, base_error = fit_mixedlm("k_star_ratio ~ accuracy", model_df)
    interaction_result, interaction_error = fit_mixedlm("k_star_ratio ~ accuracy * C(difficulty)", model_df)

    interaction_coef = math.nan
    interaction_p = math.nan
    if interaction_result is not None:
        interaction_terms = [name for name in interaction_result.params.index if name.startswith("accuracy:C(difficulty)")]
        if interaction_terms:
            term = interaction_terms[0]
            interaction_coef = float(interaction_result.params[term])
            interaction_p = float(interaction_result.pvalues[term])

    return {
        "base_status": base_error or "ok",
        "interaction_status": interaction_error or "ok",
        "base_result": base_result,
        "interaction_result": interaction_result,
        "interaction_coef": interaction_coef,
        "interaction_p": interaction_p,
        "n_obs": int(len(model_df)),
        "n_groups": int(model_df["question_id"].nunique()),
    }


def format_group_line(stats_df: pd.DataFrame, group: str, value_col: str, symbol: str) -> str:
    row = stats_df.loc[stats_df["group"] == group].iloc[0]
    return (
        f"  Questions: {int(row['n'])}, Mean {symbol}: {row[f'mean_{value_col}']:.3f}, "
        f"Median {symbol}: {row[f'median_{value_col}']:.3f}, % {symbol}>0: {row['pct_positive']:.1f}%"
    )


def build_report(
    *,
    pq_run_dir: Path,
    accuracy_source: str,
    min_points: int,
    total_t1c_questions: int,
    accuracy_questions: int,
    merged_questions: int,
    scatter_df: pd.DataFrame,
    analysis1: Analysis1Result,
    analysis2: Analysis2Result,
    analysis3: dict[str, Any],
    output_dir: Path,
    pq_details_rows: int,
) -> str:
    rho_all = analysis1.stats_df.loc[analysis1.stats_df["group"] == "all"].iloc[0]
    delta_all = analysis2.stats_df.loc[analysis2.stats_df["group"] == "all"].iloc[0]
    rho_sig = "SIGNIFICANT" if (not math.isnan(analysis1.wilcoxon_p) and analysis1.wilcoxon_p < 0.05) else "NOT SIGNIFICANT"
    delta_sig = "SIGNIFICANT" if (not math.isnan(analysis2.wilcoxon_p) and analysis2.wilcoxon_p < 0.05) else "NOT SIGNIFICANT"

    report_lines = [
        "=============================================================",
        "  Per-Question k*/L vs Accuracy: Significance Report",
        "=============================================================",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"PQ Run: {pq_run_dir}",
        f"Accuracy source: {accuracy_source}",
        "",
        "-------------------------------------------------------------",
        "  Data Summary",
        "-------------------------------------------------------------",
        f"Total questions in t1c: {total_t1c_questions}",
        f"Questions with accuracy data: {accuracy_questions}",
        f"Questions merged (both accuracy + k*/L): {merged_questions}",
        f"Questions with >= {min_points} valid L-bins: {analysis1.eligible_questions}",
        f"  - medium: {int((analysis1.detail_df['difficulty'].eq('medium') & analysis1.detail_df['n_points'].ge(min_points)).sum())}",
        f"  - hard: {int((analysis1.detail_df['difficulty'].eq('hard') & analysis1.detail_df['n_points'].ge(min_points)).sum())}",
        f"Questions excluded (<{min_points} points): {analysis1.insufficient_questions}",
        f"Questions excluded (undefined Spearman): {analysis1.undefined_questions}",
        "",
        "-------------------------------------------------------------",
        "  Analysis 1: Per-Question Spearman Correlation",
        "-------------------------------------------------------------",
        "[ALL]",
        f"  Questions analysed: {analysis1.analysed_questions}",
        f"  Mean rho: {rho_all['mean_rho']:.3f}",
        f"  Median rho: {rho_all['median_rho']:.3f}",
        f"  % questions with rho > 0: {rho_all['pct_positive']:.1f}%",
        f"  Wilcoxon signed-rank (one-sided): W={analysis1.wilcoxon_stat:.3f}, p={analysis1.wilcoxon_p:.4f}",
        f"  One-sample t-test (one-sided): t={analysis1.ttest_stat:.3f}, p={analysis1.ttest_p:.4f}",
        f"  Bootstrap 95% CI for median rho: [{rho_all['ci_lo']:.3f}, {rho_all['ci_hi']:.3f}]",
        f"  -> {rho_sig} at alpha=0.05",
        "",
        "[MEDIUM]",
        format_group_line(analysis1.stats_df, "medium", "rho", "rho"),
        "[HARD]",
        format_group_line(analysis1.stats_df, "hard", "rho", "rho"),
        "",
        "-------------------------------------------------------------",
        "  Analysis 2: L* Split Test",
        "-------------------------------------------------------------",
        "[ALL]",
        f"  Questions with valid split: {int(delta_all['n'])}",
        (
            "    "
            f"(excluded: {analysis2.no_post_questions} no L>L* data, "
            f"{analysis2.no_pre_questions} no L<=L* data, {analysis2.missing_lstar_questions} missing L*)"
        ),
        f"  Mean delta(k*/L): {delta_all['mean_delta']:.3f}  (pre-L* minus post-L*)",
        f"  Median delta(k*/L): {delta_all['median_delta']:.3f}",
        f"  % questions with delta > 0: {delta_all['pct_positive']:.1f}%",
        f"  Paired Wilcoxon (one-sided): W={analysis2.wilcoxon_stat:.3f}, p={analysis2.wilcoxon_p:.4f}",
        f"  Bootstrap 95% CI for median delta: [{delta_all['ci_lo']:.3f}, {delta_all['ci_hi']:.3f}]",
        f"  -> {delta_sig} at alpha=0.05",
        "",
        "[MEDIUM]",
        format_group_line(analysis2.stats_df, "medium", "delta", "delta"),
        "[HARD]",
        format_group_line(analysis2.stats_df, "hard", "delta", "delta"),
        "",
        "-------------------------------------------------------------",
        "  Analysis 3: Mixed-Effects Regression",
        "-------------------------------------------------------------",
        "[Base model: k*/L ~ accuracy + (1|question)]",
    ]

    base_result = analysis3["base_result"]
    if base_result is None:
        report_lines.append(f"  {analysis3['base_status']}")
    else:
        coef = float(base_result.params["accuracy"])
        se = float(base_result.bse["accuracy"])
        p_value = float(base_result.pvalues["accuracy"])
        report_lines.extend(
            [
                f"  Coef(accuracy)={coef:.4f}, SE={se:.4f}, p={p_value:.4f}",
                f"  Observations={analysis3['n_obs']}, groups={analysis3['n_groups']}",
            ]
        )

    report_lines.append("")
    report_lines.append("[Interaction model: k*/L ~ accuracy * difficulty + (1|question)]")
    interaction_result = analysis3["interaction_result"]
    if interaction_result is None:
        report_lines.append(f"  {analysis3['interaction_status']}")
    else:
        accuracy_terms = [
            name
            for name in interaction_result.params.index
            if name == "accuracy"
        ]
        if accuracy_terms:
            term = accuracy_terms[0]
            report_lines.append(
                f"  Base accuracy effect={float(interaction_result.params[term]):.4f}, "
                f"SE={float(interaction_result.bse[term]):.4f}, p={float(interaction_result.pvalues[term]):.4f}"
            )
        report_lines.append(
            f"  Interaction term (accuracy x difficulty): coef={analysis3['interaction_coef']:.4f}, p={analysis3['interaction_p']:.4f}"
        )
        moderates = "YES" if (not math.isnan(analysis3["interaction_p"]) and analysis3["interaction_p"] < 0.05) else "NO"
        report_lines.append(f"  -> Difficulty moderates the relationship: {moderates}")

    report_lines.extend(
        [
            "",
            "-------------------------------------------------------------",
            "  Plot Source Data Exported",
            "-------------------------------------------------------------",
            f"  plot_src_fig_a_rho.csv           ({len(analysis1.plot_df)} rows)",
            f"  plot_src_fig_a_rho_stats.csv     ({len(analysis1.stats_df)} rows)",
            f"  plot_src_fig_b_delta.csv         ({len(analysis2.plot_df)} rows)",
            f"  plot_src_fig_b_delta_stats.csv   ({len(analysis2.stats_df)} rows)",
            f"  plot_src_fig_c_scatter.csv       ({len(scatter_df)} rows)",
            f"  plot_src_fig_c_scatter_stats.csv ({3} rows)",
            f"  pq_correlation_details.csv       ({pq_details_rows} rows)",
            "",
            "=============================================================",
        ]
    )
    return "\n".join(report_lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-run-dir", required=True, help="Path to the per-question run directory.")
    parser.add_argument(
        "--traces-path",
        help="Optional traces.jsonl path used only when no per-question accuracy CSV is available.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of valid (question, L) bins required for Analysis 1.",
    )
    parser.add_argument(
        "--output-dir",
        help="Destination directory. Defaults to {pq-run-dir}/pq_analysis/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pq_run_dir = Path(args.pq_run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (pq_run_dir / "pq_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    lstar_df = load_lstar_metadata(pq_run_dir)
    accuracy_df, accuracy_source = load_accuracy_table(
        pq_run_dir,
        Path(args.traces_path) if args.traces_path else None,
    )
    kstar_df = load_kstar_table(pq_run_dir)

    accuracy_df = accuracy_df.copy()
    accuracy_df["question_id"] = accuracy_df["question_id"].astype(str)
    accuracy_df["L"] = pd.to_numeric(accuracy_df["L"], errors="coerce").astype("Int64")
    accuracy_df["accuracy"] = pd.to_numeric(accuracy_df["accuracy"], errors="coerce")
    if "n_traces" not in accuracy_df.columns:
        accuracy_df["n_traces"] = pd.Series(pd.NA, index=accuracy_df.index, dtype="Int64")
    accuracy_df = accuracy_df[accuracy_df["L"].notna() & accuracy_df["accuracy"].notna() & (accuracy_df["L"] >= 3)]

    kstar_df = kstar_df[kstar_df["L"].notna() & kstar_df["k_star_ratio"].notna() & (kstar_df["L"] >= 3)]

    scatter_df = accuracy_df.merge(
        kstar_df[["question_id", "L", "k_star_ratio", "n_clean", "difficulty_score", "k_star"]],
        on=["question_id", "L"],
        how="inner",
    )
    scatter_df = scatter_df.merge(
        lstar_df[["question_id", "difficulty", "difficulty_score", "l_star_A", "l_star_S", "l_star_consistent"]],
        on="question_id",
        how="left",
        suffixes=("", "_meta"),
    )
    scatter_df["difficulty_score"] = scatter_df["difficulty_score_meta"].combine_first(scatter_df["difficulty_score"])
    scatter_df["difficulty"] = scatter_df["difficulty"].fillna(scatter_df["difficulty_score"].map(infer_difficulty))
    scatter_df = scatter_df[
        [
            "question_id",
            "difficulty",
            "L",
            "accuracy",
            "k_star_ratio",
            "n_clean",
            "n_traces",
            "k_star",
            "difficulty_score",
            "l_star_A",
            "l_star_S",
            "l_star_consistent",
        ]
    ].sort_values(["question_id", "L"])

    analysis1 = run_analysis1(scatter_df, min_points=args.min_points, rng=rng)
    analysis2 = run_analysis2(kstar_df, lstar_df, rng=rng)
    scatter_stats_df = build_scatter_stats(scatter_df)
    analysis3 = run_analysis3(scatter_df)

    pq_details_df = analysis1.detail_df.merge(
        analysis2.detail_df[
            [
                "question_id",
                "l_star_A",
                "mean_kstar_ratio_pre",
                "mean_kstar_ratio_post",
                "delta",
                "n_pre",
                "n_post",
                "analysis2_status",
            ]
        ],
        on="question_id",
        how="outer",
    ).merge(
        lstar_df[["question_id", "difficulty", "difficulty_score", "l_star_S", "l_star_consistent"]],
        on="question_id",
        how="left",
        suffixes=("", "_meta"),
    )
    if "difficulty_meta" in pq_details_df.columns:
        pq_details_df["difficulty"] = pq_details_df["difficulty"].fillna(pq_details_df["difficulty_meta"])
    if "difficulty_score_meta" in pq_details_df.columns:
        pq_details_df["difficulty_score"] = pq_details_df["difficulty_score"].fillna(pq_details_df["difficulty_score_meta"])
    pq_details_df = pq_details_df.drop(columns=[col for col in ("difficulty_meta", "difficulty_score_meta") if col in pq_details_df.columns])
    pq_details_df = pq_details_df.sort_values("question_id")

    analysis1.plot_df.to_csv(output_dir / "plot_src_fig_a_rho.csv", index=False)
    analysis1.stats_df.to_csv(output_dir / "plot_src_fig_a_rho_stats.csv", index=False)
    analysis2.plot_df.to_csv(output_dir / "plot_src_fig_b_delta.csv", index=False)
    analysis2.stats_df.to_csv(output_dir / "plot_src_fig_b_delta_stats.csv", index=False)
    scatter_df[["question_id", "difficulty", "L", "accuracy", "k_star_ratio", "n_clean"]].to_csv(
        output_dir / "plot_src_fig_c_scatter.csv",
        index=False,
    )
    scatter_stats_df.to_csv(output_dir / "plot_src_fig_c_scatter_stats.csv", index=False)
    pq_details_df.to_csv(output_dir / "pq_correlation_details.csv", index=False)

    report = build_report(
        pq_run_dir=pq_run_dir,
        accuracy_source=accuracy_source,
        min_points=args.min_points,
        total_t1c_questions=int(kstar_df["question_id"].nunique()),
        accuracy_questions=int(accuracy_df["question_id"].nunique()),
        merged_questions=int(scatter_df["question_id"].nunique()),
        scatter_df=scatter_df,
        analysis1=analysis1,
        analysis2=analysis2,
        analysis3=analysis3,
        output_dir=output_dir,
        pq_details_rows=int(len(pq_details_df)),
    )
    report_path = output_dir / "significance_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
