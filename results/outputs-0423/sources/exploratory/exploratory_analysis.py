"""Exploratory statistics over Phase 1 per-question artifacts."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ConstantInputWarning


PNG_DPI = 150
SPEARMAN_ABS_THRESHOLD = 0.3
SPEARMAN_P_THRESHOLD = 0.05
DIST_CORR_THRESHOLD = 0.3
EPSILON = 1e-12


def _warn(message: str, warning_log: list[str]) -> None:
    warning_log.append(message)
    print(f"WARNING: {message}")


def _to_bool_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return series


def _discover_artifact(data_dir: Path, filename: str, preferred_dirs: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    if (data_dir / filename).exists():
        candidates.append(data_dir / filename)
    for dirname in preferred_dirs:
        candidates.append(data_dir / dirname / filename)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _load_csv(path: Path | None, *, warning_log: list[str], label: str) -> pd.DataFrame | None:
    if path is None:
        _warn(f"Missing required input for {label}.", warning_log)
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        _warn(f"Failed to read {label} from {path}: {exc}", warning_log)
        return None
    if frame.empty:
        _warn(f"{label} exists but is empty: {path}", warning_log)
        return None
    return frame


def _normalize_t1b(frame: pd.DataFrame, *, warning_log: list[str]) -> pd.DataFrame:
    result = frame.copy()
    if "question_id" not in result.columns:
        if "scope" in result.columns:
            result = result.rename(columns={"scope": "question_id"})
        else:
            _warn("t1b_step_surface.csv has neither question_id nor scope; shape features will be skipped.", warning_log)
            return pd.DataFrame()
    if "pipeline" in result.columns:
        result = result[result["pipeline"].astype(str) == "pq"].copy()
    for col in ("L", "step", "mean_nldd", "nldd_se", "mean_tas_t", "tas_t_se", "n_clean"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    if "bin_status" not in result.columns:
        result["bin_status"] = ""
    result["question_id"] = result["question_id"].astype(str)
    return result


def _normalize_t1c(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["question_id"] = result["question_id"].astype(str)
    for col in ("difficulty_score", "L", "k_star", "k_star_ratio", "n_clean"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _normalize_t2b(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["question_id"] = result["question_id"].astype(str)
    if "l_star_C" not in result.columns and "l_star_S" in result.columns:
        result = result.rename(columns={"l_star_S": "l_star_C"})
    for col in ("difficulty_score", "l_star_A", "l_star_C"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    if "l_star_consistent" in result.columns:
        result["l_star_consistent"] = (
            result["l_star_consistent"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )
    return result


def _build_curve_features(step_frame: pd.DataFrame) -> pd.DataFrame:
    if step_frame.empty:
        return pd.DataFrame(columns=["question_id", "L"])

    rows: list[dict[str, Any]] = []
    grouped = step_frame.groupby(["question_id", "L"], dropna=False, sort=False)
    for (question_id, length), group in grouped:
        group = group.sort_values("step").copy()
        nldd_values = group["mean_nldd"].dropna().to_numpy(dtype=float)
        tas_values = group["mean_tas_t"].dropna().to_numpy(dtype=float)
        nldd_peak_value = float(np.nanmax(nldd_values)) if nldd_values.size else math.nan
        nldd_total = float(np.nansum(nldd_values)) if nldd_values.size else math.nan
        if nldd_values.size and not math.isclose(nldd_total, 0.0, abs_tol=EPSILON):
            early_mask = group["step"].to_numpy(dtype=float) <= (float(length) / 2.0)
            early_mass = float(np.nansum(group.loc[early_mask, "mean_nldd"].to_numpy(dtype=float)) / nldd_total)
        else:
            early_mass = math.nan
        if nldd_values.size >= 3:
            nldd_skewness = float(stats.skew(nldd_values, bias=False, nan_policy="omit"))
        else:
            nldd_skewness = math.nan
        tas_final_match = group.loc[group["step"] == group["L"], "mean_tas_t"].dropna()
        tas_final = float(tas_final_match.iloc[-1]) if not tas_final_match.empty else math.nan
        if tas_values.size >= 2:
            diffs = np.diff(tas_values)
            tas_monotonicity = float(np.mean(diffs >= 0.0))
        else:
            tas_monotonicity = math.nan
        rows.append(
            {
                "question_id": str(question_id),
                "L": float(length),
                "nldd_peak_value": nldd_peak_value,
                "nldd_early_mass": early_mass,
                "nldd_skewness": nldd_skewness,
                "tas_final": tas_final,
                "tas_monotonicity": tas_monotonicity,
            }
        )
    return pd.DataFrame(rows)


def _build_table_a(t1c: pd.DataFrame, t2b: pd.DataFrame) -> pd.DataFrame:
    merge_keys = t2b[["question_id", "l_star_A"]].rename(columns={"l_star_A": "L"})
    lstar_bins = merge_keys.merge(t1c, on=["question_id", "L"], how="left")
    lstar_bins = lstar_bins.rename(
        columns={
            "k_star": "k_star_at_lstar",
            "k_star_ratio": "k_star_ratio_at_lstar",
            "n_clean": "n_clean_at_lstar",
            "L": "L_at_lstar",
        }
    )
    keep_cols = ["question_id", "L_at_lstar", "k_star_at_lstar", "k_star_ratio_at_lstar", "n_clean_at_lstar"]
    return t2b.merge(lstar_bins[keep_cols], on="question_id", how="left")


def _build_table_b(t1c: pd.DataFrame, t2b: pd.DataFrame, curve_features: pd.DataFrame | None) -> pd.DataFrame:
    table_b = t1c.merge(t2b, on=["question_id", "difficulty_score"], how="left")
    table_b["L_minus_l_star_A"] = table_b["L"] - table_b["l_star_A"]
    table_b["k_star_minus_l_star_A"] = table_b["k_star"] - table_b["l_star_A"]
    denominator = table_b["l_star_A"].replace({0: np.nan})
    table_b["L_div_l_star_A"] = table_b["L"] / denominator
    if curve_features is not None and not curve_features.empty:
        table_b = table_b.merge(curve_features, on=["question_id", "L"], how="left")
    return table_b


def _numeric_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).copy()
    for col in numeric.columns:
        numeric[col] = _to_bool_numeric(numeric[col])
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any(axis=0)]
    return numeric


def _expand_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric_feature_frame(frame)
    derived: dict[str, pd.Series] = {}
    columns = list(numeric.columns)
    for idx, left in enumerate(columns):
        left_values = numeric[left]
        for right in columns[idx + 1 :]:
            right_values = numeric[right]
            derived[f"diff__{left}__{right}"] = left_values - right_values
            if not np.isclose(right_values.fillna(0.0).to_numpy(dtype=float), 0.0, atol=EPSILON).any():
                derived[f"ratio__{left}__{right}"] = left_values / right_values
    expanded = pd.concat([numeric, pd.DataFrame(derived)], axis=1) if derived else numeric
    return expanded


def _pairwise_spearman(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    columns = list(frame.columns)
    if len(columns) < 2:
        corr = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
        pvals = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns, dtype=float)
        return corr, pvals, []

    valid_counts = frame.notna().astype(int).T.dot(frame.notna().astype(int))
    pairs: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = stats.spearmanr(frame, axis=0, nan_policy="omit")
    corr_values = np.array(result.statistic, dtype=float, copy=True)
    pval_values = np.array(result.pvalue, dtype=float, copy=True)
    np.fill_diagonal(corr_values, 1.0)
    np.fill_diagonal(pval_values, 0.0)
    corr = pd.DataFrame(corr_values, index=columns, columns=columns)
    pvals = pd.DataFrame(pval_values, index=columns, columns=columns)
    for idx, left in enumerate(columns):
        for right in columns[idx + 1 :]:
            n_complete = int(valid_counts.loc[left, right])
            rho = float(corr.loc[left, right]) if n_complete >= 3 else math.nan
            pvalue = float(pvals.loc[left, right]) if n_complete >= 3 else math.nan
            pairs.append(
                {
                    "feature_x": left,
                    "feature_y": right,
                    "spearman_rho": rho,
                    "p_value": pvalue,
                    "n_complete": n_complete,
                    "abs_rho": abs(rho) if not math.isnan(rho) else math.nan,
                }
            )
    return corr, pvals, pairs


def _distance_correlation_fallback(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if len(x) < 3 or len(y) < 3:
        return math.nan
    a = squareform(pdist(x, metric="euclidean"))
    b = squareform(pdist(y, metric="euclidean"))
    a_centered = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b_centered = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = float(np.mean(a_centered * b_centered))
    dvar_x2 = float(np.mean(a_centered * a_centered))
    dvar_y2 = float(np.mean(b_centered * b_centered))
    if dvar_x2 <= 0.0 or dvar_y2 <= 0.0:
        return math.nan
    dcor2 = max(dcov2, 0.0) / math.sqrt(dvar_x2 * dvar_y2)
    return float(math.sqrt(max(dcor2, 0.0)))


def _distance_correlation_pairs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    columns = list(frame.columns)
    for idx, left in enumerate(columns):
        for right in columns[idx + 1 :]:
            subset = frame[[left, right]].dropna()
            if len(subset) < 3:
                value = math.nan
            else:
                value = _distance_correlation_fallback(
                    subset[left].to_numpy(dtype=float),
                    subset[right].to_numpy(dtype=float),
                )
            pairs.append(
                {
                    "feature_x": left,
                    "feature_y": right,
                    "distance_correlation": value,
                    "n_complete": int(len(subset)),
                }
            )
    return pairs


def _prepare_pca_input(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = _numeric_feature_frame(frame)
    usable = numeric.loc[:, numeric.notna().sum(axis=0) >= 3].copy()
    usable = usable.loc[:, usable.nunique(dropna=True) > 1]
    if usable.empty:
        return usable, usable
    filled = usable.apply(lambda col: col.fillna(col.median()), axis=0)
    means = filled.mean(axis=0)
    stds = filled.std(axis=0, ddof=0).replace({0.0: np.nan})
    standardized = (filled - means) / stds
    standardized = standardized.dropna(axis=1, how="any")
    return usable, standardized


def _run_pca(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original, standardized = _prepare_pca_input(frame)
    if standardized.empty or standardized.shape[0] < 2 or standardized.shape[1] < 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    matrix = standardized.to_numpy(dtype=float)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    explained_variance = (singular_values**2) / max(matrix.shape[0] - 1, 1)
    total_variance = explained_variance.sum()
    explained_ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)

    pc_labels = [f"PC{i + 1}" for i in range(vt.shape[0])]
    scree = pd.DataFrame(
        {
            "component": pc_labels,
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_ratio,
        }
    )

    score_values = u * singular_values
    scores = pd.DataFrame(score_values, columns=pc_labels, index=standardized.index)
    if "question_id" in frame.columns:
        scores.insert(0, "question_id", frame.loc[scores.index, "question_id"].astype(str).to_numpy())
    if "difficulty_score" in frame.columns:
        scores["difficulty_score"] = pd.to_numeric(frame.loc[scores.index, "difficulty_score"], errors="coerce").to_numpy()

    loadings = pd.DataFrame(vt.T, index=standardized.columns, columns=pc_labels)
    loadings.insert(0, "feature", loadings.index)
    return scree, scores, loadings.reset_index(drop=True)


def _save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _plot_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    size = max(8, min(18, 0.55 * len(corr.columns)))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)


def _plot_scree(scree: pd.DataFrame, path: Path, title: str) -> None:
    top = scree.head(min(10, len(scree)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(top["component"], top["explained_variance_ratio"], marker="o")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_xlabel("Principal Component")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)


def _plot_biplot(scores: pd.DataFrame, loadings: pd.DataFrame, path: Path, title: str) -> None:
    if "PC1" not in scores.columns or "PC2" not in scores.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    color_values = scores["difficulty_score"] if "difficulty_score" in scores.columns else None
    scatter = ax.scatter(
        scores["PC1"],
        scores["PC2"],
        c=color_values,
        cmap="viridis" if color_values is not None else None,
        alpha=0.75,
        edgecolors="none",
    )
    if color_values is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("difficulty_score")

    score_x = float(np.nanmax(np.abs(scores["PC1"]))) if len(scores) else 1.0
    score_y = float(np.nanmax(np.abs(scores["PC2"]))) if len(scores) else 1.0
    loading_x = float(np.nanmax(np.abs(loadings["PC1"]))) if len(loadings) else 1.0
    loading_y = float(np.nanmax(np.abs(loadings["PC2"]))) if len(loadings) else 1.0
    scales = [
        (score_x / loading_x) if loading_x and not math.isclose(loading_x, 0.0, abs_tol=EPSILON) else math.nan,
        (score_y / loading_y) if loading_y and not math.isclose(loading_y, 0.0, abs_tol=EPSILON) else math.nan,
    ]
    arrow_scale = 0.7 * min(scale for scale in scales if not math.isnan(scale)) if any(not math.isnan(scale) for scale in scales) else 1.0

    for _, row in loadings.iterrows():
        ax.arrow(0.0, 0.0, row["PC1"] * arrow_scale, row["PC2"] * arrow_scale, color="firebrick", alpha=0.7, width=0.005)
        ax.text(row["PC1"] * arrow_scale * 1.08, row["PC2"] * arrow_scale * 1.08, str(row["feature"]), color="firebrick", fontsize=8)

    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)


def _plot_clustermap(abs_corr: pd.DataFrame, path: Path, title: str) -> None:
    distance_matrix = (1.0 - abs_corr).clip(lower=0.0)
    condensed = squareform(distance_matrix.to_numpy(dtype=float), checks=False)
    linkage = hierarchy.linkage(condensed, method="ward")
    cluster = sns.clustermap(
        abs_corr,
        row_linkage=linkage,
        col_linkage=linkage,
        cmap="mako",
        figsize=(max(8, 0.55 * len(abs_corr.columns)), max(8, 0.55 * len(abs_corr.columns))),
    )
    cluster.figure.suptitle(title)
    cluster.figure.savefig(path, dpi=PNG_DPI)
    plt.close(cluster.figure)


def _analyze_table(
    *,
    name: str,
    raw_table: pd.DataFrame,
    csv_dir: Path,
    figure_dir: Path,
    warning_log: list[str],
) -> dict[str, Any]:
    raw_numeric = _numeric_feature_frame(raw_table)
    if raw_numeric.shape[1] < 2:
        _warn(f"{name} has fewer than two numeric features; skipping downstream analyses.", warning_log)
        return {"name": name, "rows": int(len(raw_table)), "numeric_features": int(raw_numeric.shape[1])}

    expanded = _expand_numeric_features(raw_table)
    _save_dataframe(expanded, csv_dir / f"{name}_expanded_features.csv")

    raw_corr, raw_pvals, _ = _pairwise_spearman(raw_numeric)
    raw_corr_for_plot = raw_corr.fillna(0.0)
    _save_dataframe(raw_corr.reset_index(names="feature"), csv_dir / f"{name}_spearman_heatmap_source.csv")
    _save_dataframe(raw_pvals.reset_index(names="feature"), csv_dir / f"{name}_spearman_pvalues.csv")
    _plot_heatmap(raw_corr_for_plot, figure_dir / f"{name}_spearman_heatmap.png", f"{name} Original-Feature Spearman")

    expanded_corr, expanded_pvals, expanded_pairs = _pairwise_spearman(expanded)
    _save_dataframe(expanded_corr.reset_index(names="feature"), csv_dir / f"{name}_expanded_spearman_matrix.csv")
    _save_dataframe(expanded_pvals.reset_index(names="feature"), csv_dir / f"{name}_expanded_spearman_pvalues.csv")
    spearman_pairs = pd.DataFrame(expanded_pairs)
    spearman_top = spearman_pairs[
        (spearman_pairs["abs_rho"] > SPEARMAN_ABS_THRESHOLD) & (spearman_pairs["p_value"] < SPEARMAN_P_THRESHOLD)
    ].sort_values(["abs_rho", "p_value"], ascending=[False, True]).head(50)
    _save_dataframe(spearman_top, csv_dir / f"{name}_spearman_top50.csv")

    dcor_pairs = pd.DataFrame(_distance_correlation_pairs(raw_numeric))
    dcor_top = dcor_pairs[dcor_pairs["distance_correlation"] > DIST_CORR_THRESHOLD].sort_values(
        "distance_correlation",
        ascending=False,
    ).head(30)
    _save_dataframe(dcor_pairs, csv_dir / f"{name}_distance_correlation_all.csv")
    _save_dataframe(dcor_top, csv_dir / f"{name}_distance_correlation_top30.csv")

    scree, scores, loadings = _run_pca(raw_table)
    if not scree.empty:
        _save_dataframe(scree, csv_dir / f"{name}_pca_scree.csv")
        _plot_scree(scree, figure_dir / f"{name}_pca_scree.png", f"{name} PCA Scree Plot")
    else:
        _warn(f"{name} PCA skipped because there were not enough usable numeric columns.", warning_log)
    if not scores.empty:
        _save_dataframe(scores, csv_dir / f"{name}_pca_scores.csv")
    if not loadings.empty:
        _save_dataframe(loadings, csv_dir / f"{name}_pca_loadings.csv")
    if not scores.empty and not loadings.empty and "PC2" in scores.columns and "PC2" in loadings.columns:
        _plot_biplot(scores, loadings, figure_dir / f"{name}_pca_biplot.png", f"{name} PCA Biplot")
    else:
        _warn(f"{name} PCA biplot skipped because fewer than two principal components were available.", warning_log)

    abs_corr = raw_corr.abs().fillna(0.0)
    if abs_corr.shape[0] >= 2:
        _save_dataframe(abs_corr.reset_index(names="feature"), csv_dir / f"{name}_clustermap_source.csv")
        _plot_clustermap(abs_corr, figure_dir / f"{name}_clustermap.png", f"{name} |Spearman rho| Clustermap")
    else:
        _warn(f"{name} clustermap skipped because fewer than two features were available.", warning_log)

    return {
        "name": name,
        "rows": int(len(raw_table)),
        "numeric_features": int(raw_numeric.shape[1]),
        "expanded_features": int(expanded.shape[1]),
        "spearman_hits": int(len(spearman_top)),
        "distance_correlation_hits": int(len(dcor_top)),
        "pca_components": int(len(scree)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Root containing pq_analysis/ and optionally analysis_phase1/.")
    parser.add_argument("--output-dir", help="Single directory for both CSV and PNG outputs. Defaults to {data-dir}/exploratory.")
    parser.add_argument("--csv-dir", help="Directory for CSV outputs.")
    parser.add_argument("--figure-dir", help="Directory for PNG outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    if args.csv_dir or args.figure_dir:
        csv_dir = Path(args.csv_dir) if args.csv_dir else Path(args.output_dir or (data_dir / "exploratory"))
        figure_dir = Path(args.figure_dir) if args.figure_dir else Path(args.output_dir or (data_dir / "exploratory"))
    else:
        base_output = Path(args.output_dir) if args.output_dir else (data_dir / "exploratory")
        csv_dir = base_output
        figure_dir = base_output
    csv_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []
    artifact_paths = {
        "t1b_step_surface": _discover_artifact(data_dir, "t1b_step_surface.csv", ("pq_analysis", "analysis_phase1", "analysis")),
        "t1c_kstar_ratio": _discover_artifact(data_dir, "t1c_kstar_ratio.csv", ("pq_analysis",)),
        "t2b_lstar_difficulty": _discover_artifact(data_dir, "t2b_lstar_difficulty.csv", ("pq_analysis",)),
        "nldd_surface": _discover_artifact(data_dir, "nldd_surface.csv", ("analysis_phase1", "analysis")),
        "k_star_by_L": _discover_artifact(data_dir, "k_star_by_L.csv", ("analysis_phase1", "analysis")),
        "accuracy_by_length": _discover_artifact(data_dir, "accuracy_by_length.csv", ("analysis_phase1", "analysis")),
    }

    t1c = _load_csv(artifact_paths["t1c_kstar_ratio"], warning_log=warning_log, label="t1c_kstar_ratio.csv")
    t2b = _load_csv(artifact_paths["t2b_lstar_difficulty"], warning_log=warning_log, label="t2b_lstar_difficulty.csv")
    t1b = _load_csv(artifact_paths["t1b_step_surface"], warning_log=warning_log, label="t1b_step_surface.csv")

    if t1c is None or t2b is None:
        raise RuntimeError("Cannot continue without non-empty t1c_kstar_ratio.csv and t2b_lstar_difficulty.csv.")

    t1c = _normalize_t1c(t1c)
    t2b = _normalize_t2b(t2b)
    curve_features: pd.DataFrame | None = None
    if t1b is not None:
        t1b = _normalize_t1b(t1b, warning_log=warning_log)
        if not t1b.empty:
            curve_features = _build_curve_features(t1b)
            _save_dataframe(curve_features, csv_dir / "table_B_curve_features.csv")
        else:
            _warn("t1b_step_surface.csv could not be normalized into per-question curves; shape features skipped.", warning_log)

    table_a = _build_table_a(t1c, t2b)
    table_b = _build_table_b(t1c, t2b, curve_features)
    _save_dataframe(table_a, csv_dir / "table_A.csv")
    _save_dataframe(table_b, csv_dir / "table_B.csv")

    summaries = [
        _analyze_table(name="table_A", raw_table=table_a, csv_dir=csv_dir, figure_dir=figure_dir, warning_log=warning_log),
        _analyze_table(name="table_B", raw_table=table_b, csv_dir=csv_dir, figure_dir=figure_dir, warning_log=warning_log),
    ]

    run_summary = {
        "data_dir": str(data_dir),
        "csv_dir": str(csv_dir),
        "figure_dir": str(figure_dir),
        "artifact_paths": {key: str(value) if value is not None else None for key, value in artifact_paths.items()},
        "tables": summaries,
        "warnings": warning_log,
    }
    summary_path = csv_dir / "exploratory_run_summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"table_A_rows: {len(table_a)}")
    print(f"table_B_rows: {len(table_b)}")
    print(f"csv_dir: {csv_dir}")
    print(f"figure_dir: {figure_dir}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()
