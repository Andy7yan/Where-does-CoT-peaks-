# split_t1b.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


VALUE_COLS = [
    "L",
    "step",
    "mean_nldd",
    "nldd_se",
    "mean_tas_t",
    "tas_t_se",
    "n_clean",
    "bin_status",
]

REQUIRED_COMMON = {"L", "step", "mean_nldd", "nldd_se", "n_clean", "bin_status"}

TAS_ALIASES = {
    "mean_tas": "mean_tas_t",
    "tas_se": "tas_t_se",
}


def slugify(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def parse_pipeline_filter(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def rename_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for old, new in TAS_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing_common = REQUIRED_COMMON - set(df.columns)
    if missing_common:
        raise ValueError(f"Missing required columns: {sorted(missing_common)}")

    if "mean_tas_t" not in df.columns:
        raise ValueError("Missing TAS column. Expected 'mean_tas_t' or alias 'mean_tas'.")
    if "tas_t_se" not in df.columns:
        raise ValueError("Missing TAS SE column. Expected 'tas_t_se' or alias 'tas_se'.")

    has_question_schema = "question_id" in df.columns
    has_scope_schema = "scope" in df.columns

    if not has_question_schema and not has_scope_schema:
        raise ValueError("Input must contain either 'question_id' or 'scope'.")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["L", "step", "n_clean"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["mean_nldd", "nldd_se", "mean_tas_t", "tas_t_se"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "question_id" in out.columns:
        out["question_id"] = out["question_id"].astype(str)

    if "scope" in out.columns:
        out["scope"] = out["scope"].astype(str)

    if "pipeline" in out.columns:
        out["pipeline"] = out["pipeline"].astype(str)
    else:
        out["pipeline"] = ""

    out["bin_status"] = out["bin_status"].astype(str)

    return out


def build_group_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "question_id" in out.columns:
        out["__group_kind"] = "per_question"
        out["__pipeline_norm"] = "pq"
        out["__scope_norm"] = out["question_id"].astype(str)
        out["__file_stem"] = out["question_id"].map(slugify)
        return out

    out["__pipeline_norm"] = out["pipeline"].fillna("").astype(str).str.strip().str.lower()
    out["__pipeline_norm"] = out["__pipeline_norm"].replace("", "overall")
    out["__scope_norm"] = out["scope"].astype(str)

    def infer_group_kind(pipeline_norm: str) -> str:
        return "per_question" if pipeline_norm == "pq" else "overall"

    out["__group_kind"] = out["__pipeline_norm"].map(infer_group_kind)

    def make_file_stem(row: pd.Series) -> str:
        if row["__group_kind"] == "per_question":
            return slugify(row["__scope_norm"])
        return slugify(f"{row['__pipeline_norm']}_{row['__scope_norm']}")

    out["__file_stem"] = out.apply(make_file_stem, axis=1)
    out["question_id"] = out["__scope_norm"].where(out["__group_kind"] == "per_question", None)

    return out


def apply_filters(
    df: pd.DataFrame,
    only_ok: bool,
    pipeline_filter: set[str] | None,
) -> pd.DataFrame:
    out = df.copy()

    if pipeline_filter is not None:
        out = out[out["__pipeline_norm"].isin(pipeline_filter)].copy()

    if only_ok:
        out = out[out["bin_status"].str.lower() == "ok"].copy()

    out = out[out["L"].notna() & out["step"].notna()].copy()
    out["L"] = out["L"].astype(int)
    out["step"] = out["step"].astype(int)

    return out


def export_group_csvs(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    group_cols = ["__group_kind", "__pipeline_norm", "__scope_norm", "__file_stem"]

    for keys, gdf in df.groupby(group_cols, sort=True, dropna=False):
        group_kind, pipeline_norm, scope_norm, file_stem = keys
        gdf = gdf.sort_values(["L", "step"], kind="mergesort").copy()

        if group_kind == "per_question":
            out_df = gdf[["question_id"] + VALUE_COLS].copy()
        else:
            out_df = gdf[["scope", "pipeline"] + VALUE_COLS].copy()

        out_path = outdir / f"{file_stem}.csv"
        out_df.to_csv(out_path, index=False)

        rows.append(
            {
                "group_kind": group_kind,
                "pipeline": pipeline_norm,
                "scope": scope_norm,
                "file_stem": file_stem,
                "csv_path": str(out_path),
                "n_rows": len(gdf),
                "n_L": int(gdf["L"].nunique()),
                "min_L": int(gdf["L"].min()),
                "max_L": int(gdf["L"].max()),
                "step_max": int(gdf["step"].max()),
                "mean_n_clean": float(gdf["n_clean"].mean()) if len(gdf) else 0.0,
                "n_ok_rows": int((gdf["bin_status"].str.lower() == "ok").sum()),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(
        ["group_kind", "pipeline", "scope"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    manifest.to_csv(outdir / "_split_manifest.csv", index=False)
    return manifest


def export_group_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    group_cols = ["__group_kind", "__pipeline_norm", "__scope_norm", "__file_stem"]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("L", "size"),
            n_L=("L", "nunique"),
            min_L=("L", "min"),
            max_L=("L", "max"),
            step_max=("step", "max"),
            mean_n_clean=("n_clean", "mean"),
            n_ok_rows=("bin_status", lambda s: int((s.str.lower() == "ok").sum())),
            n_valid_nldd=("mean_nldd", lambda s: int(s.notna().sum())),
            n_valid_tas=("mean_tas_t", lambda s: int(s.notna().sum())),
        )
        .reset_index()
        .rename(
            columns={
                "__group_kind": "group_kind",
                "__pipeline_norm": "pipeline",
                "__scope_norm": "scope",
                "__file_stem": "file_stem",
            }
        )
    )

    summary["coverage_score"] = (
        summary["n_L"] * 10
        + summary["n_valid_nldd"] * 2
        + summary["n_valid_tas"]
        + summary["mean_n_clean"].fillna(0) * 0.1
    )

    summary = summary.sort_values(
        ["group_kind", "coverage_score", "scope"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    summary.to_csv(outdir / "_group_summary.csv", index=False)
    return summary


def export_selected_questions(summary: pd.DataFrame, outdir: Path, top_k: int) -> pd.DataFrame:
    pq = summary[summary["group_kind"] == "per_question"].copy()
    pq = pq.sort_values(
        ["coverage_score", "n_L", "n_valid_nldd", "mean_n_clean", "scope"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected = pq.head(top_k).copy()
    selected.to_csv(outdir / "_selected_questions.csv", index=False)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split T1-B CSV into per-question and/or overall heatmap CSV files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input T1-B CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="t1b_splits",
        help="Directory for exported split CSV files.",
    )
    parser.add_argument(
        "--only-ok",
        action="store_true",
        help="Keep only rows with bin_status == ok.",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default=None,
        help="Optional comma-separated pipeline filter, e.g. 'v6,pq'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many top per-question files to record in _selected_questions.csv.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pipeline_filter = parse_pipeline_filter(args.pipelines)

    df = pd.read_csv(input_path)
    df = rename_alias_columns(df)
    validate_columns(df)
    df = coerce_types(df)
    df = build_group_metadata(df)
    df = apply_filters(df, only_ok=args.only_ok, pipeline_filter=pipeline_filter)

    if df.empty:
        raise ValueError("No rows remain after filtering.")

    manifest = export_group_csvs(df, outdir)
    summary = export_group_summary(df, outdir)
    selected = export_selected_questions(summary, outdir, args.top_k)

    print(f"Input: {input_path}")
    print(f"Output directory: {outdir}")
    print(f"Exported files: {len(manifest)}")
    print(f"Manifest: {outdir / '_split_manifest.csv'}")
    print(f"Summary: {outdir / '_group_summary.csv'}")
    print(f"Selected per-question groups: {len(selected)}")
    print(f"Selected file: {outdir / '_selected_questions.csv'}")


if __name__ == "__main__":
    main()