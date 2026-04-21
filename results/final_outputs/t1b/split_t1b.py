# split_t1b.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_BASE_COLS = {
    "L",
    "step",
    "mean_nldd",
    "nldd_se",
    "n_clean",
    "bin_status",
}

TAS_CANDIDATES = ["mean_tas_t", "mean_tas"]
TAS_SE_CANDIDATES = ["tas_t_se", "tas_se"]


def _find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_slug(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def load_and_normalise(csv_path: Path) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tas_col = _find_first_existing(df, TAS_CANDIDATES)
    tas_se_col = _find_first_existing(df, TAS_SE_CANDIDATES)

    if tas_col is None:
        raise ValueError(
            f"Missing TAS column. Expected one of: {TAS_CANDIDATES}"
        )
    if tas_se_col is None:
        raise ValueError(
            f"Missing TAS SE column. Expected one of: {TAS_SE_CANDIDATES}"
        )

    if "question_id" not in df.columns:
        if "scope" in df.columns:
            df["question_id"] = df["scope"].astype(str)
        else:
            raise ValueError(
                "Missing question identifier. Expected 'question_id' or 'scope'."
            )

    if "pipeline" in df.columns:
        df = df[df["pipeline"].astype(str).str.lower() == "pq"].copy()

    df = df.rename(columns={tas_col: "mean_tas_t", tas_se_col: "tas_t_se"}).copy()

    numeric_cols = [
        "L",
        "step",
        "mean_nldd",
        "nldd_se",
        "mean_tas_t",
        "tas_t_se",
        "n_clean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["question_id"] = df["question_id"].astype(str)
    df["bin_status"] = df["bin_status"].astype(str)

    return df, "mean_tas_t", "tas_t_se"


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out[out["bin_status"].str.lower() == "ok"].copy()

    out = out[out["L"].notna() & out["step"].notna()].copy()
    out["L"] = out["L"].astype(int)
    out["step"] = out["step"].astype(int)

    out = out[out["L"] >= 3].copy()
    out = out[(out["step"] >= 1) & (out["step"] <= out["L"])].copy()

    return out


def build_question_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["valid_nldd_cell"] = (
        (work["step"] >= 2) &
        work["mean_nldd"].notna()
    )
    work["valid_tas_cell"] = work["mean_tas_t"].notna()

    grouped = work.groupby("question_id", dropna=False)

    summary = grouped.agg(
        n_rows=("question_id", "size"),
        n_L=("L", "nunique"),
        min_L=("L", "min"),
        max_L=("L", "max"),
        max_n_clean=("n_clean", "max"),
        mean_n_clean=("n_clean", "mean"),
        n_valid_nldd=("valid_nldd_cell", "sum"),
        n_valid_tas=("valid_tas_cell", "sum"),
    ).reset_index()

    summary["coverage_score"] = (
        summary["n_L"] * 10
        + summary["n_valid_nldd"] * 2
        + summary["n_valid_tas"] * 1
        + summary["mean_n_clean"].fillna(0) * 0.1
    )

    summary = summary.sort_values(
        by=[
            "coverage_score",
            "n_L",
            "n_valid_nldd",
            "mean_n_clean",
            "question_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return summary


def export_per_question_csvs(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    per_q_dir = outdir / "per_question"
    per_q_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    for question_id, qdf in df.groupby("question_id", sort=True):
        qdf = qdf.sort_values(["L", "step"], kind="mergesort").copy()
        slug = _safe_slug(question_id)
        out_path = per_q_dir / f"{slug}.csv"
        qdf.to_csv(out_path, index=False)

        manifest_rows.append(
            {
                "question_id": question_id,
                "slug": slug,
                "csv_path": str(out_path),
                "n_rows": len(qdf),
                "n_L": qdf["L"].nunique(),
                "min_L": int(qdf["L"].min()),
                "max_L": int(qdf["L"].max()),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["n_L", "n_rows", "question_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    manifest.to_csv(outdir / "per_question_manifest.csv", index=False)
    return manifest


def export_selected_questions(summary: pd.DataFrame, outdir: Path, top_k: int) -> pd.DataFrame:
    selected = summary.head(top_k).copy()
    selected.to_csv(outdir / "selected_questions.csv", index=False)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split t1b_step_surface.csv into per-question CSV files for T1-B plotting."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="t1b_step_surface.csv",
        help="Path to the combined t1b_step_surface.csv file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="t1b_split",
        help="Output directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many representative questions to list in selected_questions.csv.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df, _, _ = load_and_normalise(input_path)
    df = filter_valid_rows(df)

    if df.empty:
        raise ValueError("No valid PQ rows remain after filtering.")

    summary = build_question_summary(df)
    manifest = export_per_question_csvs(df, outdir)
    selected = export_selected_questions(summary, outdir, args.top_k)

    summary.to_csv(outdir / "question_summary.csv", index=False)

    print(f"Input: {input_path}")
    print(f"Output directory: {outdir}")
    print(f"Questions exported: {len(manifest)}")
    print(f"Top-{args.top_k} representative questions written to: {outdir / 'selected_questions.csv'}")
    print(f"Per-question manifest written to: {outdir / 'per_question_manifest.csv'}")
    print(f"Summary written to: {outdir / 'question_summary.csv'}")


if __name__ == "__main__":
    main()