# select_t1b.py
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def slugify(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def is_overall_file(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.startswith("v6_") or stem.startswith("overall_")


def is_usable_number(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s not in {"", "nan", "null", "none"}


def find_longest_contiguous_run(values: list[int]) -> tuple[int | None, int | None, int]:
    if not values:
        return None, None, 0

    values = sorted(set(values))
    best_start = values[0]
    best_end = values[0]
    best_len = 1

    cur_start = values[0]
    cur_end = values[0]

    for v in values[1:]:
        if v == cur_end + 1:
            cur_end = v
        else:
            cur_len = cur_end - cur_start + 1
            if cur_len > best_len:
                best_start, best_end, best_len = cur_start, cur_end, cur_len
            cur_start = v
            cur_end = v

    cur_len = cur_end - cur_start + 1
    if cur_len > best_len:
        best_start, best_end, best_len = cur_start, cur_end, cur_len

    return best_start, best_end, best_len


def analyse_file(path: Path, expected_lmin: int, only_ok: bool) -> dict | None:
    df = pd.read_csv(path)

    # 只分析 per-question 文件
    if "question_id" not in df.columns:
        return None

    if only_ok and "bin_status" in df.columns:
        df = df[df["bin_status"].astype(str).str.lower() == "ok"].copy()

    if df.empty:
        return None

    required_cols = {"question_id", "L", "step", "mean_nldd", "mean_tas_t", "n_clean"}
    if not required_cols.issubset(df.columns):
        return None

    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["n_clean"] = pd.to_numeric(df["n_clean"], errors="coerce")

    df = df[df["L"].notna() & df["step"].notna()].copy()
    if df.empty:
        return None

    df["L"] = df["L"].astype(int)
    df["step"] = df["step"].astype(int)

    question_ids = df["question_id"].dropna().astype(str).unique().tolist()
    question_id = question_ids[0] if question_ids else path.stem

    max_L = int(df["L"].max())
    min_L = int(df["L"].min())

    per_L_rows: list[dict] = []
    complete_Ls: list[int] = []

    for L in sorted(df["L"].unique()):
        sub = df[df["L"] == L].copy().sort_values("step", kind="mergesort")
        expected_steps = list(range(1, L + 1))
        actual_steps = sub["step"].tolist()

        steps_complete = actual_steps == expected_steps
        row_count_ok = len(sub) == L

        tas_complete = sub["mean_tas_t"].map(is_usable_number).all()

        if L >= 2:
            nldd_mask = sub["step"] >= 2
            nldd_complete = sub.loc[nldd_mask, "mean_nldd"].map(is_usable_number).all()
        else:
            nldd_complete = True

        is_complete = bool(steps_complete and row_count_ok and tas_complete and nldd_complete)

        if is_complete:
            complete_Ls.append(int(L))

        per_L_rows.append(
            {
                "L": int(L),
                "is_complete": is_complete,
                "n_rows": int(len(sub)),
                "mean_n_clean_L": float(sub["n_clean"].mean()) if len(sub) else 0.0,
            }
        )

    target_Ls = list(range(expected_lmin, max_L + 1))
    full_uninterrupted = (
        max_L >= expected_lmin and
        all(L in complete_Ls for L in target_Ls)
    )

    run_start, run_end, run_len = find_longest_contiguous_run([L for L in complete_Ls if L >= expected_lmin])

    if run_start is None:
        run_cells = 0
        mean_n_clean_run = 0.0
    else:
        run_cells = sum(range(run_start, run_end + 1))
        mean_n_clean_run = (
            pd.DataFrame(per_L_rows)
            .query("@run_start <= L <= @run_end")["mean_n_clean_L"]
            .mean()
        )

    full_cells = sum(range(expected_lmin, max_L + 1)) if max_L >= expected_lmin else 0
    mean_n_clean_all = float(df["n_clean"].mean()) if len(df) else 0.0

    return {
        "question_id": question_id,
        "file_stem": path.stem,
        "csv_path": str(path),
        "min_L_present": min_L,
        "max_L_present": max_L,
        "expected_lmin": expected_lmin,
        "full_uninterrupted": full_uninterrupted,
        "longest_run_start": run_start,
        "longest_run_end": run_end,
        "longest_run_len": run_len,
        "run_cells": int(run_cells),
        "full_cells": int(full_cells),
        "n_complete_L": int(len(complete_Ls)),
        "mean_n_clean_run": float(mean_n_clean_run) if pd.notna(mean_n_clean_run) else 0.0,
        "mean_n_clean_all": mean_n_clean_all,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select T1-B per-question files, prioritising uninterrupted trajectories over coverage."
    )
    parser.add_argument(
        "--indir",
        type=str,
        default="t1b_splits",
        help="Directory containing split CSV files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of selected questions to output.",
    )
    parser.add_argument(
        "--expected-lmin",
        type=int,
        default=3,
        help="Minimum L that must be included in the uninterrupted run.",
    )
    parser.add_argument(
        "--only-ok",
        action="store_true",
        help="Only use rows with bin_status == ok.",
    )
    args = parser.parse_args()

    indir = Path(args.indir).resolve()
    files = sorted(indir.glob("*.csv"))

    rows: list[dict] = []
    for path in files:
        if path.name.startswith("_"):
            continue
        if is_overall_file(path):
            continue

        result = analyse_file(path, expected_lmin=args.expected_lmin, only_ok=args.only_ok)
        if result is not None:
            rows.append(result)

    if not rows:
        raise ValueError("No per-question CSV files were found or no usable files remained.")

    summary = pd.DataFrame(rows)

    good = summary[summary["full_uninterrupted"]].copy()
    good = good.sort_values(
        by=[
            "max_L_present",
            "full_cells",
            "mean_n_clean_run",
            "question_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )

    fallback = summary[~summary["full_uninterrupted"]].copy()
    fallback = fallback.sort_values(
        by=[
            "longest_run_len",
            "longest_run_end",
            "run_cells",
            "mean_n_clean_run",
            "question_id",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )

    selected = pd.concat(
        [
            good,
            fallback,
        ],
        axis=0,
        ignore_index=True,
    ).head(args.top_k)

    summary.to_csv(indir / "_selected_summary_v2.csv", index=False)
    good.to_csv(indir / "_selected_good_v2.csv", index=False)
    selected.to_csv(indir / "_selected_questions_v2.csv", index=False)

    print(f"Input directory: {indir}")
    print(f"Total per-question files analysed: {len(summary)}")
    print(f"Fully uninterrupted candidates: {len(good)}")
    print(f"Selected: {len(selected)}")
    print(f"Summary: {indir / '_selected_summary_v2.csv'}")
    print(f"Good-only: {indir / '_selected_good_v2.csv'}")
    print(f"Selected file: {indir / '_selected_questions_v2.csv'}")


if __name__ == "__main__":
    main()