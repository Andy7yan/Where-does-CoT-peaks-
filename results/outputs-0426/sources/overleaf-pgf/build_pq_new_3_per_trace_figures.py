from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = ROOT / "results" / "outputs-0426" / "overleaf-pgf"
SOURCE_DIR = OUT_ROOT / "source-0426"
TRACE_SRC_DIR = OUT_ROOT / "pq_new_3_nldd_per_trace"
TRACE_DATA_DIR = TRACE_SRC_DIR / "data"
PQ_ANALYSIS = ROOT / "results" / "papery-pq" / "pq_analysis"
NLD_D_PATH = PQ_ANALYSIS / "nldd_per_trace.jsonl"
TAS_PATH = PQ_ANALYSIS / "tas_curve_per_trace.jsonl"


def load_jsonl_filtered(path: Path, selected_questions: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("question_id")) in selected_questions:
                rows.append(row)
    return rows


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def choose_length_and_trace(
    question_row: pd.Series,
    nldd: pd.DataFrame,
    tas: pd.DataFrame,
) -> dict[str, Any]:
    question_id = str(question_row["question_id"])
    l_star = int(question_row["l_star_A"])
    q_nldd = nldd[nldd["question_id"] == question_id].copy()
    q_tas = tas[tas["question_id"] == question_id].copy()
    if q_nldd.empty or q_tas.empty:
        raise ValueError(f"No per-trace rows for {question_id}")

    candidates: list[dict[str, Any]] = []
    for length, sub in q_nldd.groupby("length", sort=True):
        length = int(length)
        tas_sub = q_tas[q_tas["length"] == length]
        nldd_sample_counts = (
            sub.dropna(subset=["nldd_value"])
            .query("measurement_exclusion_reason.isna()", engine="python")
            .groupby(["sample_id", "source_trace_id"], as_index=False)
            .agg(
                valid_nldd=("k", "nunique"),
                k_star_trace=("k_star_trace", "first"),
                trace_tier=("trace_tier", "first"),
                max_abs_nldd=("nldd_value", lambda s: float(np.nanmax(np.abs(s.to_numpy(dtype=float))))),
                nldd_range=("nldd_value", lambda s: float(np.nanmax(s.to_numpy(dtype=float)) - np.nanmin(s.to_numpy(dtype=float)))),
            )
        )
        tas_sample_counts = (
            tas_sub.dropna(subset=["tas_value"])
            .groupby(["sample_id", "source_trace_id"], as_index=False)
            .agg(valid_tas=("step_index", "nunique"))
        )
        sample_counts = nldd_sample_counts.merge(tas_sample_counts, on=["sample_id", "source_trace_id"], how="left")
        if sample_counts.empty:
            continue
        sample_counts["valid_tas"] = sample_counts["valid_tas"].fillna(0)
        sample_counts["coverage_ratio"] = sample_counts["valid_nldd"] / max(1, length - 1)
        sample_counts["tas_ratio"] = sample_counts["valid_tas"] / max(1, length)
        full = sample_counts[
            (sample_counts["valid_nldd"] >= max(1, length - 1))
            & (sample_counts["valid_tas"] >= length)
        ].copy()
        usable = sample_counts.copy()
        usable["coverage_score"] = usable["coverage_ratio"] + 0.15 * usable["tas_ratio"]
        usable["quality_score"] = (
            usable["coverage_score"]
            - 0.0005 * usable["max_abs_nldd"].fillna(0)
            + 0.0001 * usable["nldd_range"].fillna(0)
        )
        best_sample = usable.sort_values(
            ["quality_score", "coverage_ratio", "valid_nldd", "tas_ratio", "valid_tas", "sample_id"],
            ascending=[False, False, False, False, False, True],
            kind="mergesort",
        ).iloc[0]
        candidates.append(
            {
                "length": length,
                "sample_count": int(len(sample_counts)),
                "full_sample_count": int(len(full)),
                "best_sample_id": str(best_sample["sample_id"]),
                "best_source_trace_id": str(best_sample["source_trace_id"]),
                "best_valid_nldd": int(best_sample["valid_nldd"]),
                "best_valid_tas": int(best_sample["valid_tas"]),
                "best_coverage_ratio": float(best_sample["coverage_ratio"]),
                "best_tas_ratio": float(best_sample["tas_ratio"]),
                "best_k_star_trace": finite_float(best_sample["k_star_trace"]),
                "best_trace_tier": int(best_sample["trace_tier"]),
            }
        )
    if not candidates:
        raise ValueError(f"No usable trace candidates for {question_id}")
    frame = pd.DataFrame(candidates)
    target_length = max(l_star, 8)
    frame["selection_target_length"] = target_length
    frame["target_distance"] = (frame["length"] - target_length).abs()
    frame["lstar_distance"] = (frame["length"] - l_star).abs()
    eligible = frame[(frame["sample_count"] >= 5) & (frame["best_coverage_ratio"] >= 0.5)].copy()
    if eligible.empty:
        eligible = frame.copy()
    eligible = eligible.sort_values(
        [
            "target_distance",
            "best_coverage_ratio",
            "sample_count",
            "best_valid_nldd",
            "best_tas_ratio",
            "length",
        ],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    )
    return eligible.iloc[0].to_dict()


def axis_limits(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(vals, 2))
    hi = float(np.nanpercentile(vals, 98))
    lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def write_trace_csvs(
    rank: int,
    question_row: pd.Series,
    choice: dict[str, Any],
    nldd: pd.DataFrame,
    tas: pd.DataFrame,
) -> dict[str, Any]:
    question_id = str(question_row["question_id"])
    short_id = f"{int(question_row['short_id']):04d}"
    length = int(choice["length"])
    highlight_sample_id = str(choice["best_sample_id"])
    highlight_source_trace_id = str(choice["best_source_trace_id"])

    q_nldd = nldd[(nldd["question_id"] == question_id) & (nldd["length"] == length)].copy()
    q_nldd = q_nldd[
        q_nldd["nldd_value"].notna()
        & q_nldd["measurement_exclusion_reason"].isna()
        & q_nldd["k"].between(2, length)
    ].copy()
    sample_keys = (
        q_nldd[["sample_id", "source_trace_id"]]
        .drop_duplicates()
        .sort_values(["sample_id", "source_trace_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    sample_keys["sample_idx"] = np.arange(1, len(sample_keys) + 1)
    q_nldd = q_nldd.merge(sample_keys, on=["sample_id", "source_trace_id"], how="left")
    q_nldd["is_highlight"] = (
        (q_nldd["sample_id"].astype(str) == highlight_sample_id)
        & (q_nldd["source_trace_id"].astype(str) == highlight_source_trace_id)
    ).astype(int)
    nldd_out = q_nldd[
        [
            "sample_idx",
            "sample_id",
            "source_trace_id",
            "k",
            "nldd_value",
            "k_star_trace",
            "is_highlight",
        ]
    ].sort_values(["sample_idx", "k"], kind="mergesort")

    q_tas = tas[
        (tas["question_id"] == question_id)
        & (tas["length"] == length)
        & (tas["sample_id"].astype(str) == highlight_sample_id)
        & (tas["source_trace_id"].astype(str) == highlight_source_trace_id)
    ].copy()
    q_tas = q_tas[q_tas["step_index"].between(1, length)].copy()
    tas_out = q_tas[["step_index", "tas_value"]].sort_values("step_index", kind="mergesort")

    TRACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"pq_new_3_trace_{rank:02d}_{short_id}_L{length}"
    nldd_csv = TRACE_DATA_DIR / f"{stem}_nldd.csv"
    tas_csv = TRACE_DATA_DIR / f"{stem}_tas.csv"
    nldd_out.to_csv(nldd_csv, index=False, na_rep="nan")
    tas_out.to_csv(tas_csv, index=False, na_rep="nan")

    y_min, y_max = axis_limits(nldd_out["nldd_value"])
    selected_nldd = nldd_out[nldd_out["is_highlight"] == 1]
    selected_k_star = int(selected_nldd["k_star_trace"].dropna().iloc[0]) if selected_nldd["k_star_trace"].notna().any() else int(choice["best_k_star_trace"])
    return {
        "stem": stem,
        "nldd_csv": nldd_csv.name,
        "tas_csv": tas_csv.name,
        "sample_count": int(sample_keys["sample_idx"].nunique()),
        "sample_indices": ",".join(str(int(v)) for v in sample_keys["sample_idx"].tolist()),
        "selected_k_star": selected_k_star,
        "highlight_sample_id": highlight_sample_id,
        "highlight_source_trace_id": highlight_source_trace_id,
        "y_min": y_min,
        "y_max": y_max,
    }


def standalone_open() -> str:
    return r"""\documentclass[border=4pt]{standalone}
\input{../peakcot_figure_preamble.tex}
\begin{document}
"""


def write_tex(rank: int, question_row: pd.Series, choice: dict[str, Any], artifacts: dict[str, Any]) -> Path:
    question_id = str(question_row["question_id"])
    short_id = f"{int(question_row['short_id']):04d}"
    difficulty = str(question_row["difficulty"])
    dscore = float(question_row["difficulty_score"])
    length = int(choice["length"])
    l_star = int(question_row["l_star_A"])
    x_min = 0.8
    x_max = length + 0.2
    tex = rf"""{standalone_open()}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  name=main,
  width=9.0cm,
  height=5.6cm,
  scale only axis,
  xmin={x_min:.1f},
  xmax={x_max:.1f},
  ymin={artifacts["y_min"]:.3f},
  ymax={artifacts["y_max"]:.3f},
  xlabel={{Step position $k$}},
  ylabel={{Per-trace NLDD($k$)}},
  title={{Q\texttt{{{short_id}}}, $L={length}$, {difficulty}, $d={dscore:.2f}$}},
  title style={{font=\small, yshift=1pt}},
  clip=true,
]
  \foreach \sid in {{{artifacts["sample_indices"]}}} {{
    \addplot[gray!62, line width=0.50pt, opacity=0.72, mark=none, forget plot, unbounded coords=jump]
      table[x=k, y=nldd_value, col sep=comma, restrict expr to domain={{\thisrow{{sample_idx}}}}{{\sid:\sid}}] {{data/{artifacts["nldd_csv"]}}};
  }}
  \addplot[band-blue!75!black, line width=1.35pt, mark=*, mark size=1.6pt, unbounded coords=jump]
    table[x=k, y=nldd_value, col sep=comma, restrict expr to domain={{\thisrow{{is_highlight}}}}{{1:1}}] {{data/{artifacts["nldd_csv"]}}};
  \addplot[accent-red, dashed, line width=0.85pt] coordinates {{({artifacts["selected_k_star"]},{artifacts["y_min"]:.3f}) ({artifacts["selected_k_star"]},{artifacts["y_max"]:.3f})}};
  \ifdim {l_star}pt>0pt
    \node[anchor=north east, align=right, inner sep=1.5pt, font=\scriptsize, text=accent-red, fill=white, fill opacity=0.78, text opacity=1]
      at (axis description cs:0.98,0.96) {{$k^*={artifacts["selected_k_star"]}$, $L^*={l_star}$\\{artifacts["sample_count"]} retained traces}};
  \fi
\end{{axis}}
\begin{{axis}}[
  at={{(main.south west)}},
  anchor=south west,
  width=9.0cm,
  height=5.6cm,
  scale only axis,
  xmin={x_min:.1f},
  xmax={x_max:.1f},
  ymin=0,
  ymax=1.05,
  axis y line*=right,
  axis x line=none,
  ylabel={{Example-trace TAS$_t$}},
  ytick={{0,0.25,0.5,0.75,1}},
  ylabel style={{text=hard-orange!85!black}},
  yticklabel style={{text=hard-orange!85!black}},
  clip=true,
]
  \addplot[hard-orange, thick, densely dashed, mark=square*, mark size=1.35pt]
    table[x=step_index, y=tas_value, col sep=comma] {{data/{artifacts["tas_csv"]}}};
\end{{axis}}
\end{{tikzpicture}}
\par\vspace{{3pt}}
\begin{{tikzpicture}}
\begin{{axis}}[
  hide axis,
  scale only axis,
  width=0pt,
  height=0pt,
  xmin=0,
  xmax=1,
  ymin=0,
  ymax=1,
  legend columns=4,
  legend cell align=left,
  legend style={{draw=none, fill=none, font=\footnotesize, /tikz/every even column/.append style={{column sep=8pt}}}},
]
  \addlegendimage{{gray!62, line width=0.75pt}}
  \addlegendentry{{Other retained traces: NLDD($k$)}}
  \addlegendimage{{band-blue!75!black, line width=1.35pt, mark=*, mark size=1.4pt}}
  \addlegendentry{{Example trace: NLDD($k$)}}
  \addlegendimage{{hard-orange, thick, densely dashed, mark=square*, mark size=1.25pt}}
  \addlegendentry{{Example trace: TAS$_t$}}
  \addlegendimage{{accent-red, dashed, line width=0.85pt}}
  \addlegendentry{{Example trace: $k^*$}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    TRACE_SRC_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TRACE_SRC_DIR / f"{artifacts['stem']}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def clean_generated_outputs() -> None:
    TRACE_SRC_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("pq_new_3_trace_*.tex", "pq_new_3_per_trace_manifest.csv"):
        for path in TRACE_SRC_DIR.glob(pattern):
            if path.is_file():
                path.unlink()
    for pattern in ("pq_new_3_trace_*.csv",):
        for path in TRACE_DATA_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    clean_generated_outputs()
    selected = pd.read_csv(SOURCE_DIR / "fig_pq_triple_selected.csv")
    selected_questions = set(selected["question_id"].astype(str))
    nldd = pd.DataFrame(load_jsonl_filtered(NLD_D_PATH, selected_questions))
    tas = pd.DataFrame(load_jsonl_filtered(TAS_PATH, selected_questions))
    for frame, cols in (
        (nldd, ["length", "k", "nldd_value"]),
        (tas, ["length", "step_index", "tas_value"]),
    ):
        for col in cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("sample_id", "source_trace_id", "question_id"):
        if col in nldd.columns:
            nldd[col] = nldd[col].astype(str)
        if col in tas.columns:
            tas[col] = tas[col].astype(str)

    manifest_rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        choice = choose_length_and_trace(row, nldd, tas)
        artifacts = write_trace_csvs(rank, row, choice, nldd, tas)
        tex_path = write_tex(rank, row, choice, artifacts)
        manifest_rows.append(
            {
                "rank": rank,
                "question_id": row["question_id"],
                "short_id": row["short_id"],
                "difficulty": row["difficulty"],
                "difficulty_score": row["difficulty_score"],
                "l_star_A": row["l_star_A"],
                "selection_target_L": int(choice["selection_target_length"]),
                "chosen_L": int(choice["length"]),
                "retained_trace_count": artifacts["sample_count"],
                "highlight_sample_id": artifacts["highlight_sample_id"],
                "highlight_source_trace_id": artifacts["highlight_source_trace_id"],
                "highlight_k_star_trace": artifacts["selected_k_star"],
                "highlight_valid_nldd": int(choice["best_valid_nldd"]),
                "highlight_valid_tas": int(choice["best_valid_tas"]),
                "tex_file": str(tex_path.relative_to(OUT_ROOT)),
                "nldd_csv": artifacts["nldd_csv"],
                "tas_csv": artifacts["tas_csv"],
                "y_min": artifacts["y_min"],
                "y_max": artifacts["y_max"],
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(TRACE_SRC_DIR / "pq_new_3_per_trace_manifest.csv", index=False)
    print(manifest[["rank", "question_id", "chosen_L", "retained_trace_count", "highlight_sample_id", "highlight_k_star_trace"]].to_string(index=False))


if __name__ == "__main__":
    main()
