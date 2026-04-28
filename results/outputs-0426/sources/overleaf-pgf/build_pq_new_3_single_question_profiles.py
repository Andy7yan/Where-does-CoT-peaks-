from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = ROOT / "results" / "outputs-0426" / "overleaf-pgf"
SOURCE_DIR = OUT_ROOT / "source-0426"
SINGLE_SRC_DIR = OUT_ROOT / "pq_new_3_single_question_profiles"
SINGLE_DATA_DIR = SINGLE_SRC_DIR / "data"

PROFILE_RE = re.compile(r"fig_new_3_profile_(gsm8k_platinum_\d{4})_L(\d+)\.csv$")


def longest_runs(values: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None
    for value in sorted(set(values)):
        if previous is None or value == previous + 1:
            current.append(value)
        else:
            runs.append(current)
            current = [value]
        previous = value
    if current:
        runs.append(current)
    return runs


def profile_metrics(path: Path, question_id: str, length: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["mean_nldd"] = pd.to_numeric(frame["mean_nldd"], errors="coerce")
    frame["nldd_se"] = pd.to_numeric(frame["nldd_se"], errors="coerce")
    valid = frame[frame["mean_nldd"].notna() & frame["step"].between(2, length)].copy()
    if valid.empty:
        return {
            "question_id": question_id,
            "L": length,
            "path": path,
            "valid_points": 0,
            "coverage": 0.0,
            "peak_step": -1,
            "y_min": math.nan,
            "y_max": math.nan,
            "y_range": math.nan,
            "k_star": -1,
            "valid_bin": False,
        }
    values = valid["mean_nldd"].to_numpy(dtype=float)
    peak_step = int(valid.iloc[int(np.nanargmax(values))]["step"])
    k_star = int(frame["k_star"].dropna().iloc[0]) if "k_star" in frame and frame["k_star"].notna().any() else -1
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    y_range = y_max - y_min
    coverage = float(len(valid) / max(1, length - 1))
    return {
        "question_id": question_id,
        "L": length,
        "path": path,
        "valid_points": int(len(valid)),
        "coverage": coverage,
        "peak_step": peak_step,
        "y_min": y_min,
        "y_max": y_max,
        "y_range": y_range,
        "k_star": k_star,
        "valid_bin": bool(coverage >= 0.65 and len(valid) >= 3 and y_max > 20 and y_range < 2500),
    }


def discover_profiles() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DIR.glob("fig_new_3_profile_*.csv")):
        match = PROFILE_RE.match(path.name)
        if not match:
            continue
        rows.append(profile_metrics(path, match.group(1), int(match.group(2))))
    return pd.DataFrame(rows)


def choose_question(profiles: pd.DataFrame, requested_question_id: str | None = None) -> tuple[str, list[int], pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    for question_id, sub in profiles.groupby("question_id", sort=True):
        valid_lengths = [int(value) for value in sub[sub["valid_bin"]]["L"].tolist()]
        if not valid_lengths:
            continue
        best_run = max(longest_runs(valid_lengths), key=len)
        run_frame = sub[sub["L"].isin(best_run)].copy()
        mid_peak_rate = float(((run_frame["peak_step"] > 1) & (run_frame["peak_step"] < run_frame["L"])).mean())
        candidates.append(
            {
                "question_id": question_id,
                "run_len": len(best_run),
                "run_min": min(best_run),
                "run_max": max(best_run),
                "run_lengths": best_run,
                "avg_coverage": float(run_frame["coverage"].mean()),
                "avg_range": float(run_frame["y_range"].mean()),
                "mid_peak_rate": mid_peak_rate,
            }
        )
    if not candidates:
        raise RuntimeError("No valid single-question NEW-3 candidate found.")
    summary = pd.DataFrame(candidates)
    if requested_question_id is not None:
        requested = summary[summary["question_id"] == requested_question_id].copy()
        if requested.empty:
            raise RuntimeError(f"No continuous valid NEW-3 bins found for {requested_question_id}.")
        selected = requested.sort_values(
            ["run_len", "avg_coverage", "mid_peak_rate", "avg_range"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).iloc[0]
        selected_summary = summary.sort_values(
            ["run_len", "avg_coverage", "mid_peak_rate", "avg_range"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        return str(selected["question_id"]), [int(value) for value in selected["run_lengths"]], selected_summary

    eligible = summary[summary["run_len"] >= 8].copy()
    if eligible.empty:
        eligible = summary.copy()
    # Prefer stable, readable curves among long continuous runs.
    selected = eligible.sort_values(
        ["avg_range", "run_len", "avg_coverage", "mid_peak_rate", "question_id"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    question_id = str(selected["question_id"])
    run_lengths = [int(value) for value in selected["run_lengths"]]
    selected_summary = summary.sort_values(
        ["run_len", "avg_coverage", "mid_peak_rate", "avg_range"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return question_id, run_lengths, selected_summary


def clean_outputs(short_id: str | None = None) -> None:
    SINGLE_SRC_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if short_id is None:
        source_patterns = ("pq_new_3_single_*.tex", "pq_new_3_single_manifest*.csv", "pq_new_3_single_candidate_summary.csv")
        data_patterns = ("pq_new_3_single_*.csv",)
    else:
        source_patterns = (f"pq_new_3_single_{short_id}_*.tex", f"pq_new_3_single_manifest_{short_id}.csv")
        data_patterns = (f"pq_new_3_single_{short_id}_*.csv",)
    for pattern in source_patterns:
        for path in SINGLE_SRC_DIR.glob(pattern):
            if path.is_file():
                path.unlink()
    for pattern in data_patterns:
        for path in SINGLE_DATA_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def axis_limits(frames: list[pd.DataFrame]) -> tuple[float, float]:
    values: list[float] = []
    for frame in frames:
        for _, row in frame.iterrows():
            mean = float(row["mean_nldd"]) if pd.notna(row["mean_nldd"]) else math.nan
            se = float(row["nldd_se"]) if pd.notna(row["nldd_se"]) else 0.0
            if math.isfinite(mean):
                values.extend([mean - se, mean + se])
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    lo = min(0.0, float(np.nanpercentile(arr, 2)))
    hi = float(np.nanpercentile(arr, 98))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def standalone_open() -> str:
    return r"""\documentclass[border=4pt]{standalone}
\input{../peakcot_figure_preamble.tex}
\begin{document}
"""


def legend_tex() -> str:
    return r"""
\begin{tikzpicture}
\begin{axis}[
  hide axis,
  scale only axis,
  width=0pt,
  height=0pt,
  xmin=0,
  xmax=1,
  ymin=0,
  ymax=1,
  legend columns=5,
  legend cell align=left,
  legend style={draw=none, fill=none, font=\footnotesize, /tikz/every even column/.append style={column sep=8pt}},
]
  \addlegendimage{band-blue!75!black, thick, mark=*, mark size=1.3pt}
  \addlegendentry{Mean NLDD($k$)}
  \addlegendimage{band-blue, fill=band-blue, fill opacity=0.14, area legend}
  \addlegendentry{NLDD SE band}
  \addlegendimage{hard-orange, thick, densely dashed, mark=square*, mark size=1.25pt}
  \addlegendentry{Mean TAS$_t$}
  \addlegendimage{hard-orange, fill=hard-orange, fill opacity=0.10, area legend}
  \addlegendentry{TAS SE band}
  \addlegendimage{accent-red, dashed, line width=0.85pt}
  \addlegendentry{$k^*(L)$}
\end{axis}
\end{tikzpicture}
"""


def write_one_tex(
    question_id: str,
    short_id: str,
    length: int,
    csv_name: str,
    k_star: int,
    l_star: int,
    y_min: float,
    y_max: float,
) -> Path:
    x_min = 0.8
    x_max = length + 0.2
    tex = rf"""{standalone_open()}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  name=main,
  width=8.4cm,
  height=5.2cm,
  scale only axis,
  xmin={x_min:.1f},
  xmax={x_max:.1f},
  ymin={y_min:.3f},
  ymax={y_max:.3f},
  xlabel={{Step position $k$}},
  ylabel={{Mean NLDD($k$)}},
  title={{Q\texttt{{{short_id}}}: aggregate profile at $L={length}$}},
  title style={{font=\small, yshift=1pt}},
  clip=true,
]
  \addplot[draw=none, name path=nlddHi, forget plot, unbounded coords=jump]
    table[x=step, y expr={{\thisrow{{mean_nldd}}+\thisrow{{nldd_se}}}}, col sep=comma] {{data/{csv_name}}};
  \addplot[draw=none, name path=nlddLo, forget plot, unbounded coords=jump]
    table[x=step, y expr={{\thisrow{{mean_nldd}}-\thisrow{{nldd_se}}}}, col sep=comma] {{data/{csv_name}}};
  \addplot[band-blue, fill opacity=0.14, draw=none, forget plot] fill between[of=nlddHi and nlddLo];
  \addplot[band-blue!75!black, thick, mark=*, mark size=1.35pt, unbounded coords=jump]
    table[x=step, y=mean_nldd, col sep=comma] {{data/{csv_name}}};
  \addplot[accent-red, dashed, line width=0.85pt] coordinates {{({k_star},{y_min:.3f}) ({k_star},{y_max:.3f})}};
  \node[anchor=north east, align=right, inner sep=1.5pt, font=\scriptsize, text=accent-red, fill=white, fill opacity=0.78, text opacity=1]
    at (axis description cs:0.98,0.96) {{$k^*(L)={k_star}$\\$L^*={l_star}$}};
\end{{axis}}
\begin{{axis}}[
  at={{(main.south west)}},
  anchor=south west,
  width=8.4cm,
  height=5.2cm,
  scale only axis,
  xmin={x_min:.1f},
  xmax={x_max:.1f},
  ymin=0,
  ymax=1.05,
  axis y line*=right,
  axis x line=none,
  ylabel={{Mean TAS$_t$}},
  ytick={{0,0.25,0.5,0.75,1}},
  ylabel style={{text=hard-orange!85!black}},
  yticklabel style={{text=hard-orange!85!black}},
  clip=true,
]
  \addplot[draw=none, name path=tasHi, forget plot, unbounded coords=jump]
    table[x=step, y expr={{\thisrow{{mean_tas_t}}+\thisrow{{tas_t_se}}}}, col sep=comma] {{data/{csv_name}}};
  \addplot[draw=none, name path=tasLo, forget plot, unbounded coords=jump]
    table[x=step, y expr={{\thisrow{{mean_tas_t}}-\thisrow{{tas_t_se}}}}, col sep=comma] {{data/{csv_name}}};
  \addplot[hard-orange, fill opacity=0.10, draw=none, forget plot] fill between[of=tasHi and tasLo];
  \addplot[hard-orange, thick, densely dashed, mark=square*, mark size=1.25pt, unbounded coords=jump]
    table[x=step, y=mean_tas_t, col sep=comma] {{data/{csv_name}}};
\end{{axis}}
\end{{tikzpicture}}
\par\vspace{{3pt}}
{legend_tex()}
\end{{document}}
"""
    tex_path = SINGLE_SRC_DIR / f"pq_new_3_single_{short_id}_L{length:02d}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question-id",
        default=None,
        help="Optional explicit question id, e.g. gsm8k_platinum_0221.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = discover_profiles()
    question_id, lengths, candidate_summary = choose_question(profiles, args.question_id)
    short_id = question_id.rsplit("_", 1)[-1]

    lcurve_path = SOURCE_DIR / f"fig_new_1_lcurve_{question_id}.csv"
    l_star = -1
    if lcurve_path.exists():
        lcurve = pd.read_csv(lcurve_path)
        if "l_star_A" in lcurve and lcurve["l_star_A"].notna().any():
            l_star = int(lcurve["l_star_A"].dropna().iloc[0])

    clean_outputs(None if args.question_id is None else short_id)
    candidate_summary.to_csv(SINGLE_SRC_DIR / "pq_new_3_single_candidate_summary.csv", index=False)

    frames: list[pd.DataFrame] = []
    artifacts: list[dict[str, Any]] = []
    for length in lengths:
        source_path = SOURCE_DIR / f"fig_new_3_profile_{question_id}_L{length}.csv"
        frame = pd.read_csv(source_path)
        for col in ("step", "mean_nldd", "nldd_se", "mean_tas_t", "tas_t_se", "k_star"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        out_frame = frame[["step", "mean_nldd", "nldd_se", "mean_tas_t", "tas_t_se", "k_star"]].copy()
        csv_name = f"pq_new_3_single_{short_id}_L{length:02d}.csv"
        out_frame.to_csv(SINGLE_DATA_DIR / csv_name, index=False, na_rep="nan")
        frames.append(out_frame)
        k_star = int(out_frame["k_star"].dropna().iloc[0]) if out_frame["k_star"].notna().any() else -1
        artifacts.append({"L": length, "csv_name": csv_name, "k_star": k_star})

    y_min, y_max = axis_limits(frames)
    manifest_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        tex_path = write_one_tex(
            question_id=question_id,
            short_id=short_id,
            length=int(artifact["L"]),
            csv_name=str(artifact["csv_name"]),
            k_star=int(artifact["k_star"]),
            l_star=l_star,
            y_min=y_min,
            y_max=y_max,
        )
        metrics = profiles[(profiles["question_id"] == question_id) & (profiles["L"] == int(artifact["L"]))].iloc[0]
        manifest_rows.append(
            {
                "question_id": question_id,
                "short_id": short_id,
                "l_star_A": l_star,
                "L": int(artifact["L"]),
                "k_star": int(artifact["k_star"]),
                "valid_points": int(metrics["valid_points"]),
                "coverage": float(metrics["coverage"]),
                "peak_step": int(metrics["peak_step"]),
                "tex_file": str(tex_path.relative_to(OUT_ROOT)),
                "csv_file": artifact["csv_name"],
                "shared_y_min": y_min,
                "shared_y_max": y_max,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_name = "pq_new_3_single_manifest.csv" if args.question_id is None else f"pq_new_3_single_manifest_{short_id}.csv"
    manifest.to_csv(SINGLE_SRC_DIR / manifest_name, index=False)
    print(f"Selected {question_id} with continuous valid L bins: {lengths}")
    print(manifest[["question_id", "L", "k_star", "valid_points", "coverage", "peak_step"]].to_string(index=False))


if __name__ == "__main__":
    main()
