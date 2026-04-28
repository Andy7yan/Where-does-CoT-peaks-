from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = ROOT / "results" / "outputs-0426" / "overleaf-pgf"
SOURCE_DIR = OUT_ROOT / "source-0426"
LOCAL_FIG_DIR = ROOT / "results" / "outputs-0426" / "local-figures" / "overleaf-pgf"
FIG_STEM = "fig_rev_6_kstar_ratio_violin_by_difficulty"
GLOBAL_MEDIAN = 0.643


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="nan")


def fmt_num(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def fmt_p(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def discover_t1c() -> Path:
    candidates: list[Path] = []
    scratch = os.environ.get("SCRATCH")
    if scratch:
        scratch_root = Path(scratch)
        for pattern in ("runs/*/analysis/t1c_kstar_ratio.csv", "runs/*/pq_analysis/t1c_kstar_ratio.csv"):
            candidates.extend(path for path in scratch_root.glob(pattern) if path.is_file())
    candidates.extend(
        path
        for path in [
            ROOT / "results" / "papery-pq" / "pq_analysis" / "t1c_kstar_ratio.csv",
            ROOT / "results" / "outputs-0422" / "sources" / "t1c.csv",
        ]
        if path.is_file()
    )
    candidates.extend((ROOT / "results" / "outputs-0423" / "sources").glob("**/t1c_kstar_ratio.csv"))
    if not candidates:
        raise FileNotFoundError("Could not locate t1c_kstar_ratio.csv or compatible t1c.csv")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_t1c(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"question_id", "difficulty_score", "L", "k_star", "k_star_ratio", "n_clean"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    work = frame.loc[:, ["question_id", "difficulty_score", "L", "k_star", "k_star_ratio", "n_clean"]].copy()
    work["question_id"] = work["question_id"].astype(str)
    for col in ["difficulty_score", "L", "k_star", "k_star_ratio", "n_clean"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["difficulty_score", "k_star_ratio"])
    work["difficulty_group"] = np.where(work["difficulty_score"] < 0.5, "medium", "hard")
    return work


def make_violin_outline(values: np.ndarray, xpos: float) -> pd.DataFrame:
    grid = np.linspace(0.25, 1.0, 160)
    if len(values) >= 2 and float(np.std(values)) > 1e-8:
        density = gaussian_kde(values)(grid)
    else:
        density = np.ones_like(grid)
    width = 0.34 * density / float(np.max(density))
    left = pd.DataFrame({"x": xpos - width, "y": grid})
    right = pd.DataFrame({"x": xpos + width[::-1], "y": grid[::-1]})
    return pd.concat([left, right], ignore_index=True)


def make_box_commands(stats_frame: pd.DataFrame) -> str:
    lines: list[str] = []
    for row in stats_frame.itertuples(index=False):
        x = float(row.group_x)
        lines.extend(
            [
                rf"\draw[black, line width=0.35pt] (axis cs:{x:.3f},{row.whisker_low:.6f}) -- (axis cs:{x:.3f},{row.q1:.6f});",
                rf"\draw[black, line width=0.35pt] (axis cs:{x:.3f},{row.q3:.6f}) -- (axis cs:{x:.3f},{row.whisker_high:.6f});",
                rf"\draw[black, fill=white, fill opacity=0.85, line width=0.45pt] (axis cs:{x-0.09:.3f},{row.q1:.6f}) rectangle (axis cs:{x+0.09:.3f},{row.q3:.6f});",
                rf"\draw[accent-red, line width=0.9pt] (axis cs:{x-0.12:.3f},{row.median:.6f}) -- (axis cs:{x+0.12:.3f},{row.median:.6f});",
            ]
        )
    return "\n  ".join(lines)


def export_sources(t1c: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    group_order = [("medium", 1.0), ("hard", 2.0)]
    rng = np.random.default_rng(42)
    strip_rows: list[dict[str, float | str | int]] = []
    stats_rows: list[dict[str, float | str | int]] = []

    write_csv(SOURCE_DIR / f"{FIG_STEM}_source.csv", t1c)
    for group, xpos in group_order:
        values = t1c.loc[t1c["difficulty_group"] == group, "k_star_ratio"].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        for value, dx in zip(values, jitter):
            strip_rows.append({"difficulty_group": group, "group_x": xpos, "x_jitter": xpos + float(dx), "k_star_ratio": value})

        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        stats_rows.append(
            {
                "difficulty_group": group,
                "group_x": xpos,
                "n": int(len(values)),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "whisker_low": max(float(values.min()), float(q1 - 1.5 * iqr)),
                "whisker_high": min(float(values.max()), float(q3 + 1.5 * iqr)),
            }
        )
        write_csv(SOURCE_DIR / f"{FIG_STEM}_violin_{group}.csv", make_violin_outline(values, xpos))

    strip = pd.DataFrame(strip_rows)
    box_stats = pd.DataFrame(stats_rows)
    write_csv(SOURCE_DIR / f"{FIG_STEM}_strip.csv", strip)
    write_csv(SOURCE_DIR / f"{FIG_STEM}_box_stats.csv", box_stats)

    medium = t1c.loc[t1c["difficulty_group"] == "medium", "k_star_ratio"].to_numpy(dtype=float)
    hard = t1c.loc[t1c["difficulty_group"] == "hard", "k_star_ratio"].to_numpy(dtype=float)
    mw = stats.mannwhitneyu(medium, hard, alternative="two-sided")
    summary = box_stats.copy()
    summary["mann_whitney_u"] = float(mw.statistic)
    summary["mann_whitney_p"] = float(mw.pvalue)
    summary["global_median_line"] = GLOBAL_MEDIAN
    write_csv(SOURCE_DIR / f"{FIG_STEM}_summary.csv", summary)
    return summary, make_box_commands(box_stats)


def write_preamble() -> None:
    write_text(
        OUT_ROOT / "peakcot_figure_preamble.tex",
        r"""% Shared preamble for peak-CoT pgfplots figures.
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


def write_tex(summary: pd.DataFrame, box_commands: str) -> None:
    stats_by_group = summary.set_index("difficulty_group")
    medium = stats_by_group.loc["medium"]
    hard = stats_by_group.loc["hard"]
    p_value = float(medium["mann_whitney_p"])
    u_value = float(medium["mann_whitney_u"])
    annotation = (
        rf"Median: medium={fmt_num(float(medium['median']))}, hard={fmt_num(float(hard['median']))}; "
        rf"Mann--Whitney $p={fmt_p(p_value)}$"
    )
    write_text(
        OUT_ROOT / f"{FIG_STEM}.tex",
        rf"""\ifdefined\PeakCoTMainDocument
\else
\documentclass[border=3pt]{{standalone}}
\input{{peakcot_figure_preamble.tex}}
\begin{{document}}
\fi
\newcommand{{\DataDir}}{{source-0426}}
\begin{{tikzpicture}}
\begin{{axis}}[
  peakcot base,
  width=0.66\textwidth,
  height=0.48\textwidth,
  xmin=0.45,
  xmax=2.55,
  ymin=0.25,
  ymax=1.02,
  xtick={{1,2}},
  xticklabels={{Medium,Hard}},
  ylabel={{Relative horizon $k^*/L$}},
  title={{{annotation}}},
  legend columns=3,
  legend style={{at={{(0.5,-0.20)}}, anchor=north, column sep=8pt}},
]
  \addplot[draw=medium-teal!70!black, fill=medium-teal, fill opacity=0.16, line width=0.45pt]
    table[x=x, y=y, col sep=comma] {{\DataDir/{FIG_STEM}_violin_medium.csv}} \closedcycle;
  \addlegendentry{{Medium}}
  \addplot[draw=hard-orange!75!black, fill=hard-orange, fill opacity=0.18, line width=0.45pt]
    table[x=x, y=y, col sep=comma] {{\DataDir/{FIG_STEM}_violin_hard.csv}} \closedcycle;
  \addlegendentry{{Hard}}
  \addplot+[only marks, mark=*, mark size=0.72pt, draw=none, fill=gray!45, opacity=0.42]
    table[x=x_jitter, y=k_star_ratio, col sep=comma] {{\DataDir/{FIG_STEM}_strip.csv}};
  \addplot[gray!65, dashed, line width=0.8pt, forget plot] coordinates {{(0.50,{GLOBAL_MEDIAN:.3f}) (2.50,{GLOBAL_MEDIAN:.3f})}};
  \addlegendimage{{gray!65, dashed, line width=0.8pt}}
  \addlegendentry{{Global median = {GLOBAL_MEDIAN:.3f}}}
  {box_commands}
\end{{axis}}
\end{{tikzpicture}}
\ifdefined\PeakCoTMainDocument
\else
\end{{document}}
\fi
""",
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

    t1c_path = discover_t1c()
    t1c = normalize_t1c(t1c_path)
    summary, box_commands = export_sources(t1c)
    write_preamble()
    write_tex(summary, box_commands)

    print(f"source: {t1c_path.relative_to(ROOT)}")
    for row in summary.itertuples(index=False):
        print(
            f"{row.difficulty_group.title()}: n={int(row.n)}, "
            f"median={float(row.median):.6f}, IQR=[{float(row.q1):.6f}, {float(row.q3):.6f}]"
        )
    first = summary.iloc[0]
    print(f"Mann-Whitney U={float(first['mann_whitney_u']):.1f}, p={float(first['mann_whitney_p']):.6g}")
    print(f"wrote: {OUT_ROOT / (FIG_STEM + '.tex')}")


if __name__ == "__main__":
    main()
