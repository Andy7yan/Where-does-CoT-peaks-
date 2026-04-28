from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = ROOT / "results" / "outputs-0426" / "overleaf-pgf"
SOURCE_DIR = OUT_ROOT / "source-0426"


PROFILE_RE = re.compile(r"fig_new_3_profile_(gsm8k_platinum_\d{4})_L(\d+)\.csv$")


def longest_run(values: list[int]) -> int:
    best = 0
    current = 0
    previous: int | None = None
    for value in sorted(set(values)):
        if previous is None or value == previous + 1:
            current += 1
        else:
            best = max(best, current)
            current = 1
        previous = value
    return max(best, current)


def read_profiles() -> dict[str, dict[int, dict[str, float]]]:
    profiles: dict[str, dict[int, dict[str, float]]] = {}
    for path in sorted(SOURCE_DIR.glob("fig_new_3_profile_*.csv")):
        match = PROFILE_RE.match(path.name)
        if not match:
            continue
        question_id = match.group(1)
        length = int(match.group(2))
        frame = pd.read_csv(path)
        frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
        row = frame[frame["step"] == length]
        if row.empty:
            row = frame.tail(1)
        if row.empty:
            continue
        item = row.iloc[0]
        profiles.setdefault(question_id, {})[length] = {
            "final_tas": float(item.get("mean_tas_t", math.nan)),
            "final_tas_se": float(item.get("tas_t_se", math.nan)),
            "k_star": float(item.get("k_star", math.nan)),
        }
    return profiles


def select_questions(profiles: dict[str, dict[int, dict[str, float]]], top_k: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(SOURCE_DIR.glob("fig_new_1_lcurve_*.csv")):
        question_id = path.stem.replace("fig_new_1_lcurve_", "")
        frame = pd.read_csv(path).sort_values("L")
        frame["L"] = pd.to_numeric(frame["L"], errors="coerce")
        frame = frame[frame["L"].notna()].copy()
        if frame.empty:
            continue
        lengths = [int(value) for value in frame["L"].tolist()]
        profile_lengths = sorted(profiles.get(question_id, {}).keys())
        if not profile_lengths:
            continue
        l_star = int(frame.iloc[0]["l_star_A"]) if pd.notna(frame.iloc[0]["l_star_A"]) else -1
        rows.append(
            {
                "question_id": question_id,
                "short_id": question_id.rsplit("_", 1)[-1],
                "difficulty": str(frame.iloc[0]["difficulty"]),
                "difficulty_score": float(frame.iloc[0]["difficulty_score"]),
                "l_min": min(lengths),
                "l_max": max(lengths),
                "accuracy_rows": len(lengths),
                "accuracy_span": max(lengths) - min(lengths) + 1,
                "mechanistic_rows": len(profile_lengths),
                "mechanistic_min": min(profile_lengths),
                "mechanistic_max": max(profile_lengths),
                "mechanistic_run": longest_run(profile_lengths),
                "l_star_A": l_star,
                "accuracy_range": float(frame["accuracy"].max() - frame["accuracy"].min()),
            }
        )
    summary = pd.DataFrame(rows)
    usable = summary[
        (summary["accuracy_rows"] >= 9)
        & (summary["mechanistic_rows"] >= 4)
        & (summary["l_star_A"] > 0)
    ].copy()
    selected = usable.sort_values(
        [
            "mechanistic_run",
            "mechanistic_rows",
            "accuracy_span",
            "accuracy_range",
            "difficulty_score",
            "question_id",
        ],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    ).head(top_k)
    selected.to_csv(SOURCE_DIR / "fig_pq_triple_selected.csv", index=False)
    summary.to_csv(SOURCE_DIR / "fig_pq_triple_selection_summary.csv", index=False)
    return selected


def write_question_csv(row: pd.Series, profiles: dict[str, dict[int, dict[str, float]]], rank: int) -> str:
    question_id = str(row["question_id"])
    short_id = str(row["short_id"])
    lcurve = pd.read_csv(SOURCE_DIR / f"fig_new_1_lcurve_{question_id}.csv").sort_values("L")
    records: list[dict[str, object]] = []
    for item in lcurve.itertuples(index=False):
        length = int(item.L)
        profile = profiles.get(question_id, {}).get(length, {})
        records.append(
            {
                "L": length,
                "accuracy": float(item.accuracy),
                "accuracy_se": float(item.se),
                "n_traces": int(item.n_traces),
                "sufficient": int(item.sufficient),
                "final_tas": profile.get("final_tas", math.nan),
                "final_tas_se": profile.get("final_tas_se", math.nan),
                "k_star": profile.get("k_star", math.nan),
                "l_star_A": int(item.l_star_A),
                "difficulty": str(item.difficulty),
                "difficulty_score": float(item.difficulty_score),
            }
        )
    csv_name = f"fig_pq_triple_{rank:02d}_{short_id}.csv"
    pd.DataFrame(records).to_csv(SOURCE_DIR / csv_name, index=False, na_rep="nan")
    return csv_name


def standalone_open(border: int = 4) -> str:
    return rf"""\ifdefined\PeakCoTMainDocument
  \def\PeakCoTEndDocument{{}}
\else
\documentclass[border={border}pt]{{standalone}}
\input{{peakcot_figure_preamble.tex}}
\begin{{document}}
  \def\PeakCoTEndDocument{{\end{{document}}}}
\fi
"""


def legend_tex(name: str = "pqLegend", columns: int = 3) -> str:
    return rf"""
\newcommand{{\DefinePQLegend}}{{%
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
  legend to name={name},
  legend columns={columns},
  legend cell align=left,
  legend style={{
    draw=none,
    fill=none,
    font=\footnotesize,
    /tikz/every even column/.append style={{column sep=9pt}},
    nodes={{inner sep=1pt}},
  }},
]
  \addlegendimage{{band-blue!75!black, thick, mark=*, mark size=1.2pt}}
  \addlegendentry{{Accuracy (all L)}}
  \addlegendimage{{band-blue!75!black, only marks, mark=o, mark size=1.5pt, mark options={{solid, fill=white}}}}
  \addlegendentry{{low-support accuracy bin}}
  \addlegendimage{{tas-green, thick, densely dashed, mark=triangle*, mark size=1.4pt}}
  \addlegendentry{{Final TAS(L)}}
  \addlegendimage{{hard-orange, thick, mark=square*, mark size=1.2pt}}
  \addlegendentry{{NLDD horizon $k^*(L)$}}
  \addlegendimage{{axis-gray, dashed, line width=0.8pt}}
  \addlegendentry{{$k^*=L$}}
  \addlegendimage{{accent-red, dashed, line width=0.8pt}}
  \addlegendentry{{behavioral $L^*$}}
\end{{axis}}
\end{{tikzpicture}}%
}}
"""


def panel_tex(
    csv_name: str,
    short_id: str,
    difficulty: str,
    difficulty_score: float,
    l_min: int,
    l_max: int,
    l_star: int,
    panel_w: str,
    panel_h: str,
    show_left_label: bool = True,
    show_right_label: bool = True,
    show_x_label: bool = True,
) -> str:
    xmin = max(0.5, l_min - 0.5)
    xmax = l_max + 0.5
    right_ymax = l_max + 0.5
    left_ylabel = "Accuracy / final TAS" if show_left_label else ""
    right_ylabel = "NLDD horizon $k^*(L)$" if show_right_label else ""
    x_label = "Reasoning length $L$" if show_x_label else ""
    left_ytick = "%" if show_left_label else "yticklabels={},"
    right_ytick = "%" if show_right_label else "yticklabels={},"
    return rf"""
\begin{{tikzpicture}}
  \begin{{axis}}[
    peakcot base,
    name=leftax,
    width={panel_w},
    height={panel_h},
    scale only axis,
    axis y line*=left,
    xmin={xmin:.1f},
    xmax={xmax:.1f},
    ymin=0,
    ymax=1.05,
    xtick distance=2,
    ytick={{0,0.25,0.5,0.75,1}},
    xlabel={{{x_label}}},
    ylabel={{{left_ylabel}}},
    {left_ytick}
    title={{Q\texttt{{{short_id}}}, {difficulty}, $d={difficulty_score:.2f}$}},
    title style={{font=\small, yshift=1pt}},
    clip=true,
  ]
    \addplot[draw=none, name path=accHi, forget plot]
      table[x=L, y expr={{min(1.05,\thisrow{{accuracy}}+\thisrow{{accuracy_se}})}}, col sep=comma] {{source-0426/{csv_name}}};
    \addplot[draw=none, name path=accLo, forget plot]
      table[x=L, y expr={{max(0,\thisrow{{accuracy}}-\thisrow{{accuracy_se}})}}, col sep=comma] {{source-0426/{csv_name}}};
    \addplot[band-blue, fill opacity=0.12, draw=none, forget plot] fill between[of=accHi and accLo];
    \addplot[band-blue!75!black, thick, mark=*, mark size=1.25pt]
      table[x=L, y=accuracy, col sep=comma] {{source-0426/{csv_name}}};
    \addplot[band-blue!75!black, only marks, mark=o, mark size=1.6pt, mark options={{solid, fill=white}},
      restrict expr to domain={{\thisrow{{sufficient}}}}{{0:0}}]
      table[x=L, y=accuracy, col sep=comma] {{source-0426/{csv_name}}};
    \addplot[draw=none, name path=tasHi, forget plot, unbounded coords=jump]
      table[x=L, y expr={{\thisrow{{final_tas}}+\thisrow{{final_tas_se}}}}, col sep=comma] {{source-0426/{csv_name}}};
    \addplot[draw=none, name path=tasLo, forget plot, unbounded coords=jump]
      table[x=L, y expr={{\thisrow{{final_tas}}-\thisrow{{final_tas_se}}}}, col sep=comma] {{source-0426/{csv_name}}};
    \addplot[tas-green, fill opacity=0.10, draw=none, forget plot] fill between[of=tasHi and tasLo];
    \addplot[tas-green, thick, densely dashed, mark=triangle*, mark size=1.4pt, unbounded coords=jump]
      table[x=L, y=final_tas, col sep=comma] {{source-0426/{csv_name}}};
    \ifdim {l_star}pt>0pt
      \addplot[accent-red, dashed, line width=0.8pt] coordinates {{({l_star},0) ({l_star},1.05)}};
    \fi
  \end{{axis}}
  \begin{{axis}}[
    at={{(leftax.south west)}},
    anchor=south west,
    width={panel_w},
    height={panel_h},
    scale only axis,
    axis y line*=right,
    axis x line=none,
    xmin={xmin:.1f},
    xmax={xmax:.1f},
    ymin=0,
    ymax={right_ymax:.1f},
    ytick distance=2,
    ylabel={{{right_ylabel}}},
    {right_ytick}
    clip=true,
  ]
    \addplot[axis-gray, dashed, line width=0.8pt, domain={xmin:.1f}:{xmax:.1f}, samples=2, forget plot] {{x}};
    \addplot[hard-orange, thick, mark=square*, mark size=1.25pt, unbounded coords=jump, forget plot]
      table[x=L, y=k_star, col sep=comma] {{source-0426/{csv_name}}};
  \end{{axis}}
\end{{tikzpicture}}%
"""


def write_single_tex(row: pd.Series, rank: int, csv_name: str) -> None:
    short_id = str(row["short_id"])
    text = (
        standalone_open()
        + legend_tex(columns=3)
        + "\n\\centering\n\\DefinePQLegend\n"
        + panel_tex(
            csv_name=csv_name,
            short_id=short_id,
            difficulty=str(row["difficulty"]),
            difficulty_score=float(row["difficulty_score"]),
            l_min=int(row["l_min"]),
            l_max=int(row["l_max"]),
            l_star=int(row["l_star_A"]),
            panel_w="7.2cm",
            panel_h="5.2cm",
        )
        + "\n\\par\\vspace{3pt}\n\\pgfplotslegendfromname{pqLegend}\n\n\\PeakCoTEndDocument\n"
    )
    (OUT_ROOT / f"fig_pq_triple_{rank:02d}_{short_id}.tex").write_text(text, encoding="utf-8")


def write_overview_tex(selected: pd.DataFrame, csv_names: list[str]) -> None:
    panels: list[str] = []
    for idx, (_, row) in enumerate(selected.head(3).iterrows()):
        panels.append(
            panel_tex(
                csv_name=csv_names[idx],
                short_id=str(row["short_id"]),
                difficulty=str(row["difficulty"]),
                difficulty_score=float(row["difficulty_score"]),
                l_min=int(row["l_min"]),
                l_max=int(row["l_max"]),
                l_star=int(row["l_star_A"]),
                panel_w="5.2cm",
                panel_h="3.7cm",
                show_left_label=idx == 0,
                show_right_label=idx == 2,
                show_x_label=True,
            )
        )
    text = (
        standalone_open()
        + legend_tex(columns=3)
        + "\n\\centering\n\\DefinePQLegend\n"
        + "\\begin{tabular}{@{}c@{\\hspace{5pt}}c@{\\hspace{5pt}}c@{}}\n"
        + "\n&\n".join(panels)
        + "\n\\\\[3pt]\n\\multicolumn{3}{c}{\\pgfplotslegendfromname{pqLegend}}\n\\end{tabular}\n"
        + "\n\\PeakCoTEndDocument\n"
    )
    (OUT_ROOT / "fig_rev_5_t1a_triple_metric.tex").write_text(text, encoding="utf-8")


def main() -> None:
    profiles = read_profiles()
    selected = select_questions(profiles, top_k=10)
    csv_names: list[str] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        csv_name = write_question_csv(row, profiles, rank)
        csv_names.append(csv_name)
        write_single_tex(row, rank, csv_name)
    write_overview_tex(selected, csv_names)
    print("Selected per-question triple figures:")
    print(selected[["question_id", "difficulty", "l_min", "l_max", "mechanistic_rows", "mechanistic_run", "l_star_A"]].to_string(index=False))


if __name__ == "__main__":
    main()
