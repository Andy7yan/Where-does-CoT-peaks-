"""Export T2-A/T2-B single-question plots into results/final_outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_PQ_RUN_DIR = Path("results/per-question-0419_143551")
DEFAULT_OUTPUT_ROOT = Path("results/final_outputs")
PQ_ANALYSIS_DIRNAME = "pq_analysis"
PER_QUESTION_DIRNAME = "per_question"


@dataclass(frozen=True)
class QuestionSelection:
    question_id: str
    bin_length: int
    n_retained: int
    difficulty_score: float
    l_star_a: int
    l_star_s: int
    l_star_consistent: bool


@dataclass(frozen=True)
class T2ASelection:
    question_id: str
    difficulty_score: float
    l_star_a: int
    l_star_s: int
    l_star_consistent: bool


def main() -> None:
    args = parse_args()
    pq_run_dir = Path(args.pq_run_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.random_t2a_count > 0:
        batch_artifacts = export_random_t2a_batch(
            pq_run_dir=pq_run_dir,
            output_dir=output_root / "t2a_random5",
            count=args.random_t2a_count,
            seed=args.random_seed,
            require_no_gap=args.require_no_gap,
            require_all_ok=args.require_all_ok_for_t2a,
        )
        print(f"random_t2a_count: {batch_artifacts['selected_count']}")
        print(f"random_t2a_manifest: {batch_artifacts['manifest_path']}")

    if args.t2b_all_bins:
        all_bins_artifacts = export_t2b_all_bins(
            pq_run_dir=pq_run_dir,
            output_dir=output_root / "t2b_all_bins",
            question_id=args.t2b_all_bins_question_id,
        )
        print(f"t2b_all_bins_question_id: {all_bins_artifacts['question_id']}")
        print(f"t2b_all_bins_manifest: {all_bins_artifacts['manifest_path']}")

    selection = resolve_question_selection(
        pq_run_dir=pq_run_dir,
        question_id=args.question_id,
        bin_length=args.bin_length,
    )

    t2a_dir = output_root / "t2a"
    t2b_dir = output_root / "t2b"
    t2a_dir.mkdir(parents=True, exist_ok=True)
    t2b_dir.mkdir(parents=True, exist_ok=True)

    t2a_artifacts = export_t2a(
        pq_run_dir=pq_run_dir,
        selection=selection,
        output_dir=t2a_dir,
    )
    t2b_artifacts = export_t2b(
        pq_run_dir=pq_run_dir,
        selection=selection,
        output_dir=t2b_dir,
        allow_partial=args.allow_partial,
    )

    summary = {
        "pq_run_dir": str(pq_run_dir),
        "question_id": selection.question_id,
        "bin_length": selection.bin_length,
        "n_retained": selection.n_retained,
        "difficulty_score": selection.difficulty_score,
        "l_star_a": selection.l_star_a,
        "l_star_s": selection.l_star_s,
        "l_star_consistent": selection.l_star_consistent,
        "t2a": t2a_artifacts,
        "t2b": t2b_artifacts,
    }
    summary_path = output_root / "t2_single_question_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"question_id: {selection.question_id}")
    print(f"bin_length: {selection.bin_length}")
    print(f"t2a_status: {t2a_artifacts['status']}")
    print(f"t2a_plot: {t2a_artifacts.get('plot_path')}")
    print(f"t2b_status: {t2b_artifacts['status']}")
    if t2b_artifacts.get("plot_path"):
        print(f"t2b_plot: {t2b_artifacts['plot_path']}")
    if t2b_artifacts.get("note_path"):
        print(f"t2b_note: {t2b_artifacts['note_path']}")
    print(f"summary_path: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pq-run-dir",
        default=str(DEFAULT_PQ_RUN_DIR),
        help="Per-question run directory containing per_question/ and pq_analysis/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory that will receive final_outputs/t2a and final_outputs/t2b artifacts.",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="Qualified question_id to plot. If omitted, auto-select one question.",
    )
    parser.add_argument(
        "--bin-length",
        type=int,
        default=None,
        help="Exact bin length to use for T2-B. If omitted, resolve from l_star_A or a nearby ok bin in [6, 8].",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write T2-A plus a T2-B note when per-trace NLDD input is unavailable.",
    )
    parser.add_argument(
        "--random-t2a-count",
        type=int,
        default=0,
        help="Additionally export T2-A for N randomly sampled qualified questions.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used by --random-t2a-count.",
    )
    parser.add_argument(
        "--require-no-gap",
        action="store_true",
        help="When sampling random T2-A questions, require the L values in l_curve.csv to be contiguous.",
    )
    parser.add_argument(
        "--require-all-ok-for-t2a",
        action="store_true",
        help="When sampling random T2-A questions, require every plotted L to be bin_status=ok.",
    )
    parser.add_argument(
        "--t2b-all-bins",
        action="store_true",
        help="Additionally export T2-B curves for every ok bin of one qualified question.",
    )
    parser.add_argument(
        "--t2b-all-bins-question-id",
        default=None,
        help="Qualified question_id used by --t2b-all-bins. If omitted, auto-select one with broad ok-bin coverage.",
    )
    return parser.parse_args()


def resolve_question_selection(
    *,
    pq_run_dir: Path,
    question_id: str | None,
    bin_length: int | None,
) -> QuestionSelection:
    lstar_df, bin_status_df, qualified_ids = load_plotting_views(pq_run_dir)
    lstar_df = lstar_df[lstar_df["question_id"].isin(qualified_ids)].copy()

    if question_id is None:
        return _auto_select_question(lstar_df=lstar_df, bin_status_df=bin_status_df)

    question_id = str(question_id)
    question_rows = lstar_df[lstar_df["question_id"] == question_id].copy()
    if question_rows.empty:
        raise ValueError(f"Question '{question_id}' is not present in t2b_lstar_difficulty.csv.")

    row = question_rows.iloc[0]
    resolved_length = (
        int(bin_length)
        if bin_length is not None
        else _resolve_question_bin_length(
            question_id=question_id,
            preferred_length=int(row["l_star_A"]),
            bin_status_df=bin_status_df,
        )
    )
    status_row = _require_ok_bin(bin_status_df=bin_status_df, question_id=question_id, bin_length=resolved_length)
    return QuestionSelection(
        question_id=question_id,
        bin_length=resolved_length,
        n_retained=int(status_row["n_retained"]),
        difficulty_score=float(row["difficulty_score"]),
        l_star_a=int(row["l_star_A"]),
        l_star_s=int(row["l_star_S"]),
        l_star_consistent=bool(row["l_star_consistent"]),
    )


def _auto_select_question(*, lstar_df: pd.DataFrame, bin_status_df: pd.DataFrame) -> QuestionSelection:
    candidates: list[QuestionSelection] = []
    for row in lstar_df.itertuples(index=False):
        question_id = str(row.question_id)
        medium_ok_rows = bin_status_df[
            (bin_status_df["scope"] == question_id)
            & (bin_status_df["bin_status"] == "ok")
            & (bin_status_df["L"] >= 6)
            & (bin_status_df["L"] <= 8)
        ]
        if medium_ok_rows.empty:
            continue
        medium_ok_rows = medium_ok_rows.sort_values(
            by=["n_retained", "L"],
            ascending=[False, False],
            kind="mergesort",
        )
        status_row = medium_ok_rows.iloc[0]
        candidates.append(
            QuestionSelection(
                question_id=question_id,
                bin_length=int(status_row["L"]),
                n_retained=int(status_row["n_retained"]),
                difficulty_score=float(row.difficulty_score),
                l_star_a=int(row.l_star_A),
                l_star_s=int(row.l_star_S),
                l_star_consistent=bool(row.l_star_consistent),
            )
        )
    if not candidates:
        raise ValueError("Could not find a qualified question that has an ok PQ bin inside L in [6, 8].")
    candidates.sort(
        key=lambda item: (
            int(item.l_star_consistent),
            item.n_retained,
            item.bin_length,
            -item.difficulty_score,
            item.question_id,
        ),
        reverse=True,
    )
    return candidates[0]


def _resolve_question_bin_length(
    *,
    question_id: str,
    preferred_length: int,
    bin_status_df: pd.DataFrame,
) -> int:
    ok_rows = bin_status_df[
        (bin_status_df["scope"] == question_id)
        & (bin_status_df["bin_status"] == "ok")
        & (bin_status_df["L"] >= 6)
        & (bin_status_df["L"] <= 8)
    ].copy()
    if ok_rows.empty:
        raise ValueError(f"Question '{question_id}' has no ok PQ bin in the requested medium-length range [6, 8].")
    ok_rows["distance_to_preferred"] = (ok_rows["L"] - preferred_length).abs()
    ok_rows = ok_rows.sort_values(
        by=["distance_to_preferred", "n_retained", "L"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    return int(ok_rows.iloc[0]["L"])


def _require_ok_bin(*, bin_status_df: pd.DataFrame, question_id: str, bin_length: int) -> pd.Series:
    rows = bin_status_df[
        (bin_status_df["scope"] == question_id)
        & (bin_status_df["L"] == bin_length)
        & (bin_status_df["bin_status"] == "ok")
    ]
    if rows.empty:
        raise ValueError(f"Question '{question_id}' does not have an ok PQ bin at L={bin_length}.")
    return rows.iloc[0]


def export_t2a(
    *,
    pq_run_dir: Path,
    selection: QuestionSelection,
    output_dir: Path,
) -> dict[str, Any]:
    curve_path = pq_run_dir / PER_QUESTION_DIRNAME / selection.question_id / "l_curve.csv"
    lstar_path = pq_run_dir / PER_QUESTION_DIRNAME / selection.question_id / "l_star.json"
    bin_status_path = pq_run_dir / PQ_ANALYSIS_DIRNAME / "bin_status.csv"

    curve_df = pd.read_csv(curve_path)
    curve_df["L"] = pd.to_numeric(curve_df["L"], errors="coerce")
    curve_df["accuracy"] = pd.to_numeric(curve_df["accuracy"], errors="coerce")
    curve_df["accuracy_se"] = pd.to_numeric(curve_df["accuracy_se"], errors="coerce")
    curve_df["n"] = pd.to_numeric(curve_df["n"], errors="coerce")

    bin_status_df = pd.read_csv(bin_status_path)
    bin_status_df = bin_status_df[bin_status_df["scope"].astype(str) == selection.question_id].copy()
    bin_status_df["L"] = pd.to_numeric(bin_status_df["L"], errors="coerce")
    bin_status_df["n_retained"] = pd.to_numeric(bin_status_df["n_retained"], errors="coerce")
    bin_status_df = bin_status_df[["L", "n_retained", "bin_status"]]

    merged = curve_df.merge(bin_status_df, on="L", how="left")
    lstar_payload = json.loads(lstar_path.read_text(encoding="utf-8"))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(
        merged["L"],
        merged["accuracy"],
        color="#1f6feb",
        linewidth=2.0,
        marker="o",
        label="accuracy",
    )
    ax.fill_between(
        merged["L"],
        (merged["accuracy"] - merged["accuracy_se"]).clip(lower=0.0),
        (merged["accuracy"] + merged["accuracy_se"]).clip(upper=1.0),
        color="#1f6feb",
        alpha=0.12,
    )

    ok_rows = merged[merged["bin_status"].astype(str) == "ok"].copy()
    if not ok_rows.empty:
        ax.scatter(
            ok_rows["L"],
            ok_rows["accuracy"],
            color="#2ca02c",
            s=64,
            zorder=3,
            label="PQ bin ok",
        )

    ax.axvspan(6, 8, color="#ffcc66", alpha=0.15, label="medium-length window [6, 8]")
    ax.axvline(selection.l_star_a, color="#aa3377", linestyle="--", linewidth=1.6, label="l_star_A")
    ax.axvline(selection.bin_length, color="#ff7f0e", linestyle=":", linewidth=1.6, label="selected bin")

    ax.set_title(f"T2-A L-curve: {selection.question_id}")
    ax.set_xlabel("L")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = output_dir / f"t2a_l_curve_{selection.question_id}.png"
    csv_copy_path = output_dir / f"t2a_l_curve_{selection.question_id}.csv"
    meta_path = output_dir / f"t2a_l_curve_{selection.question_id}.json"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    merged.sort_values("L", kind="mergesort").to_csv(csv_copy_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "question_id": selection.question_id,
                "selected_bin_length": selection.bin_length,
                "l_star_payload": lstar_payload,
                "plot_path": str(plot_path),
                "csv_path": str(csv_copy_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "plot_path": str(plot_path),
        "csv_path": str(csv_copy_path),
        "meta_path": str(meta_path),
    }


def export_random_t2a_batch(
    *,
    pq_run_dir: Path,
    output_dir: Path,
    count: int,
    seed: int,
    require_no_gap: bool,
    require_all_ok: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lstar_df, bin_status_df, qualified_ids = load_plotting_views(pq_run_dir)

    candidates = build_t2a_candidates(
        pq_run_dir=pq_run_dir,
        lstar_df=lstar_df,
        bin_status_df=bin_status_df,
        qualified_ids=qualified_ids,
        require_no_gap=require_no_gap,
        require_all_ok=require_all_ok,
    )
    if len(candidates) < count:
        raise ValueError(
            f"Requested {count} random T2-A questions, but only {len(candidates)} candidates satisfy the filters."
        )

    rng = random.Random(seed)
    selected = sorted(rng.sample(candidates, count), key=lambda item: item.question_id)

    manifest_rows: list[dict[str, Any]] = []
    for selection in selected:
        artifacts = export_t2a(
            pq_run_dir=pq_run_dir,
            selection=QuestionSelection(
                question_id=selection.question_id,
                bin_length=_resolve_question_bin_length(
                    question_id=selection.question_id,
                    preferred_length=selection.l_star_a,
                    bin_status_df=bin_status_df,
                ),
                n_retained=0,
                difficulty_score=selection.difficulty_score,
                l_star_a=selection.l_star_a,
                l_star_s=selection.l_star_s,
                l_star_consistent=selection.l_star_consistent,
            ),
            output_dir=output_dir,
        )
        lcurve_df = pd.read_csv(pq_run_dir / PER_QUESTION_DIRNAME / selection.question_id / "l_curve.csv")
        Ls = sorted(pd.to_numeric(lcurve_df["L"], errors="coerce").dropna().astype(int).tolist())
        manifest_rows.append(
            {
                "question_id": selection.question_id,
                "difficulty_score": selection.difficulty_score,
                "l_star_a": selection.l_star_a,
                "l_star_s": selection.l_star_s,
                "l_star_consistent": selection.l_star_consistent,
                "min_L": min(Ls),
                "max_L": max(Ls),
                "n_L": len(Ls),
                "plot_path": artifacts["plot_path"],
                "csv_path": artifacts["csv_path"],
            }
        )

    manifest_path = output_dir / "random_t2a_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return {
        "selected_count": len(manifest_rows),
        "manifest_path": str(manifest_path),
    }


def export_t2b(
    *,
    pq_run_dir: Path,
    selection: QuestionSelection,
    output_dir: Path,
    allow_partial: bool,
) -> dict[str, Any]:
    selection_path = (
        pq_run_dir
        / PER_QUESTION_DIRNAME
        / selection.question_id
        / "bins"
        / f"bin_{selection.bin_length}"
        / "selection.jsonl"
    )
    selection_rows = _load_jsonl(selection_path)
    selection_df = pd.DataFrame(selection_rows)
    selection_df["sample_id"] = selection_df["sample_id"].astype(str)
    selection_df["source_trace_id"] = selection_df["source_trace_id"].astype(str)
    selection_df["actual_num_steps"] = pd.to_numeric(selection_df["actual_num_steps"], errors="coerce")
    selection_df["trace_tier"] = pd.to_numeric(selection_df["trace_tier"], errors="coerce")

    manifest_path = output_dir / f"t2b_selection_manifest_{selection.question_id}_L{selection.bin_length}.csv"
    selection_df.sort_values("sample_id", kind="mergesort").to_csv(manifest_path, index=False)

    nldd_path = pq_run_dir / PQ_ANALYSIS_DIRNAME / "nldd_per_trace.jsonl"
    if not nldd_path.exists():
        note_path = output_dir / f"t2b_missing_input_{selection.question_id}_L{selection.bin_length}.json"
        payload = {
            "status": "missing_input",
            "message": "Cannot render T2-B because pq_analysis/nldd_per_trace.jsonl is not present locally.",
            "expected_path": str(nldd_path),
            "selection_manifest_path": str(manifest_path),
            "question_id": selection.question_id,
            "bin_length": selection.bin_length,
            "sample_count_requested": len(selection_df),
        }
        note_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if allow_partial:
            return {
                "status": "missing_input",
                "selection_manifest_path": str(manifest_path),
                "note_path": str(note_path),
            }
        raise FileNotFoundError(payload["message"] + f" Expected: {nldd_path}")

    wanted_trace_ids = set(selection_df["source_trace_id"].astype(str))
    nldd_rows = [
        row
        for row in _load_jsonl(nldd_path)
        if str(row.get("question_id")) == selection.question_id
        and int(row.get("length")) == selection.bin_length
        and str(row.get("source_trace_id")) in wanted_trace_ids
        and row.get("nldd_value") is not None
    ]
    if not nldd_rows:
        note_path = output_dir / f"t2b_missing_rows_{selection.question_id}_L{selection.bin_length}.json"
        payload = {
            "status": "missing_rows",
            "message": "nldd_per_trace.jsonl exists, but it does not contain matching per-trace NLDD rows for this PQ bin.",
            "nldd_path": str(nldd_path),
            "selection_manifest_path": str(manifest_path),
            "question_id": selection.question_id,
            "bin_length": selection.bin_length,
        }
        note_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if allow_partial:
            return {
                "status": "missing_rows",
                "selection_manifest_path": str(manifest_path),
                "note_path": str(note_path),
            }
        raise ValueError(payload["message"])

    nldd_df = pd.DataFrame(nldd_rows)
    nldd_df["sample_id"] = nldd_df["sample_id"].astype(str)
    nldd_df["source_trace_id"] = nldd_df["source_trace_id"].astype(str)
    nldd_df["k"] = pd.to_numeric(nldd_df["k"], errors="coerce")
    nldd_df["nldd_value"] = pd.to_numeric(nldd_df["nldd_value"], errors="coerce")
    nldd_df = nldd_df.merge(
        selection_df[["sample_id", "source_trace_id"]],
        on=["sample_id", "source_trace_id"],
        how="inner",
    )
    if nldd_df.empty:
        raise ValueError("Matched NLDD rows disappeared after joining against the PQ selection manifest.")

    csv_path = output_dir / f"t2b_nldd_per_sample_{selection.question_id}_L{selection.bin_length}.csv"
    nldd_df.sort_values(["sample_id", "k"], kind="mergesort").to_csv(csv_path, index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    sample_order = selection_df.sort_values("sample_id", kind="mergesort")["sample_id"].astype(str).tolist()
    palette = sns.color_palette("viridis", n_colors=max(len(sample_order), 1))
    color_by_sample = {sample_id: palette[index] for index, sample_id in enumerate(sample_order)}

    for sample_id in sample_order:
        sample_rows = nldd_df[nldd_df["sample_id"] == sample_id].sort_values("k", kind="mergesort")
        if sample_rows.empty:
            continue
        ax.plot(
            sample_rows["k"],
            sample_rows["nldd_value"],
            marker="o",
            linewidth=1.6,
            markersize=3.4,
            alpha=0.88,
            color=color_by_sample[sample_id],
            label=f"sample {sample_id}",
        )

    ax.set_title(f"T2-B NLDD(k) curves: {selection.question_id} at L={selection.bin_length}")
    ax.set_xlabel("k")
    ax.set_ylabel("NLDD(k)")
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(list(range(2, selection.bin_length + 1)))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, ncol=1, fontsize=8)
    fig.tight_layout()

    plot_path = output_dir / f"t2b_nldd_per_sample_{selection.question_id}_L{selection.bin_length}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    meta_path = output_dir / f"t2b_nldd_per_sample_{selection.question_id}_L{selection.bin_length}.json"
    meta_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "question_id": selection.question_id,
                "bin_length": selection.bin_length,
                "requested_sample_count": len(selection_df),
                "rendered_sample_count": int(nldd_df["sample_id"].nunique()),
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
                "selection_manifest_path": str(manifest_path),
                "nldd_path": str(nldd_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "plot_path": str(plot_path),
        "csv_path": str(csv_path),
        "meta_path": str(meta_path),
        "selection_manifest_path": str(manifest_path),
    }


def export_t2b_all_bins(
    *,
    pq_run_dir: Path,
    output_dir: Path,
    question_id: str | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lstar_df, bin_status_df, qualified_ids = load_plotting_views(pq_run_dir)

    selected_question_id = (
        str(question_id)
        if question_id is not None
        else auto_select_all_bins_question(
            lstar_df=lstar_df,
            bin_status_df=bin_status_df,
            qualified_ids=qualified_ids,
        )
    )
    if selected_question_id not in qualified_ids:
        raise ValueError(f"Question '{selected_question_id}' is not in the qualified-question set.")

    question_rows = lstar_df[lstar_df["question_id"] == selected_question_id].copy()
    if question_rows.empty:
        raise ValueError(f"Missing l_star row for question '{selected_question_id}'.")
    lstar_row = question_rows.iloc[0]

    ok_rows = bin_status_df[
        (bin_status_df["scope"] == selected_question_id)
        & (bin_status_df["bin_status"] == "ok")
    ].copy()
    ok_rows = ok_rows.sort_values("L", kind="mergesort")
    if ok_rows.empty:
        raise ValueError(f"Question '{selected_question_id}' has no ok bins for T2-B all-bins export.")

    question_output_dir = output_dir / selected_question_id
    question_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for row in ok_rows.itertuples(index=False):
        selection = QuestionSelection(
            question_id=selected_question_id,
            bin_length=int(row.L),
            n_retained=int(row.n_retained),
            difficulty_score=float(lstar_row["difficulty_score"]),
            l_star_a=int(lstar_row["l_star_A"]),
            l_star_s=int(lstar_row["l_star_S"]),
            l_star_consistent=bool(lstar_row["l_star_consistent"]),
        )
        artifacts = export_t2b(
            pq_run_dir=pq_run_dir,
            selection=selection,
            output_dir=question_output_dir,
            allow_partial=False,
        )
        manifest_rows.append(
            {
                "question_id": selected_question_id,
                "L": int(row.L),
                "n_retained": int(row.n_retained),
                "plot_path": artifacts["plot_path"],
                "csv_path": artifacts["csv_path"],
                "selection_manifest_path": artifacts["selection_manifest_path"],
            }
        )

    manifest_path = question_output_dir / "t2b_all_bins_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return {
        "question_id": selected_question_id,
        "bin_count": len(manifest_rows),
        "manifest_path": str(manifest_path),
    }


def auto_select_all_bins_question(
    *,
    lstar_df: pd.DataFrame,
    bin_status_df: pd.DataFrame,
    qualified_ids: set[str],
) -> str:
    candidates: list[tuple[int, int, int, str]] = []
    lstar_by_q = {
        str(row["question_id"]): row
        for _, row in lstar_df.iterrows()
    }
    for question_id in sorted(qualified_ids):
        ok_rows = bin_status_df[
            (bin_status_df["scope"] == question_id)
            & (bin_status_df["bin_status"] == "ok")
        ].copy()
        if ok_rows.empty:
            continue
        lengths = sorted(ok_rows["L"].astype(int).tolist())
        contiguous_pairs = sum(1 for left, right in zip(lengths, lengths[1:]) if right - left == 1)
        total_retained = int(ok_rows["n_retained"].sum())
        candidates.append((len(lengths), contiguous_pairs, total_retained, question_id))
    if not candidates:
        raise ValueError("Could not find any qualified question with ok bins.")
    candidates.sort(reverse=True)
    return candidates[0][3]


def build_t2a_candidates(
    *,
    pq_run_dir: Path,
    lstar_df: pd.DataFrame,
    bin_status_df: pd.DataFrame,
    qualified_ids: set[str],
    require_no_gap: bool,
    require_all_ok: bool,
) -> list[T2ASelection]:
    selections: list[T2ASelection] = []
    lstar_by_q = {
        str(row["question_id"]): row
        for _, row in lstar_df.iterrows()
        if str(row["question_id"]) in qualified_ids
    }
    for question_id in sorted(qualified_ids):
        if question_id not in lstar_by_q:
            continue
        curve_path = pq_run_dir / PER_QUESTION_DIRNAME / question_id / "l_curve.csv"
        if not curve_path.exists():
            continue
        curve_df = pd.read_csv(curve_path)
        lengths = sorted(pd.to_numeric(curve_df["L"], errors="coerce").dropna().astype(int).tolist())
        if not lengths:
            continue
        if require_no_gap and not _is_contiguous(lengths):
            continue
        if require_all_ok:
            ok_lookup = {
                int(row["L"]): str(row["bin_status"])
                for _, row in bin_status_df[bin_status_df["scope"] == question_id].iterrows()
            }
            if not all(ok_lookup.get(length) == "ok" for length in lengths):
                continue
        row = lstar_by_q[question_id]
        selections.append(
            T2ASelection(
                question_id=question_id,
                difficulty_score=float(row["difficulty_score"]),
                l_star_a=int(row["l_star_A"]),
                l_star_s=int(row["l_star_S"]),
                l_star_consistent=bool(row["l_star_consistent"]),
            )
        )
    return selections


def load_plotting_views(
    pq_run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    lstar_df = pd.read_csv(pq_run_dir / PQ_ANALYSIS_DIRNAME / "t2b_lstar_difficulty.csv")
    bin_status_df = pd.read_csv(pq_run_dir / PQ_ANALYSIS_DIRNAME / "bin_status.csv")
    qualified_df = pd.read_csv(pq_run_dir / PQ_ANALYSIS_DIRNAME / "qualified_questions_full_metrics.csv")

    lstar_df["question_id"] = lstar_df["question_id"].astype(str)
    lstar_df["l_star_A"] = pd.to_numeric(lstar_df["l_star_A"], errors="coerce")
    lstar_df["l_star_S"] = pd.to_numeric(lstar_df["l_star_S"], errors="coerce")
    lstar_df["difficulty_score"] = pd.to_numeric(lstar_df["difficulty_score"], errors="coerce")
    lstar_df["l_star_consistent"] = lstar_df["l_star_consistent"].astype(str).str.lower() == "true"

    bin_status_df["scope"] = bin_status_df["scope"].astype(str)
    bin_status_df["L"] = pd.to_numeric(bin_status_df["L"], errors="coerce")
    bin_status_df["n_retained"] = pd.to_numeric(bin_status_df["n_retained"], errors="coerce")
    bin_status_df["bin_status"] = bin_status_df["bin_status"].astype(str)

    qualified_df["question_id"] = qualified_df["question_id"].astype(str)
    qualified_ids = set(
        qualified_df.loc[
            (~qualified_df["degenerate"].astype(bool))
            & (~qualified_df["l_curve_insufficient"].astype(bool))
            & (~qualified_df["k_star_insufficient"].astype(bool)),
            "question_id",
        ]
    )
    return lstar_df, bin_status_df, qualified_ids


def _is_contiguous(lengths: list[int]) -> bool:
    return all(right - left == 1 for left, right in zip(lengths, lengths[1:]))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


if __name__ == "__main__":
    main()
