"""Build per-question T1D overview CSVs in a T1A-like schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_int(value: Any) -> int | None:
    if value in (None, "", "nan", "None", "null"):
        return None
    return int(value)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "nan", "None", "null"):
        return None
    return float(value)


def build_t1d_exports(*, pq_run_dir: Path, output_dir: Path) -> dict[str, Any]:
    per_question_root = pq_run_dir / "per_question"
    pq_analysis_dir = pq_run_dir / "pq_analysis"
    if not per_question_root.exists():
        raise FileNotFoundError(f"Missing per_question root: {per_question_root}")
    if not pq_analysis_dir.exists():
        raise FileNotFoundError(f"Missing pq_analysis root: {pq_analysis_dir}")

    lstar_rows = {
        str(row["question_id"]): {
            "difficulty_score": _to_float(row.get("difficulty_score")),
            "l_star_A": _to_int(row.get("l_star_A")),
            "l_star_S": _to_int(row.get("l_star_S")),
            "l_star_consistent": str(row.get("l_star_consistent", "")).strip(),
        }
        for row in _read_csv(pq_analysis_dir / "t2b_lstar_difficulty.csv")
    }
    kstar_rows = {
        (str(row["question_id"]), int(row["L"])): {
            "k_star": _to_int(row.get("k_star")),
            "k_star_ratio": _to_float(row.get("k_star_ratio")),
        }
        for row in _read_csv(pq_analysis_dir / "t1c_kstar_ratio.csv")
    }
    final_tas_rows = {
        (str(row["question_id"]), int(row["L"])): {
            "mean_tas": _to_float(row.get("mean_tas_t")),
            "tas_se": _to_float(row.get("tas_t_se")),
        }
        for row in _read_csv(pq_analysis_dir / "t1b_step_surface.csv")
        if _to_int(row.get("step")) == _to_int(row.get("L"))
    }

    manifest_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    export_fieldnames = [
        "question_id",
        "difficulty",
        "difficulty_score",
        "L",
        "accuracy",
        "accuracy_se",
        "k_star",
        "k_star_ratio",
        "mean_tas",
        "tas_se",
        "l_star",
        "l_star_A",
        "l_star_S",
        "l_star_consistent",
        "bin_status",
        "n_traces",
        "n_clean",
    ]

    for question_dir in sorted(path for path in per_question_root.iterdir() if path.is_dir()):
        question_id = question_dir.name
        lcurve_path = question_dir / "l_curve.csv"
        lstar_path = question_dir / "l_star.json"
        if not lcurve_path.exists() and not lstar_path.exists():
            continue

        lcurve_by_L: dict[int, dict[str, Any]] = {}
        if lcurve_path.exists():
            for row in _read_csv(lcurve_path):
                length = int(row["L"])
                lcurve_by_L[length] = {
                    "accuracy": _to_float(row.get("accuracy")),
                    "accuracy_se": _to_float(row.get("accuracy_se")),
                    "n_traces": _to_int(row.get("n")),
                }

        lstar_payload = _read_json(lstar_path) if lstar_path.exists() else {}
        bins_dir = question_dir / "bins"
        bin_rows_by_L: dict[int, dict[str, Any]] = {}
        if bins_dir.exists():
            for bin_dir in sorted(path for path in bins_dir.iterdir() if path.is_dir() and path.name.startswith("bin_")):
                summary_path = bin_dir / "bin_summary.json"
                if not summary_path.exists():
                    continue
                summary = _read_json(summary_path)
                length = int(summary["L"])
                bin_rows_by_L[length] = {
                    "bin_status": str(summary.get("bin_status", "")),
                    "n_clean": _to_int(summary.get("n_retained")),
                }

        lstar_meta = lstar_rows.get(
            question_id,
            {
                "difficulty_score": _to_float(lstar_payload.get("difficulty_score")),
                "l_star_A": _to_int(lstar_payload.get("l_star_A")),
                "l_star_S": _to_int(lstar_payload.get("l_star_S")),
                "l_star_consistent": str(lstar_payload.get("l_star_consistent", "")),
            },
        )
        difficulty = str(lstar_payload.get("difficulty", ""))

        lengths = sorted(
            set(lcurve_by_L)
            | set(bin_rows_by_L)
            | {length for qid, length in final_tas_rows if qid == question_id}
            | {length for qid, length in kstar_rows if qid == question_id}
        )
        if not lengths:
            continue

        question_rows: list[dict[str, Any]] = []
        for length in lengths:
            lcurve_row = lcurve_by_L.get(length, {})
            bin_row = bin_rows_by_L.get(length, {})
            tas_row = final_tas_rows.get((question_id, length), {})
            kstar_row = kstar_rows.get((question_id, length), {})
            row = {
                "question_id": question_id,
                "difficulty": difficulty,
                "difficulty_score": lstar_meta.get("difficulty_score"),
                "L": length,
                "accuracy": lcurve_row.get("accuracy"),
                "accuracy_se": lcurve_row.get("accuracy_se"),
                "k_star": kstar_row.get("k_star"),
                "k_star_ratio": kstar_row.get("k_star_ratio"),
                "mean_tas": tas_row.get("mean_tas"),
                "tas_se": tas_row.get("tas_se"),
                "l_star": bool(lstar_meta.get("l_star_A") == length),
                "l_star_A": lstar_meta.get("l_star_A"),
                "l_star_S": lstar_meta.get("l_star_S"),
                "l_star_consistent": lstar_meta.get("l_star_consistent"),
                "bin_status": bin_row.get("bin_status"),
                "n_traces": lcurve_row.get("n_traces"),
                "n_clean": bin_row.get("n_clean"),
            }
            question_rows.append(row)
            combined_rows.append(row)

        out_path = output_dir / f"{question_id}.csv"
        _write_csv(out_path, question_rows, export_fieldnames)
        manifest_rows.append(
            {
                "question_id": question_id,
                "csv_path": str(out_path),
                "difficulty": difficulty,
                "difficulty_score": lstar_meta.get("difficulty_score"),
                "row_count": len(question_rows),
                "min_L": min(lengths),
                "max_L": max(lengths),
                "l_star_A": lstar_meta.get("l_star_A"),
                "l_star_S": lstar_meta.get("l_star_S"),
            }
        )

    _write_csv(
        output_dir / "_manifest.csv",
        manifest_rows,
        ["question_id", "csv_path", "difficulty", "difficulty_score", "row_count", "min_L", "max_L", "l_star_A", "l_star_S"],
    )
    _write_csv(output_dir / "t1d_full.csv", combined_rows, export_fieldnames)
    return {
        "question_count": len(manifest_rows),
        "manifest_path": str(output_dir / "_manifest.csv"),
        "combined_path": str(output_dir / "t1d_full.csv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pq-run-dir",
        required=True,
        help="Per-question run directory containing per_question/ and pq_analysis/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for T1D per-question overview CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_t1d_exports(
        pq_run_dir=Path(args.pq_run_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"question_count: {artifacts['question_count']}")
    print(f"manifest_path: {artifacts['manifest_path']}")
    print(f"combined_path: {artifacts['combined_path']}")


if __name__ == "__main__":
    main()
