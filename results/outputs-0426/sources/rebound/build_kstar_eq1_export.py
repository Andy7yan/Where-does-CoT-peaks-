"""Export k*/L = 1 GSM8K-PQ bins and the retained clean traces behind them."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


EPSILON = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def load_lstar_by_question(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "pq_analysis" / "t2b_lstar_difficulty.csv"
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        result[row["question_id"]] = {
            "l_star_A": to_int(row.get("l_star_A")),
            "l_star_S": to_int(row.get("l_star_S") or row.get("l_star_C")),
            "l_star_consistent": row.get("l_star_consistent"),
        }
    return result


def load_trace_lookup(run_dir: Path) -> dict[str, dict[str, Any]]:
    traces_path = run_dir / "traces.jsonl"
    lookup: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(traces_path):
        trace_id = row.get("trace_id")
        if trace_id:
            lookup[str(trace_id)] = row
    return lookup


def per_question_roots(run_dir: Path, fallback_per_question_dir: Path | None) -> list[Path]:
    roots = [run_dir / "per_question"]
    if fallback_per_question_dir is not None:
        roots.append(fallback_per_question_dir)
    return roots


def find_bin_path(
    run_dir: Path,
    fallback_per_question_dir: Path | None,
    question_id: str,
    length: int,
    filename: str,
) -> Path | None:
    for root in per_question_roots(run_dir, fallback_per_question_dir):
        path = root / question_id / "bins" / f"bin_{length}" / filename
        if path.exists():
            return path
    return None


def load_bin_summary(
    run_dir: Path,
    fallback_per_question_dir: Path | None,
    question_id: str,
    length: int,
) -> dict[str, Any]:
    path = find_bin_path(run_dir, fallback_per_question_dir, question_id, length, "bin_summary.json")
    if path is not None:
        payload = read_json(path)
        payload["_source_path"] = str(path)
        return payload
    return {}


def load_selection_rows(
    run_dir: Path,
    fallback_per_question_dir: Path | None,
    question_id: str,
    length: int,
) -> list[dict[str, Any]]:
    path = find_bin_path(run_dir, fallback_per_question_dir, question_id, length, "selection.jsonl")
    if path is None:
        return []
    rows = list(iter_jsonl(path))
    for row in rows:
        row["_source_selection_path"] = str(path)
    return rows


def build_bin_record(
    row: dict[str, str],
    run_dir: Path,
    fallback_per_question_dir: Path | None,
    lstar_by_question: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    question_id = row["question_id"]
    length = to_int(row["L"])
    if length is None:
        raise ValueError(f"Bad L for row: {row}")
    bin_summary = load_bin_summary(run_dir, fallback_per_question_dir, question_id, length)
    lstar = lstar_by_question.get(question_id, {})
    l_star_A = lstar.get("l_star_A")
    return {
        "question_id": question_id,
        "L": length,
        "k_star": to_int(row.get("k_star")),
        "k_star_ratio": to_float(row.get("k_star_ratio")),
        "difficulty_score": to_float(row.get("difficulty_score")),
        "n_clean": to_int(row.get("n_clean")),
        "l_star_A": l_star_A,
        "l_star_S": lstar.get("l_star_S"),
        "l_star_consistent": lstar.get("l_star_consistent"),
        "L_minus_l_star_A": (length - l_star_A) if isinstance(l_star_A, int) else None,
        "bin_status": bin_summary.get("bin_status"),
        "n_total_traces": bin_summary.get("n_total_traces"),
        "n_correct": bin_summary.get("n_correct"),
        "n_tier1": bin_summary.get("n_tier1"),
        "n_tier2": bin_summary.get("n_tier2"),
        "n_failed": bin_summary.get("n_failed"),
        "n_retained": bin_summary.get("n_retained"),
        "bin_summary_source_path": bin_summary.get("_source_path"),
    }


def is_eq1(row: dict[str, str]) -> bool:
    ratio = to_float(row.get("k_star_ratio"))
    length = to_int(row.get("L"))
    k_star = to_int(row.get("k_star"))
    if ratio is None or length is None or k_star is None:
        return False
    return abs(ratio - 1.0) <= EPSILON and k_star == length


def is_lt1(row: dict[str, str]) -> bool:
    ratio = to_float(row.get("k_star_ratio"))
    return ratio is not None and ratio < 1.0 - EPSILON


def retained_trace_record(
    *,
    bin_record: dict[str, Any],
    selection: dict[str, Any],
    trace_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_trace_id = str(selection.get("source_trace_id"))
    original = trace_lookup.get(source_trace_id, {})
    return {
        "export_group": "kstar_ratio_eq_1",
        "question_id": bin_record["question_id"],
        "L": bin_record["L"],
        "k_star": bin_record["k_star"],
        "k_star_ratio": bin_record["k_star_ratio"],
        "difficulty_score": bin_record["difficulty_score"],
        "n_clean_in_bin": bin_record["n_clean"],
        "l_star_A": bin_record["l_star_A"],
        "L_minus_l_star_A": bin_record["L_minus_l_star_A"],
        "bin_summary": {
            key: bin_record.get(key)
            for key in ("bin_status", "n_total_traces", "n_correct", "n_tier1", "n_tier2", "n_failed", "n_retained")
        },
        "selection": selection,
        "trace": original,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    length_values = [float(row["L"]) for row in rows if row.get("L") is not None]
    difficulty_values = [float(row["difficulty_score"]) for row in rows if row.get("difficulty_score") is not None]
    n_clean_values = [float(row["n_clean"]) for row in rows if row.get("n_clean") is not None]
    l_minus_values = [float(row["L_minus_l_star_A"]) for row in rows if row.get("L_minus_l_star_A") is not None]
    length_hist = Counter(int(value) for value in length_values)
    return {
        "bin_count": len(rows),
        "question_count": len({row["question_id"] for row in rows}),
        "L": summarize_numbers(length_values),
        "L_histogram": dict(sorted(length_hist.items())),
        "difficulty_score": summarize_numbers(difficulty_values),
        "n_clean": summarize_numbers(n_clean_values),
        "L_minus_l_star_A": summarize_numbers(l_minus_values),
        "share_L_le_4": mean([value <= 4 for value in length_values]) if length_values else None,
        "share_L_ge_6": mean([value >= 6 for value in length_values]) if length_values else None,
        "share_L_after_lstar": mean([value > 0 for value in l_minus_values]) if l_minus_values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/papery-pq")
    parser.add_argument(
        "--fallback-per-question-dir",
        default="results/legacy/per-question-0419_143551/per_question",
        help="Optional per_question directory used when run-dir/per_question is incomplete.",
    )
    parser.add_argument("--output-dir", default="results/outputs-0426")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fallback_per_question_dir = Path(args.fallback_per_question_dir) if args.fallback_per_question_dir else None
    if fallback_per_question_dir is not None and not fallback_per_question_dir.exists():
        fallback_per_question_dir = None
    output_dir = Path(args.output_dir)
    source_dir = output_dir / "sources"
    t1c_path = run_dir / "pq_analysis" / "t1c_kstar_ratio.csv"

    lstar_by_question = load_lstar_by_question(run_dir)
    t1c_rows = read_csv(t1c_path)
    eq1_raw = [row for row in t1c_rows if is_eq1(row)]
    lt1_raw = [row for row in t1c_rows if is_lt1(row)]
    eq1_bins = [build_bin_record(row, run_dir, fallback_per_question_dir, lstar_by_question) for row in eq1_raw]
    lt1_bins = [build_bin_record(row, run_dir, fallback_per_question_dir, lstar_by_question) for row in lt1_raw]

    trace_lookup = load_trace_lookup(run_dir)
    trace_rows: list[dict[str, Any]] = []
    missing_trace_ids: list[str] = []
    for bin_record in eq1_bins:
        selections = load_selection_rows(
            run_dir,
            fallback_per_question_dir,
            bin_record["question_id"],
            int(bin_record["L"]),
        )
        for selection in selections:
            source_trace_id = str(selection.get("source_trace_id"))
            if source_trace_id not in trace_lookup:
                missing_trace_ids.append(source_trace_id)
            trace_rows.append(
                retained_trace_record(
                    bin_record=bin_record,
                    selection=selection,
                    trace_lookup=trace_lookup,
                )
            )

    bin_columns = [
        "question_id",
        "L",
        "k_star",
        "k_star_ratio",
        "difficulty_score",
        "n_clean",
        "l_star_A",
        "l_star_S",
        "l_star_consistent",
        "L_minus_l_star_A",
        "bin_status",
        "n_total_traces",
        "n_correct",
        "n_tier1",
        "n_tier2",
        "n_failed",
        "n_retained",
        "bin_summary_source_path",
    ]
    write_jsonl(output_dir / "kstar_ratio_eq1_retained_traces.jsonl", trace_rows)
    write_csv(source_dir / "kstar_ratio_eq1_bins.csv", eq1_bins, bin_columns)
    write_csv(source_dir / "kstar_ratio_lt1_bins.csv", lt1_bins, bin_columns)

    summary = {
        "run_dir": str(run_dir),
        "fallback_per_question_dir": str(fallback_per_question_dir) if fallback_per_question_dir else None,
        "t1c_path": str(t1c_path),
        "output_jsonl": str(output_dir / "kstar_ratio_eq1_retained_traces.jsonl"),
        "eq1": summarize_group(eq1_bins),
        "lt1": summarize_group(lt1_bins),
        "retained_trace_rows": len(trace_rows),
        "missing_trace_ids": missing_trace_ids,
        "notes": {
            "eq1_definition": "abs(k_star_ratio - 1) <= 1e-9 and k_star == L",
            "trace_definition": "retained clean traces listed in per_question/<question>/bins/bin_<L>/selection.jsonl, joined to the original traces.jsonl row",
        },
    }
    write_json(source_dir / "kstar_ratio_eq1_export_summary.json", summary)

    print(f"eq1_bins: {len(eq1_bins)}")
    print(f"lt1_bins: {len(lt1_bins)}")
    print(f"retained_trace_rows: {len(trace_rows)}")
    print(f"output_jsonl: {output_dir / 'kstar_ratio_eq1_retained_traces.jsonl'}")


if __name__ == "__main__":
    main()
