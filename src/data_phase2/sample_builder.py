"""Build a compact sample run from formal Stage 1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase2.difficulty_groups import _extract_question_code
from src.data_phase2.difficulty_histogram import export_difficulty_histogram


DIFFICULTIES = ("easy", "medium", "hard")


def main() -> None:
    """Build a deterministic sample fixture for later-stage development."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists. Pass --overwrite to rebuild it."
        )

    manifest = build_sample_run(
        source_run_dir=Path(args.source_run_dir),
        output_dir=output_dir,
        samples_per_difficulty=args.samples_per_difficulty,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        default="results/full_generation",
        help="Legacy full_generation directory used as the source corpus.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/sample_data",
        help="Directory that will receive the compact sample run.",
    )
    parser.add_argument(
        "--samples-per-difficulty",
        type=int,
        default=2,
        help="Maximum number of eligible traces to keep for each difficulty.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first when it already exists.",
    )
    return parser.parse_args()


def build_sample_run(
    *,
    source_run_dir: Path,
    output_dir: Path,
    samples_per_difficulty: int,
) -> dict[str, Any]:
    """Build a small sample run rooted at ``output_dir``."""

    if samples_per_difficulty <= 0:
        raise ValueError("samples_per_difficulty must be positive.")

    question_metadata_rows = _load_jsonl(source_run_dir / "question_metadata.jsonl")
    metadata_by_question = {
        str(row["question_id"]): row
        for row in question_metadata_rows
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    difficulty_root = output_dir / "difficulty"
    difficulty_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "stage1_sample_run_v1",
        "source_run_dir": str(source_run_dir),
        "notes": {
            "purpose": "Compact fixture for later-stage IO and aggregation development.",
            "source_scheme": "formal Stage 1 exports reorganized into the current difficulty/bin/sample layout",
        },
        "difficulties": {},
    }

    for difficulty in DIFFICULTIES:
        selected_traces, grouped_corruptions = _select_sample_traces(
            source_run_dir=source_run_dir,
            difficulty=difficulty,
            samples_per_difficulty=samples_per_difficulty,
            metadata_by_question=metadata_by_question,
        )
        question_ids = sorted({str(trace["question_id"]) for trace in selected_traces})
        difficulty_dir = difficulty_root / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            difficulty_dir / "question_metadata.jsonl",
            [_project_group_question_metadata(metadata_by_question[question_id]) for question_id in question_ids],
        )
        _write_jsonl(difficulty_dir / "questions.jsonl", _build_question_rows(selected_traces))
        _write_jsonl(difficulty_dir / "traces.jsonl", [_project_trace_row(row) for row in selected_traces])

        difficulty_summary = {
            "question_count": len(question_ids),
            "trace_count": len(selected_traces),
            "bins": {},
        }

        traces_by_length = _group_by_length(selected_traces)
        for length in sorted(traces_by_length):
            selection_rows: list[dict[str, Any]] = []
            bin_dir = difficulty_dir / "bins" / f"bin_{length}"
            samples_dir = bin_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            sample_id_counts: dict[str, int] = {}
            for trace in traces_by_length[length]:
                trace_id = str(trace["trace_id"])
                per_k_rows = grouped_corruptions[trace_id]
                trace_tier = _resolve_trace_tier(per_k_rows)
                k_values = [int(row["step_index"]) for row in per_k_rows]
                sample_id = _assign_sample_id(
                    str(trace["question_id"]),
                    sample_id_counts=sample_id_counts,
                )
                sample_dir = samples_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                clean_payload = _build_clean_payload(trace)
                meta_payload = _build_meta_payload(
                    trace=trace,
                    sample_id=sample_id,
                    trace_tier=trace_tier,
                    per_k_rows=per_k_rows,
                )
                _write_json(sample_dir / "clean.json", clean_payload)
                _write_json(sample_dir / "meta.json", meta_payload)
                for corruption_row in per_k_rows:
                    k = int(corruption_row["step_index"])
                    _write_json(
                        sample_dir / f"corrupt_k{k}.json",
                        _build_corrupt_payload(trace=trace, corruption_row=corruption_row),
                    )

                selection_rows.append(
                    {
                        "sample_id": sample_id,
                        "question_id": trace["question_id"],
                        "source_trace_id": trace_id,
                        "actual_num_steps": length,
                        "trace_tier": trace_tier,
                        "k_values": k_values,
                        "sample_path": f"samples/{sample_id}",
                    }
                )

            _write_jsonl(bin_dir / "selection.jsonl", selection_rows)
            difficulty_summary["bins"][str(length)] = {
                "trace_count": len(selection_rows),
                "k_values_per_trace": {
                    row["source_trace_id"]: row["k_values"]
                    for row in selection_rows
                },
            }

        manifest["difficulties"][difficulty] = difficulty_summary

    histogram_path = source_run_dir / "difficulty_histogram.csv"
    if histogram_path.exists():
        shutil.copy2(histogram_path, output_dir / "difficulty_histogram.csv")
    else:
        export_difficulty_histogram(
            question_metadata_path=source_run_dir / "question_metadata.jsonl",
            output_path=output_dir / "difficulty_histogram.csv",
        )
    _write_json(output_dir / "sample_manifest.json", manifest)
    return manifest


def _select_sample_traces(
    *,
    source_run_dir: Path,
    difficulty: str,
    samples_per_difficulty: int,
    metadata_by_question: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    difficulty_root = source_run_dir / "difficulty"
    legacy_root = source_run_dir / "difficulty_groups"
    traces = []
    root_traces_path = source_run_dir / "traces.jsonl"
    if root_traces_path.exists():
        traces = [
            trace
            for trace in _load_jsonl(root_traces_path)
            if metadata_by_question.get(str(trace["question_id"]), {}).get("difficulty_bucket") == difficulty
        ]
    if (difficulty_root / difficulty / "traces.jsonl").exists():
        corruption_rows = []
        traces_by_id = {
            str(trace["trace_id"]): trace
            for trace in traces
        }
        for selection_path in (difficulty_root / difficulty / "bins").glob("bin_*/selection.jsonl"):
            selection_rows = _load_jsonl(selection_path)
            for selection_row in selection_rows:
                sample_dir = selection_path.parent / "samples" / str(selection_row["sample_id"])
                meta_path = sample_dir / "meta.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                source_trace_id = str(selection_row.get("source_trace_id", selection_row.get("trace_id", "")))
                trace = traces_by_id.get(source_trace_id)
                if trace is None:
                    continue
                clean_steps = [str(step) for step in trace.get("steps", [])]
                per_k_by_index = {
                    int(row.get("k", row.get("step_index"))): row
                    for row in meta.get("per_k", [])
                }
                for k in meta.get("k_values", []):
                    corrupt_path = sample_dir / f"corrupt_k{k}.json"
                    if not corrupt_path.exists():
                        continue
                    corrupt_payload = json.loads(corrupt_path.read_text(encoding="utf-8"))
                    corrupt_steps = [str(step) for step in corrupt_payload.get("steps", [])]
                    if len(clean_steps) < int(k) or len(corrupt_steps) < int(k):
                        continue
                    per_k_meta = per_k_by_index.get(int(k))
                    if per_k_meta is None:
                        continue
                    corruption_rows.append(
                        {
                            "trace_id": source_trace_id,
                            "step_index": int(k),
                            "corruption_id": per_k_meta["corruption_id"],
                            "corruption_tier": int(per_k_meta.get("tier", per_k_meta.get("corruption_tier", 1))),
                            "clean_step": clean_steps[int(k) - 1],
                            "corrupt_step": corrupt_steps[int(k) - 1],
                            "token_delta": per_k_meta.get("token_delta", 0),
                            "corruption_failed": False,
                        }
                    )
    else:
        if not traces:
            traces = _load_jsonl(legacy_root / difficulty / "traces.jsonl")
        corruption_rows = _load_jsonl(legacy_root / difficulty / "corruptions.jsonl")

    grouped_corruptions: dict[str, list[dict[str, Any]]] = {}
    for row in corruption_rows:
        if bool(row.get("corruption_failed")):
            continue
        step_index = int(row["step_index"])
        if step_index < 2:
            continue
        grouped_corruptions.setdefault(str(row["trace_id"]), []).append(row)

    eligible_traces: list[dict[str, Any]] = []
    for trace in traces:
        if not trace.get("is_correct"):
            continue
        measurement_eligible = trace.get("nldd_measurement_eligible")
        if measurement_eligible is None:
            measurement_eligible = int(trace.get("actual_num_steps", 0)) >= 3
        if not measurement_eligible:
            continue
        trace_id = str(trace["trace_id"])
        length = int(trace["actual_num_steps"])
        expected_k = list(range(2, length + 1))
        per_k_rows = grouped_corruptions.get(trace_id, [])
        per_k_by_step = {
            int(row["step_index"]): row
            for row in per_k_rows
        }
        if not expected_k or any(k not in per_k_by_step for k in expected_k):
            continue
        grouped_corruptions[trace_id] = [per_k_by_step[k] for k in expected_k]
        eligible_traces.append(trace)

    selected: list[dict[str, Any]] = []
    selected_lengths: set[int] = set()
    for trace in eligible_traces:
        length = int(trace["actual_num_steps"])
        if length in selected_lengths:
            continue
        selected.append(trace)
        selected_lengths.add(length)
        if len(selected) >= samples_per_difficulty:
            break

    if len(selected) < samples_per_difficulty:
        for trace in eligible_traces:
            if trace in selected:
                continue
            selected.append(trace)
            if len(selected) >= samples_per_difficulty:
                break

    if not selected:
        raise ValueError(f"No eligible sample traces were found for difficulty '{difficulty}'.")

    return selected, {
        str(trace["trace_id"]): grouped_corruptions[str(trace["trace_id"])]
        for trace in selected
    }


def _build_meta_payload(
    *,
    sample_id: str,
    trace: dict[str, Any],
    trace_tier: int,
    per_k_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "stage1_sample_meta_v1",
        "sample_id": sample_id,
        "source_trace_id": trace["trace_id"],
        "question_id": trace["question_id"],
        "actual_num_steps": trace["actual_num_steps"],
        "trace_tier": trace_tier,
        "k_values": [int(row["step_index"]) for row in per_k_rows],
        "per_k": [
            {
                "k": int(row["step_index"]),
                "corruption_id": row["corruption_id"],
                "tier": int(row["corruption_tier"]),
                "token_delta": int(row["token_delta"]),
            }
            for row in per_k_rows
        ],
    }


def _build_clean_payload(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage1_sample_clean_v1",
        "steps": [str(step) for step in trace.get("steps", [])],
        "raw_completion": str(trace.get("raw_completion", "")),
        "final_answer_line": trace["final_answer_line"],
    }


def _build_corrupt_payload(
    *,
    trace: dict[str, Any],
    corruption_row: dict[str, Any],
) -> dict[str, Any]:
    k = int(corruption_row["step_index"])
    steps = [str(step) for step in trace.get("steps", [])]
    corrupted_steps = list(steps)
    corrupted_steps[k - 1] = str(corruption_row["corrupt_step"])
    corrupted_completion = "\n".join(
        corrupted_steps + ([str(trace["final_answer_line"])] if trace.get("final_answer_line") else [])
    )
    return {
        "schema_version": "stage1_sample_corrupt_v1",
        "step_index": k,
        "steps": corrupted_steps,
        "raw_completion": corrupted_completion,
    }


def _resolve_trace_tier(rows: list[dict[str, Any]]) -> int:
    tiers = {int(row["corruption_tier"]) for row in rows}
    return 1 if tiers == {1} else 2


def _group_by_length(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["actual_num_steps"]), []).append(row)
    return grouped


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _project_group_question_metadata(row: dict[str, Any]) -> dict[str, Any]:
    difficulty_score = row.get("difficulty_score", row.get("difficulty"))
    if not isinstance(difficulty_score, (int, float)):
        raise TypeError("question metadata rows must contain numeric 'difficulty_score'.")
    return {
        "question_id": str(row["question_id"]),
        "accuracy": float(row["accuracy"]),
        "difficulty_score": float(difficulty_score),
        "optimal_length": row.get("optimal_length"),
        "total_samples": int(row["total_samples"]),
        "correct_count": int(row["correct_count"]),
        "natural_length_distribution": dict(row["natural_length_distribution"]),
    }


def _build_question_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_question: dict[str, dict[str, Any]] = {}
    for trace in traces:
        question_id = str(trace["question_id"])
        rows_by_question.setdefault(
            question_id,
            {
                "question_id": question_id,
                "question_text": str(trace["question_text"]),
                "gold_answer": trace["gold_answer"],
            },
        )
    return [rows_by_question[question_id] for question_id in sorted(rows_by_question)]


def _project_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    actual_num_steps = int(row["actual_num_steps"])
    nldd_length_eligible = bool(row.get("nldd_length_eligible", actual_num_steps >= 3))
    nldd_measurement_eligible = bool(
        row.get("nldd_measurement_eligible", bool(row["is_correct"]) and nldd_length_eligible)
    )
    return {
        "trace_id": str(row["trace_id"]),
        "question_id": str(row["question_id"]),
        "actual_num_steps": actual_num_steps,
        "steps": [str(step) for step in row.get("steps", [])],
        "raw_completion": str(row.get("raw_completion", "")),
        "final_answer_line": row.get("final_answer_line"),
        "extracted_answer": row.get("extracted_answer"),
        "is_correct": bool(row["is_correct"]),
        "extraction_failed": bool(row["extraction_failed"]),
        "token_count": row.get("token_count"),
        "timestamp": row.get("timestamp"),
        "nldd_length_eligible": nldd_length_eligible,
        "nldd_measurement_eligible": nldd_measurement_eligible,
    }


def _assign_sample_id(question_id: str, *, sample_id_counts: dict[str, int]) -> str:
    base = _extract_question_code(question_id)
    sample_id_counts[base] = sample_id_counts.get(base, 0) + 1
    ordinal = sample_id_counts[base]
    return base if ordinal == 1 else f"{base}_{ordinal}"


if __name__ == "__main__":
    main()
