"""Export per-difficulty analysis handoff artifacts for canonical runs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable

from src.common.settings import ExperimentConfig, require_config_value
from src.data_phase1.gsm8k import build_ranked_questions, load_gsm8k_test
from src.data_phase1.pilot import build_token_counter
from src.data_phase2.coarse_analysis import DIFFICULTY_ORDER


def export_difficulty_length_groups(
    run_path: Path,
    *,
    config_path: str = "configs/stage1.yaml",
) -> dict[str, Any]:
    """Write the spec-oriented difficulty handoff directory."""

    config = ExperimentConfig.from_yaml(config_path)
    min_nldd_length = require_config_value(
        "analysis.min_nldd_length",
        config.analysis.min_nldd_length,
    )
    min_cell_size = require_config_value(
        "analysis.min_cell_size",
        config.analysis.min_cell_size,
    )
    target_traces_per_cell = require_config_value(
        "analysis.target_traces_per_cell",
        config.analysis.target_traces_per_cell,
    )

    traces = _load_jsonl(run_path / "traces.jsonl")
    metadata_rows = _load_jsonl(run_path / "question_metadata.jsonl")
    metadata_by_question = {
        str(row["question_id"]): row
        for row in metadata_rows
    }

    output_dir = run_path / "difficulty"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_counter = build_token_counter(tokenizer=None, approximate=True)
    question_lookup = _build_question_lookup(
        traces=traces,
        config=config,
    )
    grouped_traces: dict[str, list[dict[str, Any]]] = {
        difficulty: []
        for difficulty in DIFFICULTY_ORDER
    }
    grouped_metadata: dict[str, list[dict[str, Any]]] = {
        difficulty: []
        for difficulty in DIFFICULTY_ORDER
    }
    grouped_questions: dict[str, list[dict[str, Any]]] = {
        difficulty: []
        for difficulty in DIFFICULTY_ORDER
    }

    for trace in sorted(
        traces,
        key=lambda row: (
            str(row.get("question_id", "")),
            int(row.get("actual_num_steps", 0)),
            str(row.get("prompt_id", "")),
            str(row.get("trace_id", "")),
        ),
    ):
        question_meta = metadata_by_question.get(str(trace["question_id"]))
        if not question_meta:
            continue
        difficulty = question_meta.get("difficulty_bucket")
        if difficulty not in DIFFICULTY_ORDER:
            continue
        grouped_traces[difficulty].append(
            _enrich_trace_row(
                trace=trace,
                question_meta=question_meta,
                min_nldd_length=min_nldd_length,
            )
        )

    for difficulty in DIFFICULTY_ORDER:
        question_ids = sorted({str(row["question_id"]) for row in grouped_traces[difficulty]})
        grouped_metadata[difficulty] = [
            _project_group_question_metadata(metadata_by_question[question_id])
            for question_id in question_ids
        ]
        grouped_questions[difficulty] = _build_question_rows(
            grouped_traces[difficulty],
            question_lookup=question_lookup,
        )

    export_summary: dict[str, Any] = {
        "schema_version": "stage1_difficulty_export_v1",
        "min_nldd_length": min_nldd_length,
        "min_cell_size": min_cell_size,
        "target_traces_per_cell": target_traces_per_cell,
        "difficulties": {},
    }

    for difficulty in DIFFICULTY_ORDER:
        difficulty_dir = output_dir / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)

        _write_jsonl(difficulty_dir / "questions.jsonl", grouped_questions[difficulty])
        _write_jsonl(
            difficulty_dir / "traces.jsonl",
            [_project_group_trace_row(row) for row in grouped_traces[difficulty]],
        )
        _write_jsonl(difficulty_dir / "question_metadata.jsonl", grouped_metadata[difficulty])

        bin_summary = export_bins_for_difficulty(
            difficulty_dir=difficulty_dir,
            difficulty=difficulty,
            traces=grouped_traces[difficulty],
            token_counter=token_counter,
            integer_perturbation_range=tuple(config.nldd.integer_perturbation_range),
            float_perturbation_range=tuple(config.nldd.float_perturbation_range),
            enable_tier3_semantic_flip=config.nldd.enable_tier3_semantic_flip,
            corruption_token_delta_max=config.nldd.corruption_token_delta_max,
            corruption_retry_limit=config.nldd.corruption_retry_limit,
            corruption_seed=config.experiment.seed,
            min_nldd_length=min_nldd_length,
            min_cell_size=min_cell_size,
            target_traces_per_cell=target_traces_per_cell,
        )
        export_summary["difficulties"][difficulty] = {
            "question_count": len(grouped_questions[difficulty]),
            "trace_count": len(grouped_traces[difficulty]),
            **bin_summary,
        }

    return export_summary


def export_bins_for_difficulty(
    *,
    difficulty_dir: Path,
    difficulty: str,
    traces: list[dict[str, Any]],
    token_counter: Callable[[str], int],
    integer_perturbation_range: tuple[int, int],
    float_perturbation_range: tuple[float, float],
    enable_tier3_semantic_flip: bool,
    corruption_token_delta_max: int,
    corruption_retry_limit: int,
    corruption_seed: int,
    min_nldd_length: int,
    min_cell_size: int,
    target_traces_per_cell: int,
) -> dict[str, Any]:
    """Write per-length bins and sample directories for one difficulty bucket."""

    bins_dir = difficulty_dir / "bins"
    bins_dir.mkdir(parents=True, exist_ok=True)

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_length[int(trace["actual_num_steps"])].append(trace)

    kept_bin_count = 0
    selected_sample_count = 0
    length_summaries: dict[str, Any] = {}
    for actual_num_steps in sorted(by_length):
        length_rows = by_length[actual_num_steps]
        eligible_rows = [
            row
            for row in length_rows
            if row["is_correct"] and int(row["actual_num_steps"]) >= min_nldd_length
        ]
        if actual_num_steps < min_nldd_length or len(eligible_rows) < min_cell_size:
            continue

        selected_bundles = build_sample_bundles_for_length(
            difficulty=difficulty,
            traces=eligible_rows,
            token_counter=token_counter,
            integer_perturbation_range=integer_perturbation_range,
            float_perturbation_range=float_perturbation_range,
            enable_tier3_semantic_flip=enable_tier3_semantic_flip,
            corruption_token_delta_max=corruption_token_delta_max,
            corruption_retry_limit=corruption_retry_limit,
            corruption_seed=corruption_seed,
            min_cell_size=min_cell_size,
            target_traces_per_cell=target_traces_per_cell,
        )
        if not selected_bundles:
            continue

        _assign_compact_sample_ids(selected_bundles)

        bin_dir = bins_dir / f"bin_{actual_num_steps}"
        samples_dir = bin_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        selection_rows: list[dict[str, Any]] = []
        for bundle in selected_bundles:
            sample_dir = samples_dir / bundle["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            _write_json(sample_dir / "clean.json", bundle["clean_payload"])
            _write_json(sample_dir / "meta.json", bundle["meta_payload"])
            for k, payload in bundle["corrupt_payloads"].items():
                _write_json(sample_dir / f"corrupt_k{k}.json", payload)
            selection_rows.append(bundle["selection_row"])

        _write_jsonl(bin_dir / "selection.jsonl", selection_rows)
        kept_bin_count += 1
        selected_sample_count += len(selected_bundles)
        length_summaries[str(actual_num_steps)] = {
            "eligible_trace_count": len(eligible_rows),
            "selected_sample_count": len(selected_bundles),
        }

    return {
        "kept_bin_count": kept_bin_count,
        "selected_sample_count": selected_sample_count,
        "bins": length_summaries,
    }


def build_sample_bundles_for_length(
    *,
    difficulty: str,
    traces: list[dict[str, Any]],
    token_counter: Callable[[str], int],
    integer_perturbation_range: tuple[int, int],
    float_perturbation_range: tuple[float, float],
    enable_tier3_semantic_flip: bool,
    corruption_token_delta_max: int,
    corruption_retry_limit: int,
    corruption_seed: int,
    min_cell_size: int,
    target_traces_per_cell: int,
) -> list[dict[str, Any]]:
    """Select fully successful samples for one exact-length bin."""

    from src.analysis_phase1.nldd import CorruptionSelectionConfig, build_corruption_records

    complete_bundles: list[dict[str, Any]] = []
    sorted_traces = sorted(
        traces,
        key=lambda row: _stable_seed(
            f"{difficulty}:{int(row['actual_num_steps'])}:{row['trace_id']}"
        ),
    )
    for trace in sorted_traces:
        records_by_mode = build_corruption_records(
            [("root", trace)],
            token_counter=token_counter,
            token_delta_max=corruption_token_delta_max,
            retry_limit=corruption_retry_limit,
            selection=CorruptionSelectionConfig(seed=corruption_seed),
            integer_perturbation_range=integer_perturbation_range,
            float_perturbation_range=float_perturbation_range,
            enable_tier3_semantic_flip=enable_tier3_semantic_flip,
        )
        bundle = build_trace_sample_bundle(
            trace=trace,
            difficulty=difficulty,
            corruption_rows=records_by_mode["all_steps"],
        )
        if bundle is None:
            continue
        complete_bundles.append(bundle)

    if len(complete_bundles) < min_cell_size:
        return []
    return complete_bundles[:target_traces_per_cell]


def build_trace_sample_bundle(
    *,
    trace: dict[str, Any],
    difficulty: str,
    corruption_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build one sample directory payload, discarding partial corruption sweeps."""

    actual_num_steps = int(trace["actual_num_steps"])
    expected_k_values = list(range(2, actual_num_steps + 1))
    rows_by_k = {
        int(row["step_index"]): row
        for row in corruption_rows
        if str(row.get("trace_id")) == str(trace["trace_id"]) and int(row["step_index"]) >= 2
    }
    if expected_k_values != sorted(rows_by_k):
        return None

    selected_rows = [rows_by_k[k] for k in expected_k_values]
    if any(bool(row["corruption_failed"]) for row in selected_rows):
        return None

    steps = [str(step) for step in trace.get("steps", [])]
    trace_tier = max(int(row["corruption_tier"]) for row in selected_rows)

    clean_payload = {
        "steps": steps,
        "raw_completion": str(trace.get("raw_completion", "")),
        "final_answer_line": trace.get("final_answer_line"),
    }

    corrupt_payloads: dict[int, dict[str, Any]] = {}
    for row in selected_rows:
        k = int(row["step_index"])
        corrupted_steps = list(steps)
        corrupted_steps[k - 1] = str(row["corrupt_step"])
        corrupted_completion = "\n".join(
            corrupted_steps + ([str(trace["final_answer_line"])] if trace.get("final_answer_line") else [])
        )
        corrupt_payloads[k] = {
            "step_index": k,
            "steps": corrupted_steps,
            "raw_completion": corrupted_completion,
        }

    meta_payload = {
        "sample_id": None,
        "source_trace_id": str(trace["trace_id"]),
        "question_id": str(trace["question_id"]),
        "actual_num_steps": actual_num_steps,
        "trace_tier": trace_tier,
        "k_values": expected_k_values,
        "per_k": [
            {
                "k": int(row["step_index"]),
                "tier": int(row["corruption_tier"]),
                "corruption_id": str(row["corruption_id"]),
                "token_delta": row.get("token_delta"),
            }
            for row in selected_rows
        ],
    }
    selection_row = {
        "sample_id": None,
        "question_id": str(trace["question_id"]),
        "source_trace_id": str(trace["trace_id"]),
        "actual_num_steps": actual_num_steps,
        "trace_tier": trace_tier,
        "k_values": expected_k_values,
        "sample_path": None,
    }
    return {
        "sample_id": None,
        "clean_payload": clean_payload,
        "corrupt_payloads": corrupt_payloads,
        "meta_payload": meta_payload,
        "selection_row": selection_row,
    }


def _build_question_rows(
    traces: list[dict[str, Any]],
    *,
    question_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    question_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in traces:
        question_id = str(trace["question_id"])
        if question_id in seen:
            continue
        seen.add(question_id)
        question_row = question_lookup.get(question_id)
        if question_row is None:
            raise KeyError(f"Missing question payload for question_id {question_id!r}.")
        question_rows.append(
            {
                "question_id": question_id,
                "question_text": str(question_row["question_text"]),
                "gold_answer": question_row["gold_answer"],
            }
        )
    return sorted(question_rows, key=lambda row: row["question_id"])


def _build_question_lookup(
    *,
    traces: list[dict[str, Any]],
    config: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    missing_question_ids: set[str] = set()
    for trace in traces:
        question_id = str(trace["question_id"])
        question_text = trace.get("question_text")
        gold_answer = trace.get("gold_answer")
        if isinstance(question_text, str) and question_text.strip() and gold_answer is not None:
            lookup[question_id] = {
                "question_text": question_text,
                "gold_answer": gold_answer,
            }
        else:
            missing_question_ids.add(question_id)

    if not missing_question_ids:
        return lookup

    ranked_questions = build_ranked_questions(
        load_gsm8k_test(
            cache_dir=config.model.hf_cache,
            dataset_name=config.dataset.name,
            dataset_config=config.dataset.hf_config,
            split=config.dataset.split,
        ),
        hash_seed=config.dataset.order_hash_seed,
        dataset_name=config.dataset.name,
        split=config.dataset.split,
    )
    ranked_by_index = {
        _question_index_key(str(row["question_id"])): row
        for row in ranked_questions
    }
    for question_id in sorted(missing_question_ids):
        recovered = ranked_by_index.get(_question_index_key(question_id))
        if recovered is None:
            raise KeyError(
                f"Could not recover question payload for question_id {question_id!r} "
                "from the ranked dataset."
            )
        lookup[question_id] = {
            "question_text": str(recovered["question_text"]),
            "gold_answer": recovered["gold_answer"],
        }
    return lookup


def _question_index_key(question_id: str) -> int | str:
    matches = re.findall(r"\d+", question_id)
    if matches:
        return int(matches[-1])
    return question_id


def _enrich_trace_row(
    *,
    trace: dict[str, Any],
    question_meta: dict[str, Any],
    min_nldd_length: int,
) -> dict[str, Any]:
    row = dict(trace)
    difficulty_score = question_meta.get("difficulty_score", question_meta.get("difficulty"))
    if not isinstance(difficulty_score, (int, float)):
        raise TypeError("question metadata rows must contain numeric 'difficulty_score'.")
    row["question_accuracy"] = float(question_meta["accuracy"])
    row["difficulty_score"] = float(difficulty_score)
    row["difficulty_bucket"] = str(question_meta["difficulty_bucket"])
    row["nldd_min_length"] = min_nldd_length
    row["nldd_length_eligible"] = int(trace["actual_num_steps"]) >= min_nldd_length
    row["nldd_measurement_eligible"] = bool(trace["is_correct"]) and row["nldd_length_eligible"]
    return row


def _project_group_question_metadata(row: dict[str, Any]) -> dict[str, Any]:
    difficulty_score = row.get("difficulty_score", row.get("difficulty"))
    if not isinstance(difficulty_score, (int, float)):
        raise TypeError("question metadata rows must contain numeric 'difficulty_score'.")
    return {
        "question_id": str(row["question_id"]),
        "accuracy": float(row["accuracy"]),
        "difficulty_score": float(difficulty_score),
        "total_samples": int(row["total_samples"]),
        "correct_count": int(row["correct_count"]),
        "natural_length_distribution": dict(row["natural_length_distribution"]),
    }


def _project_group_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": str(row["trace_id"]),
        "question_id": str(row["question_id"]),
        "actual_num_steps": int(row["actual_num_steps"]),
        "steps": [str(step) for step in row.get("steps", [])],
        "raw_completion": str(row.get("raw_completion", "")),
        "final_answer_line": row.get("final_answer_line"),
        "extracted_answer": row.get("extracted_answer"),
        "is_correct": bool(row["is_correct"]),
        "extraction_failed": bool(row["extraction_failed"]),
        "token_count": row.get("token_count"),
        "timestamp": row.get("timestamp"),
        "nldd_length_eligible": bool(row["nldd_length_eligible"]),
        "nldd_measurement_eligible": bool(row["nldd_measurement_eligible"]),
    }


def _assign_compact_sample_ids(bundles: list[dict[str, Any]]) -> None:
    counts_by_base: dict[str, int] = defaultdict(int)
    for bundle in bundles:
        question_id = str(bundle["meta_payload"]["question_id"])
        base = _extract_question_code(question_id)
        counts_by_base[base] += 1
        ordinal = counts_by_base[base]
        sample_id = base if ordinal == 1 else f"{base}_{ordinal}"
        bundle["sample_id"] = sample_id
        bundle["meta_payload"]["sample_id"] = sample_id
        bundle["selection_row"]["sample_id"] = sample_id
        bundle["selection_row"]["sample_path"] = f"samples/{sample_id}"


def _extract_question_code(question_id: str) -> str:
    matches = re.findall(r"\d+", question_id)
    if matches:
        return matches[-1]
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", question_id).strip("_")
    return sanitized or "sample"


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


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
