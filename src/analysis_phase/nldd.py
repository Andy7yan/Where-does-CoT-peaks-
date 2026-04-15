"""NLDD helpers for corruption regeneration, v4 trace selection, and D2 scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Sequence

from src.data_phase2.coarse_analysis import DIFFICULTY_ORDER, assign_length_bin, dedupe_traces_for_analysis
from src.data_phase1.prompting import build_nldd_clean_prompt, build_nldd_corrupt_prompt
from src.common.reasoning import (
    CorruptionResult,
    DEFAULT_FLOAT_PERTURBATION_RANGE,
    corrupt_step_text_with_fallbacks,
)
from src.data_phase2.aggregation import discover_stage1_shard_paths, load_stage1_traces


TRACE_SELECTION_REQUIRED_COLUMNS = (
    "trace_id",
    "question_id",
    "difficulty",
    "length_bin",
    "selected_for_nldd",
    "selected_for_near_lstar",
)
TRACE_SELECTION_FIELDNAMES = (
    "trace_id",
    "question_id",
    "difficulty",
    "length_bin",
    "raw_length_bin",
    "actual_clean_length",
    "prompt_id",
    "selected_for_nldd",
    "selected_for_near_lstar",
    "selection_mode",
    "near_lstar_selection_mode",
)


@dataclass(frozen=True)
class CorruptionSelectionConfig:
    """Selection settings for corruption regeneration."""

    sampled_min_steps: int = 1
    sampled_max_steps: int = 2
    seed: int = 42
    include_incorrect_traces: bool = False


@dataclass(frozen=True)
class TraceSelectionConfig:
    """Selection settings for the v4 full-run NLDD trace sweep."""

    target_traces_per_cell: int
    target_traces_near_lstar: int
    per_question_trace_cap: int
    min_nldd_length: int
    seed: int = 42


def load_trace_sources(run_dir: str) -> list[tuple[str, dict[str, Any]]]:
    """Load traces from a run root or from per-shard trace files."""

    run_path = Path(run_dir)
    shard_paths = discover_stage1_shard_paths(run_path)
    records: list[tuple[str, dict[str, Any]]] = []
    if shard_paths:
        for shard_path in shard_paths:
            shard_id = shard_path.parent.name
            records.extend((shard_id, row) for row in _load_jsonl_records(shard_path))
        return records

    root_traces = run_path / "traces.jsonl"
    if root_traces.exists():
        return [("root", row) for row in _load_jsonl_records(root_traces)]

    raise FileNotFoundError(
        f"No traces found under '{run_dir}'. Expected traces.jsonl or shard traces."
    )


def build_corruption_records(
    trace_sources: list[tuple[str, dict[str, Any]]],
    *,
    token_counter: Callable[[str], int],
    token_delta_max: int,
    retry_limit: int,
    selection: CorruptionSelectionConfig,
    float_perturbation_range: tuple[float, float, float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
    use_tier3: bool | None = None,
    max_perplexity_ratio: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build all-step and sampled corruption records from existing clean traces."""

    effective_enable_tier3 = enable_tier3_semantic_flip or bool(use_tier3)
    records_by_mode = {
        "all_steps": [],
        "sampled_steps": [],
    }

    for shard_id, trace in trace_sources:
        if not selection.include_incorrect_traces and not trace.get("is_correct"):
            continue
        steps = trace.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue

        all_indices = list(range(1, len(steps) + 1))
        sampled_indices = sample_step_indices_for_trace(
            trace_id=str(trace.get("trace_id", "")),
            num_steps=len(steps),
            sampled_min_steps=selection.sampled_min_steps,
            sampled_max_steps=selection.sampled_max_steps,
            seed=selection.seed,
        )
        for mode_name, indices in (
            ("all_steps", all_indices),
            ("sampled_steps", sampled_indices),
        ):
            for step_index in indices:
                step_text = str(steps[step_index - 1])
                rng = random.Random(_stable_seed(f"{selection.seed}:{mode_name}:{trace.get('trace_id')}:{step_index}"))
                corruption = corrupt_step_text_with_fallbacks(
                    step_text,
                    rng=rng,
                    float_perturbation_range=float_perturbation_range,
                    enable_tier3_semantic_flip=effective_enable_tier3,
                    token_counter=token_counter,
                    token_delta_max=token_delta_max,
                    retry_limit=retry_limit,
                    max_perplexity_ratio=max_perplexity_ratio,
                )
                record = build_corruption_record(
                    shard_id=shard_id,
                    trace=trace,
                    step_index=step_index,
                    clean_step=step_text,
                    corruption=corruption,
                    selection_mode=mode_name,
                )
                records_by_mode[mode_name].append(record)

    return records_by_mode


def sample_step_indices_for_trace(
    *,
    trace_id: str,
    num_steps: int,
    sampled_min_steps: int,
    sampled_max_steps: int,
    seed: int,
) -> list[int]:
    """Return deterministic sampled step indices for a trace."""

    if num_steps <= 0:
        return []

    lower = max(1, sampled_min_steps)
    upper = max(lower, sampled_max_steps)
    generator = random.Random(_stable_seed(f"{seed}:{trace_id}:sampled"))
    sample_size = generator.randint(lower, upper)
    sample_size = min(sample_size, num_steps)
    return sorted(generator.sample(list(range(1, num_steps + 1)), k=sample_size))


def build_corruption_record(
    *,
    shard_id: str,
    trace: dict[str, Any],
    step_index: int,
    clean_step: str,
    corruption: CorruptionResult,
    selection_mode: str,
) -> dict[str, Any]:
    """Materialize one corruption record with prompt text and metadata."""

    prompt = None
    if not corruption.corruption_failed:
        prompt = build_nldd_corrupt_prompt(
            question=str(trace["question_text"]),
            clean_steps=[str(step) for step in trace["steps"]],
            corrupt_step=corruption.corrupt_text,
            corrupt_index=step_index - 1,
        )

    return {
        "corruption_id": f"{trace['trace_id']}_step{step_index}_{selection_mode}",
        "selection_mode": selection_mode,
        "source_shard_id": shard_id,
        "trace_id": str(trace["trace_id"]),
        "question_id": str(trace["question_id"]),
        "prompt_id": str(trace.get("prompt_id", "")),
        "step_index": step_index,
        "actual_num_steps": int(trace.get("actual_num_steps", len(trace["steps"]))),
        "clean_step": clean_step,
        "corrupt_step": corruption.corrupt_text if not corruption.corruption_failed else None,
        "corrupt_prompt": prompt,
        "corruption_failed": corruption.corruption_failed,
        "corruption_tier": corruption.corruption_tier,
        "corruption_type": corruption.corruption_type,
        "original_fragment": corruption.original_fragment,
        "corrupted_fragment": corruption.corrupted_fragment,
        "token_delta": corruption.token_delta,
        "attempts": corruption.attempts,
        "perplexity_ratio": corruption.perplexity_ratio,
        "failure_tier": corruption.failure_tier,
    }


def summarize_corruption_records(records_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build compact summary metrics for generated corruption records."""

    summary: dict[str, Any] = {}
    for mode_name, records in records_by_mode.items():
        failure_count = sum(1 for record in records if record["corruption_failed"])
        tier_counts = Counter(record["corruption_tier"] for record in records)
        type_counts = Counter(record["corruption_type"] for record in records)
        failure_tiers = Counter(
            record.get("failure_tier")
            for record in records
            if record.get("failure_tier") is not None
        )
        summary[mode_name] = {
            "records": len(records),
            "failures": failure_count,
            "failure_rate": (failure_count / len(records)) if records else 0.0,
            "corruption_tier_counts": dict(tier_counts),
            "corruption_type_counts": dict(type_counts),
            "failure_tier_counts": dict(failure_tiers),
        }
    return summary


def write_corruption_artifacts(
    output_dir: str,
    *,
    records_by_mode: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write corruption records and run metadata to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for mode_name, records in records_by_mode.items():
        path = output_path / f"{mode_name}.jsonl"
        _write_jsonl(path, records)
        paths[f"{mode_name}_path"] = str(path)

    summary_path = output_path / "corruption_summary.json"
    summary_path.write_text(
        json.dumps({"metadata": metadata, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["summary_path"] = str(summary_path)
    return paths


def build_v4_trace_selection(
    *,
    traces: list[dict[str, Any]],
    question_metadata: list[dict[str, Any]],
    coarse_analysis: dict[str, Any],
    selection_config: TraceSelectionConfig,
) -> list[dict[str, Any]]:
    """Build the v4 full-run trace selection table from Stage C artifacts."""

    metadata_by_question = {
        str(row["question_id"]): row
        for row in question_metadata
    }
    deduped_traces = dedupe_traces_for_analysis(traces)
    candidate_rows: list[dict[str, Any]] = []
    for trace in deduped_traces:
        if not trace.get("is_correct"):
            continue
        actual_length = int(trace.get("actual_num_steps", 0))
        if actual_length < selection_config.min_nldd_length:
            continue
        question_meta = metadata_by_question.get(str(trace["question_id"]))
        if not question_meta:
            continue
        difficulty = question_meta.get("difficulty_bucket")
        if difficulty not in DIFFICULTY_ORDER:
            continue

        difficulty_payload = coarse_analysis.get("difficulties", {}).get(difficulty, {})
        tertiles = difficulty_payload.get("length_tertiles", {})
        q33 = tertiles.get("q33")
        q67 = tertiles.get("q67")
        if q33 is None or q67 is None:
            continue

        raw_length_bin = assign_length_bin(actual_length, float(q33), float(q67))
        length_bin_map = difficulty_payload.get("length_bin_map", {})
        length_bin = length_bin_map.get(raw_length_bin) or raw_length_bin
        candidate_rows.append(
            {
                "trace_id": str(trace["trace_id"]),
                "question_id": str(trace["question_id"]),
                "difficulty": str(difficulty),
                "length_bin": str(length_bin),
                "raw_length_bin": str(raw_length_bin),
                "actual_clean_length": actual_length,
                "prompt_id": str(trace.get("prompt_id", "")),
                "selected_for_nldd": False,
                "selected_for_near_lstar": False,
                "selection_mode": None,
                "near_lstar_selection_mode": None,
            }
        )

    if not candidate_rows:
        return []

    rows_by_trace_id = {
        row["trace_id"]: row
        for row in candidate_rows
    }
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_cell[(row["difficulty"], row["length_bin"])].append(row)

    for difficulty, difficulty_payload in coarse_analysis.get("difficulties", {}).items():
        for entry in difficulty_payload.get("merged_length_bins", []):
            label = entry.get("label")
            if not isinstance(label, str):
                continue
            selected_rows, selection_mode = _select_rows_for_measurement(
                rows=by_cell.get((difficulty, label), []),
                target=selection_config.target_traces_per_cell,
                per_question_cap=selection_config.per_question_trace_cap,
                seed=selection_config.seed,
                salt=f"cell:{difficulty}:{label}",
            )
            for row in selected_rows:
                stored = rows_by_trace_id[row["trace_id"]]
                stored["selected_for_nldd"] = True
                stored["selection_mode"] = selection_mode

    for difficulty, difficulty_payload in coarse_analysis.get("difficulties", {}).items():
        selected_window = difficulty_payload.get("near_lstar", {}).get("selected_window")
        if not isinstance(selected_window, list) or len(selected_window) != 2:
            continue
        left, right = float(selected_window[0]), float(selected_window[1])
        difficulty_rows = [
            row
            for row in candidate_rows
            if row["difficulty"] == difficulty and left <= row["actual_clean_length"] <= right
        ]
        preferred_ids = {
            row["trace_id"]
            for row in difficulty_rows
            if row["selected_for_nldd"]
        }
        selected_rows, selection_mode = _select_rows_for_measurement(
            rows=difficulty_rows,
            target=selection_config.target_traces_near_lstar,
            per_question_cap=selection_config.per_question_trace_cap,
            seed=selection_config.seed,
            salt=f"near_lstar:{difficulty}:{left}:{right}",
            preferred_trace_ids=preferred_ids,
        )
        for row in selected_rows:
            stored = rows_by_trace_id[row["trace_id"]]
            stored["selected_for_near_lstar"] = True
            stored["near_lstar_selection_mode"] = selection_mode

    selected_union = [
        row
        for row in rows_by_trace_id.values()
        if row["selected_for_nldd"] or row["selected_for_near_lstar"]
    ]
    return sorted(
        selected_union,
        key=lambda row: (
            DIFFICULTY_ORDER.index(row["difficulty"]),
            row["length_bin"],
            row["actual_clean_length"],
            row["question_id"],
            row["trace_id"],
        ),
    )


def load_trace_selection(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the v4 trace-selection CSV."""

    selection_path = Path(path)
    if not selection_path.exists():
        raise FileNotFoundError(f"Trace selection file not found: {selection_path}")

    with selection_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            name
            for name in TRACE_SELECTION_REQUIRED_COLUMNS
            if name not in fieldnames
        ]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"trace_selection.csv is missing required columns: {joined}")

        rows: list[dict[str, Any]] = []
        seen_trace_ids: set[str] = set()
        for raw_row in reader:
            row = dict(raw_row)
            row["trace_id"] = str(row["trace_id"])
            if row["trace_id"] in seen_trace_ids:
                raise ValueError(f"trace_selection.csv contains duplicate trace_id: {row['trace_id']}")
            seen_trace_ids.add(row["trace_id"])
            row["question_id"] = str(row["question_id"])
            row["difficulty"] = str(row["difficulty"])
            row["length_bin"] = str(row["length_bin"])
            row["raw_length_bin"] = _normalize_optional_string(row.get("raw_length_bin"))
            row["prompt_id"] = str(row.get("prompt_id", ""))
            row["selection_mode"] = _normalize_optional_string(row.get("selection_mode"))
            row["near_lstar_selection_mode"] = _normalize_optional_string(
                row.get("near_lstar_selection_mode")
            )
            row["selected_for_nldd"] = _parse_bool(row["selected_for_nldd"])
            row["selected_for_near_lstar"] = _parse_bool(row["selected_for_near_lstar"])
            row["actual_clean_length"] = int(row.get("actual_clean_length") or 0)
            rows.append(row)

    if not rows:
        raise ValueError("trace_selection.csv is empty.")
    if not any(row["selected_for_nldd"] or row["selected_for_near_lstar"] for row in rows):
        raise ValueError("trace_selection.csv does not contain any selected traces.")
    return rows


def write_trace_selection(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the v4 trace-selection CSV."""

    selection_path = Path(path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACE_SELECTION_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            serialized = {name: row.get(name) for name in TRACE_SELECTION_FIELDNAMES}
            serialized["selected_for_nldd"] = "true" if row.get("selected_for_nldd") else "false"
            serialized["selected_for_near_lstar"] = (
                "true" if row.get("selected_for_near_lstar") else "false"
            )
            writer.writerow(serialized)


def build_prompt_logit_fn(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    torch_module: Any,
) -> Callable[[str], Any]:
    """Create a prompt -> final-token-logits scorer."""

    def score_prompt(prompt: str) -> Any:
        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prepared_inputs = _move_model_inputs_to_device(model_inputs, device)
        with torch_module.no_grad():
            outputs = model(**prepared_inputs)
        return outputs.logits[0, -1, :]

    return score_prompt


def calibrate_s(
    clean_traces: list[dict[str, Any]],
    *,
    prompt_logits_fn: Callable[[str], Any],
) -> float:
    """Calibrate the global normalization constant S on all correct traces."""

    std_values: list[float] = []
    for trace in clean_traces:
        if not trace.get("is_correct"):
            continue
        steps = [str(step) for step in trace.get("steps", [])]
        if not steps:
            continue
        prompt = build_nldd_clean_prompt(
            question=str(trace["question_text"]),
            steps=steps,
        )
        logits = prompt_logits_fn(prompt)
        std_values.append(_compute_vector_std(logits))

    if not std_values:
        raise ValueError("Cannot calibrate S without at least one correct trace with steps.")
    return sum(std_values) / len(std_values)


def compute_logit_margin(
    logits: object,
    correct_token_ids: Sequence[int],
    s_value: float,
) -> float:
    """Compute the standardized LD margin for one final-token logit vector."""

    if s_value <= 0:
        raise ValueError("s_value must be positive.")
    if not correct_token_ids:
        raise ValueError("correct_token_ids must not be empty.")

    values = _flatten_numeric_values(logits)
    vocab_size = len(values)
    normalized_ids = sorted({int(token_id) for token_id in correct_token_ids})
    invalid_ids = [token_id for token_id in normalized_ids if token_id < 0 or token_id >= vocab_size]
    if invalid_ids:
        raise IndexError(f"Correct token ids out of bounds for logits size {vocab_size}: {invalid_ids[:5]}")
    if len(normalized_ids) >= vocab_size:
        raise ValueError("correct_token_ids must not cover the entire vocabulary.")

    correct_max = max(values[token_id] for token_id in normalized_ids)
    correct_id_set = set(normalized_ids)
    incorrect_max = max(
        value
        for index, value in enumerate(values)
        if index not in correct_id_set
    )
    return (correct_max - incorrect_max) / s_value


def measure_nldd(
    clean_margin: float,
    corrupt_margin: float,
    *,
    ld_epsilon: float,
) -> float | None:
    """Compute the NLDD percentage drop when the clean LD is stable enough."""

    if abs(clean_margin) < ld_epsilon:
        return None
    return ((clean_margin - corrupt_margin) / abs(clean_margin)) * 100.0


def build_correct_token_ids(gold_answer: float | int | str, tokenizer: Any) -> list[int]:
    """Build the candidate correct-answer token ids for first-token LD scoring."""

    variants = _format_gold_answer_variants(gold_answer)
    token_ids: set[int] = set()
    for variant in variants:
        encoded = tokenizer(
            variant,
            add_special_tokens=False,
        )
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
        flattened = _flatten_token_ids(input_ids)
        if flattened:
            token_ids.add(int(flattened[0]))
    if not token_ids:
        raise ValueError(f"Could not tokenize any correct-answer variants for gold answer {gold_answer!r}.")
    return sorted(token_ids)


def measure_trace_profile(
    *,
    trace: dict[str, Any],
    selection_row: dict[str, Any],
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    token_counter: Callable[[str], int],
    s_value: float,
    ld_epsilon: float,
    seed: int,
    token_delta_max: int,
    retry_limit: int,
    float_perturbation_range: tuple[float, float, float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
) -> list[dict[str, Any]]:
    """Measure the full NLDD profile for one selected clean trace."""

    steps = [str(step) for step in trace.get("steps", [])]
    actual_clean_length = int(trace.get("actual_num_steps", len(steps)))
    if not steps or actual_clean_length <= 0:
        return []

    correct_token_ids = build_correct_token_ids(trace["gold_answer"], tokenizer)
    clean_prompt = build_nldd_clean_prompt(
        question=str(trace["question_text"]),
        steps=steps,
    )
    clean_logits = prompt_logits_fn(clean_prompt)
    ld_clean = compute_logit_margin(clean_logits, correct_token_ids, s_value)
    low_ld_clean = abs(ld_clean) < ld_epsilon

    rows: list[dict[str, Any]] = []
    for step_index in range(1, actual_clean_length + 1):
        step_text = steps[step_index - 1]
        rng = random.Random(_stable_seed(f"{seed}:full:{trace['trace_id']}:{step_index}"))
        corruption = corrupt_step_text_with_fallbacks(
            step_text,
            rng=rng,
            float_perturbation_range=float_perturbation_range,
            enable_tier3_semantic_flip=enable_tier3_semantic_flip,
            token_counter=token_counter,
            token_delta_max=token_delta_max,
            retry_limit=retry_limit,
            max_perplexity_ratio=None,
        )
        corruption_record = build_corruption_record(
            shard_id="root",
            trace=trace,
            step_index=step_index,
            clean_step=step_text,
            corruption=corruption,
            selection_mode="full",
        )

        ld_corrupt: float | None = None
        nldd_value: float | None = None
        exclusion_reason: str | None = None
        if low_ld_clean:
            exclusion_reason = "low_ld_clean"
        elif corruption.corruption_failed:
            exclusion_reason = "corruption_failed"
        else:
            corrupt_logits = prompt_logits_fn(str(corruption_record["corrupt_prompt"]))
            ld_corrupt = compute_logit_margin(corrupt_logits, correct_token_ids, s_value)
            nldd_value = measure_nldd(
                ld_clean,
                ld_corrupt,
                ld_epsilon=ld_epsilon,
            )

        rows.append(
            {
                "nldd_id": f"{trace['trace_id']}_step{step_index}",
                "question_id": str(trace["question_id"]),
                "clean_trace_id": str(trace["trace_id"]),
                "difficulty": str(selection_row["difficulty"]),
                "length_bin": str(selection_row["length_bin"]),
                "raw_length_bin": selection_row.get("raw_length_bin"),
                "selected_for_near_lstar": bool(selection_row["selected_for_near_lstar"]),
                "prompt_id": str(trace.get("prompt_id", "")),
                "actual_clean_length": actual_clean_length,
                "corruption_step_index": step_index,
                "corruption_tier": corruption.corruption_tier,
                "corruption_type": corruption.corruption_type,
                "corruption_failed": corruption.corruption_failed,
                "failure_tier": corruption.failure_tier,
                "ld_clean": ld_clean,
                "ld_corrupt": ld_corrupt,
                "nldd_value": nldd_value,
                "measurement_exclusion_reason": exclusion_reason,
                "selection_mode": selection_row.get("selection_mode"),
                "near_lstar_selection_mode": selection_row.get("near_lstar_selection_mode"),
                "corruption_id": corruption_record["corruption_id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    horizon = extract_trace_horizon(rows)
    for row in rows:
        row["k_star_trace"] = horizon["k_star_trace"]
        row["r_star_trace"] = horizon["r_star_trace"]
    return rows


def extract_trace_horizon(profile_rows: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    """Locate the per-trace peak-based horizon using the v4 tie-breaking rule."""

    valid_rows = [
        row
        for row in profile_rows
        if row.get("nldd_value") is not None and int(row["corruption_step_index"]) > 1
    ]
    if not valid_rows:
        return {
            "k_star_trace": None,
            "r_star_trace": None,
        }

    best_row = max(
        valid_rows,
        key=lambda row: (float(row["nldd_value"]), -int(row["corruption_step_index"])),
    )
    k_star = int(best_row["corruption_step_index"])
    length = int(best_row["actual_clean_length"])
    return {
        "k_star_trace": k_star,
        "r_star_trace": (k_star / length) if length > 0 else None,
    }


def measure_selected_traces(
    *,
    traces_by_id: dict[str, dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    token_counter: Callable[[str], int],
    s_value: float,
    ld_epsilon: float,
    seed: int,
    token_delta_max: int,
    retry_limit: int,
    float_perturbation_range: tuple[float, float, float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure all selected traces and return rows plus a compact summary."""

    measurement_rows: list[dict[str, Any]] = []
    selected_rows = [
        row
        for row in selection_rows
        if row["selected_for_nldd"] or row["selected_for_near_lstar"]
    ]
    for selection_row in selected_rows:
        trace_id = str(selection_row["trace_id"])
        trace = traces_by_id.get(trace_id)
        if trace is None:
            raise KeyError(f"Trace selection references missing trace_id: {trace_id}")
        measurement_rows.extend(
            measure_trace_profile(
                trace=trace,
                selection_row=selection_row,
                prompt_logits_fn=prompt_logits_fn,
                tokenizer=tokenizer,
                token_counter=token_counter,
                s_value=s_value,
                ld_epsilon=ld_epsilon,
                seed=seed,
                token_delta_max=token_delta_max,
                retry_limit=retry_limit,
                float_perturbation_range=float_perturbation_range,
                enable_tier3_semantic_flip=enable_tier3_semantic_flip,
            )
        )

    corruption_summary = summarize_corruption_records({"full": measurement_rows})["full"]
    trace_ids_with_low_ld = {
        str(row["clean_trace_id"])
        for row in measurement_rows
        if row.get("measurement_exclusion_reason") == "low_ld_clean"
    }
    summary = {
        "selected_trace_count": len(selected_rows),
        "selected_for_nldd_count": sum(1 for row in selected_rows if row["selected_for_nldd"]),
        "selected_for_near_lstar_count": sum(
            1 for row in selected_rows if row["selected_for_near_lstar"]
        ),
        "measured_row_count": len(measurement_rows),
        "valid_nldd_count": sum(1 for row in measurement_rows if row["nldd_value"] is not None),
        "low_ld_clean_trace_count": len(trace_ids_with_low_ld),
        "corruption": corruption_summary,
    }
    return measurement_rows, summary


def validate_nldd_full_records(
    *,
    selection_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Validate the Stage D v4 handoff invariants for nldd_full.jsonl."""

    if not selection_rows:
        raise ValueError("trace_selection.csv is empty.")
    if not records:
        raise ValueError("nldd_full.jsonl is empty.")

    expected_by_trace_id = {
        str(row["trace_id"]): int(row["actual_clean_length"])
        for row in selection_rows
        if row["selected_for_nldd"] or row["selected_for_near_lstar"]
    }
    actual_steps_by_trace_id: dict[str, set[int]] = defaultdict(set)
    for row in records:
        trace_id = str(row["clean_trace_id"])
        actual_steps_by_trace_id[trace_id].add(int(row["corruption_step_index"]))

    missing_trace_ids = sorted(set(expected_by_trace_id) - set(actual_steps_by_trace_id))
    if missing_trace_ids:
        sample = ", ".join(missing_trace_ids[:5])
        raise ValueError(f"nldd_full.jsonl is missing selected traces: {sample}")

    for trace_id, expected_length in expected_by_trace_id.items():
        expected_steps = set(range(1, expected_length + 1))
        actual_steps = actual_steps_by_trace_id[trace_id]
        if actual_steps != expected_steps:
            missing_steps = sorted(expected_steps - actual_steps)
            extra_steps = sorted(actual_steps - expected_steps)
            raise ValueError(
                "nldd_full.jsonl does not cover the full sweep for trace "
                f"{trace_id}: missing={missing_steps[:5]}, extra={extra_steps[:5]}"
            )


def write_s_calibration(
    path: str | Path,
    *,
    s_value: float,
    trace_count: int,
) -> None:
    """Persist the calibrated S constant for downstream provenance."""

    calibration_path = Path(path)
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": "stage1_s_calibration_v1",
                "s_value": s_value,
                "trace_count": trace_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_nldd_full_records(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the Stage D v4 nldd_full.jsonl file."""

    _write_jsonl(Path(path), rows)


def load_question_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Load the Stage C question metadata JSONL."""

    return _load_jsonl_records(Path(path))


def load_coarse_analysis(path: str | Path) -> dict[str, Any]:
    """Load the Stage C coarse-analysis JSON."""

    coarse_path = Path(path)
    with coarse_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("coarse_analysis.json must contain a top-level mapping.")
    return data


def load_or_build_trace_selection(
    *,
    run_dir: str,
    traces: list[dict[str, Any]],
    question_metadata: list[dict[str, Any]],
    coarse_analysis: dict[str, Any],
    selection_config: TraceSelectionConfig,
) -> tuple[list[dict[str, Any]], str]:
    """Load trace_selection.csv if present, otherwise build the v4 selection."""

    run_path = Path(run_dir)
    selection_path = run_path / "trace_selection.csv"
    if selection_path.exists():
        return load_trace_selection(selection_path), "loaded"

    rows = build_v4_trace_selection(
        traces=traces,
        question_metadata=question_metadata,
        coarse_analysis=coarse_analysis,
        selection_config=selection_config,
    )
    if not rows:
        raise ValueError("Could not build a non-empty v4 trace selection from the available artifacts.")
    write_trace_selection(selection_path, rows)
    return rows, "built"


def compute_v4_measurement_artifacts(
    *,
    run_dir: str,
    question_metadata: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    prompt_logits_fn: Callable[[str], Any],
    tokenizer: Any,
    token_counter: Callable[[str], int],
    seed: int,
    token_delta_max: int,
    retry_limit: int,
    ld_epsilon: float,
    float_perturbation_range: tuple[float, float, float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
) -> dict[str, Any]:
    """Compute and persist the v4 Stage D measurement artifacts for one run."""

    run_path = Path(run_dir)
    traces = load_stage1_traces(run_path)
    traces_by_id = {
        str(trace["trace_id"]): trace
        for trace in traces
    }
    correct_traces = [
        trace
        for trace in traces
        if trace.get("is_correct") and trace.get("steps")
    ]
    if not correct_traces:
        raise ValueError("Stage D2 requires at least one correct clean trace.")

    s_value = calibrate_s(
        correct_traces,
        prompt_logits_fn=prompt_logits_fn,
    )
    measurement_rows, measurement_summary = measure_selected_traces(
        traces_by_id=traces_by_id,
        selection_rows=selection_rows,
        prompt_logits_fn=prompt_logits_fn,
        tokenizer=tokenizer,
        token_counter=token_counter,
        s_value=s_value,
        ld_epsilon=ld_epsilon,
        seed=seed,
        token_delta_max=token_delta_max,
        retry_limit=retry_limit,
        float_perturbation_range=float_perturbation_range,
        enable_tier3_semantic_flip=enable_tier3_semantic_flip,
    )
    validate_nldd_full_records(
        selection_rows=selection_rows,
        records=measurement_rows,
    )

    s_path = run_path / "s_calibration.json"
    nldd_path = run_path / "nldd_full.jsonl"
    summary_path = run_path / "corruption_summary.json"
    write_s_calibration(
        s_path,
        s_value=s_value,
        trace_count=len(correct_traces),
    )
    write_nldd_full_records(nldd_path, measurement_rows)
    summary_payload = {
        "metadata": {
            "schema_version": "stage1_nldd_measurement_summary_v1",
            "run_dir": str(run_path),
            "selection_trace_count": len(selection_rows),
            "question_metadata_count": len(question_metadata),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": measurement_summary,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "s_value": s_value,
        "s_calibration_path": str(s_path),
        "nldd_full_path": str(nldd_path),
        "corruption_summary_path": str(summary_path),
        "measurement_summary": measurement_summary,
    }


def _select_rows_for_measurement(
    *,
    rows: list[dict[str, Any]],
    target: int,
    per_question_cap: int,
    seed: int,
    salt: str,
    preferred_trace_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if not rows:
        return [], "empty"
    if len(rows) <= target:
        return list(rows), "all_available"

    preferred_trace_ids = preferred_trace_ids or set()
    ordered_rows = sorted(
        rows,
        key=lambda row: _stable_seed(f"{seed}:{salt}:{row['trace_id']}"),
    )
    preferred_rows = [row for row in ordered_rows if row["trace_id"] in preferred_trace_ids]
    fallback_rows = [row for row in ordered_rows if row["trace_id"] not in preferred_trace_ids]

    selected: list[dict[str, Any]] = []
    question_counts: Counter[str] = Counter()
    for row in [*preferred_rows, *fallback_rows]:
        question_id = str(row["question_id"])
        if question_counts[question_id] >= per_question_cap:
            continue
        selected.append(row)
        question_counts[question_id] += 1
        if len(selected) >= target:
            break

    if len(selected) >= target:
        if preferred_rows:
            return selected, "preferred_reuse"
        if per_question_cap < len(rows):
            return selected, "capped_random_sample"
        return selected, "random_sample"

    selected_trace_ids = {row["trace_id"] for row in selected}
    relaxed_rows = [
        row
        for row in ordered_rows
        if row["trace_id"] not in selected_trace_ids
    ]
    return [*selected, *relaxed_rows], "all_after_cap"


def _format_gold_answer_variants(gold_answer: float | int | str) -> list[str]:
    if isinstance(gold_answer, str):
        base = gold_answer.strip()
    elif isinstance(gold_answer, int):
        base = str(gold_answer)
    else:
        numeric = float(gold_answer)
        if numeric.is_integer():
            base = str(int(numeric))
        else:
            base = format(numeric, "g")
    variants = [base]
    if not base.startswith(" "):
        variants.append(f" {base}")
    return variants


def _flatten_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            flattened: list[int] = []
            for item in value:
                flattened.extend(_flatten_token_ids(item))
            return flattened
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    return [int(value)]


def _flatten_numeric_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    return [float(value)]


def _compute_vector_std(logits: Any) -> float:
    values = _flatten_numeric_values(logits)
    if not values:
        raise ValueError("Cannot compute a standard deviation from an empty logit vector.")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _move_model_inputs_to_device(model_inputs: Any, device: Any) -> dict[str, Any]:
    if isinstance(model_inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
    raise TypeError("Tokenizer returned an unsupported input type for NLDD scoring.")


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Could not parse a boolean value from {value!r}.")


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
