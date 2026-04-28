"""Corruption record generation helpers for the NLDD workflow."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Any, Callable

from src.common.corruption import (
    CorruptionResult,
    DEFAULT_FLOAT_PERTURBATION_RANGE,
    DEFAULT_INTEGER_PERTURBATION_RANGE,
    corrupt_step_text_with_fallbacks,
)
from src.common.prontoqa_paper_corruption import corrupt_prontoqa_step
from src.analysis_phase1.nldd_prompts import build_canonical_corrupt_prompt
from src.data_phase1.prontoqa_paper import PRONTOQA_PAPER_TASK
from src.data_phase2.pipeline import discover_stage1_shard_paths

from src.analysis_phase1.nldd_shared import _load_jsonl_records, _stable_seed, _write_jsonl


@dataclass(frozen=True)
class CorruptionSelectionConfig:
    """Selection settings for corruption regeneration."""

    seed: int = 42
    include_incorrect_traces: bool = False


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
    integer_perturbation_range: tuple[int, int] = DEFAULT_INTEGER_PERTURBATION_RANGE,
    float_perturbation_range: tuple[float, float] = DEFAULT_FLOAT_PERTURBATION_RANGE,
    enable_tier3_semantic_flip: bool = False,
    use_tier3: bool | None = None,
    max_perplexity_ratio: float | None = None,
    task_name: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build full-analysis corruption records from existing clean traces."""

    effective_enable_tier3 = enable_tier3_semantic_flip or bool(use_tier3)
    records_by_mode = {"all_steps": []}

    for shard_id, trace in trace_sources:
        if not selection.include_incorrect_traces and not trace.get("is_correct"):
            continue
        steps = trace.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue

        for step_index in range(2, len(steps) + 1):
            step_text = str(steps[step_index - 1])
            effective_task_name = str(
                task_name or trace.get("task_name") or "gsm8k"
            )
            rng = random.Random(
                _stable_seed(f"{selection.seed}:all_steps:{trace.get('trace_id')}:{step_index}")
            )
            corruption = corrupt_step_for_task(
                step_text=step_text,
                task_name=effective_task_name,
                rng=rng,
                integer_perturbation_range=integer_perturbation_range,
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
                selection_mode="all_steps",
                task_name=effective_task_name,
            )
            records_by_mode["all_steps"].append(record)

    return records_by_mode


def build_corruption_record(
    *,
    shard_id: str,
    trace: dict[str, Any],
    step_index: int,
    clean_step: str,
    corruption: CorruptionResult,
    selection_mode: str,
    task_name: str = "gsm8k",
) -> dict[str, Any]:
    """Materialize one corruption record with prompt text and metadata."""

    prompt = None
    if not corruption.corruption_failed:
        prompt = build_canonical_corrupt_prompt(
            question=str(trace["question_text"]),
            clean_steps=[str(step) for step in trace["steps"]],
            corruption_step_index=step_index,
            corrupt_step=corruption.corrupt_text,
            task_name=task_name,
        )

    return {
        "corruption_id": f"{trace['trace_id']}_step{step_index}_{selection_mode}",
        "selection_mode": selection_mode,
        "task_name": task_name,
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


def corrupt_step_for_task(
    *,
    step_text: str,
    task_name: str,
    rng: random.Random,
    token_counter: Callable[[str], int],
    token_delta_max: int,
    retry_limit: int,
    integer_perturbation_range: tuple[int, int],
    float_perturbation_range: tuple[float, float],
    enable_tier3_semantic_flip: bool,
    max_perplexity_ratio: float | None,
) -> CorruptionResult:
    if task_name == PRONTOQA_PAPER_TASK:
        return corrupt_prontoqa_step(
            step_text,
            rng=rng,
            token_counter=token_counter,
            token_delta_max=token_delta_max,
            retry_limit=retry_limit,
            max_perplexity_ratio=max_perplexity_ratio,
        )

    return corrupt_step_text_with_fallbacks(
        step_text,
        rng=rng,
        integer_perturbation_range=integer_perturbation_range,
        float_perturbation_range=float_perturbation_range,
        enable_tier3_semantic_flip=enable_tier3_semantic_flip,
        token_counter=token_counter,
        token_delta_max=token_delta_max,
        retry_limit=retry_limit,
        max_perplexity_ratio=max_perplexity_ratio,
    )


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
