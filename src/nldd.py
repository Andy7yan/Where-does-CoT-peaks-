"""NLDD helpers for corruption regeneration and later scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable

from src.prompting import build_nldd_corrupt_prompt
from src.reasoning import CorruptionResult, corrupt_step_text_with_fallbacks
from src.reports import discover_stage1_shard_paths


@dataclass(frozen=True)
class CorruptionSelectionConfig:
    """Selection settings for corruption regeneration."""

    sampled_min_steps: int = 1
    sampled_max_steps: int = 2
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
    use_tier3: bool = False,
    max_perplexity_ratio: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build all-step and sampled corruption records from existing clean traces."""

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
                    token_counter=token_counter,
                    token_delta_max=token_delta_max,
                    retry_limit=retry_limit,
                    use_tier3=use_tier3,
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
            record["failure_tier"]
            for record in records
            if record["failure_tier"] is not None
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


def compute_logit_margin(logits: object, gold_token_id: int) -> float:
    """Compute the logit margin for a gold token."""

    raise NotImplementedError("NLDD margin computation is not implemented yet.")


def measure_nldd(clean_margin: float, corrupt_margin: float) -> float:
    """Compute an NLDD score from clean and corrupt margins."""

    raise NotImplementedError("NLDD scoring is not implemented yet.")


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
