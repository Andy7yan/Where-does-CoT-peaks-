"""CLI for Stage 1 length-controlled trace generation."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import (
    TRACE_SCHEMA_VERSION,
    LLMGenerator,
    append_traces_to_jsonl,
    generate_traces_for_question,
    load_existing_trace_ids,
    validate_output_dir_schema,
    write_run_metadata,
)
from src.prompting import load_prompt_template
from src.settings import ExperimentConfig


def main() -> None:
    """Run Stage 1 trace generation over an evaluation subset."""

    args = parse_args()
    if args.debug:
        os.environ["PEAK_COT_DEBUG"] = "1"
    config = ExperimentConfig.from_yaml(args.config)
    subset = load_subset(args.subset_path)
    selected_subset = subset[args.start_idx : args.end_idx]

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    shard_id = args.shard_id or build_default_shard_id(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        subset_size=len(subset),
    )
    shard_dir = root_output_dir / "shards" / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_path = shard_dir / "traces.jsonl"
    validate_output_dir_schema(str(shard_dir), expected_schema_version=TRACE_SCHEMA_VERSION)

    run_start = time.perf_counter()
    generator = LLMGenerator(
        model_name=config.model.name,
        dtype=config.model.dtype,
        cache_dir=config.model.hf_cache,
    )
    num_icl_groups = require_config_value(
        "generation.num_icl_groups",
        config.generation.num_icl_groups,
    )
    prompt_templates = discover_prompt_templates(
        prompts_dir="prompts",
        expected_count=num_icl_groups,
        preferred_prompt_ids=config.generation.icl_group_prompt_ids or None,
    )
    samples_per_group = require_config_value(
        "generation.samples_per_group",
        config.generation.samples_per_group,
    )
    prompt_temperatures = resolve_prompt_temperatures(
        prompt_ids=[template["prompt_id"] for template in prompt_templates],
        default_temperature=config.generation.temperature,
        configured_group_temperatures=config.generation.icl_group_temperatures,
    )
    prompt_sample_counts = resolve_prompt_sample_counts(
        prompt_ids=[template["prompt_id"] for template in prompt_templates],
        default_samples_per_group=samples_per_group,
        configured_group_sample_counts=config.generation.icl_group_sample_counts,
    )
    temperature = config.generation.temperature
    max_new_tokens = require_config_value(
        "generation.max_new_tokens",
        config.generation.max_new_tokens,
    )
    prompt_ids = [template["prompt_id"] for template in prompt_templates]
    run_metadata = build_run_metadata(
        config=config,
        prompt_ids=prompt_ids,
        temperature=temperature,
        icl_group_temperatures=prompt_temperatures,
        icl_group_sample_counts=prompt_sample_counts,
        max_new_tokens=max_new_tokens,
        num_icl_groups=num_icl_groups,
        samples_per_group=samples_per_group,
    )
    write_run_metadata(str(shard_dir), run_metadata)
    write_shard_metadata(
        shard_dir=shard_dir,
        shard_id=shard_id,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        selected_questions=len(selected_subset),
    )
    existing_trace_ids = load_existing_trace_ids(str(output_path))

    written_traces = 0
    skipped_traces = 0
    extraction_failed_count = 0

    print(
        "run_config:",
        {
            "output_dir": str(root_output_dir),
            "shard_id": shard_id,
            "selected_questions": len(selected_subset),
            "prompt_ids": prompt_ids,
            "samples_per_group": samples_per_group,
            "icl_group_sample_counts": prompt_sample_counts,
            "max_new_tokens": max_new_tokens,
            "default_temperature": temperature,
        },
    )
    if os.getenv("PEAK_COT_DEBUG") == "1":
        print(
            "run_config_details:",
            {
                "shard_dir": str(shard_dir),
                "icl_group_temperatures": prompt_temperatures,
            },
        )

    total_questions = len(selected_subset)
    for index, record in enumerate(selected_subset, start=1):
        question_start = time.perf_counter()
        traces = generate_traces_for_question(
            generator=generator,
            question_id=record["question_id"],
            question_text=record["question_text"],
            gold_answer=float(record["gold_answer"]),
            prompt_templates=prompt_templates,
            samples_per_group=samples_per_group,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_temperatures=prompt_temperatures,
            prompt_sample_counts=prompt_sample_counts,
        )

        new_traces = [trace for trace in traces if trace["trace_id"] not in existing_trace_ids]
        if new_traces:
            append_traces_to_jsonl(new_traces, str(output_path))
            for trace in new_traces:
                existing_trace_ids.add(trace["trace_id"])

        written_traces += len(new_traces)
        skipped_traces += len(traces) - len(new_traces)
        extraction_failed_count += sum(1 for trace in new_traces if trace["extraction_failed"])

        should_log_question = (
            os.getenv("PEAK_COT_DEBUG") == "1"
            or index == 1
            or index % 10 == 0
            or index == total_questions
        )
        if should_log_question:
            elapsed = time.perf_counter() - run_start
            average_per_question = elapsed / index if index else 0.0
            remaining = max(total_questions - index, 0) * average_per_question
            question_elapsed = time.perf_counter() - question_start
            correct_count = sum(1 for trace in traces if trace["is_correct"])
            question_failures = sum(1 for trace in traces if trace["extraction_failed"])
            avg_tokens = (
                sum(int(trace["token_count"]) for trace in traces) / len(traces)
                if traces
                else 0.0
            )
            print(
                "progress:",
                {
                    "questions_completed": f"{index}/{total_questions}",
                    "last_question_id": record["question_id"],
                    "question_seconds": round(question_elapsed, 2),
                    "question_correct": f"{correct_count}/{len(traces)}",
                    "question_extraction_failed": question_failures,
                    "question_avg_tokens": round(avg_tokens, 1),
                    "written_traces": written_traces,
                    "elapsed_seconds": round(elapsed, 2),
                    "eta_seconds": round(remaining, 2),
                },
            )

    total_elapsed = time.perf_counter() - run_start
    total_traces, final_extraction_failed_count = summarize_trace_file(output_path)
    extraction_fail_threshold = require_config_value(
        "analysis.max_extraction_fail_rate",
        config.analysis.max_extraction_fail_rate,
    )
    extraction_failed_rate = (
        final_extraction_failed_count / total_traces if total_traces else 0.0
    )
    threshold_exceeded = extraction_failed_rate > extraction_fail_threshold
    if threshold_exceeded:
        print(
            "[WARN] extraction_failed rate "
            f"{extraction_failed_rate:.2%} exceeds threshold "
            f"{extraction_fail_threshold:.2%}"
        )

    print(f"total_written_traces: {written_traces}")
    print(f"total_skipped_existing_traces: {skipped_traces}")
    print(f"extraction_failed_traces: {final_extraction_failed_count}")
    print(f"total_traces: {total_traces}")
    print(f"extraction_failed_rate: {extraction_failed_rate:.2%}")
    print(f"extraction_fail_threshold: {extraction_fail_threshold:.2%}")
    print(f"threshold_exceeded: {threshold_exceeded}")
    print(f"total_elapsed_seconds: {total_elapsed:.2f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage 1 trace generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment config YAML.")
    parser.add_argument("--subset-path", required=True, help="Path to eval_subset.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated run outputs.")
    parser.add_argument(
        "--shard-id",
        default=None,
        help="Optional shard label used to isolate outputs under output-dir/shards/.",
    )
    parser.add_argument("--start-idx", type=int, default=0, help="Inclusive question start index.")
    parser.add_argument("--end-idx", type=int, default=None, help="Exclusive question end index.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose tokenizer/input preparation debug logs.",
    )
    return parser.parse_args()


def discover_prompt_templates(
    prompts_dir: str,
    expected_count: int,
    preferred_prompt_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load all Stage C ICL prompt groups in filename order."""

    prompt_dir = Path(prompts_dir)
    prompt_paths = list(prompt_dir.glob("icl_*.yaml"))
    templates_by_id = {
        path.stem: load_prompt_template(path.stem, prompts_dir=prompts_dir)
        for path in prompt_paths
    }
    if preferred_prompt_ids:
        missing_prompt_ids = [
            prompt_id for prompt_id in preferred_prompt_ids if prompt_id not in templates_by_id
        ]
        if missing_prompt_ids:
            missing = ", ".join(missing_prompt_ids)
            raise ValueError(
                f"Preferred prompt ids are missing in '{prompt_dir}': {missing}."
            )
        extra_prompt_ids = sorted(set(templates_by_id) - set(preferred_prompt_ids))
        ordered_prompt_ids = [*preferred_prompt_ids, *extra_prompt_ids]
    else:
        ordered_prompt_ids = sorted(templates_by_id)
    templates = [templates_by_id[prompt_id] for prompt_id in ordered_prompt_ids]
    if len(templates) != expected_count:
        raise ValueError(
            f"Expected {expected_count} ICL prompt groups in '{prompt_dir}', found {len(templates)}."
        )
    return templates


def build_run_metadata(
    config: ExperimentConfig,
    prompt_ids: list[str],
    temperature: float | None,
    icl_group_temperatures: dict[str, float],
    icl_group_sample_counts: dict[str, int],
    max_new_tokens: int,
    num_icl_groups: int,
    samples_per_group: int,
) -> dict[str, Any]:
    """Construct run-level metadata for Stage 1 trace generation."""

    return {
        "run_id": config.experiment.run_id,
        "model_name": config.model.name,
        "dataset": f"{config.dataset.name}:{config.dataset.split}",
        "temperature": temperature,
        "icl_group_temperatures": icl_group_temperatures,
        "icl_group_sample_counts": icl_group_sample_counts,
        "max_new_tokens": max_new_tokens,
        "num_icl_groups": num_icl_groups,
        "samples_per_group": samples_per_group,
        "seed": config.experiment.seed,
        "prompt_ids": prompt_ids,
        "schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_default_shard_id(*, start_idx: int, end_idx: int | None, subset_size: int) -> str:
    """Build a stable shard label from the selected question range."""

    effective_end = subset_size if end_idx is None else end_idx
    return f"q{start_idx:04d}_{effective_end:04d}"


def write_shard_metadata(
    *,
    shard_dir: Path,
    shard_id: str,
    start_idx: int,
    end_idx: int | None,
    selected_questions: int,
) -> str:
    """Write shard-specific bookkeeping without touching the shared run root."""

    shard_meta_path = shard_dir / "shard_meta.json"
    payload = {
        "shard_id": shard_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "selected_questions": selected_questions,
    }
    shard_meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(shard_meta_path)


def load_subset(subset_path: str) -> list[dict]:
    """Load eval_subset JSONL records."""

    records: list[dict] = []
    with Path(subset_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def summarize_trace_file(trace_path: Path) -> tuple[int, int]:
    """Count final traces and extraction failures from the written trace file."""

    total_traces = 0
    extraction_failed_count = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            total_traces += 1
            if record.get("extraction_failed"):
                extraction_failed_count += 1
    return total_traces, extraction_failed_count


def require_config_value(field_path: str, value: Any) -> Any:
    """Require a config value to be present after Pilot backfill."""

    if value is None:
        raise ValueError(f"{field_path} is required before Stage 1 generation can run.")
    return value


def resolve_prompt_temperatures(
    *,
    prompt_ids: list[str],
    default_temperature: float | None,
    configured_group_temperatures: dict[str, float],
) -> dict[str, float]:
    """Resolve the effective temperature for each discovered prompt group."""

    resolved: dict[str, float] = {}
    missing_prompt_ids: list[str] = []
    for prompt_id in prompt_ids:
        if prompt_id in configured_group_temperatures:
            resolved[prompt_id] = configured_group_temperatures[prompt_id]
        elif default_temperature is not None:
            resolved[prompt_id] = default_temperature
        else:
            missing_prompt_ids.append(prompt_id)

    if missing_prompt_ids:
        missing = ", ".join(sorted(missing_prompt_ids))
        raise ValueError(
            "Missing temperatures for prompt groups with no default fallback: "
            f"{missing}"
        )

    return resolved


def resolve_prompt_sample_counts(
    *,
    prompt_ids: list[str],
    default_samples_per_group: int | None,
    configured_group_sample_counts: dict[str, int],
) -> dict[str, int]:
    """Resolve the effective sample count for each discovered prompt group."""

    resolved: dict[str, int] = {}
    missing_prompt_ids: list[str] = []
    for prompt_id in prompt_ids:
        if prompt_id in configured_group_sample_counts:
            resolved[prompt_id] = configured_group_sample_counts[prompt_id]
        elif default_samples_per_group is not None:
            resolved[prompt_id] = default_samples_per_group
        else:
            missing_prompt_ids.append(prompt_id)

    if missing_prompt_ids:
        missing = ", ".join(sorted(missing_prompt_ids))
        raise ValueError(
            "Missing sample counts for prompt groups with no default fallback: "
            f"{missing}"
        )

    return resolved


if __name__ == "__main__":
    main()
