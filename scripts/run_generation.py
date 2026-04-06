"""CLI for Stage 1 length-controlled trace generation."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.generation.length_controlled import (
    LLMGenerator,
    append_traces_to_jsonl,
    generate_traces_for_question,
    load_existing_trace_ids,
)
from src.prompts import load_prompt_template


def main() -> None:
    """Run Stage 1 trace generation over an evaluation subset."""

    args = parse_args()
    if args.debug:
        os.environ["PEAK_COT_DEBUG"] = "1"
    config = ExperimentConfig.from_yaml(args.config)
    subset = load_subset(args.subset_path)
    selected_subset = subset[args.start_idx : args.end_idx]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "traces.jsonl"

    run_start = time.perf_counter()
    generator = LLMGenerator(
        model_name=config.model.name,
        dtype=config.model.dtype,
        cache_dir=config.model.hf_cache,
    )
    prompt_template = load_prompt_template(args.prompt_id)
    existing_trace_ids = load_existing_trace_ids(str(output_path))

    written_traces = 0
    skipped_traces = 0
    extraction_failed_count = 0

    length_grid = args.length_grid if args.length_grid is not None else config.generation.length_grid
    samples_per_length = (
        args.samples_per_length
        if args.samples_per_length is not None
        else config.generation.samples_per_length
    )
    temperature = args.temperature if args.temperature is not None else config.generation.temperature
    max_new_tokens = (
        args.max_new_tokens if args.max_new_tokens is not None else config.generation.max_new_tokens
    )

    print(f"selected_questions: {len(selected_subset)}")
    print(f"length_grid: {length_grid}")
    print(f"samples_per_length: {samples_per_length}")
    print(f"temperature: {temperature}")
    print(f"max_new_tokens: {max_new_tokens}")

    total_questions = len(selected_subset)
    for index, record in enumerate(selected_subset, start=1):
        question_start = time.perf_counter()
        print(
            f"starting_question: {index}/{total_questions} "
            f"question_id={record['question_id']}"
        )
        traces = generate_traces_for_question(
            generator=generator,
            question_id=record["question_id"],
            question_text=record["question_text"],
            gold_answer=float(record["gold_answer"]),
            length_grid=length_grid,
            samples_per_length=samples_per_length,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            prompt_template=prompt_template,
            model_name=config.model.name,
        )

        new_traces = [trace for trace in traces if trace["trace_id"] not in existing_trace_ids]
        if new_traces:
            append_traces_to_jsonl(new_traces, str(output_path))
            for trace in new_traces:
                existing_trace_ids.add(trace["trace_id"])

        written_traces += len(new_traces)
        skipped_traces += len(traces) - len(new_traces)
        extraction_failed_count += sum(1 for trace in new_traces if trace["extraction_failed"])

        if index % 10 == 0 or index == total_questions:
            elapsed = time.perf_counter() - run_start
            average_per_question = elapsed / index if index else 0.0
            remaining = max(total_questions - index, 0) * average_per_question
            question_elapsed = time.perf_counter() - question_start
            print(
                f"progress: {index}/{total_questions} questions | "
                f"last_question_seconds={question_elapsed:.2f} | "
                f"elapsed_seconds={elapsed:.2f} | eta_seconds={remaining:.2f}"
            )

    total_elapsed = time.perf_counter() - run_start
    print(f"total_written_traces: {written_traces}")
    print(f"total_skipped_existing_traces: {skipped_traces}")
    print(f"extraction_failed_traces: {extraction_failed_count}")
    print(f"total_elapsed_seconds: {total_elapsed:.2f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage 1 trace generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment config YAML.")
    parser.add_argument("--subset-path", required=True, help="Path to eval_subset.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated run outputs.")
    parser.add_argument("--start-idx", type=int, default=0, help="Inclusive question start index.")
    parser.add_argument("--end-idx", type=int, default=None, help="Exclusive question end index.")
    parser.add_argument(
        "--prompt-id",
        default="len_guided_v1",
        help="Prompt template id to load from prompts/.",
    )
    parser.add_argument(
        "--length-grid",
        type=int,
        nargs="+",
        default=None,
        help="Optional override for the configured target length grid.",
    )
    parser.add_argument(
        "--samples-per-length",
        type=int,
        default=None,
        help="Optional override for samples generated at each target length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional override for generation temperature.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for maximum generated tokens.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose tokenizer/input preparation debug logs.",
    )
    return parser.parse_args()


def load_subset(subset_path: str) -> list[dict]:
    """Load eval_subset JSONL records."""

    records: list[dict] = []
    with Path(subset_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


if __name__ == "__main__":
    main()
