"""CLI for per-question trace generation over a frozen medium/hard manifest."""

from __future__ import annotations

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

from scripts.run_generation import (
    build_default_shard_id,
    discover_prompt_templates,
    require_config_value,
    summarize_trace_file,
    write_shard_metadata,
)
from src.common.settings import ExperimentConfig
from src.data_phase1.generation import (
    TRACE_SCHEMA_VERSION,
    LLMGenerator,
    append_traces_to_jsonl,
    generate_traces_for_question,
    validate_output_dir_schema,
    write_run_metadata,
)
from src.data_phase1.per_question_selection import (
    PER_QUESTION_MANIFEST_FILENAME,
    PER_QUESTION_PIPELINE_VARIANT,
    PER_QUESTION_SELECTION_META_FILENAME,
    PER_QUESTION_TARGET_SAMPLES_PER_PROMPT,
    PER_QUESTION_TARGET_TOTAL_TRACES,
    load_per_question_manifest,
    slice_per_question_manifest,
)


def main() -> None:
    """Run per-question generation over a sliced manifest shard."""

    args = parse_args()
    if args.debug:
        os.environ["PEAK_COT_DEBUG"] = "1"
    artifacts = run_per_question_generation(
        config_path=args.config,
        output_dir=args.output_dir,
        question_manifest_path=args.question_manifest,
        prompts_dir=args.prompts_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        shard_id=args.shard_id,
        preserve_root_selection_inputs=args.preserve_root_selection_inputs,
    )
    for key, value in artifacts.items():
        print(f"{key}: {value}")


def run_per_question_generation(
    *,
    config_path: str,
    output_dir: str,
    question_manifest_path: str,
    prompts_dir: str,
    start_idx: int = 0,
    end_idx: int | None = None,
    shard_id: str | None = None,
    preserve_root_selection_inputs: bool = False,
) -> dict[str, Any]:
    """Run one per-question shard and persist it in the canonical shard layout."""

    config = ExperimentConfig.from_yaml(config_path)
    manifest_rows = load_per_question_manifest(question_manifest_path)
    selected_questions = slice_per_question_manifest(
        manifest_rows,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    root_output_dir = Path(output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    if not preserve_root_selection_inputs:
        _persist_root_selection_inputs(
            output_dir=root_output_dir,
            manifest_rows=manifest_rows,
            question_manifest_path=Path(question_manifest_path),
        )

    effective_shard_id = shard_id or build_default_shard_id(
        start_idx=start_idx,
        end_idx=end_idx,
        total_questions=len(manifest_rows),
    )
    shard_dir = root_output_dir / "shards" / effective_shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    output_path = shard_dir / "traces.jsonl"
    validate_output_dir_schema(str(shard_dir), expected_schema_version=TRACE_SCHEMA_VERSION)

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
        prompts_dir=prompts_dir,
        expected_count=num_icl_groups,
        preferred_prompt_ids=config.generation.icl_group_prompt_ids or None,
    )
    prompt_ids = [template["prompt_id"] for template in prompt_templates]
    temperature = config.generation.temperature
    max_new_tokens = require_config_value(
        "generation.max_new_tokens",
        config.generation.max_new_tokens,
    )
    selection_metadata = _load_selection_metadata_if_present(root_output_dir)
    run_metadata = build_per_question_run_metadata(
        config=config,
        prompt_ids=prompt_ids,
        prompts_dir=prompts_dir,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        selection_metadata=selection_metadata,
    )
    write_run_metadata(str(shard_dir), run_metadata)
    write_shard_metadata(
        shard_dir=shard_dir,
        shard_id=effective_shard_id,
        start_idx=start_idx,
        end_idx=end_idx,
        selected_questions=len(selected_questions),
    )

    written_traces = 0
    run_start = time.perf_counter()

    print(
        "run_config:",
        {
            "pipeline_variant": PER_QUESTION_PIPELINE_VARIANT,
            "output_dir": str(root_output_dir),
            "question_manifest_path": str(Path(question_manifest_path)),
            "prompt_dir": prompts_dir,
            "shard_id": effective_shard_id,
            "selected_questions": len(selected_questions),
            "total_manifest_questions": len(manifest_rows),
            "prompt_ids": prompt_ids,
            "max_new_tokens": max_new_tokens,
            "default_temperature": temperature,
        },
    )

    total_questions = len(selected_questions)
    default_samples_per_group = config.generation.samples_per_group
    for index, record in enumerate(selected_questions, start=1):
        question_start = time.perf_counter()
        prompt_sample_count = int(record["target_samples_per_prompt"])
        prompt_sample_counts = {
            prompt_id: prompt_sample_count for prompt_id in prompt_ids
        }
        traces = generate_traces_for_question(
            generator=generator,
            question_id=str(record["question_id"]),
            question_text=str(record["question_text"]),
            gold_answer=record["gold_answer"],
            prompt_templates=prompt_templates,
            samples_per_group=default_samples_per_group,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_sample_counts=prompt_sample_counts,
        )

        append_traces_to_jsonl(traces, str(output_path))
        written_traces += len(traces)

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
                    "difficulty": record["source_difficulty_bucket"],
                    "question_seconds": round(question_elapsed, 2),
                    "question_target_total_traces": int(record["target_total_traces"]),
                    "question_target_samples_per_prompt": prompt_sample_count,
                    "question_correct": f"{correct_count}/{len(traces)}",
                    "question_extraction_failed": sum(
                        1 for trace in traces if trace["extraction_failed"]
                    ),
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

    return {
        "pipeline_variant": PER_QUESTION_PIPELINE_VARIANT,
        "run_dir": str(root_output_dir),
        "shard_dir": str(shard_dir),
        "question_manifest_path": str(root_output_dir / PER_QUESTION_MANIFEST_FILENAME),
        "selection_meta_path": str(root_output_dir / PER_QUESTION_SELECTION_META_FILENAME),
        "written_traces": written_traces,
        "total_traces": total_traces,
        "extraction_failed_traces": final_extraction_failed_count,
        "extraction_failed_rate": extraction_failed_rate,
        "extraction_fail_threshold": extraction_fail_threshold,
        "total_elapsed_seconds": total_elapsed,
    }


def build_per_question_run_metadata(
    *,
    config: ExperimentConfig,
    prompt_ids: list[str],
    prompts_dir: str,
    temperature: float | None,
    max_new_tokens: int,
    selection_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Construct shard-local run metadata for per-question generation."""

    return {
        "run_id": config.experiment.run_id,
        "model_name": config.model.name,
        "dataset": f"{config.dataset.name}:{config.dataset.split}",
        "temperature": temperature,
        "icl_group_sample_counts": dict(config.generation.icl_group_sample_counts),
        "max_new_tokens": max_new_tokens,
        "num_icl_groups": len(prompt_ids),
        "samples_per_group": config.generation.samples_per_group,
        "seed": config.experiment.seed,
        "prompt_ids": prompt_ids,
        "schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline_variant": PER_QUESTION_PIPELINE_VARIANT,
        "source_run_dir": selection_metadata.get("source_run_dir"),
        "source_run_name": selection_metadata.get("source_run_name"),
        "source_question_metadata_path": selection_metadata.get("source_question_metadata_path"),
        "prompt_dir": prompts_dir,
        "per_question_trace_policy": selection_metadata.get(
            "per_question_trace_policy",
            {
                difficulty: {
                    "target_total_traces": total,
                    "target_samples_per_prompt": PER_QUESTION_TARGET_SAMPLES_PER_PROMPT[
                        difficulty
                    ],
                }
                for difficulty, total in PER_QUESTION_TARGET_TOTAL_TRACES.items()
            },
        ),
        "selected_question_count": selection_metadata.get("selected_question_count"),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for per-question generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment config YAML.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated run outputs.")
    parser.add_argument(
        "--question-manifest",
        required=True,
        help="Per-question manifest JSONL generated from the source run.",
    )
    parser.add_argument(
        "--prompts-dir",
        required=True,
        help="Directory containing the per-question icl_*.yaml prompt templates.",
    )
    parser.add_argument(
        "--shard-id",
        default=None,
        help="Optional shard label used to isolate outputs under output-dir/shards/.",
    )
    parser.add_argument("--start-idx", type=int, default=0, help="Inclusive manifest start index.")
    parser.add_argument("--end-idx", type=int, default=None, help="Exclusive manifest end index.")
    parser.add_argument(
        "--source",
        choices=("huggingface", "local"),
        default="huggingface",
        help="Reserved for interface compatibility; selection is already frozen in the manifest.",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Reserved for interface compatibility; unused once the manifest is built.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Reserved for interface compatibility; unused once the manifest is built.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose tokenizer/input preparation debug logs.",
    )
    parser.add_argument(
        "--preserve-root-selection-inputs",
        action="store_true",
        help=(
            "Keep the existing run-root manifest and selection metadata untouched. "
            "Use this when rerunning only a repair subset against an existing run."
        ),
    )
    return parser.parse_args()


def _persist_root_selection_inputs(
    *,
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    question_manifest_path: Path,
) -> None:
    manifest_target_path = output_dir / PER_QUESTION_MANIFEST_FILENAME
    with manifest_target_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_selection_meta_path = question_manifest_path.with_name(
        PER_QUESTION_SELECTION_META_FILENAME
    )
    if source_selection_meta_path.exists():
        selection_meta = json.loads(source_selection_meta_path.read_text(encoding="utf-8"))
        (output_dir / PER_QUESTION_SELECTION_META_FILENAME).write_text(
            json.dumps(selection_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _load_selection_metadata_if_present(output_dir: Path) -> dict[str, Any]:
    selection_meta_path = output_dir / PER_QUESTION_SELECTION_META_FILENAME
    if selection_meta_path.exists():
        return json.loads(selection_meta_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "per_question_selection_v1",
        "pipeline_variant": PER_QUESTION_PIPELINE_VARIANT,
        "selected_question_count": None,
        "per_question_trace_policy": {
            difficulty: {
                "target_total_traces": total,
                "target_samples_per_prompt": PER_QUESTION_TARGET_SAMPLES_PER_PROMPT[
                    difficulty
                ],
            }
            for difficulty, total in PER_QUESTION_TARGET_TOTAL_TRACES.items()
        },
    }


if __name__ == "__main__":
    main()
