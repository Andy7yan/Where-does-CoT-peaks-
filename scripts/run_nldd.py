"""Regenerate NLDD corruption artifacts from existing clean traces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase1.generation import _load_tokenizer_with_fallback, ensure_model_available
from src.analysis_phase.nldd import (
    CorruptionSelectionConfig,
    build_corruption_records,
    load_trace_sources,
    summarize_corruption_records,
    write_corruption_artifacts,
)
from src.data_phase1.pilot import build_token_counter
from src.common.settings import ExperimentConfig


def main() -> None:
    """Regenerate corruption records from an existing Stage 1 run directory."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    token_counter, token_counter_label = build_diagnostic_token_counter(
        config=config,
        approximate=args.approximate_tokens,
    )
    trace_sources = load_trace_sources(args.source_run_dir)
    selection = CorruptionSelectionConfig(
        sampled_min_steps=args.sampled_min_steps,
        sampled_max_steps=args.sampled_max_steps,
        seed=args.seed if args.seed is not None else config.experiment.seed,
        include_incorrect_traces=args.include_incorrect_traces,
    )
    effective_enable_tier3 = config.nldd.enable_tier3_semantic_flip or args.use_tier3
    records_by_mode = build_corruption_records(
        trace_sources,
        token_counter=token_counter,
        token_delta_max=config.nldd.corruption_token_delta_max,
        retry_limit=config.nldd.corruption_retry_limit,
        selection=selection,
        float_perturbation_range=tuple(config.nldd.float_perturbation_range),
        enable_tier3_semantic_flip=effective_enable_tier3,
        max_perplexity_ratio=None,
    )
    summary = summarize_corruption_records(records_by_mode)
    metadata = {
        "source_run_dir": args.source_run_dir,
        "token_counter": token_counter_label,
        "token_delta_max": config.nldd.corruption_token_delta_max,
        "retry_limit": config.nldd.corruption_retry_limit,
        "sampled_min_steps": selection.sampled_min_steps,
        "sampled_max_steps": selection.sampled_max_steps,
        "seed": selection.seed,
        "include_incorrect_traces": selection.include_incorrect_traces,
        "tier3_semantic_flip_enabled": effective_enable_tier3,
        "perplexity_filter_enabled": False,
    }
    artifacts = write_corruption_artifacts(
        args.output_dir,
        records_by_mode=records_by_mode,
        summary=summary,
        metadata=metadata,
    )

    for key, value in artifacts.items():
        print(f"{key}: {value}")
    for mode_name, mode_summary in summary.items():
        print(
            f"{mode_name}: records={mode_summary['records']} "
            f"failures={mode_summary['failures']} "
            f"failure_rate={mode_summary['failure_rate']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True, help="Existing run directory with traces.")
    parser.add_argument("--output-dir", required=True, help="Output directory for regenerated corruption artifacts.")
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Experiment config used to resolve tokenizer and corruption settings.",
    )
    parser.add_argument(
        "--sampled-min-steps",
        type=int,
        default=1,
        help="Minimum sampled corruption points per trace for sampled mode.",
    )
    parser.add_argument(
        "--sampled-max-steps",
        type=int,
        default=2,
        help="Maximum sampled corruption points per trace for sampled mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for deterministic sampling seed.",
    )
    parser.add_argument(
        "--include-incorrect-traces",
        action="store_true",
        help="Include incorrect traces as corruption sources. Disabled by default.",
    )
    parser.add_argument(
        "--approximate-tokens",
        action="store_true",
        help="Use whitespace token counting instead of the model tokenizer.",
    )
    parser.add_argument(
        "--use-tier3",
        action="store_true",
        help="Enable Tier 3 semantic-flip fallback. Disabled by default.",
    )
    return parser.parse_args()


def build_diagnostic_token_counter(
    *,
    config: ExperimentConfig,
    approximate: bool,
) -> tuple[Callable[[str], int], str]:
    """Build the token counter used for token-delta filtering."""

    if approximate:
        return build_token_counter(tokenizer=None, approximate=True), "whitespace_approximation"

    try:
        local_model_path, _ = ensure_model_available(
            model_name=config.model.name,
            cache_dir=config.model.hf_cache,
        )
        tokenizer = _load_tokenizer_with_fallback(
            tokenizer_path=local_model_path,
            cache_dir=config.model.hf_cache,
        )
    except Exception as exc:
        print(
            "warning: failed to load the model tokenizer; "
            f"falling back to whitespace approximation: {exc}"
        )
        return build_token_counter(tokenizer=None, approximate=True), "whitespace_approximation"

    return build_token_counter(tokenizer=tokenizer, approximate=False), "model_tokenizer"


if __name__ == "__main__":
    main()
