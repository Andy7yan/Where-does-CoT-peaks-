"""CLI for preparing a deterministic GSM8K evaluation subset."""

import argparse
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gsm8k import (
    load_gsm8k_test,
    save_eval_subset,
    save_gsm8k_corpus,
    select_eval_subset,
)
from src.settings import ExperimentConfig


def main() -> None:
    """Run the subset-preparation workflow."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)

    questions = load_gsm8k_test(
        source=args.source,
        local_path=args.local_path,
        cache_dir=args.cache_dir,
    )
    corpus_path = None
    if not args.skip_full_corpus_export:
        corpus_path = save_gsm8k_corpus(
            questions,
            output_dir=args.output_dir,
            filename=args.full_corpus_filename,
        )
    subset = select_eval_subset(
        questions,
        n=resolve_subset_size(args.subset_size, config.dataset.subset_size),
        hash_seed=config.dataset.subset_hash_seed,
        start_idx=args.start_idx,
    )
    jsonl_path, meta_path = save_eval_subset(
        subset,
        args.output_dir,
        jsonl_filename=args.eval_subset_filename,
    )

    gold_answers = [record["gold_answer"] for record in subset]
    summary = {
        "total_questions": len(questions),
        "subset_size": len(subset),
        "subset_start_idx": getattr(subset, "start_idx", 0),
        "subset_end_idx_exclusive": getattr(subset, "start_idx", 0) + len(subset),
        "gold_answer_min": min(gold_answers) if gold_answers else None,
        "gold_answer_max": max(gold_answers) if gold_answers else None,
        "gold_answer_mean": statistics.fmean(gold_answers) if gold_answers else None,
        "corpus_path": corpus_path,
        "jsonl_path": jsonl_path,
        "meta_path": meta_path,
    }

    for key, value in summary.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for subset preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--source",
        choices=("huggingface", "local"),
        default="huggingface",
        help="Where to load GSM8K from.",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Path to a local GSM8K-style JSON file when using --source local.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional datasets cache directory for Hugging Face loading.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where eval_subset.jsonl and eval_subset_meta.json will be written.",
    )
    parser.add_argument(
        "--subset-size",
        default=None,
        help="Override the config subset size with an integer or 'all'.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index within the deterministic ranked corpus before slicing.",
    )
    parser.add_argument(
        "--eval-subset-filename",
        default="eval_subset.jsonl",
        help="Filename for the ranked evaluation subset JSONL.",
    )
    parser.add_argument(
        "--full-corpus-filename",
        default="gsm8k_test.jsonl",
        help="Filename for exporting the full GSM8K test corpus as JSONL.",
    )
    parser.add_argument(
        "--skip-full-corpus-export",
        action="store_true",
        help="Skip writing the full GSM8K corpus JSONL and only write the eval subset files.",
    )
    return parser.parse_args()


def resolve_subset_size(cli_value: str | None, config_value: int | None) -> int | None:
    """Resolve the subset-size override, supporting 'all' for the full ranked corpus."""

    if cli_value is None:
        return config_value
    if cli_value.lower() == "all":
        return None
    return int(cli_value)


if __name__ == "__main__":
    main()
