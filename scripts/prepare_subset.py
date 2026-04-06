"""CLI for preparing a deterministic GSM8K evaluation subset."""

import argparse
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data.gsm8k import load_gsm8k_test, save_eval_subset, select_eval_subset


def main() -> None:
    """Run the subset-preparation workflow."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)

    questions = load_gsm8k_test(
        source=args.source,
        local_path=args.local_path,
        cache_dir=args.cache_dir,
    )
    subset = select_eval_subset(
        questions,
        n=config.dataset.subset_size,
        hash_seed=config.dataset.subset_hash_seed,
    )
    jsonl_path, meta_path = save_eval_subset(subset, args.output_dir)

    gold_answers = [record["gold_answer"] for record in subset]
    summary = {
        "total_questions": len(questions),
        "subset_size": len(subset),
        "gold_answer_min": min(gold_answers) if gold_answers else None,
        "gold_answer_max": max(gold_answers) if gold_answers else None,
        "gold_answer_mean": statistics.fmean(gold_answers) if gold_answers else None,
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
