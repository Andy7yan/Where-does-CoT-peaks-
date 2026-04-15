"""Diagnose corruption failures from a Pilot run."""

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_phase1.generation import _load_tokenizer_with_fallback, ensure_model_available
from src.data_phase1.pilot import build_token_counter, diagnose_corruption_failures, load_jsonl_records
from src.common.settings import ExperimentConfig


def main() -> None:
    """Diagnose failed corruption attempts for a Pilot run directory."""

    args = parse_args()
    run_dir = Path(args.run_dir)
    traces_path = run_dir / "pilot_traces.jsonl"
    if not traces_path.exists():
        raise FileNotFoundError(f"Pilot traces not found: {traces_path}")

    config = ExperimentConfig.from_yaml(args.config)
    traces = load_jsonl_records(traces_path)
    token_counter, token_counter_label = build_diagnostic_token_counter(
        config=config,
        approximate=args.approximate_tokens,
    )
    failures = diagnose_corruption_failures(
        traces=traces,
        token_counter=token_counter,
        corruption_token_delta_max=config.nldd.corruption_token_delta_max,
    )

    output_path = run_dir / "corruption_failures.jsonl"
    write_jsonl(failures, output_path)
    print(f"wrote_failures: {output_path}")
    print(f"token_counter: {token_counter_label}")
    print_summary(traces=traces, failures=failures)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Pilot run directory.")
    parser.add_argument(
        "--config",
        default="configs/stage1.yaml",
        help="Experiment config used to resolve tokenizer settings.",
    )
    parser.add_argument(
        "--approximate-tokens",
        action="store_true",
        help="Use whitespace token counting instead of the model tokenizer.",
    )
    return parser.parse_args()


def build_diagnostic_token_counter(
    *,
    config: ExperimentConfig,
    approximate: bool,
) -> tuple[Callable[[str], int], str]:
    """Build the token counter used to enforce corruption token drift checks."""

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
            "warning: failed to load the model tokenizer for corruption diagnostics; "
            f"falling back to whitespace approximation: {exc}"
        )
        return build_token_counter(tokenizer=None, approximate=True), "whitespace_approximation"

    return build_token_counter(tokenizer=tokenizer, approximate=False), "model_tokenizer"


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write JSONL records to disk."""

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(
    *,
    traces: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    """Print grouped failure summaries."""

    correct_traces = [trace for trace in traces if trace.get("is_correct")]
    step_attempts = 0
    attempts_by_step_index: dict[int, int] = {}
    attempts_by_actual_num_steps: dict[int, int] = {}
    for trace in correct_traces:
        steps = trace.get("steps", [])
        if not isinstance(steps, list):
            continue
        actual_num_steps = int(trace.get("actual_num_steps", len(steps)))
        for step_index, step_text in enumerate(steps, start=1):
            if not isinstance(step_text, str):
                continue
            step_attempts += 1
            attempts_by_step_index[step_index] = attempts_by_step_index.get(step_index, 0) + 1
            attempts_by_actual_num_steps[actual_num_steps] = (
                attempts_by_actual_num_steps.get(actual_num_steps, 0) + 1
            )

    failures_by_reason: dict[str, int] = {}
    failures_by_step_index: dict[int, int] = {}
    failures_by_actual_num_steps: dict[int, int] = {}
    no_numeric_steps: list[str] = []
    for failure in failures:
        reason = str(failure["failure_reason"])
        step_index = int(failure["step_index"])
        actual_num_steps = int(failure["actual_num_steps"])
        failures_by_reason[reason] = failures_by_reason.get(reason, 0) + 1
        failures_by_step_index[step_index] = failures_by_step_index.get(step_index, 0) + 1
        failures_by_actual_num_steps[actual_num_steps] = (
            failures_by_actual_num_steps.get(actual_num_steps, 0) + 1
        )
        if reason == "no_numeric":
            no_numeric_steps.append(str(failure["step_text"]))

    print(f"correct_traces: {len(correct_traces)}")
    print(f"step_attempts: {step_attempts}")
    print(f"total_failures: {len(failures)}")
    print(f"overall_failure_rate: {_format_rate(len(failures), step_attempts)}")
    print("")
    print("failure_reason_counts:")
    for reason in ("no_numeric", "token_delta_exceeded", "other"):
        print(f"  {reason}: {failures_by_reason.get(reason, 0)}")
    print("")
    print("failure_rate_by_step_index:")
    for step_index in sorted(attempts_by_step_index):
        attempts = attempts_by_step_index[step_index]
        failures_for_step = failures_by_step_index.get(step_index, 0)
        print(
            "  "
            f"step_index={step_index} "
            f"rate={_format_rate(failures_for_step, attempts)} "
            f"failures={failures_for_step} attempts={attempts}"
        )
    print("")
    print("failure_rate_by_actual_num_steps:")
    for actual_num_steps in sorted(attempts_by_actual_num_steps):
        attempts = attempts_by_actual_num_steps[actual_num_steps]
        failures_for_length = failures_by_actual_num_steps.get(actual_num_steps, 0)
        print(
            "  "
            f"actual_num_steps={actual_num_steps} "
            f"rate={_format_rate(failures_for_length, attempts)} "
            f"failures={failures_for_length} attempts={attempts}"
        )
    print("")
    print("sample_no_numeric_steps:")
    sampled_steps = sample_no_numeric_steps(no_numeric_steps)
    if not sampled_steps:
        print("  <none>")
    for step_text in sampled_steps:
        print(f"  {step_text}")


def sample_no_numeric_steps(step_texts: list[str], sample_size: int = 10) -> list[str]:
    """Return a deterministic sample of no-numeric failure steps."""

    if not step_texts:
        return []
    generator = random.Random(42)
    if len(step_texts) <= sample_size:
        return step_texts
    return generator.sample(step_texts, k=sample_size)


def _format_rate(count: int, total: int) -> str:
    if total <= 0:
        return "0/0 (0.0000)"
    return f"{count}/{total} ({count / total:.4f})"


if __name__ == "__main__":
    main()
