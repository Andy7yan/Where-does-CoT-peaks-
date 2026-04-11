"""Pilot-run orchestration and reporting for Stage 1."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import statistics
from typing import Any, Callable

from src.generation import (
    TRACE_SCHEMA_VERSION,
    GenerationOutput,
    append_traces_to_jsonl,
    generate_traces_for_question,
    load_existing_trace_ids,
    write_run_metadata,
)
from src.gsm8k import load_gsm8k_test, select_eval_subset
from src.prompting import load_prompt_template
from src.reasoning import corrupt_arithmetic
from src.settings import load_settings


@dataclass(frozen=True)
class PilotOverrides:
    """Pilot-only generation parameters sourced from raw YAML."""

    num_questions: int
    num_icl_groups: int
    samples_per_group: int
    temperature: float
    max_new_tokens: int
    max_extraction_fail_rate: float


@dataclass(frozen=True)
class PilotCheckResult:
    """Structured result for a single Pilot check."""

    code: str
    title: str
    status: str
    summary: str
    metrics: dict[str, Any]


class PilotMockGenerator:
    """Deterministic local generator used for Stage D dry-runs."""

    def __init__(self, prompt_count: int, samples_per_group: int) -> None:
        self.prompt_count = prompt_count
        self.samples_per_group = samples_per_group
        self.calls = 0
        self.current_gold_answer = 0.0

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        del messages, temperature, max_new_tokens

        self.calls += 1
        group_index = ((self.calls - 1) // self.samples_per_group) % self.prompt_count
        num_steps = 1 + (group_index * 2)
        base_value = 12 + (group_index * 10)
        steps = [
            f"Step {index + 1}: Track {base_value + index} helper units before the final answer."
            for index in range(num_steps)
        ]
        answer = _format_numeric(self.current_gold_answer)
        raw_completion = "\n".join([*steps, f"#### {answer}"])
        return GenerationOutput(
            raw_completion=raw_completion,
            token_count=len(raw_completion.split()),
        )


def parse_pilot_overrides(settings: dict[str, Any]) -> PilotOverrides:
    """Extract Pilot-only generation overrides from the raw YAML mapping."""

    pilot = _require_mapping(settings, "pilot")
    return PilotOverrides(
        num_questions=_require_int(pilot, "num_questions"),
        num_icl_groups=_require_int(pilot, "num_icl_groups"),
        samples_per_group=_require_int(pilot, "samples_per_group"),
        temperature=_require_float(pilot, "temperature"),
        max_new_tokens=_require_int(pilot, "max_new_tokens"),
        max_extraction_fail_rate=_require_float(pilot, "max_extraction_fail_rate"),
    )


def discover_prompt_templates(
    prompts_dir: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Load Pilot prompt groups in filename order."""

    prompt_root = Path(prompts_dir)
    prompt_paths = sorted(prompt_root.glob("icl_*.yaml"))
    templates = [load_prompt_template(path.stem, prompts_dir=prompts_dir) for path in prompt_paths]
    if len(templates) != expected_count:
        raise ValueError(
            f"Expected {expected_count} ICL prompt groups in '{prompt_root}', found {len(templates)}."
        )
    return templates


def run_pilot(
    *,
    config_path: str,
    output_dir: str,
    source: str = "huggingface",
    cache_dir: str | None = None,
    local_path: str | None = None,
    prompts_dir: str = "prompts",
    mock: bool = False,
    data_path: str | None = None,
) -> dict[str, str]:
    """Execute the Stage D Pilot workflow and write its artifacts."""

    settings = load_settings(config_path)
    pilot = parse_pilot_overrides(settings)
    prompt_templates = discover_prompt_templates(
        prompts_dir=prompts_dir,
        expected_count=pilot.num_icl_groups,
    )

    resolved_local_path = data_path if mock and data_path is not None else local_path
    load_source = "local" if mock else source
    subset = build_pilot_subset(
        settings=settings,
        pilot=pilot,
        source=load_source,
        local_path=resolved_local_path,
        cache_dir=cache_dir,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    traces_path = output_root / "pilot_traces.jsonl"
    validate_pilot_output_dir(output_root)

    generator = (
        PilotMockGenerator(
            prompt_count=len(prompt_templates),
            samples_per_group=pilot.samples_per_group,
        )
        if mock
        else _build_real_generator(settings)
    )
    prompt_ids = [template["prompt_id"] for template in prompt_templates]
    run_metadata = build_pilot_run_metadata(settings, pilot, prompt_ids, mock=mock)
    write_run_metadata(str(output_root), run_metadata)

    existing_trace_ids = load_existing_trace_ids(str(traces_path))

    for record in subset:
        if isinstance(generator, PilotMockGenerator):
            generator.current_gold_answer = float(record["gold_answer"])
        traces = generate_traces_for_question(
            generator=generator,
            question_id=record["question_id"],
            question_text=record["question_text"],
            gold_answer=float(record["gold_answer"]),
            prompt_templates=prompt_templates,
            samples_per_group=pilot.samples_per_group,
            temperature=pilot.temperature,
            max_new_tokens=pilot.max_new_tokens,
        )
        new_traces = [trace for trace in traces if trace["trace_id"] not in existing_trace_ids]
        if new_traces:
            append_traces_to_jsonl(new_traces, str(traces_path))
            for trace in new_traces:
                existing_trace_ids.add(trace["trace_id"])

    all_traces = load_jsonl_records(traces_path)
    token_counter = build_token_counter(
        tokenizer=None if mock else generator.tokenizer,
        approximate=mock,
    )
    checks = evaluate_pilot_checks(
        traces=all_traces,
        prompt_ids=prompt_ids,
        pilot=pilot,
        token_counter=token_counter,
        corruption_token_delta_max=_require_int(
            _require_mapping(settings, "nldd"),
            "corruption_token_delta_max",
        ),
        token_counter_label="whitespace_approximation" if mock else "model_tokenizer",
    )
    report_text = render_pilot_report(
        checks=checks,
        settings=settings,
        pilot=pilot,
        prompt_ids=prompt_ids,
        traces=all_traces,
        mock=mock,
    )
    report_path = output_root / "pilot_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "pilot_traces_path": str(traces_path),
        "run_meta_path": str(output_root / "run_meta.json"),
        "pilot_report_path": str(report_path),
    }


def build_pilot_subset(
    *,
    settings: dict[str, Any],
    pilot: PilotOverrides,
    source: str,
    local_path: str | None,
    cache_dir: str | None,
) -> list[dict[str, Any]]:
    """Select the Pilot subset from GSM8K or a local GSM8K-style file."""

    dataset = _require_mapping(settings, "dataset")
    records = load_gsm8k_test(
        source=source,
        local_path=local_path,
        cache_dir=cache_dir,
    )
    return select_eval_subset(
        records,
        n=pilot.num_questions,
        hash_seed=_require_int(dataset, "subset_hash_seed"),
    )


def build_pilot_run_metadata(
    settings: dict[str, Any],
    pilot: PilotOverrides,
    prompt_ids: list[str],
    *,
    mock: bool,
) -> dict[str, Any]:
    """Construct run-level metadata for Pilot traces."""

    experiment = _require_mapping(settings, "experiment")
    dataset = _require_mapping(settings, "dataset")
    model = _require_mapping(settings, "model")
    return {
        "run_id": f"{_require_string(experiment, 'run_id')}-pilot",
        "model_name": _require_string(model, "name"),
        "dataset": f"{_require_string(dataset, 'name')}:{_require_string(dataset, 'split')}",
        "temperature": pilot.temperature,
        "max_new_tokens": pilot.max_new_tokens,
        "num_icl_groups": pilot.num_icl_groups,
        "samples_per_group": pilot.samples_per_group,
        "seed": _require_int(experiment, "seed"),
        "prompt_ids": prompt_ids,
        "schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": _utcnow_iso(),
        "mode": "mock" if mock else "real",
        "num_questions": pilot.num_questions,
    }


def evaluate_pilot_checks(
    *,
    traces: list[dict[str, Any]],
    prompt_ids: list[str],
    pilot: PilotOverrides,
    token_counter: Callable[[str], int],
    corruption_token_delta_max: int,
    token_counter_label: str,
) -> list[PilotCheckResult]:
    """Compute Pilot checks A/B/C/D/E from generated traces."""

    results = [
        evaluate_check_a(traces, prompt_ids),
        evaluate_check_b(traces),
        evaluate_check_c(traces, pilot.max_extraction_fail_rate),
        evaluate_check_d(
            traces=traces,
            token_counter=token_counter,
            corruption_token_delta_max=corruption_token_delta_max,
            token_counter_label=token_counter_label,
        ),
        evaluate_check_e(),
    ]
    return results


def evaluate_check_a(
    traces: list[dict[str, Any]],
    prompt_ids: list[str],
) -> PilotCheckResult:
    """Evaluate whether prompt groups induce distinct step-length distributions."""

    medians: list[float] = []
    missing_prompt_ids: list[str] = []
    for prompt_id in prompt_ids:
        lengths = [trace["actual_num_steps"] for trace in traces if trace["prompt_id"] == prompt_id]
        if not lengths:
            missing_prompt_ids.append(prompt_id)
            continue
        medians.append(float(statistics.median(lengths)))

    all_lengths = [trace["actual_num_steps"] for trace in traces]
    global_range = (max(all_lengths) - min(all_lengths)) if all_lengths else 0
    strictly_increasing = (
        len(medians) == len(prompt_ids)
        and all(medians[index] < medians[index + 1] for index in range(len(medians) - 1))
    )
    distinct_medians = len(set(medians))
    occupied_bins = len(set(all_lengths))

    if strictly_increasing and global_range >= 3:
        status = "PASS"
        summary = "Prompt-group median step counts increase cleanly with enough spread."
    elif distinct_medians >= 2 and (not strictly_increasing or global_range == 2):
        status = "WARN"
        summary = "Prompt groups show some separation, but the ordering or spread is weak."
    else:
        status = "FAIL"
        summary = "Prompt groups do not create enough distinct effective lengths."

    return PilotCheckResult(
        code="A",
        title="ICL Length Guidance",
        status=status,
        summary=summary,
        metrics={
            "prompt_group_medians": {
                prompt_id: (
                    float(
                        statistics.median(
                            [
                                trace["actual_num_steps"]
                                for trace in traces
                                if trace["prompt_id"] == prompt_id
                            ]
                        )
                    )
                    if any(trace["prompt_id"] == prompt_id for trace in traces)
                    else None
                )
                for prompt_id in prompt_ids
            },
            "global_step_range": global_range,
            "occupied_length_bins": occupied_bins,
            "missing_prompt_ids": missing_prompt_ids,
        },
    )


def evaluate_check_b(traces: list[dict[str, Any]]) -> PilotCheckResult:
    """Evaluate whether occupied length bins have enough total and correct traces."""

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_length[int(trace["actual_num_steps"])].append(trace)

    occupied_bins = sorted(by_length)
    total_correct = sum(1 for trace in traces if trace["is_correct"])
    bin_rows = {
        length: {
            "trace_count": len(length_traces),
            "correct_count": sum(1 for trace in length_traces if trace["is_correct"]),
        }
        for length, length_traces in by_length.items()
    }

    if (
        len(occupied_bins) >= 3
        and all(row["trace_count"] >= 2 for row in bin_rows.values())
        and all(row["correct_count"] >= 1 for row in bin_rows.values())
    ):
        status = "PASS"
        summary = "Occupied length bins have enough total and correct traces for Pilot inspection."
    elif len(occupied_bins) >= 2 and total_correct > 0:
        status = "WARN"
        summary = "At least two bins are occupied, but some bins are still sparse."
    else:
        status = "FAIL"
        summary = "Length coverage is too narrow or there are no correct traces."

    return PilotCheckResult(
        code="B",
        title="Per-Length Sample Volume",
        status=status,
        summary=summary,
        metrics={
            "occupied_bins": occupied_bins,
            "bin_counts": bin_rows,
            "total_correct_traces": total_correct,
            "median_occupied_bin_count": (
                float(statistics.median(row["trace_count"] for row in bin_rows.values()))
                if bin_rows
                else 0.0
            ),
        },
    )


def evaluate_check_c(
    traces: list[dict[str, Any]],
    max_extraction_fail_rate: float,
) -> PilotCheckResult:
    """Evaluate extraction and segmentation quality on Pilot traces."""

    total = len(traces)
    extraction_failed = sum(1 for trace in traces if trace["extraction_failed"])
    zero_step = sum(1 for trace in traces if trace["actual_num_steps"] == 0)
    extraction_failed_rate = (extraction_failed / total) if total else 1.0
    zero_step_rate = (zero_step / total) if total else 1.0
    warn_threshold = max(2 * max_extraction_fail_rate, 0.10)

    if (
        extraction_failed_rate <= max_extraction_fail_rate
        and zero_step_rate <= 0.02
    ):
        status = "PASS"
        summary = "Extraction and segmentation stay within Pilot thresholds."
    elif extraction_failed_rate <= warn_threshold and zero_step_rate <= 0.10:
        status = "WARN"
        summary = "Extraction or zero-step rates are elevated but still recoverable."
    else:
        status = "FAIL"
        summary = "Extraction or segmentation quality is too weak for a formal run."

    return PilotCheckResult(
        code="C",
        title="Segmentation And Extraction",
        status=status,
        summary=summary,
        metrics={
            "total_traces": total,
            "extraction_failed_rate": round(extraction_failed_rate, 4),
            "zero_step_rate": round(zero_step_rate, 4),
            "pass_threshold": max_extraction_fail_rate,
            "warn_threshold": round(warn_threshold, 4),
        },
    )


def evaluate_check_d(
    *,
    traces: list[dict[str, Any]],
    token_counter: Callable[[str], int],
    corruption_token_delta_max: int,
    token_counter_label: str,
) -> PilotCheckResult:
    """Evaluate single-step corruption feasibility over correct traces."""

    correct_traces = [trace for trace in traces if trace["is_correct"]]
    step_attempts = 0
    corruption_failed = 0
    token_delta_violations = 0

    for trace in correct_traces:
        for step in trace["steps"]:
            step_attempts += 1
            result = corrupt_arithmetic(step)
            if result.corruption_failed:
                corruption_failed += 1
                continue

            clean_tokens = token_counter(step)
            corrupt_tokens = token_counter(result.corrupt_text)
            if abs(clean_tokens - corrupt_tokens) > corruption_token_delta_max:
                token_delta_violations += 1

    if step_attempts == 0:
        status = "FAIL"
        summary = "No correct-trace steps were available for corruption checks."
        corruption_failed_rate = 1.0
        token_delta_violation_rate = 1.0
    else:
        corruption_failed_rate = corruption_failed / step_attempts
        token_delta_violation_rate = token_delta_violations / step_attempts
        if (
            corruption_failed_rate <= 0.15
            and token_delta_violation_rate <= 0.10
        ):
            status = "PASS"
            summary = "Arithmetic corruption is feasible on most correct-trace steps."
        elif (
            corruption_failed_rate <= 0.30
            and token_delta_violation_rate <= 0.25
        ):
            status = "WARN"
            summary = "Corruption is possible, but failure or token drift is still noticeable."
        else:
            status = "FAIL"
            summary = "Corruption quality is too unstable for a formal NLDD stage."

    return PilotCheckResult(
        code="D",
        title="Corruption Feasibility",
        status=status,
        summary=summary,
        metrics={
            "correct_traces": len(correct_traces),
            "step_attempts": step_attempts,
            "corruption_failed_rate": round(corruption_failed_rate, 4),
            "token_delta_violation_rate": round(token_delta_violation_rate, 4),
            "token_delta_max": corruption_token_delta_max,
            "token_counter": token_counter_label,
        },
    )


def evaluate_check_e() -> PilotCheckResult:
    """Report the deferred NLDD smoke status for Stage D."""

    return PilotCheckResult(
        code="E",
        title="NLDD Smoke",
        status="WARN",
        summary="Deferred to Stage F by stage-boundary decision.",
        metrics={
            "tas_plateau_threshold_default": 0.05,
            "analysis_num_spot_checks_default": 3,
        },
    )


def render_pilot_report(
    *,
    checks: list[PilotCheckResult],
    settings: dict[str, Any],
    pilot: PilotOverrides,
    prompt_ids: list[str],
    traces: list[dict[str, Any]],
    mock: bool,
) -> str:
    """Render the Pilot report markdown."""

    check_map = {check.code: check for check in checks}
    recommendation_rows = build_recommendation_rows(
        checks=check_map,
        pilot=pilot,
        traces=traces,
    )
    dataset = _require_mapping(settings, "dataset")
    lines = [
        "# Pilot Report",
        "",
        f"- Mode: {'mock' if mock else 'real'}",
        f"- Dataset: {_require_string(dataset, 'name')}:{_require_string(dataset, 'split')}",
        f"- Pilot questions: {pilot.num_questions}",
        f"- Prompt groups: {', '.join(prompt_ids)}",
        f"- Traces written: {len(traces)}",
        "",
        "## Check Results",
        "",
    ]

    for check in checks:
        lines.append(f"### {check.code}. {check.title}")
        lines.append(f"[{check.status}] {check.summary}")
        for key, value in check.metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(
        [
            "## Backfill Recommendations",
            "",
            "| Config Field | Recommended Value | Basis |",
            "|---|---:|---|",
        ]
    )
    for row in recommendation_rows:
        lines.append(
            f"| `{row['field']}` | `{row['value']}` | {row['basis']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Check E is intentionally deferred to Stage F and does not exercise `src/nldd.py` in Stage D.",
            "- `tas.plateau_threshold` and `analysis.num_spot_checks` are provisional defaults until Stage F smoke data exists.",
            "- The backfill table is intended to be copied into `configs/stage1.yaml` by hand after Pilot review.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_recommendation_rows(
    *,
    checks: dict[str, PilotCheckResult],
    pilot: PilotOverrides,
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the 9-field backfill recommendation table."""

    check_b_failed = checks["B"].status == "FAIL"
    observed_max_token_count = max((trace["token_count"] for trace in traces), default=0)
    median_occupied_bin_count = checks["B"].metrics["median_occupied_bin_count"]

    return [
        {
            "field": "dataset.subset_size",
            "value": 400 if check_b_failed else 200,
            "basis": "Use 400 when check B fails; otherwise use 200.",
        },
        {
            "field": "generation.num_icl_groups",
            "value": pilot.num_icl_groups,
            "basis": "Copy the Pilot prompt-group count.",
        },
        {
            "field": "generation.samples_per_group",
            "value": pilot.samples_per_group * 2 if check_b_failed else pilot.samples_per_group,
            "basis": "Double only when check B fails; otherwise copy the Pilot value.",
        },
        {
            "field": "generation.temperature",
            "value": pilot.temperature,
            "basis": "Copy the Pilot temperature.",
        },
        {
            "field": "generation.max_new_tokens",
            "value": max(pilot.max_new_tokens, observed_max_token_count + 32),
            "basis": "Use the larger of Pilot max_new_tokens and observed max token_count + 32.",
        },
        {
            "field": "analysis.min_bin_size",
            "value": 5 if median_occupied_bin_count >= 5 else 3,
            "basis": "Use 5 when the median occupied-bin count is at least 5; else 3.",
        },
        {
            "field": "analysis.max_extraction_fail_rate",
            "value": pilot.max_extraction_fail_rate,
            "basis": "Copy the Pilot extraction-fail threshold.",
        },
        {
            "field": "tas.plateau_threshold",
            "value": 0.05,
            "basis": "Provisional Stage D default because check E is deferred.",
        },
        {
            "field": "analysis.num_spot_checks",
            "value": 3,
            "basis": "Provisional Stage D default because check E is deferred.",
        },
    ]


def validate_pilot_output_dir(output_dir: Path) -> None:
    """Guard against mixing Pilot outputs with incompatible prior runs."""

    pilot_traces_path = output_dir / "pilot_traces.jsonl"
    run_meta_path = output_dir / "run_meta.json"
    formal_traces_path = output_dir / "traces.jsonl"

    if formal_traces_path.exists():
        raise RuntimeError(
            "Output directory already contains formal traces.jsonl. Use a fresh Pilot output directory."
        )

    if pilot_traces_path.exists() and not run_meta_path.exists():
        raise RuntimeError(
            "Output directory contains pilot_traces.jsonl but is missing run_meta.json."
        )

    if run_meta_path.exists():
        with run_meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise RuntimeError(
                "Output directory contains an incompatible run_meta.json schema_version."
            )


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk."""

    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def build_token_counter(
    *,
    tokenizer: Any | None,
    approximate: bool,
) -> Callable[[str], int]:
    """Return a token counting function for corruption checks."""

    if approximate or tokenizer is None:
        return lambda text: len(text.split())

    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _build_real_generator(settings: dict[str, Any]) -> Any:
    from src.generation import LLMGenerator

    model = _require_mapping(settings, "model")
    return LLMGenerator(
        model_name=_require_string(model, "name"),
        dtype=_require_string(model, "dtype"),
        cache_dir=_require_string(model, "hf_cache"),
    )


def _format_numeric(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Pilot config section '{key}' must be a mapping.")
    return value


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Pilot config field '{key}' must be a string.")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Pilot config field '{key}' must be an integer.")
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        raise TypeError(f"Pilot config field '{key}' must be a float.")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Pilot config field '{key}' must be a float.")


__all__ = [
    "PilotCheckResult",
    "PilotMockGenerator",
    "PilotOverrides",
    "build_pilot_run_metadata",
    "build_pilot_subset",
    "discover_prompt_templates",
    "parse_pilot_overrides",
    "render_pilot_report",
    "run_pilot",
]
