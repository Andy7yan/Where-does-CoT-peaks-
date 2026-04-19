"""Tests for the CPU postprocess pipeline over existing runs."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import yaml

from src.analysis_phase1.nldd import summarize_corruption_records
from src.data_phase2.postprocess import run_postprocess_pipeline


class FakeTokenizer:
    """Minimal tokenizer stub for deterministic analysis tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, list[int]]:
        del add_special_tokens, return_tensors
        return {"input_ids": self.encode(text, add_special_tokens=False)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1] if text.startswith(" ") else [0]


def test_run_postprocess_pipeline_rebuilds_and_analyzes_legacy_layout(monkeypatch) -> None:
    workspace = Path("tests") / f"_tmp_postprocess_{uuid.uuid4().hex}"
    run_dir = workspace / "legacy_like_run"
    config_path = workspace / "stage1_smoke.yaml"
    try:
        _build_valid_run(run_dir)
        _write_smoke_config(config_path)
        monkeypatch.setattr(
            "src.data_phase2.postprocess.load_analysis_backend",
            lambda config: {
                "tokenizer": FakeTokenizer(),
                "prompt_logits_fn": _fake_prompt_logits,
                "trace_trajectory_fn": _fake_trace_trajectory,
                "runtime_selection": SimpleNamespace(
                    requested_device="cpu",
                    resolved_device="cpu",
                    reason="test backend",
                    gpu_name=None,
                    gpu_compute_capability=None,
                ),
            },
        )

        result = run_postprocess_pipeline(
            run_dir=run_dir,
            config_path=str(config_path),
            include_analysis=True,
        )

        assert (run_dir / "difficulty" / "easy" / "questions.jsonl").exists()
        assert (run_dir / "difficulty_histogram.csv").exists()
        assert (run_dir / "analysis_phase1" / "nldd_per_trace.jsonl").exists()
        assert result["validation"]["difficulty_exports_match_traces"] is True
        assert result["analysis"]["sample_count"] >= 1
        assert result["runtime_selection"]["resolved_device"] == "cpu"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _build_valid_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    traces = [
        _trace("q1", "icl_short", 1, True, 2),
        _trace("q1", "icl_medium", 1, True, 3),
        _trace("q2", "icl_short", 1, False, 4),
        _trace("q2", "icl_medium", 1, True, 4),
    ]
    with (run_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "peak-cot-stage1-gsm8k-platinum-llama31",
                "max_new_tokens": 512,
                "icl_group_sample_counts": {
                    "icl_short": 5,
                    "icl_medium": 5,
                    "icl_detailed": 5,
                    "icl_verbose": 5,
                },
                "schema_version": "stage1_trace_v2",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_corruption_artifacts(run_dir)


def _write_corruption_artifacts(run_dir: Path) -> None:
    corruption_dir = run_dir / "corruptted_traces"
    corruption_dir.mkdir(parents=True, exist_ok=True)
    all_steps = [
        {
            "corruption_id": "q1_icl_medium_1_step2_all_steps",
            "trace_id": "q1_icl_medium_1",
            "step_index": 2,
            "corruption_failed": False,
            "corruption_tier": 1,
            "corruption_type": "numeric_result",
            "failure_tier": None,
            "clean_step": "Step 1: 3 + 1 = 4.",
            "corrupt_step": "Step 1: 3 + 1 = 5.",
            "token_delta": 0,
        },
        {
            "corruption_id": "q1_icl_medium_1_step3_all_steps",
            "trace_id": "q1_icl_medium_1",
            "step_index": 3,
            "corruption_failed": False,
            "corruption_tier": 1,
            "corruption_type": "numeric_result",
            "failure_tier": None,
            "clean_step": "Step 2: 4 + 1 = 5.",
            "corrupt_step": "Step 2: 4 + 1 = 6.",
            "token_delta": 0,
        },
        {
            "corruption_id": "q2_icl_medium_1_step2_all_steps",
            "trace_id": "q2_icl_medium_1",
            "step_index": 2,
            "corruption_failed": False,
            "corruption_tier": 2,
            "corruption_type": "operator_flip",
            "failure_tier": None,
            "clean_step": "Step 1: 3 + 1 = 4.",
            "corrupt_step": "Step 1: 3 - 1 = 2.",
            "token_delta": 0,
        },
        {
            "corruption_id": "q2_icl_medium_1_step3_all_steps",
            "trace_id": "q2_icl_medium_1",
            "step_index": 3,
            "corruption_failed": False,
            "corruption_tier": 2,
            "corruption_type": "operator_flip",
            "failure_tier": None,
            "clean_step": "Step 2: 4 + 1 = 5.",
            "corrupt_step": "Step 2: 4 - 1 = 3.",
            "token_delta": 0,
        },
        {
            "corruption_id": "q2_icl_medium_1_step4_all_steps",
            "trace_id": "q2_icl_medium_1",
            "step_index": 4,
            "corruption_failed": False,
            "corruption_tier": 2,
            "corruption_type": "operator_flip",
            "failure_tier": None,
            "clean_step": "Step 3: 5 + 1 = 6.",
            "corrupt_step": "Step 3: 5 - 1 = 4.",
            "token_delta": 0,
        },
    ]
    with (corruption_dir / "all_steps.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_steps:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_corruption_records({"all_steps": all_steps})
    (corruption_dir / "corruption_summary.json").write_text(
        json.dumps({"metadata": {"seed": 42}, "summary": summary}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_smoke_config(path: Path) -> None:
    config = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    config["analysis"]["min_bin_size"] = 1
    config["analysis"]["min_nldd_length"] = 3
    config["analysis"]["target_traces_per_cell"] = 10
    config["analysis"]["target_traces_near_lstar"] = 10
    config["analysis"]["min_near_lstar_traces"] = 1
    config["analysis"]["min_cell_size"] = 1
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _trace(
    question_id: str,
    prompt_id: str,
    sample_idx: int,
    is_correct: bool,
    actual_num_steps: int,
) -> dict[str, object]:
    steps = [f"Step {index}: {index + 2} + 1 = {index + 3}." for index in range(actual_num_steps)]
    return {
        "trace_id": f"{question_id}_{prompt_id}_{sample_idx}",
        "question_id": question_id,
        "question_text": f"Question {question_id}",
        "gold_answer": 1.0,
        "prompt_id": prompt_id,
        "raw_completion": "\n".join(steps),
        "steps": steps,
        "actual_num_steps": actual_num_steps,
        "final_answer_line": "#### 1",
        "extracted_answer": 1.0 if is_correct else 0.0,
        "is_correct": is_correct,
        "extraction_failed": False,
        "token_count": 10 + actual_num_steps,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def _fake_prompt_logits(prompt: str) -> list[float]:
    checksum = sum(ord(char) for char in prompt)
    incorrect_max = 0.4 + ((checksum % 7) * 0.1)
    return [2.0, 1.8, incorrect_max, incorrect_max - 0.2, -0.5, -1.0]


def _fake_trace_trajectory(question: str, steps: tuple[str, ...]) -> list[list[float]]:
    vectors: list[list[float]] = []
    running_chars = len(question)
    for index, step in enumerate(steps, start=1):
        running_chars += len(step)
        vectors.append([float(index), float(running_chars), float(len(step))])
    return vectors
