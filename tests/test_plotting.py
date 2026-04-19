"""Tests for Stage 1 static plotting."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from src.analysis_phase1.analysis import run_analysis
from src.analysis_phase1.per_question_analysis import run_per_question_analysis
from src.analysis_phase2.plotting import run_stage1_plotting
from src.data_phase2.per_question_pipeline import aggregate_per_question_outputs


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


def test_run_stage1_plotting_renders_spec_v8_static_plots(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    workspace = Path("tests") / f"_tmp_plotting_{uuid.uuid4().hex}"
    overall_run_dir = workspace / "overall_run"
    pq_run_dir = workspace / "pq_run"
    output_dir = workspace / "plots"
    try:
        shutil.copytree("tests/sample_data", overall_run_dir)
        run_analysis(
            run_dir=str(overall_run_dir),
            prompt_logits_fn=_fake_prompt_logits,
            tokenizer=FakeTokenizer(),
            trace_trajectory_fn=_fake_trace_trajectory,
            ld_epsilon=1.0e-6,
            tas_plateau_threshold=None,
        )

        _build_per_question_run(pq_run_dir)
        aggregate_per_question_outputs(
            pq_run_dir,
            config_path="configs/stage1_per_question.yaml",
        )
        run_per_question_analysis(
            run_dir=str(pq_run_dir),
            prompt_logits_fn=_fake_prompt_logits,
            tokenizer=FakeTokenizer(),
            trace_trajectory_fn=_fake_trace_trajectory,
            ld_epsilon=1.0e-6,
            tas_plateau_threshold=None,
            min_kstar_bins=2,
        )

        artifacts = run_stage1_plotting(
            overall_run_dir=overall_run_dir,
            pq_run_dir=pq_run_dir,
            output_dir=output_dir,
            representative_questions=["q_ok"],
            max_heatmap_questions=1,
            normalized_bins=4,
        )

        assert (overall_run_dir / "analysis" / "t1a_overview.csv").exists()
        assert (overall_run_dir / "analysis" / "t1b_step_surface.csv").exists()
        assert (output_dir / "t1a" / "t1a_easy.png").exists()
        assert (output_dir / "t1a" / "t1a_medium.png").exists()
        assert (output_dir / "t1a" / "t1a_hard.png").exists()
        assert (output_dir / "t1b" / "t1b_q_ok.png").exists()
        assert (output_dir / "t1b_norm" / "t1b_norm_q_ok.png").exists()
        assert (output_dir / "t1c_kstar_ratio_vs_difficulty.png").exists()
        assert (output_dir / "t2b_lstar_vs_difficulty.png").exists()
        assert artifacts["representative_questions"] == ["q_ok"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _build_per_question_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        {
            "question_id": "q_ok",
            "question_text": "Question q_ok",
            "gold_answer": 1.0,
            "source_difficulty_bucket": "medium",
            "target_total_traces": 22,
            "target_samples_per_prompt": 22,
        },
        {
            "question_id": "q_short",
            "question_text": "Question q_short",
            "gold_answer": 1.0,
            "source_difficulty_bucket": "hard",
            "target_total_traces": 22,
            "target_samples_per_prompt": 22,
        },
        {
            "question_id": "q_deg",
            "question_text": "Question q_deg",
            "gold_answer": 1.0,
            "source_difficulty_bucket": "hard",
            "target_total_traces": 6,
            "target_samples_per_prompt": 6,
        },
    ]
    with (run_dir / "per_question_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    shard_dir = run_dir / "shards" / "q0000_0003"
    shard_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    traces.extend(_build_question_traces("q_ok", 3, correct_count=5, wrong_count=5))
    traces.extend(_build_question_traces("q_ok", 4, correct_count=6, wrong_count=0))
    traces.extend(_build_question_traces("q_ok", 5, correct_count=5, wrong_count=1))
    traces.extend(_build_question_traces("q_short", 3, correct_count=5, wrong_count=6))
    traces.extend(_build_question_traces("q_short", 4, correct_count=5, wrong_count=6))
    traces.extend(_build_question_traces("q_deg", 3, correct_count=0, wrong_count=6))

    with (shard_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (shard_dir / "run_meta.json").write_text(
        json.dumps({"schema_version": "stage1_trace_v2"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_question_traces(
    question_id: str,
    length: int,
    *,
    correct_count: int,
    wrong_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = correct_count + wrong_count
    for index in range(total):
        is_correct = index < correct_count
        steps = [
            f"Step {step}: {step + 2} + 1 = {step + 3}."
            for step in range(length)
        ]
        rows.append(
            {
                "trace_id": f"{question_id}_L{length}_{index}",
                "question_id": question_id,
                "question_text": f"Question {question_id}",
                "gold_answer": 1.0,
                "prompt_id": "icl_short",
                "raw_completion": "\n".join(steps + ["#### 1"]),
                "steps": steps,
                "actual_num_steps": length,
                "final_answer_line": "#### 1",
                "extracted_answer": 1.0 if is_correct else 0.0,
                "is_correct": is_correct,
                "extraction_failed": False,
                "token_count": 10 + length,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )
    return rows


def _fake_prompt_logits(prompt: str) -> list[float]:
    checksum = sum(ord(char) for char in prompt)
    incorrect_max = 0.4 + ((checksum % 7) * 0.1)
    return [2.0, 1.8, incorrect_max, incorrect_max - 0.2, -0.5, -1.0]


def _fake_trace_trajectory(question: str, steps: tuple[str, ...]) -> list[list[float]]:
    vectors: list[list[float]] = [[0.0, float(len(question)), 0.0]]
    running_chars = len(question)
    for index, step in enumerate(steps, start=1):
        running_chars += len(step)
        vectors.append([float(index), float(running_chars), float(len(step))])
    return vectors
