"""Tests for the per-question data-phase and pq_analysis pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import uuid

from src.analysis_phase1.per_question_analysis import run_per_question_analysis
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


def test_per_question_data_phase_and_analysis_write_revised_outputs(monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", "/tmp")

    workspace = Path("tests") / f"_tmp_per_question_pipeline_{uuid.uuid4().hex}"
    run_dir = workspace / "pq_run"
    try:
        _build_per_question_run(run_dir)

        artifacts = aggregate_per_question_outputs(
            run_dir,
            config_path="configs/stage1_per_question.yaml",
        )

        assert Path(artifacts["question_metadata_path"]).exists()
        assert (run_dir / "per_question" / "q_ok" / "l_curve.csv").exists()
        assert (run_dir / "per_question" / "q_ok" / "l_star.json").exists()
        assert (run_dir / "per_question" / "q_ok" / "bins" / "bin_3" / "selection.jsonl").exists()

        lstar_payload = json.loads(
            (run_dir / "per_question" / "q_ok" / "l_star.json").read_text(encoding="utf-8")
        )
        assert "l_star_S" in lstar_payload
        assert "l_star_C" not in lstar_payload
        assert lstar_payload["degenerate"] is False

        degenerate_payload = json.loads(
            (run_dir / "per_question" / "q_deg" / "l_star.json").read_text(encoding="utf-8")
        )
        assert degenerate_payload["degenerate"] is True

        analysis = run_per_question_analysis(
            run_dir=str(run_dir),
            prompt_logits_fn=_fake_prompt_logits,
            tokenizer=FakeTokenizer(),
            trace_trajectory_fn=_fake_trace_trajectory,
            ld_epsilon=1.0e-6,
            tas_plateau_threshold=None,
            min_kstar_bins=2,
        )

        output_dir = run_dir / "pq_analysis"
        assert Path(analysis["analysis_dir"]) == output_dir
        assert (output_dir / "t1b_step_surface.csv").exists()
        assert (output_dir / "t1c_kstar_ratio.csv").exists()
        assert (output_dir / "t2b_lstar_difficulty.csv").exists()
        assert (output_dir / "bin_status.csv").exists()
        assert (output_dir / "failure_stats.csv").exists()
        assert (output_dir / "S_calibration.json").exists()
        assert not (output_dir / "per_question_summary.csv").exists()

        with (output_dir / "t2b_lstar_difficulty.csv").open("r", encoding="utf-8", newline="") as handle:
            lstar_rows = list(csv.DictReader(handle))
        assert {row["question_id"] for row in lstar_rows} == {"q_ok"}
        assert "l_star_S" in lstar_rows[0]

        with (output_dir / "bin_status.csv").open("r", encoding="utf-8", newline="") as handle:
            bin_status_rows = list(csv.DictReader(handle))
        assert any(row["scope"] == "q_ok" and row["pipeline"] == "pq" for row in bin_status_rows)
        assert any(row["scope"] == "q_short" and row["bin_status"] == "ok" for row in bin_status_rows)

        with (output_dir / "failure_stats.csv").open("r", encoding="utf-8", newline="") as handle:
            failure_rows = {row["question_id"]: row for row in csv.DictReader(handle)}
        assert failure_rows["q_deg"]["degenerate"] == "True"
        assert failure_rows["q_short"]["l_curve_insufficient"] == "True"
        assert failure_rows["q_ok"]["k_star_insufficient"] == "False"
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
