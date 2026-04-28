"""Tests for the per-question data-phase and pq_analysis pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import uuid

from src.analysis_phase1.per_question_analysis import build_kstar_by_bin, run_per_question_analysis
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
        assert (run_dir / "per_question" / "q_ok" / "bins" / "bin_3" / "samples" / "1" / "corrupt_k2_full.json").exists()

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
            prompt_logits_batch_fn=_fake_prompt_logits_batch,
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
        assert (output_dir / "nldd_per_trace.jsonl").exists()
        assert (output_dir / "tas_curve_per_trace.jsonl").exists()
        assert (output_dir / "trace_profiles.jsonl").exists()
        assert not (output_dir / "per_question_summary.csv").exists()

        with (output_dir / "t2b_lstar_difficulty.csv").open("r", encoding="utf-8", newline="") as handle:
            lstar_rows = list(csv.DictReader(handle))
        assert {row["question_id"] for row in lstar_rows} == {"q_ok"}
        assert "l_star_S" in lstar_rows[0]

        with (output_dir / "t1b_step_surface.csv").open("r", encoding="utf-8", newline="") as handle:
            t1b_rows = list(csv.DictReader(handle))
        assert "question_id" in t1b_rows[0]
        assert "scope" not in t1b_rows[0]
        assert "pipeline" not in t1b_rows[0]
        assert any(row["question_id"] == "q_ok" for row in t1b_rows)

        with (output_dir / "bin_status.csv").open("r", encoding="utf-8", newline="") as handle:
            bin_status_rows = list(csv.DictReader(handle))
        assert any(row["scope"] == "q_ok" and row["pipeline"] == "pq" for row in bin_status_rows)
        assert any(row["scope"] == "q_short" and row["bin_status"] == "strict_ok" for row in bin_status_rows)
        partial_row = next(row for row in bin_status_rows if row["scope"] == "q_partial")
        assert partial_row["bin_status"] == "relaxed_ok"
        assert partial_row["n_valid_k"] == "2"
        assert partial_row["failed_k_count"] == "1"

        with (output_dir / "failure_stats.csv").open("r", encoding="utf-8", newline="") as handle:
            failure_rows = {row["question_id"]: row for row in csv.DictReader(handle)}
        assert failure_rows["q_deg"]["degenerate"] == "True"
        assert failure_rows["q_short"]["l_curve_insufficient"] == "True"
        assert failure_rows["q_ok"]["k_star_insufficient"] == "False"

        with (output_dir / "trace_profiles.jsonl").open("r", encoding="utf-8") as handle:
            profile_rows = [json.loads(line) for line in handle if line.strip()]
        assert profile_rows
        sample_profile = next(row for row in profile_rows if row["question_id"] == "q_ok" and row["length"] == 3)
        assert sample_profile["pipeline"] == "pq"
        assert sample_profile["clean_trace"]["steps"]
        assert sample_profile["corruption_profiles"]
        assert sample_profile["tas_curve"]
        assert sample_profile["corruption_profiles"][0]["k"] >= 2

        with (output_dir / "t1c_kstar_ratio.csv").open("r", encoding="utf-8", newline="") as handle:
            kstar_rows = list(csv.DictReader(handle))
        q_ok_row = next(row for row in kstar_rows if row["question_id"] == "q_ok" and row["L"] == "3")
        assert q_ok_row["k_star"].isdigit()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_build_kstar_by_bin_uses_argmax_of_mean_curve() -> None:
    rows = [
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s1",
            "k": 2,
            "nldd_value": 50.0,
            "k_star_trace": 2,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s1",
            "k": 4,
            "nldd_value": 10.0,
            "k_star_trace": 2,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s1",
            "k": 5,
            "nldd_value": 0.0,
            "k_star_trace": 2,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s2",
            "k": 2,
            "nldd_value": 10.0,
            "k_star_trace": 5,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s2",
            "k": 4,
            "nldd_value": 40.0,
            "k_star_trace": 5,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s2",
            "k": 5,
            "nldd_value": 50.0,
            "k_star_trace": 5,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s3",
            "k": 2,
            "nldd_value": 10.0,
            "k_star_trace": 4,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s3",
            "k": 4,
            "nldd_value": 45.0,
            "k_star_trace": 4,
        },
        {
            "question_id": "q_ok",
            "length": 5,
            "sample_id": "s3",
            "k": 5,
            "nldd_value": 0.0,
            "k_star_trace": 4,
        },
    ]

    kstar_by_bin = build_kstar_by_bin(rows)

    assert kstar_by_bin == {
        ("q_ok", 5): {
            "k_star": 4,
            "n_clean": 3,
        }
    }


def test_build_kstar_by_bin_keeps_legacy_sparse_k_behavior() -> None:
    rows = [
        {"question_id": "q_ok", "length": 5, "k": 2, "nldd_value": 10.0},
        {"question_id": "q_ok", "length": 5, "k": 2, "nldd_value": 20.0},
        {"question_id": "q_ok", "length": 5, "k": 4, "nldd_value": 99.0},
    ]

    assert build_kstar_by_bin(rows) == {
        ("q_ok", 5): {
            "k_star": 4,
            "n_clean": 1,
        }
    }


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
        {
            "question_id": "q_partial",
            "question_text": "Question q_partial",
            "gold_answer": 1.0,
            "source_difficulty_bucket": "hard",
            "target_total_traces": 2,
            "target_samples_per_prompt": 2,
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
    traces.extend(_build_partial_corruption_traces("q_partial", correct_count=2, wrong_count=1))

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


def _build_partial_corruption_traces(
    question_id: str,
    *,
    correct_count: int,
    wrong_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    steps = [
        "We start with the given amount.",
        "2 + 1 = 3.",
        "This sentence has no arithmetic relation.",
        "3 + 1 = 4.",
    ]
    for index in range(correct_count + wrong_count):
        is_correct = index < correct_count
        rows.append(
            {
                "trace_id": f"{question_id}_partial_{index}",
                "question_id": question_id,
                "question_text": f"Question {question_id}",
                "gold_answer": 1.0,
                "prompt_id": "icl_short",
                "raw_completion": "\n".join(steps + ["#### 1"]),
                "steps": steps,
                "actual_num_steps": len(steps),
                "final_answer_line": "#### 1",
                "extracted_answer": 1.0 if is_correct else 0.0,
                "is_correct": is_correct,
                "extraction_failed": False,
                "token_count": 20,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )
    return rows


def _fake_prompt_logits(prompt: str) -> list[float]:
    checksum = sum(ord(char) for char in prompt)
    incorrect_max = 0.4 + ((checksum % 7) * 0.1)
    return [2.0, 1.8, incorrect_max, incorrect_max - 0.2, -0.5, -1.0]


def _fake_prompt_logits_batch(prompts: list[str]) -> list[list[float]]:
    return [_fake_prompt_logits(prompt) for prompt in prompts]


def _fake_trace_trajectory(question: str, steps: tuple[str, ...]) -> list[list[float]]:
    vectors: list[list[float]] = [[0.0, float(len(question)), 0.0]]
    running_chars = len(question)
    for index, step in enumerate(steps, start=1):
        running_chars += len(step)
        vectors.append([float(index), float(running_chars), float(len(step))])
    return vectors
