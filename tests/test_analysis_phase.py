"""Tests for the analysis workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid

from src.analysis_phase1.analysis import (
    build_trace_trajectory_fn,
    compute_tas_curve_from_vectors,
    compute_tas_from_vectors,
)
from src.analysis_phase1.io import load_analysis_samples


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


def test_load_analysis_samples_reads_compact_sample_layout() -> None:
    samples = load_analysis_samples("tests/sample_data")

    assert samples
    assert samples[0].sample_id.isdigit()
    assert samples[0].clean_steps
    assert samples[0].k_values[0] >= 2
    assert samples[0].question_text


def test_run_analysis_phase_main_writes_analysis_outputs(monkeypatch) -> None:
    workspace = Path("tests") / f"_tmp_analysis_phase_{uuid.uuid4().hex}"
    run_dir = workspace / "sample_run"
    try:
        shutil.copytree("tests/sample_data", run_dir)

        import scripts.run_analysis_phase as runner

        monkeypatch.setattr(
            runner,
            "load_analysis_backend",
            lambda config: {
                "tokenizer": FakeTokenizer(),
                "prompt_logits_fn": _fake_prompt_logits,
                "trace_trajectory_fn": _fake_trace_trajectory,
                "runtime_selection": SimpleNamespace(
                    requested_device="cpu",
                    resolved_device="cpu",
                ),
            },
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_analysis_phase.py",
                "--run-dir",
                str(run_dir),
                "--config",
                "configs/stage1.yaml",
            ],
        )

        runner.main()

        analysis_dir = run_dir / "analysis_phase1"
        assert (analysis_dir / "accuracy_by_length.csv").exists()
        assert (analysis_dir / "S_calibration.json").exists()
        assert (analysis_dir / "nldd_per_trace.jsonl").exists()
        assert (analysis_dir / "tas_per_trace.jsonl").exists()
        assert (analysis_dir / "tas_curve_per_trace.jsonl").exists()
        assert (analysis_dir / "tas_curve.csv").exists()
        assert (analysis_dir / "nldd_surface.csv").exists()
        assert (analysis_dir / "k_star_by_L.csv").exists()
        assert (analysis_dir / "L_star.csv").exists()
        assert (analysis_dir / "bin_status.csv").exists()
        assert (analysis_dir / "failure_stats.csv").exists()
        assert "nldd_surface_path" in runner.run_analysis(
            run_dir=str(run_dir),
            prompt_logits_fn=_fake_prompt_logits,
            tokenizer=FakeTokenizer(),
            trace_trajectory_fn=_fake_trace_trajectory,
            ld_epsilon=1.0e-6,
            tas_plateau_threshold=None,
        )

        with (analysis_dir / "nldd_per_trace.jsonl").open("r", encoding="utf-8") as handle:
            nldd_rows = [json.loads(line) for line in handle if line.strip()]
        with (analysis_dir / "tas_per_trace.jsonl").open("r", encoding="utf-8") as handle:
            tas_rows = [json.loads(line) for line in handle if line.strip()]
        with (analysis_dir / "tas_curve_per_trace.jsonl").open("r", encoding="utf-8") as handle:
            tas_curve_rows = [json.loads(line) for line in handle if line.strip()]
        with (analysis_dir / "accuracy_by_length.csv").open("r", encoding="utf-8", newline="") as handle:
            accuracy_rows = list(csv.DictReader(handle))

        assert nldd_rows
        assert tas_rows
        assert tas_curve_rows
        assert accuracy_rows
        assert {row["difficulty"] for row in accuracy_rows} <= {"easy", "medium", "hard"}
        assert all("sample_id" in row for row in nldd_rows)
        assert all("tas_value" in row for row in tas_rows)
        assert all("step_index" in row for row in tas_curve_rows)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_build_trace_trajectory_fn_includes_h0_state() -> None:
    seen_prompts: list[str] = []

    def fake_hidden(prompt: str) -> list[float]:
        seen_prompts.append(prompt)
        return [float(len(seen_prompts))]

    trajectory_fn = build_trace_trajectory_fn(prompt_hidden_state_fn=fake_hidden)

    vectors = trajectory_fn("Question?", ("step one", "step two"))

    assert len(vectors) == 3
    assert seen_prompts[0] == "Question?\n\n\n\n#### "


def test_compute_tas_from_vectors_uses_l2_geometric_efficiency() -> None:
    tas_value, plateau_step = compute_tas_from_vectors(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        plateau_threshold=None,
    )

    assert round(tas_value, 6) == round((2.0 ** 0.5) / 2.0, 6)
    assert plateau_step is None


def test_compute_tas_from_vectors_tracks_plateau_on_l2_step_lengths() -> None:
    tas_value, plateau_step = compute_tas_from_vectors(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.01],
        ],
        plateau_threshold=0.02,
    )

    assert tas_value > 0.0
    assert plateau_step == 3


def test_compute_tas_curve_from_vectors_treats_each_prefix_as_endpoint() -> None:
    curve = compute_tas_curve_from_vectors(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        plateau_threshold=None,
    )

    assert [point["step_index"] for point in curve] == [1, 2]
    assert curve[0]["tas_value"] == 1.0
    assert round(curve[1]["tas_value"], 6) == round((2.0 ** 0.5) / 2.0, 6)


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
