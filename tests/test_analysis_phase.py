"""Tests for the analysis workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid

from src.analysis_phase.io import load_analysis_samples


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

        analysis_dir = run_dir / "analysis"
        assert (analysis_dir / "accuracy_by_length.csv").exists()
        assert (analysis_dir / "S_calibration.json").exists()
        assert (analysis_dir / "nldd_per_trace.jsonl").exists()
        assert (analysis_dir / "tas_per_trace.jsonl").exists()
        assert (analysis_dir / "nldd_surface.csv").exists()
        assert (analysis_dir / "k_star_by_L.csv").exists()
        assert (analysis_dir / "L_star.csv").exists()
        assert (analysis_dir / "bin_status.csv").exists()
        assert (analysis_dir / "failure_stats.csv").exists()

        with (analysis_dir / "nldd_per_trace.jsonl").open("r", encoding="utf-8") as handle:
            nldd_rows = [json.loads(line) for line in handle if line.strip()]
        with (analysis_dir / "tas_per_trace.jsonl").open("r", encoding="utf-8") as handle:
            tas_rows = [json.loads(line) for line in handle if line.strip()]
        with (analysis_dir / "accuracy_by_length.csv").open("r", encoding="utf-8", newline="") as handle:
            accuracy_rows = list(csv.DictReader(handle))

        assert nldd_rows
        assert tas_rows
        assert accuracy_rows
        assert {row["difficulty"] for row in accuracy_rows} <= {"easy", "medium", "hard"}
        assert all("sample_id" in row for row in nldd_rows)
        assert all("tas_value" in row for row in tas_rows)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


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
