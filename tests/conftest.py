"""Shared pytest fixtures for peak-CoT."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_gsm8k() -> list[dict]:
    """Load the bundled sample records."""

    sample_path = Path(__file__).resolve().parents[1] / "data" / "gsm8k_sample.json"
    if sample_path.exists():
        return json.loads(sample_path.read_text(encoding="utf-8"))
    return [
        {
            "question": f"What is {index} + {index + 1}?",
            "answer": f"Add {index} and {index + 1}.\n#### {2 * index + 1}",
        }
        for index in range(50)
    ]


@pytest.fixture
def sample_question(sample_gsm8k: list[dict]) -> dict:
    """Return the first bundled sample question."""

    return sample_gsm8k[0]
