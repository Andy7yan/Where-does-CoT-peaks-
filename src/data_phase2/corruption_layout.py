"""Helpers for canonical Stage 1 artifact layout."""

from __future__ import annotations

from pathlib import Path


CORRUPTION_ARTIFACT_FILENAMES = (
    "all_steps.jsonl",
    "corruption_summary.json",
)

LEGACY_CORRUPTION_DIRNAME = "corruptted_traces"


def resolve_corruption_artifact_path(run_path: Path, filename: str) -> Path:
    """Resolve one corruption artifact in flat-root or legacy nested layout."""

    root_path = run_path / filename
    if root_path.exists():
        return root_path
    return run_path / LEGACY_CORRUPTION_DIRNAME / filename


def corruption_artifact_exists(run_path: Path, filename: str) -> bool:
    """Return whether one corruption artifact exists in either supported layout."""

    return resolve_corruption_artifact_path(run_path, filename).exists()


def resolve_corruption_artifact_dir(run_path: Path) -> Path:
    """Return the directory currently holding the corruption artifacts."""

    root_matches = [run_path / filename for filename in CORRUPTION_ARTIFACT_FILENAMES]
    if any(path.exists() for path in root_matches):
        return run_path
    return run_path / LEGACY_CORRUPTION_DIRNAME
