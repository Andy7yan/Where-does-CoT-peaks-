"""Flatten and clean a canonical full-generation result directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.canonical_artifacts import CORRUPTION_ARTIFACT_FILENAMES
from src.difficulty_groups import export_difficulty_length_groups


def main() -> None:
    args = parse_args()
    run_path = Path(args.run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    moved = flatten_corruption_artifacts(run_path)
    manifest = export_difficulty_length_groups(run_path)

    if args.delete_legacy and (run_path / "legacy").exists():
        shutil.rmtree(run_path / "legacy")
    if args.delete_smoke_dir:
        smoke_path = Path(args.delete_smoke_dir)
        if smoke_path.exists():
            shutil.rmtree(smoke_path)

    print(f"run_dir: {run_path}")
    print(f"flattened_corruption_files: {moved}")
    for difficulty, payload in manifest["difficulty_groups"].items():
        counts = payload["counts"]
        print(
            f"{difficulty}: all={counts['all']} "
            f"lt3={counts['lt3_excluded']} "
            f"short={counts['short']} medium={counts['medium']} "
            f"detailed={counts['detailed']} verbose={counts['verbose']}"
        )
    print(f"excluded_from_difficulty: {manifest['excluded_from_difficulty_count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Canonical full_generation directory to rewrite.")
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Delete run_dir/legacy after flattening and export generation.",
    )
    parser.add_argument(
        "--delete-smoke-dir",
        default=None,
        help="Optional temporary smoke directory to delete after merging.",
    )
    return parser.parse_args()


def flatten_corruption_artifacts(run_path: Path) -> list[str]:
    """Move canonical corruption files to the run root and remove the old folder."""

    legacy_dir = run_path / "corruptted_traces"
    moved: list[str] = []
    for filename in CORRUPTION_ARTIFACT_FILENAMES:
        nested_path = legacy_dir / filename
        root_path = run_path / filename
        if not nested_path.exists():
            continue
        if root_path.exists():
            if root_path.stat().st_size != nested_path.stat().st_size:
                raise ValueError(
                    f"Root artifact already exists with different contents: {root_path}"
                )
            try:
                nested_path.unlink()
                moved.append(f"deduped_duplicate:{filename}")
            except PermissionError:
                moved.append(f"duplicate_left_in_place:{filename}")
            continue
        try:
            shutil.move(str(nested_path), str(root_path))
            moved.append(filename)
        except PermissionError:
            shutil.copy2(str(nested_path), str(root_path))
            moved.append(f"copied:{filename}")

    if legacy_dir.exists() and not any(legacy_dir.iterdir()):
        legacy_dir.rmdir()
    return moved


if __name__ == "__main__":
    main()
