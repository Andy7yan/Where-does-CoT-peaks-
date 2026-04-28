"""Export a slim k*/L = 1 JSONL containing only trace-level thinking content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_thinking_row(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace") or {}
    steps = trace.get("steps") or []
    return {"thinking_steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        default="results/outputs-0426/kstar_ratio_eq1_retained_traces.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="results/outputs-0426/kstar_ratio_eq1_thinking_traces_only.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in iter_jsonl(input_path):
            slim = build_thinking_row(row)
            handle.write(json.dumps(slim, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1

    print(f"rows: {count}")
    print(f"output_jsonl: {output_path}")


if __name__ == "__main__":
    main()
