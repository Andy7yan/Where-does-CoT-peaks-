"""Validate prompt/config consistency before formal generation submission."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_generation import discover_prompt_templates
from src.common.settings import ExperimentConfig
from src.data_phase1.prompting import inspect_prompt_templates


def main() -> None:
    """Validate generation preflight assumptions and print a concise summary."""

    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    num_icl_groups = require_value(
        "generation.num_icl_groups",
        config.generation.num_icl_groups,
    )
    prompt_ids = config.generation.icl_group_prompt_ids
    try:
        templates = discover_prompt_templates(
            prompts_dir=args.prompts_dir,
            expected_count=num_icl_groups,
            preferred_prompt_ids=prompt_ids or None,
        )
    except Exception:
        prompt_dir, inventory = inspect_prompt_templates(args.prompts_dir)
        print(f"prompt_preflight_expected: {','.join(prompt_ids)}")
        if inventory:
            discovered_rows = [
                f"{row['filename']}=>{row['prompt_id']}" for row in inventory
            ]
            print(f"prompt_preflight_inventory[{prompt_dir}]: {','.join(discovered_rows)}")
        else:
            print(f"prompt_preflight_inventory[{prompt_dir}]: <empty>")
        raise
    discovered_prompt_ids = [template["prompt_id"] for template in templates]
    print(f"prompt_preflight_ok: {','.join(discovered_prompt_ids)}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the generation preflight check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the experiment config YAML.")
    parser.add_argument(
        "--prompts-dir",
        default="prompts",
        help="Directory containing the ICL prompt YAML files.",
    )
    return parser.parse_args()


def require_value(field_path: str, value: int | None) -> int:
    """Require a non-null integer config field."""

    if value is None:
        raise ValueError(f"{field_path} must be set before formal generation submission.")
    return value


if __name__ == "__main__":
    main()
