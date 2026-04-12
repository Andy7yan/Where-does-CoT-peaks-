"""GSM8K loading and subset-selection utilities."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from src.reasoning import extract_answer


class EvalSubset(list[dict[str, Any]]):
    """List-like container that carries subset metadata for serialization."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        hash_seed: int,
        start_idx: int = 0,
        total_questions: int | None = None,
        dataset: str = "gsm8k",
        split: str = "test",
    ) -> None:
        super().__init__(records)
        self.hash_seed = hash_seed
        self.start_idx = start_idx
        self.total_questions = total_questions
        self.dataset = dataset
        self.split = split


def load_gsm8k_test(
    source: str = "huggingface",
    local_path: str | None = None,
    cache_dir: str | None = None,
) -> list[dict]:
    """Load the GSM8K test split from Hugging Face or a local JSON file."""

    if source == "local":
        if local_path is None:
            raise ValueError("local_path is required when source='local'.")
        return _load_local_records(local_path)

    if source == "huggingface":
        return _load_huggingface_records(cache_dir=cache_dir)

    raise ValueError(f"Unsupported GSM8K source: {source}")


def parse_gold_answer(raw_answer: str) -> float:
    """Extract the numeric gold answer from a GSM8K answer field."""

    result = extract_answer(raw_answer)
    if result.value is None:
        raise ValueError("Could not parse a gold answer from the provided GSM8K record.")
    return result.value


def select_eval_subset(
    questions: list[dict],
    n: int | None = 200,
    hash_seed: int = 42,
    start_idx: int = 0,
) -> list[dict]:
    """Select a deterministic evaluation subset using salted SHA-256 sorting."""

    ranked_records: list[tuple[str, dict[str, Any]]] = []
    for record in questions:
        question_text = _require_text_field(record, "question")
        rank = hashlib.sha256(f"{hash_seed}:{question_text}".encode("utf-8")).hexdigest()
        ranked_records.append((rank, record))

    ranked_records.sort(key=lambda item: item[0])
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative.")
    if start_idx > len(ranked_records):
        raise ValueError(
            f"start_idx {start_idx} exceeds the ranked corpus size {len(ranked_records)}."
        )
    end_idx = len(ranked_records) if n is None else min(start_idx + n, len(ranked_records))
    subset_records: list[dict[str, Any]] = []
    for global_index, (_, record) in enumerate(
        ranked_records[start_idx:end_idx],
        start=start_idx,
    ):
        subset_records.append(
            {
                "question_id": f"gsm8k_{global_index:04d}",
                "question_text": _require_text_field(record, "question"),
                "gold_answer": parse_gold_answer(_require_text_field(record, "answer")),
            }
        )

    return EvalSubset(
        subset_records,
        hash_seed=hash_seed,
        start_idx=start_idx,
        total_questions=len(ranked_records),
        dataset="gsm8k",
        split="test",
    )


def save_eval_subset(
    subset: list[dict],
    output_dir: str,
    *,
    jsonl_filename: str = "eval_subset.jsonl",
    meta_filename: str | None = None,
) -> tuple[str, str]:
    """Write the evaluation subset JSONL and metadata JSON files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path / jsonl_filename
    if meta_filename is None:
        meta_stem = Path(jsonl_filename).stem
        meta_filename = f"{meta_stem}_meta.json"
    meta_path = output_path / meta_filename

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in subset:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "n": len(subset),
        "hash_seed": getattr(subset, "hash_seed", None),
        "start_idx": getattr(subset, "start_idx", 0),
        "total_questions": getattr(subset, "total_questions", None),
        "dataset": getattr(subset, "dataset", "gsm8k"),
        "split": getattr(subset, "split", "test"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(jsonl_path), str(meta_path)


def save_gsm8k_corpus(
    records: list[dict],
    output_dir: str,
    filename: str = "gsm8k_test.jsonl",
) -> str:
    """Write the raw GSM8K question/answer corpus as JSONL."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    corpus_path = output_path / filename

    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in records:
            normalized = _normalize_record(record)
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    return str(corpus_path)


def _load_local_records(local_path: str) -> list[dict[str, str]]:
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local GSM8K file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("Local GSM8K JSON must contain a list of records.")

    return [_normalize_record(record) for record in data]


def _load_huggingface_records(cache_dir: str | None = None) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for source='huggingface'. "
            "Install project dependencies or use source='local' in offline environments."
        ) from exc

    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    return [_normalize_record(record) for record in dataset]


def _normalize_record(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question": _require_text_field(record, "question"),
        "answer": _require_text_field(record, "answer"),
    }


def _require_text_field(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"GSM8K record field '{field}' must be a string.")
    return value
