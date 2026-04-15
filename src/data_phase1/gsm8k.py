"""Dataset loading and ranked question-record utilities for GSM8K-family benchmarks."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common.reasoning import extract_answer


DEFAULT_DATASET_NAME = "madrylab/gsm8k-platinum"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_QUESTION_ID_PREFIX = "gsm8k_platinum"


class QuestionSlice(list[dict[str, Any]]):
    """List-like container that carries ranked-question metadata for serialization."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        hash_seed: int,
        start_idx: int = 0,
        total_questions: int | None = None,
        dataset: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_DATASET_SPLIT,
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
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_DATASET_SPLIT,
) -> list[dict]:
    """Load GSM8K-Platinum or a compatible local JSON/JSONL file."""

    if source == "local":
        if local_path is None:
            raise ValueError("local_path is required when source='local'.")
        return _load_local_records(local_path)

    if source == "huggingface":
        return _load_huggingface_records(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            cache_dir=cache_dir,
        )

    raise ValueError(f"Unsupported dataset source: {source}")


def parse_gold_answer(raw_answer: str) -> float:
    """Extract the numeric gold answer from a GSM8K-style answer field."""

    result = extract_answer(raw_answer)
    if result.value is None:
        raise ValueError("Could not parse a gold answer from the provided dataset record.")
    return result.value


def build_ranked_questions(
    questions: list[dict],
    *,
    hash_seed: int = 42,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_DATASET_SPLIT,
    question_id_prefix: str | None = None,
) -> QuestionSlice:
    """Build the full ranked question table used by formal generation and pilot slicing."""

    effective_question_id_prefix = (
        DEFAULT_QUESTION_ID_PREFIX
        if question_id_prefix is None and dataset_name == DEFAULT_DATASET_NAME
        else _dataset_name_to_prefix(dataset_name)
        if question_id_prefix is None
        else question_id_prefix
    )

    ranked_records: list[tuple[str, dict[str, Any]]] = []
    for record in questions:
        question_text = _require_text_field(record, "question")
        rank = hashlib.sha256(f"{hash_seed}:{question_text}".encode("utf-8")).hexdigest()
        ranked_records.append((rank, record))

    ranked_records.sort(key=lambda item: item[0])

    output_records: list[dict[str, Any]] = []
    for global_index, (_, record) in enumerate(ranked_records):
        output_records.append(
            {
                "question_id": f"{effective_question_id_prefix}_{global_index:04d}",
                "question_text": _require_text_field(record, "question"),
                "gold_answer": parse_gold_answer(_require_text_field(record, "answer")),
            }
        )

    return QuestionSlice(
        output_records,
        hash_seed=hash_seed,
        start_idx=0,
        total_questions=len(ranked_records),
        dataset=dataset_name,
        split=split,
    )


def slice_question_records(
    question_records: list[dict[str, Any]],
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> QuestionSlice:
    """Take a deterministic slice from the ranked full question table."""

    total_questions = len(question_records)
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative.")
    if start_idx > total_questions:
        raise ValueError(
            f"start_idx {start_idx} exceeds the ranked corpus size {total_questions}."
        )
    effective_end_idx = total_questions if end_idx is None else end_idx
    if effective_end_idx < start_idx:
        raise ValueError("end_idx must be greater than or equal to start_idx.")
    if effective_end_idx > total_questions:
        raise ValueError(
            f"end_idx {effective_end_idx} exceeds the ranked corpus size {total_questions}."
        )

    source_slice = question_records[start_idx:effective_end_idx]
    return QuestionSlice(
        list(source_slice),
        hash_seed=getattr(question_records, "hash_seed", 42),
        start_idx=start_idx,
        total_questions=getattr(question_records, "total_questions", total_questions),
        dataset=getattr(question_records, "dataset", DEFAULT_DATASET_NAME),
        split=getattr(question_records, "split", DEFAULT_DATASET_SPLIT),
    )


def save_question_slice(
    question_records: list[dict[str, Any]],
    output_dir: str,
    *,
    jsonl_filename: str = "gsm8k_ranked_questions.jsonl",
    meta_filename: str | None = None,
) -> tuple[str, str]:
    """Write ranked question records and metadata JSON files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path / jsonl_filename
    if meta_filename is None:
        meta_stem = Path(jsonl_filename).stem
        meta_filename = f"{meta_stem}_meta.json"
    meta_path = output_path / meta_filename

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in question_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "n": len(question_records),
        "hash_seed": getattr(question_records, "hash_seed", None),
        "start_idx": getattr(question_records, "start_idx", 0),
        "total_questions": getattr(question_records, "total_questions", None),
        "dataset": getattr(question_records, "dataset", DEFAULT_DATASET_NAME),
        "split": getattr(question_records, "split", DEFAULT_DATASET_SPLIT),
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
    filename: str = "gsm8k_platinum_test.jsonl",
) -> str:
    """Write the raw GSM8K-Platinum question/answer corpus as JSONL."""

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
        raise FileNotFoundError(f"Local dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(_normalize_record(json.loads(stripped)))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("Local dataset JSON must contain a list of records.")

    return [_normalize_record(record) for record in data]


def _load_huggingface_records(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    cache_dir: str | None = None,
) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for source='huggingface'. "
            "Install project dependencies or use source='local' in offline environments."
        ) from exc

    if dataset_config is None:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=cache_dir,
        )
    return [_normalize_record(record) for record in dataset]


def _normalize_record(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question": _require_text_field(record, "question"),
        "answer": _require_text_field(record, "answer"),
    }


def _dataset_name_to_prefix(dataset_name: str) -> str:
    return dataset_name.split("/")[-1].replace("-", "_")


def _require_text_field(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"Dataset record field '{field}' must be a string.")
    return value
