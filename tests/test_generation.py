"""Tests for generation-side trace assembly and JSONL helpers."""

import json
from pathlib import Path
import shutil
import uuid

from src.generation.length_controlled import (
    GenerationOutput,
    LLMGenerator,
    _move_model_inputs_to_device,
    _snapshot_has_required_files,
    _looks_like_tensor,
    append_traces_to_jsonl,
    generate_traces_for_question,
    load_existing_trace_ids,
)


class FakeGenerator:
    """Small stand-in generator for local unit tests."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> GenerationOutput:
        self.calls += 1
        target_length = "unknown"
        system_message = messages[0]["content"]
        if "exactly " in system_message and " reasoning steps" in system_message:
            target_length = system_message.split("exactly ", maxsplit=1)[1].split(
                " reasoning steps",
                maxsplit=1,
            )[0]
        return GenerationOutput(
            raw_completion=(
                f"Step 1: Work toward target {target_length}.\n"
                "Step 2: Finish the arithmetic.\n"
                "#### 4"
            ),
            token_count=7,
        )


def test_generate_traces_for_question_builds_trace_schema() -> None:
    generator = FakeGenerator()
    prompt_template = {
        "prompt_id": "len_guided_v1",
        "system": "Use exactly {target_length} reasoning steps.",
        "few_shot": [],
        "user_template": "{question}",
    }

    traces = generate_traces_for_question(
        generator=generator,
        question_id="gsm8k_0001",
        question_text="What is 2 + 2?",
        gold_answer=4.0,
        length_grid=[3, 5],
        samples_per_length=2,
        temperature=0.7,
        max_new_tokens=64,
        prompt_template=prompt_template,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )

    assert len(traces) == 4
    assert traces[0]["trace_id"] == "gsm8k_0001_L3_1"
    assert traces[-1]["trace_id"] == "gsm8k_0001_L5_2"
    assert traces[0]["prompt_id"] == "len_guided_v1"
    assert traces[0]["actual_num_steps"] == 2
    assert traces[0]["final_answer_line"] == "#### 4"
    assert traces[0]["extracted_answer"] == 4.0
    assert traces[0]["is_correct"] is True
    assert traces[0]["token_count"] == 7
    assert "timestamp" in traces[0]
    assert generator.calls == 4


def test_load_existing_trace_ids_returns_empty_for_missing_file() -> None:
    assert load_existing_trace_ids("tests/_missing_trace_file.jsonl") == set()


def test_append_traces_to_jsonl_and_reload_ids() -> None:
    temp_dir = Path("tests") / f"_tmp_generation_{uuid.uuid4().hex}"
    output_path = temp_dir / "traces.jsonl"
    traces = [
        {"trace_id": "trace_1", "value": 1},
        {"trace_id": "trace_2", "value": 2},
    ]

    try:
        append_traces_to_jsonl(traces, str(output_path))
        lines = output_path.read_text(encoding="utf-8").splitlines()
        loaded = [json.loads(line) for line in lines]
        existing_ids = load_existing_trace_ids(str(output_path))

        assert loaded == traces
        assert existing_ids == {"trace_1", "trace_2"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_snapshot_has_required_files_detects_complete_layout() -> None:
    temp_dir = Path("tests") / f"_tmp_snapshot_complete_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
            (temp_dir / filename).write_text("{}", encoding="utf-8")

        assert _snapshot_has_required_files(temp_dir) is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_snapshot_has_required_files_detects_incomplete_layout() -> None:
    temp_dir = Path("tests") / f"_tmp_snapshot_incomplete_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "config.json").write_text("{}", encoding="utf-8")

        assert _snapshot_has_required_files(temp_dir) is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_move_model_inputs_to_device_accepts_dict_like_with_to() -> None:
    class FakeBatch(dict):
        def __init__(self) -> None:
            super().__init__({"input_ids": "ids", "attention_mask": "mask"})
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    batch = FakeBatch()
    result = _move_model_inputs_to_device(batch, "cuda:0")

    assert batch.seen_device == "cuda:0"
    assert result["input_ids"] == "ids"
    assert result["attention_mask"] == "mask"


def test_move_model_inputs_to_device_accepts_mapping_like_object() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.shape = (1, 3)
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    class FakeBatchEncoding:
        def __init__(self) -> None:
            self.payload = {
                "input_ids": FakeTensor(),
                "attention_mask": FakeTensor(),
            }
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

        def items(self):
            return self.payload.items()

    batch = FakeBatchEncoding()
    result = _move_model_inputs_to_device(batch, "cpu")

    assert batch.seen_device == "cpu"
    assert result["input_ids"].seen_device == "cpu"
    assert result["attention_mask"].seen_device == "cpu"


def test_move_model_inputs_to_device_accepts_tensor_like() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.shape = (1, 3)
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    tensor = FakeTensor()
    result = _move_model_inputs_to_device(tensor, "cpu")

    assert result["input_ids"] is tensor
    assert tensor.seen_device == "cpu"


def test_looks_like_tensor_checks_shape_and_to() -> None:
    class Tensorish:
        shape = (1,)

        def to(self, device):
            return self

    class NotTensorish:
        pass

    assert _looks_like_tensor(Tensorish()) is True
    assert _looks_like_tensor(NotTensorish()) is False


def test_prepare_model_inputs_falls_back_to_string_tokenization() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.shape = (1, 4)
            self.seen_device = None

        def to(self, device):
            self.seen_device = device
            return self

    class FakeTokenizer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=None, return_tensors=None):
            self.calls.append(f"template:{tokenize}")
            if tokenize:
                raise TypeError("unsupported backend")
            return "rendered prompt"

        def __call__(self, prompt, return_tensors="pt", add_special_tokens=False):
            self.calls.append("tokenizer_call")
            assert prompt == "rendered prompt"
            return {"input_ids": FakeTensor()}

    generator = object.__new__(LLMGenerator)
    generator.tokenizer = FakeTokenizer()
    generator.device = "cpu"

    result = LLMGenerator._prepare_model_inputs(generator, [{"role": "user", "content": "hi"}])

    assert generator.tokenizer.calls == ["template:True", "template:False", "tokenizer_call"]
    assert "input_ids" in result
    assert result["input_ids"].seen_device == "cpu"
