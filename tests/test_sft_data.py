import torch

from l20_pretrain.sft_data import IGNORE_INDEX, SFTTokenDataset, encode_sft_example, render_sft_example
from l20_pretrain.train_sft import SFTDatasetConfig, create_sft_source, load_sft_config


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, verbose: bool = False) -> dict[str, list[int]]:
        del add_special_tokens, verbose
        return {"input_ids": [ord(ch) % 31 + 1 for ch in text]}


def test_instruction_sft_masks_prompt_tokens() -> None:
    encoded = encode_sft_example(
        {"instruction": "Say hello", "input": "to Ada", "output": "Hello, Ada."},
        TinyTokenizer(),
        block_size=80,
    )

    assert encoded is not None
    labels = encoded["labels"]
    supervised = torch.nonzero(labels != IGNORE_INDEX).flatten()
    assert supervised.numel() > 0
    assert supervised[0].item() > 0
    assert torch.all(labels[: supervised[0]] == IGNORE_INDEX)
    assert labels[supervised[-1]].item() == TinyTokenizer.eos_token_id


def test_messages_sft_uses_last_assistant_turn_as_target() -> None:
    example = {
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "3 + 3?"},
            {"role": "assistant", "content": "6"},
        ]
    }

    prompt, response = render_sft_example(example)
    assert "2 + 2?" in prompt
    assert "### Response:\n4" in prompt
    assert response == "6"


def test_sft_dataset_emits_fixed_shape_rows() -> None:
    dataset = SFTTokenDataset(
        [{"prompt": "Capital of France?", "response": "Paris."}],
        TinyTokenizer(),
        block_size=96,
    )
    row = next(iter(dataset))
    assert row["input_ids"].shape == (96,)
    assert row["attention_mask"].shape == (96,)
    assert row["labels"].shape == (96,)


def test_sft_config_loads_default_recipe() -> None:
    config = load_sft_config("configs/l20_edu_135m_sft.yaml")
    assert config.base_model == "AliceYin/l20-edu-135m"
    assert config.dataset.config_name == "default"
    assert config.dataset.train_on_prompt is False
    assert config.sequences_per_step == 64


def test_local_sft_eval_source_uses_eval_jsonl(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text('{"prompt":"train","response":"A"}\n', encoding="utf-8")
    eval_path.write_text('{"prompt":"eval","response":"B"}\n', encoding="utf-8")

    config = SFTDatasetConfig(
        local_jsonl_path=str(train_path),
        eval_local_jsonl_path=str(eval_path),
        split="train",
        eval_split="eval",
    )

    train_row = next(iter(create_sft_source(config)))
    eval_row = next(iter(create_sft_source(config, split="eval")))
    assert train_row["prompt"] == "train"
    assert eval_row["prompt"] == "eval"


def test_local_sft_source_can_be_reiterated(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        '{"prompt":"first","response":"A"}\n{"prompt":"second","response":"B"}\n',
        encoding="utf-8",
    )
    config = SFTDatasetConfig(local_jsonl_path=str(train_path))

    source = create_sft_source(config)

    assert [row["prompt"] for row in source] == ["first", "second"]
    assert [row["prompt"] for row in source] == ["first", "second"]
