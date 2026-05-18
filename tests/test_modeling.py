from l20_pretrain.config import ModelConfig
from l20_pretrain.modeling import build_model, count_parameters, pad_to_multiple


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __len__(self) -> int:
        return 101


def test_vocab_padding() -> None:
    assert pad_to_multiple(101, 64) == 128


def test_build_tiny_model() -> None:
    config = ModelConfig(
        block_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_multiple=64,
    )
    model = build_model(config, TinyTokenizer())
    assert model.config.vocab_size == 128
    assert count_parameters(model) > 0
