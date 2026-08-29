import math

import pytest

from l20_pretrain.config import ModelConfig
from l20_pretrain.modeling import (
    build_model,
    build_model_config,
    count_parameters,
    pad_to_multiple,
)


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __len__(self) -> int:
        return 101


class DummyTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = None

    def __len__(self) -> int:
        return 49_152


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


def test_initializer_range_is_forwarded_to_llama_config() -> None:
    config = ModelConfig(initializer_range=1.0 / 24.0, vocab_multiple=1)

    model_config = build_model_config(config, DummyTokenizer())

    assert model_config.initializer_range == 1.0 / 24.0
    assert model_config.vocab_size == 49_152


def test_nanotron_style_init_scales_residual_output_projections() -> None:
    config = ModelConfig(
        block_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        initializer_range=0.04,
        scale_residual_projections=True,
        vocab_multiple=1,
    )

    model = build_model(config, DummyTokenizer())
    q_std = float(model.model.layers[0].self_attn.q_proj.weight.std().detach())
    o_std = float(model.model.layers[0].self_attn.o_proj.weight.std().detach())
    up_std = float(model.model.layers[0].mlp.up_proj.weight.std().detach())
    down_std = float(model.model.layers[0].mlp.down_proj.weight.std().detach())

    assert q_std == pytest.approx(0.04, rel=0.08)
    assert up_std == pytest.approx(0.04, rel=0.08)
    expected_residual_std = 0.04 / math.sqrt(4.0)
    assert o_std == pytest.approx(expected_residual_std, rel=0.08)
    assert down_std == pytest.approx(expected_residual_std, rel=0.08)
