from __future__ import annotations

from typing import Any

from transformers import LlamaConfig, LlamaForCausalLM

from .config import ModelConfig


def pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def build_model_config(config: ModelConfig, tokenizer: Any) -> LlamaConfig:
    vocab_size = pad_to_multiple(len(tokenizer), config.vocab_multiple)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    model_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.block_size,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        attention_dropout=config.attention_dropout,
        tie_word_embeddings=config.tie_word_embeddings,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    if config.attn_implementation:
        model_config._attn_implementation = config.attn_implementation
    return model_config


def build_model(config: ModelConfig, tokenizer: Any) -> LlamaForCausalLM:
    return LlamaForCausalLM(build_model_config(config, tokenizer))


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
