from l20_pretrain.config import load_config


def test_token_budget() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.tokens_per_step == 256
    assert config.planned_tokens == 1280


def test_architecture_ablation_uses_matching_data_budget() -> None:
    deepthin = load_config("configs/l20_135m_deepthin.yaml")
    wide = load_config("configs/l20_wide_140m_baseline.yaml")

    assert deepthin.tokenizer_name == wide.tokenizer_name
    assert deepthin.dataset.name == wide.dataset.name
    assert deepthin.dataset.config_name == wide.dataset.config_name
    assert deepthin.dataset.min_score == wide.dataset.min_score
    assert deepthin.dataset.min_int_score == wide.dataset.min_int_score
    assert deepthin.model.block_size == wide.model.block_size

    ratio = deepthin.planned_tokens / wide.planned_tokens
    assert 0.995 <= ratio <= 1.005
