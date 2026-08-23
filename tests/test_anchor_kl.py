import pytest
import torch

from l20_pretrain.config import load_config
from l20_pretrain.train import token_kl_divergence, training_stream_schedule


def test_anchor_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 7, 11)
    labels = torch.arange(14).reshape(2, 7)

    loss = token_kl_divergence(logits, logits, labels, chunk_size=2)

    assert abs(float(loss)) < 1e-6


def test_anchor_kl_is_positive_and_backpropagates_with_mask_and_stride() -> None:
    student = torch.randn(2, 9, 13, requires_grad=True)
    teacher = torch.randn(2, 9, 13)
    labels = torch.arange(18).reshape(2, 9)
    labels[:, 3] = -100

    loss = token_kl_divergence(
        student,
        teacher,
        labels,
        temperature=1.5,
        stride=2,
        chunk_size=2,
    )
    loss.backward()

    assert float(loss.detach()) > 0
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"stride": 0}, "stride"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_anchor_kl_rejects_invalid_controls(kwargs: dict[str, float], match: str) -> None:
    logits = torch.randn(1, 3, 5)
    labels = torch.ones(1, 3, dtype=torch.long)

    with pytest.raises(ValueError, match=match):
        token_kl_divergence(logits, logits, labels, **kwargs)


def test_two_stream_schedule_is_deterministic() -> None:
    config = load_config("configs/l20_v2_stage_b_two_stream_005.yaml")

    assert training_stream_schedule(config) == (
        "retention",
        "retention",
        "retention",
        "target",
        "target",
    )


def test_single_stream_schedule_preserves_existing_configs() -> None:
    config = load_config("configs/l20_v2_stage_a_anchor_kl_005.yaml")

    assert training_stream_schedule(config) == ("target",) * 4
