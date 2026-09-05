import numpy as np

from l20_pretrain.provenance import verify_shard_directory
from l20_pretrain.smoke_shards import create_synthetic_smoke_shards


def test_synthetic_smoke_shards_are_distinct_and_reproducible(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "block_size": 8,
        "train_blocks": 4,
        "val_blocks": 2,
        "vocab_size": 31,
        "seed": 17,
    }

    create_synthetic_smoke_shards(first, **kwargs)
    create_synthetic_smoke_shards(second, **kwargs)
    verify_shard_directory(first)
    verify_shard_directory(second)

    first_train = np.fromfile(first / "train.bin", dtype=np.uint32)
    first_val = np.fromfile(first / "val.bin", dtype=np.uint32)
    second_train = np.fromfile(second / "train.bin", dtype=np.uint32)
    assert np.array_equal(first_train, second_train)
    assert not np.array_equal(first_train[: first_val.size], first_val)
