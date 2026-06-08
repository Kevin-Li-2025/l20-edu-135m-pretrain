from collections import Counter

import numpy as np

from l20_pretrain.prepare_mixture_shards import SourceSpec, copy_tokenized_replay_source
from l20_pretrain.quality import normalize_code_text


def test_normalize_code_text_preserves_indentation() -> None:
    text = normalize_code_text("def f():   \n    return 1\n    return 2\n")

    assert "def f():" in text
    assert "    return 1" in text
    assert "    return 2" in text


def test_copy_tokenized_replay_source_writes_quota(tmp_path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenized_dir.mkdir()
    np.arange(100, dtype=np.uint32).tofile(tokenized_dir / "train.bin")

    source = SourceSpec(
        name="edu-replay",
        kind="tokenized_replay",
        dataset="",
        tokenized_path=str(tokenized_dir),
        sample_seed=123,
    )
    counters: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    source_tokens = {"edu-replay": 0}
    output_path = tmp_path / "replay.bin"

    with output_path.open("wb") as handle:
        written = copy_tokenized_replay_source(
            source,
            train_handle=handle,
            quota=20,
            target_tokens=20,
            train_tokens=0,
            block_size=8,
            counters=counters,
            source_counter=source_counter,
            source_tokens=source_tokens,
        )

    replay = np.fromfile(output_path, dtype=np.uint32)
    assert written == 20
    assert replay.shape == (20,)
    assert source_tokens["edu-replay"] == 20
    assert counters["tokenized_replay_chunks"] > 0
    assert source_counter["train_kept"] > 0
