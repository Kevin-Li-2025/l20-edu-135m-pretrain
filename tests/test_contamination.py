from pathlib import Path

from l20_pretrain.contamination import contamination_report, word_ngrams


def test_word_ngrams_normalizes_text() -> None:
    assert "hello world" in word_ngrams("Hello, world!", 2)


def test_contamination_report(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    benchmark = tmp_path / "benchmark.jsonl"
    train.write_text("alpha beta gamma delta epsilon zeta eta theta iota kappa", encoding="utf-8")
    benchmark.write_text(
        '{"prompt": "alpha beta gamma delta epsilon zeta eta theta iota kappa"}\n',
        encoding="utf-8",
    )

    report = contamination_report(train, benchmark, ngram=5, fail_threshold=0.0)
    assert report.overlap_ngrams > 0
    assert report.status == "fail"
