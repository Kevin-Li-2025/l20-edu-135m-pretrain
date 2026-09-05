from pathlib import Path

from l20_pretrain.data_guard import (
    BenchmarkContaminationIndex,
    CrossSourceDataGuard,
    canonical_segment,
    minhash_signature,
    signature_similarity,
    token_lcs_ratio,
    word_shingle_hashes,
)


def test_canonical_segment_masks_template_values() -> None:
    left = canonical_segment("Order 123 at https://example.com/a")
    right = canonical_segment("Order 999 at https://other.test/b")
    assert left == right


def test_minhash_detects_near_duplicate() -> None:
    base = " ".join(f"word{index}" for index in range(300))
    edited = base.replace("word150", "changed")
    assert signature_similarity(minhash_signature(base), minhash_signature(edited)) >= 0.82


def test_bottom_k_shingles_are_stable_under_prefix_insertion() -> None:
    base = " ".join(f"token{index}" for index in range(6000))
    edited = "new prefix words " + base
    assert signature_similarity(minhash_signature(base), minhash_signature(edited)) >= 0.95


def test_token_lcs_ratio_uses_benchmark_length() -> None:
    benchmark = "alpha beta gamma delta epsilon".split()
    document = "prefix alpha beta gamma delta suffix".split()
    assert token_lcs_ratio(document, benchmark) == 0.8


def test_benchmark_contamination_requires_ngram_and_lcs(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmarks.jsonl"
    benchmark_path.write_text(
        '{"benchmark":"arc","text":"alpha beta gamma delta epsilon zeta eta theta"}\n',
        encoding="utf-8",
    )
    index = BenchmarkContaminationIndex(benchmark_path, ngram=3, lcs_threshold=0.6)
    assert index.match("prefix alpha beta gamma delta epsilon suffix") == ("arc", 0.625)


def test_cross_source_guard_persists_near_duplicates(tmp_path: Path) -> None:
    guard = CrossSourceDataGuard(tmp_path / "guard.sqlite", similarity_threshold=0.8)
    text = " ".join(f"educational token{index}" for index in range(300))
    decision, signature, segments = guard.evaluate(text)
    assert decision.keep
    guard.add(text=decision.text, source="first", signature=signature, segments=segments)
    guard.connection.commit()

    edited = text.replace("token150", "replacement")
    duplicate, _, _ = guard.evaluate(edited)
    guard.close()
    assert not duplicate.keep
    assert duplicate.reason == "near_duplicate"


def test_word_shingle_hashes_accept_unicode_text() -> None:
    hashes = word_shingle_hashes("教育 quality 数据 pipeline with explicit bytes", n=3)

    assert hashes.size > 0
