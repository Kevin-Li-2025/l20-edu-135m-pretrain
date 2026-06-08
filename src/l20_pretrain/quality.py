from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
import unicodedata


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
BAD_PHRASES = (
    "enable javascript",
    "cookie policy",
    "privacy policy",
    "all rights reserved",
    "lorem ipsum",
    "subscribe to our newsletter",
)
CODE_KEYWORD_RE = re.compile(
    r"\b("
    r"def|class|return|import|from|for|while|if|else|elif|try|catch|except|public|private|"
    r"static|void|int|float|double|char|const|let|var|function|async|await|SELECT|FROM|WHERE"
    r")\b"
)


@dataclass(frozen=True)
class QualityDecision:
    keep: bool
    reason: str = "ok"


def normalize_text(text: str, *, max_chars: int | None = None) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    lines = []
    previous = ""
    for raw_line in text.splitlines():
        line = SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    normalized = "\n".join(lines).strip()
    normalized = BLANK_RE.sub("\n\n", normalized)
    if max_chars is not None and len(normalized) > max_chars:
        normalized = normalized[:max_chars].rsplit("\n", 1)[0].strip() or normalized[:max_chars]
    return normalized


def normalize_code_text(text: str, *, max_chars: int | None = None) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    lines = [raw_line.rstrip() for raw_line in text.splitlines()]
    normalized = "\n".join(lines).strip()
    normalized = BLANK_RE.sub("\n\n", normalized)
    if max_chars is not None and len(normalized) > max_chars:
        normalized = normalized[:max_chars].rsplit("\n", 1)[0].strip() or normalized[:max_chars]
    return normalized


def stable_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def repeated_line_fraction(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if len(line.strip()) >= 20]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(1, len(lines))


def repeated_ngram_fraction(words: list[str], n: int = 5) -> float:
    if len(words) < n * 4:
        return 0.0
    ngrams = zip(*(words[i:] for i in range(n)))
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    total = max(1, len(words) - n + 1)
    return repeated / total


def quality_filter(
    text: str,
    *,
    min_chars: int = 500,
    max_repeated_line_fraction: float = 0.25,
    max_repeated_ngram_fraction: float = 0.18,
    min_alpha_ratio: float = 0.45,
    max_digit_ratio: float = 0.30,
    min_unique_word_ratio: float = 0.18,
) -> QualityDecision:
    if len(text) < min_chars:
        return QualityDecision(False, "too_short")

    lower = text[:5000].lower()
    if any(phrase in lower for phrase in BAD_PHRASES):
        return QualityDecision(False, "boilerplate")

    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return QualityDecision(False, "empty")
    alpha_ratio = sum(char.isalpha() for char in non_space) / len(non_space)
    digit_ratio = sum(char.isdigit() for char in non_space) / len(non_space)
    if alpha_ratio < min_alpha_ratio:
        return QualityDecision(False, "low_alpha")
    if digit_ratio > max_digit_ratio:
        return QualityDecision(False, "high_digit")

    words = [word.lower() for word in WORD_RE.findall(text)]
    if len(words) < 80:
        return QualityDecision(False, "too_few_words")
    unique_word_ratio = len(set(words)) / len(words)
    if unique_word_ratio < min_unique_word_ratio:
        return QualityDecision(False, "low_unique_words")

    if repeated_line_fraction(text) > max_repeated_line_fraction:
        return QualityDecision(False, "repeated_lines")
    if repeated_ngram_fraction(words) > max_repeated_ngram_fraction:
        return QualityDecision(False, "repeated_ngrams")

    return QualityDecision(True)


def code_quality_filter(
    text: str,
    *,
    min_chars: int = 200,
    max_line_length: int = 2000,
    max_average_line_length: float = 180.0,
    max_repeated_line_fraction: float = 0.35,
    min_unique_line_ratio: float = 0.20,
) -> QualityDecision:
    if len(text) < min_chars:
        return QualityDecision(False, "too_short")

    if "\ufffd" in text:
        return QualityDecision(False, "decode_error")

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return QualityDecision(False, "too_few_lines")

    line_lengths = [len(line) for line in lines]
    if max(line_lengths) > max_line_length:
        return QualityDecision(False, "very_long_line")
    if sum(line_lengths) / len(line_lengths) > max_average_line_length:
        return QualityDecision(False, "minified")

    unique_line_ratio = len(set(lines)) / len(lines)
    if unique_line_ratio < min_unique_line_ratio:
        return QualityDecision(False, "low_unique_lines")
    if repeated_line_fraction(text) > max_repeated_line_fraction:
        return QualityDecision(False, "repeated_lines")

    code_punctuation = sum(text.count(mark) for mark in ("{", "}", "(", ")", ";", "=", ".", "_", "<", ">"))
    keyword_count = len(CODE_KEYWORD_RE.findall(text))
    if code_punctuation < 8 and keyword_count < 2:
        return QualityDecision(False, "low_code_signal")

    return QualityDecision(True)
