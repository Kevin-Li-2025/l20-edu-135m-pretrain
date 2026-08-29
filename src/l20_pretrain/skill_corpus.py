from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any

from .contamination import iter_strings, normalize_tokens
from .data_guard import CrossSourceDataGuard
from .quality import normalize_text, quality_filter, stable_hash


SKILL_TAGS = {
    "arc_science",
    "hellaswag_continuation",
    "lambada_cloze",
    "piqa_physical",
    "winogrande_coreference",
    "textbook_reasoning",
    "python_edu",
    "general_edu",
}

QUESTION_RE = re.compile(r"\?|which of the following|choose the|multiple choice", re.IGNORECASE)
SCIENCE_RE = re.compile(
    r"\b(force|energy|cell|planet|temperature|water|organism|electric|chemical|"
    r"gravity|ecosystem|molecule|experiment|evidence|hypothesis)\b",
    re.IGNORECASE,
)
CONTINUATION_RE = re.compile(r"\b(then|after|before|while|suddenly|next|finally|because)\b", re.IGNORECASE)
PHYSICAL_RE = re.compile(
    r"\b(push|pull|lift|drop|heat|cool|break|cut|hold|pour|open|close|move|"
    r"object|tool|container|surface|weight|friction)\b",
    re.IGNORECASE,
)
COREFERENCE_RE = re.compile(r"\b(he|she|they|him|her|them|his|hers|their|it)\b", re.IGNORECASE)
CODE_RE = re.compile(r"\b(def|class|return|import|for|while|if|else|function|print)\b")
REASONING_RE = re.compile(r"\b(therefore|because|so|step|explain|reason|answer|solution)\b", re.IGNORECASE)
ANSWER_RE = re.compile(r"\b(answer|solution|correct option|final)\b", re.IGNORECASE)
ANSWER_LABEL_RE = re.compile(
    r"\b(?:answer|correct(?:\s+option|\s+continuation)?|label)\s*(?:is|:)?\s*([ABCD])\b",
    re.IGNORECASE,
)
TEMPLATE_NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
TEMPLATE_QUOTE_RE = re.compile(r'"[^"]{4,}"|\'[^\']{4,}\'')
TEMPLATE_CHOICE_RE = re.compile(r"\b(option|choice|answer|label|correct option)\s*[: ]\s*[abcd]\b", re.IGNORECASE)


@dataclass(frozen=True)
class SkillRecord:
    text: str
    source: str
    skill: str
    quality_score: float
    digest: str
    token_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Rejection:
    source: str
    reason: str
    digest: str | None = None
    skill: str | None = None


class BenchmarkSimilarityIndex:
    def __init__(self, path: str | Path, *, threshold: float = 0.50):
        self.path = Path(path)
        self.threshold = threshold
        self.records: list[tuple[str, set[str]]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                tokens = set(normalize_tokens(str(payload.get("text", ""))))
                if tokens:
                    self.records.append((str(payload.get("benchmark", "unknown")), tokens))

    def match(self, text: str) -> tuple[str, float] | None:
        tokens = set(normalize_tokens(text))
        if not tokens:
            return None
        best: tuple[str, float] | None = None
        for benchmark, benchmark_tokens in self.records:
            intersection = len(tokens & benchmark_tokens)
            if intersection == 0:
                continue
            jaccard = intersection / len(tokens | benchmark_tokens)
            containment = intersection / min(len(tokens), len(benchmark_tokens))
            score = max(jaccard, containment)
            if score >= self.threshold and (best is None or score > best[1]):
                best = (benchmark, score)
        return best


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                yield {"text": line}
                continue
            if isinstance(payload, dict):
                yield payload
            else:
                yield {"text": payload}


def iter_text_records(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk in re.split(r"\n\s*\n+", text):
        chunk = chunk.strip()
        if chunk:
            yield {"text": chunk}


def iter_source_records(paths: Iterable[str | Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    for raw_path in paths:
        path = Path(raw_path)
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for file_path in files:
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix == ".jsonl":
                iterator = iter_jsonl_records(file_path)
            elif suffix in {".txt", ".md"}:
                iterator = iter_text_records(file_path)
            elif suffix == ".json":
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
                iterator = ({"text": value} for value in iter_strings(payload))
            else:
                continue
            for record in iterator:
                yield file_path, record


def record_text(record: dict[str, Any], text_fields: tuple[str, ...]) -> str:
    for field in text_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    strings = list(iter_strings(record))
    return "\n\n".join(strings)


def infer_skill(text: str, declared: str | None = None) -> str:
    if declared in SKILL_TAGS:
        return declared
    lower = text.lower()
    tokens = normalize_tokens(text)
    if CODE_RE.search(text):
        return "python_edu"
    if SCIENCE_RE.search(text) and (QUESTION_RE.search(text) or REASONING_RE.search(text)):
        return "arc_science"
    if PHYSICAL_RE.search(text) and (QUESTION_RE.search(text) or ANSWER_RE.search(text)):
        return "piqa_physical"
    if COREFERENCE_RE.search(text) and QUESTION_RE.search(text):
        return "winogrande_coreference"
    if len(tokens) >= 120 and (lower.rstrip().endswith(".") or lower.rstrip().endswith('"')):
        return "lambada_cloze"
    if CONTINUATION_RE.search(text) and len(tokens) >= 80:
        return "hellaswag_continuation"
    if REASONING_RE.search(text):
        return "textbook_reasoning"
    return "general_edu"


def template_signature(text: str, skill: str, *, max_tokens: int = 96) -> str:
    text = TEMPLATE_QUOTE_RE.sub("<quote>", text.lower())
    text = TEMPLATE_CHOICE_RE.sub(r"\1 <label>", text)
    text = TEMPLATE_NUMBER_RE.sub("<num>", text)
    tokens = normalize_tokens(text)[:max_tokens]
    if not tokens:
        return stable_hash(skill)
    return stable_hash(skill + "\n" + " ".join(tokens))


def extract_answer_label(record: dict[str, Any], text: str) -> str | None:
    for key in ("answer", "label", "correct", "correct_answer", "target"):
        value = record.get(key)
        if isinstance(value, str):
            stripped = value.strip().upper()
            if stripped in {"A", "B", "C", "D"}:
                return stripped
    match = ANSWER_LABEL_RE.search(text)
    return match.group(1).upper() if match else None


def lexical_quality_score(text: str, skill: str) -> float:
    tokens = normalize_tokens(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    length_score = min(1.0, len(tokens) / 256)
    reasoning_bonus = 0.12 if REASONING_RE.search(text) else 0.0
    answer_bonus = 0.08 if ANSWER_RE.search(text) else 0.0
    skill_bonus = 0.0
    if skill == "arc_science" and SCIENCE_RE.search(text):
        skill_bonus = 0.12
    elif skill == "hellaswag_continuation" and CONTINUATION_RE.search(text):
        skill_bonus = 0.10
    elif skill == "piqa_physical" and PHYSICAL_RE.search(text):
        skill_bonus = 0.12
    elif skill == "winogrande_coreference" and COREFERENCE_RE.search(text):
        skill_bonus = 0.08
    elif skill == "python_edu" and CODE_RE.search(text):
        skill_bonus = 0.12
    return min(1.0, 0.50 * unique_ratio + 0.30 * length_score + reasoning_bonus + answer_bonus + skill_bonus)


def build_skill_record(
    *,
    text: str,
    source: str,
    skill: str,
    metadata: dict[str, Any] | None = None,
) -> SkillRecord:
    tokens = normalize_tokens(text)
    digest = stable_hash(text)
    return SkillRecord(
        text=text,
        source=source,
        skill=skill,
        quality_score=lexical_quality_score(text, skill),
        digest=digest,
        token_count=len(tokens),
        metadata=metadata or {},
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def clean_skill_corpus(
    *,
    input_paths: Iterable[str | Path],
    out_jsonl: str | Path,
    guard_index: str | Path,
    contamination_path: str | Path | None = None,
    text_fields: tuple[str, ...] = ("text", "content", "prompt", "response"),
    min_quality_score: float = 0.34,
    min_chars: int = 240,
    max_chars: int = 12000,
    max_records: int | None = None,
    source_name: str | None = None,
    benchmark_similarity_path: str | Path | None = None,
    benchmark_similarity_threshold: float = 0.50,
    max_template_repeats: int = 200,
    max_answer_label_count: int | None = None,
) -> dict[str, Any]:
    out_path = Path(out_jsonl)
    reject_path = out_path.with_suffix(".rejected.jsonl")
    guard = CrossSourceDataGuard(
        guard_index,
        contamination_path=contamination_path,
        max_duplicate_segment_fraction=0.25,
    )
    benchmark_similarity = (
        BenchmarkSimilarityIndex(
            benchmark_similarity_path,
            threshold=benchmark_similarity_threshold,
        )
        if benchmark_similarity_path
        else None
    )
    kept: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    answer_label_counts: Counter[str] = Counter()
    seen = 0

    try:
        for file_path, payload in iter_source_records(input_paths):
            if max_records is not None and seen >= max_records:
                break
            seen += 1
            source = source_name or str(file_path)
            raw_text = record_text(payload, text_fields)
            text = normalize_text(raw_text, max_chars=max_chars)
            digest = stable_hash(text) if text else None
            if not text:
                counters["empty"] += 1
                rejections.append(asdict(Rejection(source=source, reason="empty", digest=digest)))
                continue
            decision = quality_filter(text, min_chars=min_chars)
            if not decision.keep:
                counters[decision.reason] += 1
                rejections.append(asdict(Rejection(source=source, reason=decision.reason, digest=digest)))
                continue

            declared_skill = payload.get("skill") if isinstance(payload.get("skill"), str) else None
            skill = infer_skill(text, declared_skill)
            record = build_skill_record(
                text=text,
                source=source,
                skill=skill,
                metadata={key: value for key, value in payload.items() if key not in set(text_fields)},
            )
            if record.quality_score < min_quality_score:
                counters["low_skill_quality"] += 1
                rejections.append(
                    asdict(
                        Rejection(
                            source=source,
                            reason="low_skill_quality",
                            digest=record.digest,
                            skill=skill,
                        )
                    )
                )
                continue

            signature = template_signature(text, skill)
            if template_counts[signature] >= max_template_repeats:
                counters["template_cap"] += 1
                rejections.append(
                    asdict(Rejection(source=source, reason="template_cap", digest=record.digest, skill=skill))
                )
                continue

            answer_label = extract_answer_label(payload, text)
            if answer_label and max_answer_label_count is not None and answer_label_counts[answer_label] >= max_answer_label_count:
                counters[f"answer_label_cap_{answer_label}"] += 1
                rejections.append(
                    asdict(
                        Rejection(
                            source=source,
                            reason=f"answer_label_cap_{answer_label}",
                            digest=record.digest,
                            skill=skill,
                        )
                    )
                )
                continue

            guard_decision, signature, segments = guard.evaluate(text)
            if not guard_decision.keep:
                reason = guard_decision.reason
                if guard_decision.benchmark:
                    reason = f"{reason}:{guard_decision.benchmark}"
                counters[reason] += 1
                rejections.append(
                    asdict(Rejection(source=source, reason=reason, digest=record.digest, skill=skill))
                )
                continue

            if benchmark_similarity:
                match = benchmark_similarity.match(text)
                if match:
                    benchmark, score = match
                    counters[f"benchmark_similarity:{benchmark}"] += 1
                    rejections.append(
                        asdict(
                            Rejection(
                                source=source,
                                reason=f"benchmark_similarity:{benchmark}:{score:.3f}",
                                digest=record.digest,
                                skill=skill,
                            )
                        )
                    )
                    continue

            guard.add(text=text, source=source, signature=signature, segments=segments)
            row = asdict(record)
            if answer_label:
                row["answer_label"] = answer_label
                answer_label_counts[answer_label] += 1
            kept.append(row)
            counters["kept"] += 1
            skill_counts[skill] += 1
            source_counts[source] += 1
            token_counts[skill] += record.token_count
            template_counts[template_signature(text, skill)] += 1
    finally:
        guard.close()

    write_jsonl(out_path, kept)
    write_jsonl(reject_path, rejections)
    manifest = {
        "status": "pass",
        "input_records": seen,
        "kept_records": len(kept),
        "rejected_records": len(rejections),
        "output": str(out_path),
        "rejected_output": str(reject_path),
        "guard_index": str(guard_index),
        "contamination_path": str(contamination_path) if contamination_path else None,
        "benchmark_similarity_path": str(benchmark_similarity_path) if benchmark_similarity_path else None,
        "benchmark_similarity_threshold": benchmark_similarity_threshold,
        "min_quality_score": min_quality_score,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "max_template_repeats": max_template_repeats,
        "max_answer_label_count": max_answer_label_count,
        "counters": dict(sorted(counters.items())),
        "answer_label_counts": dict(sorted(answer_label_counts.items())),
        "skill_counts": dict(sorted(skill_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "skill_token_counts": dict(sorted(token_counts.items())),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
