#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from l20_pretrain.skill_corpus import clean_skill_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and tag skill-targeted data for 135M curriculum training."
    )
    parser.add_argument("inputs", nargs="+", help="input files or directories: jsonl/json/txt/md")
    parser.add_argument("--out", required=True, help="clean output jsonl")
    parser.add_argument("--guard-index", required=True, help="SQLite MinHash/segment guard index")
    parser.add_argument("--contamination-path", help="benchmark contamination jsonl index")
    parser.add_argument("--source-name", help="override source name in output records")
    parser.add_argument("--benchmark-similarity-path", help="benchmark jsonl for lexical similarity screening")
    parser.add_argument("--benchmark-similarity-threshold", type=float, default=0.50)
    parser.add_argument("--text-fields", default="text,content,prompt,response")
    parser.add_argument("--min-quality-score", type=float, default=0.34)
    parser.add_argument("--min-chars", type=int, default=240)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-template-repeats", type=int, default=200)
    parser.add_argument("--max-answer-label-count", type=int)
    args = parser.parse_args()

    manifest = clean_skill_corpus(
        input_paths=args.inputs,
        out_jsonl=args.out,
        guard_index=args.guard_index,
        contamination_path=args.contamination_path,
        text_fields=tuple(field.strip() for field in args.text_fields.split(",") if field.strip()),
        min_quality_score=args.min_quality_score,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_records=args.max_records,
        source_name=args.source_name,
        benchmark_similarity_path=args.benchmark_similarity_path,
        benchmark_similarity_threshold=args.benchmark_similarity_threshold,
        max_template_repeats=args.max_template_repeats,
        max_answer_label_count=args.max_answer_label_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
