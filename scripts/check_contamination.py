#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.contamination import contamination_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple n-gram contamination check.")
    parser.add_argument("--train", required=True, help="Local training text sample file or directory.")
    parser.add_argument("--benchmark", required=True, help="Benchmark sample/log file or directory.")
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--fail-threshold", type=float, default=0.001)
    parser.add_argument("--max-train-texts", type=int, default=None)
    parser.add_argument("--max-benchmark-texts", type=int, default=None)
    parser.add_argument("--out", default="eval_results/contamination_report.json")
    args = parser.parse_args()

    report = contamination_report(
        args.train,
        args.benchmark,
        ngram=args.ngram,
        fail_threshold=args.fail_threshold,
        max_train_texts=args.max_train_texts,
        max_benchmark_texts=args.max_benchmark_texts,
    )
    payload = {
        "train_ngrams": report.train_ngrams,
        "benchmark_ngrams": report.benchmark_ngrams,
        "overlap_ngrams": report.overlap_ngrams,
        "overlap_rate": report.overlap_rate,
        "status": report.status,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if report.status == "pass" else 2)


if __name__ == "__main__":
    main()
