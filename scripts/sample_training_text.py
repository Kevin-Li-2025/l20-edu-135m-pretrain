#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from l20_pretrain.env import set_default_hf_home

set_default_hf_home()

from l20_pretrain.config import load_config
from l20_pretrain.data import create_source, iter_filtered_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample filtered training texts for contamination checks.")
    parser.add_argument("config", type=str)
    parser.add_argument("--docs", type=int, default=10000)
    parser.add_argument("--out", default="data/train_sample.txt")
    args = parser.parse_args()

    config = load_config(args.config)
    source = create_source(config.dataset)
    texts = iter_filtered_texts(
        source,
        text_column=config.dataset.text_column,
        min_chars=config.dataset.min_chars,
        max_chars=config.dataset.max_chars,
        min_score=config.dataset.min_score,
        min_int_score=config.dataset.min_int_score,
        max_docs=args.docs,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for text in texts:
            normalized = " ".join(text.split())
            if normalized:
                handle.write(normalized)
                handle.write("\n\n")
    print(out_path)


if __name__ == "__main__":
    main()
