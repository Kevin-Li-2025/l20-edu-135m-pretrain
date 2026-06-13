#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def emit(handle, benchmark: str, text: str) -> None:
    text = text.strip()
    if text:
        handle.write(json.dumps({"benchmark": benchmark, "text": text}, ensure_ascii=True) + "\n")


def iter_dataset(name: str, config: str | None, splits: tuple[str, ...]):
    for split in splits:
        kwargs = {"path": name, "split": split}
        if config:
            kwargs["name"] = config
        yield from load_dataset(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/benchmark_contamination/eval_5tasks.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as handle:
        for row in iter_dataset("allenai/ai2_arc", "ARC-Challenge", ("validation", "test")):
            choices = " ".join(row["choices"]["text"])
            emit(handle, "arc_challenge", f"{row['question']} {choices}")

        for row in iter_dataset("Rowan/hellaswag", None, ("validation",)):
            emit(handle, "hellaswag", f"{row['ctx']} {' '.join(row['endings'])}")

        for row in iter_dataset("baber/piqa", None, ("validation",)):
            emit(handle, "piqa", f"{row['goal']} {row['sol1']} {row['sol2']}")

        for row in iter_dataset("EleutherAI/lambada_openai", None, ("test",)):
            emit(handle, "lambada_openai", row["text"])

        for row in iter_dataset("allenai/winogrande", "winogrande_xl", ("validation",)):
            emit(
                handle,
                "winogrande",
                f"{row['sentence']} {row['option1']} {row['option2']}",
            )

    print(json.dumps({"out": str(out), "bytes": out.stat().st_size}))


if __name__ == "__main__":
    main()
