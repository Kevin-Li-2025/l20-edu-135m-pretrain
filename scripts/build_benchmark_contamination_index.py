#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset


def emit(handle: Any, benchmark: str, text: str) -> None:
    text = text.strip()
    if text:
        handle.write(
            json.dumps({"benchmark": benchmark, "text": text}, ensure_ascii=True)
            + "\n"
        )


def rows(
    name: str, config: str | None, splits: tuple[str, ...]
) -> Iterator[dict[str, Any]]:
    for split in splits:
        kwargs: dict[str, Any] = {"path": name, "split": split}
        if config:
            kwargs["name"] = config
        yield from load_dataset(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build six-task decontamination text index.")
    parser.add_argument(
        "--out", default="data/benchmark_contamination/eval_6tasks.jsonl"
    )
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for config, benchmark in (
            ("ARC-Challenge", "arc_challenge"),
            ("ARC-Easy", "arc_easy"),
        ):
            for row in rows("allenai/ai2_arc", config, ("validation", "test")):
                emit(
                    handle,
                    benchmark,
                    f"{row['question']} {' '.join(row['choices']['text'])}",
                )
        for row in rows("Rowan/hellaswag", None, ("validation",)):
            emit(handle, "hellaswag", f"{row['ctx']} {' '.join(row['endings'])}")
        for row in rows("baber/piqa", None, ("validation",)):
            emit(handle, "piqa", f"{row['goal']} {row['sol1']} {row['sol2']}")
        for row in rows("EleutherAI/lambada_openai", None, ("test",)):
            emit(handle, "lambada_openai", row["text"])
        for row in rows("allenai/winogrande", "winogrande_xl", ("validation",)):
            emit(
                handle,
                "winogrande",
                f"{row['sentence']} {row['option1']} {row['option2']}",
            )

    print(
        json.dumps(
            {"event": "done", "path": str(output), "bytes": output.stat().st_size}
        )
    )


if __name__ == "__main__":
    main()
