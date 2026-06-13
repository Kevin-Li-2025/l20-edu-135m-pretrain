#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-final-fallback", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    evaluations: list[tuple[float, int]] = []
    for line in Path(args.log).read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "eval":
            evaluations.append((float(row["loss"]), int(row["step"])))
    available = {int(path.name.split("-")[1]): path for path in run_dir.glob("step-*")}
    candidates = [(loss, step, available[step]) for loss, step in evaluations if step in available]
    if candidates:
        loss, step, checkpoint = min(candidates)
    elif args.allow_final_fallback:
        checkpoint = (run_dir / "final").resolve()
        step = int(checkpoint.name.split("-")[1])
        loss = None
    else:
        raise SystemExit(
            "No saved checkpoint has a matching validation-loss record; "
            "refusing to label the final checkpoint as best."
        )
    if not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"Selected checkpoint is missing model.safetensors: {checkpoint}")
    payload = {"checkpoint": str(checkpoint), "step": step, "eval_loss": loss}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
