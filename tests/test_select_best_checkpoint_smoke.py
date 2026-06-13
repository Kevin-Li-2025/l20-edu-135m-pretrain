from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    root = Path("/tmp/select_best_fixture")
    shutil.rmtree(root, ignore_errors=True)
    for step in (100, 200):
        checkpoint = root / f"step-{step:06d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").touch()
    log = Path("/tmp/select_best.log")
    log.write_text(
        "\n".join(
            [
                json.dumps({"event": "eval", "step": 100, "loss": 2.5}),
                json.dumps({"event": "eval", "step": 200, "loss": 2.3}),
            ]
        )
    )
    command = [
        sys.executable,
        "scripts/select_best_pretrain_checkpoint.py",
        "--log",
        str(log),
        "--run-dir",
        str(root),
        "--out",
        "/tmp/select_best.json",
    ]
    subprocess.run(command, check=True)
    assert json.loads(Path("/tmp/select_best.json").read_text())["step"] == 200
    log.write_text("")
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "refusing" in failed.stderr
    print("select_best_checkpoint_fault_injection_ok")


if __name__ == "__main__":
    main()
