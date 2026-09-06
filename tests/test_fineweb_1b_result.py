import json
from pathlib import Path

from scripts.verify_fineweb_1b_result import verify_result


ROOT = Path(__file__).resolve().parents[1]


def test_committed_fineweb_1b_receipt_is_internally_consistent() -> None:
    path = ROOT / "results/fineweb_1b/factorial_20260906.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    verify_result(payload, repo_root=ROOT)
