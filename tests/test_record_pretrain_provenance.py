from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "record_pretrain_provenance.py"
SPEC = importlib.util.spec_from_file_location("record_pretrain_provenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_hashes_files_in_stable_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.bin").write_bytes(b"second")
    (source / "a.bin").write_bytes(b"first")

    manifest = MODULE.build_manifest(
        [("data", str(source))],
        [("dataset_revision", "abc123")],
    )

    artifacts = manifest["artifacts"]
    assert [item["relative_path"] for item in artifacts] == ["a.bin", "b.bin"]
    assert artifacts[0]["sha256"] == MODULE.sha256_file(source / "a.bin")
    assert manifest["records"] == {"dataset_revision": "abc123"}
