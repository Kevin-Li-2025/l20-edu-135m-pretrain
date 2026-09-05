from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from l20_pretrain.provenance import verify_shard_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify immutable packed-shard manifests.")
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="Check files, byte sizes, and token counts without reading full SHA256 hashes.",
    )
    args = parser.parse_args()

    for directory in args.directories:
        metadata = verify_shard_directory(directory, verify_hashes=not args.size_only)
        print(
            f"{directory}: verified; train_tokens={int(metadata['train_tokens']):,}; "
            f"val_tokens={int(metadata['val_tokens']):,}; hashes={not args.size_only}"
        )


if __name__ == "__main__":
    main()
