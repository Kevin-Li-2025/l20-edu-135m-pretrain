#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.is_file():
        return token_path.read_text(encoding="utf-8").strip()
    raise SystemExit("HF_TOKEN is unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a model artifact without embedded credentials.")
    parser.add_argument("--repo-id", default="AliceYin/l20-edu-135m")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--path-in-repo", default="")
    parser.add_argument("--commit-message", default="Upload model artifact")
    args = parser.parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist: {folder}")
    api = HfApi(token=resolve_token())
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=folder,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )
    print(result)


if __name__ == "__main__":
    main()
