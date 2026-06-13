#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


MODEL_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "*.yaml",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="AliceYin/l20-edu-135m")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--revision-dir", default="stage4-best")
    parser.add_argument("--root-model", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache/huggingface/token"
        token = token_path.read_text().strip() if token_path.is_file() else None
    if not token:
        raise SystemExit("HF_TOKEN is unavailable")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    archive_result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.model_dir,
        path_in_repo=f"releases/{args.revision_dir}",
        allow_patterns=MODEL_PATTERNS,
        commit_message=f"Archive {args.revision_dir} model",
    )
    if args.root_model:
        root_result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=args.model_dir,
            path_in_repo="",
            allow_patterns=MODEL_PATTERNS,
            delete_patterns=[
                "config.json",
                "generation_config.json",
                "model*.safetensors",
                "pytorch_model*.bin",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ],
            commit_message=f"Publish {args.revision_dir} as the default model",
        )
        print(root_result)
    eval_result = None
    if not args.skip_eval:
        eval_result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=args.eval_dir,
            path_in_repo="eval_results/stage4-release",
            commit_message="Upload Stage 4 evaluation artifacts",
        )
    card_result = api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=args.model_card,
        path_in_repo="README.md",
        commit_message=f"Update model card for {args.revision_dir}",
    )
    info = api.model_info(args.repo_id, files_metadata=False)
    filenames = {item.rfilename for item in info.siblings or []}
    if args.root_model and not {"config.json", "tokenizer.json"}.issubset(filenames):
        raise RuntimeError("Published root model is missing config.json or tokenizer.json")
    print(archive_result)
    if eval_result is not None:
        print(eval_result)
    print(card_result)
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
