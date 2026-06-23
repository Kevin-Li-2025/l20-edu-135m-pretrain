#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from l20_pretrain.rlvr_rewards import gsm8k_reward_func


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM8K RLVR with TRL GRPO.")
    parser.add_argument("--model", default="AliceYin/l20-edu-135m")
    parser.add_argument("--train-data", default="data/rlvr/gsm8k_train.jsonl")
    parser.add_argument("--output-dir", default="runs/l20-edu-135m-rlvr-gsm8k-grpo")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-completion-length", type=int, default=320)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use-vllm", action="store_true")
    args = parser.parse_args()

    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=[],
        use_vllm=args.use_vllm,
    )
    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=gsm8k_reward_func,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(Path(args.output_dir) / "final")
    tokenizer.save_pretrained(Path(args.output_dir) / "final")


if __name__ == "__main__":
    main()
