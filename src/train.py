"""
Lightweight fine-tuning script for causal language models using Hugging Face transformers.

Example:
    python -m src.train --train_file data/sample/train.txt --validation_file data/sample/validation.txt --output_dir artifacts/sample-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class TrainingConfig:
    model_name: str
    train_file: str
    validation_file: str
    output_dir: str
    num_train_epochs: int
    batch_size: int
    learning_rate: float
    block_size: int
    warmup_steps: int
    weight_decay: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    seed: int
    use_cpu: bool


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM on custom text data.")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Base model to start from.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to a UTF-8 text file for training.")
    parser.add_argument(
        "--validation_file", type=str, required=True, help="Path to a UTF-8 text file for validation."
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/run", help="Directory to save checkpoints.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Optimizer learning rate.")
    parser.add_argument("--block_size", type=int, default=128, help="Context window length for training sequences.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps for the scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--logging_steps", type=int, default=10, help="How often to log training metrics.")
    parser.add_argument("--eval_steps", type=int, default=50, help="How often to run eval.")
    parser.add_argument("--save_steps", type=int, default=200, help="How often to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU training even if CUDA is available.")
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def tokenize_function(tokenizer, examples: Dict[str, List[str]]):
    return tokenizer(examples["text"])


def group_texts(examples: Dict[str, List[List[int]]], block_size: int):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def main():
    config = parse_args()
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset(
        "text",
        data_files={"train": config.train_file, "validation": config.validation_file},
    )

    tokenized = raw_datasets.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )

    lm_datasets = tokenized.map(
        lambda x: group_texts(x, config.block_size),
        batched=True,
        num_proc=1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available() and not config.use_cpu,
        gradient_accumulation_steps=1,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
