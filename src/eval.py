"""
Evaluation utilities for causal LMs.

Examples:
    # Perplexity on held-out test data
    python -m src.eval --model_name artifacts/sample-run --test_file data/sample/test.txt

    # Perplexity on base model
    python -m src.eval --model_name distilgpt2 --test_file data/sample/test.txt --max_samples 10

    # Generate responses for prompts and compute ROUGE-L against references
    python -m src.eval --model_name distilgpt2 --prompts_file data/sample/test.txt --references_file data/sample/test.txt --num_return_sequences 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(loss))


def compute_perplexity(
    model_name: str,
    test_file: str,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> float:
    dataset = load_dataset("text", data_files={"test": test_file})["test"]
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()

    losses: List[float] = []
    with torch.no_grad():
        for sample in dataset:
            inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
            input_ids = inputs.input_ids.to(target_device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return perplexity_from_loss(avg_loss)


def generate_outputs(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 64,
    num_return_sequences: int = 1,
) -> List[str]:
    gen_pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device_map="auto",
    )
    results: List[str] = []
    for prompt in prompts:
        outputs = gen_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.7,
        )
        texts = [o["generated_text"] for o in outputs]
        results.extend(texts)
    return results


def compute_rouge(predictions: List[str], references: List[str]) -> dict:
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a causal LM for perplexity and ROUGE.")
    parser.add_argument("--model_name", type=str, required=True, help="Model checkpoint or directory.")
    parser.add_argument("--test_file", type=str, help="Path to held-out test text file for perplexity.")
    parser.add_argument("--prompts_file", type=str, help="File with prompts (one per line) for generation metrics.")
    parser.add_argument("--references_file", type=str, help="Reference completions (one per line) for ROUGE-L.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick checks.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generation length for evaluation.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of generations per prompt.")
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}

    if args.test_file:
        ppl = compute_perplexity(
            model_name=args.model_name,
            test_file=args.test_file,
            max_samples=args.max_samples,
        )
        results["perplexity"] = ppl

    if args.prompts_file and args.references_file:
        prompts = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        refs = Path(args.references_file).read_text(encoding="utf-8").splitlines()
        if len(prompts) != len(refs):
            raise SystemExit("Prompts and references must have the same number of lines.")
        preds = generate_outputs(
            model_name=args.model_name,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
        )
        if len(preds) != len(refs):
            refs = refs * (len(preds) // len(refs))
            refs = refs[: len(preds)]
        rouge_scores = compute_rouge(preds, refs)
        results["rouge"] = rouge_scores

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
