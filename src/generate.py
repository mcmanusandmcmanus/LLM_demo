"""
CLI helper to run text generation against the configured LLM service.

Usage:
    python -m src.generate "Write a product description for a solar-powered lantern."
"""

from __future__ import annotations

import argparse

from .llm_service import LLMService


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a local model.")
    parser.add_argument("prompt", type=str, help="Prompt to send to the model.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model override.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling probability.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k cutoff.")
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLMService(model_name=args.model_name or None)
    completion = llm.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(completion)


if __name__ == "__main__":
    main()
