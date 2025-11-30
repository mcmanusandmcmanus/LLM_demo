"""
Utility helpers for preparing text datasets for causal language models.

Usage (split a raw text file into train/val/test):
    python -m src.data_utils --input_file data/raw.txt --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --lower --dedupe
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List


def clean_text(line: str, lower: bool = False) -> str:
    text = line.strip()
    if lower:
        text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_lines(path: Path, lower: bool = False, dedupe: bool = False) -> List[str]:
    lines = [clean_text(line, lower=lower) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if dedupe:
        # Preserve order while removing duplicates.
        seen = set()
        unique = []
        for line in lines:
            if line not in seen:
                unique.append(line)
                seen.add(line)
        lines = unique
    return lines


def split_dataset(
    lines: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42
) -> tuple[List[str], List[str], List[str]]:
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")
    rng = random.Random(seed)
    rng.shuffle(lines)
    n = len(lines)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = lines[:n_train]
    val = lines[n_train : n_train + n_val]
    test = lines[n_train + n_val :]
    return train, val, test


def write_split(name: str, lines: List[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Split and clean a raw text file into train/val/test files.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw text file (one example per line).")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory for split files.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--lower", action="store_true", help="Lowercase all text.")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate lines.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    lines = load_lines(input_path, lower=args.lower, dedupe=args.dedupe)
    if not lines:
        raise SystemExit("Input file is empty after cleaning.")

    train, val, test = split_dataset(lines, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)

    write_split("train", train, output_dir)
    write_split("validation", val, output_dir)
    write_split("test", test, output_dir)
    print(f"Saved splits to {output_dir} (train={len(train)}, val={len(val)}, test={len(test)})")


if __name__ == "__main__":
    main()
