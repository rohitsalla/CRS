"""
utils.py

General-purpose utilities for:
    • Dataset loading and normalization
    • Accuracy, softmax, entropy, confidence
    • File operations and logging
    • Text normalization
    • Timing utilities
    • Seed fixing for reproducibility
"""

import os
import json
import csv
import time
import random
import logging
import string
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch


# =====================================================================================
# Logging utilities
# =====================================================================================

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Configures and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = get_logger("CRS-UTILS")


# =====================================================================================
# Seed fixing
# =====================================================================================

def set_seed(seed: int = 42):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


# =====================================================================================
# File utilities
# =====================================================================================

def ensure_dir(path: str):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
    logger.info(f"Wrote JSON to {path}")


def read_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# =====================================================================================
# Dataset utilities
# =====================================================================================

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower().strip()

    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    while "  " in s:
        s = s.replace("  ", " ")

    return s


def exact_match(pred: str, gold: str) -> bool:
    """Case-insensitive exact match after normalization."""
    return normalize_answer(pred) == normalize_answer(gold)


def load_qa_dataset(path: str) -> List[Dict[str, str]]:
    """
    Loads any QA dataset in JSON or CSV.
    Required fields:
        question, answer
    """
    ext = os.path.splitext(path)[1]

    if ext == ".json":
        data = read_json(path)
        if isinstance(data, dict):
            # assume {"data": [...]}
            data = data.get("data", [])
    elif ext == ".csv":
        data = read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    clean = []
    for row in data:
        if "question" in row and "answer" in row:
            clean.append({
                "question": row["question"],
                "answer": row["answer"]
            })

    logger.info(f"Loaded dataset: {path} ({len(clean)} samples)")
    return clean


# =====================================================================================
# Text helpers
# =====================================================================================

def clean_text(s: str) -> str:
    """Removes weird characters and compresses spaces."""
    s = s.replace("\n", " ").replace("\t", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def strip_non_ascii(s: str) -> str:
    """Keeps only ASCII characters."""
    return "".join(c for c in s if ord(c) < 128)


# =====================================================================================
# Accuracy & statistics
# =====================================================================================

def compute_accuracy(preds: List[str], golds: List[str]) -> float:
    """Exact-match accuracy for QA tasks."""
    correct = sum(1 for p, g in zip(preds, golds) if exact_match(p, g))
    return correct / len(preds)


def softmax(x: np.ndarray) -> np.ndarray:
    """Applies softmax to a NumPy array."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def entropy(probs: np.ndarray) -> float:
    """Computes the Shannon entropy of a probability distribution."""
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def confidence_from_logits(logits: torch.Tensor) -> float:
    """Returns the maximum softmax probability of logits."""
    probs = torch.softmax(logits, dim=-1)
    return float(torch.max(probs).item())


# =====================================================================================
# Statistical Tools
# =====================================================================================

def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return float(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def bootstrap_ci(values: List[float], n_samples: int = 200, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Computes bootstrap confidence interval.
    """
    if len(values) < 2:
        return (values[0], values[0])

    values = np.array(values)
    means = []

    for _ in range(n_samples):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())

    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


# =====================================================================================
# Timer utility
# =====================================================================================

class Timer:
    """Simple timer for benchmarking blocks of code."""

    def __init__(self):
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"Elapsed: {self.duration:.2f}s")


# =====================================================================================
# Device helper
# =====================================================================================

def get_device() -> str:
    """Returns 'cuda' if GPU available else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================================
# JSONL support
# =====================================================================================

def read_jsonl(path: str) -> List[Any]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(items: List[Any], path: str):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Wrote JSONL file: {path}")
