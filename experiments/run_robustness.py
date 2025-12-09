#!/usr/bin/env python3
"""
run_robustness.py

Computes the Robustness Pillar (R) of the Composite Reliability Score (CRS):

    R = 1 - (AvgAccuracyDrop / CleanAccuracy)

Perturbations evaluated:
    1. Noisy Input (typos)
    2. Paraphrased Input (semantic rephrasing)
    3. Adversarial Rewrite (text-based adversarial variant)

Outputs:
    robustness_results/
        - accuracy_report.json
        - accuracy_report.csv
        - robustness_barplot.png
        - debug_samples.txt

"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Utility Helpers
# ======================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# ======================================================================
# Loading
# ======================================================================

def load_predictions(path):
    ext = os.path.splitext(path)[1]
    if ext == ".npz":
        data = np.load(path)
        return data["logits"], data["labels"]
    else:
        raise ValueError("Unsupported format. Use .npz")


def load_texts(path):
    return open(path).read().splitlines()


# ======================================================================
# Accuracy
# ======================================================================

def compute_accuracy_from_logits(logits, labels):
    preds = logits.argmax(axis=1)
    return float((preds == labels).mean())


# ======================================================================
# Perturbations
# ======================================================================

def add_noise(texts, rate=0.05):
    """Simulate typos."""
    noisy = []
    for t in texts:
        chars = list(t)
        k = max(1, int(len(chars) * rate))
        for _ in range(k):
            i = np.random.randint(0, len(chars))
            chars[i] = chr(ord(chars[i]) + np.random.randint(-1, 2))
        noisy.append("".join(chars))
    return noisy


def paraphrase_texts(texts):
    """
    Dummy paraphraser. In real experiments, swap with MarianMT or Pegasus.
    """
    return ["[PARA] " + t for t in texts]


def adversarial_texts(texts):
    """
    Simple adversarial transformation placeholder.
    Replace with TextAttack or OpenAttack for real adversarial QA.
    """
    return ["[ADV] " + t for t in texts]


# ======================================================================
# Plotting
# ======================================================================

def plot_accuracy_drop(clean_acc, noisy_acc, para_acc, adv_acc, outpath):
    drops = [
        clean_acc - noisy_acc,
        clean_acc - para_acc,
        clean_acc - adv_acc,
    ]
    labels = ["Noisy", "Paraphrased", "Adversarial"]

    plt.figure(figsize=(6,4))
    plt.bar(labels, drops, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylabel("Accuracy Drop")
    plt.title("Robustness Under Perturbations")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ======================================================================
# Robustness Score
# ======================================================================

def compute_robustness(clean_acc, noisy_acc, para_acc, adv_acc):
    avg_drop = (
        (clean_acc - noisy_acc)
      + (clean_acc - para_acc)
      + (clean_acc - adv_acc)
    ) / 3.0

    return 1.0 - (avg_drop / (clean_acc + 1e-8))


# ======================================================================
# (Optional) Simulated Model Re-run
# Replace this with HuggingFace inference for real evaluation.
# ======================================================================

def simulate_perturbed_logits(original_logits, noise_std):
    """
    Instead of calling the model again, simply add noise to logits.
    """
    return original_logits + np.random.normal(0, noise_std, original_logits.shape)


# ======================================================================
# Main Robustness Pipeline
# ======================================================================

def run_robustness(logits_file, text_file, outdir):
    ensure_dir(outdir)
    print("\n=== Running Robustness Evaluation ===")

    # 1. Load logits, labels, texts
    logits, labels = load_predictions(logits_file)
    texts = load_texts(text_file)

    # 2. Compute clean accuracy
    clean_acc = compute_accuracy_from_logits(logits, labels)

    # 3. Generate perturbations
    noisy_texts = add_noise(texts)
    para_texts = paraphrase_texts(texts)
    adv_texts  = adversarial_texts(texts)

    # 4. (Simulated) Compute perturbed logits
    # Replace these with real model calls in production
    noisy_logits = simulate_perturbed_logits(logits, noise_std=0.02)
    para_logits  = simulate_perturbed_logits(logits, noise_std=0.03)
    adv_logits   = simulate_perturbed_logits(logits, noise_std=0.04)

    # 5. Compute accuracies
    noisy_acc = compute_accuracy_from_logits(noisy_logits, labels)
    para_acc  = compute_accuracy_from_logits(para_logits, labels)
    adv_acc   = compute_accuracy_from_logits(adv_logits, labels)

    # 6. Compute final robustness metric
    R = compute_robustness(clean_acc, noisy_acc, para_acc, adv_acc)

    # 7. Save outputs
    report_json = {
        "clean_accuracy": float(clean_acc),
        "noisy_accuracy": float(noisy_acc),
        "paraphrased_accuracy": float(para_acc),
        "adversarial_accuracy": float(adv_acc),
        "robustness_score_R": float(R)
    }

    json.dump(report_json, open(os.path.join(outdir, "accuracy_report.json"), "w"), indent=2)

    # CSV
    with open(os.path.join(outdir, "accuracy_report.csv"), "w") as f:
        f.write("Metric,Value\n")
        for k, v in report_json.items():
            f.write(f"{k},{v}\n")

    # Plot
    plot_accuracy_drop(
        clean_acc,
        noisy_acc,
        para_acc,
        adv_acc,
        os.path.join(outdir, "robustness_barplot.png")
    )

    # Debug: save sample transformed queries
    with open(os.path.join(outdir, "debug_samples.txt"), "w") as f:
        for i in range(min(10, len(texts))):
            f.write(f"[CLEAN] {texts[i]}\n")
            f.write(f"[NOISY] {noisy_texts[i]}\n")
            f.write(f"[PARA ] {para_texts[i]}\n")
            f.write(f"[ADV  ] {adv_texts[i]}\n\n")

    print("=== Robustness Computation Finished ===")
    print(f"Robustness Score R = {R:.4f}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True, help="Path to predictions .npz")
    parser.add_argument("--texts", required=True, help="Text prompts file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    run_robustness(args.logits, args.texts, args.outdir)


if __name__ == "__main__":
    main()
