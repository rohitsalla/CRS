#!/usr/bin/env python3
"""
run_uncertainty.py

Computes the Uncertainty Pillar (U) of the Composite Reliability Score (CRS):

    U = (AUROC - 0.5) / 0.5

Uncertainty methods evaluated:
    1. MC Dropout (approximate Bayesian uncertainty)
    2. Deep Ensemble (variance across multiple checkpoints)

Outputs:
    uncertainty_results/
        - results.json
        - results.csv
        - roc_curve.png
        - uncertainty_histograms.png
        - debug_misclassified.txt

"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
# Loading Predictions
# ======================================================================

def load_logits_labels(path):
    data = np.load(path)
    logits = data["logits"]
    labels = data["labels"]
    return logits, labels


# ======================================================================
# Uncertainty Metrics
# ======================================================================

def predictive_entropy(prob):
    """Entropy of prediction over classes."""
    eps = 1e-9
    return -np.sum(prob * np.log(prob + eps), axis=1)


def predictive_variance(probs_list):
    probs_stack = np.stack(probs_list, axis=0)  # [T, N, C]
    return probs_stack.var(axis=0).mean(axis=1)  # variance per sample


def incorrect_mask(logits, labels):
    preds = logits.argmax(axis=1)
    return (preds != labels).astype(np.int32)


# ======================================================================
# AUROC Calculation
# ======================================================================

def compute_auroc(uncert_scores, labels, logits):
    """
    AUROC for error detection.
    Higher uncertainty should correspond to incorrect answers.
    """
    incorrect = incorrect_mask(logits, labels)
    fpr, tpr, _ = roc_curve(incorrect, uncert_scores)
    score = auc(fpr, tpr)
    return score, fpr, tpr


# ======================================================================
# MC Dropout Evaluation
# ======================================================================

def mc_dropout_logits(original_logits, T=10, noise_std=0.02):
    """
    Simulates MC dropout by adding Gaussian perturbations.
    Replace with real HF dropout calls for actual study.
    """
    runs = []
    for _ in range(T):
        perturbed = original_logits + np.random.normal(0, noise_std, original_logits.shape)
        runs.append(perturbed)
    return runs


def evaluate_mc_dropout(logits, labels):
    dropout_runs = mc_dropout_logits(logits)
    probs_list = [softmax(l) for l in dropout_runs]

    # Mean probability for predictive distribution
    mean_probs = np.mean(probs_list, axis=0)

    # Uncertainty = entropy
    entropy_scores = predictive_entropy(mean_probs)

    auroc, fpr, tpr = compute_auroc(entropy_scores, labels, logits)

    return {
        "method": "mc_dropout",
        "entropy_scores": entropy_scores,
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr
    }


# ======================================================================
# Ensemble Evaluation
# ======================================================================

def ensemble_logits(original_logits, num_models=3, noise_std=0.03):
    """
    Simulates ensemble predictors by adding structured variance.
    Replace with real checkpoints for production runs.
    """
    runs = []
    for i in range(num_models):
        perturbed = original_logits + np.random.normal(0, noise_std*(i+1), original_logits.shape)
        runs.append(perturbed)
    return runs


def evaluate_ensemble(logits, labels):
    ens_runs = ensemble_logits(logits)
    probs_list = [softmax(l) for l in ens_runs]

    # Uncertainty = variance of predicted class probabilities
    var_scores = predictive_variance(probs_list)

    auroc, fpr, tpr = compute_auroc(var_scores, labels, logits)

    return {
        "method": "ensemble",
        "variance_scores": var_scores,
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr
    }


# ======================================================================
# Plotting
# ======================================================================

def plot_roc(fpr_mc, tpr_mc, fpr_ens, tpr_ens, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(fpr_mc, tpr_mc, label="MC Dropout", linewidth=2)
    plt.plot(fpr_ens, tpr_ens, label="Ensemble", linewidth=2)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Error Detection ROC Curve")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_uncert_hist(mc_scores, ens_scores, outpath):
    plt.figure(figsize=(7,5))
    plt.hist(mc_scores, bins=40, alpha=0.6, label="MC Dropout")
    plt.hist(ens_scores, bins=40, alpha=0.6, label="Ensemble")
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Frequency")
    plt.title("Uncertainty Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ======================================================================
# Debug Output
# ======================================================================

def save_debug_misclassified(logits, labels, uncertainty_scores, outpath):
    preds = logits.argmax(axis=1)
    incorrect_idx = np.where(preds != labels)[0]

    with open(outpath, "w") as f:
        for i in incorrect_idx[:50]:
            f.write(f"Sample {i}\n")
            f.write(f"  Label: {labels[i]}\n")
            f.write(f"  Pred:  {preds[i]}\n")
            f.write(f"  Uncertainty: {uncertainty_scores[i]:.4f}\n\n")


# ======================================================================
# Main Pipeline
# ======================================================================

def run_uncertainty(logits_file, outdir):
    ensure_dir(outdir)
    print("\n=== Running Uncertainty Evaluation ===")

    # Load predictions
    logits, labels = load_logits_labels(logits_file)

    # Evaluate both UQ methods
    mc_results  = evaluate_mc_dropout(logits, labels)
    ens_results = evaluate_ensemble(logits, labels)

    best_method = "ensemble" if ens_results["auroc"] > mc_results["auroc"] else "mc_dropout"
    best_auroc  = max(mc_results["auroc"], ens_results["auroc"])
    U = (best_auroc - 0.5) / 0.5  # normalized CRS metric

    result = {
        "mc_dropout_auroc": float(mc_results["auroc"]),
        "ensemble_auroc": float(ens_results["auroc"]),
        "best_method": best_method,
        "best_auroc": float(best_auroc),
        "U_score": float(U)
    }

    # Save JSON + CSV
    json.dump(result, open(os.path.join(outdir, "results.json"), "w"), indent=2)
    with open(os.path.join(outdir, "results.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in result.items():
            f.write(f"{k},{v}\n")

    # Plots
    plot_roc(
        mc_results["fpr"], mc_results["tpr"],
        ens_results["fpr"], ens_results["tpr"],
        os.path.join(outdir, "roc_curve.png")
    )

    plot_uncert_hist(
        mc_results["entropy_scores"],
        ens_results["variance_scores"],
        os.path.join(outdir, "uncertainty_histograms.png")
    )

    # Debug
    combined_unc = (
        mc_results["entropy_scores"] if best_method == "mc_dropout"
        else ens_results["variance_scores"]
    )

    save_debug_misclassified(
        logits, labels, combined_unc,
        os.path.join(outdir, "debug_misclassified.txt")
    )

    print(f"Best UQ Method: {best_method}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Uncertainty Score U = {U:.4f}")
    print("=== Uncertainty Computation Finished ===")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True, help="Path to logits_and_labels.npz")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    run_uncertainty(args.logits, args.outdir)


if __name__ == "__main__":
    main()
