"""
run_calibration.py

Compute calibration metrics (ECE, Brier, NLL) for model outputs and apply
post-hoc calibration methods (Temperature Scaling, Isotonic Regression).

Expected input: a NumPy .npz file or CSV file containing:
    - logits: shape (N, C)  (preferred)
    - labels: shape (N,)    integer class indices in [0, C-1]

For CSV, we assume:
    - a column called "label"
    - columns called "logit_0", "logit_1", ..., "logit_{C-1}"

The script will:
    1. Load predictions.
    2. Compute baseline metrics.
    3. Optionally fit Temperature Scaling and Isotonic Regression.
    4. Recompute metrics after calibration.
    5. Optionally plot reliability diagrams.
    6. Dump a summary CSV for all methods.

Usage example:
    python run_calibration.py \
        --input predictions.npz \
        --out-dir results/calibration \
        --num-bins 15 \
        --methods none temperature isotonic \
        --plot

"""

import argparse
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax."""
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable log-softmax."""
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits)
    log_sum = np.log(np.sum(exps, axis=axis, keepdims=True))
    return logits - log_sum


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load logits and labels from a .npz file."""
    data = np.load(path)
    if "logits" not in data or "labels" not in data:
        raise ValueError(
            f"NPZ file {path} must contain 'logits' and 'labels' arrays."
        )
    logits = data["logits"]
    labels = data["labels"]
    return logits, labels


def _load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load logits and labels from a CSV file."""
    if pd is None:
        raise ImportError(
            "pandas is required to load CSV files. "
            "Install via `pip install pandas`."
        )
    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    label_col = df["label"].to_numpy()
    # infer logit columns
    logit_cols = [c for c in df.columns if c.startswith("logit_")]
    if not logit_cols:
        raise ValueError(
            "CSV must contain columns named 'logit_0', 'logit_1', ... etc."
        )
    logit_cols = sorted(logit_cols, key=lambda x: int(x.split("_")[1]))
    logits = df[logit_cols].to_numpy(dtype=np.float32)
    return logits, label_col


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load logits and labels from NPZ or CSV.

    Returns:
        logits: (N, C)
        labels: (N,)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return _load_npz(path)
    elif ext == ".csv":
        return _load_csv(path)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Use .npz or .csv."
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute mean Negative Log-Likelihood (NLL) for a batch.

    Args:
        logits: (N, C)
        labels: (N,)
    """
    log_probs = log_softmax(logits, axis=-1)
    n = logits.shape[0]
    # log_probs[range(n), labels] shape (N,)
    example_log_probs = log_probs[np.arange(n), labels]
    nll = - example_log_probs.mean()
    return float(nll)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multi-class Brier score (mean squared error in probability space).

    Args:
        probs: (N, C) probabilities
        labels: (N,)
    """
    n, c = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    bs = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return float(bs)


@dataclass
class ECEStats:
    ece: float
    bin_acc: np.ndarray
    bin_conf: np.ndarray
    bin_count: np.ndarray


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> ECEStats:
    """
    Compute Expected Calibration Error (ECE) and per-bin statistics.

    Args:
        probs: (N, C) probabilities
        labels: (N,)
        num_bins: number of bins in [0, 1]

    Returns:
        ECEStats with scalar ece and per-bin arrays.
    """
    n = probs.shape[0]
    # max probability and predicted class
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correctness = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    # Clip indices to valid range
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_acc = np.zeros(num_bins, dtype=np.float32)
    bin_conf = np.zeros(num_bins, dtype=np.float32)
    bin_count = np.zeros(num_bins, dtype=np.int64)

    for b in range(num_bins):
        mask = bin_indices == b
        count = np.sum(mask)
        bin_count[b] = count
        if count > 0:
            bin_acc[b] = correctness[mask].mean()
            bin_conf[b] = confidences[mask].mean()
        else:
            bin_acc[b] = 0.0
            bin_conf[b] = 0.0

    ece = 0.0
    for b in range(num_bins):
        if bin_count[b] > 0:
            ece += (bin_count[b] / n) * abs(bin_acc[b] - bin_conf[b])

    return ECEStats(ece=float(ece),
                    bin_acc=bin_acc,
                    bin_conf=bin_conf,
                    bin_count=bin_count)


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """
    Simple scalar Temperature Scaling:
        logits' = logits / T

    T is chosen to minimize NLL on a calibration set.
    We use a basic grid search plus local refinement to avoid SciPy dependency.
    """

    def __init__(self, init_temp: float = 1.0):
        self.temperature = init_temp

    def _nll_for_temp(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temp: float
    ) -> float:
        scaled_logits = logits / temp
        return negative_log_likelihood(scaled_logits, labels)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Fit temperature using a simple search.

        Strategy:
            1. Coarse grid over [0.1, 5.0].
            2. Take best temp, refine in local neighborhood.

        Returns:
            best_T (float)
        """
        # coarse grid
        candidates = np.linspace(0.1, 5.0, 30)
        best_T = 1.0
        best_nll = float("inf")
        for T in candidates:
            nll = self._nll_for_temp(logits, labels, T)
            if nll < best_nll:
                best_nll = nll
                best_T = T

        # local refinement: small search around best_T
        small_steps = np.linspace(-0.5, 0.5, 21)
        for delta in small_steps:
            T = best_T + delta
            if T <= 0.02:
                continue
            nll = self._nll_for_temp(logits, labels, T)
            if nll < best_nll:
                best_nll = nll
                best_T = T

        self.temperature = best_T
        return best_T

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature and return scaled logits."""
        return logits / self.temperature

    def to_dict(self) -> Dict:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, d: Dict) -> "TemperatureScaler":
        return cls(init_temp=float(d["temperature"]))


class IsotonicCalibrator:
    """
    Multi-class isotonic regression by calibrating top-1 confidence.

    We map raw max-confidences to calibrated confidences and
    re-normalize class probabilities by:

        p_calibrated(y_hat) = f(confidence_raw)
        p_calibrated(other) = p_raw(other) * (1 - f(conf_raw)) / (1 - conf_raw)

    This is a simple heuristic but works reasonably well in practice.
    Requires scikit-learn.
    """

    def __init__(self):
        if IsotonicRegression is None:
            raise ImportError(
                "IsotonicRegression requires scikit-learn. "
                "Install via `pip install scikit-learn`."
            )
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit isotonic regression on max confidence vs correctness.

        Args:
            probs: (N, C) probabilities
            labels: (N,)
        """
        conf = probs.max(axis=-1)
        preds = probs.argmax(axis=-1)
        correctness = (preds == labels).astype(np.float32)
        # Fit mapping from conf in [0,1] to empirical accuracy in [0,1]
        self._iso.fit(conf, correctness)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression to probabilities.

        Args:
            probs: (N, C)
        Returns:
            calibrated_probs: (N, C)
        """
        conf_raw = probs.max(axis=-1)
        preds = probs.argmax(axis=-1)
        conf_cal = self._iso.predict(conf_raw)

        # Avoid division by zero
        eps = 1e-8
        conf_raw_clipped = np.clip(conf_raw, eps, 1.0 - eps)

        # scaling factor for non-top classes
        scale_other = (1.0 - conf_cal) / (1.0 - conf_raw_clipped)

        calibrated = probs.copy()
        n, c = probs.shape
        for i in range(n):
            top = preds[i]
            calibrated[i, :] *= scale_other[i]
            calibrated[i, top] = conf_cal[i]
        # Clip and renormalize to fix numerical issues
        calibrated = np.clip(calibrated, 0.0, 1.0)
        calibrated /= calibrated.sum(axis=-1, keepdims=True)
        return calibrated

    def to_dict(self) -> Dict:
        # sklearn isotonic regression stores thresholds internally as attributes
        # but serialization is messy; easiest is to save via joblib or pickle.
        # Here, we just serialize using pickle to a separate file if needed.
        raise NotImplementedError("Use joblib/pickle to save isotonic model.")


# ---------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    ece_stats: ECEStats,
    filename: str,
    title: str,
) -> None:
    """
    Plot a standard reliability diagram.

    Args:
        ece_stats: ECEStats with per-bin info.
        filename: output path (PNG).
        title: plot title.
    """
    bin_acc = ece_stats.bin_acc
    bin_conf = ece_stats.bin_conf
    bin_count = ece_stats.bin_count

    num_bins = len(bin_acc)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

    # Only plot bins with samples
    mask = bin_count > 0
    plt.bar(
        bin_centers[mask],
        bin_acc[mask],
        width=1.0 / num_bins,
        edgecolor="black",
        alpha=0.7,
        label="Empirical"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title + f" (ECE={ece_stats.ece:.3f})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------

@dataclass
class MethodMetrics:
    method: str
    ece: float
    brier: float
    nll: float


def evaluate_method(
    method: str,
    logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
    out_dir: Optional[str],
    plot: bool,
) -> MethodMetrics:
    """
    Evaluate one calibration method and optionally save reliability diagram.

    Args:
        method: "none", "temperature", or "isotonic"
        logits: (N, C)
        labels: (N,)
        num_bins: ECE bins
        out_dir: directory to save plots/params
        plot: whether to plot reliability diagram

    Returns:
        MethodMetrics with metrics for this method.
    """
    if method not in {"none", "temperature", "isotonic"}:
        raise ValueError(f"Unknown method: {method}")

    method_dir = None
    if out_dir is not None:
        method_dir = os.path.join(out_dir, method)
        ensure_dir(method_dir)

    # --- Calibration transform ---
    if method == "none":
        calibrated_logits = logits
        probs = softmax(calibrated_logits, axis=-1)

    elif method == "temperature":
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        calibrated_logits = scaler.transform(logits)
        probs = softmax(calibrated_logits, axis=-1)

        # Save temperature parameter
        if method_dir is not None:
            params_path = os.path.join(method_dir, "temperature.json")
            with open(params_path, "w") as f:
                json.dump(scaler.to_dict(), f, indent=2)

    else:  # isotonic
        # For isotonic we work directly in probability space
        raw_probs = softmax(logits, axis=-1)
        calibrator = IsotonicCalibrator()
        calibrator.fit(raw_probs, labels)
        probs = calibrator.transform(raw_probs)
        # We can derive calibrated logits by inverse-softmax if needed,
        # but for metrics we only need probabilities.
        calibrated_logits = None

    # --- Metrics ---
    if calibrated_logits is not None:
        nll = negative_log_likelihood(calibrated_logits, labels)
    else:
        # approximate NLL via log of probs (numerically safe)
        eps = 1e-12
        n = probs.shape[0]
        log_probs = np.log(np.clip(probs, eps, 1.0))
        nll = float(-log_probs[np.arange(n), labels].mean())

    bs = brier_score(probs, labels)
    ece_stats = expected_calibration_error(probs, labels, num_bins=num_bins)
    ece = ece_stats.ece

    # --- Plot reliability diagram ---
    if plot and method_dir is not None:
        plot_path = os.path.join(method_dir, "reliability_diagram.png")
        title = f"{method.capitalize()} Calibration"
        plot_reliability_diagram(ece_stats, plot_path, title)

    return MethodMetrics(
        method=method,
        ece=ece,
        brier=bs,
        nll=nll,
    )


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibration and compute reliability metrics."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to NPZ or CSV file with logits and labels.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save calibration results (plots, params, summary).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of bins for ECE.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["none", "temperature", "isotonic"],
        help="Calibration methods to run: none temperature isotonic",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, save reliability diagram plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading predictions from {args.input}")
    logits, labels = load_predictions(args.input)
    n, c = logits.shape
    print(f"[INFO] Loaded logits: shape={logits.shape}, labels shape={labels.shape}")

    if args.out_dir is not None:
        ensure_dir(args.out_dir)

    # Evaluate each requested method
    all_metrics: List[MethodMetrics] = []
    for m in args.methods:
        print(f"[INFO] Evaluating method: {m}")
        metrics = evaluate_method(
            method=m,
            logits=logits,
            labels=labels,
            num_bins=args.num_bins,
            out_dir=args.out_dir,
            plot=args.plot,
        )
        all_metrics.append(metrics)
        print(
            f"  -> ECE={metrics.ece:.4f}, "
            f"Brier={metrics.brier:.4f}, "
            f"NLL={metrics.nll:.4f}"
        )

    # Save summary CSV + JSON
    if args.out_dir is not None:
        summary_csv = os.path.join(args.out_dir, "calibration_summary.csv")
        summary_json = os.path.join(args.out_dir, "calibration_summary.json")

        # CSV
        header = "method,ece,brier,nll\n"
        with open(summary_csv, "w") as f:
            f.write(header)
            for m in all_metrics:
                f.write(f"{m.method},{m.ece:.6f},{m.brier:.6f},{m.nll:.6f}\n")

        # JSON
        with open(summary_json, "w") as f:
            json.dump(
                [asdict(m) for m in all_metrics],
                f,
                indent=2,
            )

        print(f"[INFO] Saved summary to {summary_csv} and {summary_json}")


if __name__ == "__main__":
    main()
