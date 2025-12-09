"""
calibration.py

Core utilities for evaluating and improving calibration of
classification or QA-style models using:

    • Expected Calibration Error (ECE)
    • Brier Score
    • Negative Log-Likelihood (NLL)
    • Temperature Scaling
    • Isotonic Regression

This module is model-agnostic: it works with numpy arrays of logits
or probabilities and does not depend on any specific deep learning
framework.

Typical usage:
--------------
from calibration import CalibrationEvaluator

evaluator = CalibrationEvaluator(n_bins=15)
results = evaluator.evaluate(
    y_true=y_true,
    logits=logits,
    apply_temperature=True,
    apply_isotonic=True,
)

print(results["baseline"]["ece"])
print(results["temperature"]["ece"])
print(results["isotonic"]["ece"])
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Literal

import numpy as np
from scipy.special import log_softmax
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Dataclasses for config and results
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Configuration for calibration evaluation."""
    n_bins: int = 15
    eps: float = 1e-12  # numerical stability


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    ece: float
    brier: float
    nll: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def softmax_logits(logits: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute probabilities from logits with numerical stability."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N, C), got shape {logits.shape}")
    log_probs = log_softmax(logits, axis=-1)
    probs = np.exp(log_probs)
    probs = np.clip(probs, eps, 1.0 - eps)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
    eps: float = 1e-12,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins
    in predicted confidence.

    ECE = sum_k (|B_k| / N) * |acc(B_k) - conf(B_k)|

    Args:
        y_true: (N,) integer labels.
        probs:  (N, C) predicted probabilities.
        n_bins: number of confidence bins.
        eps:    small value to avoid division by zero.

    Returns:
        scalar ECE in [0, 1].
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {probs.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D (N,), got shape {y_true.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have same number of samples")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        # bin: (a, b]
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()

        if bin_count == 0:
            continue

        bin_acc = correctness[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        weight = bin_count / float(N)

        ece += weight * abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Multi-class Brier score.

    Brier = (1/N) * sum_i sum_c (p_ic - y_ic)^2

    Where y_ic is one-hot.

    Args:
        y_true: (N,) labels.
        probs:  (N, C) probabilities.

    Returns:
        scalar Brier score.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {probs.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D (N,), got shape {y_true.shape}")
    N, C = probs.shape

    y_onehot = np.zeros((N, C), dtype=float)
    y_onehot[np.arange(N), y_true] = 1.0

    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def negative_log_likelihood(
    y_true: np.ndarray,
    logits: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """
    Compute NLL from logits or probabilities.

    Args:
        y_true:  (N,) integer labels.
        logits:  (N, C) raw logits.
        probs:   (N, C) probabilities. Exactly one of logits/probs must be provided.
        eps:     small epsilon for numerical stability.

    Returns:
        scalar NLL.
    """
    if (logits is None) == (probs is None):
        raise ValueError("Provide exactly one of logits or probs.")

    if probs is None:
        probs = softmax_logits(logits, eps=eps)

    N = len(y_true)
    p_true = probs[np.arange(N), y_true]
    p_true = np.clip(p_true, eps, 1.0)
    return float(-np.mean(np.log(p_true)))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def _nll_temperature(
    T: float,
    y_true: np.ndarray,
    logits: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """NLL as a function of temperature for optimization."""
    scaled_logits = logits / max(T, eps)
    return negative_log_likelihood(y_true, logits=scaled_logits, eps=eps)


def fit_temperature(
    y_true: np.ndarray,
    logits: np.ndarray,
    init_T: float = 1.0,
    bounds: Tuple[float, float] = (0.05, 10.0),
    eps: float = 1e-12,
) -> float:
    """
    Fit a single temperature parameter using validation logits and labels.

    Args:
        y_true: (N,) labels.
        logits: (N, C) validation logits.
        init_T: initial temperature.
        bounds: lower and upper bounds for T.
        eps:    numerical stability.

    Returns:
        optimal temperature T*.
    """
    def obj_fn(t_array: np.ndarray) -> float:
        T = float(t_array[0])
        return _nll_temperature(T, y_true, logits, eps=eps)

    res = minimize(
        obj_fn,
        x0=np.array([init_T], dtype=float),
        bounds=[bounds],
        method="L-BFGS-B",
    )

    T_opt = float(res.x[0])
    return max(T_opt, eps)


def apply_temperature(
    logits: np.ndarray,
    T: float,
) -> np.ndarray:
    """
    Apply learned temperature to logits.

    Args:
        logits: (N, C) logits.
        T:      scalar temperature.

    Returns:
        scaled logits (N, C).
    """
    return logits / T


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """
    One-dimensional isotonic regression on maximum predicted confidence.

    This treats the model's predicted max-probability as the input and
    the correctness (0/1) as the target, learning a monotonic mapping
    from confidence to empirical accuracy.
    """

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, y_true: np.ndarray, probs: np.ndarray) -> None:
        """
        Fit isotonic regression using validation data.

        Args:
            y_true: (N,) labels.
            probs:  (N, C) probabilities.
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correctness = (predictions == y_true).astype(float)

        self._iso.fit(confidences, correctness)
        self._fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities by rescaling their maximum confidence
        according to the learned isotonic mapping. We only adjust the
        predicted class confidence and re-normalize.

        Args:
            probs: (N, C) pre-calibrated probabilities.

        Returns:
            calibrated_probs: (N, C)
        """
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fit() before transform().")

        N, C = probs.shape
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)

        # Evaluate isotonic mapping on the current confidences
        calibrated_conf = self._iso.predict(confidences)

        # Create a copy and update only the predicted class confidence
        calibrated_probs = probs.copy()
        for i in range(N):
            pred = predictions[i]
            original_conf = probs[i, pred]
            if original_conf <= 0.0:
                continue
            scale = calibrated_conf[i] / original_conf
            calibrated_probs[i] *= scale
            # Renormalize
            s = calibrated_probs[i].sum()
            if s > 0:
                calibrated_probs[i] /= s

        return calibrated_probs


# ---------------------------------------------------------------------------
# Calibration Evaluator
# ---------------------------------------------------------------------------

class CalibrationEvaluator:
    """
    High-level calibration evaluator.

    Provides:
        • Baseline metrics from logits or probabilities
        • Temperature scaling (global temperature)
        • Isotonic regression on predicted confidence
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.iso_calibrator: Optional[IsotonicCalibrator] = None
        self.temperature: Optional[float] = None

    # ------------------- low-level wrappers -------------------

    def _compute_metrics_from_probs(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> CalibrationMetrics:
        ece = expected_calibration_error(
            y_true=y_true,
            probs=probs,
            n_bins=self.config.n_bins,
            eps=self.config.eps,
        )
        brier = brier_score(y_true=y_true, probs=probs)

        if logits is not None:
            nll = negative_log_likelihood(
                y_true=y_true,
                logits=logits,
                eps=self.config.eps,
            )
        else:
            nll = negative_log_likelihood(
                y_true=y_true,
                probs=probs,
                eps=self.config.eps,
            )

        return CalibrationMetrics(ece=ece, brier=brier, nll=nll)

    # ------------------- public API -------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        apply_temperature: bool = True,
        apply_isotonic: bool = True,
        val_split: Literal["same", "half"] = "half",
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate calibration and (optionally) fit and apply calibration methods.

        Args:
            y_true: (N,) labels.
            logits: (N, C) logits. Provide either logits or probs.
            probs:  (N, C) probabilities.
            apply_temperature: whether to fit and evaluate temperature scaling.
            apply_isotonic:    whether to fit and evaluate isotonic regression.
            val_split:         how to choose validation subset for calibration.
                               "half" → first half for calibration, second half for eval.
                               "same" → use all examples for both (optimistic).

        Returns:
            dict with keys "baseline", "temperature", "isotonic" (if enabled),
            each mapping to a dict of metrics { "ece", "brier", "nll" }.
        """
        if (logits is None) == (probs is None):
            raise ValueError("Provide exactly one of logits or probs.")

        y_true = np.asarray(y_true, dtype=int)
        N = len(y_true)

        if logits is not None:
            logits = np.asarray(logits, dtype=float)
            if logits.shape[0] != N:
                raise ValueError("logits and y_true mismatch in number of samples.")
            base_probs = softmax_logits(logits, eps=self.config.eps)
        else:
            probs = np.asarray(probs, dtype=float)
            if probs.shape[0] != N:
                raise ValueError("probs and y_true mismatch in number of samples.")
            base_probs = probs
            logits = None  # we may not have logits, NLL computed from probs

        results: Dict[str, Dict[str, float]] = {}

        # Baseline
        baseline_metrics = self._compute_metrics_from_probs(
            y_true=y_true,
            probs=base_probs,
            logits=logits,
        )
        results["baseline"] = baseline_metrics.to_dict()

        # Split data for calibration if needed
        if val_split == "half" and N >= 4:
            mid = N // 2
            idx_val = np.arange(0, mid)
            idx_eval = np.arange(mid, N)
        else:
            # Use all data for both fitting and evaluation
            idx_val = np.arange(N)
            idx_eval = np.arange(N)

        # Temperature scaling
        if apply_temperature and logits is not None:
            T = fit_temperature(
                y_true=y_true[idx_val],
                logits=logits[idx_val],
                eps=self.config.eps,
            )
            self.temperature = T
            scaled_logits = apply_temperature(logits[idx_eval], T)
            scaled_probs = softmax_logits(scaled_logits, eps=self.config.eps)

            temp_metrics = self._compute_metrics_from_probs(
                y_true=y_true[idx_eval],
                probs=scaled_probs,
                logits=scaled_logits,
            )
            results["temperature"] = temp_metrics.to_dict()

        # Isotonic regression
        if apply_isotonic:
            self.iso_calibrator = IsotonicCalibrator()
            self.iso_calibrator.fit(
                y_true=y_true[idx_val],
                probs=base_probs[idx_val],
            )
            iso_probs = self.iso_calibrator.transform(base_probs[idx_eval])

            iso_metrics = self._compute_metrics_from_probs(
                y_true=y_true[idx_eval],
                probs=iso_probs,
                logits=None if logits is None else logits[idx_eval],
            )
            results["isotonic"] = iso_metrics.to_dict()

        return results
