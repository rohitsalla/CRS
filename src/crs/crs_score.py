"""
crs_score.py

Implements the Composite Reliability Score (CRS) for evaluating
Large Language Model reliability across three pillars:

    • Calibration (C)
    • Robustness (R)
    • Uncertainty (U)

CRS = α*C + β*R + γ*U

This module contains:
    - Metric normalization
    - CRS score computation
    - Reliability tier assignment
    - Optional weight sensitivity analysis
    - Optional leave-one-out dataset stability analysis

Designed to be fully compatible with:
    run_crs_pipeline.py
    run_calibration.py
    run_robustness.py
    run_uncertainty.py
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PillarScores:
    """
    Holds raw (unnormalized) metrics for a single model.

    calibration: lower is better (ECE, Brier, NLL combined)
    robustness: higher is better
    uncertainty: higher is better
    """
    calibration: float
    robustness: float
    uncertainty: float

@dataclass
class NormalizedScores:
    """Normalized (0–1) pillar scores."""
    C: float
    R: float
    U: float

@dataclass
class CRSResult:
    """Final CRS score and reliability category."""
    model_name: str
    C: float
    R: float
    U: float
    crs: float
    tier: str

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# CRS Calculator
# ---------------------------------------------------------------------------

class CRSCalculator:
    """
    Compute the Composite Reliability Score (CRS) from raw metric
    outputs collected from Calibration, Robustness, and Uncertainty pipelines.
    """

    def __init__(
        self,
        weight_calib: float = 1/3,
        weight_robust: float = 1/3,
        weight_uncert: float = 1/3,
        normalization: str = "minmax",
    ):
        """
        Args:
            weight_calib: α
            weight_robust: β
            weight_uncert: γ
            normalization: {"minmax", "zscore", "percentile"}
        """
        self.weights = (weight_calib, weight_robust, weight_uncert)
        self.normalization = normalization

    # ----------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------

    @staticmethod
    def _invert(value: float) -> float:
        """For calibration, lower is better → invert for normalization."""
        return -value

    @staticmethod
    def _minmax(values: List[float]) -> List[float]:
        vals = np.array(values, dtype=float)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return [0.5] * len(vals)
        return list((vals - lo) / (hi - lo))

    @staticmethod
    def _zscore(values: List[float]) -> List[float]:
        vals = np.array(values, dtype=float)
        mean, std = vals.mean(), vals.std()
        if std == 0:
            return [0.0] * len(vals)
        z = (vals - mean) / std
        # Normalize to 0–1
        z = (z - z.min()) / (z.max() - z.min())
        return list(z)

    @staticmethod
    def _percentile(values: List[float]) -> List[float]:
        vals = np.array(values, dtype=float)
        ranks = np.argsort(np.argsort(vals))
        percentiles = ranks / (len(vals) - 1)
        return list(percentiles)

    # ----------------------------------------------------------------------
    # Normalization wrapper
    # ----------------------------------------------------------------------

    def _normalize(self, values: List[float]) -> List[float]:
        if self.normalization == "minmax":
            return self._minmax(values)
        if self.normalization == "zscore":
            return self._zscore(values)
        if self.normalization == "percentile":
            return self._percentile(values)
        raise ValueError(f"Unknown normalization method: {self.normalization}")

    # ----------------------------------------------------------------------
    # Main CRS computation
    # ----------------------------------------------------------------------

    def compute_crs(
        self,
        model_metrics: Dict[str, PillarScores],
    ) -> List[CRSResult]:
        """
        Compute CRS for all models.

        Args:
            model_metrics:
                {
                  "model_name": PillarScores(calibration, robustness, uncertainty),
                  ...
                }

        Returns:
            List[CRSResult]
        """
        model_names = list(model_metrics.keys())

        # Extract raw lists
        raw_C = [self._invert(model_metrics[m].calibration) for m in model_names]
        raw_R = [model_metrics[m].robustness for m in model_names]
        raw_U = [model_metrics[m].uncertainty for m in model_names]

        # Normalize each pillar
        C_norm = self._normalize(raw_C)
        R_norm = self._normalize(raw_R)
        U_norm = self._normalize(raw_U)

        α, β, γ = self.weights

        results = []
        for i, model_name in enumerate(model_names):
            C, R, U = C_norm[i], R_norm[i], U_norm[i]
            crs = α * C + β * R + γ * U
            tier = self._assign_tier(crs)

            results.append(
                CRSResult(
                    model_name=model_name,
                    C=C, R=R, U=U,
                    crs=crs,
                    tier=tier,
                )
            )

        # Sort by CRS descending
        results.sort(key=lambda x: x.crs, reverse=True)
        return results

    # ----------------------------------------------------------------------
    # Reliability Tier Mapping
    # ----------------------------------------------------------------------

    @staticmethod
    def _assign_tier(crs: float) -> str:
        if crs >= 0.80:
            return "Highly Reliable"
        if crs >= 0.65:
            return "Moderately Reliable"
        return "Unreliable"

    # ----------------------------------------------------------------------
    # Optional: Weight Sensitivity
    # ----------------------------------------------------------------------

    def weight_sensitivity(
        self,
        metrics: Dict[str, PillarScores],
        weight_sets: List[Tuple[float, float, float]],
    ) -> Dict[str, List[float]]:
        """
        For reviewer: checks if rankings remain stable under multiple weight choices.
        Returns dict mapping model → list of CRS scores across weight sets.
        """
        stability = {m: [] for m in metrics}

        for (a, b, c) in weight_sets:
            self.weights = (a, b, c)
            crs_results = self.compute_crs(metrics)
            for r in crs_results:
                stability[r.model_name].append(r.crs)

        return stability

    # ----------------------------------------------------------------------
    # Optional: Leave-One-Out Dataset Stability
    # ----------------------------------------------------------------------

    def dataset_stability(
        self,
        per_dataset_metrics: Dict[str, Dict[str, PillarScores]],
    ) -> Dict[str, float]:
        """
        Compute average deviation in CRS when dropping each dataset.
        Used to show CRS does not overfit a single dataset.

        per_dataset_metrics:
            {
                "dataset1": {"model1": PillarScores(...), "model2": ...},
                "dataset2": {...},
                ...
            }

        Returns:
            dict: {model_name: avg deviation across leave-one-out runs}
        """
        datasets = list(per_dataset_metrics.keys())
        all_models = list(next(iter(per_dataset_metrics.values())).keys())

        # Compute full CRS with all datasets
        full_scores = self._average_dataset_scores(per_dataset_metrics)
        full_crs = {r.model_name: r.crs for r in self.compute_crs(full_scores)}

        deviations = {m: [] for m in all_models}

        # Leave-one-out
        for ds in datasets:
            subset = {
                d: per_dataset_metrics[d]
                for d in datasets if d != ds
            }
            avg_scores = self._average_dataset_scores(subset)
            crs_subset = {
                r.model_name: r.crs
                for r in self.compute_crs(avg_scores)
            }
            for m in all_models:
                deviations[m].append(abs(full_crs[m] - crs_subset[m]))

        # Average deviation per model
        return {m: float(np.mean(devs)) for m, devs in deviations.items()}

    @staticmethod
    def _average_dataset_scores(
        per_dataset: Dict[str, Dict[str, PillarScores]]
    ) -> Dict[str, PillarScores]:
        """
        Average calibration, robustness, and uncertainty scores across datasets.
        """
        models = list(next(iter(per_dataset.values())).keys())
        out = {}

        for m in models:
            C_list, R_list, U_list = [], [], []
            for ds in per_dataset:
                item = per_dataset[ds][m]
                C_list.append(item.calibration)
                R_list.append(item.robustness)
                U_list.append(item.uncertainty)
            out[m] = PillarScores(
                calibration=float(np.mean(C_list)),
                robustness=float(np.mean(R_list)),
                uncertainty=float(np.mean(U_list)),
            )
        return out
