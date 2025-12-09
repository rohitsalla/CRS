"""
CRS Evaluation Toolkit
======================

A unified framework for computing the Composite Reliability Score (CRS)
for Large Language Models (LLMs). CRS integrates three reliability pillars:

    • Calibration (C)
    • Robustness (R)
    • Uncertainty Quantification (U)

This package contains modular evaluators for each pillar as well as a
pipeline wrapper for full CRS computation.

Directory Structure
-------------------
crs/
    __init__.py
    calibration/
        run_calibration.py
        metrics.py
    robustness/
        run_robustness.py
        perturbations.py
    uncertainty/
        run_uncertainty.py
        metrics.py
    pipeline/
        run_crs_pipeline.py

Public API
----------
Users may import the following classes:

    from crs import (
        CalibrationEvaluator,
        RobustnessEvaluator,
        UncertaintyEvaluator,
        CRSPipeline,
    )

Author: Rohit Kumar Salla (Virginia Tech)
Paper: Beyond Hallucinations: A Composite Score for Measuring Reliability
"""

# Version metadata
__version__ = "1.0.0"
__author__ = "Rohit Kumar Salla"
__email__ = "rohits25@vt.edu"
__license__ = "MIT"


# ---------------------------------------------------------------------
# Import exposed classes from submodules
# ---------------------------------------------------------------------

# Calibration
try:
    from .calibration.run_calibration import CalibrationEvaluator
except Exception:  # allows importing even if submodule missing during setup
    CalibrationEvaluator = None

# Robustness
try:
    from .robustness.run_robustness import RobustnessEvaluator
except Exception:
    RobustnessEvaluator = None

# Uncertainty
try:
    from .uncertainty.run_uncertainty import UncertaintyEvaluator
except Exception:
    UncertaintyEvaluator = None

# CRS Pipeline
try:
    from .pipeline.run_crs_pipeline import CRSPipeline
except Exception:
    CRSPipeline = None


# ---------------------------------------------------------------------
# Public symbols exposed by the package
# ---------------------------------------------------------------------
__all__ = [
    "__version__",
    "CalibrationEvaluator",
    "RobustnessEvaluator",
    "UncertaintyEvaluator",
    "CRSPipeline",
]
