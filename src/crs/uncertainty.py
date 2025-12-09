"""
uncertainty.py

Provides predictive uncertainty estimation using:
    • Monte-Carlo Dropout
    • Deep Ensembles
    • AUROC for error detection

Outputs a normalized UQ score:
      U = (AUROC - 0.5) / 0.5

Compatible with CRS pipeline.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# =====================================================================================
# Utility Classes
# =====================================================================================

@dataclass
class UQResult:
    """Stores AUROC and raw uncertainty scores."""
    auroc: float
    uncertainties: List[float]
    correct_labels: List[int]   # 1 = correct, 0 = incorrect


# =====================================================================================
# Model Wrapper for Answer + Probabilities
# =====================================================================================

class UQModelWrapper:
    """
    Makes causal-LM QA predictions and extracts:
        - predicted answer (string)
        - confidence score (softmax probability)
    """

    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" else torch.float32,
        ).to(self.device)

        self.model.eval()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        self.eos = self.tokenizer.eos_token or "</s>"

    def answer_with_logits(
        self,
        question: str,
        max_length: int = 64,
    ) -> Tuple[str, torch.Tensor]:

        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=inputs["input_ids"].shape[1] + max_length,
            )

        # decode answer string
        generated = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        answer = generated.split("Answer:")[-1].strip()

        # final token logits
        final_logits = output.scores[-1][0]  # (vocab_size)

        return answer, final_logits

    @staticmethod
    def exact_match(pred: str, gold: str) -> bool:
        return pred.lower().strip() == gold.lower().strip()


# =====================================================================================
# Monte Carlo Dropout Uncertainty
# =====================================================================================

class MonteCarloDropout:
    """Applies dropout at inference and measures prediction variance."""

    def __init__(self, wrapper: UQModelWrapper, passes: int = 10):
        self.wrapper = wrapper
        self.passes = passes

        # Enable dropout
        self._enable_dropout(self.wrapper.model)

    def _enable_dropout(self, module):
        """Recursively enables dropout layers for MC sampling."""
        if isinstance(module, torch.nn.Dropout):
            module.train()
        for child in module.children():
            self._enable_dropout(child)

    def measure_uncertainty(self, question: str) -> float:
        """
        Returns uncertainty = variance of softmax probabilities (scalar)
        """
        probs = []

        for _ in range(self.passes):
            _, logits = self.wrapper.answer_with_logits(question)
            p = torch.softmax(logits, dim=-1)
            probs.append(p.unsqueeze(0))

        probs = torch.cat(probs, dim=0)  # (k, vocab)
        var = torch.var(probs, dim=0).mean().item()

        return var


# =====================================================================================
# Ensemble Uncertainty
# =====================================================================================

class DeepEnsembleUQ:
    """Uses multiple model checkpoints to estimate prediction variance."""

    def __init__(self, model_names: List[str], device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = []
        for name in model_names:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.float16 if "cuda" else torch.float32,
            ).to(self.device)
            mdl.eval()
            self.models.append((tok, mdl))

    def measure_uncertainty(self, question: str) -> float:
        logits_list = []

        for tok, mdl in self.models:
            prompt = f"Question: {question}\nAnswer:"
            inputs = tok(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = mdl.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_length=inputs["input_ids"].shape[1] + 64,
                )
            logits = out.scores[-1][0]
            logits_list.append(logits.unsqueeze(0))

        logits_stack = torch.cat(logits_list, dim=0)
        var = torch.var(logits_stack, dim=0).mean().item()

        return var


# =====================================================================================
# AUROC Computation
# =====================================================================================

class UQEvaluator:
    """
    Computes:
        • MC Dropout AUROC
        • Ensemble AUROC
        • Normalized UQ score
    """

    def __init__(self, model_name: str, ensemble_models: Optional[List[str]] = None):
        self.wrapper = UQModelWrapper(model_name)

        self.mc_dropout = MonteCarloDropout(self.wrapper, passes=10)
        self.ensemble = None

        if ensemble_models:
            self.ensemble = DeepEnsembleUQ(ensemble_models)

    # --------------------------------------------------------------------------
    # Evaluate uncertainty on dataset
    # --------------------------------------------------------------------------

    def evaluate(
        self,
        dataset: List[Dict],
        method: str = "mc",
        max_samples: Optional[int] = None,
    ) -> UQResult:

        if max_samples:
            dataset = dataset[:max_samples]

        uncertainties = []
        labels = []

        for item in dataset:
            q = item["question"]
            gold = item["answer"]

            pred, _ = self.wrapper.answer_with_logits(q)
            correct = int(self.wrapper.exact_match(pred, gold))

            # pick method
            if method == "mc":
                u = self.mc_dropout.measure_uncertainty(q)
            elif method == "ensemble":
                if self.ensemble is None:
                    raise ValueError("No ensemble models provided.")
                u = self.ensemble.measure_uncertainty(q)
            else:
                raise ValueError("Invalid UQ method.")

            uncertainties.append(u)
            labels.append(correct)

        # Higher uncertainty should correlate with incorrect predictions
        auroc = roc_auc_score(labels, uncertainties)

        return UQResult(
            auroc=auroc,
            uncertainties=uncertainties,
            correct_labels=labels,
        )

    # --------------------------------------------------------------------------
    # Returns normalized score used by CRS
    # --------------------------------------------------------------------------

    @staticmethod
    def normalize_auroc(auroc: float) -> float:
        """
        Maps AUROC → [0,1]
        AUROC=0.5 → 0
        AUROC=1.0 → 1
        """
        score = (auroc - 0.5) / 0.5
        return max(0.0, min(1.0, score))

    # --------------------------------------------------------------------------
    # Compare MC vs Ensemble, return best
    # --------------------------------------------------------------------------

    def evaluate_best(self, dataset: List[Dict], max_samples: int = None) -> Tuple[str, float]:
        """
        Returns (best_method, normalized_score)
        """
        result_mc = self.evaluate(dataset, method="mc", max_samples=max_samples)
        best = ("mc", self.normalize_auroc(result_mc.auroc))

        if self.ensemble:
            result_en = self.evaluate(dataset, method="ensemble", max_samples=max_samples)
            score_en = self.normalize_auroc(result_en.auroc)

            if score_en > best[1]:
                best = ("ensemble", score_en)

        return best
