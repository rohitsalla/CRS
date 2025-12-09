"""
robustness.py

Provides perturbation methods and robustness evaluation tools
for the CRS (Composite Reliability Score) pipeline.

Implements:
    • Noise perturbation (character swaps, insertions, deletions)
    • Paraphrasing (back-translation)
    • Adversarial synonym attacks (TextFooler-style)
    • Robustness evaluation against clean accuracy
    • Aggregation of per-dataset robustness results

Outputs:
    robustness_score = 1 - (avg_drop / avg_clean_accuracy)
"""

from __future__ import annotations

import random
import string
from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import nltk
from nltk.corpus import wordnet as wn


# ------------------------------------------------------------------------------------
# Download required NLTK resources
# ------------------------------------------------------------------------------------
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ------------------------------------------------------------------------------------
# Data Structure for Robustness Output
# ------------------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    clean_accuracy: float
    perturbed_accuracy: float
    accuracy_drop: float


# ------------------------------------------------------------------------------------
# Perturbation Suite
# ------------------------------------------------------------------------------------

class PerturbationGenerator:
    """Generates noisy, paraphrased, and adversarial perturbations."""

    def __init__(
        self,
        paraphrase_model: str = "Helsinki-NLP/opus-mt-en-fr",
        reverse_paraphrase_model: str = "Helsinki-NLP/opus-mt-fr-en",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        # Load translation models for back-translation
        self.tokenizer_fr = AutoTokenizer.from_pretrained(paraphrase_model)
        self.model_fr = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model).to(device)

        self.tokenizer_en = AutoTokenizer.from_pretrained(reverse_paraphrase_model)
        self.model_en = AutoModelForSeq2SeqLM.from_pretrained(reverse_paraphrase_model).to(device)

    # --------------------------------------------------------------------------
    # Noise Perturbations (Typo Simulation)
    # --------------------------------------------------------------------------

    def _swap_chars(self, word: str) -> str:
        if len(word) < 2:
            return word
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]

    def _delete_char(self, word: str) -> str:
        if len(word) < 2:
            return word
        i = random.randint(0, len(word) - 1)
        return word[:i] + word[i+1:]

    def _insert_char(self, word: str) -> str:
        i = random.randint(0, len(word))
        c = random.choice(string.ascii_lowercase)
        return word[:i] + c + word[i:]

    def apply_noise(self, text: str, prob: float = 0.15) -> str:
        words = text.split()
        new_words = []

        for w in words:
            if random.random() > prob:
                new_words.append(w)
                continue

            op = random.choice(["swap", "delete", "insert"])
            if op == "swap":
                new_words.append(self._swap_chars(w))
            elif op == "delete":
                new_words.append(self._delete_char(w))
            else:
                new_words.append(self._insert_char(w))

        return " ".join(new_words)

    # --------------------------------------------------------------------------
    # Back-Translation for Paraphrasing
    # --------------------------------------------------------------------------

    def paraphrase(self, text: str, max_length: int = 128) -> str:
        """English → French → English paraphrasing."""
        # EN → FR
        inputs = self.tokenizer_fr(text, return_tensors="pt", truncation=True).to(self.device)
        intermediate = self.model_fr.generate(**inputs, max_length=max_length)
        fr_text = self.tokenizer_fr.decode(intermediate[0], skip_special_tokens=True)

        # FR → EN
        inputs = self.tokenizer_en(fr_text, return_tensors="pt", truncation=True).to(self.device)
        final = self.model_en.generate(**inputs, max_length=max_length)
        return self.tokenizer_en.decode(final[0], skip_special_tokens=True)

    # --------------------------------------------------------------------------
    # Adversarial Synonym Replacement
    # --------------------------------------------------------------------------

    def _get_synonym(self, word: str) -> str:
        syns = wn.synsets(word)
        if not syns:
            return word
        lemmas = syns[0].lemma_names()
        if not lemmas:
            return word
        s = random.choice(lemmas)
        # Avoid underscore tokens
        return s.replace("_", " ")

    def adversarial_attack(self, text: str, prob: float = 0.12) -> str:
        """TextFooler-style synonym substitution."""
        tokens = nltk.word_tokenize(text)
        adv_tokens = []

        for tok in tokens:
            if tok.isalpha() and random.random() < prob:
                adv_tokens.append(self._get_synonym(tok))
            else:
                adv_tokens.append(tok)

        return " ".join(adv_tokens)


# ------------------------------------------------------------------------------------
# Model Wrappers for QA Evaluation
# ------------------------------------------------------------------------------------

class QAInference:
    """
    Wrapper around HF models for extractive QA or generative QA evaluation.
    """

    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if "cuda" else torch.float32
        ).to(self.device)

        # Special tokens
        self.eos = self.tokenizer.eos_token or "</s>"

    def answer(self, question: str, context: str = "", max_length: int = 64) -> str:
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs, max_length=inputs["input_ids"].shape[1] + max_length
        )
        ans = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return ans.split("Answer:")[-1].strip()

    # For exact match QA scoring
    @staticmethod
    def exact_match(pred: str, gold: str) -> bool:
        return pred.lower().strip() == gold.lower().strip()


# ------------------------------------------------------------------------------------
# Robustness Evaluator
# ------------------------------------------------------------------------------------

class RobustnessEvaluator:
    """Computes robustness metrics across datasets and perturbations."""

    def __init__(self, model_name: str, device: str = None):
        self.qa = QAInference(model_name, device=device)
        self.perturber = PerturbationGenerator()

    # ----------------------------------------------------------------------
    # Evaluate on a dataset
    # ----------------------------------------------------------------------

    def evaluate_dataset(
        self,
        dataset: List[Dict],
        perturb_fn: Callable[[str], str],
        max_samples: Optional[int] = None,
    ) -> RobustnessResult:
        """
        dataset: list of {"question": ..., "answer": ...}
        perturb_fn: noise / paraphrase / adversarial function
        """

        if max_samples:
            dataset = dataset[:max_samples]

        clean_correct = 0
        perturbed_correct = 0

        for item in dataset:
            q = item["question"]
            gold = item["answer"]

            # Clean prediction
            pred_clean = self.qa.answer(q)
            if self.qa.exact_match(pred_clean, gold):
                clean_correct += 1

            # Perturbed question
            q_pert = perturb_fn(q)

            pred_pert = self.qa.answer(q_pert)
            if self.qa.exact_match(pred_pert, gold):
                perturbed_correct += 1

        clean_acc = clean_correct / len(dataset)
        pert_acc = perturbed_correct / len(dataset)
        drop = clean_acc - pert_acc

        return RobustnessResult(
            clean_accuracy=clean_acc,
            perturbed_accuracy=pert_acc,
            accuracy_drop=drop,
        )

    # ----------------------------------------------------------------------
    # Combine robustness across perturbation types
    # ----------------------------------------------------------------------

    def compute_overall_robustness(
        self,
        dataset: List[Dict],
        max_samples: int = None,
    ) -> Dict[str, RobustnessResult]:

        results = {
            "noise": self.evaluate_dataset(dataset, self.perturber.apply_noise, max_samples),
            "paraphrase": self.evaluate_dataset(dataset, self.perturber.paraphrase, max_samples),
            "adversarial": self.evaluate_dataset(dataset, self.perturber.adversarial_attack, max_samples),
        }

        return results

    # ----------------------------------------------------------------------
    # Final robustness score for CRS
    # ----------------------------------------------------------------------

    @staticmethod
    def compute_final_score(results: Dict[str, RobustnessResult]) -> float:
        """
        Robustness score = 1 - (avg_drop / avg_clean_accuracy)
        """
        drops = [r.accuracy_drop for r in results.values()]
        clean = [r.clean_accuracy for r in results.values()]

        avg_drop = sum(drops) / len(drops)
        avg_clean = sum(clean) / len(clean)

        if avg_clean == 0:
            return 0.0

        score = 1 - (avg_drop / avg_clean)
        return max(0.0, min(1.0, score))  # clamp to [0,1]
