"""
Contradiction Detection Agent - Detects contradictions between statements using NLI.

Uses microsoft/deberta-v3-large-mnli for natural language inference:
compare claims pairwise and return pairs classified as contradictory.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.hf_config import HF_TOKEN

# MNLI label order for microsoft/deberta-v3-large-mnli (check model card if different)
NLI_LABELS = ["entailment", "neutral", "contradiction"]


class ContradictionAgent:
    """
    Splits document into claims and uses NLI (DeBERTa-v3-large-mnli) to detect
    contradictory statement pairs.
    """

    NLI_MODEL_ID = "microsoft/deberta-v3-large-mnli"

    def __init__(self, device: Optional[str] = None):
        """
        Initialize NLI model for premise-hypothesis contradiction detection.

        Args:
            device: 'cuda', 'cpu', or None for auto.
        """
        self._model = None
        self._tokenizer = None
        self._device = device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.NLI_MODEL_ID,
                token=HF_TOKEN,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.NLI_MODEL_ID,
                token=HF_TOKEN,
            )
            self._model.to(self._device)
            self._model.eval()
            # Use model's label order if available
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._nli_labels = [self._model.config.id2label[i] for i in sorted(self._model.config.id2label)]
            else:
                self._nli_labels = NLI_LABELS

    def _split_into_claims(self, text: str, max_claims: int = 50) -> List[str]:
        """Split document into sentence-like claims for pairwise comparison."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:max_claims]

    def _nli_label(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Run NLI: premise vs hypothesis. Returns (label, score).
        Labels: contradiction, neutral, entailment.
        """
        self._load_model()
        inputs = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        idx = int(probs.argmax())
        labels = getattr(self, "_nli_labels", NLI_LABELS)
        label = labels[idx].lower() if idx < len(labels) else "neutral"
        score = float(probs[idx])
        return label, score

    def _check_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        """Check if claim1 and claim2 are in contradiction. Returns (is_contradiction, confidence)."""
        label1, score1 = self._nli_label(claim1, claim2)
        label2, score2 = self._nli_label(claim2, claim1)
        if label1 == "contradiction" or label2 == "contradiction":
            conf = score1 if label1 == "contradiction" else score2
            if label1 == "contradiction" and label2 == "contradiction":
                conf = max(score1, score2)
            return True, conf
        return False, 0.0

    def detect(self, document_text: str, min_confidence: float = 0.7, max_pairs: int = 100) -> List[Dict[str, Any]]:
        """
        Detect contradictory statement pairs in the document.

        Args:
            document_text: Cleaned document text.
            min_confidence: Minimum NLI score to report a contradiction.
            max_pairs: Maximum number of pairs to evaluate (for performance).

        Returns:
            List of {"statement_1", "statement_2", "confidence"}.
        """
        claims = self._split_into_claims(document_text)
        if len(claims) < 2:
            return []

        contradictions = []
        pairs_checked = 0
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                if pairs_checked >= max_pairs:
                    break
                is_contra, conf = self._check_contradiction(claims[i], claims[j])
                if is_contra and conf >= min_confidence:
                    contradictions.append({
                        "statement_1": claims[i],
                        "statement_2": claims[j],
                        "confidence": round(conf, 4),
                    })
                pairs_checked += 1
            if pairs_checked >= max_pairs:
                break

        return contradictions
