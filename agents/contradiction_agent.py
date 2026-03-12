"""
Contradiction Detection Agent - Detects contradictions between statements using NLI.

Uses cross-encoder/nli-deberta-v3-base for natural language inference:
compare claims pairwise and return pairs classified as contradictory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.hf_config import HF_TOKEN

NLI_LABELS = ["entailment", "neutral", "contradiction"]


class ContradictionAgent:
    """
    Splits document into claims and uses NLI (DeBERTa-v3-base) to detect
    contradictory statement pairs.
    """

    NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
    _SKIP_PREFIXES = (
        "indiankanoon",
        "http://",
        "https://",
        "coram",
        "versus",
        "petitioner",
        "respondent",
    )
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "that", "this", "it",
        "as", "at", "from", "into", "under", "against", "shall", "may", "can",
        "would", "could", "should", "than", "then", "there", "their", "his", "her",
        "its", "our", "your", "who", "whom", "which", "what", "when",
    }

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._tokenizer = None
        self._device = device
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.NLI_MODEL_ID,
                    token=HF_TOKEN,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.NLI_MODEL_ID,
                    token=HF_TOKEN,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to load contradiction model '{self.NLI_MODEL_ID}': {exc}"
                ) from exc
            self._model.to(self._device)
            self._model.eval()
            if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                self._nli_labels = [self._model.config.id2label[i] for i in sorted(self._model.config.id2label)]
            else:
                self._nli_labels = NLI_LABELS

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _clean_claim(self, text: str) -> str:
        s = self._normalize_whitespace(text)
        s = re.sub(r"https?://\S+", "", s)
        s = re.sub(r"\bwww\.\S+", "", s)
        s = re.sub(r"^\s*\d+\s*[\).:-]\s*", "", s)
        s = re.sub(r"^\s*[-*•]\s*", "", s)
        return self._normalize_whitespace(s)

    def _is_valid_claim(self, claim: str) -> bool:
        low = claim.lower()
        if len(claim) < 35:
            return False
        if low.startswith(self._SKIP_PREFIXES):
            return False
        # Reject near-title/party-only lines and citation-heavy noise.
        if low.count(" v. ") + low.count(" vs ") + low.count(" versus ") > 0 and len(low.split()) < 12:
            return False
        if re.search(r"\b\d{4}\b", low) and len(low.split()) < 10:
            return False
        alpha_ratio = sum(ch.isalpha() for ch in claim) / max(len(claim), 1)
        if alpha_ratio < 0.6:
            return False
        return True

    def _split_into_claims(self, text: str, max_claims: int = 35) -> List[str]:
        """Split document into substantive claim sentences for comparison."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims: List[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            cleaned = self._clean_claim(sentence)
            if not self._is_valid_claim(cleaned):
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            claims.append(cleaned)
            if len(claims) >= max_claims:
                break
        return claims

    def _claim_terms(self, claim: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z]{3,}", claim.lower())
        return {t for t in tokens if t not in self._STOPWORDS}

    def _should_compare_pair(self, claim1: str, claim2: str) -> bool:
        terms1 = self._claim_terms(claim1)
        terms2 = self._claim_terms(claim2)
        if not terms1 or not terms2:
            return False
        overlap = len(terms1.intersection(terms2))
        min_terms = min(len(terms1), len(terms2))
        # Compare only semantically related statements to reduce false positives.
        return overlap >= 2 and (overlap / max(min_terms, 1)) >= 0.2

    @staticmethod
    def _normalize_label(label: str) -> str:
        l = label.lower().strip()
        if "contra" in l:
            return "contradiction"
        if "entail" in l:
            return "entailment"
        if "neutral" in l:
            return "neutral"
        return l

    def _nli_distribution(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Run NLI and return normalized label probabilities."""
        import torch

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
        labels = getattr(self, "_nli_labels", NLI_LABELS)
        dist: Dict[str, float] = {}
        for idx, p in enumerate(probs):
            raw = labels[idx] if idx < len(labels) else str(idx)
            dist[self._normalize_label(raw)] = float(p)
        return dist

    def _check_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        """Check if claim1 and claim2 are in contradiction. Returns (is_contradiction, confidence)."""
        d1 = self._nli_distribution(claim1, claim2)
        d2 = self._nli_distribution(claim2, claim1)
        c1 = d1.get("contradiction", 0.0)
        c2 = d2.get("contradiction", 0.0)
        e1 = d1.get("entailment", 0.0)
        e2 = d2.get("entailment", 0.0)
        conf = max(c1, c2)
        # Require contradiction to dominate and avoid strong entailment pairs.
        if conf >= 0.85 and max(e1, e2) < 0.6:
            return True, conf
        return False, 0.0

    def detect(self, document_text: str, min_confidence: float = 0.88, max_pairs: int = 60) -> List[Dict[str, Any]]:
        """
        Detect contradictory statement pairs in the document.

        Returns:
            List of {"statement_1", "statement_2", "confidence"}.
        """
        claims = self._split_into_claims(document_text)
        if len(claims) < 2:
            return []

        contradictions: List[Dict[str, Any]] = []
        pairs_checked = 0
        seen_pairs: set[Tuple[str, str]] = set()
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                if pairs_checked >= max_pairs:
                    break
                c1, c2 = claims[i], claims[j]
                if not self._should_compare_pair(c1, c2):
                    continue
                key = tuple(sorted((c1.lower(), c2.lower())))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                is_contra, conf = self._check_contradiction(c1, c2)
                if is_contra and conf >= min_confidence:
                    contradictions.append({
                        "statement_1": c1,
                        "statement_2": c2,
                        "confidence": round(conf, 4),
                    })
                pairs_checked += 1
            if pairs_checked >= max_pairs:
                break

        contradictions.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return contradictions[:8]
