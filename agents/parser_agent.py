"""
Parser Agent - Cleans and normalizes text extracted from legal PDFs.

Uses Mistral-7B-Instruct (optional) or rule-based logic to normalize whitespace,
fix OCR artifacts, and produce clean text suitable for downstream agents.
"""

import re
from typing import Optional
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.llm_loader import LLMLoader


class ParserAgent:
    """
    Cleans and normalizes raw PDF text using rule-based preprocessing
    and optional LLM-based refinement (Mistral-7B-Instruct-v0.2).
    """

    PARSER_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

    def __init__(self, use_llm: bool = True, llm_loader: Optional[LLMLoader] = None):
        """
        Initialize the parser agent.

        Args:
            use_llm: If True, use Mistral for additional normalization.
            llm_loader: Optional pre-initialized LLM loader (otherwise created on first use).
        """
        self.use_llm = use_llm
        self._llm = llm_loader

    def _get_llm(self) -> Optional[LLMLoader]:
        if self._llm is None and self.use_llm:
            self._llm = LLMLoader(
                self.PARSER_MODEL_ID,
                max_new_tokens=2048,
                temperature=0.1,
            )
        return self._llm

    def _preprocess(self, text: str) -> str:
        """Rule-based cleanup: whitespace, line breaks, common artifacts."""
        if not text or not text.strip():
            return ""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple newlines to double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces to single
        text = re.sub(r"[ \t]+", " ", text)
        # Strip per line and rejoin
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n\n".join(lines)

    def _llm_clean(self, text: str) -> str:
        """Use LLM to normalize legal text (fix obvious OCR/formatting issues)."""
        llm = self._get_llm()
        if llm is None:
            return text
        # Truncate if too long for model context
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Document truncated for parsing...]"
        prompt = (
            "You are a legal document parser. Normalize the following extracted legal text. "
            "Fix obvious OCR errors, normalize spacing and line breaks. Output ONLY the cleaned text, no commentary.\n\n"
            f"Text:\n{text}"
        )
        try:
            out = llm.generate(prompt, max_new_tokens=2048)
            return out.strip() if out else text
        except Exception:
            return text

    def parse(self, raw_text: str) -> str:
        """
        Clean and normalize raw text from a PDF.

        Args:
            raw_text: Raw extracted text from load_pdf_text.

        Returns:
            Cleaned, normalized text.
        """
        cleaned = self._preprocess(raw_text)
        if self.use_llm and cleaned:
            cleaned = self._llm_clean(cleaned)
        return cleaned
