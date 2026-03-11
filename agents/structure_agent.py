"""
Structure Extraction Agent - Extracts structured legal metadata from judgments.

Uses Qwen2-7B-Instruct to produce JSON with case name, court, parties,
sections of law, precedents, and final decision.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.llm_loader import LLMLoader


class StructureAgent:
    """
    Extracts structured legal metadata from document text.
    Model: Qwen/Qwen2-7B-Instruct (open-source)
    """

    STRUCTURE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"

    OUTPUT_SCHEMA = {
        "case_name": "",
        "court": "",
        "judge": "",
        "petitioner": "",
        "respondent": "",
        "sections_of_law": [],
        "precedents": [],
        "final_decision": "",
    }

    def __init__(self, llm_loader: Optional[LLMLoader] = None):
        self._llm = llm_loader

    def _get_llm(self) -> LLMLoader:
        if self._llm is None:
            self._llm = LLMLoader(
                self.STRUCTURE_MODEL_ID,
                max_new_tokens=1024,
                temperature=0.2,
            )
        return self._llm

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, handling markdown code blocks."""
        response = response.strip()
        # Try to find JSON block
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass
        return dict(self.OUTPUT_SCHEMA)

    def _ensure_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected keys exist with correct types."""
        out = dict(self.OUTPUT_SCHEMA)
        for key in out:
            if key in data:
                val = data[key]
                if key in ("sections_of_law", "precedents") and not isinstance(val, list):
                    out[key] = [val] if isinstance(val, str) else []
                else:
                    out[key] = val
        return out

    def extract(self, document_text: str) -> Dict[str, Any]:
        """
        Extract structured metadata from legal document text.

        Args:
            document_text: Full or truncated cleaned document text.

        Returns:
            Dict with case_name, court, judge, petitioner, respondent,
            sections_of_law, precedents, final_decision.
        """
        # Use first ~6000 chars for structure extraction to fit context
        text_sample = document_text[:6000] if len(document_text) > 6000 else document_text
        prompt = (
            "Extract legal case metadata from the following judgment text. "
            "Respond with a single JSON object only, no other text. Use this exact structure:\n"
            '{"case_name": "", "court": "", "judge": "", "petitioner": "", "respondent": "", '
            '"sections_of_law": [], "precedents": [], "final_decision": ""}\n\n'
            f"Document:\n{text_sample}"
        )
        llm = self._get_llm()
        try:
            response = llm.generate(prompt, max_new_tokens=1024)
            data = self._extract_json_from_response(response)
            return self._ensure_schema(data)
        except Exception:
            return dict(self.OUTPUT_SCHEMA)
