"""
Structure Extraction Agent - Extracts structured legal metadata from judgments.

Uses Qwen2-1.5B-Instruct (optional) plus heuristics to produce JSON with case name, court, parties,
sections of law, precedents, and final decision.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.llm_loader import LLMLoader
from utils.legal_normalizer import normalize_outcome


class StructureAgent:
    """
    Extracts structured legal metadata from document text.
    Model: Qwen/Qwen2-1.5B-Instruct (optional) with heuristic fallback
    """

    STRUCTURE_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

    OUTPUT_SCHEMA = {
        "case_name": "",
        "court": "",
        "judge": "",
        "petitioner": "",
        "respondent": "",
        "main_issue": "",
        "petitioner_arguments": [],
        "respondent_arguments": [],
        "sections_of_law": [],
        "precedents": [],
        "court_reasoning": [],
        "final_decision": "",
        "outcome_normalized": "",
    }

    @staticmethod
    def _dedupe_lines(items: List[str], max_items: int = 10) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            clean = " ".join(str(item).split()).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def _extract_sentences_by_keywords(text: str, keywords: List[str], max_items: int = 6) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        matched: List[str] = []
        for s in sentences:
            line = s.strip()
            if len(line) < 20:
                continue
            low = line.lower()
            if any(k in low for k in keywords):
                matched.append(line[:260])
                if len(matched) >= max_items:
                    break
        return StructureAgent._dedupe_lines(matched, max_items=max_items)

    def _build_salient_sample(self, text: str, max_chars: int = 6000) -> str:
        if len(text) <= max_chars:
            return text
        head = text[:2200]
        tail = text[-1800:]
        keyword_lines: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if len(line) < 25:
                continue
            low = line.lower()
            if any(
                token in low
                for token in (
                    "section ", "issue", "petitioner", "respondent", "appellant",
                    "therefore", "held", "order", "judgment", "v.", "vs.", "versus",
                )
            ):
                keyword_lines.append(line)
                if len(keyword_lines) >= 35:
                    break
        mid = "\n".join(keyword_lines)
        return f"{head}\n\n{mid}\n\n{tail}"[:max_chars]

    def _fallback_extract(self, document_text: str) -> Dict[str, Any]:
        out = dict(self.OUTPUT_SCHEMA)
        lines = [line.strip() for line in document_text.splitlines() if line.strip()]
        first_block = lines[:80]
        tail_block = lines[-120:] if len(lines) > 120 else lines

        for line in first_block:
            if re.search(r"\s(v\.|vs\.?|versus)\s", line, flags=re.IGNORECASE):
                out["case_name"] = line[:180]
                break
        for line in first_block:
            if re.search(r"\bcourt\b", line, flags=re.IGNORECASE):
                out["court"] = line[:180]
                break
        for line in first_block:
            if re.search(r"\b(justice|coram|hon'?ble)\b", line, flags=re.IGNORECASE):
                out["judge"] = line[:180]
                break
        decision_keywords = ("dismissed", "allowed", "disposed", "acquitted", "convicted", "guilty")
        for line in reversed(tail_block):
            if any(k in line.lower() for k in decision_keywords):
                out["final_decision"] = line[:240]
                break

        sections: List[str] = []
        for match in re.finditer(
            r"\bsection\s*([0-9A-Za-z\-]+)\s*(?:of\s+the\s+)?([A-Za-z .()]{0,40})",
            document_text,
            flags=re.IGNORECASE,
        ):
            sec = match.group(1).strip()
            act_hint = (match.group(2) or "").strip()
            if sec:
                sections.append(f"Section {sec} {act_hint}".strip())
            if len(sections) >= 10:
                break
        out["sections_of_law"] = self._dedupe_lines(sections, max_items=10)

        precedent_candidates = re.findall(
            r"\b([A-Z][A-Za-z0-9.&,'()\- ]{2,80}\s+(?:v\.|vs\.?|versus)\s+[A-Z][A-Za-z0-9.&,'()\- ]{2,80})\b",
            document_text,
        )
        out["precedents"] = self._dedupe_lines(precedent_candidates, max_items=10)

        issue_lines = self._extract_sentences_by_keywords(
            document_text,
            keywords=["issue", "question for consideration", "point for determination", "dispute"],
            max_items=2,
        )
        out["main_issue"] = issue_lines[0] if issue_lines else ""
        out["petitioner_arguments"] = self._extract_sentences_by_keywords(
            document_text,
            keywords=["petitioner submitted", "appellant submitted", "for the petitioner", "learned counsel for the petitioner"],
            max_items=6,
        )
        out["respondent_arguments"] = self._extract_sentences_by_keywords(
            document_text,
            keywords=["respondent submitted", "for the respondent", "state submitted", "public prosecutor"],
            max_items=6,
        )
        out["court_reasoning"] = self._extract_sentences_by_keywords(
            document_text,
            keywords=["we hold", "it is clear", "therefore", "in view of", "this court finds", "observed that"],
            max_items=8,
        )
        out["outcome_normalized"] = normalize_outcome(
            str(out.get("final_decision", "")),
            str(out.get("outcome_normalized", "")),
        )
        return out

    def _coalesce(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(primary)
        for key, value in secondary.items():
            if key not in merged:
                merged[key] = value
                continue
            current = merged.get(key)
            if isinstance(current, list):
                if not current and isinstance(value, list):
                    merged[key] = value
            else:
                if (current is None or str(current).strip() == "") and str(value).strip():
                    merged[key] = value
        return self._ensure_schema(merged)

    def __init__(self, llm_loader: Optional[LLMLoader] = None, use_llm: bool = True):
        self._llm = llm_loader
        self.use_llm = use_llm

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
                if key in (
                    "sections_of_law",
                    "precedents",
                    "petitioner_arguments",
                    "respondent_arguments",
                    "court_reasoning",
                ) and not isinstance(val, list):
                    out[key] = [val] if isinstance(val, str) else []
                else:
                    out[key] = val
        return out

    def _normalize_outcome(self, final_decision: str, outcome_from_model: str = "") -> str:
        return normalize_outcome(final_decision, outcome_from_model)

    def extract(self, document_text: str) -> Dict[str, Any]:
        """
        Extract structured metadata from legal document text.

        Args:
            document_text: Full or truncated cleaned document text.

        Returns:
            Dict with case_name, court, judge, petitioner, respondent,
            sections_of_law, precedents, final_decision.
        """
        fallback = self._fallback_extract(document_text)
        if not self.use_llm:
            return fallback
        # Use salient sample for more robust extraction quality.
        text_sample = self._build_salient_sample(document_text, max_chars=6000)
        prompt = (
            "Extract legal case metadata from the following judgment text. "
            "Respond with a single JSON object only, no other text. Use this exact structure:\n"
            '{"case_name": "", "court": "", "judge": "", "petitioner": "", "respondent": "", '
            '"main_issue": "", "petitioner_arguments": [], "respondent_arguments": [], '
            '"sections_of_law": [], "precedents": [], "court_reasoning": [], '
            '"final_decision": "", "outcome_normalized": ""}\n\n'
            f"Document:\n{text_sample}"
        )
        llm = self._get_llm()
        try:
            response = llm.generate(prompt, max_new_tokens=1024)
            data = self._extract_json_from_response(response)
            ensured = self._ensure_schema(data)
            merged = self._coalesce(fallback, ensured)
            merged["outcome_normalized"] = self._normalize_outcome(
                str(merged.get("final_decision", "")),
                str(merged.get("outcome_normalized", "")),
            )
            return merged
        except Exception:
            fallback["outcome_normalized"] = self._normalize_outcome(
                str(fallback.get("final_decision", "")),
                str(fallback.get("outcome_normalized", "")),
            )
            return fallback
