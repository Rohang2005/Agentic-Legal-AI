"""
Final Review Agent - Post-processes extracted outputs for frontend readability.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from models.llm_loader import LLMLoader


class FinalReviewAgent:
    """Normalizes and formats extraction outputs before UI rendering."""

    REVIEW_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        use_llm: bool = True,
        require_gpu: bool = False,
        model_id: str = REVIEW_MODEL_ID,
    ):
        self.use_llm = use_llm
        self.require_gpu = require_gpu
        self.model_id = model_id
        self._llm: LLMLoader | None = None

    def _can_use_llm(self) -> bool:
        if not self.use_llm:
            return False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if self.require_gpu and not has_cuda:
            return False
        # Avoid loading 7B review model on CPU-only environments.
        return has_cuda

    def _get_llm(self) -> LLMLoader:
        if self._llm is None:
            self._llm = LLMLoader(
                self.model_id,
                require_gpu=self.require_gpu,
                load_in_4bit=True,
                max_new_tokens=384,
                do_sample=False,
                temperature=0.1,
            )
        return self._llm

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, Any] | None:
        raw = (text or "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _llm_refine(
        self,
        metadata: Dict[str, Any],
        timeline: List[Dict[str, str]],
        contradictions: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        llm = self._get_llm()
        payload = {
            "metadata": metadata,
            "timeline": timeline[:8],
            "contradictions_count": len(contradictions),
        }
        prompt = (
            "You are a legal case output reviewer. Improve readability only; do not invent facts.\n"
            "Given the extracted case payload, return JSON only with this schema:\n"
            '{"headline":{"case_name":"","court":"","outcome":""},"summary":["", "", ""]}\n'
            "Rules:\n"
            "- Keep summary to 3-6 concise bullets.\n"
            "- Use only facts present in input.\n"
            "- If a field is missing, keep it empty.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
        )
        try:
            out = llm.generate(prompt, max_new_tokens=384)
            parsed = self._extract_json_block(out)
            if not parsed:
                return None
            headline = parsed.get("headline", {})
            summary = parsed.get("summary", [])
            if not isinstance(headline, dict) or not isinstance(summary, list):
                return None
            return {"headline": headline, "summary": summary}
        except Exception:
            return None

    @staticmethod
    def _clean_text(value: Any, max_chars: int = 320, prefer_sentence: bool = False) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        text = text.strip(" -\t")
        text = text.lstrip()
        text = re.sub(r"^\d+\s*[\).:-]\s+", "", text)
        if len(text) <= max_chars:
            return text

        # Prefer not to cut in the middle of a sentence.
        if prefer_sentence:
            sentence_end = max(text.rfind(".", 0, max_chars + 1), text.rfind("?", 0, max_chars + 1), text.rfind("!", 0, max_chars + 1))
            if sentence_end >= int(max_chars * 0.55):
                return text[: sentence_end + 1].strip()

        # Fallback: cut at last whitespace before limit.
        cut = text.rfind(" ", 0, max_chars + 1)
        if cut < int(max_chars * 0.5):
            cut = max_chars
        clipped = text[:cut].strip()
        if clipped.endswith((".", "?", "!")):
            return clipped
        return f"{clipped}..."

    @staticmethod
    def _looks_incomplete(text: str) -> bool:
        if not text:
            return True
        low = text.lower().strip()
        if "..." in low:
            return True
        if low.endswith((",", ";", ":", "-", " i.e.", " i.e.,")):
            return True
        if low.endswith((" and", " or", " to", " of", " for", " with", " through", " by", " under")):
            return True
        return False

    @staticmethod
    def _to_list(
        value: Any,
        max_items: int = 10,
        max_chars: int = 320,
        prefer_sentence: bool = False,
        drop_incomplete: bool = True,
    ) -> List[str]:
        if value is None:
            return []
        raw = value if isinstance(value, list) else [value]
        out: List[str] = []
        seen: set[str] = set()
        for item in raw:
            cleaned = FinalReviewAgent._clean_text(item, max_chars=max_chars, prefer_sentence=prefer_sentence)
            if not cleaned:
                continue
            if drop_incomplete and FinalReviewAgent._looks_incomplete(cleaned):
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        case_name = FinalReviewAgent._clean_text(metadata.get("case_name", ""), 400)
        court = FinalReviewAgent._clean_text(metadata.get("court", ""), 300)
        judge = FinalReviewAgent._clean_text(metadata.get("judge", ""), 260)
        petitioner = FinalReviewAgent._clean_text(metadata.get("petitioner", ""), 260)
        respondent = FinalReviewAgent._clean_text(metadata.get("respondent", ""), 260)
        main_issue = FinalReviewAgent._clean_text(metadata.get("main_issue", ""), 520, prefer_sentence=True)
        final_decision = FinalReviewAgent._clean_text(metadata.get("final_decision", ""), 520, prefer_sentence=True)
        if FinalReviewAgent._looks_incomplete(case_name):
            case_name = ""
        if FinalReviewAgent._looks_incomplete(court):
            court = ""
        if FinalReviewAgent._looks_incomplete(judge):
            judge = ""
        if FinalReviewAgent._looks_incomplete(petitioner):
            petitioner = ""
        if FinalReviewAgent._looks_incomplete(respondent):
            respondent = ""
        if FinalReviewAgent._looks_incomplete(main_issue):
            main_issue = ""
        if FinalReviewAgent._looks_incomplete(final_decision):
            final_decision = ""
        return {
            "case_name": case_name,
            "court": court,
            "judge": judge,
            "petitioner": petitioner,
            "respondent": respondent,
            "main_issue": main_issue,
            "petitioner_arguments": FinalReviewAgent._to_list(
                metadata.get("petitioner_arguments", []),
                max_items=8,
                max_chars=420,
                prefer_sentence=True,
            ),
            "respondent_arguments": FinalReviewAgent._to_list(
                metadata.get("respondent_arguments", []),
                max_items=8,
                max_chars=420,
                prefer_sentence=True,
            ),
            "sections_of_law": FinalReviewAgent._to_list(
                metadata.get("sections_of_law", []),
                max_items=12,
                max_chars=180,
                prefer_sentence=False,
                drop_incomplete=False,
            ),
            "precedents": FinalReviewAgent._to_list(
                metadata.get("precedents", []),
                max_items=10,
                max_chars=260,
                prefer_sentence=False,
                drop_incomplete=False,
            ),
            "court_reasoning": FinalReviewAgent._to_list(
                metadata.get("court_reasoning", []),
                max_items=10,
                max_chars=460,
                prefer_sentence=True,
            ),
            "final_decision": final_decision,
            "outcome_normalized": FinalReviewAgent._clean_text(metadata.get("outcome_normalized", ""), 50),
        }

    @staticmethod
    def _normalize_timeline(timeline: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in timeline or []:
            if not isinstance(item, dict):
                continue
            date = FinalReviewAgent._clean_text(item.get("date", ""), 40)
            event = FinalReviewAgent._clean_text(item.get("event", ""), 380, prefer_sentence=True)
            if not event:
                continue
            if FinalReviewAgent._looks_incomplete(event):
                continue
            key = f"{date}|{event}".lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"date": date, "event": event})
            if len(out) >= 15:
                break
        return out

    @staticmethod
    def _normalize_contradictions(contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in contradictions or []:
            if not isinstance(item, dict):
                continue
            s1 = FinalReviewAgent._clean_text(item.get("statement_1", ""), 280)
            s2 = FinalReviewAgent._clean_text(item.get("statement_2", ""), 280)
            if len(s1) < 25 or len(s2) < 25:
                continue
            if FinalReviewAgent._looks_incomplete(s1) or FinalReviewAgent._looks_incomplete(s2):
                continue
            if s1.lower() == s2.lower():
                continue
            key = "|".join(sorted((s1.lower(), s2.lower())))
            if key in seen:
                continue
            seen.add(key)
            confidence_raw = item.get("confidence", 0.0)
            try:
                confidence = round(float(confidence_raw), 4)
            except Exception:
                confidence = 0.0
            out.append(
                {
                    "statement_1": s1,
                    "statement_2": s2,
                    "confidence": confidence,
                }
            )
            if len(out) >= 8:
                break
        out.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        return out

    @staticmethod
    def _build_summary(metadata: Dict[str, Any], timeline: List[Dict[str, str]], contradictions: List[Dict[str, Any]]) -> List[str]:
        summary: List[str] = []
        if metadata.get("main_issue"):
            summary.append(f"Main issue: {metadata['main_issue']}")
        if metadata.get("final_decision"):
            summary.append(f"Final decision: {metadata['final_decision']}")
        if metadata.get("outcome_normalized"):
            summary.append(f"Outcome: {metadata['outcome_normalized']}")
        if metadata.get("sections_of_law"):
            sections = ", ".join(metadata["sections_of_law"][:3])
            summary.append(f"Key sections: {sections}")
        if timeline:
            summary.append(f"Timeline events extracted: {len(timeline)}")
        if contradictions:
            summary.append(f"Potential contradictions detected: {len(contradictions)}")
        return summary[:6]

    def review(
        self,
        metadata: Dict[str, Any],
        timeline: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        reviewed_metadata = self._normalize_metadata(metadata or {})
        reviewed_timeline = self._normalize_timeline(timeline or [])
        reviewed_contradictions = self._normalize_contradictions(contradictions or [])
        reviewed_warnings = self._to_list(warnings or [], max_items=25)
        review_notes: List[str] = []

        headline = {
            "case_name": reviewed_metadata.get("case_name", ""),
            "court": reviewed_metadata.get("court", ""),
            "outcome": reviewed_metadata.get("outcome_normalized", ""),
        }
        summary = self._build_summary(reviewed_metadata, reviewed_timeline, reviewed_contradictions)

        if self._can_use_llm():
            refined = self._llm_refine(
                metadata=reviewed_metadata,
                timeline=reviewed_timeline,
                contradictions=reviewed_contradictions,
            )
            if refined:
                llm_headline = refined.get("headline", {})
                llm_summary = refined.get("summary", [])
                if isinstance(llm_headline, dict):
                    headline = {
                        "case_name": self._clean_text(llm_headline.get("case_name", headline.get("case_name", "")), 400)
                        or headline.get("case_name", ""),
                        "court": self._clean_text(llm_headline.get("court", headline.get("court", "")), 300)
                        or headline.get("court", ""),
                        "outcome": self._clean_text(llm_headline.get("outcome", headline.get("outcome", "")), 50)
                        or headline.get("outcome", ""),
                    }
                if isinstance(llm_summary, list):
                    summary = self._to_list(
                        llm_summary,
                        max_items=6,
                        max_chars=320,
                        prefer_sentence=True,
                    ) or summary
                review_notes.append(f"final_review_llm:{self.model_id}")
            else:
                review_notes.append("final_review_llm_fallback_used")
        else:
            if self.use_llm:
                review_notes.append("final_review_llm_unavailable_no_cuda")

        return {
            "headline": headline,
            "summary": summary,
            "metadata": reviewed_metadata,
            "timeline": reviewed_timeline,
            "contradictions": reviewed_contradictions,
            "warnings": reviewed_warnings + review_notes,
        }

