"""
Timeline Extraction Agent - Extracts chronological events from legal judgments.

Uses heuristic date extraction with optional Qwen2-1.5B-Instruct enrichment.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.llm_loader import LLMLoader


class TimelineAgent:
    """
    Extracts a timeline of events from legal document text.
    Model: Qwen/Qwen2-1.5B-Instruct (optional) with heuristic fallback
    """

    TIMELINE_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

    @staticmethod
    def _dedupe_events(events: List[Dict[str, Any]], max_items: int = 15) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in events:
            date = str(item.get("date", "")).strip()
            event = str(item.get("event", "")).strip()
            if not event:
                continue
            key = f"{date}|{event}".lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"date": date, "event": event})
            if len(out) >= max_items:
                break
        return out

    def _fallback_timeline(self, text: str, max_items: int = 12) -> List[Dict[str, Any]]:
        date_pattern = re.compile(
            r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
            r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
            r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"
        )
        sentences = re.split(r"(?<=[.!?])\s+", text)
        events: List[Dict[str, Any]] = []
        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 15:
                continue
            m = date_pattern.search(s)
            if not m:
                continue
            events.append({"date": m.group(1), "event": s[:240]})
            if len(events) >= max_items:
                break
        return self._dedupe_events(events, max_items=max_items)

    def __init__(self, llm_loader: Optional[LLMLoader] = None, use_llm: bool = True):
        self._llm = llm_loader
        self.use_llm = use_llm

    def _get_llm(self) -> LLMLoader:
        if self._llm is None:
            self._llm = LLMLoader(
                self.TIMELINE_MODEL_ID,
                max_new_tokens=1024,
                temperature=0.2,
            )
        return self._llm

    def _parse_timeline_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM output into list of {date, event} dicts."""
        response = response.strip()
        events = []
        # Try JSON array
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(response[start:end])
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            events.append({
                                "date": str(item.get("date", "")),
                                "event": str(item.get("event", "")),
                            })
                        elif isinstance(item, str):
                            events.append({"date": "", "event": item})
                    return self._dedupe_events(events)
            except json.JSONDecodeError:
                pass
        # Fallback: split by newlines and look for date-like + text
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match "date: ..." or "YYYY-MM-DD ..." or "- date: ..."
            m = re.match(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|[\w]+\s+\d{1,2},?\s+\d{4}|date:\s*)?(.+)$", line, re.I)
            if m:
                rest = m.group(1).strip()
                date_match = re.match(r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|[\w]+\s+\d{1,2},?\s+\d{4})\s*[:\-]\s*(.+)$", rest, re.I)
                if date_match:
                    events.append({"date": date_match.group(1), "event": date_match.group(2)})
                else:
                    events.append({"date": "", "event": rest})
        return self._dedupe_events(events)

    def extract(self, document_text: str) -> List[Dict[str, Any]]:
        """
        Extract chronological timeline of events from document.

        Args:
            document_text: Cleaned document text.

        Returns:
            List of {"date": str, "event": str} in chronological order.
        """
        fallback = self._fallback_timeline(document_text)
        if not self.use_llm:
            return fallback
        text_sample = document_text[:6000] if len(document_text) > 6000 else document_text
        prompt = (
            "From the following legal judgment text, extract a chronological timeline of events. "
            "Respond with a JSON array only. Each element must have 'date' and 'event' keys. "
            "Example: [{\"date\": \"2020-01-15\", \"event\": \"Filing of petition\"}, ...]\n\n"
            f"Document:\n{text_sample}"
        )
        llm = self._get_llm()
        try:
            response = llm.generate(prompt, max_new_tokens=1024)
            parsed = self._parse_timeline_response(response)
            return parsed if parsed else fallback
        except Exception:
            return fallback
