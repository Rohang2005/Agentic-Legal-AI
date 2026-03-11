"""
Utilities for normalizing extracted legal metadata into queryable fields.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def normalize_outcome(final_decision: str, outcome_hint: str = "") -> str:
    text = f"{outcome_hint} {final_decision}".lower()
    if any(token in text for token in ("acquit", "acquittal", "not guilty")):
        return "acquittal"
    if any(token in text for token in ("convict", "conviction", "guilty", "sentenced")):
        return "conviction"
    if "partly allowed" in text or "partially allowed" in text:
        return "partly_allowed"
    if "dismissed" in text or "rejected" in text:
        return "dismissed"
    if "allowed" in text:
        return "allowed"
    if "disposed" in text:
        return "disposed"
    return "unknown"


def _normalize_section_item(item: Any) -> Optional[Dict[str, str]]:
    raw = str(item).strip()
    if not raw:
        return None
    section_match = re.search(r"\bsection\s*([0-9A-Za-z\-]+)\b", raw, flags=re.IGNORECASE)
    num_match = section_match.group(1) if section_match else ""
    act = ""
    if "ipc" in raw.lower() or "indian penal code" in raw.lower():
        act = "IPC"
    elif "crpc" in raw.lower():
        act = "CrPC"
    elif "cpc" in raw.lower():
        act = "CPC"
    return {
        "act": act,
        "section": num_match,
        "raw_text": raw,
    }


def normalize_sections(sections_of_law: List[Any]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in sections_of_law or []:
        if isinstance(item, dict):
            raw = str(item.get("raw_text") or item.get("text") or item).strip()
            candidate = _normalize_section_item(raw)
        else:
            candidate = _normalize_section_item(item)
        if candidate:
            normalized.append(candidate)
    return normalized


def normalize_case_record(document_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    final_decision = str(metadata.get("final_decision", ""))
    outcome_hint = str(metadata.get("outcome_normalized", ""))
    normalized_outcome = normalize_outcome(final_decision, outcome_hint)
    normalized_sections = normalize_sections(metadata.get("sections_of_law", []))
    record = dict(metadata)
    record["document_id"] = document_id
    record["outcome_normalized"] = normalized_outcome
    record["sections_normalized"] = normalized_sections
    return record


def parse_nl_search_query(query: str) -> Dict[str, str]:
    query_l = (query or "").lower()
    section = ""
    section_match = re.search(r"section\s*([0-9A-Za-z\-]+)", query_l, flags=re.IGNORECASE)
    if section_match:
        section = section_match.group(1)
    act = "IPC" if "ipc" in query_l or "indian penal code" in query_l else ""
    outcome = ""
    if "acquitt" in query_l or "not guilty" in query_l:
        outcome = "acquittal"
    elif "convict" in query_l or "guilty" in query_l:
        outcome = "conviction"
    elif "dismiss" in query_l:
        outcome = "dismissed"
    elif "partly allowed" in query_l or "partially allowed" in query_l:
        outcome = "partly_allowed"
    elif "allowed" in query_l:
        outcome = "allowed"
    return {"act": act, "section": section, "outcome": outcome}
