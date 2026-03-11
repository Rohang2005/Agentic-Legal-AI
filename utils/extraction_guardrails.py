"""
Validation and provenance helpers for extraction outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


REQUIRED_LIST_FIELDS = [
    "petitioner_arguments",
    "respondent_arguments",
    "sections_of_law",
    "precedents",
    "court_reasoning",
]


def build_provenance(model_name: str, stage: str, notes: str = "") -> Dict[str, str]:
    return {
        "stage": stage,
        "model": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }


def validate_extraction(metadata: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if not str(metadata.get("case_name", "")).strip():
        warnings.append("case_name is empty")
    if not str(metadata.get("final_decision", "")).strip():
        warnings.append("final_decision is empty")
    for field in REQUIRED_LIST_FIELDS:
        value = metadata.get(field, [])
        if not isinstance(value, list):
            warnings.append(f"{field} should be a list")
        elif len(value) == 0:
            warnings.append(f"{field} is empty")
    outcome = str(metadata.get("outcome_normalized", "unknown"))
    if outcome == "unknown":
        warnings.append("outcome_normalized is unknown")
    return warnings
