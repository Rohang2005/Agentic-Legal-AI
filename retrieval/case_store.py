"""
Persistent case metadata store backed by JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


class CaseStore:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path or "data/case_store.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.storage_path.exists():
            self._write([])

    def _read(self) -> List[Dict[str, Any]]:
        with self.storage_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, records: List[Dict[str, Any]]) -> None:
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def upsert_case(self, case_record: Dict[str, Any]) -> None:
        with self._lock:
            records = self._read()
            doc_id = str(case_record.get("document_id", ""))
            replaced = False
            for i, record in enumerate(records):
                if str(record.get("document_id", "")) == doc_id:
                    records[i] = case_record
                    replaced = True
                    break
            if not replaced:
                records.append(case_record)
            self._write(records)

    def list_cases(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read()

    def search_cases(
        self,
        act: str = "",
        section: str = "",
        outcome: str = "",
        court: str = "",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        act_l = act.lower().strip()
        section_l = section.lower().strip()
        outcome_l = outcome.lower().strip()
        court_l = court.lower().strip()
        matches: List[Dict[str, Any]] = []
        with self._lock:
            records = self._read()

        for record in records:
            if outcome_l and str(record.get("outcome_normalized", "")).lower() != outcome_l:
                continue
            if court_l and court_l not in str(record.get("court", "")).lower():
                continue

            if act_l or section_l:
                sections = record.get("sections_normalized", [])
                has_section = False
                for item in sections:
                    item_act = str(item.get("act", "")).lower()
                    item_section = str(item.get("section", "")).lower()
                    if act_l and act_l != item_act:
                        continue
                    if section_l and section_l != item_section:
                        continue
                    has_section = True
                    break
                if not has_section:
                    continue

            matches.append(record)
            if len(matches) >= limit:
                break

        return matches
