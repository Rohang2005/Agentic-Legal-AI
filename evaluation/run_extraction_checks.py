"""
Lightweight extraction quality checks for structured metadata outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.extraction_guardrails import validate_extraction


SAMPLE_PATH = Path("evaluation/sample_extractions.json")


def load_samples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def run_checks(samples: List[Dict[str, Any]]) -> int:
    total_warnings = 0
    for idx, sample in enumerate(samples, start=1):
        metadata = sample.get("metadata", {})
        warnings = validate_extraction(metadata)
        total_warnings += len(warnings)
        print(f"[sample-{idx}] warnings={len(warnings)}")
        for item in warnings:
            print(f"  - {item}")
    return total_warnings


if __name__ == "__main__":
    samples = load_samples(SAMPLE_PATH)
    if not samples:
        print("No evaluation samples found at evaluation/sample_extractions.json")
        raise SystemExit(0)
    warnings_count = run_checks(samples)
    print(f"Completed checks. Total warnings: {warnings_count}")
