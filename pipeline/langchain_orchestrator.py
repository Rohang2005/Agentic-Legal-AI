"""
LangChain-based pipeline orchestrator.

Uses LangChain chains for structure extraction, timeline extraction, and RAG Q&A
while keeping the same interface as LegalPipelineOrchestrator.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import re

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.parser_agent import ParserAgent
from agents.contradiction_agent import ContradictionAgent
from models.llm_loader import LLMLoader
from models.embedding_model import EmbeddingModel
from retrieval.vector_store import LegalVectorStore
from retrieval.case_store import CaseStore
from utils.text_chunker import TextChunker
from utils.pdf_loader import load_pdf_text
from utils.legal_normalizer import normalize_case_record, normalize_outcome
from utils.extraction_guardrails import build_provenance, validate_extraction

from .langchain_components import (
    get_structure_chain,
    get_timeline_chain,
    get_rag_chain,
    get_issue_chain,
    get_petitioner_arguments_chain,
    get_respondent_arguments_chain,
    get_reasoning_chain,
    parse_structure_output,
    parse_timeline_output,
    parse_single_field_json,
)


class LangChainOrchestrator:
    """
    Pipeline orchestrator using LangChain for structure, timeline, and RAG.
    Same interface as LegalPipelineOrchestrator: run_from_pdf, run_from_text, ask.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_llm_parser: bool = False,
        enable_contradiction_detection: bool = False,
        vector_store: Optional[LegalVectorStore] = None,
        index_path: Optional[Path] = None,
        case_store: Optional[CaseStore] = None,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.parser = ParserAgent(use_llm=use_llm_parser)
        self.contradiction_agent = ContradictionAgent()
        self.embedding_model = EmbeddingModel()
        self.vector_store = vector_store or LegalVectorStore(
            embedding_model=self.embedding_model,
            index_path=index_path,
        )

        # Reuse one loader to avoid repeated heavyweight model initialization per chain.
        self._llm_loader = LLMLoader(
            "Qwen/Qwen2-1.5B-Instruct",
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.2,
        )
        self._structure_chain = get_structure_chain(llm_loader=self._llm_loader)
        self._timeline_chain = get_timeline_chain(llm_loader=self._llm_loader)
        self._rag_chain = get_rag_chain(self.vector_store, llm_loader=self._llm_loader)
        self._issue_chain = get_issue_chain(llm_loader=self._llm_loader)
        self._petitioner_arguments_chain = get_petitioner_arguments_chain(llm_loader=self._llm_loader)
        self._respondent_arguments_chain = get_respondent_arguments_chain(llm_loader=self._llm_loader)
        self._reasoning_chain = get_reasoning_chain(llm_loader=self._llm_loader)
        self.enable_contradiction_detection = enable_contradiction_detection
        self.case_store = case_store or CaseStore()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._last_metadata: Optional[Dict[str, Any]] = None
        self._last_timeline: Optional[List[Dict[str, Any]]] = None
        self._last_contradictions: Optional[List[Dict[str, Any]]] = None
        self._last_cleaned_text: Optional[str] = None
        self._last_provenance: List[Dict[str, str]] = []
        self._last_warnings: List[str] = []

    def _fallback_metadata_from_text(self, metadata: Dict[str, Any], cleaned_text: str) -> Dict[str, Any]:
        """Best-effort heuristic fill for key fields when LLM JSON is sparse."""
        merged = dict(metadata)
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        first_block = lines[:80]
        tail_block = lines[-120:] if len(lines) > 120 else lines

        if not merged.get("case_name"):
            for line in first_block:
                if re.search(r"\b(v\.|vs\.?|versus)\b", line, flags=re.IGNORECASE):
                    merged["case_name"] = line[:180]
                    break

        if not merged.get("court"):
            for line in first_block:
                if re.search(r"\bcourt\b", line, flags=re.IGNORECASE):
                    merged["court"] = line[:180]
                    break

        if not merged.get("judge"):
            for line in first_block:
                if re.search(r"\b(justice|coram|hon'?ble)\b", line, flags=re.IGNORECASE):
                    merged["judge"] = line[:180]
                    break

        if not merged.get("final_decision"):
            decision_keywords = ("dismissed", "allowed", "disposed", "acquitted", "convicted", "guilty")
            for line in reversed(tail_block):
                if any(k in line.lower() for k in decision_keywords):
                    merged["final_decision"] = line[:240]
                    break

        if not merged.get("sections_of_law"):
            sections = []
            for match in re.finditer(
                r"\bsection\s*([0-9A-Za-z\-]+)\s*(?:of\s+the\s+)?([A-Za-z .()]{0,40})",
                cleaned_text,
                flags=re.IGNORECASE,
            ):
                sec = match.group(1).strip()
                act_hint = (match.group(2) or "").strip()
                if sec:
                    sections.append(f"Section {sec} {act_hint}".strip())
                if len(sections) >= 8:
                    break
            if sections:
                merged["sections_of_law"] = sections

        merged["outcome_normalized"] = normalize_outcome(
            str(merged.get("final_decision", "")),
            str(merged.get("outcome_normalized", "")),
        )
        return merged

    def _fallback_timeline_from_text(self, cleaned_text: str) -> List[Dict[str, Any]]:
        """Extract a lightweight timeline by date-like markers in text."""
        date_pattern = re.compile(
            r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
            r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
            r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"
        )
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
        timeline: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for sentence in sentences:
            if len(sentence.strip()) < 15:
                continue
            match = date_pattern.search(sentence)
            if not match:
                continue
            date = match.group(1)
            event = sentence.strip()[:220]
            key = f"{date}|{event}"
            if key in seen:
                continue
            seen.add(key)
            timeline.append({"date": date, "event": event})
            if len(timeline) >= 12:
                break
        return timeline

    def _merge_specialized_fields(self, metadata: Dict[str, Any], document_text: str) -> Dict[str, Any]:
        merged = dict(metadata)
        text_sample = document_text[:8000] if len(document_text) > 8000 else document_text

        if not merged.get("main_issue"):
            issue_raw = self._issue_chain.invoke({"document": text_sample})
            issue = parse_single_field_json(str(issue_raw), "main_issue", "")
            if issue:
                merged["main_issue"] = issue

        if not merged.get("petitioner_arguments"):
            pet_raw = self._petitioner_arguments_chain.invoke({"document": text_sample})
            petitioner_arguments = parse_single_field_json(str(pet_raw), "petitioner_arguments", [])
            if petitioner_arguments:
                merged["petitioner_arguments"] = petitioner_arguments

        if not merged.get("respondent_arguments"):
            resp_raw = self._respondent_arguments_chain.invoke({"document": text_sample})
            respondent_arguments = parse_single_field_json(str(resp_raw), "respondent_arguments", [])
            if respondent_arguments:
                merged["respondent_arguments"] = respondent_arguments

        if not merged.get("court_reasoning"):
            reasoning_raw = self._reasoning_chain.invoke({"document": text_sample})
            court_reasoning = parse_single_field_json(str(reasoning_raw), "court_reasoning", [])
            if court_reasoning:
                merged["court_reasoning"] = court_reasoning

        merged["outcome_normalized"] = normalize_outcome(
            str(merged.get("final_decision", "")),
            str(merged.get("outcome_normalized", "")),
        )
        return merged

    def run_from_pdf(self, pdf_path: str | Path, document_id: Optional[str] = None) -> Dict[str, Any]:
        raw_text = load_pdf_text(pdf_path)
        doc_id = document_id or Path(pdf_path).stem
        return self.run_from_text(raw_text, document_id=doc_id)

    def run_from_text(self, raw_text: str, document_id: str = "latest") -> Dict[str, Any]:
        try:
            cleaned = self.parser.parse(raw_text)
        except Exception as e:
            raise RuntimeError(f"Parsing failed: {e}") from e
        self._last_cleaned_text = cleaned

        chunks = self.chunker.chunk(cleaned)
        text_sample = cleaned[:6000] if len(cleaned) > 6000 else cleaned

        try:
            structure_raw = self._structure_chain.invoke({"document": text_sample})
            metadata = parse_structure_output(
                structure_raw if isinstance(structure_raw, str) else str(structure_raw)
            )
        except Exception as e:
            raise RuntimeError(f"Structure extraction failed (LangChain): {e}") from e
        try:
            metadata = self._merge_specialized_fields(metadata, cleaned)
        except Exception as e:
            self.logger.warning("Specialized merge failed: %s", e)
        metadata = self._fallback_metadata_from_text(metadata, cleaned)

        try:
            timeline_raw = self._timeline_chain.invoke({"document": text_sample})
            timeline = parse_timeline_output(
                timeline_raw if isinstance(timeline_raw, str) else str(timeline_raw)
            )
        except Exception as e:
            raise RuntimeError(f"Timeline extraction failed (LangChain): {e}") from e
        if not timeline:
            timeline = self._fallback_timeline_from_text(cleaned)

        if self.enable_contradiction_detection:
            try:
                # Keep contradiction checks bounded; full pairwise NLI is expensive on CPU.
                contradictions = self.contradiction_agent.detect(cleaned, max_pairs=24)
            except Exception as e:
                raise RuntimeError(f"Contradiction detection failed: {e}") from e
        else:
            contradictions = []

        try:
            self.vector_store.add_texts(chunks)
        except Exception as e:
            raise RuntimeError(f"Embedding/indexing failed: {e}") from e

        normalized_record = normalize_case_record(document_id, metadata)
        self.case_store.upsert_case(normalized_record)
        self._last_provenance = [
            build_provenance("Qwen/Qwen2-7B-Instruct", "structure"),
            build_provenance("Qwen/Qwen2-7B-Instruct", "issue"),
            build_provenance("Qwen/Qwen2-7B-Instruct", "petitioner_arguments"),
            build_provenance("Qwen/Qwen2-7B-Instruct", "respondent_arguments"),
            build_provenance("Qwen/Qwen2-7B-Instruct", "court_reasoning"),
        ]
        self._last_warnings = validate_extraction(metadata)
        for warning in self._last_warnings:
            self.logger.warning("Extraction warning: %s", warning)

        self._last_metadata = metadata
        self._last_timeline = timeline
        self._last_contradictions = contradictions

        return {
            "cleaned_text": cleaned,
            "metadata": metadata,
            "timeline": timeline,
            "contradictions": contradictions,
            "chunks_indexed": len(chunks),
            "provenance": self._last_provenance,
            "warnings": self._last_warnings,
        }

    def ask(self, question: str) -> str:
        try:
            result = self._rag_chain.invoke({"question": question, "input": question})
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            return f"Error generating answer: {e}"

    def get_last_metadata(self) -> Optional[Dict[str, Any]]:
        return self._last_metadata

    def get_last_timeline(self) -> Optional[List[Dict[str, Any]]]:
        return self._last_timeline

    def get_last_contradictions(self) -> Optional[List[Dict[str, Any]]]:
        return self._last_contradictions

    def get_last_cleaned_text(self) -> Optional[str]:
        return self._last_cleaned_text

    def get_last_provenance(self) -> List[Dict[str, str]]:
        return self._last_provenance

    def get_last_warnings(self) -> List[str]:
        return self._last_warnings

    def save_index(self, path: Optional[Path] = None) -> None:
        self.vector_store.save(path)

    def load_index(self, path: Optional[Path] = None) -> None:
        self.vector_store.load(path)
