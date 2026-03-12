"""
LangChain-based pipeline orchestrator.

Uses LangChain chains for structure extraction, timeline extraction, and RAG Q&A
while keeping the same interface as LegalPipelineOrchestrator.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import re
import hashlib
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.parser_agent import ParserAgent
from agents.contradiction_agent import ContradictionAgent
from agents.final_review_agent import FinalReviewAgent
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
        enable_llm_enrichment: bool = False,
        require_gpu: bool = False,
        vector_store: Optional[LegalVectorStore] = None,
        index_path: Optional[Path] = None,
        case_store: Optional[CaseStore] = None,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.parser = ParserAgent(use_llm=use_llm_parser)
        self.contradiction_agent = ContradictionAgent()
        self.final_review_agent = FinalReviewAgent(use_llm=True, require_gpu=require_gpu)
        self.require_gpu = require_gpu
        self.embedding_model = EmbeddingModel(require_gpu=require_gpu)
        self.vector_store = vector_store or LegalVectorStore(
            embedding_model=self.embedding_model,
            index_path=index_path,
        )

        # Reuse one loader to avoid repeated heavyweight model initialization per chain.
        self.enable_llm_enrichment = enable_llm_enrichment
        self._llm_loader = LLMLoader(
            "Qwen/Qwen2-1.5B-Instruct",
            require_gpu=require_gpu,
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
        self._last_reviewed_output: Dict[str, Any] = {}
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _base_metadata_schema() -> Dict[str, Any]:
        return {
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
    def _coalesce_metadata(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
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
        return merged

    @staticmethod
    def _dedupe_lines(items: List[str], max_items: int = 8) -> List[str]:
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
        candidates = re.split(r"(?<=[.!?])\s+", text)
        matched: List[str] = []
        for sentence in candidates:
            s = sentence.strip()
            if len(s) < 20:
                continue
            low = s.lower()
            if any(token in low for token in keywords):
                matched.append(s[:260])
                if len(matched) >= max_items:
                    break
        return LangChainOrchestrator._dedupe_lines(matched, max_items=max_items)

    def _build_salient_sample(self, text: str, max_chars: int = 6000) -> str:
        """Compose a compact but informative sample for LLM extraction."""
        if len(text) <= max_chars:
            return text
        head = text[:2200]
        tail = text[-1800:]
        keyword_lines: List[str] = []
        for line in text.splitlines():
            s = line.strip()
            if len(s) < 25:
                continue
            low = s.lower()
            if any(
                token in low
                for token in (
                    "section ",
                    "issue",
                    "petitioner",
                    "respondent",
                    "appellant",
                    "therefore",
                    "held",
                    "order",
                    "judgment",
                    "v.",
                    "vs.",
                    "versus",
                )
            ):
                keyword_lines.append(s)
                if len(keyword_lines) >= 35:
                    break
        mid = "\n".join(keyword_lines)
        sample = f"{head}\n\n{mid}\n\n{tail}".strip()
        return sample[:max_chars]

    def _fallback_metadata_from_text(self, metadata: Dict[str, Any], cleaned_text: str) -> Dict[str, Any]:
        """Best-effort heuristic fill for key fields when LLM JSON is sparse."""
        merged = dict(metadata)
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        first_block = lines[:80]
        tail_block = lines[-120:] if len(lines) > 120 else lines

        if not merged.get("case_name"):
            for line in first_block:
                if re.search(r"\s(v\.|vs\.?|versus)\s", line, flags=re.IGNORECASE):
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

        if not merged.get("precedents"):
            precedent_candidates = re.findall(
                r"\b([A-Z][A-Za-z0-9.&,'()\- ]{2,80}\s+(?:v\.|vs\.?|versus)\s+[A-Z][A-Za-z0-9.&,'()\- ]{2,80})\b",
                cleaned_text,
            )
            merged["precedents"] = self._dedupe_lines(precedent_candidates, max_items=10)

        if not merged.get("main_issue"):
            issue_lines = self._extract_sentences_by_keywords(
                cleaned_text,
                keywords=["issue", "question for consideration", "point for determination", "dispute"],
                max_items=2,
            )
            if issue_lines:
                merged["main_issue"] = issue_lines[0]

        if not merged.get("petitioner_arguments"):
            merged["petitioner_arguments"] = self._extract_sentences_by_keywords(
                cleaned_text,
                keywords=["petitioner submitted", "appellant submitted", "for the petitioner", "learned counsel for the petitioner"],
                max_items=6,
            )

        if not merged.get("respondent_arguments"):
            merged["respondent_arguments"] = self._extract_sentences_by_keywords(
                cleaned_text,
                keywords=["respondent submitted", "for the respondent", "state submitted", "public prosecutor"],
                max_items=6,
            )

        if not merged.get("court_reasoning"):
            merged["court_reasoning"] = self._extract_sentences_by_keywords(
                cleaned_text,
                keywords=["we hold", "it is clear", "therefore", "in view of", "this court finds", "observed that"],
                max_items=8,
            )

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
        text_sample = self._build_salient_sample(document_text, max_chars=8000)

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
        started = time.perf_counter()
        cache_key = hashlib.sha1(raw_text.encode("utf-8", errors="ignore")).hexdigest()
        if cache_key in self._analysis_cache:
            cached = self._analysis_cache[cache_key]
            self._last_metadata = cached.get("metadata")
            self._last_timeline = cached.get("timeline")
            self._last_contradictions = cached.get("contradictions")
            self._last_cleaned_text = cached.get("cleaned_text")
            self._last_provenance = cached.get("provenance", [])
            self._last_warnings = cached.get("warnings", [])
            self._last_reviewed_output = cached.get("final_review", {})
            return dict(cached)

        try:
            cleaned = self.parser.parse(raw_text)
        except Exception as e:
            raise RuntimeError(f"Parsing failed: {e}") from e
        self._last_cleaned_text = cleaned

        chunks = self.chunker.chunk(cleaned)
        text_sample = self._build_salient_sample(cleaned, max_chars=6000)

        metadata = self._fallback_metadata_from_text(self._base_metadata_schema(), cleaned)
        critical_missing = any(
            not str(metadata.get(field, "")).strip()
            for field in ("case_name", "court", "final_decision")
        )

        if self.enable_llm_enrichment and critical_missing:
            try:
                structure_raw = self._structure_chain.invoke({"document": text_sample})
                llm_metadata = parse_structure_output(
                    structure_raw if isinstance(structure_raw, str) else str(structure_raw)
                )
                metadata = self._coalesce_metadata(metadata, llm_metadata)
            except Exception as e:
                self.logger.warning("Structure extraction failed (LangChain): %s", e)

        if self.enable_llm_enrichment:
            try:
                metadata = self._merge_specialized_fields(metadata, cleaned)
            except Exception as e:
                self.logger.warning("Specialized merge failed: %s", e)
        metadata = self._fallback_metadata_from_text(metadata, cleaned)

        timeline = self._fallback_timeline_from_text(cleaned)
        if self.enable_llm_enrichment and len(timeline) < 1:
            try:
                timeline_raw = self._timeline_chain.invoke({"document": text_sample})
                llm_timeline = parse_timeline_output(
                    timeline_raw if isinstance(timeline_raw, str) else str(timeline_raw)
                )
                if llm_timeline:
                    timeline = llm_timeline
            except Exception as e:
                self.logger.warning("Timeline extraction failed (LangChain): %s", e)

        contradiction_warning = ""
        if self.enable_contradiction_detection:
            try:
                # Keep contradiction checks bounded; full pairwise NLI is expensive on CPU.
                contradictions = self.contradiction_agent.detect(cleaned, max_pairs=24)
            except Exception as e:
                self.logger.warning("Contradiction detection unavailable: %s", e)
                contradictions = []
                contradiction_warning = "contradiction_detection_unavailable"
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
        if contradiction_warning:
            self._last_warnings.append(contradiction_warning)
        for warning in self._last_warnings:
            self.logger.warning("Extraction warning: %s", warning)

        self._last_metadata = metadata
        self._last_timeline = timeline
        self._last_contradictions = contradictions
        self._last_reviewed_output = self.final_review_agent.review(
            metadata=metadata,
            timeline=timeline,
            contradictions=contradictions,
            warnings=self._last_warnings,
        )

        result = {
            "cleaned_text": cleaned,
            "metadata": metadata,
            "timeline": timeline,
            "contradictions": contradictions,
            "final_review": self._last_reviewed_output,
            "chunks_indexed": len(chunks),
            "provenance": self._last_provenance,
            "warnings": self._last_warnings,
            "processing_ms": int((time.perf_counter() - started) * 1000),
        }
        self._analysis_cache[cache_key] = dict(result)
        return result

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

    def get_last_reviewed_output(self) -> Dict[str, Any]:
        return self._last_reviewed_output

    def save_index(self, path: Optional[Path] = None) -> None:
        self.vector_store.save(path)

    def load_index(self, path: Optional[Path] = None) -> None:
        self.vector_store.load(path)
