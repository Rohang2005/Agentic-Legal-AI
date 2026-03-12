"""
Pipeline Orchestrator - Coordinates the full legal document analysis flow.

Flow: PDF/Text -> Parser -> Chunk -> Structure -> Timeline -> Contradiction -> Embeddings/FAISS -> Research (RAG).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.parser_agent import ParserAgent
from agents.structure_agent import StructureAgent
from agents.timeline_agent import TimelineAgent
from agents.contradiction_agent import ContradictionAgent
from agents.research_agent import ResearchAgent
from models.embedding_model import EmbeddingModel
from retrieval.vector_store import LegalVectorStore
from retrieval.case_store import CaseStore
from utils.text_chunker import TextChunker
from utils.pdf_loader import load_pdf_text
from utils.legal_normalizer import normalize_case_record
from utils.extraction_guardrails import build_provenance, validate_extraction


class LegalPipelineOrchestrator:
    """
    Runs the full pipeline: parse PDF, extract structure/timeline/contradictions,
    index chunks in FAISS, and expose research agent for Q&A.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_llm_parser: bool = False,
        use_llm_extractors: bool = False,
        vector_store: Optional[LegalVectorStore] = None,
        index_path: Optional[Path] = None,
        case_store: Optional[CaseStore] = None,
    ):
        """
        Args:
            chunk_size: Character size for document chunks.
            chunk_overlap: Overlap between chunks.
            use_llm_parser: Whether to use Mistral for parsing (slower, more accurate).
            vector_store: Optional shared vector store. If None, creates one.
            index_path: Optional path to persist FAISS index.
        """
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.parser = ParserAgent(use_llm=use_llm_parser)
        self.structure_agent = StructureAgent(use_llm=use_llm_extractors)
        self.timeline_agent = TimelineAgent(use_llm=use_llm_extractors)
        self.contradiction_agent = ContradictionAgent()
        self.embedding_model = EmbeddingModel()
        self.vector_store = vector_store or LegalVectorStore(
            embedding_model=self.embedding_model,
            index_path=index_path,
        )
        self.research_agent = ResearchAgent(vector_store=self.vector_store)
        self.case_store = case_store or CaseStore()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._last_metadata: Optional[Dict[str, Any]] = None
        self._last_timeline: Optional[List[Dict[str, Any]]] = None
        self._last_contradictions: Optional[List[Dict[str, Any]]] = None
        self._last_cleaned_text: Optional[str] = None
        self._last_provenance: List[Dict[str, str]] = []
        self._last_warnings: List[str] = []

    def run_from_pdf(self, pdf_path: str | Path, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full pipeline from a PDF file.

        Args:
            pdf_path: Path to the judgment PDF.

        Returns:
            Dict with keys: cleaned_text, metadata, timeline, contradictions, chunks_indexed.
        """
        raw_text = load_pdf_text(pdf_path)
        doc_id = document_id or Path(pdf_path).stem
        return self.run_from_text(raw_text, document_id=doc_id)

    def run_from_text(self, raw_text: str, document_id: str = "latest") -> Dict[str, Any]:
        """
        Run the full pipeline from raw document text.

        Args:
            raw_text: Raw text of the judgment.

        Returns:
            Dict with keys: cleaned_text, metadata, timeline, contradictions, chunks_indexed.
        """
        try:
            cleaned = self.parser.parse(raw_text)
        except Exception as e:
            raise RuntimeError(f"Parsing failed: {e}") from e
        self._last_cleaned_text = cleaned

        chunks = self.chunker.chunk(cleaned)
        try:
            metadata = self.structure_agent.extract(cleaned)
        except Exception as e:
            raise RuntimeError(f"Structure extraction failed (e.g. model download or GPU): {e}") from e
        try:
            timeline = self.timeline_agent.extract(cleaned)
        except Exception as e:
            raise RuntimeError(f"Timeline extraction failed: {e}") from e
        try:
            contradictions = self.contradiction_agent.detect(cleaned)
        except Exception as e:
            self.logger.warning("Contradiction detection unavailable: %s", e)
            contradictions = []
        try:
            self.vector_store.add_texts(chunks)
        except Exception as e:
            raise RuntimeError(f"Embedding/indexing failed: {e}") from e
        normalized_record = normalize_case_record(document_id, metadata)
        self.case_store.upsert_case(normalized_record)

        self._last_provenance = [
            build_provenance("Qwen/Qwen2-1.5B-Instruct", "structure"),
            build_provenance("Qwen/Qwen2-1.5B-Instruct", "timeline"),
            build_provenance("cross-encoder/nli-deberta-v3-base", "contradiction"),
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
        """Delegate to research agent for RAG-based answer."""
        return self.research_agent.ask(question)

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
        """Persist FAISS index and metadata."""
        self.vector_store.save(path)

    def load_index(self, path: Optional[Path] = None) -> None:
        """Load FAISS index and metadata (e.g. after restart)."""
        self.vector_store.load(path)
