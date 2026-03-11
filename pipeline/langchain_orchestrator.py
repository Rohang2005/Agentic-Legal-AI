"""
LangChain-based pipeline orchestrator.

Uses LangChain chains for structure extraction, timeline extraction, and RAG Q&A
while keeping the same interface as LegalPipelineOrchestrator.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.parser_agent import ParserAgent
from agents.contradiction_agent import ContradictionAgent
from models.embedding_model import EmbeddingModel
from retrieval.vector_store import LegalVectorStore
from utils.text_chunker import TextChunker
from utils.pdf_loader import load_pdf_text

from .langchain_components import (
    get_structure_chain,
    get_timeline_chain,
    get_rag_chain,
    parse_structure_output,
    parse_timeline_output,
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
        vector_store: Optional[LegalVectorStore] = None,
        index_path: Optional[Path] = None,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.parser = ParserAgent(use_llm=use_llm_parser)
        self.contradiction_agent = ContradictionAgent()
        self.embedding_model = EmbeddingModel()
        self.vector_store = vector_store or LegalVectorStore(
            embedding_model=self.embedding_model,
            index_path=index_path,
        )

        self._structure_chain = get_structure_chain()
        self._timeline_chain = get_timeline_chain()
        self._rag_chain = get_rag_chain(self.vector_store)

        self._last_metadata: Optional[Dict[str, Any]] = None
        self._last_timeline: Optional[List[Dict[str, Any]]] = None
        self._last_contradictions: Optional[List[Dict[str, Any]]] = None
        self._last_cleaned_text: Optional[str] = None

    def run_from_pdf(self, pdf_path: str | Path) -> Dict[str, Any]:
        raw_text = load_pdf_text(pdf_path)
        return self.run_from_text(raw_text)

    def run_from_text(self, raw_text: str) -> Dict[str, Any]:
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
            timeline_raw = self._timeline_chain.invoke({"document": text_sample})
            timeline = parse_timeline_output(
                timeline_raw if isinstance(timeline_raw, str) else str(timeline_raw)
            )
        except Exception as e:
            raise RuntimeError(f"Timeline extraction failed (LangChain): {e}") from e

        try:
            contradictions = self.contradiction_agent.detect(cleaned)
        except Exception as e:
            raise RuntimeError(f"Contradiction detection failed: {e}") from e

        try:
            self.vector_store.add_texts(chunks)
        except Exception as e:
            raise RuntimeError(f"Embedding/indexing failed: {e}") from e

        self._last_metadata = metadata
        self._last_timeline = timeline
        self._last_contradictions = contradictions

        return {
            "cleaned_text": cleaned,
            "metadata": metadata,
            "timeline": timeline,
            "contradictions": contradictions,
            "chunks_indexed": len(chunks),
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

    def save_index(self, path: Optional[Path] = None) -> None:
        self.vector_store.save(path)

    def load_index(self, path: Optional[Path] = None) -> None:
        self.vector_store.load(path)
