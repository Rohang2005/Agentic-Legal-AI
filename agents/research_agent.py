"""
Research Agent - RAG-based question answering over legal documents.

Uses Llama-3.1-8B-Instruct with retrieved chunks from FAISS for context.
"""

import sys
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.llm_loader import LLMLoader
from retrieval.vector_store import LegalVectorStore


class ResearchAgent:
    """
    Answers user questions about the document using RAG:
    retrieve relevant chunks from vector store, then generate answer with LLM.
    Model: meta-llama/Llama-3.1-8B-Instruct (requires HuggingFace access)
    """

    RESEARCH_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

    def __init__(
        self,
        vector_store: LegalVectorStore,
        llm_loader: Optional[LLMLoader] = None,
        top_k: int = 5,
        max_context_chars: int = 4000,
    ):
        """
        Args:
            vector_store: FAISS-backed store of document chunks.
            llm_loader: Optional LLM for generation (default: Llama-3.1-8B).
            top_k: Number of chunks to retrieve per query.
            max_context_chars: Max characters of context to pass to LLM.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self._llm = llm_loader

    def _get_llm(self) -> LLMLoader:
        if self._llm is None:
            self._llm = LLMLoader(
                self.RESEARCH_MODEL_ID,
                max_new_tokens=512,
                temperature=0.3,
            )
        return self._llm

    def _build_context(self, query: str) -> str:
        """Retrieve relevant chunks and format as context string."""
        if self.vector_store.is_empty:
            return ""
        results = self.vector_store.similarity_search(query, k=self.top_k)
        parts = []
        total = 0
        for text, _score in results:
            if total + len(text) > self.max_context_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n\n---\n\n".join(parts)

    def ask(self, question: str) -> str:
        """
        Answer a question about the document using RAG.

        Args:
            question: User question (e.g. "What was the court's reasoning?").

        Returns:
            Generated answer string.
        """
        context = self._build_context(question)
        if not context:
            return "No document has been indexed yet. Please upload and analyze a judgment first."
        prompt = (
            "You are a legal assistant. Answer the following question based ONLY on the provided document excerpts. "
            "If the excerpts do not contain enough information, say so. Be concise and accurate.\n\n"
            "Document excerpts:\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        llm = self._get_llm()
        try:
            return llm.generate(prompt, max_new_tokens=512).strip()
        except Exception as e:
            return f"Error generating answer: {e}"
