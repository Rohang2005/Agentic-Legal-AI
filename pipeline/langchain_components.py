"""
LangChain components for the Legal Agent System.

Provides chains (structure, timeline, RAG) and a retriever wrapper using LangChain.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from models.llm_loader import LLMLoader
from retrieval.vector_store import LegalVectorStore


# --- Helper: extract prompt string from LCEL payload (string or PromptValue) ---

def _prompt_to_str(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if hasattr(payload, "to_string"):
        return payload.to_string()
    if isinstance(payload, dict):
        return payload.get("prompt") or payload.get("text") or payload.get("input") or str(payload)
    return str(payload)


def get_structure_chain(llm_loader: Optional[LLMLoader] = None):
    """LangChain chain: document text -> structured metadata (JSON)."""
    if llm_loader is None:
        llm_loader = LLMLoader("Qwen/Qwen2-7B-Instruct", max_new_tokens=1024, temperature=0.2)
    prompt = PromptTemplate.from_template(
        "Extract legal case metadata from the following judgment text. "
        "Respond with a single JSON object only, no other text. Use this exact structure:\n"
        '{{"case_name": "", "court": "", "judge": "", "petitioner": "", "respondent": "", '
        '"sections_of_law": [], "precedents": [], "final_decision": ""}}\n\n'
        "Document:\n{document}"
    )

    def _generate(x: Any) -> str:
        return llm_loader.generate(_prompt_to_str(x), max_new_tokens=1024)

    runnable = RunnableLambda(_generate)
    chain = prompt | runnable
    return chain


def get_timeline_chain(llm_loader: Optional[LLMLoader] = None):
    """LangChain chain: document text -> timeline list (JSON array)."""
    if llm_loader is None:
        llm_loader = LLMLoader("Qwen/Qwen2-7B-Instruct", max_new_tokens=1024, temperature=0.2)
    prompt = PromptTemplate.from_template(
        "From the following legal judgment text, extract a chronological timeline of events. "
        "Respond with a JSON array only. Each element must have 'date' and 'event' keys. "
        "Example: [{{\"date\": \"2020-01-15\", \"event\": \"Filing of petition\"}}, ...]\n\n"
        "Document:\n{document}"
    )

    def _generate(x: Any) -> str:
        return llm_loader.generate(_prompt_to_str(x), max_new_tokens=1024)

    runnable = RunnableLambda(_generate)
    chain = prompt | runnable
    return chain


def parse_structure_output(raw: str) -> Dict[str, Any]:
    """Parse LLM output to structure dict; fallback to empty schema."""
    import json
    import re
    schema = {
        "case_name": "", "court": "", "judge": "", "petitioner": "", "respondent": "",
        "sections_of_law": [], "precedents": [], "final_decision": "",
    }
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start:end])
            for k in schema:
                if k in data:
                    v = data[k]
                    if k in ("sections_of_law", "precedents") and not isinstance(v, list):
                        schema[k] = [v] if isinstance(v, str) else []
                    else:
                        schema[k] = v
        except json.JSONDecodeError:
            pass
    return schema


def parse_timeline_output(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM output to list of {date, event}."""
    import json
    import re
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            arr = json.loads(raw[start:end])
            if isinstance(arr, list):
                return [{"date": str(x.get("date", "")), "event": str(x.get("event", ""))} for x in arr if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    return []


class LegalRetriever:
    """LangChain-style retriever that wraps LegalVectorStore and returns List[Document]."""

    def __init__(self, vector_store: LegalVectorStore, k: int = 5):
        self.vector_store = vector_store
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        if self.vector_store.is_empty:
            return []
        results = self.vector_store.similarity_search(query, k=self.k)
        return [Document(page_content=text, metadata={"score": score}) for text, score in results]


def get_rag_chain(vector_store: LegalVectorStore, llm_loader: Optional[LLMLoader] = None, top_k: int = 5):
    """Build a RAG chain: question -> retrieve context -> prompt -> LLM -> answer."""
    if llm_loader is None:
        llm_loader = LLMLoader("meta-llama/Llama-3.1-8B-Instruct", max_new_tokens=512, temperature=0.3)
    retriever = LegalRetriever(vector_store, k=top_k)
    prompt = PromptTemplate.from_template(
        "You are a legal assistant. Answer the following question based ONLY on the provided document excerpts. "
        "If the excerpts do not contain enough information, say so. Be concise and accurate.\n\n"
        "Document excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def _rag_step(inputs: Dict[str, Any]) -> str:
        question = inputs.get("question", inputs.get("input", ""))
        if isinstance(question, dict):
            question = question.get("question", question.get("input", ""))
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        if not context:
            return "No document has been indexed yet. Please upload and analyze a judgment first."
        formatted = prompt.format(context=context, question=question)
        return llm_loader.generate(formatted, max_new_tokens=512).strip()

    return RunnableLambda(_rag_step)
