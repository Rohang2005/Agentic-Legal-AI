# Legal Agent System

A multi-agent AI system for analyzing legal case judgments. It transforms long, unstructured legal PDFs into **structured legal intelligence** and supports **interactive question answering** using only open-source HuggingFace models.

## System Architecture

```
PDF Upload
    ↓
Parser Agent (Mistral-7B or rule-based) — clean & normalize text
    ↓
Text Chunking (overlapping chunks)
    ↓
Structure Agent (Qwen2-7B) — case metadata (parties, court, sections, precedents, decision)
    ↓
Timeline Agent (Qwen2-7B) — chronological events
    ↓
Contradiction Agent (DeBERTa-v3-mnli) — NLI-based contradiction detection
    ↓
Embeddings (BGE-large-en-v1.5) + FAISS index
    ↓
Research Agent (Llama-3.1-8B) — RAG question answering
```

### Models (open-source; Llama requires HuggingFace access)

| Component | HuggingFace model | Purpose |
|-----------|-------------------|--------|
| **Parser Agent** | `mistralai/Mistral-7B-Instruct-v0.2` | Optional LLM to clean and normalize PDF-extracted text (whitespace, OCR-style fixes). Default pipeline uses rule-based parsing only. |
| **Structure Agent** | `Qwen/Qwen2-7B-Instruct` | Extract structured case metadata: case name, court, judge, petitioner, respondent, sections of law, precedents, final decision (output as JSON). |
| **Timeline Agent** | `Qwen/Qwen2-7B-Instruct` | Extract a chronological list of events from the judgment (output: list of `{date, event}`). |
| **Contradiction Agent** | `microsoft/deberta-v3-large-mnli` | Natural language inference (NLI): split document into claims, compare pairs, return statement pairs classified as contradictory with confidence scores. |
| **Research Agent** | `meta-llama/Llama-3.1-8B-Instruct` | RAG: retrieve relevant chunks from the FAISS index, then generate answers to user questions about the case. Requires HuggingFace access and `HF_TOKEN` in `config/hf_config.py`. |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | Encode document chunks into dense vectors for FAISS indexing and similarity search (retrieval for RAG). |

Mistral, Qwen, DeBERTa, and BGE are open on HuggingFace. Llama-3.1-8B-Instruct requires accepting the model terms and using a HuggingFace token (e.g. in `config/hf_config.py`).

### LangChain integration

The **default** pipeline uses **LangChain** for orchestration:

- **Structure** and **timeline** extraction are implemented as LangChain LCEL chains (PromptTemplate | LLM runnable).
- **RAG** (ask questions) uses a LangChain retrieval chain: retriever over the FAISS index → context + question → LLM.
- The existing **parser**, **contradiction agent**, **chunker**, and **vector store** are unchanged; only the LLM-calling steps use LangChain.

To use the original non-LangChain orchestrator, import `LegalPipelineOrchestrator` from `pipeline.orchestrator` instead of `LangChainOrchestrator` from `pipeline.langchain_orchestrator` in the API and Streamlit app.

---

## Project Structure

```
legal_agent_system/
├── agents/
│   ├── parser_agent.py       # Text cleaning (Mistral or rule-based)
│   ├── structure_agent.py    # Metadata extraction (Qwen2-7B)
│   ├── timeline_agent.py    # Event timeline (Qwen2-7B)
│   ├── contradiction_agent.py # NLI contradiction detection (DeBERTa)
│   └── research_agent.py    # RAG Q&A (Llama-3.1-8B)
├── models/
│   ├── llm_loader.py        # HuggingFace causal LM loader (optional 8-bit)
│   └── embedding_model.py   # BGE sentence embeddings
├── pipeline/
│   ├── orchestrator.py       # End-to-end pipeline (original)
│   ├── langchain_components.py  # LangChain chains (structure, timeline, RAG)
│   └── langchain_orchestrator.py # LangChain-based orchestrator (default)
├── retrieval/
│   └── vector_store.py      # FAISS + BGE
├── utils/
│   ├── pdf_loader.py        # PyPDF text extraction
│   └── text_chunker.py      # Overlapping chunking
├── api/
│   └── app.py               # FastAPI backend
├── ui/
│   └── streamlit_app.py     # Streamlit demo UI
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- ~16GB+ RAM recommended (or GPU for faster inference)
- For **Llama-3.1-8B-Instruct**: accept the model terms on HuggingFace and set `HF_TOKEN` in `config/hf_config.py` (or use `huggingface-cli login`). Mistral, Qwen, DeBERTa, and BGE are open; token optional for them.

---

## Installation

```bash
cd legal_agent_system
pip install -r requirements.txt
```

Optional: for 8-bit loading to reduce VRAM (in `models/llm_loader.py` you can pass `load_in_8bit=True` when instantiating `LLMLoader`).

---

## How to Run

### 1. FastAPI backend

From the `legal_agent_system` directory:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

- **POST /upload_document** — upload a PDF; returns `document_id`
- **POST /analyze_document** — body: `document_id=<id>`; returns metadata, timeline, contradictions, chunks_indexed
- **POST /ask_question** — body: `{"question": "..."}`; returns RAG answer
- **POST /search_cases** — structured case search with filters (`act`, `section`, `outcome`, `court`) or natural-language query text
- **GET /health** — health check

### 2. Streamlit demo UI

From the `legal_agent_system` directory:

```bash
streamlit run ui/streamlit_app.py
```

In the UI you can:

1. Upload a legal judgment PDF  
2. Click **Analyze document**  
3. View **Metadata**, **Timeline**, and **Contradictions**  
4. Use the **Chat** tab to ask questions (e.g. “What was the court’s reasoning?”, “Which IPC sections were cited?”, “Summarize the judgment”)

---

## How to Test the Pipeline

### From Python (orchestrator only)

```python
from pathlib import Path
from pipeline.orchestrator import LegalPipelineOrchestrator

orch = LegalPipelineOrchestrator(use_llm_parser=False)
result = orch.run_from_pdf(Path("path/to/judgment.pdf"))

print(result["metadata"])
print(result["timeline"])
print(result["contradictions"])
print(result["chunks_indexed"])

answer = orch.ask("What was the court's reasoning?")
print(answer)
```

### With FastAPI

1. Upload: `curl -X POST -F "file=@judgment.pdf" http://localhost:8000/upload_document`  
2. Analyze: `curl -X POST -F "document_id=<id>" http://localhost:8000/analyze_document`  
3. Ask: `curl -X POST -H "Content-Type: application/json" -d '{"question":"What was the court\'s reasoning?"}' http://localhost:8000/ask_question`
4. Search: `curl -X POST -H "Content-Type: application/json" -d '{"act":"IPC","section":"420","outcome":"acquittal"}' http://localhost:8000/search_cases`

### Extraction quality checks

Run sample guardrail checks:

```bash
python evaluation/run_extraction_checks.py
```

---

## Model Notes

- **Open models**: Mistral-7B, Qwen2-7B, DeBERTa-mnli, and BGE are open on HuggingFace. **Llama-3.1-8B-Instruct** requires accepting the model terms and a HuggingFace token in `config/hf_config.py`.
- **Memory**: 7B/8B models are RAM/VRAM heavy. A GPU with 8GB+ VRAM is recommended; use `load_in_8bit=True` in `LLMLoader` to reduce usage.
- **Parser**: By default the pipeline uses rule-based parsing only (`use_llm_parser=False`). Set `use_llm_parser=True` to use Mistral for LLM-based parsing (slower, potentially better normalization).

---

## License

Use of each HuggingFace model is subject to its respective license (Mistral, Qwen, Meta Llama, Microsoft DeBERTa, BAAI BGE).
