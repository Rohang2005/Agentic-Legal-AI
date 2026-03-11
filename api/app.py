"""
FastAPI backend for the Legal Agent System.

Endpoints: POST /upload_document, POST /analyze_document, POST /ask_question.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.langchain_orchestrator import LangChainOrchestrator

app = FastAPI(
    title="Legal Agent System API",
    description="Multi-agent AI for analyzing legal judgments",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory orchestrator and session storage (use Redis/DB in production)
ORCHESTRATOR: Optional[LangChainOrchestrator] = None
UPLOAD_DIR = Path(tempfile.gettempdir()) / "legal_agent_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_orchestrator() -> LangChainOrchestrator:
    global ORCHESTRATOR
    if ORCHESTRATOR is None:
        ORCHESTRATOR = LangChainOrchestrator(use_llm_parser=False)
    return ORCHESTRATOR


class AnalyzeResponse(BaseModel):
    metadata: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/upload_document", summary="Upload a PDF document")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a legal judgment PDF. The file is stored temporarily; call /analyze_document
    with the returned document_id to run the analysis pipeline.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    doc_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{doc_id}.pdf"
    try:
        with path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    return {"document_id": doc_id, "filename": file.filename}


@app.post("/analyze_document", response_model=AnalyzeResponse, summary="Analyze uploaded document")
async def analyze_document(document_id: str = Form(...)) -> AnalyzeResponse:
    """
    Run the full analysis pipeline on an uploaded document (by document_id from /upload_document).
    Returns structured metadata, timeline, and contradictions.
    """
    path = UPLOAD_DIR / f"{document_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found. Upload first.")
    orch = get_orchestrator()
    try:
        result = orch.run_from_pdf(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AnalyzeResponse(
        metadata=result["metadata"],
        timeline=result["timeline"],
        contradictions=result["contradictions"],
        chunks_indexed=result["chunks_indexed"],
    )


@app.post("/ask_question", response_model=AskResponse, summary="Ask a question about the document")
async def ask_question(body: AskRequest) -> AskResponse:
    """
    Ask a natural language question about the analyzed document. Uses RAG over the indexed chunks.
    """
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    orch = get_orchestrator()
    answer = orch.ask(body.question.strip())
    return AskResponse(answer=answer)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
