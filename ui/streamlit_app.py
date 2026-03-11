"""
Streamlit UI for the Legal Agent System.

Flow: Upload PDF → Analyze Case → View metadata, timeline, contradictions → Ask questions.
"""

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from pipeline.langchain_orchestrator import LangChainOrchestrator

st.set_page_config(page_title="Agentic AI Legal Case Analyzer", page_icon="⚖️", layout="wide")

# Title
st.title("Agentic AI Legal Case Analyzer")

# Step 1 — Upload PDF
st.subheader("Step 1 — Upload PDF")
uploaded = st.file_uploader("Upload a legal judgment PDF", type=["pdf"], key="pdf_upload")

# Step 2 — Analyze Document
st.subheader("Step 2 — Analyze Document")
analyze_clicked = st.button("**Analyze Case**", type="primary")

def get_orchestrator():
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = LangChainOrchestrator(use_llm_parser=False)
    return st.session_state.orchestrator

if uploaded is not None and analyze_clicked:
    with st.spinner("Analyzing document (parsing, structure, timeline, contradictions, indexing)..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            orch = get_orchestrator()
            result = orch.run_from_pdf(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            st.session_state["analysis"] = result
            st.session_state["analyzed"] = True
            st.success("Analysis complete.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            with st.expander("Show full error details", expanded=False):
                st.code(traceback.format_exc())
            st.session_state["analyzed"] = False
else:
    if "analyzed" not in st.session_state:
        st.session_state["analyzed"] = False

# Step 3 — Show Results (after analysis)
if st.session_state.get("analyzed") and "analysis" in st.session_state:
    analysis = st.session_state["analysis"]

    st.subheader("Case Metadata")
    st.json(analysis.get("metadata", {}))

    st.subheader("Timeline")
    timeline = analysis.get("timeline", [])
    if timeline:
        st.table(timeline)
    else:
        st.write("No timeline events extracted.")

    st.subheader("Contradictions")
    contradictions = analysis.get("contradictions", [])
    if contradictions:
        st.write(contradictions)
    else:
        st.write("No contradictions detected.")

    # Step 4 — Ask Questions
    st.subheader("Ask a question about this case")
    question = st.text_input(
        "Ask a question about this case",
        key="question_input",
        placeholder="e.g. What was the court's reasoning?",
    )
    if question:
        orch = get_orchestrator()
        with st.spinner("Searching document and generating answer..."):
            answer = orch.ask(question)
        st.markdown("**Answer:**")
        st.write(answer)
else:
    st.info("Upload a PDF and click **Analyze Case** to get started.")
