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
from utils.legal_normalizer import parse_nl_search_query

st.set_page_config(page_title="Agentic AI Legal Case Analyzer", page_icon="⚖️", layout="wide")

# Title
st.title("Agentic AI Legal Case Analyzer")

# Step 1 — Upload PDF
st.subheader("Step 1 — Upload PDF")
uploaded = st.file_uploader("Upload a legal judgment PDF", type=["pdf"], key="pdf_upload")

# Step 2 — Analyze Document
st.subheader("Step 2 — Analyze Document")
analyze_clicked = st.button("**Analyze Case**", type="primary")
run_contradiction_detection = st.checkbox(
    "Run contradiction detection (slower)",
    value=False,
    help="This loads a separate NLI model and can add significant latency on CPU.",
)
st.caption("First run can still be slow while models are downloaded and initialized.")

def get_orchestrator(enable_contradiction_detection: bool = False):
    if (
        "orchestrator" not in st.session_state
        or st.session_state.get("orch_enable_contra") != enable_contradiction_detection
    ):
        st.session_state.orchestrator = LangChainOrchestrator(
            use_llm_parser=False,
            enable_contradiction_detection=enable_contradiction_detection,
        )
        st.session_state["orch_enable_contra"] = enable_contradiction_detection
    return st.session_state.orchestrator

if uploaded is not None and analyze_clicked:
    with st.spinner("Analyzing document (parsing, structure, timeline, contradictions, indexing)..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            orch = get_orchestrator(enable_contradiction_detection=run_contradiction_detection)
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
    warnings = analysis.get("warnings", [])
    if warnings:
        st.warning("Extraction warnings: " + " | ".join(warnings))
    with st.expander("Provenance", expanded=False):
        st.json(analysis.get("provenance", []))

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
        orch = get_orchestrator(enable_contradiction_detection=run_contradiction_detection)
        with st.spinner("Searching document and generating answer..."):
            answer = orch.ask(question)
        st.markdown("**Answer:**")
        st.write(answer)

    st.subheader("Search Across Analyzed Cases")
    nl_query = st.text_input(
        "Natural language filter query",
        key="search_nl_query",
        placeholder="e.g. Show cases citing Section 420 IPC with acquittal outcomes.",
    )
    parsed = parse_nl_search_query(nl_query) if nl_query else {"act": "", "section": "", "outcome": ""}
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        act = st.text_input("Act", value=parsed.get("act", ""), key="search_act")
    with col2:
        section = st.text_input("Section", value=parsed.get("section", ""), key="search_section")
    with col3:
        outcome = st.text_input("Outcome", value=parsed.get("outcome", ""), key="search_outcome")
    with col4:
        court = st.text_input("Court", value="", key="search_court")

    if st.button("Search Cases"):
        orch = get_orchestrator(enable_contradiction_detection=run_contradiction_detection)
        matches = orch.case_store.search_cases(
            act=act,
            section=section,
            outcome=outcome,
            court=court,
            limit=100,
        )
        st.write(f"Found {len(matches)} case(s).")
        if matches:
            rows = []
            for item in matches:
                rows.append(
                    {
                        "document_id": item.get("document_id", ""),
                        "case_name": item.get("case_name", ""),
                        "court": item.get("court", ""),
                        "outcome": item.get("outcome_normalized", ""),
                        "sections": ", ".join(
                            f"{x.get('act', '')} {x.get('section', '')}".strip()
                            for x in item.get("sections_normalized", [])
                        ),
                        "main_issue": item.get("main_issue", ""),
                    }
                )
            st.table(rows)
else:
    st.info("Upload a PDF and click **Analyze Case** to get started.")
