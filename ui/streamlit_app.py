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
require_gpu = st.checkbox(
    "Require GPU runtime",
    value=False,
    help="When enabled, analysis fails fast unless CUDA-enabled PyTorch is available.",
)
enable_llm_enrichment = st.checkbox(
    "Deep LLM enrichment (slower, can improve extraction)",
    value=False,
    help="Runs additional generative extraction passes for issue/arguments/reasoning/timeline.",
)
run_contradiction_detection = st.checkbox(
    "Run contradiction detection (slower)",
    value=False,
    help="This loads a separate NLI model and can add significant latency on CPU.",
)
st.caption("First run can still be slow while models are downloaded and initialized.")

try:
    import torch

    if torch.cuda.is_available():
        st.caption(f"Runtime device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        st.caption("Runtime device: CPU (CUDA not available in this Python env)")
except Exception:
    st.caption("Runtime device: unknown")


def _normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _render_field(label: str, value: str):
    clean = str(value).strip() if value is not None else ""
    st.markdown(f"**{label}:** {clean if clean else 'Not available'}")


def _render_list_section(title: str, items):
    data = _normalize_to_list(items)
    st.markdown(f"**{title}**")
    if not data:
        st.caption("Not available")
        return
    for item in data:
        st.markdown(f"- {item}")


def _render_metadata(metadata: dict):
    st.subheader("Case Metadata")
    left, right = st.columns(2)
    with left:
        _render_field("Case Name", metadata.get("case_name", ""))
        _render_field("Court", metadata.get("court", ""))
        _render_field("Judge", metadata.get("judge", ""))
        _render_field("Petitioner", metadata.get("petitioner", ""))
        _render_field("Respondent", metadata.get("respondent", ""))
    with right:
        _render_field("Main Issue", metadata.get("main_issue", ""))
        _render_field("Final Decision", metadata.get("final_decision", ""))
        _render_field("Outcome", metadata.get("outcome_normalized", ""))

    st.markdown("---")
    _render_list_section("Petitioner Arguments", metadata.get("petitioner_arguments", []))
    _render_list_section("Respondent Arguments", metadata.get("respondent_arguments", []))
    _render_list_section("Court Reasoning", metadata.get("court_reasoning", []))
    _render_list_section("Sections of Law", metadata.get("sections_of_law", []))
    _render_list_section("Precedents", metadata.get("precedents", []))

    with st.expander("View raw metadata JSON", expanded=False):
        st.json(metadata)


def _render_contradictions(contradictions):
    st.subheader("Contradictions")
    if not contradictions:
        st.write("No contradictions detected.")
        return

    st.caption(f"Detected {len(contradictions)} potential contradiction(s).")
    for idx, item in enumerate(contradictions, start=1):
        c1 = str(item.get("statement_1", "")).strip()
        c2 = str(item.get("statement_2", "")).strip()
        confidence = item.get("confidence", "")
        title = f"Contradiction {idx}"
        if confidence != "":
            title += f" (confidence: {confidence})"
        with st.expander(title, expanded=False):
            st.markdown("**Statement 1**")
            st.write(c1 if c1 else "Not available")
            st.markdown("**Statement 2**")
            st.write(c2 if c2 else "Not available")

    with st.expander("View raw contradictions JSON", expanded=False):
        st.json(contradictions)


def _split_user_and_internal_warnings(warnings):
    items = _normalize_to_list(warnings)
    internal = [w for w in items if w.startswith("final_review_")]
    user_visible = [w for w in items if not w.startswith("final_review_")]
    return user_visible, internal


def get_orchestrator(
    enable_contradiction_detection: bool = False,
    require_gpu_runtime: bool = False,
    llm_enrichment: bool = False,
):
    if (
        "orchestrator" not in st.session_state
        or st.session_state.get("orch_enable_contra") != enable_contradiction_detection
        or st.session_state.get("orch_require_gpu") != require_gpu_runtime
        or st.session_state.get("orch_llm_enrichment") != llm_enrichment
    ):
        st.session_state.orchestrator = LangChainOrchestrator(
            use_llm_parser=False,
            enable_contradiction_detection=enable_contradiction_detection,
            enable_llm_enrichment=llm_enrichment,
            require_gpu=require_gpu_runtime,
        )
        st.session_state["orch_enable_contra"] = enable_contradiction_detection
        st.session_state["orch_require_gpu"] = require_gpu_runtime
        st.session_state["orch_llm_enrichment"] = llm_enrichment
    return st.session_state.orchestrator

if uploaded is not None and analyze_clicked:
    with st.spinner("Analyzing document (parsing, structure, timeline, contradictions, indexing)..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            orch = get_orchestrator(
                enable_contradiction_detection=run_contradiction_detection,
                require_gpu_runtime=require_gpu,
                llm_enrichment=enable_llm_enrichment,
            )
            result = orch.run_from_pdf(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            st.session_state["analysis"] = result
            st.session_state["analyzed"] = True
            elapsed = int(result.get("processing_ms", 0))
            st.success(f"Analysis complete in {elapsed} ms.")
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
    reviewed = analysis.get("final_review", {}) if isinstance(analysis, dict) else {}
    metadata_view = reviewed.get("metadata", analysis.get("metadata", {}))
    timeline_view = reviewed.get("timeline", analysis.get("timeline", []))
    contradictions_view = reviewed.get("contradictions", analysis.get("contradictions", []))
    warnings_view = reviewed.get("warnings", analysis.get("warnings", []))
    headline = reviewed.get("headline", {})
    summary = reviewed.get("summary", [])

    if headline:
        st.subheader("Case Snapshot")
        col1, col2, col3 = st.columns(3)
        with col1:
            _render_field("Case", headline.get("case_name", ""))
        with col2:
            _render_field("Court", headline.get("court", ""))
        with col3:
            _render_field("Outcome", headline.get("outcome", ""))

    if summary:
        st.subheader("Quick Summary")
        for bullet in summary:
            st.markdown(f"- {bullet}")

    _render_metadata(metadata_view)
    user_warnings, internal_warnings = _split_user_and_internal_warnings(warnings_view)
    if user_warnings:
        st.warning("Extraction warnings: " + " | ".join(user_warnings))
    if internal_warnings:
        with st.expander("Internal review notes", expanded=False):
            for note in internal_warnings:
                st.caption(note)
    with st.expander("Provenance", expanded=False):
        st.json(analysis.get("provenance", []))

    st.subheader("Timeline")
    if timeline_view:
        st.table(timeline_view)
    else:
        st.write("No timeline events extracted.")

    _render_contradictions(contradictions_view)

    # Step 4 — Ask Questions
    st.subheader("Ask a question about this case")
    question = st.text_input(
        "Ask a question about this case",
        key="question_input",
        placeholder="e.g. What was the court's reasoning?",
    )
    if question:
        orch = get_orchestrator(
            enable_contradiction_detection=run_contradiction_detection,
            require_gpu_runtime=require_gpu,
            llm_enrichment=enable_llm_enrichment,
        )
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
        orch = get_orchestrator(
            enable_contradiction_detection=run_contradiction_detection,
            require_gpu_runtime=require_gpu,
            llm_enrichment=enable_llm_enrichment,
        )
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
