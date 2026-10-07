"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from finrag import config
from finrag.embeddings import get_embedder
from finrag.graph import FinanceRAGAgent, load_agent
from finrag.ingest import build_index, discover
from finrag.llm import get_llm
from finrag.rerank import get_reranker
from finrag.tools import load_financials

st.set_page_config(page_title="Agentic Finance RAG", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Building the vector index...")
def get_agent() -> FinanceRAGAgent:
    return load_agent()


st.title("📈 Agentic Finance RAG")
st.caption("Ask questions about financial documents. A LangGraph planner routes each question to retrieval, exact calculation and charting. "
           "The bundled sample is **synthetic data about a fictional company**.")

with st.sidebar:
    st.subheader("Setup")
    st.write(f"**LLM:** `{config.LLM_PROVIDER if get_llm() else 'mock (rule-based)'}`")
    st.write(f"**Embeddings:** `{config.EMBEDDINGS}` · **Re-ranker:** `{config.RERANKER}`")
    st.write("**Indexed files:**")
    for p in discover():
        st.write(f"- {p.name}")
    uploads = st.file_uploader("Add documents (.md, .txt, .pdf, .csv)", accept_multiple_files=True, type=["md", "txt", "pdf", "csv"])
    if uploads and st.button("Index uploaded files with the sample"):
        tmp = Path(tempfile.mkdtemp())
        paths = discover()
        for f in uploads:
            dest = tmp / f.name
            dest.write_bytes(f.getvalue())
            paths.append(dest)
        store = build_index(paths)
        st.session_state["agent"] = FinanceRAGAgent(store, get_reranker(), get_llm())
        st.success(f"Indexed {len(store.chunks)} chunks.")
    if st.button("Rebuild index"):
        get_agent.clear()
        st.session_state.pop("agent", None)

    st.subheader("Try")
    examples = [
        "How did revenue grow from Q1 FY2025 to Q2 FY2026, and what drove the margin improvement?",
        "What are the main risks to operating margin?",
        "Plot cloud and services revenue as a bar chart",
        "What is the CAGR of cloud revenue from Q1 FY2025 to Q2 FY2026?",
    ]
    for ex in examples:
        if st.button(ex, width="stretch"):
            st.session_state["question"] = ex

agent = st.session_state.get("agent") or get_agent()
question = st.text_input("Your question", value=st.session_state.get("question", ""))

if question:
    with st.spinner("Planning and running tools..."):
        result = agent.ask(question)
    st.markdown("### Answer")
    st.write(result["answer"])
    if result.get("chart") is not None:
        st.plotly_chart(result["chart"], width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Agent trace", expanded=True):
            for step in result["trace"]:
                st.code(step, language=None)
        if result.get("calc"):
            with st.expander("Calculations"):
                st.json(result["calc"])
    with col2:
        with st.expander(f"Sources ({len(result.get('contexts', []))})", expanded=True):
            for c in result.get("contexts", []):
                st.markdown(f"**{c['source']}** · {c['section']} · score {c['score']}")
                st.caption(c["text"])

with st.expander("Quarterly figures (synthetic, USD m)"):
    st.dataframe(load_financials(), width="stretch")
