from pathlib import Path

import pytest

from finrag.embeddings import HashingEmbedder
from finrag.graph import FinanceRAGAgent, rule_based_plan
from finrag.ingest import chunk_documents, chunk_text, discover
from finrag.rerank import BM25Reranker
from finrag.tools import calculate, load_financials, make_chart, safe_eval
from finrag.vectorstore import FaissStore


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    store = FaissStore(HashingEmbedder())
    store.add(chunk_documents(discover()))
    store.save(tmp_path_factory.mktemp("idx"))
    return FinanceRAGAgent(store, BM25Reranker(), llm=None)


def test_chunking_overlaps_and_respects_size():
    text = " ".join(f"Sentence number {i} about revenue." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 5 and all(len(c) <= 360 for c in chunks)


def test_vector_store_roundtrip(tmp_path):
    store = FaissStore(HashingEmbedder())
    store.add(chunk_documents(discover()))
    store.save(tmp_path)
    loaded = FaissStore.load(tmp_path, HashingEmbedder())
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.search("foreign exchange rupee", 3)


def test_calculations_match_table():
    df = load_financials()
    g = calculate({"op": "growth", "metric": "revenue", "start": "Q2 FY2025", "end": "q2 fy 2026"}, df)
    assert g["pct"] == pytest.approx(20.86, abs=0.01)
    m = calculate({"op": "margin", "metric": "operating_income", "end": "Q2 FY2026"}, df)
    assert m["pct"] == pytest.approx(23.05, abs=0.01)
    assert calculate({"op": "total", "metric": "revenue", "start": "Q1 FY2025", "end": "Q4 FY2025"}, df)["value"] == pytest.approx(522.7)
    assert safe_eval("(2 + 3) * 4") == 20
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")


def test_planner_routes():
    df = load_financials()
    assert rule_based_plan("What are the main risks to operating margin?", df)["actions"] == ["retrieve"]
    assert rule_based_plan("Plot cloud revenue over time", df)["actions"] == ["chart"]
    p = rule_based_plan("How did revenue grow from Q1 FY2025 to Q2 FY2026 and why?", df)
    assert p["actions"] == ["retrieve", "calculate"] and p["calc"][0]["op"] == "growth"


def test_agent_end_to_end(agent):
    out = agent.ask("How exposed is the company to foreign exchange?")
    assert any(c["source"] == "risk_factors.md" for c in out["contexts"])
    assert "rupee" in out["answer"].lower()

    out = agent.ask("What was net income growth in Q2 FY2026 year over year? Show a bar chart.")
    assert "+30.2%" in out["answer"]
    assert out["chart"] is not None and len(make_chart({"metrics": ["net_income"]}).data) == 1
