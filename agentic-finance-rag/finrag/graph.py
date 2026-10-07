"""The agent: a LangGraph state machine.

    planner -> router -> (retrieve | calculate | chart)* -> answer

The planner (LLM, or a rule-based fallback in mock mode) decides which tools a question
needs; the router walks that plan, each tool writes its result into the shared state,
and the answer node composes a cited response.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .embeddings import tokenize
from .rerank import STOP
from .tools import METRICS, calculate, load_financials, make_chart

PLANNER_SYSTEM = """You are the planner of a financial research agent. Decide which tools answer the question.
Tools: "retrieve" (search report/transcript/risk text), "calculate" (exact maths on the quarterly table), "chart" (Plotly chart of table metrics).
Table columns: {columns}. Periods: {periods}.
Return JSON: {{"actions": [subset of "retrieve","calculate","chart" in order],
 "calc": {{"op": "growth|margin|cagr|average|total|expression", "metric": column, "start": period, "end": period, "expression": str}} or null,
 "chart": {{"metrics": [columns], "kind": "line|bar"}} or null}}"""

ANSWER_SYSTEM = """You answer questions about a company's financial documents.
Use ONLY the provided context and tool results. Cite document sources in square brackets, e.g. [risk_factors.md].
Copy numbers from the calculation result exactly. If the context does not contain the answer, say so. Be concise."""

EXPLAIN_WORDS = ("why", "drove", "driver", "explain", "risk", "guidance", "outlook", "strategy", "said", "according", "reason",
                 "management", "describe", "summar", "competition", "competitive", "expos", "expect", "plan", "who", "what is", "what are")
MARGIN_METRIC = {"gross": "gross_profit", "operating": "operating_income", "net": "net_income"}


class AgentState(TypedDict, total=False):
    question: str
    pending: list[str]
    next: str
    calc_spec: list[dict]
    chart_spec: dict | None
    contexts: list[dict]
    calc: list[dict]
    chart: Any
    answer: str
    trace: list[str]


def _periods(text: str) -> list[str]:
    return [f"Q{q} FY{y}" for q, y in re.findall(r"q\s*([1-4])\s*fy\s*(\d{4})", text.lower())]


def rule_based_plan(question: str, df: pd.DataFrame) -> dict:
    q = question.lower()
    found: dict[str, int] = {}
    for key in sorted(METRICS, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(key)}\b", q)
        if m and METRICS[key] not in found:
            found[METRICS[key]] = m.start()
    if "revenue" in found and ({"cloud_revenue", "services_revenue", "cost_of_revenue"} & found.keys()):
        found.pop("revenue")  # "cloud revenue" should not also chart total revenue
    metrics = sorted(found, key=found.get) or ["revenue"]
    periods = _periods(q)
    first, last = df["period"].iloc[0], df["period"].iloc[-1]

    ops: list[str] = []
    if re.fullmatch(r"[\d\s.+\-*/()]+", q.strip(" ?=")) and re.search(r"\d\s*[+\-*/]", q):
        ops.append("expression")
    else:
        if "cagr" in q or "compound" in q:
            ops.append("cagr")
        elif re.search(r"\b(grow|grew|growth|increased?|changed?|rise|rose|declined?|fell|yoy)\b|year[- ]over[- ]year", q):
            ops.append("growth")
        if "margin" in q:
            ops.append("margin")
        if re.search(r"\b(average|mean)\b", q):
            ops.append("average")
        if re.search(r"\b(total|sum|combined)\b", q):
            ops.append("total")
    explanatory = any(w in q for w in EXPLAIN_WORDS)
    if explanatory and not periods and "expression" not in ops:
        ops = []  # "what are the risks to operating margin?" is a reading question, not a calculation

    calcs: list[dict] = []
    for op in ops:
        if op == "expression":
            calcs.append({"op": op, "expression": q.strip(" ?=")})
            continue
        start, end = (periods[0], periods[-1]) if len(periods) >= 2 else (None, periods[0] if periods else None)
        metric = metrics[0]
        if op == "margin":
            metric = next((col for word, col in MARGIN_METRIC.items() if f"{word} margin" in q), "gross_profit" if metric == "revenue" else metric)
        if op == "growth" and len(periods) == 1:  # "growth in Q2 FY2026" -> year over year
            qtr, yr = re.match(r"Q(\d) FY(\d{4})", periods[0]).groups()
            start, end = f"Q{qtr} FY{int(yr) - 1}", periods[0]
        calcs.append({"op": op, "metric": metric, "start": start or first, "end": end or last})

    chart = None
    if re.search(r"chart|plot|graph|visuali|trend|over time", q):
        chart = {"metrics": metrics, "kind": "bar" if "bar" in q else "line"}

    actions = []
    if explanatory or (not calcs and chart is None):
        actions.append("retrieve")
    if calcs:
        actions.append("calculate")
    if chart:
        actions.append("chart")
    return {"actions": actions, "calc": calcs, "chart": chart}


class FinanceRAGAgent:
    def __init__(self, store, reranker, llm=None, df: pd.DataFrame | None = None, k: int = 12, top_k: int = 4) -> None:
        self.store, self.reranker, self.llm = store, reranker, llm
        self.df = load_financials() if df is None else df
        self.k, self.top_k = k, top_k
        self.graph = self._build()

    # ---- nodes ---------------------------------------------------------
    def planner(self, state: AgentState) -> AgentState:
        q = state["question"]
        plan = None
        if self.llm is not None:
            try:
                plan = self.llm.json(
                    PLANNER_SYSTEM.format(columns=", ".join(self.df.columns[1:]), periods=", ".join(self.df["period"])), q
                )
            except Exception as exc:  # fall back rather than fail the request
                state.setdefault("trace", []).append(f"planner LLM failed ({exc}); using rules")
        if not plan or not plan.get("actions"):
            plan = rule_based_plan(q, self.df)
        actions = [a for a in plan.get("actions", []) if a in ("retrieve", "calculate", "chart")] or ["retrieve"]
        trace = state.get("trace", []) + [f"planner: {' -> '.join(actions)}"]
        calc = plan.get("calc")
        calc_specs = [calc] if isinstance(calc, dict) else [c for c in (calc or []) if isinstance(c, dict)]
        return {"pending": actions, "calc_spec": calc_specs, "chart_spec": plan.get("chart"), "trace": trace,
                "contexts": [], "calc": [], "chart": None}

    def router(self, state: AgentState) -> AgentState:
        pending = list(state.get("pending", []))
        return {"next": pending.pop(0) if pending else "answer", "pending": pending}

    def retrieve(self, state: AgentState) -> AgentState:
        hits = self.reranker.rerank(state["question"], self.store.search(state["question"], self.k), self.top_k)
        contexts = [{"source": c.source, "title": c.meta.get("title", ""), "section": c.section, "text": c.text, "score": round(s, 3)} for c, s in hits]
        return {"contexts": contexts, "trace": state["trace"] + [f"retrieve: {len(contexts)} chunks from {sorted({c['source'] for c in contexts})}"]}

    def calculate(self, state: AgentState) -> AgentState:
        specs = state.get("calc_spec") or [{"op": "growth", "metric": "revenue"}]
        results, trace = [], list(state["trace"])
        for spec in specs:
            try:
                results.append(calculate(spec, self.df))
            except Exception as exc:
                results.append({"error": str(exc), "text": f"Calculation failed: {exc}"})
            trace.append(f"calculate: {json.dumps(spec)}")
        return {"calc": results, "trace": trace}

    def chart(self, state: AgentState) -> AgentState:
        spec = state.get("chart_spec") or {"metrics": ["revenue"]}
        return {"chart": make_chart(spec, self.df), "trace": state["trace"] + [f"chart: {json.dumps(spec)}"]}

    def answer(self, state: AgentState) -> AgentState:
        text = self._write_with_llm(state) if self.llm is not None else None
        return {"answer": text or self._write_extractive(state), "trace": state["trace"] + ["answer"]}

    # ---- answer writers --------------------------------------------------
    def _write_with_llm(self, state: AgentState) -> str | None:
        ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in state.get("contexts", []))
        calc = " ".join(c["text"] for c in state.get("calc") or []) or "none"
        chart = "a chart was generated and is shown to the user" if state.get("chart") is not None else "none"
        try:
            return self.llm.chat(ANSWER_SYSTEM, f"Question: {state['question']}\n\nContext:\n{ctx or 'none'}\n\nCalculation: {calc}\nChart: {chart}")
        except Exception:
            return None

    def _write_extractive(self, state: AgentState) -> str:
        parts = []
        for c in state.get("calc") or []:
            parts.append(c["text"])
        if state.get("contexts"):
            qtok = {t for t in tokenize(state["question"]) if t not in STOP}
            sentences = []
            for c in state["contexts"]:
                if state.get("calc") and c["source"].endswith(".csv"):
                    continue  # the calculation already reports the table numbers
                sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", c["text"]) if x.strip()]
                where = set(tokenize(f"{c.get('title', '')} {c['section']}"))
                for i, s in enumerate(sents):
                    overlap = len(qtok & (set(tokenize(s)) | where))
                    if not overlap or len(s) <= 30:
                        continue
                    if s.endswith("?"):  # a matching question in a transcript: use the answer that follows
                        s = " ".join(sents[i + 1 : i + 3])
                        if not s:
                            continue
                    sentences.append((overlap + c["score"], s, c["source"]))
            seen, picked = set(), []
            for _, s, src in sorted(sentences, key=lambda x: x[0], reverse=True):
                if not any(s in p or p in s for p in seen):
                    seen.add(s)
                    picked.append(f"{s} [{src}]")
                if len(picked) == 3:
                    break
            if picked:
                parts.append("From the documents: " + " ".join(picked))
        if state.get("chart") is not None:
            parts.append("Chart generated: " + state["chart"].layout.title.text + ".")
        return "\n\n".join(parts) or "I couldn't find that in the documents."

    # ---- graph -----------------------------------------------------------
    def _build(self):
        g = StateGraph(AgentState)
        for name in ("planner", "router", "retrieve", "calculate", "chart", "answer"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "planner")
        g.add_edge("planner", "router")
        g.add_conditional_edges("router", lambda s: s["next"], {k: k for k in ("retrieve", "calculate", "chart", "answer")})
        for tool in ("retrieve", "calculate", "chart"):
            g.add_edge(tool, "router")
        g.add_edge("answer", END)
        return g.compile()

    def ask(self, question: str) -> AgentState:
        return self.graph.invoke({"question": question, "trace": []})


def load_agent(rebuild: bool = False) -> FinanceRAGAgent:
    """Load (or build) the index and return a ready agent."""
    from . import config
    from .embeddings import get_embedder
    from .ingest import build_index
    from .llm import get_llm
    from .rerank import get_reranker
    from .vectorstore import FaissStore

    embedder = get_embedder()
    store = None
    if not rebuild and FaissStore.exists(config.INDEX_DIR):
        try:
            store = FaissStore.load(config.INDEX_DIR, embedder)
        except ValueError:
            store = None
    if store is None:
        store = build_index(index_dir=config.INDEX_DIR)
    return FinanceRAGAgent(store, get_reranker(), get_llm())
