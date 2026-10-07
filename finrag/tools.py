"""Deterministic tools the planner can call: financial calculations and Plotly charts.

Numbers in answers come from here, never from the LLM.
"""
from __future__ import annotations

import ast
import operator
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from . import config

METRICS = {
    "revenue": "revenue", "sales": "revenue", "net income": "net_income", "profit": "net_income",
    "operating income": "operating_income", "ebit": "operating_income", "gross profit": "gross_profit",
    "operating expenses": "operating_expenses", "opex": "operating_expenses", "cost of revenue": "cost_of_revenue",
    "cloud": "cloud_revenue", "services": "services_revenue",
}
LABELS = {
    "revenue": "Revenue", "net_income": "Net income", "operating_income": "Operating income", "gross_profit": "Gross profit",
    "operating_expenses": "Operating expenses", "cost_of_revenue": "Cost of revenue", "cloud_revenue": "Cloud revenue",
    "services_revenue": "Services revenue",
}


def load_financials(path: Path = config.FINANCIALS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _norm_period(p: str) -> str:
    """'q2 fy 2026' / 'Q2FY2026' -> 'Q2 FY2026'"""
    m = re.match(r"\s*Q\s*([1-4])\s*FY\s*(\d{4})\s*$", p.upper())
    return f"Q{m.group(1)} FY{m.group(2)}" if m else p.strip().upper()


def _row(df: pd.DataFrame, period: str) -> pd.Series:
    match = df[df["period"].str.upper() == _norm_period(period)]
    if match.empty:
        raise ValueError(f"unknown period {period!r}; available: {', '.join(df['period'])}")
    return match.iloc[0]


# ---- safe arithmetic --------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** are allowed")

    return float(ev(ast.parse(expr, mode="eval")))


# ---- financial calculator --------------------------------------------
def calculate(spec: dict, df: pd.DataFrame | None = None) -> dict:
    """spec: {"op": growth|margin|cagr|average|total|expression, "metric", "start", "end", "expression"}"""
    df = load_financials() if df is None else df
    op = spec.get("op", "growth")
    metric = spec.get("metric", "revenue")
    start = spec.get("start") or df["period"].iloc[0]
    end = spec.get("end") or df["period"].iloc[-1]

    if op == "expression":
        value = safe_eval(spec["expression"])
        return {"op": op, "expression": spec["expression"], "value": round(value, 4), "text": f"{spec['expression']} = {value:,.4g}"}

    if metric not in df.columns:
        raise ValueError(f"unknown metric {metric!r}")
    label = LABELS.get(metric, metric)

    if op == "growth":
        a, b = float(_row(df, start)[metric]), float(_row(df, end)[metric])
        pct = (b - a) / a * 100
        return {"op": op, "metric": metric, "start": start, "end": end, "from": a, "to": b, "change": round(b - a, 2), "pct": round(pct, 2),
                "text": f"{label} went from {a:,.1f} in {start} to {b:,.1f} in {end} (USD m), a change of {b - a:+,.1f} ({pct:+.1f}%)."}
    if op == "margin":
        r = _row(df, end)
        pct = float(r[metric]) / float(r["revenue"]) * 100
        name = {"gross_profit": "Gross", "operating_income": "Operating", "net_income": "Net"}.get(metric, label)
        return {"op": op, "metric": metric, "period": end, "pct": round(pct, 2),
                "text": f"{name} margin in {end} was {pct:.1f}% ({float(r[metric]):,.1f} / {float(r['revenue']):,.1f} USD m)."}
    if op == "cagr":
        a, b = float(_row(df, start)[metric]), float(_row(df, end)[metric])
        i0, i1 = df.index[df["period"].str.upper() == _norm_period(start)][0], df.index[df["period"].str.upper() == _norm_period(end)][0]
        quarters = max(int(i1 - i0), 1)
        q_rate = (b / a) ** (1 / quarters) - 1
        annual = (1 + q_rate) ** 4 - 1
        return {"op": op, "metric": metric, "start": start, "end": end, "quarterly_pct": round(q_rate * 100, 2), "annualised_pct": round(annual * 100, 2),
                "text": f"{label} compounded {q_rate * 100:.2f}% per quarter from {start} to {end}, about {annual * 100:.1f}% annualised."}
    if op in ("average", "total"):
        sel = df.loc[df.index[df["period"].str.upper() == _norm_period(start)][0] : df.index[df["period"].str.upper() == _norm_period(end)][0], metric]
        value = float(sel.mean() if op == "average" else sel.sum())
        return {"op": op, "metric": metric, "start": start, "end": end, "value": round(value, 2),
                "text": f"{op.title()} {label.lower()} from {start} to {end}: {value:,.1f} USD m across {len(sel)} quarters."}
    raise ValueError(f"unknown op {op!r}")


# ---- chart ------------------------------------------------------------
def make_chart(spec: dict, df: pd.DataFrame | None = None) -> go.Figure:
    """spec: {"metrics": [...], "kind": "line"|"bar", "title": str}"""
    df = load_financials() if df is None else df
    metrics = [m for m in spec.get("metrics", ["revenue"]) if m in df.columns] or ["revenue"]
    kind = spec.get("kind", "line")
    fig = go.Figure()
    for m in metrics:
        name = LABELS.get(m, m)
        if kind == "bar":
            fig.add_bar(x=df["period"], y=df[m], name=name)
        else:
            fig.add_scatter(x=df["period"], y=df[m], mode="lines+markers", name=name)
    fig.update_layout(title=spec.get("title") or " vs ".join(LABELS.get(m, m) for m in metrics) + " by quarter (USD m, synthetic data)",
                      xaxis_title="Quarter", yaxis_title="USD millions", barmode="group", template="plotly_white")
    return fig
