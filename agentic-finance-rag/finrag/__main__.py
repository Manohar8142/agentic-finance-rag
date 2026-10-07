"""CLI:  python -m finrag "How did revenue grow from Q1 FY2025 to Q2 FY2026, and what drove it?" [--rebuild]"""
from __future__ import annotations

import sys

from .graph import load_agent


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--rebuild"]
    question = " ".join(args) or "How did revenue grow from Q1 FY2025 to Q2 FY2026, and what drove the margin improvement?"
    agent = load_agent(rebuild="--rebuild" in sys.argv)
    out = agent.ask(question)
    print("Q:", question)
    print("\nTRACE:\n  " + "\n  ".join(out["trace"]))
    print("\nANSWER:\n" + out["answer"])
    if out.get("chart") is not None:
        path = "chart.html"
        out["chart"].write_html(path)
        print(f"\n(chart saved to {path})")


if __name__ == "__main__":
    main()
