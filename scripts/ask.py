"""Ask FinLens from the terminal.

Run:  python scripts/ask.py "Compare Nilgiri and Sahyadri operating margins for FY25"
      python scripts/ask.py            (interactive chat with memory)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finlens.service import FinLens  # noqa: E402


def show(r):
    print(f"\n{r['answer']}\n")
    print(f"  route: {r['route']}   confidence: {r['confidence']}   tools: {', '.join(r['tools_used']) or '-'}")
    if r["rewritten_query"]:
        print(f"  standalone query: {r['rewritten_query']}")
    if r["unanswerable_parts"]:
        print(f"  not answerable: {'; '.join(r['unanswerable_parts'])}")
    print(f"  latency: {r['latency_ms']} ms   cost: ${r['cost_usd']}   trace: {r['trace_id']}\n")


def main():
    app = FinLens()
    if len(sys.argv) > 1:
        show(app.ask(" ".join(sys.argv[1:])))
        return
    print("FinLens chat. Empty line to quit.")
    while True:
        q = input("> ").strip()
        if not q:
            break
        show(app.ask(q, session_id="cli"))


if __name__ == "__main__":
    main()
