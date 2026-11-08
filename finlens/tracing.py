"""Minimal tracing: every query records its rewrites, retrievals, tool calls,
tokens, latency and cost as one JSON line in data/traces.jsonl.

The same span structure maps directly onto Langfuse / Arize Phoenix if you
want a hosted UI later.
"""
import contextvars
import json
import time
import uuid
from contextlib import contextmanager

from finlens.config import TRACE_PATH

_current = contextvars.ContextVar("trace", default=None)


class Trace:
    def __init__(self, question: str, persist: bool = True):
        self.id = uuid.uuid4().hex[:12]
        self.question = question
        self.persist = persist     # False for private upload workspaces: nothing is written to the trace log
        self.spans = []
        self.tokens_in = self.tokens_out = 0
        self.start = time.perf_counter()
        self.listeners = []        # callbacks for live status (used by SSE streaming)

    @property
    def cost_usd(self) -> float:
        return 0.0

    def event(self, name: str, **data):
        self.spans.append({"name": name, "t_ms": round((time.perf_counter() - self.start) * 1000, 1), **data})
        for cb in self.listeners:
            cb(name, data)

    @contextmanager
    def span(self, name: str, **data):
        t0 = time.perf_counter()
        for cb in self.listeners:
            cb(name, data)
        yield
        self.spans.append({"name": name, "ms": round((time.perf_counter() - t0) * 1000, 1), **data})

    def add_usage(self, tokens_in: int, tokens_out: int):
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    def finish(self, answer: dict) -> dict:
        record = {"trace_id": self.id, "question": self.question, "spans": self.spans,
                  "latency_ms": round((time.perf_counter() - self.start) * 1000, 1),
                  "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                  "cost_usd": round(self.cost_usd, 5), "answer": answer}
        if self.persist:
            TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        return record


def start_trace(question: str, persist: bool = True) -> Trace:
    t = Trace(question, persist)
    _current.set(t)
    return t


def current() -> Trace | None:
    return _current.get()


def recent(n: int = 20) -> list[dict]:
    if not TRACE_PATH.exists():
        return []
    lines = TRACE_PATH.read_text(encoding="utf-8").splitlines()[-n:]
    return [json.loads(l) for l in reversed(lines)]
