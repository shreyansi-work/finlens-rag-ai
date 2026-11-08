"""FinLens: the one entry point used by the CLI, the API and the evaluator.

ask() = memory -> query understanding -> cache -> agent -> reflection -> trace
"""
from finlens import cache, memory, tracing
from finlens.agent.agent import answer_question
from finlens.config import Settings, settings
from finlens.db import connect
from finlens.generation import Answer
from finlens.query.understanding import understand


class FinLens:
    def __init__(self, s: Settings = settings):
        self.s = s
        self.conn = connect(s.db_path)
        self.years = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT fiscal_year FROM documents WHERE fiscal_year IS NOT NULL ORDER BY fiscal_year")]

    def ask(self, question: str, session_id: str = "default", on_event=None, use_memory: bool = True,
            allow_llm: bool | None = None, persist_trace: bool = True) -> dict:
        """allow_llm=False forbids the hosted LLM for this question (used for private uploads unless the user
        opted in); None keeps the configured behaviour. persist_trace=False keeps the question out of the log."""
        s = self.s if allow_llm is None else self.s.with_(use_synthesis=self.s.use_synthesis and allow_llm)
        trace = tracing.start_trace(question, persist_trace)
        if on_event:
            trace.listeners.append(on_event)
        mem = memory.get(session_id) if use_memory else memory.Memory()

        with trace.span("understand"):
            plan = understand(question, mem.history(), self.s, self.years)
            if not self.s.use_agent:          # single-shot RAG baseline
                plan.route, plan.sub_queries = "vector", []
        trace.event("query", rewritten=plan.query, route=plan.route, filters=plan.filters,
                    variants=plan.variants, memory_facts=mem.facts)

        cached = cache.lookup(self.conn, plan.query, self.s.cache_threshold) if self.s.use_cache else None
        if cached:
            trace.event("cache_hit")
            answer = Answer(**{**cached, "cached": True})
        else:
            with trace.span("agent", route=plan.route):
                answer = answer_question(plan, self.conn, s)
            if self.s.use_cache and not answer.answer.startswith("I don't know"):
                cache.store(self.conn, plan.query, answer.model_dump())

        result = answer.model_dump()
        record = trace.finish(result)
        mem.add(question, answer.answer)
        return {**result, "rewritten_query": plan.query, "filters": plan.filters, "trace_id": trace.id,
                "latency_ms": record["latency_ms"], "cost_usd": record["cost_usd"], "spans": record["spans"]}

    def feedback(self, question: str, answer: str, rating: int, reason: str = ""):
        """Thumbs up/down. Bad answers become candidates for the eval set."""
        self.conn.execute("INSERT INTO feedback (question, answer, rating, reason) VALUES (?,?,?,?)",
                          (question, answer, rating, reason))
        self.conn.commit()
