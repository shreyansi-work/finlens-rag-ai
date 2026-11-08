"""Semantic cache: reuse an answer when a new question is nearly the same as a
cached one AND the corpus has not changed since it was cached."""
import json

import numpy as np

from finlens.db import corpus_version
from finlens.retrieval.embeddings import quick_embed
from finlens.text import find_tickers, tokens


def _signature(q: str) -> tuple:
    """Companies and numbers/years must match exactly; similarity alone is not enough."""
    return tuple(sorted(find_tickers(q))), tuple(sorted(t for t in tokens(q) if any(c.isdigit() for c in t)))


def lookup(conn, question: str, threshold: float) -> dict | None:
    version = corpus_version(conn)
    rows = conn.execute("SELECT question, embedding, answer_json FROM cache WHERE corpus_version=?", (version,)).fetchall()
    if not rows:
        return None
    q = quick_embed([question])[0]
    best, best_sim = None, 0.0
    for r in rows:
        if _signature(r["question"]) != _signature(question):
            continue
        sim = float(np.frombuffer(r["embedding"], dtype=np.float32) @ q)
        if sim > best_sim:
            best, best_sim = r, sim
    if best is not None and best_sim >= threshold:
        return json.loads(best["answer_json"])
    return None


def store(conn, question: str, answer: dict):
    emb = quick_embed([question])[0].astype(np.float32).tobytes()
    conn.execute("INSERT INTO cache (question, embedding, corpus_version, answer_json) VALUES (?,?,?,?)",
                 (question, emb, corpus_version(conn), json.dumps(answer)))
    conn.commit()
