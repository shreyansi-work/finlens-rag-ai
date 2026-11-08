"""Retrieval: metadata pre-filter -> dense + BM25 -> RRF -> rerank -> MMR
-> parent expansion -> contextual compression.

Each step can be switched off in Settings so its effect can be measured.
"""
import os
import re

import numpy as np

from finlens.config import Settings
from finlens.retrieval import index
from finlens.text import content_tokens, sentences


def rrf(result_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _filter_sql(filters: dict, levels: tuple, s: Settings) -> tuple[str, list]:
    where = ["c.active=1", f"c.level IN ({','.join('?' * len(levels))})"]
    args = list(levels)
    if s.use_filters:
        for col in ("ticker", "fiscal_year", "doc_type", "quarter"):
            vals = filters.get(col)
            if vals:
                vals = vals if isinstance(vals, list) else [vals]
                where.append(f"c.{col} IN ({','.join('?' * len(vals))})")
                args += vals
    return " AND ".join(where), args


def dense_rank(query: str, cand_ids: np.ndarray, db_path: str, limit: int) -> list[int]:
    embedder, ids, vecs = index.load(db_path)
    mask = np.isin(ids, cand_ids)
    if not mask.any():
        return []
    q = embedder.embed([query])[0]
    sims = vecs[mask] @ q
    order = np.argsort(-sims)[:limit]
    return [int(x) for x in ids[mask][order]]


def bm25_rank(conn, query: str, where: str, args: list, limit: int) -> list[int]:
    terms = [t for t in content_tokens(query) if re.match(r"^[a-z0-9]+$", t)]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    sql = (f"SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
           f"WHERE chunks_fts MATCH ? AND {where} ORDER BY bm25(chunks_fts) LIMIT ?")
    return [r[0] for r in conn.execute(sql, [match] + args + [limit])]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
_cross_encoder = None


def rerank(query: str, rows: list[dict], db_path: str) -> list[dict]:
    """Cross-encoder when available (FINLENS_RERANKER=ce), else a light reranker
    mixing dense similarity with query-term coverage (numbers and years weigh 2x)."""
    global _cross_encoder
    if os.getenv("FINLENS_RERANKER") == "ce":
        from sentence_transformers import CrossEncoder
        _cross_encoder = _cross_encoder or CrossEncoder("BAAI/bge-reranker-base")
        scores = _cross_encoder.predict([(query, r["embed_text"]) for r in rows])
    else:
        embedder, _, _ = index.load(db_path)
        qv = embedder.embed([query])[0]
        cv = embedder.embed([r["embed_text"] for r in rows])
        q_terms = set(content_tokens(query))
        weights = {t: (2.0 if any(ch.isdigit() for ch in t) else 1.0) for t in q_terms}
        total = sum(weights.values()) or 1
        scores = []
        for r, v in zip(rows, cv):
            toks = set(content_tokens(r["embed_text"]))
            coverage = sum(w for t, w in weights.items() if t in toks) / total
            scores.append(0.4 * float(v @ qv) + 0.6 * coverage)
    for r, sc in zip(rows, scores):
        r["score"] = float(sc)
    return sorted(rows, key=lambda r: -r["score"])


def mmr(query: str, rows: list[dict], k: int, db_path: str, lam: float = 0.7) -> list[dict]:
    if len(rows) <= k:
        return rows
    embedder, _, _ = index.load(db_path)
    vecs = embedder.embed([r["embed_text"] for r in rows])
    rel = np.array([r["score"] for r in rows])
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
    chosen = [0]
    while len(chosen) < k:
        best, best_score = None, -1e9
        for i in range(len(rows)):
            if i in chosen:
                continue
            redundancy = max(float(vecs[i] @ vecs[j]) for j in chosen)
            score = lam * rel[i] - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
    return [rows[i] for i in chosen]


def compress(query: str, text: str) -> str:
    """Contextual compression: keep sentences that share terms with the query."""
    q = set(content_tokens(query))
    sents = sentences(text)
    kept = [x for x in sents if q & set(content_tokens(x))]
    return " ".join(kept) if kept else text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def search(conn, query: str, filters: dict, s: Settings, variants: list[str] | None = None,
           levels=("child", "table"), k: int | None = None) -> list[dict]:
    k = k or s.final_k
    where, args = _filter_sql(filters, levels, s)
    cand = np.array([r[0] for r in conn.execute(f"SELECT c.chunk_id FROM chunks c WHERE {where}", args)])
    if len(cand) == 0:
        return []

    lists = []
    for q in [query] + (variants or []):
        lists.append(dense_rank(q, cand, s.db_path, s.candidates_k))
        if s.use_hybrid:
            lists.append(bm25_rank(conn, q, where, args, s.candidates_k))
    ranked = rrf(lists)[: s.candidates_k]
    if not ranked:
        return []

    by_id = {r["chunk_id"]: dict(r) for r in conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(ranked))})", ranked)}
    rows = [by_id[i] for i in ranked]
    for i, r in enumerate(rows):
        r["score"] = 1 / (i + 1)
    if s.use_rerank:
        rows = rerank(query, rows, s.db_path)
    rows = mmr(query, rows, k, s.db_path) if s.use_mmr else rows[:k]

    hits, seen_parents = [], set()
    for r in rows:
        text = r["text"]
        if s.use_parent_expansion and r["parent_id"]:
            if r["parent_id"] in seen_parents:
                continue
            seen_parents.add(r["parent_id"])
            parent = conn.execute("SELECT text FROM chunks WHERE chunk_id=?", (r["parent_id"],)).fetchone()
            text = parent["text"] if parent else text
        if s.use_compression and r["level"] != "table":
            text = compress(query, text)
        hits.append({**r, "context": text})
    return hits
