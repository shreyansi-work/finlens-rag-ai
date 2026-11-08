"""Dense vector index: vectors for every active chunk, stored next to the DB.

With a few thousand chunks a brute-force matrix product is faster than an ANN
index. HNSW (pgvector / Qdrant) is the upgrade path once the corpus grows.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from finlens.config import Settings
from finlens.retrieval.embeddings import get_embedder_class


def _paths(db_path: str) -> tuple[Path, Path]:
    p = Path(db_path)
    return p.with_suffix(".vectors.npz"), p.with_suffix(".embedder.npz")


def build_vectors(conn, s: Settings) -> int:
    rows = conn.execute(
        "SELECT chunk_id, embed_text FROM chunks WHERE active=1 AND level NOT IN ('parent', 'toc')").fetchall()
    if not rows:
        return 0
    ids = np.array([r[0] for r in rows])
    texts = [r[1] for r in rows]
    embedder = get_embedder_class()().fit(texts)
    vecs = embedder.embed(texts)
    vec_path, emb_path = _paths(s.db_path)
    np.savez(vec_path, ids=ids, vecs=vecs)
    embedder.save(emb_path)
    load.cache_clear()
    return len(ids)


def vectors_exist(db_path: str) -> bool:
    return all(p.exists() for p in _paths(db_path))


@lru_cache(maxsize=8)
def load(db_path: str):
    """Returns (embedder, chunk_ids, vectors). Cached per DB file."""
    vec_path, emb_path = _paths(db_path)
    d = np.load(vec_path)
    return get_embedder_class().load(emb_path), d["ids"], d["vecs"]
