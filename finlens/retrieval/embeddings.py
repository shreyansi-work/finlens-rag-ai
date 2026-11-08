"""Dense embeddings.

Default: a small, dependency-free model. Text is hashed into TF-IDF features
(words + word pairs), then projected with LSA (truncated SVD) into a dense
vector fitted on the corpus. It captures co-occurrence ("attrition" ~
"employees leaving") without downloading a model, which keeps the project
runnable offline.

Optional: set FINLENS_EMBEDDER=st to use sentence-transformers (bge-small)
if it is installed. The rest of the pipeline does not change.
"""
import os
import zlib
from pathlib import Path

import numpy as np

from finlens.text import content_tokens

HASH_DIM = 4096


def _features(text: str) -> dict[int, float]:
    toks = content_tokens(text)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    feats: dict[int, float] = {}
    for g in grams:
        h = zlib.crc32(g.encode())
        idx, sign = h % HASH_DIM, 1.0 if (h >> 31) & 1 else -1.0
        feats[idx] = feats.get(idx, 0.0) + sign
    return feats


def hashed_matrix(texts: list[str]) -> np.ndarray:
    m = np.zeros((len(texts), HASH_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for j, v in _features(t).items():
            if v:
                m[i, j] = np.sign(v) * np.log1p(abs(v))
    return m


def normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def quick_embed(texts: list[str]) -> np.ndarray:
    """Unfitted hashed vectors. Used for semantic chunking and the cache key."""
    return normalize(hashed_matrix(texts))


class LSAEmbedder:
    def __init__(self, idf=None, components=None):
        self.idf, self.components = idf, components

    def fit(self, texts: list[str], dim: int = 128) -> "LSAEmbedder":
        x = hashed_matrix(texts)
        df = (x != 0).sum(axis=0)
        self.idf = np.log((1 + len(texts)) / (1 + df)).astype(np.float32) + 1
        x = normalize(x * self.idf)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self.components = vt[: min(dim, vt.shape[0])].astype(np.float32)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        x = normalize(hashed_matrix(texts) * self.idf)
        return normalize(x @ self.components.T)

    def save(self, path: Path):
        np.savez(path, idf=self.idf, components=self.components)

    @classmethod
    def load(cls, path: Path) -> "LSAEmbedder":
        d = np.load(path)
        return cls(d["idf"], d["components"])


class STEmbedder:
    """sentence-transformers backend (optional)."""
    def __init__(self, name="BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def fit(self, texts, dim=None):
        return self

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()


def get_embedder_class():
    return STEmbedder if os.getenv("FINLENS_EMBEDDER") == "st" else LSAEmbedder
