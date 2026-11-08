"""SQLite storage: one file holds documents, chunks, BM25 index, financial
tables, the knowledge graph, cache and feedback.

SQLite plays the role Postgres + pgvector + full-text search would play in a
larger deployment. The schema is deliberately plain so it ports easily.
"""
import hashlib
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,      -- e.g. INFY_FY25_annual_report
    filename    TEXT NOT NULL,
    company     TEXT, ticker TEXT, fiscal_year TEXT, quarter TEXT, doc_type TEXT,
    file_hash   TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    num_pages   INTEGER,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS document_versions (
    doc_id TEXT, version INTEGER, file_hash TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    parent_id   INTEGER,               -- NULL for parents and summaries
    level       TEXT NOT NULL,         -- child | parent | table | summary
    company TEXT, ticker TEXT, fiscal_year TEXT, quarter TEXT, doc_type TEXT,
    section     TEXT,
    page        INTEGER,
    text        TEXT NOT NULL,         -- what the LLM sees
    embed_text  TEXT NOT NULL,         -- what gets embedded (text + contextual header)
    chunk_hash  TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1,
    version     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_chunks_meta ON chunks(active, level, ticker, fiscal_year, doc_type);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(embed_text, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS financials (
    ticker TEXT, company TEXT, fiscal_year TEXT, quarter TEXT,  -- quarter = 'FY' for full year
    metric TEXT, value REAL, unit TEXT,
    doc_id TEXT, page INTEGER,
    UNIQUE(ticker, fiscal_year, quarter, metric)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    subject TEXT, relation TEXT, object TEXT,
    ticker TEXT, fiscal_year TEXT, doc_id TEXT, page INTEGER,
    UNIQUE(subject, relation, object, fiscal_year)
);
CREATE TABLE IF NOT EXISTS cache (
    question TEXT, embedding BLOB, corpus_version TEXT, answer_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    question TEXT, answer TEXT, rating INTEGER, reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """Read-only connection used by the text-to-SQL tool (SQL safety)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def corpus_version(conn: sqlite3.Connection) -> str:
    """Changes whenever any document changes; used to invalidate the cache."""
    rows = conn.execute("SELECT doc_id, file_hash FROM documents ORDER BY doc_id").fetchall()
    joined = "|".join(f"{r[0]}:{r[1]}" for r in rows)
    return hashlib.sha1(joined.encode()).hexdigest()[:12]
