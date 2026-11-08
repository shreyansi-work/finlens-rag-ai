"""Ingestion pipeline: PDF -> parsed pages -> chunks, tables, summaries, graph.

Run through scripts/ingest.py. Re-running is incremental: unchanged files are
skipped, and for changed files only chunks whose hash changed are replaced
(old versions are kept with active=0 for history).
"""
import hashlib
import math
import re
from pathlib import Path

import numpy as np

from finlens import llm
from finlens.config import Settings
from finlens.db import connect
from finlens.ingestion.chunking import child_chunks, parent_chunks, sections_from_pages
from finlens.ingestion.parser import DocumentError, metadata_from_filename, parse_document
from finlens.retrieval import index
from finlens.retrieval.embeddings import quick_embed
from finlens.text import company_name, numbers_in, sentences, sha

DOC_TYPE_LABEL = {"annual_report": "Annual Report", "transcript": "Earnings Call Transcript",
                  "results": "Quarterly Results"}
PERIOD_RE = re.compile(r"^(?:(Q[1-4])\s*)?(FY\d{2})$")


def ingest_folder(folder: Path, s: Settings, log=print) -> dict:
    conn = connect(s.db_path)
    stats = {"new": 0, "updated": 0, "skipped": 0}
    # Oldest fiscal year first, so restated figures from newer reports win.
    files = sorted(folder.glob("*.pdf"), key=lambda p: (metadata_from_filename(p)["fiscal_year"] or "", p.name))
    for path in files:
        status = ingest_file(conn, path, s)
        stats[status] += 1
        log(f"  {status:8s} {path.name}")
    if stats["new"] or stats["updated"] or not index.vectors_exist(s.db_path):
        n = index.build_vectors(conn, s)
        log(f"  embedded {n} chunks")
    conn.close()
    return stats


def ingest_file(conn, path: Path, s: Settings, meta: dict | None = None) -> str:
    """Ingest one PDF (or DOCX). `meta` is given for user uploads: a plain document with no ticker or fiscal
    year, so no financial tables or graph edges are extracted and only the text is indexed."""
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    doc_id = path.stem
    generic = meta is not None
    if generic:
        meta = {"ticker": "", "fiscal_year": None, "quarter": None, "doc_type": "document", **meta, "generic": True}
    else:
        meta = metadata_from_filename(path)
        meta["company"] = company_name(meta["ticker"])

    row = conn.execute("SELECT file_hash, version FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if row and row["file_hash"] == file_hash:
        return "skipped"
    dup = conn.execute("SELECT doc_id FROM documents WHERE file_hash=? AND doc_id!=?", (file_hash, doc_id)).fetchone()
    if dup:
        return "skipped"          # same file already ingested under another name
    version = row["version"] + 1 if row else 1

    pages = parse_document(path)
    sections, tables = sections_from_pages(pages)
    if generic and sum(len(sec.text) for sec in sections) < 40:
        raise DocumentError("No readable text found. Scanned or image-only files are not supported.")
    conn.execute("INSERT OR REPLACE INTO documents (doc_id, filename, company, ticker, fiscal_year, quarter, doc_type,"
                 " file_hash, version, num_pages) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (doc_id, path.name, meta["company"], meta["ticker"], meta["fiscal_year"], meta["quarter"],
                  meta["doc_type"], file_hash, version, len(pages)))
    conn.execute("INSERT INTO document_versions (doc_id, version, file_hash) VALUES (?,?,?)", (doc_id, version, file_hash))

    chunks = build_chunks(doc_id, meta, sections, tables, s)
    upsert_chunks(conn, doc_id, meta, chunks, version)
    if not generic:
        store_financials(conn, doc_id, meta, tables)
        if s.build_graph:
            store_graph(conn, doc_id, meta, pages)
    if s.build_summaries:
        build_summaries(conn, doc_id, meta, version)
    conn.commit()
    return "updated" if row else "new"


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------
def contextual_header(meta: dict, section: str, text: str, s: Settings) -> str:
    """One line of context prepended before embedding (contextual retrieval)."""
    if meta.get("generic"):
        base = f'This chunk is from the document "{meta["company"]}", section {section}.'
    else:
        period = " ".join(x for x in (meta["quarter"], meta["fiscal_year"]) if x)
        base = f"This chunk is from {meta['company']} ({meta['ticker']}) {period} {DOC_TYPE_LABEL.get(meta['doc_type'], '')}, section {section}."
    if llm.enabled():
        extra = llm.complete(f"In one short sentence, say what this passage is about.\n\n{text[:1500]}")
        return f"{base} {extra}"
    return base


def build_chunks(doc_id, meta, sections, tables, s: Settings) -> list[dict]:
    parents, owner = parent_chunks(sections, s.parent_tokens)
    out = []
    for i, p in enumerate(parents):
        out.append({"key": f"P{i}", "level": "parent", "parent": None, "section": p.heading,
                    "page": p.page, "text": p.text})
    seen = set()
    for sec, pi in zip(sections, owner):
        prose = "\n".join(l for l in sec.text.splitlines() if " | " not in l)   # tables get their own chunks
        for piece in child_chunks(prose, s.chunk_strategy, s.child_tokens, s.chunk_overlap, quick_embed):
            if piece in seen:            # dedup identical chunks within a document
                continue
            seen.add(piece)
            out.append({"key": None, "level": "child", "parent": f"P{pi}", "section": sec.heading,
                        "page": sec.page, "text": piece})
    for t in tables:
        out.append({"key": None, "level": "table", "parent": None, "section": t.heading,
                    "page": t.page, "text": f"Table: {t.heading}\n{t.text}"})
    for c in out:
        header = contextual_header(meta, c["section"], c["text"], s) if s.contextual_headers else ""
        c["embed_text"] = f"{header}\n{c['text']}".strip()
        c["hash"] = sha(c["level"] + c["embed_text"])
    return out


def upsert_chunks(conn, doc_id, meta, chunks, version):
    existing = {r["chunk_hash"]: r["chunk_id"] for r in conn.execute(
        "SELECT chunk_id, chunk_hash FROM chunks WHERE doc_id=? AND active=1 AND level!='summary'", (doc_id,))}
    keep = set()
    key_to_id = {}
    for c in chunks:                                     # parents come first in the list
        parent_id = key_to_id.get(c["parent"])
        if c["hash"] in existing:
            cid = existing[c["hash"]]
            conn.execute("UPDATE chunks SET parent_id=? WHERE chunk_id=?", (parent_id, cid))
        else:
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, parent_id, level, company, ticker, fiscal_year, quarter, doc_type,"
                " section, page, text, embed_text, chunk_hash, version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, parent_id, c["level"], meta["company"], meta["ticker"], meta["fiscal_year"],
                 meta["quarter"], meta["doc_type"], c["section"], c["page"], c["text"], c["embed_text"],
                 c["hash"], version))
            cid = cur.lastrowid
            conn.execute("INSERT INTO chunks_fts (rowid, embed_text) VALUES (?,?)", (cid, c["embed_text"]))
        keep.add(cid)
        if c["key"]:
            key_to_id[c["key"]] = cid
    for cid in set(existing.values()) - keep:           # changed chunks: retire old version
        conn.execute("UPDATE chunks SET active=0 WHERE chunk_id=?", (cid,))
        conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))


# ---------------------------------------------------------------------------
# Tables -> SQL
# ---------------------------------------------------------------------------
def normalize_metric(label: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label.strip())
    name, unit = (m.group(1), m.group(2)) if m else (label.strip(), "")
    return name.lower(), unit


def store_financials(conn, doc_id, meta, tables):
    for t in tables:
        rows = [r for r in (line.split(" | ") for line in t.text.splitlines())]
        header = rows[0]
        periods = [PERIOD_RE.match(h.strip()) for h in header[1:]]
        if not all(periods):
            continue
        for r in rows[1:]:
            metric, unit = normalize_metric(r[0])
            for cell, p in zip(r[1:], periods):
                vals = numbers_in(cell)
                if not vals:
                    continue
                # A report's own-year figures win over comparatives in later reports ("as reported").
                verb = "INSERT OR REPLACE" if p.group(2) == meta["fiscal_year"] else "INSERT OR IGNORE"
                conn.execute(f"{verb} INTO financials VALUES (?,?,?,?,?,?,?,?,?)",
                             (meta["ticker"], meta["company"], p.group(2), p.group(1) or "FY", metric,
                              vals[0], unit, doc_id, t.page))


# ---------------------------------------------------------------------------
# Knowledge graph (GraphRAG-lite)
# ---------------------------------------------------------------------------
KV_RELATIONS = {"statutory auditor": "audited_by", "chief executive officer": "has_ceo",
                "chief financial officer": "has_cfo", "subsidiaries": "has_subsidiary"}
ENTITY_RE = re.compile(r"\b(?:given )?to ((?:[A-Z][\w&.]*\s)+(?:Private Limited|Limited|LLP))")


def store_graph(conn, doc_id, meta, pages):
    company = meta["company"]

    def add(rel, obj, page):
        conn.execute("INSERT OR IGNORE INTO graph_edges VALUES (?,?,?,?,?,?,?)",
                     (company, rel, obj.strip(), meta["ticker"], meta["fiscal_year"], doc_id, page))

    for p in pages:
        text = "\n".join(b.text for b in p.blocks if b.kind == "text")
        for line in text.splitlines():
            key, _, value = line.partition(":")
            rel = KV_RELATIONS.get(key.strip().lower())
            if rel and value.strip():
                for v in value.split(";"):
                    add(rel, v, p.number)
        for sent in sentences(text.replace("\n", " ")):
            if "related" in sent.lower() or "emphasis of matter" in sent.lower() or "promoter" in sent.lower():
                for ent in ENTITY_RE.findall(sent):
                    flagged = "draw attention" in sent.lower() or "without prior review" in text.lower()
                    add("related_party_flagged_by_auditor" if flagged else "related_party", ent, p.number)
        if llm.enabled():
            triples = llm.complete_json(
                "Extract (subject, relation, object) triples about companies, subsidiaries, auditors, executives "
                f"and related parties from this filing page of {company}. Return a JSON list of 3-item lists.\n\n{text[:3000]}")
            for tr in triples or []:
                if isinstance(tr, list) and len(tr) == 3:
                    conn.execute("INSERT OR IGNORE INTO graph_edges VALUES (?,?,?,?,?,?,?)",
                                 (str(tr[0]), str(tr[1]), str(tr[2]), meta["ticker"], meta["fiscal_year"], doc_id, p.number))


# ---------------------------------------------------------------------------
# Hierarchical summaries (RAPTOR-lite)
# ---------------------------------------------------------------------------
def summarize(texts: list[str], max_sents: int = 4) -> str:
    joined = "\n".join(texts)
    if llm.enabled():
        return llm.complete(f"Summarise the key facts in 3-4 sentences. Keep all numbers exact.\n\n{joined[:6000]}")
    # Extractive fallback: the lead sentence of each passage (skipping boilerplate), in order.
    picked = []
    for t in texts:
        prose = " ".join(l for l in t.splitlines() if " | " not in l)
        for x in sentences(prose):
            if len(x.split()) > 5 and ":" not in x and "table below" not in x.lower() and x not in picked:
                picked.append(x)
                break
    return " ".join(picked[:max_sents]) or joined[:500]


def kmeans(x: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    rng = np.random.default_rng(0)
    centers = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        labels = np.argmax(x @ centers.T, axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = x[labels == j].mean(axis=0)
    return labels


def build_summaries(conn, doc_id, meta, version):
    old = conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=? AND level='summary' AND active=1", (doc_id,)).fetchall()
    for r in old:
        conn.execute("UPDATE chunks SET active=0 WHERE chunk_id=?", (r[0],))
        conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (r[0],))
    rows = conn.execute("SELECT text, page FROM chunks WHERE doc_id=? AND active=1 AND level='child'",
                        (doc_id,)).fetchall()
    if not rows:
        return
    texts = [r["text"] for r in rows]
    k = max(1, math.ceil(len(texts) / 5))
    labels = kmeans(quick_embed(texts), k) if k > 1 else np.zeros(len(texts), dtype=int)
    level1 = []
    for j in sorted(set(labels.tolist())):
        members = [i for i in range(len(texts)) if labels[i] == j]
        level1.append((summarize([texts[i] for i in members]), min(rows[i]["page"] for i in members)))
    level2 = summarize([t for t, _ in level1], max_sents=6) if len(level1) > 1 else None
    period = " ".join(x for x in (meta["quarter"], meta["fiscal_year"]) if x)
    title = f"Summary of {meta['company']} {period} {DOC_TYPE_LABEL.get(meta['doc_type'], '')}".replace("  ", " ").strip()
    items = [(t, p, "Section cluster summary") for t, p in level1]
    if level2:
        items.insert(0, (level2, 1, "Document summary"))
    for text, page, section in items:
        embed_text = f"{title}. {text}"
        cur = conn.execute(
            "INSERT INTO chunks (doc_id, level, company, ticker, fiscal_year, quarter, doc_type, section, page,"
            " text, embed_text, chunk_hash, version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_id, "summary", meta["company"], meta["ticker"], meta["fiscal_year"], meta["quarter"],
             meta["doc_type"], section, page, text, embed_text, sha(embed_text), version))
        conn.execute("INSERT INTO chunks_fts (rowid, embed_text) VALUES (?,?)", (cur.lastrowid, embed_text))
