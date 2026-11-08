"""FastAPI app: JSON + streaming (SSE) endpoints and the web UI.

Run:  uvicorn finlens.api:app --port 8000
"""
import asyncio
import json
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from finlens import llm, tracing
from finlens.config import RAW_DIR, ROOT, local_model, settings
from finlens.service import FinLens

app = FastAPI(title="FinLens")
_app: FinLens | None = None
_lock = threading.Lock()

STATUS_TEXT = {
    "understand": "Understanding the question",
    "tool:search_filings": "Searching filings",
    "tool:query_financials": "Running SQL on financial tables",
    "tool:graph_lookup": "Looking up relationships",
    "tool:calculator": "Calculating",
    "tool:get_live_quote": "Fetching live quote",
    "tool:web_search": "Searching the web",
    "crag_retry": "Evidence looked weak, rewriting the query and retrying",
    "injection_blocked": "Ignored instructions hidden inside a document",
    "cache_hit": "Answered from cache",
}


def service() -> FinLens:
    global _app
    if _app is None:
        _app = FinLens()
    return _app


class AskIn(BaseModel):
    question: str
    session_id: str = "web"


class FeedbackIn(BaseModel):
    question: str
    answer: str
    rating: int              # 1 = thumbs up, -1 = thumbs down
    reason: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.post("/api/ask")
def ask(body: AskIn):
    with _lock:
        return service().ask(body.question, body.session_id)


@app.get("/api/ask/stream")
async def ask_stream(q: str, session_id: str = "web"):
    """Server-sent events: live status lines, then the final answer."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(name, data):
        if name in STATUS_TEXT:
            loop.call_soon_threadsafe(queue.put_nowait, ("status", {"text": STATUS_TEXT[name]}))

    def worker():
        try:
            with _lock:
                result = service().ask(q, session_id, on_event=on_event)
            loop.call_soon_threadsafe(queue.put_nowait, ("answer", result))
        except Exception as e:  # surface errors to the UI instead of hanging
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"text": str(e)}))

    threading.Thread(target=worker, daemon=True).start()

    async def events():
        while True:
            kind, data = await queue.get()
            yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
            if kind in ("answer", "error"):
                break

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    service().feedback(body.question, body.answer, body.rating, body.reason)
    if body.rating < 0:   # bad answers become candidates for the golden set
        with open(ROOT / "eval" / "feedback_candidates.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(body.model_dump()) + "\n")
    return {"ok": True}


@app.get("/api/info")
def info():
    c = service().conn
    count = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
    mode = "ollama" if llm.provider() == "ollama" else "offline"
    return {"mode": mode, "model": local_model(settings.llm_model),
            "documents": count("SELECT COUNT(*) FROM documents"),
            "chunks": count("SELECT COUNT(*) FROM chunks WHERE active=1"),
            "financial_rows": count("SELECT COUNT(*) FROM financials"),
            "graph_edges": count("SELECT COUNT(*) FROM graph_edges")}


@app.get("/api/documents")
def documents():
    rows = service().conn.execute(
        "SELECT doc_id, company, ticker, fiscal_year, quarter, doc_type, num_pages, version FROM documents "
        "ORDER BY ticker, fiscal_year, doc_type").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/page")
def page(doc: str, page: int):
    rows = service().conn.execute(
        "SELECT section, text FROM chunks WHERE doc_id=? AND page=? AND active=1 AND level='parent' ORDER BY chunk_id",
        (doc, page)).fetchall()
    return {"doc": doc, "page": page, "sections": [{"section": r["section"], "text": _tidy(r["text"])} for r in rows]}


def _tidy(text: str) -> str:
    """Re-join lines that the PDF wrapped, keeping headings and table rows on their own lines."""
    blocks = []
    for block in text.split("\n\n"):
        keep, prose = [], []
        for line in block.splitlines():
            if line.startswith("## ") or " | " in line:
                if prose:
                    keep.append(" ".join(prose))
                    prose = []
                keep.append(line)
            else:
                prose.append(line.strip())
        if prose:
            keep.append(" ".join(prose))
        blocks.append("\n".join(keep))
    return "\n\n".join(blocks)


@app.get("/api/traces")
def traces(n: int = 20):
    return tracing.recent(n)


@app.get("/files/{name}")
def pdf(name: str):
    path = (RAW_DIR / name).resolve()
    if path.parent != RAW_DIR.resolve() or not path.exists() or path.suffix != ".pdf":
        raise HTTPException(404)
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/eval")
def eval_results():
    p = ROOT / "eval" / "results" / "latest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
