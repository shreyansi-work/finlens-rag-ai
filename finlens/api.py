"""FastAPI app: JSON + streaming (SSE) endpoints and the web UI.

Run:  uvicorn finlens.api:app --port 8000
"""
import json
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from finlens import llm, tracing
from finlens.config import RAW_DIR, ROOT, local_model, settings
from finlens.service import FinLens
from finlens.streaming import STATUS_TEXT, sse  # noqa: F401  (STATUS_TEXT re-exported for compatibility)
from finlens.text import tidy_page_text
from finlens.workspace_api import router as workspace_router

app = FastAPI(title="FinLens")
app.include_router(workspace_router)
_app: FinLens | None = None
_lock = threading.Lock()


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
    def run(on_event):
        with _lock:
            return service().ask(q, session_id, on_event=on_event)
    return await sse(run)


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
    mode = llm.provider()                       # ollama | groq | offline
    return {"mode": mode, "model": local_model(settings.llm_model), "synthesis": llm.synthesis_enabled(),
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
    return {"doc": doc, "page": page, "sections": [{"section": r["section"], "text": tidy_page_text(r["text"])} for r in rows]}


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
