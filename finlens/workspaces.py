"""Private, anonymous, auto-expiring workspaces for user-uploaded documents (PDF / DOCX).

No accounts: a visitor's browser gets a random unguessable workspace id in an HttpOnly cookie. Everything a
visitor uploads lives in its own folder with its own SQLite database and vector index, so one visitor can never
search another's documents. Workspaces expire after TTL_SECONDS of inactivity, are wiped when the server starts,
and can be deleted by the user at any time.

Hard limits (constants below) protect a small free host: file size, page count, number of files, uploads per
minute, number of workspaces, and a zip-bomb guard for DOCX. File contents are validated, never trusted from
the extension or the client's file name; stored files get server-generated names.
"""
import hashlib
import io
import logging
import os
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from finlens import llm
from finlens.config import DATA_DIR, settings
from finlens.db import connect
from finlens.ingestion.parser import MAX_DOCX_XML_BYTES, DocumentError
from finlens.ingestion.pipeline import ingest_file
from finlens.retrieval import index
from finlens.service import FinLens
from finlens.text import tidy_page_text

log = logging.getLogger("finlens.workspaces")

ROOT_DIR = DATA_DIR / "workspaces"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_FILES = 3
MAX_PDF_PAGES = 80
TTL_SECONDS = 3 * 3600
MAX_WORKSPACES = 40
MAX_UPLOADS_PER_MINUTE = 5
MAX_ZIP_ENTRIES = 2000
MAX_ZIP_RATIO = 200
SYNC = False                    # tests set this to ingest inline instead of in the background thread

ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
_INGEST_LOCK = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


class UploadError(Exception):
    """A rejected upload or request. `status` is the HTTP status; `message` is safe to show to the user."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class DocInfo:
    doc_id: str
    title: str
    kind: str                   # pdf | docx
    size: int
    status: str = "processing"  # processing | ready | error
    error: str = ""
    pages: int = 0
    digest: str = ""            # sha256 of the content: the same file under another name is a duplicate

    def public(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "kind": self.kind, "size": self.size,
                "status": self.status, "error": self.error, "pages": self.pages}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def clean_title(filename: str) -> str:
    stem = Path(re.sub(r"[\x00-\x1f\x7f/\\]", " ", filename or "")).stem
    return re.sub(r"\s+", " ", stem).strip()[:80] or "Document"


def detect_kind(filename: str, data: bytes) -> str:
    """Decide pdf / docx from the content and require the extension to agree. Raises UploadError otherwise."""
    ext = Path(filename or "").suffix.lower()
    if ext not in (".pdf", ".docx"):
        raise UploadError(415, "Only PDF and DOCX files are supported.")
    if ext == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise UploadError(415, "That file is not a valid PDF.")
        return "pdf"
    if not data.startswith(b"PK\x03\x04"):
        raise UploadError(415, "That file is not a valid DOCX.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            entries = zf.infolist()
            names = {e.filename for e in entries}
            if len(entries) > MAX_ZIP_ENTRIES or "word/document.xml" not in names:
                raise UploadError(415, "That file is not a valid DOCX.")
            total = sum(e.file_size for e in entries)
            if zf.getinfo("word/document.xml").file_size > MAX_DOCX_XML_BYTES or total > MAX_ZIP_RATIO * len(data):
                raise UploadError(413, "This DOCX expands to an unreasonable size.")   # zip-bomb guard
    except zipfile.BadZipFile:
        raise UploadError(415, "That file is not a valid DOCX.") from None
    return "docx"


def check_pdf(data: bytes) -> None:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadError(422, "Password-protected PDFs are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise UploadError(413, f"PDFs are limited to {MAX_PDF_PAGES} pages on this demo.")
    except UploadError:
        raise
    except Exception:
        raise UploadError(422, "That PDF could not be read.") from None


# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, ws_id: str, path: Path):
        self.id, self.path = ws_id, path
        self.created = self.last_used = time.time()
        self.docs: dict[str, DocInfo] = {}
        self.upload_times: list[float] = []
        self.lock = threading.RLock()          # guards docs / uploads
        self.ask_lock = threading.Lock()       # one question at a time per workspace
        self._app: FinLens | None = None
        (path / "raw").mkdir(parents=True, exist_ok=True)

    @property
    def raw(self) -> Path:
        return self.path / "raw"

    @property
    def db_path(self) -> str:
        return str(self.path / "ws.db")

    def settings(self):
        # An LLM may answer from the document when it is a hosted one (fast; needs the visitor's consent, see ask()).
        # A local Ollama model on a CPU takes 1-3 minutes per answer and was measured to be worse than the
        # deterministic answer, so it is opt-in: FINLENS_WORKSPACE_LLM=1.
        use_llm = llm.is_remote() or os.getenv("FINLENS_WORKSPACE_LLM") == "1"
        return settings.with_(db_path=self.db_path, generic_docs=True, use_filters=False, use_decomposition=False,
                              use_contextualize=False, use_multi_query=False, use_hyde=False, use_step_back=False,
                              build_graph=False, build_summaries=False, llm_agent=False, use_synthesis=use_llm)

    def app(self) -> FinLens:
        with self.lock:
            if self._app is None:
                self._app = FinLens(self.settings())
            return self._app

    def reset_app(self) -> None:
        with self.lock:
            if self._app is not None:
                try:
                    self._app.conn.close()
                except Exception:
                    pass
                self._app = None

    def touch(self) -> None:
        self.last_used = time.time()

    def expires_in(self) -> int:
        return max(0, int(TTL_SECONDS - (time.time() - self.last_used)))

    def ready_docs(self) -> list[DocInfo]:
        return [d for d in self.docs.values() if d.status == "ready"]


class Manager:
    def __init__(self, root: Path):
        self.root = root
        self.items: dict[str, Workspace] = {}
        self.lock = threading.RLock()
        shutil.rmtree(root, ignore_errors=True)      # nothing from a previous run is kept
        root.mkdir(parents=True, exist_ok=True)

    # --- lifecycle ---------------------------------------------------------
    def create(self) -> Workspace:
        import secrets
        with self.lock:
            self.cleanup()
            if len(self.items) >= MAX_WORKSPACES:
                raise UploadError(503, "The demo is busy right now. Please try again in a little while.")
            ws_id = secrets.token_urlsafe(24)
            ws = Workspace(ws_id, self.root / ws_id)
            self.items[ws_id] = ws
            return ws

    def get(self, ws_id: str | None) -> Workspace | None:
        if not ws_id or not ID_RE.match(ws_id):
            return None
        with self.lock:
            ws = self.items.get(ws_id)
            if ws and time.time() - ws.last_used > TTL_SECONDS:
                self.delete(ws_id)
                return None
            if ws:
                ws.touch()
            return ws

    def cleanup(self) -> None:
        with self.lock:
            now = time.time()
            for ws_id, ws in list(self.items.items()):
                if now - ws.last_used > TTL_SECONDS:
                    self.delete(ws_id)

    def delete(self, ws_id: str) -> None:
        with self.lock:
            ws = self.items.pop(ws_id, None)
        if ws:
            ws.reset_app()
            shutil.rmtree(ws.path, ignore_errors=True)

    # --- documents ---------------------------------------------------------
    def add_document(self, ws: Workspace, filename: str, data: bytes) -> DocInfo:
        now = time.time()
        with ws.lock:
            ws.upload_times = [t for t in ws.upload_times if now - t < 60]
            if len(ws.upload_times) >= MAX_UPLOADS_PER_MINUTE:
                raise UploadError(429, "Too many uploads. Please wait a minute.")
        if not data:
            raise UploadError(400, "That file is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise UploadError(413, f"Files are limited to {MAX_FILE_BYTES // (1024 * 1024)} MB on this demo.")
        kind = detect_kind(filename, data)
        if kind == "pdf":
            check_pdf(data)
        title = clean_title(filename)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:30].strip("_") or "doc"
        digest = hashlib.sha256(data).hexdigest()
        doc_id = f"{slug}_{digest[:6]}"                                    # [A-Za-z0-9_]+ : valid in citations
        with ws.lock:
            if doc_id in ws.docs or any(d.digest == digest and d.status != "error" for d in ws.docs.values()):
                raise UploadError(409, "You have already uploaded this document.")
            active = [d for d in ws.docs.values() if d.status != "error"]
            if len(active) >= MAX_FILES:
                raise UploadError(409, f"You can keep up to {MAX_FILES} documents. Delete one first.")
            (ws.raw / f"{doc_id}.{kind}").write_bytes(data)
            doc = DocInfo(doc_id=doc_id, title=title, kind=kind, size=len(data), digest=digest)
            ws.docs[doc_id] = doc
            ws.upload_times.append(now)
        if SYNC:
            self._ingest(ws, doc)
        else:
            _executor.submit(self._ingest, ws, doc)
        return doc

    def _ingest(self, ws: Workspace, doc: DocInfo) -> None:
        path = ws.raw / f"{doc.doc_id}.{doc.kind}"
        try:
            with _INGEST_LOCK:
                if doc.doc_id not in ws.docs:
                    return                                   # deleted while queued
                s = ws.settings()
                conn = connect(s.db_path)
                try:
                    ingest_file(conn, path, s, meta={"company": doc.title})
                    row = conn.execute("SELECT num_pages FROM documents WHERE doc_id=?", (doc.doc_id,)).fetchone()
                    index.build_vectors(conn, s)
                finally:
                    conn.close()
            with ws.lock:
                doc.pages = int(row["num_pages"] or 0) if row else 0
                doc.status = "ready"
                ws.reset_app()
        except DocumentError as e:
            self._fail(ws, doc, str(e))
        except Exception as e:
            log.error("ingestion failed: %s", type(e).__name__)    # no traceback: paths contain the file's name
            self._fail(ws, doc, "That file could not be processed.")

    def _fail(self, ws: Workspace, doc: DocInfo, message: str) -> None:
        with ws.lock:
            doc.status, doc.error = "error", message
        (ws.raw / f"{doc.doc_id}.{doc.kind}").unlink(missing_ok=True)

    def delete_document(self, ws: Workspace, doc_id: str) -> bool:
        with ws.lock:
            doc = ws.docs.get(doc_id)
            if not doc:
                return False
        with _INGEST_LOCK:
            with ws.lock:
                ws.docs.pop(doc_id, None)
            conn = connect(ws.db_path)
            try:
                ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))]
                conn.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
                conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
                conn.execute("DELETE FROM document_versions WHERE doc_id=?", (doc_id,))
                conn.commit()
                remaining = conn.execute("SELECT COUNT(*) FROM chunks WHERE active=1").fetchone()[0]
                if remaining:
                    index.build_vectors(conn, ws.settings())
                else:
                    for p in (Path(ws.db_path).with_suffix(".vectors.npz"), Path(ws.db_path).with_suffix(".embedder.npz")):
                        p.unlink(missing_ok=True)
                    index.load.cache_clear()
            finally:
                conn.close()
        (ws.raw / f"{doc_id}.{doc.kind}").unlink(missing_ok=True)
        ws.reset_app()
        return True

    # --- queries -----------------------------------------------------------
    def ask(self, ws: Workspace, question: str, allow_llm: bool, on_event=None) -> dict:
        if not ws.ready_docs():
            raise UploadError(409, "Upload a document first.")
        # A local LLM (Ollama) keeps data on this machine, so it needs no consent; a hosted one (Groq) does.
        may_use_llm = allow_llm or not llm.is_remote()
        with ws.ask_lock:
            return ws.app().ask(question, session_id=f"ws-{ws.id}", on_event=on_event, use_memory=False,
                                allow_llm=may_use_llm, persist_trace=False)

    def page(self, ws: Workspace, doc_id: str, page: int) -> dict | None:
        if doc_id not in ws.docs:
            return None
        conn = connect(ws.db_path)
        try:
            rows = conn.execute(
                "SELECT section, text FROM chunks WHERE doc_id=? AND page=? AND active=1 AND level='parent' "
                "ORDER BY chunk_id", (doc_id, page)).fetchall()
        finally:
            conn.close()
        return {"doc": doc_id, "page": page,
                "sections": [{"section": r["section"], "text": tidy_page_text(r["text"])} for r in rows]}

    def file_path(self, ws: Workspace, doc_id: str) -> Path | None:
        doc = ws.docs.get(doc_id)
        if not doc or doc.kind != "pdf" or doc.status != "ready":
            return None
        path = ws.raw / f"{doc_id}.pdf"
        return path if path.exists() else None

    def status(self, ws: Workspace | None) -> dict:
        """State for the UI. With no workspace yet (nothing uploaded) it is empty: workspaces are only created by
        an upload, so merely visiting the page cannot use up the limited number of slots."""
        docs = []
        if ws:
            with ws.lock:
                docs = [d.public() for d in ws.docs.values()]
        return {"docs": docs, "expires_in_s": ws.expires_in() if ws else 0,
                "limits": {"max_files": MAX_FILES, "max_mb": MAX_FILE_BYTES // (1024 * 1024),
                           "max_pdf_pages": MAX_PDF_PAGES, "ttl_hours": TTL_SECONDS // 3600}}


_manager: Manager | None = None


def get_manager() -> Manager:
    global _manager
    if _manager is None:
        _manager = Manager(ROOT_DIR)
    return _manager
