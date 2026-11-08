"""HTTP endpoints for private upload workspaces (see finlens/workspaces.py).

Identity is an unguessable random id in an HttpOnly, SameSite=Lax cookie; there are no accounts. Every response
is `Cache-Control: no-store`. Uploads are sent as the raw request body (no multipart dependency) with the file
name in the X-Filename header, streamed and size-limited before anything is parsed.
"""
from urllib.parse import unquote

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from finlens import llm
from finlens.streaming import sse
from finlens.workspaces import MAX_FILE_BYTES, TTL_SECONDS, UploadError, Workspace, get_manager

router = APIRouter(prefix="/api/workspace")
COOKIE = "fl_ws"


class WorkspaceAsk(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    allow_llm: bool = False          # explicit opt-in: may short excerpts go to the hosted LLM?


def _secure(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _fail(e: UploadError):
    return HTTPException(status_code=e.status, detail=e.message)


def _existing(request: Request) -> Workspace:
    ws = get_manager().get(request.cookies.get(COOKIE))
    if ws is None:
        raise HTTPException(status_code=404, detail="No workspace yet. Reload the page.")
    return ws


def _get_or_create(request: Request, response: Response) -> Workspace:
    _no_store(response)
    ws = get_manager().get(request.cookies.get(COOKIE))
    if ws is None:
        try:
            ws = get_manager().create()
        except UploadError as e:
            raise _fail(e) from None
        response.set_cookie(COOKIE, ws.id, max_age=TTL_SECONDS, httponly=True, samesite="lax",
                            secure=_secure(request), path="/")
    return ws


@router.get("")
def workspace_status(request: Request, response: Response):
    _no_store(response)
    ws = get_manager().get(request.cookies.get(COOKIE))      # read-only: never creates a workspace
    return {**get_manager().status(ws), "synthesis": llm.synthesis_enabled()}


@router.post("/upload", status_code=202)
async def upload(request: Request, response: Response, x_filename: str = Header(default="")):
    ws = _get_or_create(request, response)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="That file is too large for this demo.")
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="That file is too large for this demo.")
    try:
        doc = await run_in_threadpool(get_manager().add_document, ws, unquote(x_filename)[:200], bytes(buf))
    except UploadError as e:
        raise _fail(e) from None
    return doc.public()


@router.delete("/docs/{doc_id}")
async def delete_document(doc_id: str, request: Request, response: Response):
    _no_store(response)
    ws = _existing(request)
    if not await run_in_threadpool(get_manager().delete_document, ws, doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"ok": True}


@router.delete("")
def delete_workspace(request: Request, response: Response):
    _no_store(response)
    ws = get_manager().get(request.cookies.get(COOKIE))
    if ws:
        get_manager().delete(ws.id)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.post("/ask")
async def ask(body: WorkspaceAsk, request: Request, response: Response):
    _no_store(response)
    ws = _existing(request)
    try:
        return await run_in_threadpool(get_manager().ask, ws, body.question, body.allow_llm)
    except UploadError as e:
        raise _fail(e) from None


@router.post("/ask/stream")
async def ask_stream(body: WorkspaceAsk, request: Request):
    """Streams status lines then the answer. POST (not GET) so the question travels in the body and never appears
    in a URL, where access logs and proxies would record it."""
    ws = _existing(request)
    if not ws.ready_docs():
        raise HTTPException(status_code=409, detail="Upload a document first.")
    resp = await sse(lambda on_event: get_manager().ask(ws, body.question, body.allow_llm, on_event),
                     expose_errors=False)
    _no_store(resp)
    return resp


@router.get("/page")
def page(doc: str, page: int, request: Request, response: Response):
    _no_store(response)
    data = get_manager().page(_existing(request), doc, page)
    if data is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return data


@router.get("/file/{doc_id}")
def file(doc_id: str, request: Request):
    path = get_manager().file_path(_existing(request), doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="File not available.")
    return FileResponse(path, media_type="application/pdf",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Content-Disposition": 'inline; filename="document.pdf"'})
