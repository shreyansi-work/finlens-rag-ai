"""Server-sent events for live status lines followed by the final answer. Shared by the sample-corpus API and the
private upload workspaces so both stream the same way."""
import asyncio
import json
import threading
from collections.abc import Callable

from fastapi.responses import StreamingResponse

STATUS_TEXT = {
    "understand": "Understanding the question",
    "tool:search_filings": "Searching filings",
    "tool:query_financials": "Running SQL on financial tables",
    "tool:graph_lookup": "Looking up relationships",
    "tool:calculator": "Calculating",
    "tool:get_live_quote": "Fetching live quote",
    "tool:web_search": "Searching the web",
    "synthesize": "Writing the answer and checking every number against the sources",
    "crag_retry": "Evidence looked weak, rewriting the query and retrying",
    "injection_blocked": "Ignored instructions hidden inside a document",
    "cache_hit": "Answered from cache",
}


async def sse(run_ask: Callable[[Callable], dict], expose_errors: bool = True) -> StreamingResponse:
    """Run `run_ask(on_event)` in a worker thread and stream its status events, then the answer.
    expose_errors=False replaces exception text with a generic message (used for user-uploaded documents)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(name, data):
        if name in STATUS_TEXT:
            loop.call_soon_threadsafe(queue.put_nowait, ("status", {"text": STATUS_TEXT[name]}))

    def worker():
        try:
            result = run_ask(on_event)
            loop.call_soon_threadsafe(queue.put_nowait, ("answer", result))
        except Exception as e:  # surface errors to the UI instead of hanging
            text = str(e) if expose_errors else "Something went wrong while answering. Please try again."
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"text": text}))

    threading.Thread(target=worker, daemon=True).start()

    async def events():
        while True:
            kind, data = await queue.get()
            yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
            if kind in ("answer", "error"):
                break

    return StreamingResponse(events(), media_type="text/event-stream")
