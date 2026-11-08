"""LLM wrapper with two free backends.

Ollama is a local model. It is free, needs no API key, and no prompt leaves
the machine. Start it with `ollama serve` and pull a model once
(`ollama pull llama3.2:3b`). It powers the helper steps (rewriting, grading, text-to-SQL).

Groq (FINLENS_LLM_PROVIDER=groq, key in GROQ_API_KEY) is a free hosted API used for ONE call per
question: writing the final answer, which finlens/synthesis.py then verifies. It is rate limited,
so any 429 / timeout / bad key trips a circuit breaker and the deterministic answer is used instead.

The paid Anthropic API is not called. `FINLENS_LLM_PROVIDER=anthropic`,
`FINLENS_ALLOW_PAID`, and `ANTHROPIC_API_KEY` are ignored.

The LLM is optional. If Ollama is not running, or FINLENS_LLM_PROVIDER=none,
every caller falls back to a deterministic offline path, so the project always
runs and the eval stays reproducible.

Responses use Anthropic-style content blocks (`.type` / `.text` / `.name` /
`.input`, plus `.stop_reason` and `.usage`) so the agent loop can stay unchanged.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

from finlens import tracing
from finlens.config import local_model, settings

OLLAMA_NUM_CTX = 8192          # Ollama's default context is small; RAG prompts need more
OLLAMA_TIMEOUT_S = 300         # small local models can be slow on CPU
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
REMOTE_TIMEOUT_S = 25          # a hosted API that is slow is treated as down: the demo must not hang


class LLMUnavailable(RuntimeError):
    """The hosted LLM is rate-limited, down or misconfigured. Callers fall back to the deterministic path."""


# Circuit breaker for the hosted API: after a failure we stop calling it for a while.
_breaker_until = 0.0
_last_call = 0.0


def _open_breaker(seconds: float) -> None:
    global _breaker_until
    _breaker_until = time.monotonic() + seconds


def breaker_open() -> bool:
    return time.monotonic() < _breaker_until


def enabled() -> bool:
    """Helper steps (rewriting, grading, text-to-SQL) use the local Ollama model only."""
    return settings.llm_enabled and settings.llm_provider == "ollama"


def groq_ready() -> bool:
    return settings.llm_provider == "groq" and bool(os.environ.get("GROQ_API_KEY")) and not breaker_open()


def synthesis_enabled() -> bool:
    """Should the LLM write the final answer? Groq: needs a key and a closed breaker. Ollama: explicit opt-in."""
    if not settings.use_synthesis:
        return False
    return groq_ready() if settings.llm_provider == "groq" else enabled()


def provider() -> str:
    if enabled():
        return "ollama"
    if settings.llm_provider == "groq" and os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "offline"


# ---------------------------------------------------------------------------
# Ollama backend (converts Anthropic-style messages/tools to Ollama's chat API)
# ---------------------------------------------------------------------------
def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _to_ollama_messages(system: str, messages: list) -> list[dict]:
    out = [{"role": "system", "content": system}] if system else []
    names: dict[str, str] = {}                      # tool_use id -> tool name
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = "".join(_get(b, "text", "") for b in content if _get(b, "type") == "text")
            calls = []
            for b in content:
                if _get(b, "type") == "tool_use":
                    names[_get(b, "id")] = _get(b, "name")
                    calls.append({"function": {"name": _get(b, "name"), "arguments": _get(b, "input") or {}}})
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:                                        # user turn carrying tool results
            for b in content:
                if _get(b, "type") == "tool_result":
                    out.append({"role": "tool", "tool_name": names.get(_get(b, "tool_use_id"), ""),
                                "content": str(_get(b, "content", ""))})
                elif _get(b, "type") == "text":
                    out.append({"role": "user", "content": _get(b, "text", "")})
    return out


def _to_ollama_tools(tools: list | None) -> list[dict] | None:
    if not tools:
        return None
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
            for t in tools]


def _create_ollama(system: str = "", messages: list | None = None, tools: list | None = None,
                   max_tokens: int = 1024, **_ignored):
    body = {"model": local_model(settings.llm_model), "stream": False,
            "messages": _to_ollama_messages(system, messages or []),
            "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": max_tokens, "temperature": 0}}
    if tools:
        body["tools"] = _to_ollama_tools(tools)
    req = urllib.request.Request(settings.ollama_host.rstrip("/") + "/api/chat",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as r:
        data = json.load(r)

    msg = data.get("message", {})
    blocks = []
    if msg.get("content"):
        blocks.append(SimpleNamespace(type="text", text=msg["content"]))
    for i, call in enumerate(msg.get("tool_calls") or []):
        fn = call.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):                    # some models return the arguments as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        blocks.append(SimpleNamespace(type="tool_use", id=f"call_{i}_{fn.get('name', '')}",
                                      name=fn.get("name", ""), input=args))
    has_tool = any(b.type == "tool_use" for b in blocks)
    return SimpleNamespace(
        content=blocks, stop_reason="tool_use" if has_tool else "end_turn",
        usage=SimpleNamespace(input_tokens=data.get("prompt_eval_count", 0),
                              output_tokens=data.get("eval_count", 0)))


# ---------------------------------------------------------------------------
# Groq backend (free hosted API, OpenAI-compatible). Plain text chat only.
# The key is read from the environment at call time and never stored, logged or put in a trace.
# ---------------------------------------------------------------------------
def _throttle() -> None:
    """Optional spacing between calls (FINLENS_LLM_MIN_INTERVAL_S) so long evals stay under token-per-minute caps."""
    global _last_call
    gap = float(os.environ.get("FINLENS_LLM_MIN_INTERVAL_S", "0") or 0)
    wait = _last_call + gap - time.monotonic()
    if gap and wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _create_groq(system: str = "", messages: list | None = None, max_tokens: int = 1024, **_ignored):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMUnavailable("GROQ_API_KEY is not set")
    if breaker_open():
        raise LLMUnavailable("hosted LLM temporarily disabled after an error")
    chat = ([{"role": "system", "content": system}] if system else [])
    for m in messages or []:
        if not isinstance(m["content"], str):
            raise LLMUnavailable("the hosted backend supports plain text messages only")
        chat.append({"role": m["role"], "content": m["content"]})
    body = {"model": settings.llm_model, "messages": chat, "temperature": 0, "max_completion_tokens": max_tokens}
    req = urllib.request.Request(
        GROQ_BASE_URL + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "finlens/1.0"})   # the default urllib agent is blocked by Groq's edge
    _throttle()
    try:
        with urllib.request.urlopen(req, timeout=REMOTE_TIMEOUT_S) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            try:
                wait = float(e.headers.get("retry-after", 30))
            except (TypeError, ValueError):
                wait = 30.0
            _open_breaker(min(max(wait, 5.0), 120.0))
        elif e.code in (401, 403):
            _open_breaker(600.0)             # bad or revoked key: stop hammering the API
        else:
            _open_breaker(30.0)
        raise LLMUnavailable(f"hosted LLM returned HTTP {e.code}") from None   # never include headers or the key
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        _open_breaker(30.0)
        raise LLMUnavailable("hosted LLM unreachable or returned invalid JSON") from None

    choices = data.get("choices") or [{}]
    text = ((choices[0].get("message") or {}).get("content")) or ""
    usage = data.get("usage") or {}
    blocks = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(content=blocks, stop_reason="end_turn",
                           usage=SimpleNamespace(input_tokens=usage.get("prompt_tokens", 0),
                                                 output_tokens=usage.get("completion_tokens", 0)))


# ---------------------------------------------------------------------------
# Public API (same for all backends)
# ---------------------------------------------------------------------------
def create(**kwargs):
    """Chat completion with usage tracking. Only Ollama (local) and Groq (free hosted) can run."""
    if settings.llm_provider == "groq":
        resp = _create_groq(**kwargs)
    elif settings.llm_provider == "ollama":
        resp = _create_ollama(**kwargs)
    else:
        raise RuntimeError("No LLM backend is enabled (the Anthropic API is never used).")
    t = tracing.current()
    if t:
        t.add_usage(resp.usage.input_tokens, resp.usage.output_tokens)
    return resp


def complete(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    resp = create(system=system, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if resp.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def complete_json(prompt: str, system: str = "", max_tokens: int = 1024):
    """Ask for JSON and parse the first JSON object/array in the reply."""
    text = complete(prompt + "\n\nReply with JSON only.", system, max_tokens)
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None
