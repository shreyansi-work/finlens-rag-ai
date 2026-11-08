"""Local LLM wrapper. The only backend is Ollama.

Ollama is a local model. It is free, needs no API key, and no prompt leaves
the machine. Start it with `ollama serve` and pull a model once
(`ollama pull llama3.2:3b`).

The paid Anthropic API is not called. `FINLENS_LLM_PROVIDER=anthropic`,
`FINLENS_ALLOW_PAID`, and `ANTHROPIC_API_KEY` are ignored.

The LLM is optional. If Ollama is not running, or FINLENS_LLM_PROVIDER=none,
every caller falls back to a deterministic offline path, so the project always
runs and the eval stays reproducible.

Responses use Anthropic-style content blocks (`.type` / `.text` / `.name` /
`.input`, plus `.stop_reason` and `.usage`) so the agent loop can stay unchanged.
"""
import json
import re
import urllib.request
from types import SimpleNamespace

from finlens import tracing
from finlens.config import local_model, settings

OLLAMA_NUM_CTX = 8192          # Ollama's default context is small; RAG prompts need more
OLLAMA_TIMEOUT_S = 300         # small local models can be slow on CPU


def enabled() -> bool:
    return settings.llm_enabled and settings.llm_provider == "ollama"


def provider() -> str:
    return "ollama" if enabled() else "offline"


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
# Public API (same for both backends)
# ---------------------------------------------------------------------------
def create(**kwargs):
    """Chat completion with usage tracking. Talks only to the local Ollama server."""
    if settings.llm_provider != "ollama":
        raise RuntimeError("The Anthropic API is disabled. Only the local Ollama backend can run.")
    resp = _create_ollama(**kwargs)
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
