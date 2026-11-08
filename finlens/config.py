"""All settings and technique flags live here.

Every RAG technique can be switched on/off so the evaluation script can
measure its impact (see scripts/evaluate.py and the ablation table).
Values can be overridden with environment variables, e.g. FINLENS_USE_HYDE=1.
"""
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "finlens.db"
TRACE_PATH = DATA_DIR / "traces.jsonl"


LOCAL_MODEL = "llama3.2:3b"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"   # on Groq's free plan; override with FINLENS_LLM_MODEL


def _provider() -> str:
    """ollama (free local, default) | groq (free hosted API, answer synthesis only) | none (rule-based).
    Anthropic is never selected."""
    p = os.getenv("FINLENS_LLM_PROVIDER", "ollama").strip().lower()
    return p if p in ("none", "groq") else "ollama"


def _paid_model(name: str) -> bool:
    n = (name or "").strip().lower()
    return n.startswith("claude") or "anthropic" in n


def local_model(name: str | None) -> str:
    """An Ollama model name. Claude and Anthropic names are replaced."""
    name = (name or "").strip()
    return LOCAL_MODEL if not name or _paid_model(name) else name


def _default_model() -> str:
    return local_model(os.getenv("FINLENS_LLM_MODEL"))


def _ollama_host() -> str:
    host = os.getenv("FINLENS_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    return host if host.startswith("http") else "http://" + host


def _ollama_ready() -> bool:
    """True only if the Ollama server answers and the chosen model is pulled."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(_ollama_host().rstrip("/") + "/api/tags", timeout=1.5) as r:
            names = {m.get("name") for m in json.load(r).get("models", [])}
    except Exception:
        return False
    model = _default_model()
    return model in names or f"{model}:latest" in names


def _llm_ready() -> bool:
    if _provider() == "ollama":
        return _ollama_ready()
    if _provider() == "groq":
        return bool(os.getenv("GROQ_API_KEY"))     # presence only; the key is read at call time, never stored
    return False


@dataclass
class Settings:
    # --- storage ---
    db_path: str = str(DB_PATH)

    # --- ingestion / chunking ---
    chunk_strategy: str = "recursive"      # fixed | recursive | structure | semantic
    child_tokens: int = 200                 # small chunks that get embedded
    parent_tokens: int = 1500               # big chunks that go to the LLM
    chunk_overlap: int = 30
    contextual_headers: bool = True         # prepend "This chunk is from ..." before embedding
    build_summaries: bool = True            # RAPTOR-lite summary levels
    build_graph: bool = True                # GraphRAG-lite entity/relation table

    # --- retrieval ---
    use_hybrid: bool = True                 # dense + BM25 fused with RRF (off = dense only)
    use_filters: bool = True                # metadata pre-filtering (self-querying)
    use_rerank: bool = True
    use_mmr: bool = True
    use_parent_expansion: bool = True
    use_compression: bool = True
    use_context_ordering: bool = True
    candidates_k: int = 50                  # hybrid candidates before reranking
    final_k: int = 6                        # chunks kept after reranking

    # --- query understanding ---
    use_contextualize: bool = True
    use_routing: bool = True
    use_decomposition: bool = True
    # Off by default: no measured gain on the sample set once filters + reranking are on
    # (see eval/results/ablation.md). Turn on to re-measure on your own corpus.
    use_multi_query: bool = False
    use_hyde: bool = False
    use_step_back: bool = False

    # --- agent ---
    use_agent: bool = True                  # tools + planner (off = single-shot RAG)
    # Free-form tool calling stays off. The deterministic planner runs the tools.
    llm_agent: bool = False
    use_crag: bool = True                   # grade retrieved chunks, retry on failure
    use_reflection: bool = True             # drop claims without supporting evidence
    max_steps: int = 8
    max_cost_usd: float = 0.05              # hard ceiling per agentic query
    allow_web_search: bool = False

    # --- production ---
    use_cache: bool = True
    cache_threshold: float = 0.95

    # --- LLM (optional; everything runs offline without it) ---
    # ollama = free local model (default) | groq = free hosted API (answer synthesis only) | none = rule-based.
    # Anthropic is not used.
    llm_provider: str = field(default_factory=lambda: _provider())
    llm_model: str = field(default_factory=lambda: _default_model())
    ollama_host: str = field(default_factory=lambda: _ollama_host())
    llm_enabled: bool = field(default_factory=lambda: _llm_ready())
    # LLM writes the final answer from the verified draft; kept only if every number and citation checks out.
    # On by default for groq (the one LLM call per question that fits its free token limits), off otherwise.
    use_synthesis: bool = field(default_factory=lambda: _provider() == "groq")

    def __post_init__(self):
        # Every construction path keeps the backend on Ollama, Groq or rule-based "none".
        provider = (self.llm_provider or "").strip().lower()
        self.llm_provider = provider if provider in ("none", "groq") else "ollama"
        if self.llm_provider == "groq" and (not self.llm_model or self.llm_model == LOCAL_MODEL):
            self.llm_model = GROQ_DEFAULT_MODEL
        self.llm_model = local_model(self.llm_model)
        if self.llm_provider != "ollama":
            self.llm_agent = False

    def with_(self, **changes) -> "Settings":
        return replace(self, **changes)


def _from_env(s: Settings) -> Settings:
    changes = {}
    for f in fields(s):
        raw = os.getenv("FINLENS_" + f.name.upper())
        if raw is None:
            continue
        cur = getattr(s, f.name)
        if isinstance(cur, bool):
            changes[f.name] = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            changes[f.name] = int(raw)
        elif isinstance(cur, float):
            changes[f.name] = float(raw)
        else:
            changes[f.name] = raw
    return replace(s, **changes)


settings = _from_env(Settings())
