"""Tools the agent can call. Each returns a plain dict that is JSON-serialisable,
so it can be passed straight back to the LLM as a tool_result.

  search_filings    hybrid retrieval (+ Corrective RAG)
  query_financials  text-to-SQL over extracted tables (read-only, validated)
  graph_lookup      relationship questions
  calculator        all arithmetic (the LLM never does maths itself)
  get_live_quote    current price from a free API
  web_search        fallback outside the corpus, always labelled external
"""
import ast
import json
import operator
import re
import time
import urllib.parse
import urllib.request

from finlens import llm, tracing
from finlens.config import Settings
from finlens.db import connect_readonly
from finlens.query.understanding import FORWARD, FORWARD_TERMS, METRIC_SYNONYMS, find_metrics, step_back
from finlens.retrieval.search import search
from finlens.text import COMPANIES, content_tokens, find_tickers


def cite(row) -> dict:
    return {"doc": row["doc_id"], "page": row["page"]}


# ---------------------------------------------------------------------------
# search_filings + Corrective RAG
# ---------------------------------------------------------------------------
GENERIC = {"annual", "report", "company", "limited", "year", "financial", "fy23", "fy24", "fy25", "fy",
           "q1", "q2", "q3", "q4", "indian", "what", "reported", "overview", "summary", "summarise",
           "summarize", "give", "key", "themes"}
for _info in COMPANIES.values():
    GENERIC.update(t for a in _info["aliases"] + [_info["name"].lower()] for t in a.split())


def key_terms(query: str) -> set[str]:
    terms = {t for t in content_tokens(query) if t not in GENERIC}
    if FORWARD.search(query):
        # "What do they expect next year?" is answered by guidance wording ("we guide ...").
        terms |= FORWARD_TERMS
    # "Is the workforce growing?" is answered by passages that say "headcount".
    for metric in find_metrics(query):
        terms |= set(metric.split())
    return terms


def grade(query: str, hits: list[dict]) -> list[dict]:
    """CRAG grader: keep only chunks relevant to the query."""
    if not hits:
        return []
    if llm.enabled():
        listing = "\n\n".join(f"[{i}] {h['context'][:600]}" for i, h in enumerate(hits))
        keep = llm.complete_json(f"Question: {query}\n\nWhich passages help answer it? "
                                 f"Return a JSON list of passage numbers.\n\n{listing}")
        if isinstance(keep, list):
            return [hits[i] for i in keep if isinstance(i, int) and 0 <= i < len(hits)]
    terms = key_terms(query)
    if not terms:
        return hits
    return [h for h in hits if terms & set(content_tokens(h["embed_text"] + " " + h["context"]))]


def search_filings(conn, s: Settings, query: str, ticker=None, fiscal_years=None, doc_type=None,
                   summaries: bool = False, variants=None) -> dict:
    filters = {"ticker": _as_list(ticker), "fiscal_year": _as_list(fiscal_years), "doc_type": _as_list(doc_type)}
    levels = ("summary", "child") if summaries else ("child", "table")
    hits = search(conn, query, filters, s, variants=variants, levels=levels)
    t = tracing.current()
    if s.use_crag:
        good = grade(query, hits)
        if not good:
            # Retry 1: rewrite the query and relax the year/doc-type filters.
            rewritten = (llm.complete(f"Rewrite this as a better search query for company filings: {query}")
                         if llm.enabled() else f"{query} {step_back(query, filters)}")
            if t:
                t.event("crag_retry", rewritten=rewritten)
            # Company and year are hard constraints; only the doc-type filter is relaxed.
            hits = search(conn, rewritten, {"ticker": filters["ticker"], "fiscal_year": filters["fiscal_year"]},
                          s, levels=levels)
            good = grade(query, hits)
        if not good and s.allow_web_search:
            web = web_search(query)
            return {"ok": web["ok"], "external": True, "results": web.get("results", []), "citations": []}
        hits = good
    results = [{"doc": h["doc_id"], "page": h["page"], "section": h["section"], "text": h["context"],
                "chunk_id": h["chunk_id"], "score": round(h["score"], 3)} for h in hits]
    return {"ok": bool(results), "results": results, "citations": [cite(h) for h in hits]}


def _as_list(x):
    if not x:
        return None
    return x if isinstance(x, list) else [x]


# ---------------------------------------------------------------------------
# query_financials: text-to-SQL with safety checks
# ---------------------------------------------------------------------------
ALLOWED_TABLES = {"financials"}
SCHEMA_DOC = ("financials(ticker TEXT, company TEXT, fiscal_year TEXT like 'FY25', quarter TEXT 'FY' for full "
              "year or 'Q1'..'Q4', metric TEXT lowercase e.g. 'revenue','operating margin','net profit','eps',"
              "'attrition','headcount','total income','net interest margin','gross npa','ebitda margin',"
              "'loan growth', value REAL, unit TEXT, doc_id TEXT, page INTEGER)")
REQUIRED_COLUMNS = {"ticker", "fiscal_year", "metric", "value", "unit", "doc_id", "page"}
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", re.I)


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise ValueError("only one statement allowed")
    if not re.match(r"^\s*select\b", sql, re.I):
        raise ValueError("only SELECT is allowed")
    if FORBIDDEN.search(sql):
        raise ValueError("forbidden keyword")
    tables = set(t.lower() for t in re.findall(r"\b(?:from|join)\s+([a-z_]+)", sql, re.I))
    if not tables or not tables <= ALLOWED_TABLES:
        raise ValueError(f"table not allowed: {tables - ALLOWED_TABLES or 'none'}")
    if not re.search(r"\blimit\b", sql, re.I):
        sql += " LIMIT 200"
    return sql


def run_sql(db_path: str, sql: str, params=(), timeout_s: float = 2.0) -> list[dict]:
    conn = connect_readonly(db_path)                       # read-only connection
    deadline = time.time() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 1000)   # query timeout
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def build_sql(question: str, filters: dict) -> tuple[str, list]:
    """Offline text-to-SQL for the common pattern: metric(s) x company x year(s)."""
    metrics = find_metrics(question) or list(METRIC_SYNONYMS)
    where, args = [f"metric IN ({','.join('?' * len(metrics))})"], list(metrics)
    for col in ("ticker", "fiscal_year"):
        if filters.get(col):
            where.append(f"{col} IN ({','.join('?' * len(filters[col]))})")
            args += filters[col]
    where.append(f"quarter IN ({','.join('?' * len(filters['quarter']))})" if filters.get("quarter") else "quarter = 'FY'")
    args += filters.get("quarter") or []
    sql = f"SELECT * FROM financials WHERE {' AND '.join(where)} ORDER BY ticker, metric, fiscal_year"
    return sql, args


def query_financials(conn, s: Settings, question: str, filters: dict | None = None) -> dict:
    filters = filters or {}
    try:
        rows = []
        if llm.enabled():
            try:
                sql = llm.complete(f"Schema: {SCHEMA_DOC}\nWrite one SQLite SELECT query answering: {question}\n"
                                   f"Output only SQL.").strip().strip("`").removeprefix("sql").strip()
                rows = run_sql(s.db_path, validate_sql(sql))
                # Small models often select only some columns ("SELECT value ..."). The agent needs full rows
                # (fiscal_year, unit, doc_id, page for the answer and its citation), so anything less is rejected.
                if rows and not REQUIRED_COLUMNS <= rows[0].keys():
                    rows = []
            except Exception:       # small models often write invalid SQL: use the rule-based builder instead
                rows = []
        if not rows:
            sql, args = build_sql(question, filters)
            rows = run_sql(s.db_path, validate_sql(sql), args)
    except Exception as e:  # invalid SQL, timeout, etc. are reported, never raised
        return {"ok": False, "error": str(e), "rows": [], "citations": []}
    return {"ok": bool(rows), "sql": sql, "rows": rows,
            "citations": [{"doc": r["doc_id"], "page": r["page"]} for r in rows if "doc_id" in r]}


# ---------------------------------------------------------------------------
# graph_lookup
# ---------------------------------------------------------------------------
RELATION_WORDS = {"auditor": "audited_by", "audit": "audited_by", "subsidiar": "has_subsidiary", "ceo": "has_ceo",
                  "chief executive": "has_ceo", "cfo": "has_cfo", "chief financial": "has_cfo",
                  "flag": "related_party_flagged_by_auditor", "related": "related_party"}


def relation_from_text(text: str) -> str | None:
    low = text.lower()
    if ("related" in low and "flag" in low) or "emphasis of matter" in low or "qualif" in low:
        return "related_party_flagged_by_auditor"
    return next((rel for word, rel in RELATION_WORDS.items() if word in low), None)


def graph_lookup(conn, s: Settings, entity: str | None = None, relation: str | None = None) -> dict:
    where, args = [], []
    if entity:
        tickers = find_tickers(entity)
        if tickers:
            where.append(f"ticker IN ({','.join('?' * len(tickers))})")
            args += tickers
        else:
            where.append("(subject LIKE ? OR object LIKE ?)")
            args += [f"%{entity}%", f"%{entity}%"]
    if relation:
        where.append("relation LIKE ?")
        args.append(f"{relation}%")
    sql = "SELECT * FROM graph_edges" + (f" WHERE {' AND '.join(where)}" if where else "")
    sql += " ORDER BY fiscal_year DESC LIMIT 50"
    rows = [dict(r) for r in conn.execute(sql, args)]
    return {"ok": bool(rows), "edges": rows, "citations": [{"doc": r["doc_id"], "page": r["page"]} for r in rows]}


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression: str) -> dict:
    """Safe arithmetic: numbers and + - * / ** ( ) only."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    try:
        value = ev(ast.parse(expression.replace(",", ""), mode="eval").body)
        return {"ok": True, "expression": expression, "result": round(value, 4)}
    except Exception as e:
        return {"ok": False, "expression": expression, "error": str(e)}


# ---------------------------------------------------------------------------
# get_live_quote / web_search (network; fail safely)
# ---------------------------------------------------------------------------
def get_live_quote(ticker: str) -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}.NS"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as r:
            meta = json.load(r)["chart"]["result"][0]["meta"]
        return {"ok": True, "ticker": ticker, "price": meta["regularMarketPrice"], "currency": meta.get("currency"),
                "source": "Yahoo Finance (external, live)"}
    except Exception as e:
        return {"ok": False, "ticker": ticker, "error": f"live quote unavailable: {type(e).__name__}"}


def web_search(query: str) -> dict:
    url = "https://api.duckduckgo.com/?format=json&no_html=1&q=" + urllib.parse.quote(query)
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.load(r)
        results = [{"text": x.get("Text"), "url": x.get("FirstURL")} for x in data.get("RelatedTopics", []) if x.get("Text")]
        return {"ok": bool(results), "external": True, "results": results[:5]}
    except Exception as e:
        return {"ok": False, "external": True, "error": f"web search unavailable: {type(e).__name__}"}


# ---------------------------------------------------------------------------
# Tool schemas for the LLM (Anthropic tool format)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {"name": "search_filings",
     "description": "Hybrid search over company filings. Use for qualitative info: management commentary, "
                    "reasons, risks, guidance, strategy, auditor remarks.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}, "ticker": {"type": "string"},
         "fiscal_years": {"type": "array", "items": {"type": "string"}},
         "doc_type": {"type": "string", "enum": ["annual_report", "transcript", "results"]},
         "summaries": {"type": "boolean", "description": "search document summaries (broad questions)"}},
         "required": ["query"]}},
    {"name": "query_financials",
     "description": f"Answer numeric questions from extracted financial tables via SQL. Table: {SCHEMA_DOC}",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
    {"name": "graph_lookup",
     "description": "Relationships: auditors, CEO/CFO, subsidiaries, related parties flagged by auditors.",
     "input_schema": {"type": "object", "properties": {
         "entity": {"type": "string"},
         "relation": {"type": "string", "enum": ["audited_by", "has_ceo", "has_cfo", "has_subsidiary",
                                                 "related_party", "related_party_flagged_by_auditor"]}}}},
    {"name": "calculator", "description": "Evaluate arithmetic. ALWAYS use for any computation on numbers.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "get_live_quote", "description": "Current market price for an NSE ticker (live, external).",
     "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "web_search", "description": "Fallback web search for data outside the corpus. Label results as external.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def run_tool(name: str, args: dict, conn, s: Settings) -> dict:
    t = tracing.current()
    with (t.span(f"tool:{name}", args=args) if t else _null()):
        if name == "search_filings":
            return search_filings(conn, s, **args)
        if name == "query_financials":
            return query_financials(conn, s, args["question"], args.get("filters"))
        if name == "graph_lookup":
            return graph_lookup(conn, s, args.get("entity"), args.get("relation"))
        if name == "calculator":
            return calculator(args["expression"])
        if name == "get_live_quote":
            return get_live_quote(args["ticker"])
        if name == "web_search":
            return web_search(args["query"]) if s.allow_web_search else {"ok": False, "error": "web search disabled"}
    return {"ok": False, "error": f"unknown tool {name}"}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
