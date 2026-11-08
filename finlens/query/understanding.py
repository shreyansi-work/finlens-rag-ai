"""Query understanding: contextualize, route, extract filters, decompose,
expand (multi-query), HyDE and step-back.

Every function has an offline rule-based version and, when an API key is set,
an LLM version. Each is switchable in Settings so its effect can be measured.
"""
import re
from dataclasses import dataclass, field

from finlens import llm
from finlens.config import Settings
from finlens.text import COMPANIES, company_name, content_tokens, find_tickers, possessive

METRIC_SYNONYMS = {
    "revenue": ["revenue", "sales", "turnover", "top line", "topline"],
    "total income": ["total income"],
    "operating margin": ["operating margin", "ebit margin", "margins", "margin"],
    "ebitda margin": ["ebitda margin", "ebitda"],
    "net profit": ["net profit", "profit after tax", "pat", "profit", "profitable", "profitability"],
    "eps": ["eps", "earnings per share"],
    "attrition": ["attrition", "employee turnover", "churn", "leaving", "quit", "quitting", "resign",
                  "resigned", "resignations"],
    "headcount": ["headcount", "employees", "workforce", "staff", "people work", "how many people",
                  "employee count"],
    "net interest margin": ["net interest margin", "nim"],
    "gross npa": ["gross npa", "npa", "non-performing", "bad loans", "bad loan"],
    "loan growth": ["loan growth", "credit growth"],
}
ROUTES = ("sql", "vector", "graph", "live", "summary", "direct")
# Forward-looking questions are answered by management guidance (calls, results), not report boilerplate.
FORWARD = re.compile(r"\b(guid\w*|expect\w*|forecast\w*|outlook|going forward|"
                     r"(next|coming|upcoming) (fiscal |financial )?(year|quarter))\b", re.I)
FORWARD_TERMS = {"guide", "guided", "guidance", "outlook", "expect", "expects", "expected", "forecast"}


@dataclass
class QueryPlan:
    original: str
    query: str                                   # contextualized standalone question
    route: str = "vector"
    filters: dict = field(default_factory=dict)
    metrics: list = field(default_factory=list)
    sub_queries: list = field(default_factory=list)
    variants: list = field(default_factory=list)  # multi-query / HyDE / step-back texts
    wants_reasons: bool = False
    forward_looking: bool = False
    known_years: list = field(default_factory=list)


# ---------------------------------------------------------------------------
def contextualize(question: str, history: list[dict]) -> str:
    """'What about Nilgiri?' -> 'What was Nilgiri's operating margin in FY25?'"""
    last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
    if not last_user:
        return question
    if llm.enabled():
        out = llm.complete(f"Chat history:\n{_fmt(history)}\n\nRewrite the follow-up question as a standalone "
                           f"question. Output only the question.\n\nFollow-up: {question}")
        return out or question
    q_tickers, prev_tickers = find_tickers(question), find_tickers(last_user)
    alias_words = {w for t in q_tickers for a in COMPANIES[t]["aliases"] for w in a.split()}
    rest = [t for t in content_tokens(question)
            if t not in alias_words and not re.fullmatch(r"fy\d\d|q[1-4]|s", t)]
    # A follow-up only names a new company or year: "What about Nilgiri?", "And FY24?"
    is_followup = not rest or bool(re.match(r"^\s*((what|how) about|and|same for)\b", question.lower())) and len(rest) <= 1
    if not is_followup:
        return question
    new = last_user
    if q_tickers and prev_tickers:
        for alias in COMPANIES[prev_tickers[0]]["aliases"] + [company_name(prev_tickers[0])]:
            name = company_name(q_tickers[0])
            new = re.sub(rf"\b{re.escape(alias)}('s)?\b", lambda m: possessive(name) if m.group(1) else name,
                         new, flags=re.I)
    new_year = re.search(r"\bFY\d\d\b", question, re.I)
    if new_year and re.search(r"\bFY\d\d\b", new):
        new = re.sub(r"\bFY\d\d\b", new_year.group(0).upper(), new)
    return new


def _fmt(history):
    return "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])


def extract_filters(question: str, known_years: list[str]) -> dict:
    """Self-querying: natural language -> metadata filters."""
    q = question.lower()
    f: dict = {}
    tickers = find_tickers(question)
    if tickers:
        f["ticker"] = tickers
    years = sorted({f"FY{m[-2:]}" for m in re.findall(r"\bfy\s?'?(\d{2,4})\b", q)})
    years += sorted({f"FY{m[-2:]}" for m in re.findall(r"\b20\d\d-(\d\d)\b", q)})
    m = re.search(r"last (\d+|two|three|four|five) (?:fiscal )?years", q)
    if m:
        n = {"two": 2, "three": 3, "four": 4, "five": 5}.get(m.group(1)) or int(m.group(1))
        years = known_years[-n:]
    if re.search(r"\bfrom (fy\s?\d\d) to (fy\s?\d\d)\b", q) and len(years) == 2:
        lo, hi = sorted(years)
        years = [y for y in known_years if lo <= y <= hi]
    if years:
        f["fiscal_year"] = sorted(set(years))
    quarter = re.findall(r"\b(q[1-4])\b", q)
    if quarter:
        f["quarter"] = sorted({x.upper() for x in quarter})
    if FORWARD.search(q) or re.search(r"earnings call|transcript", q):
        f["doc_type"] = ["transcript", "results"]
    return f


def find_metrics(question: str) -> list[str]:
    q = question.lower()
    found = []
    for metric, words in METRIC_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            found.append(metric)
    # 'operating margin' also matches 'margin'; 'ebitda margin' beats generic margin, etc.
    if "ebitda margin" in found or "net interest margin" in found:
        found = [m for m in found if m != "operating margin"]
    if "earnings per share" in q or re.search(r"\beps\b", q):
        found = [m for m in found if m != "net profit"]
    # "employees leaving" is about attrition, not headcount.
    if "attrition" in found and "headcount" not in q:
        found = [m for m in found if m != "headcount"]
    return found


def route(question: str, metrics: list[str]) -> str:
    q = question.lower()
    if re.fullmatch(r"\s*(hi|hello|hey|thanks|thank you|ok|okay)[!. ]*", q):
        return "direct"
    if re.search(r"share price|stock price|current price|live|quote|p/e|pe ratio|market cap", q):
        return "live"
    if re.search(r"\b(audit\w*|subsidiar\w*|ceo|cfo|chief executive|chief financial|related[- ]party|who is|who are)\b", q):
        return "graph"
    if re.search(r"\b(summari[sz]e|overview|summary|key themes|broadly)\b", q):
        return "summary"
    if re.search(r"guid|outlook|commentary|management (say|said)", q):
        return "vector"
    if metrics and re.search(r"what was|how much|how many|compare|difference|trend|change|grow\w*|increas\w*|decreas\w*|"
                              r"higher|lower|in fy|\d", q):
        return "sql"
    return "vector"


def decompose(question: str, filters: dict) -> list[str]:
    """'Compare A and B margins' -> one sub-question per company."""
    tickers = filters.get("ticker") or []
    if len(tickers) < 2:
        return []
    subs = []
    for t in tickers:
        q = question
        for other in tickers:
            if other != t:
                for alias in COMPANIES[other]["aliases"] + [company_name(other)]:
                    q = re.sub(rf"\b{re.escape(alias)}\b", "", q, flags=re.I)
        subs.append(re.sub(r"\s+", " ", f"{company_name(t)}: {q}").strip())
    return subs


def multi_query(question: str, metrics: list[str]) -> list[str]:
    if llm.enabled():
        out = llm.complete_json(f"Write 3 different search queries for: {question}. JSON list of strings.")
        return [x for x in (out or []) if isinstance(x, str)][:3]
    out = []
    for m in metrics:
        for syn in METRIC_SYNONYMS[m][1:3]:
            out.append(re.sub(rf"\b{re.escape(m)}\b", syn, question, flags=re.I))
    return [x for x in out if x != question][:3]


def hyde(question: str, metrics: list[str], filters: dict) -> str:
    """Hypothetical answer, embedded instead of / alongside the question."""
    if llm.enabled():
        return llm.complete(f"Write a short passage from an Indian company's annual report that would answer: {question}")
    company = company_name((filters.get("ticker") or ["The company"])[0])
    year = (filters.get("fiscal_year") or ["the year"])[-1]
    metric = metrics[0] if metrics else "performance"
    return f"In {year}, {company} reported {metric} of X. Management said {metric} changed because of demand and costs."


def step_back(question: str, filters: dict) -> str:
    company = company_name((filters.get("ticker") or [""])[0])
    return f"{company} business overview, performance and risks".strip()


# ---------------------------------------------------------------------------
def understand(question: str, history: list[dict], s: Settings, known_years: list[str]) -> QueryPlan:
    query = contextualize(question, history) if s.use_contextualize and history else question
    metrics = find_metrics(query)
    plan = QueryPlan(original=question, query=query, metrics=metrics, known_years=known_years)
    plan.filters = extract_filters(query, known_years)
    plan.route = route(query, metrics) if s.use_routing else "vector"
    plan.forward_looking = bool(FORWARD.search(query))
    plan.wants_reasons = bool(re.search(r"\b(why|reason|reasons|explain|driver|because)\b", query.lower()))
    if s.use_decomposition:
        plan.sub_queries = decompose(query, plan.filters)
    if s.use_multi_query:
        plan.variants += multi_query(query, metrics)
    if s.use_hyde:
        plan.variants.append(hyde(query, metrics, plan.filters))
    if s.use_step_back:
        plan.variants.append(step_back(query, plan.filters))
    return plan
