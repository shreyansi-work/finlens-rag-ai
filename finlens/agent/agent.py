"""The agent: plans, calls tools, writes a cited answer, checks it.

The planner-executor picks a plan, runs tools step by step, and composes the
answer from the evidence with citations. A local Ollama model may help with
side steps. The paid Claude API is never called. The self-reflection check
drops unsupported numbers.
"""
import json
import re

from finlens import llm, outline, synthesis, tracing
from finlens.agent import tools as T
from finlens.config import Settings
from finlens.generation import (
    FINAL_FORMAT,
    SYSTEM_PROMPT,
    Answer,
    Citation,
    abstain,
    order_for_prompt,
    reflect,
    sanitize,
    wrap_documents,
)
from finlens.query.understanding import QueryPlan
from finlens.text import company_name, content_tokens, possessive, sentences

CAUSAL = re.compile(r"\b(because|due to|driven by|as a result|after|as industry|as hiring|as the)\b", re.I)
REL_NOUN = {"audited_by": "statutory auditor", "has_ceo": "CEO", "has_cfo": "CFO",
            "has_subsidiary": "subsidiary", "related_party": "related party",
            "related_party_flagged_by_auditor": "related party flagged by the auditor"}
SPEECH_WORDS = {"say", "said", "says", "saying", "comment", "comments", "commented", "talk", "talked", "mention",
                "mentioned", "call", "earnings", "transcript", "remarks", "tell", "told"}
UNIT_FMT = {"%": "{v}%", "INR crore": "INR {v} crore", "INR": "INR {v}"}
GRAPH_VOCAB = {"auditor", "statutory", "subsidiaries", "subsidiary", "ceo", "cfo", "chief", "executive",
               "financial", "officer", "who", "related", "party", "transactions", "transaction", "flag",
               "flagged", "companies", "company", "same", "share", "which", "raise", "matter", "emphasis",
               "audit", "audits", "audited", "audit"}


class Run:
    """Collects evidence, citations and tools used while the plan executes."""
    def __init__(self, conn, s: Settings):
        self.conn, self.s = conn, s
        self.tools_used, self.citations, self.evidence, self.computed = [], [], [], []
        self.missing = []
        self.passages = []          # (doc, page, text) of what was retrieved: what an LLM may cite

    def call(self, name, **args):
        out = T.run_tool(name, args, self.conn, self.s)
        self.tools_used.append(name)
        for r in out.get("results", []):
            r["text"], poisoned = sanitize(r["text"])
            if poisoned and tracing.current():
                tracing.current().event("injection_blocked", doc=r["doc"], page=r["page"])
        self.citations += out.get("citations", [])
        self.evidence += [r["text"] for r in out.get("results", [])]
        self.passages += [(r["doc"], r["page"], r["text"]) for r in out.get("results", [])]
        self.evidence += [json.dumps(r) for r in out.get("rows", []) + out.get("edges", [])]
        if name == "calculator" and out.get("ok"):
            self.computed.append(out["result"])
        return out


# ---------------------------------------------------------------------------
# Offline planner-executor
# ---------------------------------------------------------------------------
def fmt_value(v: float, unit: str) -> str:
    s = f"{v:,.0f}" if v >= 1000 else f"{v:g}"
    return UNIT_FMT.get(unit, "{v} " + unit if unit else "{v}").format(v=s).strip()


def cite_str(doc, page):
    return f"[{doc}, p.{page}]"


def describe_rows(rows: list[dict], plan: QueryPlan, run: Run) -> list[str]:
    """Turn SQL rows into cited sentences, with trends and comparisons via the calculator."""
    out = []
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["ticker"], r["metric"]), []).append(r)
    for (ticker, metric), rs in groups.items():
        rs.sort(key=lambda r: r["fiscal_year"])
        name = company_name(ticker)
        if len(rs) == 1:
            r = rs[0]
            period = f"{'' if r['quarter'] == 'FY' else r['quarter'] + ' '}{r['fiscal_year']}"
            out.append(f"{possessive(name)} {metric} was {fmt_value(r['value'], r['unit'])} in {period} {cite_str(r['doc_id'], r['page'])}.")
        else:
            series = ", ".join(f"{fmt_value(r['value'], r['unit'])} in {r['fiscal_year']} {cite_str(r['doc_id'], r['page'])}" for r in rs)
            first, last = rs[0], rs[-1]
            diff = run.call("calculator", expression=f"{last['value']} - {first['value']}")["result"]
            pct = run.call("calculator", expression=f"({last['value']} - {first['value']}) / {first['value']} * 100")["result"]
            direction = "rose" if diff > 0 else "fell" if diff < 0 else "was unchanged"
            shape = ""
            if len(rs) >= 3:
                mids = [r["value"] for r in rs]
                if mids[1] < mids[0] and mids[2] > mids[1]:
                    shape = " It fell first and then rose."
                elif mids[1] > mids[0] and mids[2] < mids[1]:
                    shape = " It rose first and then fell."
            points = " percentage points" if first["unit"] == "%" else ""
            out.append(f"{possessive(name)} {metric}: {series}. Overall it {direction} by {abs(diff):g}{points} "
                       f"({pct:+.2f}%) from {first['fiscal_year']} to {last['fiscal_year']}.{shape}")
    # Comparison across two companies for the same metric and year.
    q = plan.query.lower()
    if re.search(r"compare|difference|higher|lower|vs|versus", q):
        by_metric: dict = {}
        for r in rows:
            by_metric.setdefault((r["metric"], r["fiscal_year"], r["quarter"]), []).append(r)
        for (metric, fy, _), rs in by_metric.items():
            if len({r["ticker"] for r in rs}) == 2:
                a, b = sorted(rs, key=lambda r: -r["value"])
                d = run.call("calculator", expression=f"{a['value']} - {b['value']}")["result"]
                out.append(f"In {fy}, {company_name(a['ticker'])} had the higher {metric}; the difference is "
                           f"{fmt_value(round(d, 4), a['unit'])} (calculated: {a['value']} - {b['value']}).")
    return out


# Question words that say what kind of answer is wanted, not what it is about ("how LONG is the warranty").
WEAK_TERMS = {"long", "kind", "type", "way", "need", "needs", "provide", "give", "get", "use", "used", "using",
              "make", "take", "mention", "mentioned", "say", "says", "said", "include", "included", "list",
              "tell", "explain", "describe", "details", "detail", "information", "info", "document", "documents"}


def _clip(text: str, limit: int = 320) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _candidate_sentences(text: str, generic: bool) -> list[str]:
    """Uploads are full of bullet lists with no punctuation, which otherwise become one giant 'sentence'."""
    if not generic:
        return sentences(text)
    return [_clip(s) for piece in re.split(r"\s*[•●▪]\s*|\n+", text) for s in sentences(piece)]


def structure_parts(run: "Run", plan: QueryPlan) -> list[str]:
    """Uploads: answer 'summarize', 'key points', 'how many X' and 'which X' from the document's structure.
    Every item is cited with the real page it was found on."""
    conn = run.conn
    if plan.route == "enumerate":
        items = outline.enumerate_labels(conn, plan.noun)
        if not items:
            if plan.ask_kind != "list" or plan.noun not in outline.STRUCTURAL_NOUNS:
                return []                                # nothing countable/structural: search the text
            # "which modules are there" when nothing is labelled that way: show how the document is organised
            parts = []
            for text, doc, page in outline.summary_parts(conn, lead_in=False):
                note = f"Nothing in the document is labelled “{plan.noun}”, so here is how it is organised. {text}"
                run.evidence.append(text)
                run.passages.append((doc, page, text))
                parts.append(note)
            return parts
        run.computed.append(len(items))                  # the count itself is a computed, checkable figure
        for i in items:
            text = f"{plan.noun.capitalize()} {i['label']}: {i['title']}".strip(": ")
            run.evidence.append(text)
            run.passages.append((i["doc"], i["page"], text))
        return outline.enumeration_parts(plan.noun, items, plan.ask_kind == "count")
    parts = []
    for text, doc, page in outline.summary_parts(conn):
        run.evidence.append(text)
        run.passages.append((doc, page, text))
        parts.append(text)
    return parts


def best_sentences(results: list[dict], query: str, limit: int = 3, must: set | None = None,
                   generic: bool = False) -> list[str]:
    """Pick the most relevant evidence sentences, at most one per document first.
    If `must` is given, a sentence has to contain at least one of those terms."""
    terms = T.key_terms(query)
    if generic:
        terms = terms - WEAK_TERMS
    scored, seen = [], set()
    for r in results:
        head = set(content_tokens(r.get("section") or "")) if generic else set()
        for sent in _candidate_sentences(r["text"], generic):
            toks = set(content_tokens(sent))
            if " | " in sent or sent in seen or (must and not must & toks):
                continue
            seen.add(sent)
            overlap = len(terms & toks) + (0.5 if CAUSAL.search(sent) else 0)
            if generic and len(terms) >= 2:
                # Uploads: one shared word ("CEO") is not evidence, so questions the document cannot answer
                # abstain. The section heading counts as context: sentences under "9. Out of Scope" need not
                # repeat those words themselves.
                if len(terms & (toks | head)) < 2:
                    continue
                scored.append((overlap + 0.5 * len(terms & head), r["score"], sent, r["doc"], r["page"]))
                continue
            if terms & toks or not terms:          # broad question: keep sentences in rank order
                scored.append((overlap, r["score"], sent, r["doc"], r["page"]))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    chosen, docs = [], set()
    for pass_ in (0, 1):
        for ov, _, sent, doc, page in scored:
            item = f"{sent} {cite_str(doc, page)}"
            if item in chosen or len(chosen) >= limit:
                continue
            if pass_ == 0 and doc in docs:
                continue
            chosen.append(item)
            docs.add(doc)
    return chosen


def best_table_rows(results: list[dict], query: str, limit: int = 2) -> list[str]:
    """Uploaded documents: answer from table rows when no prose sentence matches the question.
    A row must share a term with the question; it is rendered as 'Column: value' pairs and cited."""
    terms = T.key_terms(query)
    scored = []
    for r in results:
        lines = [x for x in r["text"].splitlines() if " | " in x]
        if len(lines) < 2:
            continue
        header = [c.strip() for c in lines[0].split(" | ")]
        header_hit = len(terms & set(content_tokens(" ".join(header))))
        for line in lines[1:]:
            cells = [c.strip() for c in line.split(" | ")]
            row_hit = len(terms & set(content_tokens(line)))
            if row_hit:
                text = "; ".join(f"{h}: {c}" for h, c in zip(header, cells))
                scored.append((row_hit + 0.5 * header_hit, f"{text} {cite_str(r['doc'], r['page'])}"))
    scored.sort(key=lambda x: -x[0])
    out = []
    for _, item in scored:
        if item not in out:
            out.append(item)
    return out[:limit]


def person_statements(run: Run, person: str, query: str, f: dict, about: set, limit: int = 2) -> list[str]:
    """What a named executive said, taken from 'Name (Role): ...' turns in transcripts and results.
    A turn counts only if it mentions one of the `about` terms (a salary question is not answered by deal wins)."""
    # "What did the CEO say on the call?" names no topic, so any of their turns answers it.
    topic = {t for t in about if t not in SPEECH_WORDS and not re.fullmatch(r"fy\d\d|q[1-4]", t)}
    res = run.call("search_filings", query=f"{person} {query}", ticker=f.get("ticker"),
                   fiscal_years=f.get("fiscal_year"), doc_type=f.get("doc_type"))
    turn = re.compile(rf"({re.escape(person)} \([^)]*\):.*?)(?=\s+[A-Z][\w.]*(?: [A-Z][\w.]*)* \([^)]*\):|$)")
    out = []
    for r in res["results"]:
        text = " ".join(r["text"].split())
        for m in turn.finditer(text):
            if topic and not topic & set(content_tokens(m.group(1))):
                continue
            item = f"{m.group(1).strip()} {cite_str(r['doc'], r['page'])}"
            if item not in out:
                out.append(item)
    return out[:limit]


def locate_page(conn, cited_sentence: str) -> str:
    """Summaries are built from many pages; point the citation at the page the sentence came from."""
    m = re.match(r"(.*) \[([A-Za-z0-9_]+), p\.(\d+)\]$", cited_sentence)
    if not m:
        return cited_sentence
    sent, doc = m.group(1), m.group(2)
    probe = " ".join(sent.split()[:6])
    for r in conn.execute("SELECT page, text FROM chunks WHERE doc_id=? AND level='child' AND active=1", (doc,)):
        if probe in " ".join(r["text"].split()):
            return f"{sent} {cite_str(doc, r['page'])}"
    return cited_sentence


def plan_and_execute(plan: QueryPlan, run: Run) -> Answer:
    f, q = plan.filters, plan.query
    t = tracing.current()
    if t:
        t.event("plan", route=plan.route, filters=f, sub_queries=plan.sub_queries)
    parts: list[str] = []
    confidence = "medium"

    known = plan.known_years
    future = [y for y in f.get("fiscal_year") or [] if known and y > known[-1]]
    if future:
        # Forward-looking question (guidance, dividends to come): search the latest filings instead.
        f["fiscal_year"] = [y for y in f["fiscal_year"] if y not in future] or None
    if plan.forward_looking and not f.get("fiscal_year") and known:
        # Older reports' "next year" is already past: only the latest filings speak about what comes next.
        f["fiscal_year"] = known[-1:]
    unknown = [y for y in f.get("fiscal_year") or [] if y not in known]
    if unknown and plan.route != "live":
        return abstain(f"The corpus has no filings for {', '.join(unknown)} "
                       f"(available: {', '.join(plan.known_years)}).", [], plan.route)

    if plan.route == "direct":
        return Answer(answer="Hello! Ask me about the companies in the filings corpus.", confidence="high", route="direct")

    if plan.route in ("outline", "enumerate"):
        parts += structure_parts(run, plan)
        if parts:
            confidence = "high" if plan.route == "enumerate" else "medium"
        else:
            plan.route = "vector"                  # nothing countable or structural: search the text instead

    if plan.route in ("outline", "enumerate"):
        pass                                       # already answered from the document's structure
    elif plan.route == "live":
        for ticker in f.get("ticker") or []:
            quote = run.call("get_live_quote", ticker=ticker)
            eps = run.call("query_financials", question=f"{ticker} eps", filters={"ticker": [ticker]})
            latest = max(eps["rows"], key=lambda r: r["fiscal_year"]) if eps["rows"] else None
            if latest:
                parts.append(f"{possessive(company_name(ticker))} latest reported EPS is {fmt_value(latest['value'], latest['unit'])} "
                             f"for {latest['fiscal_year']} {cite_str(latest['doc_id'], latest['page'])}.")
            if quote["ok"]:
                pe = run.call("calculator", expression=f"{quote['price']} / {latest['value']}") if latest else None
                parts.append(f"The current share price is {quote['price']} {quote.get('currency', '')} "
                             f"(external live data: {quote['source']}).")
                if pe and pe["ok"]:
                    parts.append(f"P/E = {quote['price']} / {latest['value']} = {pe['result']}.")
            else:
                run.missing.append(f"current share price of {company_name(ticker)} ({quote['error']})")
        if not f.get("ticker"):
            run.missing.append("which company you mean")

    elif plan.route == "graph":
        relation = T.relation_from_text(q)
        entity = company_name(f["ticker"][0]) if f.get("ticker") else None
        res = run.call("graph_lookup", entity=entity, relation=relation)
        edges = res["edges"]
        if "same" in q.lower() and relation and edges:
            # Multi-hop: find the object (e.g. the auditor), then who else has it.
            target = edges[0]["object"]
            others = run.call("graph_lookup", entity=target, relation=relation)["edges"]
            names = sorted({e["subject"] for e in others if e["subject"] != edges[0]["subject"]})
            noun = REL_NOUN.get(relation, relation)
            parts.append(f"{possessive(edges[0]['subject'])} {noun} is {target} "
                         f"{cite_str(edges[0]['doc_id'], edges[0]['page'])}.")
            if names:
                cites = " ".join(sorted({cite_str(e['doc_id'], e['page']) for e in others if e['subject'] in names})[:3])
                parts.append(f"Other companies with the same {noun}: {', '.join(names)} {cites}.")
        else:
            latest_year = max((e["fiscal_year"] for e in edges), default=None)
            for e in edges:
                if e["fiscal_year"] == latest_year or relation == "related_party_flagged_by_auditor":
                    parts.append(f"{possessive(e['subject'])} {REL_NOUN.get(e['relation'], e['relation'])}: "
                                 f"{e['object']} ({e['fiscal_year']}) {cite_str(e['doc_id'], e['page'])}.")
            if relation == "related_party_flagged_by_auditor":
                # Pull the auditor's own wording for each flagged party.
                for e in edges:
                    res = run.call("search_filings", query=f"{e['object']} auditor emphasis of matter",
                                   ticker=e["ticker"], fiscal_years=[e["fiscal_year"]])
                    parts += [f"{x} {cite_str(r['doc'], r['page'])}" for r in res["results"][:3]
                              for x in sentences(r["text"]) if e["object"] in x][:1]
        leftover = T.key_terms(q) - GRAPH_VOCAB
        said = []
        if leftover and edges and relation in ("has_cfo", "has_ceo"):
            # "What does the CFO expect ...": answer with that person's own words, not report boilerplate.
            person = max(edges, key=lambda e: e["fiscal_year"])["object"]
            said = person_statements(run, person, q, f, leftover)
            parts += said
            if not said and plan.forward_looking:
                run.missing.append(f"statements by {person} on this in the filings")
                leftover = set()
        if (leftover and not said) or not edges or relation in (None, "related_party"):
            res = run.call("search_filings", query=q, ticker=f.get("ticker"), fiscal_years=f.get("fiscal_year"))
            found = best_sentences(res["results"], q, 2)
            if leftover and not any(set(content_tokens(x)) & leftover for x in found):
                run.missing.append(f"information about {', '.join(sorted(leftover))}")
                parts = []
            else:
                parts += found
        confidence = "high" if edges else "medium"

    elif plan.route == "summary":
        years = f.get("fiscal_year") or plan.known_years[-1:]
        res = run.call("search_filings", query=q, ticker=f.get("ticker"), fiscal_years=years, summaries=True)
        # Document-level summaries first (RAPTOR top level), then the rest.
        res["results"].sort(key=lambda r: r["section"] != "Document summary")
        for sent in best_sentences(res["results"], q, 4):
            parts.append(locate_page(run.conn, sent))

    else:  # sql or vector
        rows = []
        if plan.route == "sql":
            res = run.call("query_financials", question=q, filters=f)
            rows = res["rows"]
            parts += describe_rows(rows, plan, run)
            confidence = "high" if rows else "medium"
        if not rows or plan.wants_reasons or plan.route == "vector":
            queries = plan.sub_queries or [q]
            # Evidence must mention the metric asked about (stops 'interest-rate' answering 'attrition rate').
            must = {w for m in plan.metrics for w in m.split()}
            if plan.wants_reasons and plan.metrics:
                queries = [f"{' '.join(plan.metrics)} reasons why {sub}" for sub in queries]
            pool = []
            for sub in queries:
                tick = [x for x in (f.get("ticker") or []) if company_name(x).lower() in sub.lower()] or f.get("ticker")
                res = run.call("search_filings", query=sub, ticker=tick, fiscal_years=f.get("fiscal_year"),
                               doc_type=f.get("doc_type"), variants=plan.variants or None)
                pool += res["results"]
                limit = 3 if len(queries) == 1 else 2
                if plan.wants_reasons and f.get("fiscal_year") and len(f["fiscal_year"]) > 1:
                    limit = len(f["fiscal_year"])
                parts += best_sentences(res["results"], sub, limit, must or None, generic=run.s.generic_docs)
            if run.s.generic_docs and not parts:
                parts += best_table_rows(pool, q)            # user uploads: facts that live only in a table

    if not parts:
        reason = "; ".join(run.missing) or ("The uploaded documents do not contain this information."
                                            if run.s.generic_docs
                                            else "The filings in the corpus do not contain this information.")
        a = abstain(reason, run.tools_used, plan.route)
        if run.missing:
            a.answer = f"I don't know: {reason}."
        return a
    answer = " ".join(parts)
    if run.missing:
        answer += " Not available: " + "; ".join(run.missing) + "."
        confidence = "low" if confidence == "medium" else "medium"
    return Answer(answer=answer, citations=_citations_in(answer), confidence=confidence,
                  tools_used=_dedup(run.tools_used), unanswerable_parts=run.missing, route=plan.route)


def _citations_in(text: str) -> list[Citation]:
    seen = []
    for doc, page in re.findall(r"\[([A-Za-z0-9_]+), p\.(\d+)\]", text):
        c = Citation(doc=doc, page=int(page))
        if c not in seen:
            seen.append(c)
    return seen


def _dedup(xs):
    return list(dict.fromkeys(xs))


# ---------------------------------------------------------------------------
# LLM mode: ReAct loop with tool calling
# ---------------------------------------------------------------------------
def react(plan: QueryPlan, run: Run) -> Answer:
    s = run.s
    hint = f"Detected filters: {json.dumps(plan.filters)}. Suggested route: {plan.route}."
    messages = [{"role": "user", "content": f"{plan.query}\n\n({hint})\n\n{FINAL_FORMAT}"}]
    t = tracing.current()
    for step in range(s.max_steps):
        if t and t.cost_usd > s.max_cost_usd:
            return abstain("Cost budget exceeded before an answer was found.", run.tools_used, plan.route)
        resp = llm.create(system=SYSTEM_PROMPT, tools=T.TOOL_SCHEMAS, messages=messages, max_tokens=4000,
                          output_config={"effort": "medium"})
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return _parse_final(text, run, plan)
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = run.call(block.name, **block.input)
                if out.get("results"):
                    out = {**out, "results": order_for_prompt(out["results"])}
                    content = wrap_documents(out["results"])
                else:
                    content = json.dumps(out, default=str)[:8000]
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
        messages.append({"role": "user", "content": results})
    return abstain("Step budget exceeded.", run.tools_used, plan.route)


def _parse_final(text: str, run: Run, plan: QueryPlan) -> Answer:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {"answer": text}
        return Answer(**{k: v for k, v in data.items() if k in Answer.model_fields},
                      tools_used=_dedup(run.tools_used), route=plan.route)
    except Exception:
        return Answer(answer=text, citations=[], tools_used=_dedup(run.tools_used), route=plan.route)


# ---------------------------------------------------------------------------
def answer_question(plan: QueryPlan, conn, s: Settings) -> Answer:
    run = Run(conn, s)
    use_react = llm.enabled() and s.use_agent and s.llm_agent
    ans = react(plan, run) if use_react else plan_and_execute(plan, run)
    if s.use_reflection and not ans.answer.startswith("I don't know"):
        ans = reflect(ans, run.evidence, run.computed)
    if s.generic_docs:
        # Uploads: an LLM may answer from the document map + retrieved passages (counting, listing, semantic
        # questions); used only if every citation and figure verifies. Callers decide whether it may run.
        use_llm = s.use_synthesis and llm.synthesis_enabled()
        return synthesis.refine_grounded(plan.query, ans, run.passages,
                                         outline.outline_lines(conn) if use_llm else "", s)
    # Optional: an LLM rewords the verified draft; kept only if every number and citation still checks out.
    return synthesis.refine(plan.query, ans, run.evidence, s)
