"""Document structure for uploaded files: outline, counts and lists.

Questions like "summarize this", "what are the key points", "how many milestones" or "which phases are there"
are about the document's *shape*, not about one passage, so sentence retrieval cannot answer them. These helpers
read the sections the parser found (headings with their real page numbers) and answer from that structure.
Every item carries its page, so every statement is cited.
"""
import re

from finlens.text import sentences

SKIP_HEADINGS = {"introduction", "table of contents", "contents", "thank you", "agenda"}
TOP_LEVEL = re.compile(r"^\d+[.)]\s+\S")
# Words that are about the question itself, not a thing that could be counted or listed.
NOT_A_NOUN = {"key", "main", "following", "different", "total", "things", "points", "ones", "other", "major",
              "important", "various", "some", "many", "all", "these", "those", "there", "here", "document",
              "documents", "pages", "page", "words", "are", "is", "was", "were", "do", "does", "did", "we", "you",
              "they", "it", "i", "of", "in", "to", "the", "a", "an"}
BROAD = re.compile(
    r"\b(summari[sz]e|summary|overview|key (points?|takeaways?|ideas?)|main (points?|ideas?|topics?)|highlights?|"
    r"tl;?dr|outline|table of contents|structure|"
    r"what('s| is| does)? (this|the) (document|file|pdf|proposal|report|paper|contract)( about| cover| contain| say)?|"
    r"what does (this|the) (document|file|pdf|proposal|report) (cover|contain|say|describe)|"
    r"about (this|the) (document|file|pdf))\b", re.I)
COUNT_Q = re.compile(r"\bhow many ([a-z][a-z-]*)(?: [a-z-]+)?", re.I)
# "which are the modules", "what are the phases": plural only. "What IS the warranty period" is a lookup, not a list.
LIST_Q = re.compile(r"\b(?:which|what|whic|wich|wat)\s+(?:are|were)\s+(?:all\s+)?(?:the\s+)?([a-z][a-z-]*s)\b", re.I)
LIST_VERB = re.compile(r"\b(?:list|name|enumerate|show)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?([a-z][a-z-]*s)\b", re.I)
# Nouns that describe a document's structure. Only for these may "which modules are there" fall back to showing
# the outline when nothing is labelled that way; "what are the risks" must search the text instead.
STRUCTURAL_NOUNS = {"section", "part", "module", "component", "chapter", "topic", "area", "feature", "phase",
                    "stage", "step", "flow", "role", "deliverable", "workstream", "heading", "appendix", "annexure",
                    "screen", "service", "app", "application", "tab", "page"}


def singular(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def enumeration_noun(question: str) -> tuple[str, str]:
    """('count' | 'list', noun) for questions like 'how many milestones' / 'which phases are there', else ('', '')."""
    for kind, rx in (("count", COUNT_Q), ("list", LIST_Q), ("list", LIST_VERB)):
        m = rx.search(question)
        if m and m.group(1).lower() not in NOT_A_NOUN:
            return kind, singular(m.group(1))
    return "", ""


def is_broad(question: str) -> bool:
    return bool(BROAD.search(question))


def sections(conn) -> list[dict]:
    """Every section in reading order: {doc, title, heading, page, body}. Parent chunks hold '## heading\\nbody'
    blocks and never span a page for uploads, so the page is exact."""
    rows = conn.execute(
        "SELECT c.doc_id, d.company AS title, c.page, c.text FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE c.level='parent' AND c.active=1 ORDER BY c.chunk_id").fetchall()
    out = []
    for r in rows:
        for block in re.split(r"(?:^|\n\n)## ", r["text"]):
            if not block.strip():
                continue
            heading, _, body = block.partition("\n")
            out.append({"doc": r["doc_id"], "title": r["title"], "heading": " ".join(heading.split()),
                        "page": r["page"], "body": body.strip()})
    return out


def _cite(doc: str, page: int) -> str:
    return f"[{doc}, p.{page}]"


def _first_sentences(text: str, limit: int = 2, max_chars: int = 420) -> str:
    out = ""
    for s in sentences(re.sub(r"[•●▪]", ". ", text)):
        if len(s.split()) < 4:
            continue
        if out and len(out) + len(s) > max_chars:
            break
        out = f"{out} {s}".strip()
        if len(out.split(". ")) >= limit:
            break
    return out[:max_chars].rstrip()


def _toc_outline(conn, by_doc: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Top-level sections taken from the document's own table of contents, with their real pages.
    Numbered entries ('1. Overview') are the top level when the TOC has them; the TOC's printed page is only a
    fallback, because the real page is looked up from the section that carries the same heading."""
    result = {}
    for r in conn.execute("SELECT doc_id, text FROM chunks WHERE level='toc' AND active=1"):
        entries = []
        for line in r["text"].splitlines():
            title, _, page = line.partition("\t")
            if title.strip() and page.isdigit():
                entries.append((title.strip(), int(page)))
        numbered = [e for e in entries if TOP_LEVEL.match(e[0])]
        chosen = numbered if len(numbered) >= 3 else entries
        heads: dict[str, dict] = {}
        for e in by_doc.get(r["doc_id"], []):
            heads.setdefault(" ".join(e["heading"].lower().split()), e)       # first occurrence wins
        out = []
        for title, printed in chosen:
            hit = heads.get(" ".join(title.lower().split()))
            out.append({"doc": r["doc_id"], "heading": title, "page": hit["page"] if hit else printed})
        if len(out) >= 3:
            result[r["doc_id"]] = out
    return result


def outline_entries(conn) -> dict[str, list[dict]]:
    """Per document, the headings worth showing: the table of contents' top level when the document has one,
    else its numbered top-level headings, else all headings."""
    by_doc: dict[str, list[dict]] = {}
    for s in sections(conn):
        if s["heading"].lower() in SKIP_HEADINGS or len(s["heading"]) < 3:
            continue
        by_doc.setdefault(s["doc"], []).append(s)
    result = _toc_outline(conn, by_doc)
    for doc, entries in by_doc.items():
        if doc in result:
            continue
        seen, uniq = set(), []
        for e in entries:
            key = e["heading"].lower()
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        top = [e for e in uniq if TOP_LEVEL.match(e["heading"])]
        # Top-level numbered sections ("1. Overview" ... "12. General Notes") are the best outline;
        # sub-items like "1. Onboarding" restart at 1, so only keep a numbering that runs upwards.
        ordered, last = [], 0
        for e in top:
            n = int(re.match(r"\d+", e["heading"]).group())
            if n > last:
                ordered.append(e)
                last = n
        result[doc] = ordered if len(ordered) >= 3 else uniq
    return result


def summary_parts(conn, max_items: int = 14, lead_in: bool = True) -> list[tuple[str, str, int]]:
    """[(sentence-with-citation, doc, page)] describing what each document is and how it is organised.
    lead_in=False returns only the 'organised into' listing."""
    secs = sections(conn)
    entries = outline_entries(conn)
    parts = []
    for doc, items in entries.items():
        title = next((s["title"] for s in secs if s["doc"] == doc), doc)
        lead = next((s for s in secs if s["doc"] == doc and s["heading"].lower() not in SKIP_HEADINGS
                     and len(s["body"].split()) >= 25), None)
        if lead and lead_in:
            parts.append((f"{title}: {_first_sentences(lead['body'])} {_cite(doc, lead['page'])}", doc, lead["page"]))
        shown = items[:max_items]
        listing = "; ".join(f"{e['heading']} {_cite(doc, e['page'])}" for e in shown)
        more = f" and {len(items) - len(shown)} more" if len(items) > len(shown) else ""
        parts.append((f"It is organised into: {listing}{more}.", doc, shown[0]["page"] if shown else 1))
    return parts


def outline_lines(conn, max_chars: int = 2600) -> str:
    """Compact 'heading (DOC, p.N)' map of the documents, given to an LLM so it can see the whole structure."""
    lines, used = [], 0
    for doc, items in outline_entries(conn).items():
        for e in items:
            line = f"- {e['heading']} {_cite(doc, e['page'])}"
            if used + len(line) > max_chars:
                return "\n".join(lines)
            lines.append(line)
            used += len(line)
    return "\n".join(lines)


def enumerate_labels(conn, noun: str) -> list[dict]:
    """Distinct '<Noun> <label>' items such as 'Milestone 3: Card and Pocket Money', in order, with their page.
    Needs at least a numeric/letter label, so 'milestones' in a sentence does not count as an item."""
    label = r"(\d{1,3}|[A-Z]\b|[IVX]{1,5}\b)"
    rx = re.compile(rf"(?i:\b{re.escape(noun)})s?\s+{label}(?:\s*[:\u2013-]\s*([A-Z][^\n.;|•]{{2,80}}))?")
    found: dict[str, dict] = {}
    for s in sections(conn):
        for from_heading, text in ((True, s["heading"]), (False, s["body"])):
            for m in rx.finditer(text):
                key = m.group(1).upper()
                title = (m.group(2) or "").strip().rstrip(" ,")
                cand = {"label": key, "title": title, "doc": s["doc"], "page": s["page"], "heading": from_heading}
                cur = found.get(key)
                if cur is None:
                    cand["order"] = len(found)                   # items stay in the order first seen
                    found[key] = cand
                elif title and (not cur["title"] or (from_heading and not cur["heading"])):
                    cand["order"] = cur["order"]                 # a titled heading beats a passing mention
                    found[key] = cand
    # Numeric labels must form a consecutive run ("Payment 1 ... Payment 6"); a stray "payments 100%" is not an item.
    nums = sorted(int(k) for k in found if k.isdigit())
    if nums:
        run = [nums[0]]
        for n in nums[1:]:
            if n != run[-1] + 1:
                break
            run.append(n)
        found = {k: v for k, v in found.items() if not k.isdigit() or int(k) in run}
    items = sorted(found.values(), key=lambda x: x["order"])
    return items if len(items) >= 2 else []


def enumeration_parts(noun: str, items: list[dict], count: bool) -> list[str]:
    word = noun if noun.endswith("s") else noun + "s"
    listing = "; ".join(f"{noun.capitalize()} {i['label']}{': ' + i['title'] if i['title'] else ''} "
                        f"{_cite(i['doc'], i['page'])}" for i in items)
    lead = f"The document describes {len(items)} {word}" if count else f"The document lists these {word}"
    return [f"{lead}: {listing}."]
