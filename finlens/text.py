"""Small text helpers shared across modules."""
import hashlib
import re

STOPWORDS = set("""
a an the and or of in on for to from by with at as is are was were be been being this that these those
it its what which who whom how why when where did do does done has have had their them they than then
there about over under into per vs versus between compare comparison also any all each can could would
should will shall may might our we you your i me my give show tell please much many
""".split())

# Fictional companies used by the bundled sample corpus, plus a few real
# NSE tickers so real filings dropped into data/raw/ are recognised too.
COMPANIES = {
    "ARVL": {"name": "Aravali Infotech", "aliases": ["aravali", "arvl"]},
    "NILCS": {"name": "Nilgiri Consultancy Services", "aliases": ["nilgiri", "nilcs"]},
    "SAHSW": {"name": "Sahyadri Softworks", "aliases": ["sahyadri", "sahsw"]},
    "GANGABK": {"name": "Ganga Commercial Bank", "aliases": ["ganga commercial bank", "ganga bank", "ganga", "gangabk"]},
    "NARMIND": {"name": "Narmada Industries", "aliases": ["narmada industries", "narmada", "narmind"]},
    "INFY": {"name": "Infosys", "aliases": ["infosys", "infy"]},
    "TCS": {"name": "Tata Consultancy Services", "aliases": ["tcs", "tata consultancy"]},
    "WIPRO": {"name": "Wipro", "aliases": ["wipro"]},
    "HDFCBANK": {"name": "HDFC Bank", "aliases": ["hdfc bank", "hdfcbank"]},
    "RELIANCE": {"name": "Reliance Industries", "aliases": ["reliance industries", "reliance", "ril"]},
}


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def company_name(ticker: str) -> str:
    return COMPANIES.get(ticker, {}).get("name", ticker)


def find_tickers(text: str) -> list[str]:
    """Tickers mentioned in the text, in order of first mention."""
    low = text.lower()
    found = []
    for ticker, info in COMPANIES.items():
        positions = [m.start() for a in info["aliases"] for m in re.finditer(rf"\b{re.escape(a)}\b", low)]
        if positions:
            found.append((min(positions), ticker))
    return [t for _, t in sorted(found)]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    text = "\n".join(l for l in text.splitlines() if not l.startswith("## "))   # drop section headings
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def numbers_in(text: str) -> list[float]:
    """Numbers in a string, understanding Indian digit grouping like 1,53,670."""
    out = []
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def tidy_page_text(text: str) -> str:
    """Re-join lines that the PDF wrapped, keeping headings and table rows on their own lines."""
    blocks = []
    for block in text.split("\n\n"):
        keep, prose = [], []
        for line in block.splitlines():
            if line.startswith("## ") or " | " in line:
                if prose:
                    keep.append(" ".join(prose))
                    prose = []
                keep.append(line)
            else:
                prose.append(line.strip())
        if prose:
            keep.append(" ".join(prose))
        blocks.append("\n".join(keep))
    return "\n\n".join(blocks)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def word_count(text: str) -> int:
    return len(text.split())
