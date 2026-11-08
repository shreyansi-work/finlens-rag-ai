"""Layout-aware PDF parsing: pages -> headings, paragraphs and tables.

Uses pypdf's layout mode, which keeps column spacing, so table rows can be
recognised as lines whose cells are separated by 2+ spaces. Repeated page
headers/footers are removed. Scanned pages with no text layer fall back to OCR
when pytesseract + pdf2image are installed (optional).
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<fy>FY\d{2})(?:_(?P<q>Q[1-4]))?_(?P<type>annual_report|transcript|results)$")
NUMERIC_CELL = re.compile(r"^-?[\d,]+(\.\d+)?%?$")
PERIOD_CELL = re.compile(r"^(Q[1-4]\s*)?FY\d{2}$")


@dataclass
class Block:
    kind: str                 # heading | text | table
    text: str
    rows: list = field(default_factory=list)   # table rows (list of cells)


@dataclass
class Page:
    number: int
    blocks: list


def metadata_from_filename(path: Path) -> dict:
    m = FILENAME_RE.match(path.stem)
    if not m:
        return {"ticker": path.stem.split("_")[0].upper(), "fiscal_year": None, "quarter": None,
                "doc_type": "annual_report"}
    return {"ticker": m["ticker"], "fiscal_year": m["fy"], "quarter": m["q"], "doc_type": m["type"]}


def _ocr_page(path: Path, page_no: int) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return ""
    images = convert_from_path(str(path), first_page=page_no, last_page=page_no)
    return pytesseract.image_to_string(images[0]) if images else ""


def _is_table_row(line: str) -> bool:
    cells = re.split(r"\s{2,}", line.strip())
    if len(cells) < 2:
        return False
    rest = cells[1:]
    return all(NUMERIC_CELL.match(c) or PERIOD_CELL.match(c) for c in rest)


def _is_heading(line: str) -> bool:
    s = line.strip()
    return (0 < len(s) <= 80 and len(s.split()) <= 10 and s[0].isupper()
            and not s.endswith((".", ",", ";")) and ":" not in s and not _is_table_row(s))


def parse_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    raw_pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or ""
        if len(text.strip()) < 20:
            text = _ocr_page(path, i) or text
        raw_pages.append(text.splitlines())

    # Lines repeated on most pages are running headers/footers.
    def norm(l):
        return re.sub(r"\d+", "#", l.strip())
    counts = Counter(norm(l) for lines in raw_pages for l in set(lines) if l.strip())
    boilerplate = {k for k, v in counts.items() if len(raw_pages) > 1 and v >= max(2, len(raw_pages) * 0.6)}

    pages = []
    for number, lines in enumerate(raw_pages, start=1):
        blocks, para, table = [], [], []

        def flush_para():
            if para:
                blocks.append(Block("text", "\n".join(para)))
                para.clear()

        def flush_table():
            if table:
                rows = [re.split(r"\s{2,}", l.strip()) for l in table]
                text = "\n".join(" | ".join(r) for r in rows)
                blocks.append(Block("table", text, rows))
                table.clear()

        for line in lines:
            s = line.strip()
            if not s or norm(line) in boilerplate:
                flush_para()          # blank line ends a paragraph, but not a table
                continue
            if _is_table_row(line) or (not table and _looks_like_table_header(s)):
                flush_para()
                table.append(s)
                continue
            flush_table()
            if _is_heading(s) and not para:
                blocks.append(Block("heading", s))
            else:
                para.append(re.sub(r"\s+", " ", s))
        flush_para()
        flush_table()
        pages.append(Page(number, blocks))
    return pages


def _looks_like_table_header(s: str) -> bool:
    cells = re.split(r"\s{2,}", s)
    return len(cells) >= 2 and all(PERIOD_CELL.match(c) for c in cells[1:])
