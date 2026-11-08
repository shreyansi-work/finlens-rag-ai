"""Chunking strategies.

Four strategies for the small "child" chunks that get embedded:
  fixed      - fixed word windows with overlap
  recursive  - split on paragraphs, then lines, then sentences, then words
  structure  - one chunk per heading section (split further only if too long)
  semantic   - start a new chunk where sentence-to-sentence similarity drops

Parents are always structure-aware sections (~1,500 tokens) so the LLM gets
the full surrounding context (parent-child / small-to-big retrieval).
Word counts are used as a cheap stand-in for tokens.
"""
from dataclasses import dataclass

from finlens.text import sentences, word_count


@dataclass
class Section:
    heading: str
    page: int
    text: str


def sections_from_pages(pages, split_pages: bool = False) -> tuple[list[Section], list[Section]]:
    """Group parsed blocks into heading sections. Tables are returned separately.

    split_pages=True (user uploads) ends a section at every page break, so a section's page number is where
    its text really is and citations point at the right page; a heading that has no body of its own is kept
    as a tiny section so its words stay searchable."""
    sections, tables = [], []
    heading, page, buf = "Introduction", 1, []
    emitted = True                                  # the default heading has nothing to lose

    def flush():
        nonlocal emitted
        if buf:
            sections.append(Section(heading, page, "\n\n".join(buf)))
            buf.clear()
            emitted = True

    for p in pages:
        if split_pages:
            flush()
        for b in p.blocks:
            if b.kind == "heading":
                flush()
                if split_pages and not emitted:
                    sections.append(Section(heading, page, heading))
                heading, page, emitted = b.text, p.number, False
            elif b.kind == "toc":
                continue                      # table-of-contents entries are stored separately, not searched
            elif b.kind == "table":
                tables.append(Section(heading, p.number, b.text))
                buf.append(b.text)
            else:
                if not buf:
                    page = p.number
                buf.append(b.text)
    flush()
    if split_pages and not emitted:
        sections.append(Section(heading, page, heading))
    return sections, tables


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def recursive_chunks(text: str, size: int, seps=("\n\n", "\n", ". ", " ")) -> list[str]:
    if word_count(text) <= size or not seps:
        return [text.strip()] if text.strip() else []
    sep, rest = seps[0], seps[1:]
    out, cur = [], ""
    for piece in text.split(sep):
        candidate = f"{cur}{sep}{piece}" if cur else piece
        if word_count(candidate) <= size:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            if word_count(piece) > size:
                out.extend(recursive_chunks(piece, size, rest))
                cur = ""
            else:
                cur = piece
    if cur.strip():
        out.append(cur)
    return [c.strip() for c in out if c.strip()]


def semantic_chunks(text: str, size: int, embed, threshold: float = 0.15) -> list[str]:
    sents = sentences(text.replace("\n", " "))
    if len(sents) <= 1:
        return [text.strip()] if text.strip() else []
    vecs = embed(sents)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    out, cur = [], [sents[0]]
    for sent, sim in zip(sents[1:], sims):
        if sim < threshold or word_count(" ".join(cur + [sent])) > size:
            out.append(" ".join(cur))
            cur = [sent]
        else:
            cur.append(sent)
    out.append(" ".join(cur))
    return out


def child_chunks(text: str, strategy: str, size: int, overlap: int, embed=None) -> list[str]:
    if strategy == "fixed":
        return fixed_chunks(text, size, overlap)
    if strategy == "recursive":
        return recursive_chunks(text, size)
    if strategy == "structure":
        return recursive_chunks(text, size, seps=("\n\n", " "))  # whole section unless too long
    if strategy == "semantic":
        return semantic_chunks(text, size, embed)
    raise ValueError(f"unknown chunk strategy: {strategy}")


def parent_chunks(sections: list[Section], max_tokens: int) -> tuple[list[Section], list[int]]:
    """Merge small neighbouring sections on the same page into parents of up to
    max_tokens. Returns the parents and, for each section, its parent index."""
    parents, owner = [], []
    for s in sections:
        last = parents[-1] if parents else None
        if last and last.page == s.page and word_count(last.text) + word_count(s.text) <= max_tokens:
            last.text += f"\n\n## {s.heading}\n{s.text}"
        else:
            parents.append(Section(s.heading, s.page, f"## {s.heading}\n{s.text}"))
        owner.append(len(parents) - 1)
    return parents, owner
