"""Answer quality on uploaded documents: correct page citations, structure-aware answers (summary, key points,
counts, lists), footer removal, abstention, and the verifier for LLM-written answers.

The fixture PDF mimics a real proposal: cover, table of contents with dotted leaders, numbered headings, labelled
milestones, a running footer whose padding changes with the page number, and a table total ("payments 100%").
These are the failure modes found when a real work-scope PDF was uploaded."""
import sqlite3
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from finlens import api, llm, synthesis, tracing, workspaces
from finlens.config import settings
from finlens.generation import Answer, Citation
from finlens.outline import enumeration_noun, is_broad

LEAD = ("Project Atlas is a mobile platform that helps small schools collect fees, issue receipts and report "
        "attendance to parents through one shared application with simple controls for administrators.")


def build_pdf() -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(False)            # otherwise each footer (inside the margin) would add a blank page
    state = {"n": 0}

    def page(heading=None):
        pdf.add_page()
        state["n"] += 1
        pdf.set_font("Helvetica", size=11)
        if heading:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)

    def line(text, bold=False):
        pdf.set_font("Helvetica", "B" if bold else "", 12 if bold else 11)
        pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)

    def para(text):
        pdf.multi_cell(0, 6, text)
        pdf.ln(2)

    def footer():
        pdf.set_y(-15)
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 6, f"(c) 2026 Acme Ltd. All rights reserved.{' ' * (20 - state['n'])}| {state['n']}")

    page()
    line("Project Atlas: Work Scope", bold=True)
    para("Presented by Jane Doe, CEO of Acme Ltd. This proposal is valid for 30 days.")
    footer()
    page("Table of Contents")
    for title, n in [("1. Overview", 3), ("2. Milestones", 4), ("3. Payment", 5)]:
        line(f"{title} {'.' * 70} {n}")
    footer()
    page("1. Overview")
    para(LEAD)
    footer()
    page("2. Milestones")
    line("Milestone 1: Discovery", bold=True)
    para("The team interviews schools and writes the requirements document in the first two weeks.")
    line("Milestone 2: Build", bold=True)
    para("Developers implement the fee module and the attendance module during weeks three to eight.")
    line("Milestone 3: Launch", bold=True)
    para("The application goes live in a pilot of ten schools in week nine.")
    footer()
    page("3. Payment")
    para("The total fee is 1200 dollars paid in three instalments.")
    para("Payment 1: Kick-off. Payment 2: Midpoint. Payment 3: Final handover. Total payments 100%.")
    footer()
    return bytes(pdf.output())


@pytest.fixture(scope="module")
def pdf_bytes():
    return build_pdf()


@pytest.fixture
def env(tmp_path, monkeypatch, pdf_bytes):
    monkeypatch.setattr(workspaces, "_manager", workspaces.Manager(tmp_path / "ws"))
    monkeypatch.setattr(workspaces, "SYNC", True)
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")
    client = TestClient(api.app)
    doc = client.post("/api/workspace/upload", content=pdf_bytes,
                      headers={"X-Filename": quote("Atlas Scope.pdf")}).json()
    assert doc["status"] == "ready", doc
    return client, doc["doc_id"], next(iter(workspaces.get_manager().items.values()))


def ask(client, q):
    return client.post("/api/workspace/ask", json={"question": q}).json()


# ---------------------------------------------------------------- citations point at the real page
def test_citations_use_the_real_page(env):
    client, doc, _ = env
    answer = ask(client, "What is the total fee?")["answer"]
    assert "1200" in answer and f"[{doc}, p.5]" in answer
    assert "p.3" not in answer and "p.1]" not in answer


def test_sections_never_span_pages(env):
    _, _, ws = env
    conn = sqlite3.connect(ws.db_path)
    pages = {r[0] for r in conn.execute("SELECT DISTINCT page FROM chunks WHERE level='parent'")}
    assert {3, 4, 5} <= pages                      # each page is its own section: no "everything is p.12"


# ---------------------------------------------------------------- structure-aware answers
def test_summary_uses_the_documents_own_outline_with_exact_pages(env):
    client, doc, _ = env
    r = ask(client, "Summarize this document.")
    assert r["route"] == "outline" and not r["answer"].startswith("I don't know")
    assert "Project Atlas is a mobile platform" in r["answer"]
    for title, page in [("1. Overview", 3), ("2. Milestones", 4), ("3. Payment", 5)]:
        assert f"{title} [{doc}, p.{page}]" in r["answer"]


@pytest.mark.parametrize("q", ["What are the key points?", "main points please", "what is this document about"])
def test_broad_questions_are_not_refused(env, q):
    r = ask(env[0], q)
    assert r["route"] == "outline" and not r["answer"].startswith("I don't know")


def test_how_many_counts_labelled_items_with_pages(env):
    client, doc, _ = env
    r = ask(client, "how many milestones are there")
    assert r["route"] == "enumerate" and "3 milestones" in r["answer"]
    for label, page in [("Milestone 1: Discovery", 4), ("Milestone 2: Build", 4), ("Milestone 3: Launch", 4)]:
        assert f"{label} [{doc}, p.{page}]" in r["answer"]


def test_a_stray_percentage_is_not_counted_as_an_item(env):
    answer = ask(env[0], "how many payments are there")["answer"]
    assert "3 payments" in answer and "100" not in answer


def test_which_modules_falls_back_to_the_outline_when_nothing_is_labelled_that_way(env):
    r = ask(env[0], "whic are the modules need to create")
    assert "Nothing in the document is labelled" in r["answer"] and "2. Milestones" in r["answer"]


def test_what_are_the_risks_searches_the_text_instead_of_the_outline(env):
    # "risks" is not a structural noun, so there is no outline fallback; no risk is described, so it abstains.
    assert ask(env[0], "what are the risks mentioned")["answer"].startswith("I don't know")


# ---------------------------------------------------------------- cleanliness and abstention
def test_footers_and_table_of_contents_do_not_pollute_answers(env):
    client, _, ws = env
    conn = sqlite3.connect(ws.db_path)
    searchable = " ".join(r[0] for r in conn.execute("SELECT text FROM chunks WHERE level IN ('child','parent','table')"))
    assert "All rights reserved" not in searchable and "......" not in searchable
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE level='toc'").fetchone()[0] == 1   # kept for the outline


def test_one_shared_word_is_not_evidence(env):
    # The cover says "CEO of Acme". A question about Google's CEO must not be answered with it.
    assert ask(env[0], "Who is the CEO of Google?")["answer"].startswith("I don't know")


def test_a_section_heading_counts_as_context(env):
    answer = ask(env[0], "What is the payment total?")["answer"]
    assert "1200" in answer and not answer.startswith("I don't know")


def test_question_classifiers():
    assert is_broad("Summarize this document.") and is_broad("key points?") and not is_broad("What is the fee?")
    assert enumeration_noun("how many milestones are there") == ("count", "milestone")
    assert enumeration_noun("which are the phases") == ("list", "phase")
    assert enumeration_noun("What is the warranty period?") == ("", "")      # singular "is": a lookup, not a list
    assert enumeration_noun("how many are there") == ("", "")                 # "are" is not a noun


# ---------------------------------------------------------------- verifier for LLM-written answers
DOC = "Atlas_ab12cd"
EVIDENCE = "Milestone 1: Discovery. Milestone 2: Build. Milestone 3: Launch. The total fee is 1200 dollars."
ALLOWED = {(DOC, "4"), (DOC, "5")}


@pytest.mark.parametrize("text, reason", [
    (f"There are 3 milestones [{DOC}, p.4].", None),
    (f"There are three milestones [{DOC}, p.4].", None),                      # a word that matches the evidence's count
    (f"1. Discovery [{DOC}, p.4]\n2. Build [{DOC}, p.4]\n3. Launch [{DOC}, p.4]", None),   # list numbering is fine
    ("I don't know. The document does not contain this.", None),
    (f"There are 3 milestones [{DOC}, p.9].", "invented citation"),
    (f"The fee is 900 dollars [{DOC}, p.5].", "unsupported number"),
    (f"There are seven milestones [{DOC}, p.4].", "unsupported number word"),
    ("There are 3 milestones.", "no citations"),
    (f"Cite as [DOC_ID, p.N] like [{DOC}, p.4].", "echoed the instructions"),
    ("", "empty"),
])
def test_verify_grounded(text, reason):
    got = synthesis.verify_grounded(text, ALLOWED, EVIDENCE)
    assert (got is None) if reason is None else (reason in (got or ""))


def llm_settings():
    return settings.with_(generic_docs=True, use_synthesis=True)


def draft(text="Some extracted sentence.", confidence="medium"):
    return Answer(answer=text, citations=[Citation(doc=DOC, page=4)], confidence=confidence)


PASSAGES = [(DOC, 4, EVIDENCE), (DOC, 5, "The total fee is 1200 dollars.")]


def test_verified_llm_answer_replaces_a_weak_draft(monkeypatch):
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: f"The fee is 1200 dollars [{DOC}, p.5].")
    out = synthesis.refine_grounded("What is the fee?", draft(), PASSAGES, f"- Payment [{DOC}, p.5]", llm_settings())
    assert out.synthesized and out.answer.endswith("[Atlas_ab12cd, p.5].") and out.citations[0].page == 5


def test_an_llm_refusal_never_replaces_a_real_draft(monkeypatch):
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "I don't know. The document does not contain this.")
    out = synthesis.refine_grounded("What is the fee?", draft("The total fee is 1200 dollars."), PASSAGES, "", llm_settings())
    assert not out.synthesized and out.answer.startswith("The total fee")


def test_an_exact_high_confidence_answer_is_never_sent_to_the_llm(monkeypatch):
    called = []
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: called.append(1) or "x")
    out = synthesis.refine_grounded("how many?", draft("The document describes 3 milestones.", "high"),
                                    PASSAGES, "", llm_settings())
    assert not called and not out.synthesized


def test_llm_failure_keeps_the_draft(monkeypatch):
    def boom(*a, **k):
        raise llm.LLMUnavailable("HTTP 429")
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", boom)
    out = synthesis.refine_grounded("q", draft(), PASSAGES, "", llm_settings())
    assert not out.synthesized and out.answer == "Some extracted sentence."
