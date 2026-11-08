"""HTTP layer: endpoints, SSE streaming, file serving guards and a UTF-8 regression test."""
import json

import pytest
from fastapi.testclient import TestClient

from finlens import api


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def test_index_is_served_as_utf8(client):
    # Regression: read_text() without an encoding garbled "·" into "Â·" on Windows.
    r = client.get("/")
    assert r.status_code == 200
    assert "charset=utf-8" in r.headers["content-type"].lower()
    assert "Â·" not in r.text and "Ollama ·" in r.text


def test_info_and_documents(client):
    info = client.get("/api/info").json()
    assert info["documents"] >= 18 and info["chunks"] > 0 and info["mode"] in ("offline", "ollama")
    docs = client.get("/api/documents").json()
    assert {d["ticker"] for d in docs} >= {"ARVL", "NILCS", "SAHSW", "GANGABK", "NARMIND"}


def test_ask_returns_answer_with_citations_and_trace(client):
    r = client.post("/api/ask", json={"question": "What was Aravali's operating margin in FY25?", "session_id": "api-t"})
    assert r.status_code == 200
    body = r.json()
    assert "21.1" in body["answer"] and body["citations"] and body["trace_id"]


def test_ask_rejects_missing_question(client):
    assert client.post("/api/ask", json={}).status_code == 422


def test_stream_emits_status_then_answer(client):
    with client.stream("GET", "/api/ask/stream", params={"q": "What was Nilgiri's revenue in FY25?"}) as r:
        text = "".join(r.iter_text())
    assert "event: answer" in text
    payload = text.split("event: answer\ndata: ", 1)[1].split("\n\n", 1)[0]
    assert "255,324" in json.loads(payload)["answer"]


def test_page_endpoint_returns_sections(client):
    data = client.get("/api/page", params={"doc": "ARVL_FY25_annual_report", "page": 3}).json()
    assert data["sections"] and "revenue" in " ".join(s["text"] for s in data["sections"]).lower()


@pytest.mark.parametrize("name", ["finlens.db", "../finlens.db", "..%2Ffinlens.db", "ARVL_FY25_annual_report.txt"])
def test_files_endpoint_only_serves_pdfs_inside_raw_dir(client, name):
    assert client.get(f"/files/{name}").status_code == 404


def test_files_endpoint_serves_a_real_pdf(client):
    r = client.get("/files/ARVL_FY25_annual_report.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"


def test_negative_feedback_is_queued_for_the_golden_set(client, monkeypatch, tmp_path):
    (tmp_path / "eval").mkdir()
    monkeypatch.setattr(api, "ROOT", tmp_path)
    seen = []
    monkeypatch.setattr(api.service(), "feedback", lambda *a: seen.append(a))
    body = {"question": "q", "answer": "a", "rating": -1, "reason": "wrong number"}
    assert client.post("/api/feedback", json=body).json() == {"ok": True}
    assert seen == [("q", "a", -1, "wrong number")]
    line = (tmp_path / "eval" / "feedback_candidates.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["rating"] == -1
