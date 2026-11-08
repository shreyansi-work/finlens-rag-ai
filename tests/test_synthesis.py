"""Verified answer synthesis and the free hosted-LLM (Groq) backend. No network: urlopen is faked."""
import io
import json
import urllib.error

import pytest

from finlens import llm, synthesis
from finlens.config import settings
from finlens.service import FinLens

QUESTION = "Compare Nilgiri and Sahyadri operating margins for FY25 and compute the difference."
GOOD = ("Nilgiri's operating margin was 24.3% [NILCS_FY25_annual_report, p.3] against Sahyadri's 17.1% "
        "[SAHSW_FY25_annual_report, p.3], so Nilgiri was ahead by 7.2%.")


@pytest.fixture(scope="module")
def draft():
    app = FinLens(settings.with_(use_cache=False, use_synthesis=False))
    return app.ask(QUESTION, use_memory=False)


# ---------------------------------------------------------------- verify()
def test_verify_accepts_a_faithful_rewrite(draft):
    assert synthesis.verify(draft["answer"], GOOD, []) is None


@pytest.mark.parametrize("bad, reason", [
    (GOOD.replace("24.3%", "29.3%"), "unsupported number"),                       # invented figure
    (GOOD + " Margins will reach 99% next year.", "unsupported number"),          # injected claim
    (GOOD.replace("SAHSW_FY25_annual_report, p.3", "SAHSW_FY25_annual_report, p.9"), "invented citation"),
    ("Nilgiri was ahead by 7.2% [NILCS_FY25_annual_report, p.3].", "dropped number"),   # lost 17.1 and 24.3? kept 7.2 only
    ("Nilgiri earned 24.3%, Sahyadri 17.1%, difference 7.2%.", "dropped all citations"),
    ("I don't know.", "changed the abstention decision"),
    ("", "empty"),
])
def test_verify_rejects_unsafe_rewrites(draft, bad, reason):
    assert reason in (synthesis.verify(draft["answer"], bad, []) or "")


# ---------------------------------------------------------------- refine() through the real pipeline
def test_accepted_rewrite_replaces_the_draft(monkeypatch):
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: GOOD)
    r = FinLens(settings.with_(use_cache=False, use_synthesis=True)).ask(QUESTION, use_memory=False)
    assert r["synthesized"] is True and r["answer"] == GOOD
    assert any(s["name"] == "synthesis_accepted" for s in r["spans"])


def test_rejected_rewrite_keeps_the_deterministic_answer(monkeypatch, draft):
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Nilgiri's margin was 99% [NILCS_FY25_annual_report, p.3].")
    r = FinLens(settings.with_(use_cache=False, use_synthesis=True)).ask(QUESTION, use_memory=False)
    assert r["synthesized"] is False and r["answer"] == draft["answer"]
    assert any(s["name"] == "synthesis_rejected" for s in r["spans"])


def test_llm_failure_keeps_the_deterministic_answer(monkeypatch, draft):
    def boom(*a, **k):
        raise llm.LLMUnavailable("hosted LLM returned HTTP 429")
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", boom)
    r = FinLens(settings.with_(use_cache=False, use_synthesis=True)).ask(QUESTION, use_memory=False)
    assert r["answer"] == draft["answer"] and r["synthesized"] is False


def test_abstentions_are_never_sent_to_the_llm(monkeypatch):
    called = []
    monkeypatch.setattr(llm, "synthesis_enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: called.append(1) or "x")
    r = FinLens(settings.with_(use_cache=False, use_synthesis=True)).ask("What was Aravali's revenue in FY21?",
                                                                         use_memory=False)
    assert r["answer"].startswith("I don't know") and not called


def test_offline_by_default(draft):
    assert draft["synthesized"] is False


# ---------------------------------------------------------------- Groq backend
class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm, "settings", settings.with_(llm_provider="groq", use_synthesis=True, llm_enabled=True))
    monkeypatch.setattr(llm, "_breaker_until", 0.0)
    return llm


def test_groq_defaults_and_key_gating(monkeypatch):
    s = settings.with_(llm_provider="groq", llm_model="llama3.2:3b")
    assert s.llm_model == "openai/gpt-oss-20b" and s.llm_agent is False
    monkeypatch.setattr(llm, "settings", s)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.provider() == "offline" and not llm.synthesis_enabled()


def test_groq_call_parses_the_response_and_sends_the_key_only_in_the_header(groq, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["auth"], seen["body"] = req.full_url, req.headers.get("Authorization"), json.loads(req.data)
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "hello"}}],
                                        "usage": {"prompt_tokens": 11, "completion_tokens": 2}}).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    assert llm.complete("hi", system="be brief") == "hello"
    assert seen["url"].endswith("/openai/v1/chat/completions") and seen["auth"] == "Bearer test-key-not-real"
    assert "test-key-not-real" not in json.dumps(seen["body"])
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be brief"}


def test_rate_limit_opens_the_breaker_and_never_leaks_the_key(groq, monkeypatch):
    def limited(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"retry-after": "7"}, io.BytesIO(b""))
    monkeypatch.setattr(llm.urllib.request, "urlopen", limited)
    with pytest.raises(llm.LLMUnavailable) as e:
        llm.complete("hi")
    assert "429" in str(e.value) and "test-key-not-real" not in str(e.value)
    assert llm.breaker_open() and not llm.synthesis_enabled()
    monkeypatch.setattr(llm, "_breaker_until", 0.0)           # the cool-down passed
    assert llm.synthesis_enabled()


@pytest.mark.parametrize("code, min_wait", [(401, 500), (500, 20)])
def test_other_errors_also_trip_the_breaker(groq, monkeypatch, code, min_wait):
    def failing(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, io.BytesIO(b""))
    monkeypatch.setattr(llm.urllib.request, "urlopen", failing)
    with pytest.raises(llm.LLMUnavailable):
        llm.complete("hi")
    assert llm._breaker_until - llm.time.monotonic() > min_wait
