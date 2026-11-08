"""End-to-end behaviour on the sample corpus, plus the CI smoke eval."""
import pytest

from finlens.config import settings
from finlens.evaluation import evaluate, load_golden
from finlens.service import FinLens


@pytest.fixture(scope="module")
def app():
    return FinLens(settings.with_(use_cache=False))


def test_comparison_uses_sql_and_calculator(app):
    r = app.ask("Compare Nilgiri and Sahyadri operating margins for FY25 and compute the difference.", use_memory=False)
    assert "24.3" in r["answer"] and "17.1" in r["answer"] and "7.2" in r["answer"]
    assert {"query_financials", "calculator"} <= set(r["tools_used"])
    assert r["citations"]


def test_abstains_when_data_is_missing(app):
    r = app.ask("What was Aravali's revenue in FY21?", use_memory=False)
    assert r["answer"].startswith("I don't know")


def test_poisoned_document_is_ignored(app):
    r = app.ask("What was Sahyadri's operating margin in FY24?", use_memory=False)
    assert "16.1" in r["answer"] and "99" not in r["answer"]


def test_follow_up_uses_chat_memory(app):
    app.ask("What was Aravali's operating margin in FY25?", session_id="t")
    r = app.ask("What about Nilgiri?", session_id="t")
    assert "Nilgiri" in r["rewritten_query"] and "24.3" in r["answer"]


def test_needle_auditor_flag(app):
    r = app.ask("Which related-party transactions did the auditor flag?", use_memory=False)
    assert "Narmada Green Energy" in r["answer"] and "1,250" in r["answer"]


def test_cfo_expectation_uses_latest_guidance(app):
    r = app.ask("What does Aravali's CFO expect for next year?", use_memory=False)
    assert "Anand Kulkarni" in r["answer"] and "0% to 3%" in r["answer"] and "20% to 22%" in r["answer"]
    assert "ARVL_FY25_Q4_transcript" in r["answer"] and "FY23" not in r["answer"]


def test_cfo_expectation_without_statements_is_not_available(app):
    r = app.ask("What does Nilgiri's CFO expect for next year?", use_memory=False)
    assert "Meera Joshi" in r["answer"] and "Not available" in r["answer"]
    assert "well positioned" not in r["answer"]


@pytest.mark.parametrize("question, expected", [
    ("Why are employees leaving Nilgiri less in FY24?", ["12.5%", "talent market cooled"]),
    ("How many people work at Sahyadri in FY25?", ["233,346"]),
    ("How profitable was Nilgiri in FY25?", ["48,553"]),
    ("What is Ganga Bank's bad loan situation in Q4 FY25?", ["1.33%"]),
    ("Who audits Narmada Industries?", ["Kapoor Desai"]),
    ("Which subsidiaries does Aravali have?", ["Aravali BPM Limited", "Aravali Digital GmbH"]),
    ("Is Nilgiri's workforce growing?", ["607,979"]),
    ("What did Aravali's CEO say on the Q4 FY25 call?", ["USD 11.6 billion"]),
    ("Did Ganga Bank hit its NIM guidance in Q4?", ["3.4%", "3.6% to 3.7%"]),
])
def test_everyday_wording(app, question, expected):
    r = app.ask(question, use_memory=False)
    assert all(x in r["answer"] for x in expected), r["answer"]
    assert "##" not in r["answer"]


def test_smoke_eval_meets_targets():
    """CI gate: 30 questions across all types must stay above the NFR targets."""
    from scripts.evaluate import smoke_set
    result = evaluate(settings, smoke_set(load_golden()))["summary"]
    assert result["faithfulness"] >= 0.9
    assert result["numeric_em"] >= 0.85
    assert result["abstention_accuracy"] >= 0.8
    assert result["injection_resisted"] == 1.0
