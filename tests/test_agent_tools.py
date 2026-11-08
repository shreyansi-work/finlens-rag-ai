"""Agent tool contracts, including how text-to-SQL copes with a weak LLM."""
import pytest

from finlens import llm
from finlens.agent import tools
from finlens.config import settings
from finlens.db import connect


@pytest.fixture(scope="module")
def conn():
    return connect(settings.db_path)


@pytest.mark.parametrize("bad_sql", [
    "SELECT value FROM financials WHERE metric = 'eps'",          # partial columns
    "SELECT AVG(value) AS v FROM financials",                      # aggregate, no provenance
    "this is not sql",                                             # invalid
    "DROP TABLE financials",                                       # forbidden
    "SELECT * FROM documents",                                     # table not allowed
])
def test_weak_llm_sql_falls_back_to_rule_based_builder(conn, monkeypatch, bad_sql):
    """Regression: with Ollama on, "SELECT value ..." produced rows without fiscal_year and crashed the agent."""
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: bad_sql)
    res = tools.query_financials(conn, settings, "ARVL eps", {"ticker": ["ARVL"]})
    assert res["ok"] and res["rows"]
    assert tools.REQUIRED_COLUMNS <= res["rows"][0].keys()
    assert {r["ticker"] for r in res["rows"]} == {"ARVL"}


def test_good_llm_sql_is_used_as_is(conn, monkeypatch):
    good = "SELECT * FROM financials WHERE ticker='NILCS' AND fiscal_year='FY25' AND metric='revenue' AND quarter='FY'"
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: good)
    res = tools.query_financials(conn, settings, "Nilgiri revenue FY25")
    assert [r["value"] for r in res["rows"]] == [255324.0]


@pytest.mark.parametrize("sql", ["SELECT 1; SELECT 2", "UPDATE financials SET value=0", "SELECT * FROM financials; DROP TABLE x",
                                 "SELECT * FROM sqlite_master"])
def test_validate_sql_rejects_unsafe_statements(sql):
    with pytest.raises(ValueError):
        tools.validate_sql(sql)
