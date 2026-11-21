import pytest

from src.sql_agent.guardrails import GuardrailError, enforce_sql_guardrails


def test_plain_select_gets_limit():
    out = enforce_sql_guardrails("SELECT * FROM t", max_rows=50)
    assert "LIMIT 50" in out.upper()


def test_large_limit_is_capped_small_kept():
    assert "LIMIT 50" in enforce_sql_guardrails("SELECT * FROM t LIMIT 9999", 50).upper()
    assert "LIMIT 5" in enforce_sql_guardrails("SELECT * FROM t LIMIT 5", 50).upper()


def test_cte_select_allowed():
    enforce_sql_guardrails("WITH a AS (SELECT 1 AS x) SELECT x FROM a")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM t",
        "DROP TABLE t",
        "UPDATE t SET a=1",
        "INSERT INTO t VALUES (1)",
        "SELECT 1; DROP TABLE t",
        "PRAGMA writable_schema=1",
        "",
        "WITH x AS (SELECT 1) DELETE FROM t",
    ],
)
def test_blocked(sql):
    with pytest.raises(GuardrailError):
        enforce_sql_guardrails(sql)
