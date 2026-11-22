import pytest
import yaml
from fastapi.testclient import TestClient

from api import create_app
from src.sql_agent.executor import execute_query
from src.sql_agent.guardrails import GuardrailError, enforce_sql_guardrails
from src.sql_agent.postprocess import build_all, resolve_names
from src.sql_agent.postprocess.detect import detect_columns
from tests.conftest import FakeLLM

CURRENCY_CFG = {"base_currency": "SGD", "conversion_rates": {"SGD": 1.0, "USD": 1.35}}
PP_CFG = {"default_enabled": [], "currency": CURRENCY_CFG}
FN = "SELECT AVG(convert_currency(price)) AS avg_price FROM bookings"


@pytest.fixture
def make_client(tmp_path):
    def _make(db, responses, sample_limit=0):
        cfg = {
            "default_db": str(db),
            "llm": {"model_name": "x", "model_provider": "groq"},
            "retrieval": {"sample_limit": sample_limit},
            "guardrails": {"max_rows": 100},
            "agent_loop": {"max_attempts": 3},
            "uploads": {"dir": str(tmp_path / "up"), "max_mb": 1},
            "checkpoint_db": str(tmp_path / "ck.db"),
            "postprocess": PP_CFG,
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(cfg))
        llm = FakeLLM(responses)
        return TestClient(create_app(str(p), llm=llm)), llm

    return _make


# -- detection ------------------------------------------------------------
def test_detects_currency_column(priced_db):
    found = detect_columns(str(priced_db), build_all(PP_CFG))
    assert [(d.table, d.column, d.converter) for d in found] == [
        ("bookings", "price", "currency")
    ]
    assert found[0].example in ("SGD$ 100.00", "USD$ 100.00")


def test_detects_nothing_in_unrelated_db(shop_db):
    assert detect_columns(str(shop_db), build_all(PP_CFG)) == []


def test_numeric_columns_are_not_scanned(tmp_path):
    import sqlite3

    p = tmp_path / "n.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE t (amount REAL, code INTEGER)")
    c.execute("INSERT INTO t VALUES (1.5, 7)")
    c.commit()
    c.close()
    assert detect_columns(str(p), build_all(PP_CFG)) == []


def test_resolve_names_expands_auto_only_to_detected():
    assert resolve_names(["auto"], ["currency"]) == ["currency"]
    assert resolve_names(["auto"], []) == []
    assert resolve_names(["currency", "auto"], ["currency"]) == ["currency"]


# -- SQL functions --------------------------------------------------------
def test_sql_function_makes_aggregates_correct(priced_db):
    conv = build_all(PP_CFG)[0]
    fns = {conv.sql_name: conv.sql_function()}
    df = execute_query(str(priced_db), FN, functions=fns)
    assert df["avg_price"].iloc[0] == pytest.approx((100.0 + 135.0) / 2)


def test_sql_function_returns_null_not_error_on_bad_input(priced_db):
    conv = build_all(PP_CFG)[0]
    fns = {conv.sql_name: conv.sql_function()}
    df = execute_query(
        str(priced_db), "SELECT convert_currency('not money') AS v", functions=fns
    )
    assert df["v"].iloc[0] is None or df["v"].isna().iloc[0]


def test_guardrail_allows_function_calls_but_still_blocks_writes():
    assert "convert_currency" in enforce_sql_guardrails(FN, 100).lower()
    with pytest.raises(GuardrailError):
        enforce_sql_guardrails("DELETE FROM bookings", 100)


def test_read_only_connection_unchanged_by_functions(priced_db):
    conv = build_all(PP_CFG)[0]
    fns = {conv.sql_name: conv.sql_function()}
    with pytest.raises(Exception):
        execute_query(str(priced_db), "UPDATE bookings SET price='x'", functions=fns)


# -- end to end through the API ------------------------------------------
def test_auto_adds_hint_runs_function_and_reports_it(make_client, priced_db):
    c, llm = make_client(priced_db, [FN, "Average is 117.5 SGD."])
    with c:
        r = c.post("/chat", json={"message": "avg price", "conversions": ["auto"]}).json()
    assert not r["failed"]
    assert r["rows"][0]["avg_price"] == pytest.approx(117.5)
    assert "convert_currency(price)" in llm.prompts[0]  # hint in the generate prompt
    assert "SGD$ 100.00" not in llm.prompts[0]  # sample_limit 0: no example value
    assert r["conversions_applied"] == [
        {"column": "price", "converter": "currency", "detail": "to SGD (in SQL)"}
    ]
    assert "in SQL" in llm.prompts[-1]  # the summary is told


def test_hint_includes_example_only_when_samples_allowed(make_client, priced_db):
    c, llm = make_client(priced_db, [FN, "ok"], sample_limit=3)
    with c:
        c.post("/chat", json={"message": "avg", "conversions": ["auto"]})
    assert "such as '" in llm.prompts[0]


def test_auto_is_a_noop_on_a_database_without_convertible_columns(make_client, shop_db):
    c, llm = make_client(shop_db, ["SELECT name FROM customers", "Ann, Bob."])
    with c:
        r = c.post("/chat", json={"message": "names", "conversions": ["auto"]}).json()
    assert not r["failed"] and r["conversions_applied"] == []
    assert "Text columns with convertible values" not in llm.prompts[0]


def test_nothing_is_added_when_conversions_are_off(make_client, priced_db):
    c, llm = make_client(priced_db, ["SELECT price FROM bookings", "ok"])
    with c:
        r = c.post("/chat", json={"message": "prices", "conversions": []}).json()
    assert r["conversions_applied"] == []
    assert "convert_currency" not in llm.prompts[0]


def test_function_unavailable_when_not_enabled_triggers_retry(make_client, priced_db):
    c, llm = make_client(priced_db, [FN, "SELECT price FROM bookings", "ok"])
    with c:
        r = c.post("/chat", json={"message": "avg", "conversions": []}).json()
    assert r["attempts"] == 2 and "no such function" in llm.prompts[1].lower()


def test_row_path_does_not_convert_twice(make_client, priced_db):
    sql = "SELECT id, convert_currency(price) AS price FROM bookings WHERE price IS NOT NULL ORDER BY id"
    c, _ = make_client(priced_db, [sql, "ok"])
    with c:
        r = c.post("/chat", json={"message": "prices", "conversions": ["auto"]}).json()
    assert [row["price"] for row in r["rows"]] == [100.0, 135.0]


def test_schema_endpoint_lists_convertible_columns(make_client, priced_db):
    c, _ = make_client(priced_db, ["x"])
    with c:
        body = c.get("/schema").json()
    assert body["convertible"] == [
        {"table": "bookings", "column": "price", "converter": "currency"}
    ]


# -- raw-column guard -----------------------------------------------------
@pytest.mark.parametrize(
    "raw_sql",
    [
        "SELECT MAX(price) FROM bookings",
        "SELECT MIN(price) FROM bookings",
        "SELECT AVG(price) FROM bookings",
        "SELECT SUM(price) FROM bookings",
        "SELECT id FROM bookings ORDER BY price",
        "SELECT id FROM bookings WHERE price > 'USD$ 50'",
    ],
)
def test_raw_use_of_a_detected_column_is_sent_back_for_retry(make_client, priced_db, raw_sql):
    c, llm = make_client(priced_db, [raw_sql, FN, "ok"])
    with c:
        r = c.post("/chat", json={"message": "q", "conversions": ["auto"]}).json()
    assert r["attempts"] == 2 and not r["failed"]
    assert "Do not use these columns raw" in llm.prompts[1]
    assert "convert_currency(price)" in llm.prompts[1]


@pytest.mark.parametrize(
    "ok_sql",
    [
        "SELECT price FROM bookings",
        "SELECT COUNT(price) FROM bookings",
        "SELECT room, COUNT(*) FROM bookings GROUP BY price",
        "SELECT id FROM bookings WHERE convert_currency(price) > 50",
        "SELECT id FROM bookings ORDER BY convert_currency(price)",
    ],
)
def test_wrapped_or_non_comparing_use_is_allowed(make_client, priced_db, ok_sql):
    c, _ = make_client(priced_db, [ok_sql, "ok"])
    with c:
        r = c.post("/chat", json={"message": "q", "conversions": ["auto"]}).json()
    assert r["attempts"] == 1 and not r["failed"]


def test_guard_is_off_when_conversions_are_off(make_client, priced_db):
    c, _ = make_client(priced_db, ["SELECT MAX(price) FROM bookings", "ok"])
    with c:
        r = c.post("/chat", json={"message": "q", "conversions": []}).json()
    assert r["attempts"] == 1 and not r["failed"]
