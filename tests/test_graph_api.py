import hashlib

import pytest
import yaml
from fastapi.testclient import TestClient

from api import create_app
from tests.conftest import FakeLLM


@pytest.fixture
def make_client(tmp_path, shop_db):
    def _make(responses, **overrides):
        cfg = {
            "default_db": str(shop_db),
            "llm": {"model_name": "x", "model_provider": "groq"},
            "retrieval": {"sample_limit": 0},
            "guardrails": {"max_rows": 100},
            "agent_loop": {"max_attempts": 3},
            "uploads": {"dir": str(tmp_path / "up"), "max_mb": 1},
            "checkpoint_db": str(tmp_path / "ck.db"),
        }
        cfg.update(overrides)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg))
        llm = FakeLLM(responses)
        return TestClient(create_app(str(path), llm=llm)), llm

    return _make


def test_happy_path(make_client):
    c, llm = make_client(
        ["SELECT city, COUNT(*) AS n FROM customers GROUP BY city", "Two cities."]
    )
    with c:
        r = c.post("/chat", json={"message": "customers per city"}).json()
    assert r["failed"] is False and r["attempts"] == 1
    assert r["columns"] == ["city", "n"] and len(r["rows"]) == 2
    assert r["answer"] == "Two cities."


def test_retry_feeds_error_back(make_client):
    c, llm = make_client(
        [
            "SELECT nope FROM customers",  # execution error
            "SELECT name FROM customers WHERE city = 'Atlantis'",  # empty
            "SELECT name FROM customers",
            "Ann and Bob.",
        ]
    )
    with c:
        r = c.post("/chat", json={"message": "names"}).json()
    assert r["attempts"] == 3 and not r["failed"]
    assert "### Previous attempt failed" in llm.prompts[1]  # retry_feedback.md
    assert "SQL: SELECT nope FROM customers" in llm.prompts[1]
    assert "Write a corrected query" in llm.prompts[1]
    assert "### Previous attempt failed" not in llm.prompts[0]
    assert "no such column" in llm.prompts[1]
    assert "returned no rows" in llm.prompts[2]


def test_guardrail_exhaustion_fails_cleanly(make_client, shop_db):
    before = hashlib.sha256(shop_db.read_bytes()).hexdigest()
    c, llm = make_client(["DELETE FROM customers"])
    with c:
        r = c.post("/chat", json={"message": "delete everyone"}).json()
    assert r["failed"] is True and r["attempts"] == 3
    assert r["rows"] == []
    assert hashlib.sha256(shop_db.read_bytes()).hexdigest() == before


def test_decline_returns_no_query_rows_or_retries(make_client):
    c, llm = make_client(["CANNOT_ANSWER"])
    with c:
        r = c.post("/chat", json={"message": "delete everyone"}).json()
    assert r["declined"] is True and r["failed"] is False
    assert r["sql"] == "" and r["rows"] == [] and r["columns"] == []
    assert r["attempts"] == 0
    assert len(llm.prompts) == 1  # no execution, no retry, no summarize call
    assert "read-only" in r["answer"]


def test_decline_after_failed_attempt(make_client):
    c, llm = make_client(["SELECT nope FROM customers", "CANNOT_ANSWER"])
    with c:
        r = c.post("/chat", json={"message": "q"}).json()
    assert r["declined"] is True and r["rows"] == []


def test_declined_flag_resets_next_turn(make_client):
    c, _ = make_client(["CANNOT_ANSWER", "SELECT name FROM customers", "Ann, Bob."])
    with c:
        first = c.post("/chat", json={"message": "x", "thread_id": "t"}).json()
        second = c.post("/chat", json={"message": "names", "thread_id": "t"}).json()
    assert first["declined"] is True
    assert second["declined"] is False and len(second["rows"]) == 2


def test_upload_uses_uploaded_schema(make_client, wide_db):
    c, llm = make_client(["SELECT COUNT(*) AS n FROM t3", "Zero."])
    with c:
        up = c.post(
            "/upload-db", files={"file": ("my.db", wide_db.read_bytes())}
        ).json()
        r = c.post(
            "/chat", json={"message": "count t3", "db_id": up["db_id"]}
        ).json()
    assert not r["failed"]
    assert "CREATE TABLE t3" in llm.prompts[0]
    assert "customers" not in llm.prompts[0]


@pytest.mark.parametrize(
    "name,content", [("a.txt", b"SQLite format 3\x00"), ("a.db", b"not sqlite")]
)
def test_upload_rejects_bad_files(make_client, name, content):
    c, _ = make_client(["x"])
    with c:
        assert c.post("/upload-db", files={"file": (name, content)}).status_code == 400


def test_unknown_db_404(make_client):
    c, _ = make_client(["x"])
    with c:
        assert c.post("/chat", json={"message": "q", "db_id": "nope"}).status_code == 404


def test_old_max_attempts_location_fails_at_startup(tmp_path, shop_db):
    cfg = {
        "default_db": str(shop_db),
        "llm": {"model_name": "x", "model_provider": "groq"},
        "guardrails": {"max_rows": 100, "max_attempts": 3},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="agent_loop"):
        create_app(str(path), llm=FakeLLM(["x"]))


def test_agent_loop_max_attempts_is_honoured(make_client):
    c, llm = make_client(["DELETE FROM customers"], agent_loop={"max_attempts": 2})
    with c:
        r = c.post("/chat", json={"message": "x"}).json()
    assert r["attempts"] == 2 and r["failed"] is True


def test_summary_is_told_when_the_result_was_capped(make_client):
    many = (
        "SELECT 1 AS n FROM customers a, customers b, customers c, customers d, "
        "customers e, customers f, customers g"
    )  # 2^7 = 128 rows, capped at max_rows = 100
    c, llm = make_client([many, "Lots of rows."])
    with c:
        r = c.post("/chat", json={"message": "list them"}).json()
    assert r["truncated"] is True
    summary = llm.prompts[-1]
    assert "Rows returned: 100" in summary
    assert "the row limit" in summary and "do not state 100 as the total" in summary


def test_summary_has_no_cap_note_when_the_result_is_complete(make_client):
    c, llm = make_client(
        ["SELECT city, COUNT(*) AS n FROM customers GROUP BY city", "Two cities."]
    )
    with c:
        r = c.post("/chat", json={"message": "customers per city"}).json()
    assert r["truncated"] is False
    assert "row limit" not in llm.prompts[-1]
