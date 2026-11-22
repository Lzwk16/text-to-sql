import hashlib

import pytest
import yaml
from fastapi.testclient import TestClient

from api import create_app
from tests.conftest import FakeLLM

Q1 = "SELECT city, COUNT(*) AS n FROM customers GROUP BY city"
Q2 = "SELECT city, COUNT(*) AS n FROM customers WHERE city = 'Paris' GROUP BY city"


def write_config(tmp_path, db, history_turns=5):
    cfg = {
        "default_db": str(db),
        "llm": {"model_name": "x", "model_provider": "groq"},
        "retrieval": {"sample_limit": 0},
        "guardrails": {"max_rows": 100},
        "agent_loop": {"max_attempts": 3, "history_turns": history_turns},
        "uploads": {"dir": str(tmp_path / "up"), "max_mb": 1},
        "checkpoint_db": str(tmp_path / "ck.db"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


@pytest.fixture
def make_client(tmp_path, shop_db):
    def _make(responses, history_turns=5, db=None):
        llm = FakeLLM(responses)
        cfg = write_config(tmp_path, db or shop_db, history_turns)
        return TestClient(create_app(cfg, llm=llm)), llm

    return _make


def chat(c, message, thread="t1", **extra):
    return c.post("/chat", json={"message": message, "thread_id": thread, **extra}).json()


def generate_prompts(llm):
    """Prompts sent to the SQL-writing step (every other call is a summary)."""
    return llm.prompts[0::2]


def test_first_turn_has_no_history_block(make_client):
    c, llm = make_client([Q1, "Two cities."])
    with c:
        chat(c, "customers per city")
    assert "Earlier in this conversation" not in llm.prompts[0]


def test_follow_up_sees_the_previous_question_and_sql(make_client):
    c, llm = make_client([Q1, "Two cities.", Q2, "One city."])
    with c:
        chat(c, "customers per city")
        r = chat(c, "now only Paris")
    assert not r["failed"]
    second = generate_prompts(llm)[1]
    assert "Earlier in this conversation" in second
    assert "Question: customers per city" in second
    assert f"SQL: {Q1}" in second
    assert "Returned 2 rows with columns: city, n" in second


def test_result_rows_are_never_put_in_the_history(make_client):
    c, llm = make_client(
        ["SELECT name FROM customers ORDER BY id", "Ann and Bob.", Q1, "ok"]
    )
    with c:
        chat(c, "names")
        chat(c, "and cities")
    block = generate_prompts(llm)[1].split("Earlier in this conversation")[1]
    assert "Ann" not in block and "Bob" not in block


def test_window_limits_how_many_turns_are_shown(make_client):
    qs = [f"SELECT {i} AS n" for i in range(1, 5)]
    responses = [x for q in qs for x in (q, f"answer {q}")] + [qs[0], "ok"]
    c, llm = make_client(responses, history_turns=2)
    with c:
        for i in range(1, 5):
            chat(c, f"question number {i}")
        chat(c, "question number 5")
    fifth = generate_prompts(llm)[4]
    assert "question number 4" in fifth and "question number 3" in fifth
    assert "question number 2" not in fifth and "question number 1" not in fifth


def test_zero_turns_turns_the_feature_off(make_client):
    c, llm = make_client([Q1, "ok", Q2, "ok"], history_turns=0)
    with c:
        chat(c, "customers per city")
        chat(c, "now only Paris")
    assert "Earlier in this conversation" not in generate_prompts(llm)[1]


def test_a_new_thread_starts_with_no_history(make_client):
    c, llm = make_client([Q1, "ok", Q2, "ok"])
    with c:
        chat(c, "customers per city", thread="a")
        chat(c, "now only Paris", thread="b")
    assert "Earlier in this conversation" not in generate_prompts(llm)[1]


def test_history_does_not_cross_databases_on_the_same_thread(make_client, tmp_path):
    import sqlite3

    other = tmp_path / "other.db"
    con = sqlite3.connect(other)
    con.execute("CREATE TABLE things (id INTEGER, label TEXT)")
    con.execute("INSERT INTO things VALUES (1, 'a')")
    con.commit()
    con.close()
    c, llm = make_client([Q1, "ok", "SELECT label FROM things", "ok"])
    with c:
        chat(c, "customers per city")  # default database
        db_id = c.post("/upload-db", files={"file": ("o.db", other.read_bytes())}).json()["db_id"]
        chat(c, "list labels", db_id=db_id)  # same thread, other database
    assert "Earlier in this conversation" not in generate_prompts(llm)[1]


def test_declined_turns_are_not_remembered(make_client):
    c, llm = make_client(["CANNOT_ANSWER", Q1, "ok", Q2, "ok"])
    with c:
        chat(c, "delete everything")  # declined
        chat(c, "customers per city")
        chat(c, "now only Paris")
    prompts = [p for p in llm.prompts if "Generate one SQLite SELECT" in p]
    assert "Earlier in this conversation" not in prompts[1]  # after the decline
    assert "delete everything" not in prompts[2]
    assert "Question: customers per city" in prompts[2]


def test_failed_turns_are_not_remembered(make_client):
    c, llm = make_client(["DELETE FROM customers", "DELETE FROM customers", "DELETE FROM customers", Q1, "ok"])
    with c:
        r = chat(c, "delete everyone")
        assert r["failed"] is True
        chat(c, "customers per city")
    last = [p for p in llm.prompts if "Generate one SQLite SELECT" in p][-1]
    assert "Earlier in this conversation" not in last
    assert "DELETE FROM customers" not in last.split("### Database Schema")[1].split("### Answer")[0].replace("Previous attempt failed", "")


def test_history_survives_an_app_restart(tmp_path, shop_db):
    cfg = write_config(tmp_path, shop_db)
    first = FakeLLM([Q1, "Two cities."])
    with TestClient(create_app(cfg, llm=first)) as c:
        chat(c, "customers per city")
    second = FakeLLM([Q2, "One city."])  # a new process: new app, same checkpoint file
    with TestClient(create_app(cfg, llm=second)) as c:
        chat(c, "now only Paris")
    assert f"SQL: {Q1}" in second.prompts[0]


def test_user_database_is_not_modified(make_client, shop_db):
    before = hashlib.sha256(shop_db.read_bytes()).hexdigest()
    c, _ = make_client([Q1, "ok", Q2, "ok"])
    with c:
        chat(c, "a")
        chat(c, "b")
    assert hashlib.sha256(shop_db.read_bytes()).hexdigest() == before
