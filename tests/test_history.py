import hashlib

import pytest
import yaml
from fastapi.testclient import TestClient

from api import create_app
from tests.conftest import FakeLLM
from tests.test_currency import CURRENCY_CFG

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
        # Off unless a request asks for it; needed by the convert_* test.
        "postprocess": {"default_enabled": [], "currency": CURRENCY_CFG},
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
    return c.post(
        "/chat", json={"message": message, "thread_id": thread, **extra}
    ).json()


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
        db_id = c.post(
            "/upload-db", files={"file": ("o.db", other.read_bytes())}
        ).json()["db_id"]
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
    c, llm = make_client(
        [
            "DELETE FROM customers",
            "DELETE FROM customers",
            "DELETE FROM customers",
            Q1,
            "ok",
        ]
    )
    with c:
        r = chat(c, "delete everyone")
        assert r["failed"] is True
        chat(c, "customers per city")
    last = [p for p in llm.prompts if "Generate one SQLite SELECT" in p][-1]
    assert "Earlier in this conversation" not in last
    # No history block and no retry feedback: the failed DELETE left no trace.
    assert "DELETE" not in last.split("### Database Schema")[1]


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


def history_block(prompt):
    """Text of the history block only (between its heading and the answer)."""
    return prompt.split("Earlier in this conversation")[1].split("### Answer")[0]


def test_saved_sql_is_the_model_sql_without_the_injected_limit(make_client):
    c, llm = make_client([Q1, "ok", Q2, "ok"])
    with c:
        r = chat(c, "customers per city")
        assert "LIMIT 100" in r["sql"]  # the query that ran was capped
        chat(c, "now only Paris")
    block = history_block(generate_prompts(llm)[1])
    assert f"SQL: {Q1}\n" in block
    assert "LIMIT" not in block


def test_write_follow_up_is_still_told_to_decline(make_client):
    c, llm = make_client([Q1, "ok", "CANNOT_ANSWER"])
    with c:
        chat(c, "customers per city")
        r = chat(c, "now delete them")
    assert r["declined"] is True and not r["failed"]
    block = history_block(generate_prompts(llm)[1])
    assert "asks to change data" in block and "CANNOT_ANSWER" in block


def test_default_database_shares_history_whether_named_or_not(make_client):
    c, llm = make_client([Q1, "ok", Q2, "ok"])
    with c:
        chat(c, "customers per city")  # db_id omitted
        chat(c, "now only Paris", db_id="default")
    assert "Question: customers per city" in generate_prompts(llm)[1]


def test_history_is_dropped_when_the_default_database_changes(tmp_path, shop_db):
    import sqlite3

    other = tmp_path / "other.db"
    con = sqlite3.connect(other)
    con.execute("CREATE TABLE things (id INTEGER, label TEXT)")
    con.execute("INSERT INTO things VALUES (1, 'a')")
    con.commit()
    con.close()
    first = FakeLLM([Q1, "ok"])
    with TestClient(create_app(write_config(tmp_path, shop_db), llm=first)) as c:
        chat(c, "customers per city")
    second = FakeLLM(["SELECT label FROM things", "ok"])
    # Same checkpoint file, but the operator pointed default_db elsewhere.
    with TestClient(create_app(write_config(tmp_path, other), llm=second)) as c:
        chat(c, "list labels")
    assert "Earlier in this conversation" not in second.prompts[0]


def test_long_sql_is_left_out_of_history_instead_of_cut(make_client):
    long_sql = (
        "SELECT city FROM customers WHERE "
        + " OR ".join(f"name = 'x{i}'" for i in range(200))
        + " OR 1 = 1"
    )
    c, llm = make_client([long_sql, "ok", Q1, "ok"])
    with c:
        chat(c, "cities")
        chat(c, "and counts")
    block = history_block(generate_prompts(llm)[1])
    assert "Question: cities" in block
    assert "name = 'x0'" not in block
    assert "too long" in block


def test_long_question_is_shortened_in_history(make_client):
    question = "customers per city " + "please " * 300
    c, llm = make_client([Q1, "ok", Q2, "ok"])
    with c:
        chat(c, question)
        chat(c, "now only Paris")
    block = history_block(generate_prompts(llm)[1])
    question_line = next(x for x in block.splitlines() if "Question:" in x)
    assert len(question_line) < 600 and question_line.rstrip().endswith("...")


def test_follow_up_keeps_tables_from_history_on_large_schemas(make_client, fk_db):
    import sqlite3

    from tests.test_retriever import StubRetriever

    con = sqlite3.connect(fk_db)
    con.execute("INSERT INTO customers VALUES (1, 'Ann', 'x')")
    con.commit()
    con.close()
    first = "SELECT city, COUNT(*) AS n FROM customers GROUP BY city"
    c, llm = make_client(
        [first, "ok", "SELECT 1 FROM customers WHERE city = 'x'", "ok"], db=fk_db
    )
    with c:
        hits = c.app.state.provider._retrievers
        hits["default"] = StubRetriever(["customers"])
        chat(c, "customers per city")
        hits["default"] = StubRetriever(["filler0"])  # follow-up text matches nothing
        chat(c, "now only Paris")
    second = generate_prompts(llm)[1]
    schema = second.split("### Database Schema")[1].split("Earlier in")[0]
    assert "CREATE TABLE customers" in schema
    assert "CREATE TABLE filler0" in schema


def seed(db, *statements):
    import sqlite3

    con = sqlite3.connect(db)
    for s in statements:
        con.execute(s)
    con.commit()
    con.close()


def large_schema_client(make_client, fk_db, responses):
    """Client on the 12-table DB whose retrieval only ever finds filler0."""
    from tests.test_retriever import StubRetriever

    c, llm = make_client(responses, db=fk_db)
    return (
        c,
        llm,
        lambda: c.app.state.provider._retrievers.__setitem__(
            "default", StubRetriever(["filler0"])
        ),
    )


def schema_part(prompt):
    return prompt.split("### Database Schema")[1].split("Earlier in")[0]


def test_long_sql_follow_up_still_gets_its_tables(make_client, fk_db):
    seed(fk_db, "INSERT INTO customers VALUES (1, 'Ann', 'x')")
    long_sql = (
        "SELECT city FROM customers WHERE "
        + " OR ".join(f"name = 'x{i}'" for i in range(200))
        + " OR 1 = 1"
    )
    c, llm, miss = large_schema_client(
        make_client, fk_db, [long_sql, "ok", "SELECT 1 FROM customers", "ok"]
    )
    with c:
        miss()
        chat(c, "cities")
        chat(c, "now only Paris")
    second = generate_prompts(llm)[1]
    assert "too long" in history_block(second)
    assert "CREATE TABLE customers" in schema_part(second)


def test_only_the_most_recent_turns_add_tables(make_client, fk_db):
    seed(
        fk_db,
        "INSERT INTO customers VALUES (1, 'Ann', 'x')",
        "INSERT INTO products VALUES (1, 'Pen', 2.0)",
        "INSERT INTO orders VALUES (1, 1, '2026-01-01')",
    )
    turns = [
        "SELECT id FROM orders",
        "SELECT id FROM products",
        "SELECT id FROM customers",
    ]
    responses = [x for q in turns for x in (q, "ok")] + ["SELECT 1", "ok"]
    c, llm, miss = large_schema_client(make_client, fk_db, responses)
    with c:
        miss()
        for i in range(3):
            chat(c, f"question {i}")
        chat(c, "how many products are there")
    schema = schema_part(generate_prompts(llm)[3])
    assert "CREATE TABLE customers" in schema and "CREATE TABLE products" in schema
    assert "CREATE TABLE orders" not in schema  # three turns back
    assert "question 0" in history_block(generate_prompts(llm)[3])  # still shown


def test_tables_added_from_history_are_size_capped(make_client, fk_db, monkeypatch):
    import src.sql_agent.nodes as nodes

    monkeypatch.setattr(nodes, "MAX_HISTORY_SCHEMA", 10)
    seed(fk_db, "INSERT INTO customers VALUES (1, 'Ann', 'x')")
    c, llm, miss = large_schema_client(
        make_client, fk_db, ["SELECT id FROM customers", "ok", "SELECT 1", "ok"]
    )
    with c:
        miss()
        chat(c, "ids")
        chat(c, "and names")
    assert "CREATE TABLE customers" not in schema_part(generate_prompts(llm)[1])


def test_capped_row_count_is_shown_as_a_lower_bound(make_client):
    many = (
        "SELECT 1 AS n FROM customers a, customers b, customers c, customers d, "
        "customers e, customers f, customers g"
    )  # 2^7 = 128 rows, capped at max_rows = 100
    c, llm = make_client([many, "ok", Q1, "ok"])
    with c:
        r = chat(c, "lots")
        assert r["truncated"] is True
        chat(c, "how many was that")
    assert "the row cap, so there may be more" in history_block(
        generate_prompts(llm)[1]
    )


def test_database_identity_uses_the_resolved_path(tmp_path, shop_db):
    from src.sql_agent.nodes import AgentNodes
    from src.sql_agent.retriever import SchemaProvider
    from utils.db_registry import DatabaseRegistry

    (tmp_path / "sub").mkdir()
    roundabout = tmp_path / "sub" / ".." / shop_db.name
    reg = DatabaseRegistry(upload_dir=str(tmp_path / "up"), default_db=str(roundabout))
    nodes = AgentNodes(FakeLLM(["x"]), reg, SchemaProvider(sample_limit=0), {})
    assert nodes._db_identity({}) == ("default", str(shop_db.resolve()))


def test_a_table_too_big_for_the_cap_does_not_block_smaller_ones(
    make_client, fk_db, monkeypatch
):
    import src.sql_agent.nodes as nodes
    from utils.schema import introspect

    seed(
        fk_db,
        "INSERT INTO customers VALUES (1, 'Ann', 'x')",
        "INSERT INTO orders VALUES (1, 1, '2026-01-01')",
    )
    sizes = {t.name: len(t.render()) + 2 for t in introspect(str(fk_db), 0)}
    assert sizes["orders"] > sizes["customers"]
    monkeypatch.setattr(nodes, "MAX_HISTORY_SCHEMA", sizes["customers"])
    responses = ["SELECT id FROM customers", "ok", "SELECT id FROM orders", "ok"]
    c, llm, miss = large_schema_client(make_client, fk_db, responses + ["SELECT 1"])
    with c:
        miss()
        chat(c, "customer ids")
        chat(c, "order ids")  # newest turn: orders is checked first, and too big
        chat(c, "and for Paris")
    schema = schema_part(generate_prompts(llm)[2])
    assert "CREATE TABLE orders" not in schema
    assert "CREATE TABLE customers" in schema


def test_history_notes_conversion_functions_that_are_off_now(make_client, priced_db):
    used = "SELECT MAX(convert_currency(price)) AS m FROM bookings"
    c, llm = make_client([used, "ok", used, "ok", used, "ok"], db=priced_db)
    with c:
        chat(c, "max price", conversions=["currency"])
        chat(c, "same again", conversions=["currency"])  # still on: no note
        chat(c, "and without conversion", conversions=[])
    still_on = history_block(generate_prompts(llm)[1])
    now_off = history_block(generate_prompts(llm)[2])
    assert "not available" not in still_on
    assert "convert_currency is not available" in now_off


def test_summarize_skips_history_when_the_database_is_gone(tmp_path, shop_db):
    import asyncio

    from src.sql_agent.nodes import AgentNodes
    from src.sql_agent.retriever import SchemaProvider
    from utils.db_registry import DatabaseRegistry

    reg = DatabaseRegistry(upload_dir=str(tmp_path / "up"), default_db=str(shop_db))
    nodes = AgentNodes(FakeLLM(["Two cities."]), reg, SchemaProvider(sample_limit=0))
    state = {
        "db_id": "removed-upload",  # cleaned up after the query ran
        "question": "q",
        "sql": Q1,
        "columns": ["city"],
        "rows": [{"city": "Paris"}],
    }
    out = asyncio.run(nodes.summarize(state))
    assert out["answer"] == "Two cities."
    assert "history" not in out
