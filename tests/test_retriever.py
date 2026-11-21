import pytest

from src.sql_agent.retriever import SchemaProvider
from utils.schema import introspect


class StubRetriever:
    """Returns fixed hits so FK expansion can be tested without any model."""

    def __init__(self, names):
        from langchain_core.documents import Document

        self.docs = [Document(page_content=n, metadata={"table": n}) for n in names]

    def invoke(self, question):
        return self.docs


def _provider_with_hits(fk_db, hits):
    p = SchemaProvider(sample_limit=0, full_schema_max_tables=8, k=5)
    p._retrievers["db"] = StubRetriever(hits)
    return p


def names_in(schema_text):
    return {
        line.split()[2].strip('"')
        for line in schema_text.splitlines()
        if line.upper().startswith("CREATE TABLE")
    }


def test_fk_expansion_adds_parents_and_children(fk_db):
    p = _provider_with_hits(fk_db, ["orders"])
    got = names_in(p.schema_for("db", str(fk_db), "orders per customer"))
    # orders -> customers (parent) and order_items -> orders (child)
    assert got == {"orders", "customers", "order_items"}


def test_fk_expansion_excludes_unrelated(fk_db):
    p = _provider_with_hits(fk_db, ["customers"])
    got = names_in(p.schema_for("db", str(fk_db), "where do customers live"))
    assert got == {"customers", "orders"}
    assert not any(n.startswith("filler") for n in got)


@pytest.mark.slow
def test_real_hybrid_retriever_finds_relevant_tables(fk_db):
    p = SchemaProvider(sample_limit=0, full_schema_max_tables=8, k=3)
    got = names_in(p.schema_for("real", str(fk_db), "total quantity of each product ordered"))
    assert {"order_items", "products"} <= got


def test_tokenizer_splits_quoted_and_underscored_names():
    from src.sql_agent.retriever import tokenize

    assert tokenize('"Cabin Comfort" REAL, cruise_pre, onboardWifi') == [
        "cabin", "comfort", "real", "cruise", "pre", "onboard", "wifi",
    ]


@pytest.mark.slow
def test_quoted_column_names_are_findable(tmp_path):
    """Regression: quoted multi-word columns must not make BM25 miss the table."""
    import sqlite3

    path = tmp_path / "q.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE survey ("Cabin Comfort" REAL, "Onboard Wifi Service" TEXT)')
    for i in range(10):
        conn.execute(f"CREATE TABLE f{i} (id INTEGER PRIMARY KEY, x TEXT)")
    conn.commit()
    conn.close()
    p = SchemaProvider(sample_limit=0, full_schema_max_tables=8, k=3)
    got = names_in(p.schema_for("q", str(path), "average Cabin Comfort score"))
    assert "survey" in got


import asyncio

from src.sql_agent.nodes import AgentNodes
from tests.conftest import FakeLLM
from utils.db_registry import DatabaseRegistry


def _nodes(fk_db, tmp_path, responses, hits):
    reg = DatabaseRegistry(upload_dir=str(tmp_path / "up"), default_db=str(fk_db))
    prov = SchemaProvider(sample_limit=0, full_schema_max_tables=8, k=5)
    prov._retrievers["default"] = StubRetriever(hits)  # retrieval misses
    llm = FakeLLM(responses)
    return AgentNodes(llm, reg, prov, {}), llm


def test_decline_on_partial_schema_retries_with_full_schema(fk_db, tmp_path):
    nodes, llm = _nodes(
        fk_db, tmp_path, ["CANNOT_ANSWER", "SELECT 1 FROM customers"], ["filler0"]
    )
    out = asyncio.run(nodes.generate_sql({"question": "customers?", "messages": []}))
    assert out["declined"] is False and "customers" in out["sql"]
    assert len(llm.prompts) == 2
    assert "CREATE TABLE customers" not in llm.prompts[0]
    assert "CREATE TABLE customers" in llm.prompts[1]  # full schema on the retry


def test_decline_stands_when_full_schema_also_declines(fk_db, tmp_path):
    nodes, llm = _nodes(fk_db, tmp_path, ["CANNOT_ANSWER"], ["filler0"])
    out = asyncio.run(nodes.generate_sql({"question": "delete all", "messages": []}))
    assert out["declined"] is True and len(llm.prompts) == 2


def test_no_second_call_when_schema_was_already_full(shop_db, tmp_path):
    reg = DatabaseRegistry(upload_dir=str(tmp_path / "up"), default_db=str(shop_db))
    llm = FakeLLM(["CANNOT_ANSWER"])
    nodes = AgentNodes(llm, reg, SchemaProvider(sample_limit=0), {})
    out = asyncio.run(nodes.generate_sql({"question": "x", "messages": []}))
    assert out["declined"] is True and len(llm.prompts) == 1
