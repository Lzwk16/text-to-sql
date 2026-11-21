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
