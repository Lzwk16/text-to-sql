import hashlib
import sqlite3

import pytest

from src.sql_agent.executor import execute_query
from src.sql_agent.retriever import SchemaProvider
from utils.schema import introspect


def test_executor_returns_dataframe(shop_db):
    df = execute_query(str(shop_db), "SELECT COUNT(*) AS n FROM orders")
    assert df.iloc[0]["n"] == 3


def test_executor_is_read_only(shop_db):
    before = hashlib.sha256(shop_db.read_bytes()).hexdigest()
    with pytest.raises(Exception):
        execute_query(str(shop_db), "DELETE FROM orders")
    with pytest.raises(sqlite3.OperationalError):
        from src.sql_agent.executor import connect_readonly

        connect_readonly(str(shop_db)).execute("DROP TABLE orders")
    assert hashlib.sha256(shop_db.read_bytes()).hexdigest() == before


def test_introspect_fks_and_samples(shop_db):
    tables = {t.name: t for t in introspect(str(shop_db), sample_limit=2)}
    assert set(tables) == {"customers", "orders"}
    assert tables["orders"].referenced_tables == ["customers"]
    assert tables["orders"].row_count == 3
    assert len(tables["orders"].samples) == 2


def test_no_samples_when_disabled(shop_db):
    tables = introspect(str(shop_db), sample_limit=0)
    assert all(t.samples == [] for t in tables)


def test_small_db_gets_full_schema(shop_db):
    text = SchemaProvider().schema_for("a", str(shop_db), "anything")
    assert "customers" in text and "orders" in text


def test_schema_cache_is_per_db(shop_db, wide_db):
    p = SchemaProvider()
    assert "customers" in p.schema_for("a", str(shop_db), "q")
    assert {t.name for t in p.tables("b", str(wide_db))} >= {"t0", "t11"}
    assert {t.name for t in p.tables("a", str(shop_db))} == {"customers", "orders"}
