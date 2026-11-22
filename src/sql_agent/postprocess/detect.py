"""Find columns a converter recognises, from sampled values (read-only)."""

import sqlite3
from typing import List

import pandas as pd

from src.sql_agent.executor import connect_readonly
from src.sql_agent.postprocess.base import Converter, Detected

SAMPLE_VALUES = 200  # distinct non-null values examined per column
COLUMN_BUDGET_S = 3.0  # a slow column is skipped, never fatal


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_text_affinity(declared_type: str) -> bool:
    """SQLite rule: CHAR, CLOB or TEXT in the type, or no declared type at all."""
    t = (declared_type or "").upper()
    return t == "" or any(k in t for k in ("CHAR", "CLOB", "TEXT"))


def detect_columns(db_path: str, converters: List[Converter]) -> List[Detected]:
    """Return every (table, column) that some converter claims.

    Only text columns are sampled. A column claimed by one converter is not offered
    to later ones, matching ``apply_converters``.
    """
    found: List[Detected] = []
    if not converters:
        return found
    conn = connect_readonly(db_path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        columns = {
            t: [
                r[1]
                for r in conn.execute(f"PRAGMA table_info({_quote(t)})")
                if _is_text_affinity(r[2])
            ]
            for t in tables
        }
    finally:
        conn.close()

    for table, cols in columns.items():
        for col in cols:
            values = _sample(db_path, table, col)
            if not values:
                continue
            series = pd.Series(values)
            for conv in converters:
                if conv.detect(series):
                    found.append(
                        Detected(table, col, conv.name, _example(conv, values))
                    )
                    break
    return found


def _sample(db_path: str, table: str, column: str) -> List[str]:
    conn = connect_readonly(db_path, timeout_s=COLUMN_BUDGET_S)
    try:
        q, t = _quote(column), _quote(table)
        rows = conn.execute(
            f"SELECT DISTINCT {q} FROM {t} WHERE {q} IS NOT NULL LIMIT {SAMPLE_VALUES}"
        ).fetchall()
        return [r[0] for r in rows if isinstance(r[0], str)]
    except sqlite3.Error:  # includes the interrupt raised by the time budget
        return []
    finally:
        conn.close()


def _example(conv: Converter, values: List[str]):
    for v in values:
        if conv.convert_value(v) is not None:
            return v
    return None
