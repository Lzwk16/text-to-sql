"""Read-only SQLite execution."""

import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import quote

import pandas as pd


def connect_readonly(
    db_path: str,
    timeout_s: float = 10.0,
    functions: Optional[Dict[str, Callable]] = None,
) -> sqlite3.Connection:
    """Open ``db_path`` read-only, aborting queries that exceed ``timeout_s``.

    ``functions`` maps names to pure one-argument Python callables made available
    to SQL (for example ``convert_currency``). They run inside SQLite, so they must
    not do I/O or touch the database.
    """
    uri = f"file:{quote(str(Path(db_path).resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    for name, fn in (functions or {}).items():
        conn.create_function(name, 1, fn, deterministic=True)
    deadline = time.monotonic() + timeout_s
    # Returning non-zero from the progress handler interrupts the running query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    return conn


def execute_query(
    db_path: str,
    sql: str,
    timeout_s: float = 10.0,
    functions: Optional[Dict[str, Callable]] = None,
) -> pd.DataFrame:
    """Run ``sql`` against ``db_path`` read-only and return a DataFrame."""
    conn = connect_readonly(db_path, timeout_s, functions)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()
