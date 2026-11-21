"""Schema introspection for an arbitrary SQLite database."""

from dataclasses import dataclass, field
from typing import List

from src.sql_agent.executor import connect_readonly


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass
class TableInfo:
    name: str
    ddl: str
    columns: List[str]
    foreign_keys: List[str] = field(default_factory=list)
    referenced_tables: List[str] = field(default_factory=list)
    row_count: int = 0
    samples: List[dict] = field(default_factory=list)

    def render(self) -> str:
        """Text block shown to the LLM (and indexed by the retriever)."""
        parts = [self.ddl.strip().rstrip(";") + ";", f"-- {self.row_count} rows"]
        if self.samples:
            parts.append("-- sample rows:")
            parts.extend(f"--   {row}" for row in self.samples)
        return "\n".join(parts)


def introspect(db_path: str, sample_limit: int = 3) -> List[TableInfo]:
    """Return DDL, FK links, row counts and optional sample rows per table."""
    conn = connect_readonly(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = []
        for name, ddl in cur.fetchall():
            q = _quote(name)
            columns = [r[1] for r in cur.execute(f"PRAGMA table_info({q})")]
            fks = []
            refs = []
            for r in cur.execute(f"PRAGMA foreign_key_list({q})"):
                fks.append(f"{name}.{r[3]} -> {r[2]}.{r[4]}")
                refs.append(r[2])
            row_count = cur.execute(f"SELECT COUNT(*) FROM {q}").fetchone()[0]
            samples = []
            if sample_limit > 0:
                rows = cur.execute(f"SELECT * FROM {q} LIMIT {int(sample_limit)}")
                samples = [dict(zip(columns, row)) for row in rows.fetchall()]
            tables.append(
                TableInfo(
                    name=name,
                    ddl=ddl or "",
                    columns=columns,
                    foreign_keys=fks,
                    referenced_tables=sorted(set(refs)),
                    row_count=row_count,
                    samples=samples,
                )
            )
        return tables
    finally:
        conn.close()
