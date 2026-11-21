"""SQL safety checks: single read-only SELECT statement with a capped LIMIT."""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


class GuardrailError(Exception):
    """Raised when generated SQL is not allowed to run."""


def enforce_sql_guardrails(sql: str, max_rows: int = 500) -> str:
    """Validate ``sql`` and return a safe, LIMIT-capped SQLite statement.

    Raises GuardrailError for parse failures, multiple statements, or anything
    that is not a plain SELECT (including CTEs that wrap writes).
    """
    if not sql or not sql.strip():
        raise GuardrailError("The model returned an empty query.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as exc:
        raise GuardrailError(f"SQL could not be parsed: {exc}") from exc

    if len(statements) != 1:
        raise GuardrailError("Exactly one SQL statement is allowed.")

    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        raise GuardrailError(
            f"Query type '{type(stmt).__name__}' is blocked. "
            "Only read-only SELECT queries are permitted."
        )

    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Command)
    if stmt.find(*forbidden) is not None:
        raise GuardrailError("Data-modifying statements are not permitted.")

    limit = stmt.args.get("limit")
    if limit is None:
        stmt = stmt.limit(max_rows)
    else:
        try:
            current = int(limit.expression.name)
        except (ValueError, AttributeError):
            current = None
        if current is None or current > max_rows:
            stmt = stmt.limit(max_rows)

    return stmt.sql(dialect="sqlite")
