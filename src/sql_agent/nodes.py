"""Graph nodes: generate -> validate+execute -> (retry | summarize)."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import sqlglot
from sqlglot import exp
from langchain_core.messages import AIMessage, HumanMessage

from utils.db_registry import DatabaseRegistry
from src.sql_agent import prompts
from src.sql_agent.executor import execute_query
from src.sql_agent.guardrails import GuardrailError, enforce_sql_guardrails
from src.sql_agent.postprocess import (
    apply_converters,
    build_all,
    build_converters,
    resolve_names,
)
from src.sql_agent.postprocess.base import Applied, Converter, Detected
from src.sql_agent.retriever import SchemaProvider
from src.sql_agent.state import AgentState
from utils.helpers import clean_sql_output

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
DECLINE_TOKEN = "CANNOT_ANSWER"
DECLINE_MESSAGE = (
    "I can only answer read-only questions about the data in this database, "
    "so I can't help with that request. Try asking about the contents of the tables."
)
# Per-turn limits for conversation history (characters).
MAX_HISTORY_SQL = 2000
MAX_HISTORY_QUESTION = 500
# On partial schemas, tables used by this many recent turns are added to the
# prompt, up to this many characters of schema text.
HISTORY_TABLE_TURNS = 2
MAX_HISTORY_SCHEMA = 10000


class AgentNodes:
    def __init__(
        self,
        llm,
        registry: DatabaseRegistry,
        schema_provider: SchemaProvider,
        guardrails_cfg: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        summary_rows: int = 20,
        postprocess_cfg: Optional[Dict[str, Any]] = None,
        agent_loop_cfg: Optional[Dict[str, Any]] = None,
    ):
        cfg = guardrails_cfg or {}
        loop = agent_loop_cfg or {}
        self.postprocess_cfg = postprocess_cfg or {}
        self.llm = llm
        self.registry = registry
        self.schema = schema_provider
        self.max_rows = cfg.get("max_rows", 500)
        self.max_attempts = loop.get("max_attempts", 3)
        self.history_turns = int(loop.get("history_turns", 5))
        self.timeout_s = cfg.get("query_timeout_s", 10)
        self.prompt_template = prompt_template or prompts.load("generate_sql")
        self.summary_rows = summary_rows

    # -- conversions -----------------------------------------------------
    def _active(self, state: AgentState) -> Tuple[List[Converter], List[Detected]]:
        """Converters enabled for this request and the DB columns they claim.

        ``auto`` expands to the converters that detected a column in this database.
        Detection is scanned once per database and cached by the schema provider.
        """
        requested = state.get("conversions") or []
        if not requested:
            return [], []
        key = state.get("db_id") or self.registry.default_id
        path = str(self.registry.resolve(state.get("db_id")))
        options = state.get("conversion_options") or {}
        detected = self.schema.convertible(
            key, path, build_all(self.postprocess_cfg, options)
        )
        names = resolve_names(requested, {d.converter for d in detected})
        converters = build_converters(self.postprocess_cfg, names, options)
        active = {c.name for c in converters}
        return converters, [d for d in detected if d.converter in active]

    def _hints(
        self, converters: List[Converter], detected: List[Detected], tables: set
    ) -> str:
        """Prompt text naming SQL conversion functions for detected columns."""
        by_name = {c.name: c for c in converters}
        show_example = self.schema.sample_limit > 0  # same privacy rule as samples
        lines = [
            "- " + by_name[d.converter].hint(
                d.table, d.column, d.example if show_example else None
            )
            for d in detected
            if d.table in tables and d.converter in by_name
        ]
        if not lines:
            return ""
        return prompts.render("conversion_hints", lines="\n".join(lines))

    @staticmethod
    def _raw_usage(
        sql: str, detected: List[Detected], converters: List[Converter]
    ) -> str:
        """Error text if SQL aggregates, sorts or compares a detected column raw.

        MIN/MAX/ORDER BY on text like '31/08/2023 9:58' compares characters, not
        dates, and the answer looks plausible, so this is sent back for a retry.
        """
        if not detected:
            return ""
        by_col = {d.column.lower(): d for d in detected}
        names = {c.name: c.sql_name for c in converters}
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.SqlglotError:
            return ""
        raw = []
        for node in tree.find_all(exp.Min, exp.Max, exp.Avg, exp.Sum, exp.Ordered,
                                  exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between):
            operands = (
                [node.this]
                if isinstance(
                    node,
                    (exp.Min, exp.Max, exp.Avg, exp.Sum, exp.Ordered, exp.Between),
                )
                else [node.this, node.expression]
            )
            for op in operands:
                if isinstance(op, exp.Column) and op.name.lower() in by_col:
                    raw.append(by_col[op.name.lower()])
        if not raw:
            return ""
        parts = [
            f"'{d.column}' (table {d.table}) holds text; use "
            f"{names[d.converter]}({d.column}) instead of the raw column"
            for d in {id(d): d for d in raw}.values()
            if d.converter in names
        ]
        return (
            "Do not use these columns raw in MIN, MAX, AVG, SUM, ORDER BY or "
            "comparisons: " + "; ".join(parts) + "."
        ) if parts else ""

    @staticmethod
    def _sql_conversions(
        sql: str, converters: List[Converter], detected: Optional[List[Detected]] = None
    ) -> List[Applied]:
        """Conversion functions the model used inside its SQL (they run in SQLite)."""
        by_sql = {c.sql_name: c for c in converters}
        examples = {d.column.lower(): d.example for d in (detected or [])}
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.SqlglotError:
            return []
        out, seen = [], set()
        for call in tree.find_all(exp.Anonymous):
            conv = by_sql.get(str(call.name).lower())
            if conv is None or not call.expressions:
                continue
            arg = call.expressions[0]
            column = arg.sql(dialect="sqlite")
            sample = (
                examples.get(arg.name.lower()) if isinstance(arg, exp.Column) else None
            )
            if (column, conv.name) not in seen:
                seen.add((column, conv.name))
                out.append(
                    Applied(
                        column.strip('"'),
                        conv.name,
                        f"{conv.describe(sample)} (in SQL)",
                    )
                )
        return out

    # -- conversation history ---------------------------------------------
    def _db_identity(self, state: AgentState) -> Tuple[Optional[str], str]:
        """(db key, file path) naming the request's database.

        ``None`` and ``"default"`` are the same database, and the resolved path
        guards against ``default_db`` pointing at a different file after a
        restart (a relative or absolute spelling of the same file still matches).
        """
        key = state.get("db_id") or self.registry.default_id
        return key, str(Path(self.registry.resolve(state.get("db_id"))).resolve())

    def _history_turns(self, state: AgentState) -> List[Dict[str, Any]]:
        """Last few successful turns on this database, oldest first."""
        if self.history_turns <= 0:
            return []
        key, path = self._db_identity(state)
        same_db = [
            h
            for h in (state.get("history") or [])
            if h.get("db_id") == key and h.get("db_path") == path
        ]
        return same_db[-self.history_turns :]

    @classmethod
    def _history_block(
        cls, turns: List[Dict[str, Any]], available: Optional[set] = None
    ) -> str:
        """Prompt text listing earlier turns, for follow-up questions.

        ``available`` holds the ``convert_*`` SQL functions registered this turn;
        earlier SQL calling any other one would fail with "no such function".
        """
        if not turns:
            return ""
        lines = []
        for i, h in enumerate(turns, 1):
            cols = ", ".join(h.get("columns") or [])
            sql = h.get("sql") or "(not shown: too long to repeat)"
            count = h.get("row_count", 0)
            if h.get("truncated"):
                # Reaching the cap may also be a complete result of that size.
                count = f"{count} rows (the row cap, so there may be more)"
            else:
                count = f"{count} rows"
            line = (
                f"{i}. Question: {h['question']}\n   SQL: {sql}\n"
                f"   Returned {count} with columns: {cols}"
            )
            missing = sorted(cls._convert_calls(h.get("sql")) - (available or set()))
            if missing:
                line += (
                    f"\n   Note: {', '.join(missing)} is not available now; "
                    "use the raw column instead."
                )
            lines.append(line)
        return prompts.render("history", turns="\n".join(lines))

    @staticmethod
    def _convert_calls(sql: Optional[str]) -> set:
        """Lower-cased ``convert_*`` function names a query calls."""
        if not sql or "convert_" not in sql.lower():
            return set()
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.SqlglotError:
            return set()
        if tree is None:
            return set()
        names = (str(f.name).lower() for f in tree.find_all(exp.Anonymous))
        return {n for n in names if n.startswith("convert_")}

    @staticmethod
    def _sql_tables(sql: str) -> List[str]:
        """Lower-cased table names a query reads (CTE names included)."""
        try:
            tree = sqlglot.parse_one(sql or "", read="sqlite")
        except sqlglot.errors.SqlglotError:
            return []
        if tree is None:
            return []
        return list(dict.fromkeys(t.name.lower() for t in tree.find_all(exp.Table)))

    def _with_history_tables(
        self, shown: list, turns: List[Dict[str, Any]], all_tables: list
    ) -> list:
        """``shown`` plus the tables the most recent turns used, size-capped.

        Retrieval sees only the new text ("now only Paris"), so a follow-up would
        otherwise lose the tables its earlier SQL used. Newest turn first, so the
        cap keeps the tables most likely referred to.
        """
        by_name = {t.name.lower(): t for t in all_tables}
        have = {t.name.lower() for t in shown}
        added, size = [], 0
        for h in reversed(turns[-HISTORY_TABLE_TURNS:]):
            # Saved at the end of the turn; older entries only have the SQL.
            for name in h.get("tables") or self._sql_tables(h.get("sql", "")):
                table = by_name.get(name)
                if table is None or name in have:
                    continue  # CTE alias or already shown
                cost = len(table.render()) + 2
                if size + cost > MAX_HISTORY_SCHEMA:
                    continue  # too big; a smaller table may still fit
                size += cost
                added.append(table)
                have.add(name)
        return shown + added

    # -- nodes -----------------------------------------------------------
    async def generate_sql(self, state: AgentState) -> dict:
        question = state.get("question") or state["messages"][-1].content
        db_path = str(self.registry.resolve(state.get("db_id")))
        key = state.get("db_id") or self.registry.default_id
        shown = self.schema.select_tables(key, db_path, question)
        turns = self._history_turns(state)
        if turns and self.schema.is_partial(key, db_path):
            shown = self._with_history_tables(
                shown, turns, self.schema.tables(key, db_path)
            )
        db_schema = "\n\n".join(t.render() for t in shown)
        converters, detected = self._active(state)
        hints = self._hints(converters, detected, {t.name for t in shown})
        history = self._history_block(turns, {c.sql_name for c in converters})

        feedback = ""
        if state.get("error"):
            feedback = prompts.render(
                "retry_feedback", sql=state.get("sql", ""), error=state["error"]
            )

        async def ask(schema_text: str, conversion_hints: str) -> str:
            prompt = self.prompt_template.format(
                question=question,
                db_schema=schema_text,
                feedback=feedback,
                conversion_hints=conversion_hints,
                history=history,
            )
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return _THINK_RE.sub("", str(response.content))

        text = await ask(db_schema, hints)
        if DECLINE_TOKEN in text.upper() and self.schema.is_partial(key, db_path):
            # Retrieval may have hidden the needed tables; retry once with all
            # of them (new information) before telling the user we can't help.
            full = self.schema.full_schema(key, db_path)
            if full is not None:
                all_tables = {t.name for t in self.schema.tables(key, db_path)}
                text = await ask(full, self._hints(converters, detected, all_tables))
        if DECLINE_TOKEN in text.upper():
            return {"question": question, "sql": "", "declined": True}
        return {"question": question, "sql": clean_sql_output(text), "declined": False}

    async def execute_sql(self, state: AgentState) -> dict:
        attempts = state.get("attempts", 0) + 1
        trace = list(state.get("trace", []))
        sql = state.get("sql", "")
        db_path = str(self.registry.resolve(state.get("db_id")))

        def fail(msg: str) -> dict:
            trace.append(f"Attempt {attempts}: {msg}\n{sql}")
            return {"attempts": attempts, "error": msg, "trace": trace}

        try:
            safe_sql = enforce_sql_guardrails(sql, self.max_rows)
        except GuardrailError as exc:
            return fail(f"Guardrail: {exc}")

        try:
            converters, detected = self._active(state)
        except Exception as exc:
            return fail(f"Execution error: {exc}")
        raw_problem = self._raw_usage(safe_sql, detected, converters)
        if raw_problem:
            return fail(raw_problem)

        try:
            functions = {c.sql_name: c.sql_function() for c in converters}
            df = execute_query(db_path, safe_sql, self.timeout_s, functions)
        except Exception as exc:  # sqlite3.Error, pandas wrapper errors, interrupts
            return fail(f"Execution error: {exc}")

        if df.empty:
            return fail(
                "The query ran but returned no rows. Check filter values, "
                "joins and case sensitivity."
            )

        trace.append(f"Attempt {attempts}: ok ({len(df)} rows)\n{safe_sql}")
        clean = df.astype(object).where(df.notna(), None)
        return {
            "attempts": attempts,
            "error": None,
            "sql": safe_sql,
            "model_sql": sql,  # without the injected LIMIT, for follow-ups
            "columns": list(df.columns),
            "rows": clean.to_dict("records"),
            "truncated": len(df) >= self.max_rows,
            "trace": trace,
        }

    async def postprocess(self, state: AgentState) -> dict:
        """Optional, per-request conversion of numbers-as-text in the result rows.

        Runs before ``summarize`` so the written answer and the table agree. Each
        enabled converter only touches columns it detects, so databases without such
        data are unaffected. The database is never written.
        """
        converters, detected = self._active(state)
        if not converters or not state.get("rows"):
            return {"conversions_applied": []}
        in_sql = self._sql_conversions(state.get("sql", ""), converters, detected)
        df = pd.DataFrame(state["rows"], columns=state.get("columns"))
        out, applied = apply_converters(df, converters)
        everything = in_sql + applied
        if not everything:
            return {"conversions_applied": []}
        update: dict = {"conversions_applied": [a.as_dict() for a in everything]}
        if applied:
            clean = out.astype(object).where(out.notna(), None)
            update["rows"] = clean.to_dict("records")
        return update

    async def summarize(self, state: AgentState) -> dict:
        rows = state.get("rows", [])
        sample = pd.DataFrame(rows[: self.summary_rows]).to_string(index=False)
        note = ""
        applied = state.get("conversions_applied") or []
        if applied:
            parts = "; ".join(
                f"column '{a['column']}' was converted {a['detail']} ({a['converter']})"
                for a in applied
            )
            note = prompts.render("conversion_note", parts=parts)
        prompt = prompts.render(
            "summarize",
            note=note,
            question=state["question"],
            row_count=len(rows),
            # Without this the model reports the capped count as the total.
            capped=(
                prompts.render("row_cap_note", row_count=len(rows))
                if state.get("truncated")
                else ""
            ),
            shown=self.summary_rows,
            sample=sample,
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        answer = str(response.content).strip()
        update: dict = {"answer": answer, "messages": [AIMessage(content=answer)]}
        identity = None
        if self.history_turns > 0:
            try:
                identity = self._db_identity(state)
            except KeyError:  # upload cleaned up after the query ran: still answer
                identity = None
        if identity is not None:
            # Only successful turns reach summarize. Keep a bounded tail so the saved
            # state does not grow without limit; all databases share one list, and
            # _history_turns filters by database.
            key, path = identity
            question = state["question"]
            if len(question) > MAX_HISTORY_QUESTION:
                question = question[:MAX_HISTORY_QUESTION].rstrip() + " ..."
            # Save the model's own SQL: the guardrail's LIMIT would cap follow-ups
            # that build on it. Too-long SQL is dropped, never cut mid-statement.
            sql = state.get("model_sql") or state.get("sql", "")
            entry = {
                "db_id": key,
                "db_path": path,
                "question": question,
                "sql": sql if len(sql) <= MAX_HISTORY_SQL else "",
                # Kept even when the SQL text is dropped, for follow-ups.
                "tables": self._sql_tables(sql),
                "columns": [str(c) for c in state.get("columns", [])],
                "row_count": len(rows),
                "truncated": bool(state.get("truncated")),
            }
            update["history"] = [*(state.get("history") or []), entry][-50:]
        return update

    async def give_up(self, state: AgentState) -> dict:
        answer = (
            f"I couldn't produce a working query after {state.get('attempts', 0)} "
            f"attempts. Last problem: {state.get('error')}"
        )
        return {
            "answer": answer,
            "failed": True,
            "messages": [AIMessage(content=answer)],
        }

    async def decline(self, state: AgentState) -> dict:
        """Request can't be answered read-only: no query, no rows, no retry."""
        return {
            "answer": DECLINE_MESSAGE,
            "declined": True,
            "error": None,
            "rows": [],
            "columns": [],
            "messages": [AIMessage(content=DECLINE_MESSAGE)],
        }

    # -- routing ---------------------------------------------------------
    def route_generate(self, state: AgentState) -> str:
        return "decline" if state.get("declined") else "execute_sql"

    def route(self, state: AgentState) -> str:
        if not state.get("error"):
            return "postprocess"
        if state.get("attempts", 0) < self.max_attempts:
            return "retry"
        return "give_up"
