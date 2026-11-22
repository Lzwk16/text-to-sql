"""Graph nodes: generate -> validate+execute -> (retry | summarize)."""

import re
from typing import Any, Dict, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from utils.db_registry import DatabaseRegistry
from src.sql_agent import prompts
from src.sql_agent.executor import execute_query
from src.sql_agent.guardrails import GuardrailError, enforce_sql_guardrails
from src.sql_agent.postprocess import apply_converters, build_converters
from src.sql_agent.retriever import SchemaProvider
from src.sql_agent.state import AgentState
from utils.helpers import clean_sql_output

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
DECLINE_TOKEN = "CANNOT_ANSWER"
DECLINE_MESSAGE = (
    "I can only answer read-only questions about the data in this database, "
    "so I can't help with that request. Try asking about the contents of the tables."
)


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
        self.timeout_s = cfg.get("query_timeout_s", 10)
        self.prompt_template = prompt_template or prompts.load("generate_sql")
        self.summary_rows = summary_rows

    # -- nodes -----------------------------------------------------------
    async def generate_sql(self, state: AgentState) -> dict:
        question = state.get("question") or state["messages"][-1].content
        db_path = str(self.registry.resolve(state.get("db_id")))
        db_schema = self.schema.schema_for(
            state.get("db_id") or self.registry.default_id, db_path, question
        )

        feedback = ""
        if state.get("error"):
            feedback = prompts.render(
                "retry_feedback", sql=state.get("sql", ""), error=state["error"]
            )
        key = state.get("db_id") or self.registry.default_id

        async def ask(schema_text: str) -> str:
            prompt = self.prompt_template.format(
                question=question, db_schema=schema_text, feedback=feedback
            )
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return _THINK_RE.sub("", str(response.content))

        text = await ask(db_schema)
        if DECLINE_TOKEN in text.upper() and self.schema.is_partial(key, db_path):
            # Retrieval may have hidden the needed tables; retry once with all
            # of them (new information) before telling the user we can't help.
            full = self.schema.full_schema(key, db_path)
            if full is not None:
                text = await ask(full)
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
            df = execute_query(db_path, safe_sql, self.timeout_s)
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
        enabled = state.get("conversions") or []
        converters = build_converters(self.postprocess_cfg, enabled)
        if not converters or not state.get("rows"):
            return {"conversions_applied": []}
        df = pd.DataFrame(state["rows"], columns=state.get("columns"))
        out, applied = apply_converters(df, converters)
        if not applied:
            return {"conversions_applied": []}
        clean = out.astype(object).where(out.notna(), None)
        return {
            "rows": clean.to_dict("records"),
            "conversions_applied": [a.as_dict() for a in applied],
        }

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
            shown=self.summary_rows,
            sample=sample,
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        answer = str(response.content).strip()
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    async def give_up(self, state: AgentState) -> dict:
        answer = (
            f"I couldn't produce a working query after {state.get('attempts', 0)} "
            f"attempts. Last problem: {state.get('error')}"
        )
        return {"answer": answer, "failed": True, "messages": [AIMessage(content=answer)]}

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
