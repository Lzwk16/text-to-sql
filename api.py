"""FastAPI backend: database upload and chat over the LangGraph agent."""

import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from utils.db_registry import DatabaseRegistry, InvalidDatabase
from src.sql_agent.graph import build_graph
from utils.llm import build_llm
from src.sql_agent.nodes import AgentNodes
from src.sql_agent.postprocess import (
    UnknownConverter,
    available,
    build_all,
    validate_names,
    validate_options,
)
from src.sql_agent.retriever import SchemaProvider

load_dotenv()


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    db_id: Optional[str] = None
    # Converter names to apply; None -> postprocess.default_enabled from config
    conversions: Optional[List[str]] = None
    # Per-converter settings, e.g. {"units": {"length": "mi"}}; see GET /conversions
    conversion_options: Optional[Dict[str, Dict[str, str]]] = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    sql: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    truncated: bool
    failed: bool
    declined: bool = False
    conversions_applied: List[Dict[str, str]] = []
    attempts: int
    trace: List[str]


def load_config(path: str = "utils/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def cleanup_loop(registry: DatabaseRegistry, interval_s: float) -> None:
    """Delete expired uploads on a schedule, not only when someone uploads."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            registry.cleanup_expired()
        except Exception:  # never let a cleanup error kill the loop
            pass


def create_app(config_path: str = "utils/config.yaml", llm=None) -> FastAPI:
    """App factory. Pass ``llm`` to inject a fake model in tests."""
    cfg = load_config(config_path)
    pp_cfg = cfg.get("postprocess", {})
    pp_defaults = validate_names(pp_cfg.get("default_enabled", []))  # fail fast
    if "max_attempts" in cfg.get("guardrails", {}):
        raise ValueError(
            "max_attempts moved from 'guardrails' to 'agent_loop' in the config."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        up = cfg.get("uploads", {})
        registry = DatabaseRegistry(
            upload_dir=up.get("dir", "temp_uploads"),
            max_bytes=up.get("max_mb", 200) * 1024 * 1024,
            ttl_s=int(up.get("ttl_hours", 24) * 3600),
            default_db=cfg.get("default_db"),
        )
        r = cfg.get("retrieval", {})
        provider = SchemaProvider(**r)
        registry.on_remove = provider.forget
        nodes = AgentNodes(
            llm or build_llm(cfg["llm"]),
            registry,
            provider,
            cfg.get("guardrails"),
            postprocess_cfg=pp_cfg,
            agent_loop_cfg=cfg.get("agent_loop"),
        )
        ckpt = Path(cfg.get("checkpoint_db", "data/checkpoints.db"))
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
            app.state.registry = registry
            app.state.provider = provider
            app.state.graph = build_graph(nodes, saver)
            task = asyncio.create_task(
                cleanup_loop(registry, up.get("cleanup_interval_min", 30) * 60)
            )
            try:
                yield
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Text-to-SQL Agent API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/conversions")
    async def conversions():
        return {"available": available(pp_cfg), "default": pp_defaults}

    @app.post("/upload-db")
    async def upload_db(file: UploadFile = File(...)):
        registry: DatabaseRegistry = app.state.registry
        content = await file.read(registry.max_bytes + 1)
        try:
            db_id = registry.add_upload(file.filename or "", content)
        except InvalidDatabase as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        registry.cleanup_expired()
        return {"db_id": db_id}

    @app.get("/schema")
    async def schema(db_id: Optional[str] = None):
        try:
            path = app.state.registry.resolve(db_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        key = db_id or app.state.registry.default_id
        tables = app.state.provider.tables(key, str(path))
        convertible = app.state.provider.convertible(
            key, str(path), build_all(pp_cfg)
        )
        return {
            "tables": [
                {"name": t.name, "columns": t.columns, "rows": t.row_count}
                for t in tables
            ],
            "convertible": [d.as_dict() for d in convertible],
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        try:
            app.state.registry.resolve(req.db_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        thread_id = req.thread_id or f"session_{uuid.uuid4().hex[:8]}"
        try:
            conversions = (
                pp_defaults
                if req.conversions is None
                else validate_names(req.conversions)
            )
            options = validate_options(pp_cfg, req.conversion_options)
        except UnknownConverter as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        # Reset per-turn fields: the checkpointer persists state across turns.
        inputs = {
            "messages": [HumanMessage(content=req.message)],
            "question": req.message,
            "db_id": req.db_id,
            "sql": "",
            "error": None,
            "attempts": 0,
            "rows": [],
            "columns": [],
            "truncated": False,
            "failed": False,
            "declined": False,
            "conversions": conversions,
            "conversion_options": options,
            "conversions_applied": [],
            "answer": "",
            "trace": [],
        }
        try:
            final = await app.state.graph.ainvoke(
                inputs, config={"configurable": {"thread_id": thread_id}}
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}")

        rows, columns = final.get("rows", []), final.get("columns", [])
        return ChatResponse(
            thread_id=thread_id,
            answer=final.get("answer", ""),
            sql=final.get("sql", ""),
            columns=columns,
            rows=rows,
            truncated=final.get("truncated", False),
            failed=final.get("failed", False),
            declined=final.get("declined", False),
            conversions_applied=final.get("conversions_applied", []),
            attempts=final.get("attempts", 0),
            trace=final.get("trace", []),
        )

    return app


app = create_app()
