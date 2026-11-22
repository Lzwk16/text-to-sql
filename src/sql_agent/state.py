"""Shared state passed between graph nodes."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    db_id: Optional[str]
    question: str
    sql: str
    error: Optional[str]  # guardrail/execution/empty-result feedback for retries
    attempts: int
    columns: List[str]
    rows: List[Dict[str, Any]]
    truncated: bool
    answer: str
    failed: bool
    declined: bool  # model said the request can't be answered read-only
    conversions: List[str]  # converter names enabled for this request
    conversion_options: Dict[str, Dict[str, str]]  # per-converter settings chosen
    conversions_applied: List[Dict[str, str]]  # columns actually converted
    trace: List[str]  # human-readable log of each attempt
