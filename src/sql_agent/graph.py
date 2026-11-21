"""LangGraph assembly for the text-to-SQL agent."""

from langgraph.graph import END, START, StateGraph

from src.sql_agent.nodes import AgentNodes
from src.sql_agent.state import AgentState


def build_graph(nodes: AgentNodes, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("generate_sql", nodes.generate_sql)
    g.add_node("execute_sql", nodes.execute_sql)
    g.add_node("postprocess", nodes.postprocess)
    g.add_node("summarize", nodes.summarize)
    g.add_node("give_up", nodes.give_up)
    g.add_node("decline", nodes.decline)

    g.add_edge(START, "generate_sql")
    g.add_conditional_edges(
        "generate_sql",
        nodes.route_generate,
        {"execute_sql": "execute_sql", "decline": "decline"},
    )
    g.add_edge("decline", END)
    g.add_conditional_edges(
        "execute_sql",
        nodes.route,
        {"retry": "generate_sql", "postprocess": "postprocess", "give_up": "give_up"},
    )
    g.add_edge("postprocess", "summarize")
    g.add_edge("summarize", END)
    g.add_edge("give_up", END)
    return g.compile(checkpointer=checkpointer)
