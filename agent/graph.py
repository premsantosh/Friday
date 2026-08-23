"""
Graph topology. `build_graph` returns an *uncompiled* StateGraph; the engine
compiles it per invocation with a loop-local checkpointer (see checkpoint.py).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import make_nodes
from .state import AgentState
from .tools import ToolSet


def build_graph(deps, tool_set: ToolSet) -> StateGraph:
    nodes = make_nodes(deps, tool_set)
    g = StateGraph(AgentState)
    g.add_node("prepare_context", nodes.prepare_context)
    g.add_node("agent", nodes.agent)
    g.add_node("finalize", nodes.finalize)
    g.add_edge(START, "prepare_context")
    g.add_edge("prepare_context", "agent")

    if nodes.has_tools:
        g.add_node("tools", ToolNode(tool_set.tools))
        g.add_node("overflow", nodes.overflow)
        g.add_conditional_edges("agent", nodes.route_after_agent,
                                {"tools": "tools", "overflow": "overflow", "finalize": "finalize"})
        g.add_conditional_edges("tools", nodes.route_after_tools,
                                {"agent": "agent", "finalize": "finalize"})
        g.add_edge("overflow", "finalize")
    else:
        g.add_edge("agent", "finalize")

    g.add_edge("finalize", END)
    return g
