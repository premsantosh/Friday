"""
Agent graph state.

A closed TypedDict: every value is JSON/msgpack-serializable so the whole state
round-trips through the checkpointer. Live objects (browser pages, device
controllers, the LLM client) never enter state; they live in `EngineDeps` and
are closed over by the node functions.
"""

from __future__ import annotations

from typing import Annotated, List

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict):
    # Chat history for this thread (`chat:{user_id}:{epoch}`). The system
    # prompt is built fresh each turn and prepended at model-call time; it is
    # never stored here, so it can't accumulate in checkpoints.
    messages: Annotated[List[AnyMessage], add_messages]
    user_id: str
    # Per-turn memory/search context (from ContextBuilder + SearchEnhancer);
    # hidden from traces, see agent/tracing.py.
    context_block: str
    # Set by the `start_task` handoff tool: the legacy session's opening line,
    # returned to the user verbatim (slot prompts / confirmation summaries must
    # reach the user exactly as written — ActionGate approval hashing depends
    # on what was shown).
    handoff_message: str
    # Tool round-trips taken this turn; capped by AgentConfig.max_tool_iterations.
    tool_iterations: int
    # Memory-confidence feedback (mirrors LLMProvider._pending/_last_retrieved_fact_keys):
    # keys retrieved for *this* turn, and those retrieved for the previous turn,
    # which this turn's user text either confirms or corrects.
    pending_fact_keys: NotRequired[List[str]]
    retrieved_fact_keys: NotRequired[List[str]]
    # Whether the intent cache may learn from this turn (raw, un-enriched input).
    cacheable: NotRequired[bool]
