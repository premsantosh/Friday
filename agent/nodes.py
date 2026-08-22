"""
Graph nodes. Closures over `EngineDeps` (live objects never enter state).

    START -> prepare_context -> agent
      agent --(tool_calls & iterations <= cap)--> tools --(handoff_message set)--> finalize
                                                   tools --(else)-----------------> agent
      agent --(tool_calls & over cap)--> overflow -> finalize
      agent --(plain text)-----------------------------------------------------> finalize
    finalize -> END

prepare_context  deterministic: sarcasm-command check (shared LLMProvider, so
                 personality sliders stay unified across engines), memory
                 context via ContextBuilder, search context via SearchEnhancer
                 — replicating LLMProvider._prepare_request.
agent            model.bind_tools(tools).ainvoke([system] + messages). The system
                 prompt is rebuilt every turn (date/time stay accurate, mirrors
                 _refresh_system_prompt) and is never stored in state.
tools            langgraph ToolNode (async).
overflow         closes the orphaned tool calls with ERROR results + apologises,
                 so history stays valid for the next turn.
finalize         mirrors LLMProvider._record_exchange (log_turn, background
                 fact extraction, confidence feedback, response-cache write for
                 tool-free turns) then pair-aware history trimming.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from llm.providers import generate_personality_prompt

from .tracing import CONTEXT_CLOSE, CONTEXT_OPEN
from .tools import FAILURE_PREFIXES, START_TASK, ToolSet

logger = logging.getLogger(__name__)

TOOL_GUIDANCE = """TOOLS:
- You act on the home and other capabilities ONLY through the provided tools. Call a tool whenever the request maps to one; never claim an action was done unless a tool result says so.
- A tool result starting with "ERROR:" means the action failed: say briefly what failed, in character.
- A result starting with "DECLINED:" or "REFUSED:" means the action was not carried out: acknowledge and stop. Do not retry, do not ask again.
- Chain tool calls when a request needs several steps (look something up, then act on it). One tool call at a time.
- `start_task` hands the conversation to a multi-turn task; after calling it, stop.
- Answer general questions and small talk directly, without tools. Keep spoken replies short."""


# ------------------------------------------------------------ message utils

def message_text(msg: Any) -> str:
    """Plain text of a message (Anthropic replies may be a list of content blocks)."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content) if content is not None else ""


def last_human_text(messages: Sequence[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return message_text(m)
    return ""


def last_ai_text(messages: Sequence[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            text = message_text(m)
            if text:
                return text
            if not getattr(m, "tool_calls", None):
                return ""
        if isinstance(m, HumanMessage):
            break
    return ""


def turn_tool_calls(messages: Sequence[AnyMessage]) -> List[Tuple[str, Dict[str, Any], str]]:
    """(tool name, args, result text) for every tool call since the last HumanMessage."""
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break
    calls: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    order: List[str] = []
    results: Dict[str, str] = {}
    for m in messages[start:]:
        if isinstance(m, AIMessage):
            for c in getattr(m, "tool_calls", None) or []:
                calls[c["id"]] = (c["name"], dict(c.get("args") or {}))
                order.append(c["id"])
        elif isinstance(m, ToolMessage):
            results[m.tool_call_id] = message_text(m)
    return [(calls[i][0], calls[i][1], results.get(i, "")) for i in order if i in calls]


def succeeded(result_text: str) -> bool:
    return bool(result_text) and not result_text.lstrip().startswith(FAILURE_PREFIXES)


def trim_plan(messages: Sequence[AnyMessage], max_messages: int) -> List[RemoveMessage]:
    """RemoveMessages that drop the oldest history while keeping whole turns:
    the cut always lands on a HumanMessage, so an AI tool_call is never
    separated from its ToolMessages (Anthropic rejects orphaned tool blocks)."""
    if max_messages <= 0 or len(messages) <= max_messages:
        return []
    cut = len(messages) - max_messages
    while cut < len(messages) and not isinstance(messages[cut], HumanMessage):
        cut += 1
    if cut >= len(messages):
        # Window holds no full turn boundary: keep from the last HumanMessage.
        cut = 0
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                cut = i
                break
    return [RemoveMessage(id=m.id) for m in messages[:cut] if getattr(m, "id", None)]


# ------------------------------------------------------------------- nodes

def build_system_prompt(personality, context_block: str, has_tools: bool) -> str:
    parts = [generate_personality_prompt(personality)]
    if has_tools:
        parts.append(TOOL_GUIDANCE)
    if context_block:
        parts.append(f"{CONTEXT_OPEN}\n{context_block}\n{CONTEXT_CLOSE}")
    return "\n\n".join(parts)


def make_nodes(deps, tool_set: ToolSet) -> SimpleNamespace:
    llm = deps.llm
    cfg = deps.agent_config
    has_tools = bool(tool_set.tools)

    model = deps.model
    if has_tools:
        try:
            model = deps.model.bind_tools(tool_set.tools, parallel_tool_calls=False)
        except TypeError:
            model = deps.model.bind_tools(tool_set.tools)

    async def _search_context(text: str) -> str:
        enhancer = getattr(llm, "search_enhancer", None)
        if enhancer is None:
            return ""
        if llm.context_builder is None:
            found = await asyncio.to_thread(enhancer.enhance, text)
            if found:
                llm.stats["search_queries"] += 1
            return found or ""
        fp = llm.context_builder.query_fingerprint(text)
        found = llm.cache.get_cached_search(fp)
        if found is None:
            found = await asyncio.to_thread(enhancer.enhance, text)
            if found:
                llm.cache.cache_search(fp, found)
                llm.stats["search_queries"] += 1
        return found or ""

    async def prepare_context(state) -> Dict[str, Any]:
        text = last_human_text(state["messages"])
        llm._check_sarcasm_command(text)
        llm.stats["total_requests"] += 1
        parts: List[str] = []
        pending_keys: List[str] = []
        if llm.context_builder is not None:
            ctx = llm.context_builder.build_context(text)
            pending_keys = list(ctx.get("retrieved_fact_keys") or [])
            formatted = llm.context_builder.format_system_prompt("", ctx).strip()
            if formatted:
                parts.append(formatted)
        try:
            search = await _search_context(text)
        except Exception:
            logger.warning("search enhancer failed; continuing without it", exc_info=True)
            search = ""
        if search:
            parts.append(search)
        return {"context_block": "\n".join(parts), "pending_fact_keys": pending_keys}

    async def agent(state) -> Dict[str, Any]:
        system = SystemMessage(content=build_system_prompt(
            llm.personality, state.get("context_block", ""), has_tools))
        response = await model.ainvoke([system, *state["messages"]])
        llm.stats["llm_calls"] += 1
        update: Dict[str, Any] = {"messages": [response]}
        if getattr(response, "tool_calls", None):
            update["tool_iterations"] = int(state.get("tool_iterations", 0)) + 1
        return update

    def route_after_agent(state) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "finalize"
        if int(state.get("tool_iterations", 0)) > cfg.max_tool_iterations:
            return "overflow"
        return "tools"

    def route_after_tools(state) -> str:
        return "finalize" if state.get("handoff_message") else "agent"

    async def overflow(state) -> Dict[str, Any]:
        last = state["messages"][-1]
        title = llm.personality.user_title
        msgs: List[AnyMessage] = [
            ToolMessage(content="ERROR: tool budget for this request is exhausted; stop.",
                        tool_call_id=c["id"])
            for c in (getattr(last, "tool_calls", None) or [])
        ]
        msgs.append(AIMessage(content=(
            f"I'm afraid that took more steps than I am permitted in one go, {title}. "
            f"Might I suggest breaking it into smaller requests?")))
        return {"messages": msgs}

    async def finalize(state) -> Dict[str, Any]:
        messages = state["messages"]
        update: Dict[str, Any] = {}
        handoff = state.get("handoff_message")
        text = last_human_text(messages)
        reply = last_ai_text(messages)

        if llm.store is not None and not handoff and text:
            # Confidence feedback: this turn's text confirms/corrects last turn's facts.
            prev_keys = state.get("retrieved_fact_keys") or []
            if prev_keys:
                if llm.extractor.is_correction(text):
                    for key in prev_keys:
                        llm.store.drop_confidence(key)
                else:
                    for key in prev_keys:
                        llm.store.bump_confidence(key)
            llm.store.log_turn("user", text)
            if reply:
                llm.store.log_turn("assistant", reply)
                used_tools = bool(turn_tool_calls(messages))
                if not used_tools and llm.context_builder is not None:
                    llm.cache.cache_response(llm.context_builder.query_fingerprint(text), reply)
                threading.Thread(target=llm._extract_and_store, args=(text, reply),
                                 daemon=True).start()
            update["retrieved_fact_keys"] = list(state.get("pending_fact_keys") or [])
            update["pending_fact_keys"] = []

        removals = trim_plan(messages, cfg.history_max_messages)
        if removals:
            update["messages"] = removals
        return update

    return SimpleNamespace(
        prepare_context=prepare_context,
        agent=agent,
        overflow=overflow,
        finalize=finalize,
        route_after_agent=route_after_agent,
        route_after_tools=route_after_tools,
        has_tools=has_tools,
    )
