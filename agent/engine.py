"""
AgentEngine — the facade VoiceAssistant talks to.

    handle(text, user_id)      one user turn → spoken reply (str)
    has_pending_interrupt()    is a confirmation waiting for this user?
    cancel(user_id)            abandon a pending confirmation (global escape)
    reset(user_id)             "clear history": bump the thread epoch
    run_due_wakes()            Phase 4: service scheduled wake-ups (runner tick)

Thread id scheme: `chat:{user_id}:{epoch}`; task sessions remain the
SessionManager's. Voice/terminal use user_id "default", Telegram uses the chat
id and Voice PE the room name — distinct threads today (documented, not fixed).

handle():
  1. If the thread is paused on a confirmation → this turn is the answer.
     Parsed deterministically (harness parse_confirmation; UNCLEAR re-asks,
     never approves). YES resumes the graph via Command(resume=…) — this is how
     a confirmation survives a restart or a channel switch. NO closes the
     pending tool call without an LLM round-trip. A confirmation older than the
     conversation timeout lapses: closed as declined, the user is told, and the
     turn is processed fresh.
  2. Response-cache fast path (LLM-free); the exchange is appended to the
     thread so history stays coherent.
  3. Otherwise ainvoke. Interrupted → return the question; handoff → return it
     verbatim; else the last AI text.
  4. Post-run: Layer-A context update + intent-cache writeback — only when the
     run resolved via exactly one successful *simple* (ungated) workflow tool
     on un-enriched input, preserving the cache-poisoning guard in
     core/assistant.py. Gated tools are never cached: a cache hit at Step 2
     would execute the workflow directly and bypass the confirmation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from config import AgentConfig
from core.harness import ConfirmDecision, parse_confirmation

from .checkpoint import CheckpointerProvider
from .graph import build_graph
from .nodes import last_ai_text, succeeded, turn_tool_calls
from .store import AgentStore, Wake
from .tools import DECLINED_PREFIX, ToolSet, build_tools
from .tracing import trace_config

logger = logging.getLogger(__name__)

SET_ASIDE_LINE = "Very well, {title}. I've set that aside."


@dataclass
class EngineDeps:
    """Live collaborators (never checkpointed)."""
    llm: Any                      # LLMProvider: personality, store, cache, context_builder, extractor, search_enhancer, stats
    workflows: Any                # WorkflowManager
    model: Any                    # BaseChatModel
    agent_config: AgentConfig
    sessions: Any = None          # SessionManager (legacy multi-turn tasks)
    context: Any = None           # ConversationContext (Layer A)
    intent_cache: Any = None      # IntentCache
    gate: Any = None              # ActionGate for gated tools
    store: Optional[AgentStore] = None   # epochs + wakes
    confirmation_timeout_s: float = 600.0


def _pending_interrupt(state) -> Optional[Dict[str, Any]]:
    """The payload of the first pending interrupt on a thread, or None."""
    if state is None:
        return None
    interrupts = getattr(state, "interrupts", None) or ()
    if not interrupts:
        for task in getattr(state, "tasks", None) or ():
            interrupts = tuple(getattr(task, "interrupts", None) or ())
            if interrupts:
                break
    if not interrupts:
        return None
    value = interrupts[0].value
    return value if isinstance(value, dict) else {"type": "confirmation", "question": str(value)}


class AgentEngine:
    def __init__(self, deps: EngineDeps, *, checkpoints: CheckpointerProvider,
                 tracer=None, tools: Optional[ToolSet] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.deps = deps
        self.store: AgentStore = deps.store or AgentStore()
        deps.store = self.store
        self._checkpoints = checkpoints
        self._tracer = tracer
        self._clock = clock
        self.tool_set = tools or build_tools(
            deps.workflows, sessions=deps.sessions, context=deps.context,
            gate=deps.gate, store=self.store, clock=clock)
        self._graph = build_graph(deps, self.tool_set)

    # ------------------------------------------------------------ factory
    @classmethod
    def from_assistant(cls, assistant) -> "AgentEngine":
        from core.harness import ActionGate, AuditLog

        from .models import build_chat_model
        from .store import SqliteAgentStore
        from .tracing import build_tracer

        cfg = assistant.config
        ephemeral = bool(cfg.llm.ephemeral)
        model = build_chat_model(cfg.llm, cfg.agent)
        checkpoints = CheckpointerProvider(None if ephemeral else cfg.agent.checkpoint_path)
        store = AgentStore() if ephemeral else SqliteAgentStore(cfg.agent.checkpoint_path)
        gate = ActionGate.with_defaults(kill_switch_env=cfg.agent.kill_switch_env,
                                        audit=AuditLog.from_env())
        deps = EngineDeps(
            llm=assistant.llm,
            workflows=assistant.workflows,
            model=model,
            agent_config=cfg.agent,
            sessions=assistant.sessions,
            context=assistant.context,
            intent_cache=assistant.intent_cache,
            gate=gate,
            store=store,
            confirmation_timeout_s=cfg.conversation.default_session_timeout_s,
        )
        return cls(deps, checkpoints=checkpoints, tracer=build_tracer(cfg.agent))

    # ------------------------------------------------------------ helpers
    @property
    def ephemeral(self) -> bool:
        return self._checkpoints.ephemeral

    @property
    def _title(self) -> str:
        return getattr(self.deps.llm.personality, "user_title", "sir")

    def thread_id(self, user_id: str) -> str:
        return f"chat:{user_id}:{self.store.get_epoch(user_id)}"

    def _cfg(self, thread_id: str, *, user_id: str = "", run_kind: str = "") -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if run_kind:
            cfg.update(trace_config(self._tracer, user_id=user_id, thread_id=thread_id,
                                    run_kind=run_kind))
        return cfg

    def describe(self) -> str:
        names = [t.name for t in self.tool_set.tools]
        return (f"langgraph ({'ephemeral' if self.ephemeral else self._checkpoints.path}); "
                f"tools: {', '.join(names) if names else 'none'}")

    # -------------------------------------------------------------- turns
    async def handle(self, text: str, user_id: str = "default", *, cacheable: bool = True) -> str:
        thread_id = self.thread_id(user_id)
        async with self._checkpoints.open() as saver:
            app = self._graph.compile(checkpointer=saver)
            state = await app.aget_state(self._cfg(thread_id))
            prefix = ""

            pending = _pending_interrupt(state)
            if pending is not None:
                if self._expired(pending):
                    await self._close_pending(app, thread_id, state, pending, "let it lapse")
                    prefix = f"That confirmation lapsed, {self._title}, so I set it aside. "
                else:
                    return await self._answer_pending(app, thread_id, state, pending, text, user_id)

            cached = self._cached_response(text)
            if cached is not None:
                self.deps.llm.stats["cache_hits"] += 1
                try:
                    await app.aupdate_state(
                        self._cfg(thread_id),
                        {"messages": [HumanMessage(content=text), AIMessage(content=cached)],
                         "user_id": user_id, "handoff_message": ""},
                        as_node="finalize")
                except Exception:
                    logger.debug("could not append cached exchange to thread", exc_info=True)
                return prefix + cached

            out = await app.ainvoke(
                {"messages": [HumanMessage(content=text)], "user_id": user_id,
                 "handoff_message": "", "tool_iterations": 0, "cacheable": cacheable},
                self._cfg(thread_id, user_id=user_id, run_kind="fresh"))
            reply = self._reply_from(out)
            self._post_run(out, text, cacheable=cacheable)
            return prefix + reply

    async def has_pending_interrupt(self, user_id: str = "default") -> bool:
        thread_id = self.thread_id(user_id)
        async with self._checkpoints.open() as saver:
            app = self._graph.compile(checkpointer=saver)
            state = await app.aget_state(self._cfg(thread_id))
            return _pending_interrupt(state) is not None

    async def cancel(self, user_id: str = "default") -> bool:
        """Abandon a pending confirmation without an LLM round-trip. Returns
        True when there was one."""
        thread_id = self.thread_id(user_id)
        async with self._checkpoints.open() as saver:
            app = self._graph.compile(checkpointer=saver)
            state = await app.aget_state(self._cfg(thread_id))
            pending = _pending_interrupt(state)
            if pending is None:
                return False
            await self._close_pending(app, thread_id, state, pending, "cancelled")
            return True

    def reset(self, user_id: str = "default") -> None:
        """Start a fresh thread for this user (old checkpoints stay inspectable)."""
        self.store.bump_epoch(user_id)

    # ------------------------------------------------------------ wake-ups
    async def run_due_wakes(self) -> List[str]:
        """Service due agent_wakes rows; returns the replies to notify."""
        replies: List[str] = []
        for wake in self.store.due_wakes(self._clock()):
            try:
                reply = await self.wake(wake)
            except Exception:
                logger.exception("agent wake %s failed; dropping it", wake.wake_id)
                self.store.delete_wake(wake.wake_id)
                continue
            if reply is None:
                continue            # postponed: a confirmation is pending on that thread
            self.store.delete_wake(wake.wake_id)
            if reply:
                replies.append(reply)
        return replies

    async def wake(self, wake: Wake) -> Optional[str]:
        """Resume the user's thread with the wake note. None = postponed."""
        thread_id = self.thread_id(wake.user_id)
        async with self._checkpoints.open() as saver:
            app = self._graph.compile(checkpointer=saver)
            state = await app.aget_state(self._cfg(thread_id))
            if _pending_interrupt(state) is not None:
                return None
            note = str(wake.payload.get("note", "")).strip()
            text = ("(Scheduled wake-up you set earlier; the user is not speaking. "
                    f"Do what you planned and report briefly: {note})")
            out = await app.ainvoke(
                {"messages": [HumanMessage(content=text)], "user_id": wake.user_id,
                 "handoff_message": "", "tool_iterations": 0, "cacheable": False},
                self._cfg(thread_id, user_id=wake.user_id, run_kind="wake"))
            reply = self._reply_from(out)
            self._post_run(out, text, cacheable=False)
            return reply

    # ----------------------------------------------------------- internals
    def _expired(self, pending: Dict[str, Any]) -> bool:
        asked_at = pending.get("asked_at")
        if not isinstance(asked_at, (int, float)):
            return False
        return (self._clock() - float(asked_at)) > float(self.deps.confirmation_timeout_s)

    async def _answer_pending(self, app, thread_id: str, state, pending: Dict[str, Any],
                              text: str, user_id: str) -> str:
        question = str(pending.get("question") or "Shall I proceed?")
        decision = parse_confirmation(text)   # strict gate: UNCLEAR never approves
        if decision == ConfirmDecision.YES:
            out = await app.ainvoke(Command(resume={"decision": "yes"}),
                                    self._cfg(thread_id, user_id=user_id, run_kind="resume"))
            reply = self._reply_from(out)
            self._post_run(out, text, cacheable=False)
            return reply
        if decision == ConfirmDecision.NO:
            await self._close_pending(app, thread_id, state, pending, "declined")
            return SET_ASIDE_LINE.format(title=self._title)
        return f"I need a yes or a no, {self._title}. {question}"

    async def _close_pending(self, app, thread_id: str, state, pending: Dict[str, Any],
                             reason: str) -> None:
        """Close the paused tool call with a DECLINED result and a fixed spoken
        line, written as the `tools` and `finalize` nodes so the thread is idle
        again and the history stays valid (no orphaned tool_use)."""
        cfg = self._cfg(thread_id)
        tool_call_ids: List[str] = []
        if pending.get("tool_call_id"):
            tool_call_ids = [str(pending["tool_call_id"])]
        else:
            messages = (state.values or {}).get("messages", []) if state is not None else []
            for m in reversed(messages):
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    tool_call_ids = [c["id"] for c in m.tool_calls]
                    break
        if tool_call_ids:
            await app.aupdate_state(cfg, {"messages": [
                ToolMessage(content=f"{DECLINED_PREFIX} the user {reason}; do not retry.",
                            tool_call_id=tid) for tid in tool_call_ids]}, as_node="tools")
        await app.aupdate_state(cfg, {
            "messages": [AIMessage(content=SET_ASIDE_LINE.format(title=self._title))],
            "handoff_message": "",
        }, as_node="finalize")

    def _cached_response(self, text: str) -> Optional[str]:
        llm = self.deps.llm
        cb = getattr(llm, "context_builder", None)
        cache = getattr(llm, "cache", None)
        if cb is None or cache is None:
            return None
        try:
            return cache.get_cached_response(cb.query_fingerprint(text))
        except Exception:
            logger.debug("response cache lookup failed", exc_info=True)
            return None

    def _reply_from(self, out: Dict[str, Any]) -> str:
        interrupts = out.get("__interrupt__") or ()
        if interrupts:
            value = interrupts[0].value
            if isinstance(value, dict):
                return str(value.get("question") or "Shall I proceed?")
            return str(value)
        if out.get("handoff_message"):
            return str(out["handoff_message"])
        text = last_ai_text(out.get("messages", []))
        return text or f"I'm afraid I have nothing useful to say to that, {self._title}."

    def _post_run(self, out: Dict[str, Any], text: str, *, cacheable: bool) -> None:
        if out.get("__interrupt__") or out.get("handoff_message"):
            return
        calls = turn_tool_calls(out.get("messages", []))
        if not calls:
            return
        ok = [(n, a, r) for n, a, r in calls if n in self.tool_set.simple_names and succeeded(r)]
        if not ok:
            return
        name, args, _ = ok[-1]
        workflow = self.tool_set.workflow_for.get(name)
        entities = dict(args.get("entities") or {})
        if workflow is not None and self.deps.context is not None:
            try:
                self.deps.context.update(workflow.name, entities, text)
            except Exception:
                logger.debug("context update failed", exc_info=True)
        if (cacheable and self.deps.intent_cache is not None and workflow is not None
                and len(calls) == 1 and len(ok) == 1):
            try:
                self.deps.intent_cache.store(text, workflow.name, entities)
            except Exception:
                logger.debug("intent cache writeback failed", exc_info=True)
