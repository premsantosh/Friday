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

Failure containment: if a run raises *after* a tool already executed, the
engine does not re-raise (the legacy fallback would re-execute the request);
it closes any orphaned tool call, records an apology on the thread and returns
it. A raise only reaches VoiceAssistant when nothing side-effecting happened.
Orphaned tool calls left by a cancelled run are repaired at the start of the
next turn so the history stays acceptable to the provider.
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
from .nodes import (build_system_prompt, last_ai_text, last_human_text,
                    succeeded, turn_tool_calls)
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


def _orphaned_tool_call_ids(messages) -> List[str]:
    """tool_call ids since the last HumanMessage that have no ToolMessage —
    left behind when a tool step was cancelled (Ctrl+C, channel shutdown) or
    the process died mid-tool. Anthropic rejects such a history outright."""
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break
    called: List[str] = []
    answered = set()
    for m in messages[start:]:
        if isinstance(m, AIMessage):
            called.extend(c["id"] for c in (getattr(m, "tool_calls", None) or []))
        elif isinstance(m, ToolMessage):
            answered.add(m.tool_call_id)
    return [cid for cid in called if cid not in answered]


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


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _snapshot_messages(messages) -> List[Dict[str, str]]:
    """History as role dicts, replay-input shaped: everything up to and
    including the current user message. The trailing assistant reply is
    dropped and tool-calling turns are skipped, matching what
    LLMProvider._record_exchange snapshots on the legacy path."""
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _message_text(m.content)})
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            t = _message_text(m.content)
            if t:
                out.append({"role": "assistant", "content": t})
    while out and out[-1]["role"] == "assistant":
        out.pop()
    return out


class AgentEngine:
    # What the last handle() turn was: "chat" | "tools" | "interrupt" |
    # "handoff" | "cached" | "confirm" | "error". VoiceAssistant maps this to
    # the research route; turns are serialized by its _turn_lock, so a plain
    # attribute is safe.
    last_turn_kind: Optional[str] = None

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
            gate=deps.gate,
            # schedule_wakeup only where a BackgroundTaskRunner can fire it:
            # ephemeral (--chat/--test) has no runner, so don't offer the tool.
            store=None if checkpoints.ephemeral else self.store,
            clock=clock)
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
                    self.last_turn_kind = "confirm"
                    return await self._answer_pending(app, thread_id, state, pending, text, user_id)
            else:
                await self._repair_orphans(app, thread_id, state)

            # Mirrors LLMProvider._prepare_request: sarcasm sliders and the request
            # counter move even on a response-cache hit.
            llm = self.deps.llm
            llm._check_sarcasm_command(text)
            llm.stats["total_requests"] += 1
            cached = self._cached_response(text)
            if cached is not None:
                llm.stats["cache_hits"] += 1
                try:
                    await app.aupdate_state(
                        self._cfg(thread_id),
                        {"messages": [HumanMessage(content=text), AIMessage(content=cached)],
                         "user_id": user_id, "handoff_message": ""},
                        as_node="finalize")
                except Exception:
                    logger.debug("could not append cached exchange to thread", exc_info=True)
                self.last_turn_kind = "cached"
                return prefix + cached

            try:
                out = await app.ainvoke(
                    {"messages": [HumanMessage(content=text)], "user_id": user_id,
                     "handoff_message": "", "tool_iterations": 0},
                    self._cfg(thread_id, user_id=user_id, run_kind="fresh"))
            except Exception:
                recovered = await self._recover_after_error(app, thread_id, text)
                if recovered is None:
                    raise
                self.last_turn_kind = "error"
                return prefix + recovered
            reply = self._reply_from(out)
            self._post_run(out, text, cacheable=cacheable)
            if out.get("__interrupt__"):
                self.last_turn_kind = "interrupt"
            elif out.get("handoff_message"):
                self.last_turn_kind = "handoff"
            elif turn_tool_calls(out.get("messages", [])):
                self.last_turn_kind = "tools"
            else:
                self.last_turn_kind = "chat"
                self._record_research(out, text, reply)
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
            await self._repair_orphans(app, thread_id, state)
            note = str(wake.payload.get("note", "")).strip()
            text = ("(Scheduled wake-up you set earlier; the user is not speaking. "
                    f"Do what you planned and report briefly: {note})")
            try:
                out = await app.ainvoke(
                    {"messages": [HumanMessage(content=text)], "user_id": wake.user_id,
                     "handoff_message": "", "tool_iterations": 0},
                    self._cfg(thread_id, user_id=wake.user_id, run_kind="wake"))
            except Exception:
                recovered = await self._recover_after_error(app, thread_id, text)
                if recovered is None:
                    raise
                return recovered
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
            try:
                out = await app.ainvoke(Command(resume={"decision": "yes"}),
                                        self._cfg(thread_id, user_id=user_id, run_kind="resume"))
            except Exception:
                recovered = await self._recover_after_error(app, thread_id, None)
                if recovered is None:
                    raise
                return recovered
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

    async def _repair_orphans(self, app, thread_id: str, state) -> bool:
        """Close tool calls that never got a result (cancelled/killed mid-tool)
        with an ERROR ToolMessage so the thread's history is valid again.
        Returns True when something was repaired."""
        messages = (getattr(state, "values", None) or {}).get("messages", []) if state is not None else []
        orphans = _orphaned_tool_call_ids(messages)
        if not orphans:
            return False
        logger.warning("Repairing %d orphaned tool call(s) on %s", len(orphans), thread_id)
        await app.aupdate_state(self._cfg(thread_id), {"messages": [
            ToolMessage(content="ERROR: that step was interrupted before it finished; "
                                "do not assume it ran.", tool_call_id=cid)
            for cid in orphans]}, as_node="tools")
        return True

    async def _recover_after_error(self, app, thread_id: str, text: Optional[str]) -> Optional[str]:
        """A run blew up (provider error, timeout…). If nothing side-effecting
        happened this turn, return None so the caller re-raises and
        VoiceAssistant falls back to the legacy router. If a tool already ran,
        the legacy path must NOT re-execute the request: close any orphaned
        call, record an in-character apology on the thread, and return it."""
        try:
            state = await app.aget_state(self._cfg(thread_id))
        except Exception:
            logger.debug("could not read thread state after an error", exc_info=True)
            return None
        messages = (getattr(state, "values", None) or {}).get("messages", [])
        if text is not None and last_human_text(messages) != text:
            return None                      # the failed turn never reached the thread
        executed = [c for c in turn_tool_calls(messages) if c[2]]
        if not executed and not _orphaned_tool_call_ids(messages):
            return None
        await self._repair_orphans(app, thread_id, state)
        done = ", ".join(sorted({n for n, _, _ in executed})) or "the first step"
        apology = (f"I'm afraid something went wrong partway through, {self._title}: "
                   f"I completed {done} but could not finish. Do check before asking again.")
        try:
            await app.aupdate_state(self._cfg(thread_id),
                                    {"messages": [AIMessage(content=apology)], "handoff_message": ""},
                                    as_node="finalize")
        except Exception:
            logger.debug("could not record the apology on the thread", exc_info=True)
        return apology

    def _record_research(self, out: Dict[str, Any], text: str, reply: str) -> None:
        """Mirror LLMProvider._record_exchange for pure-chat engine turns.

        The learning loop's shadow/replay pipeline feeds on route="chat"
        exchanges carrying a context snapshot; without this, enabling the
        engine starves the study exactly like the pre-237e9c5 ingress bug.
        Never affects the reply.
        """
        recorder = getattr(self.deps.llm, "research_recorder", None)
        if recorder is None or not reply:
            return
        try:
            system_prompt = build_system_prompt(
                self.deps.llm.personality, out.get("context_block", ""),
                bool(self.tool_set.tools))
            recorder.record_chat(
                text, reply,
                model=self.deps.llm.get_name(),
                context_snapshot={"system_prompt": system_prompt,
                                  "messages": _snapshot_messages(out.get("messages", []))},
            )
        except Exception:
            logger.debug("research recorder failed for engine turn", exc_info=True)

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
