"""Shared fakes for the agent-engine tests: a scripted chat model, a stand-in
LLMProvider (with an optional in-memory memory stack), fake workflows, and
builders for an AgentEngine / a VoiceAssistant wired without audio or network."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent import AgentEngine, EngineDeps
from agent.checkpoint import CheckpointerProvider
from agent.store import AgentStore
from agent.tools import build_tools
from config import AgentConfig, AssistantConfig, PersonalityConfig
from core.conversation import TurnResult
from llm.router import RouteResult
from memory.cache import FridayCache
from workflows.base import (
    ConversationalWorkflow,
    Workflow,
    WorkflowManager,
    WorkflowResult,
    WorkflowStatus,
    WorkflowTrigger,
)


# ----------------------------------------------------------------- model

def tool_call(name: str, args: Optional[Dict[str, Any]] = None, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": dict(args or {}), "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """Returns scripted AIMessages in order (an Exception entry raises); records
    every call's input messages. When the script runs dry it answers with
    `default_reply`, so multi-turn tests don't need to script every turn."""

    script: List[Any] = Field(default_factory=list)
    calls: List[Any] = Field(default_factory=list)
    bound_tools: List[Any] = Field(default_factory=list)
    bind_kwargs: Dict[str, Any] = Field(default_factory=dict)
    default_reply: str = "Indeed, sir."

    def push(self, *items) -> "ScriptedChatModel":
        self.script.extend(items)
        return self

    def _next(self, messages):
        self.calls.append(list(messages))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return AIMessage(content=self.default_reply)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        self.bind_kwargs = dict(kwargs)
        return self


# -------------------------------------------------------------- memory

class FakeStore:
    def __init__(self):
        self.turns: List[tuple] = []
        self.confidence: List[tuple] = []
        self.facts: Dict[str, Any] = {}

    def log_turn(self, role, content):
        self.turns.append((role, content))

    def bump_confidence(self, key, delta=0.05):
        self.confidence.append(("bump", key))

    def drop_confidence(self, key, delta=0.3):
        self.confidence.append(("drop", key))

    def remember(self, **kw):
        self.facts[kw["key"]] = kw


class FakeContextBuilder:
    def __init__(self, cache, *, preferences: str = "", keys=None):
        self.cache = cache
        self.preferences = preferences
        self.keys = list(keys or [])

    def query_fingerprint(self, query: str) -> str:
        normalized = re.sub(r"\s+", " ", query.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def build_context(self, text: str) -> dict:
        cached = self.cache.get_cached_response(self.query_fingerprint(text))
        if cached:
            return {"cached_response": cached}
        return {"conversation_history": [], "retrieved_fact_keys": list(self.keys),
                "preferences": self.preferences or None}

    def format_system_prompt(self, base: str, context: dict) -> str:
        parts = [base]
        if context.get("preferences"):
            parts.append(f"<user_preferences>\n{context['preferences']}\n</user_preferences>")
        return "\n".join(parts)


class FakeExtractor:
    def is_correction(self, text: str) -> bool:
        return text.lower().startswith(("no,", "wrong", "actually"))


class FakeLLM:
    """Stand-in for llm.providers.LLMProvider as the engine uses it."""

    def __init__(self, *, ephemeral: bool = True, preferences: str = "", keys=None):
        self.personality = PersonalityConfig(name="Jarvis", user_title="sir")
        self.cache = FridayCache()
        if ephemeral:
            self.store = None
            self.context_builder = None
            self.extractor = None
        else:
            self.store = FakeStore()
            self.context_builder = FakeContextBuilder(self.cache, preferences=preferences, keys=keys)
            self.extractor = FakeExtractor()
        self.search_enhancer = None
        self.stats = {"total_requests": 0, "cache_hits": 0, "llm_calls": 0,
                      "ollama_extractions": 0, "facts_stored": 0, "search_queries": 0}
        self.sarcasm_checks: List[str] = []
        self.extracted: List[tuple] = []
        self.generate_calls: List[str] = []
        self.conversation_history: List[dict] = []

    def _check_sarcasm_command(self, text: str) -> bool:
        self.sarcasm_checks.append(text)
        return False

    def _extract_and_store(self, user_input: str, response: str) -> None:
        self.extracted.append((user_input, response))

    def generate_response(self, text: str) -> str:
        self.generate_calls.append(text)
        return f"legacy: {text}"

    def clear_history(self) -> None:
        self.conversation_history = []


# ----------------------------------------------------------- workflows

class EchoTimeWorkflow(Workflow):
    """Simple single-shot workflow; obscure trigger vocabulary kept from the
    era of Step-1 keyword matching (now removed) — harmless, and it keeps the
    tool descriptions distinctive."""

    def __init__(self, message: str = "It is noon, sir."):
        self.calls: List[tuple] = []
        self.message = message

    @property
    def name(self):
        return "time_check"

    @property
    def description(self):
        return "Tell the current time"

    @property
    def trigger(self):
        return WorkflowTrigger(keywords=["chronometer"], examples=["what time is it", "tell me the time"])

    async def execute(self, intent, entities):
        self.calls.append((intent, dict(entities)))
        return WorkflowResult(status=WorkflowStatus.SUCCESS, message=self.message,
                              data={"intent": intent})


class FailingWorkflow(Workflow):
    @property
    def name(self):
        return "broken"

    @property
    def description(self):
        return "A capability that is down"

    @property
    def trigger(self):
        return WorkflowTrigger(keywords=["flibbertigibbet"], examples=["use the broken thing"])

    async def execute(self, intent, entities):
        return WorkflowResult(status=WorkflowStatus.FAILURE,
                              message="The widget is being uncooperative, sir.", error="boom")


class FakeLockWorkflow(Workflow):
    """Records executes; `fail` makes it report FAILURE."""

    def __init__(self, name: str = "locks", fail: bool = False):
        self._name = name
        self.fail = fail
        self.calls: List[tuple] = []

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Control door locks"

    @property
    def trigger(self):
        return WorkflowTrigger(keywords=["deadbolt"], examples=["lock the front door", "unlock the back door"])

    async def execute(self, intent, entities):
        self.calls.append((intent, dict(entities)))
        if self.fail:
            return WorkflowResult(status=WorkflowStatus.FAILURE, message="The lock did not respond, sir.",
                                  error="timeout")
        action = entities.get("action", "lock")
        door = entities.get("door", "front")
        state = "unlocked" if action == "unlock" else "locked"
        return WorkflowResult(status=WorkflowStatus.SUCCESS, message=f"The {door} door is {state}, sir.")


class FakeReminderWorkflow(ConversationalWorkflow):
    """2-slot conversational workflow (same shape as tests/test_conversation.py)."""

    session_timeout_s = 600

    @property
    def name(self):
        return "reminder"

    @property
    def description(self):
        return "Set a reminder."

    @property
    def trigger(self):
        return WorkflowTrigger(keywords=["remind"], examples=["remind me to..."])

    async def start(self, intent, entities, session):
        return TurnResult.ask("What should I remind you about, sir?", next_state="collect_task")

    async def resume(self, text, session):
        if session.fsm_state == "collect_task":
            return TurnResult.ask("And when?", slots_update={"task": text}, next_state="collect_time")
        if session.fsm_state == "collect_time":
            return TurnResult.confirm(f"Remind you to {session.slots['task']} at {text}?",
                                      slots_update={"when": text}, next_state="confirm")
        if session.fsm_state == "confirm":
            if text.strip().lower() in ("yes", "y", "correct", "yep"):
                return TurnResult.complete("Consider it done, sir.")
            return TurnResult.cancel("As you wish, sir.")
        return TurnResult.complete("")


class RecordingIntentCache:
    def __init__(self):
        self.stored: List[tuple] = []

    def query(self, text):
        return None

    def store(self, text, workflow_name, entities):
        self.stored.append((text, workflow_name, dict(entities)))


class RecordingContext:
    """ConversationContext stand-in: enrich is identity, update is recorded."""

    def __init__(self):
        self.updates: List[tuple] = []

    enabled = True

    def enrich(self, text):
        return text

    def update(self, workflow_name, entities=None, text=""):
        self.updates.append((workflow_name, dict(entities or {}), text))

    def clear(self):
        pass


class StubRouter:
    """Legacy router stand-in. Classification only: RouteResult carries no reply
    (llm/router.py), so a no-workflow turn falls through to FakeLLM.generate_response."""

    def __init__(self):
        self.calls: List[str] = []

    def route(self, text, workflow_manager):
        self.calls.append(text)
        return RouteResult(workflow_name=None)


# ------------------------------------------------------------ builders

def make_workflow_manager(*workflows) -> WorkflowManager:
    wm = WorkflowManager()
    for wf in workflows:
        wm.register(wf)
    return wm


def make_engine(*, workflows=(), model=None, llm=None, sessions=None, context=None,
                intent_cache=None, gate=None, gate_specs=None, checkpoint_path=None,
                store=None, clock=time.time, agent_config=None, tracer=None,
                with_wakeups: bool = True, confirmation_timeout_s: float = 600.0) -> AgentEngine:
    wm = workflows if isinstance(workflows, WorkflowManager) else make_workflow_manager(*workflows)
    llm = llm or FakeLLM()
    model = model or ScriptedChatModel()
    cfg = agent_config or AgentConfig(engine="langgraph")
    store = store or AgentStore()
    deps = EngineDeps(llm=llm, workflows=wm, model=model, agent_config=cfg, sessions=sessions,
                      context=context, intent_cache=intent_cache, gate=gate, store=store,
                      confirmation_timeout_s=confirmation_timeout_s)
    tools = build_tools(wm, sessions=sessions, context=context, gate=gate,
                        store=store if with_wakeups else None, gate_specs=gate_specs, clock=clock)
    return AgentEngine(deps, checkpoints=CheckpointerProvider(checkpoint_path), tracer=tracer,
                       tools=tools, clock=clock)


def make_assistant(*, engine=None, workflows=None, sessions=None, llm=None,
                   intent_cache=None, router=None, context=None):
    """A VoiceAssistant with process_input wired but no audio/STT/TTS/network."""
    from core.assistant import VoiceAssistant

    a = VoiceAssistant.__new__(VoiceAssistant)
    a.config = AssistantConfig()
    a.llm = llm or FakeLLM()
    a.workflows = workflows or WorkflowManager()
    a.intent_cache = intent_cache
    a.intent_router = router or StubRouter()
    a._context_enabled = False
    a.context = context or RecordingContext()
    a.sessions = sessions
    a.background_runner = None
    a.agent_engine = engine
    # Research substrate hooks (core/assistant.py __init__): off, like production
    # without FRIDAY_RESEARCH=1.
    a.research_recorder = None
    a.last_route = None
    a.last_outcome = None
    return a
