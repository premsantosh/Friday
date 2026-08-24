"""Regression guard for the bug that starved the learning loop.

For a week, free-chat turns returned `route.response` — a one-sentence reply
drafted by the intent-classifier prompt, which has no personality, no history,
no remembered facts and no search. Two consequences:

  * production quality: chat replies sounded like a help menu, not Friday;
  * the study: LLMProvider._record_exchange never ran, so no context snapshot
    was ever captured, so the shadow runner, the nightly replay and every arm
    had nothing to work with. The live DB showed 4 exchanges and 0 shadow rows.

These tests assert the chat route goes through the LLM provider, and that
internally synthesized prompts stay out of the study's corpus.
"""

from __future__ import annotations

import threading

import pytest

from core.assistant import VoiceAssistant
from llm.router import RouteResult
from workflows.base import WorkflowResult, WorkflowStatus


class FakeLLM:
    def __init__(self, reply: str = "Something light, sir."):
        self.reply = reply
        self.calls: list[tuple[str, bool]] = []

    def generate_response(self, user_input: str, *, record_research: bool = True) -> str:
        self.calls.append((user_input, record_research))
        return self.reply


class FakeRouter:
    """Stands in for IntentRouter, returning a fixed classification."""

    def __init__(self, workflow_name=None):
        self.result = RouteResult(workflow_name=workflow_name)

    def route(self, text, workflow_manager):
        return self.result


class FakeWorkflows:
    def __init__(self, workflows=None):
        self.workflows = workflows or {}


class FakeContext:
    def enrich(self, text):
        return text

    def update(self, *args, **kwargs):
        pass


def _assistant(llm, router, workflows=None) -> VoiceAssistant:
    """Build the minimum surface _process_input_inner touches, no real config."""
    a = VoiceAssistant.__new__(VoiceAssistant)
    a.llm = llm
    a.intent_router = router
    a.workflows = workflows or FakeWorkflows()
    a.context = FakeContext()
    a._context_enabled = True
    a.sessions = None
    a.intent_cache = None
    a.agent_engine = None
    a.background_runner = None
    a.research_recorder = None
    a.last_route = None
    a.last_outcome = None
    a._turn_lock = threading.Lock()
    a.config = type("C", (), {"debug_mode": False})()
    return a


@pytest.mark.asyncio
async def test_chat_route_always_calls_the_llm():
    """Even when the classifier volunteers a reply, the LLM provider answers."""
    llm = FakeLLM("Something light, sir.")
    a = _assistant(llm, FakeRouter(workflow_name=None))

    reply = await a._process_input_inner("what should I make for dinner")

    assert reply == "Something light, sir."
    assert llm.calls == [("what should I make for dinner", True)]
    assert a.last_route == "chat"


@pytest.mark.asyncio
async def test_workflow_failure_reply_is_not_recorded_as_research():
    """The failure prompt is synthesized by us, so it must not enter the corpus."""
    llm = FakeLLM("The lights are unreachable, sir.")

    class FailingWorkflow:
        name = "philips_hue"

        async def execute(self, text, entities):
            return WorkflowResult(status=WorkflowStatus.FAILURE,
                                  message="failed", error="bridge offline")

    workflows = FakeWorkflows({"philips_hue": FailingWorkflow()})
    a = _assistant(llm, FakeRouter(workflow_name="philips_hue"), workflows)

    reply = await a._process_input_inner("turn on the lights")

    assert reply == "The lights are unreachable, sir."
    assert len(llm.calls) == 1
    prompt, record_research = llm.calls[0]
    assert record_research is False, "synthetic failure prompts must not be recorded"
    assert "bridge offline" in prompt
    assert a.last_outcome == "failure"


@pytest.mark.asyncio
async def test_workflow_success_marks_outcome_success():
    llm = FakeLLM()

    class OkWorkflow:
        name = "philips_hue"

        async def execute(self, text, entities):
            return WorkflowResult(status=WorkflowStatus.SUCCESS, message="Done, sir.")

    workflows = FakeWorkflows({"philips_hue": OkWorkflow()})
    a = _assistant(llm, FakeRouter(workflow_name="philips_hue"), workflows)

    reply = await a._process_input_inner("turn on the lights")

    assert reply == "Done, sir."
    assert a.last_outcome == "success"
    assert llm.calls == [], "a successful workflow needs no LLM call"


@pytest.mark.asyncio
async def test_free_text_never_executes_a_workflow_without_the_router():
    """The Step-1 keyword fast-path is gone. "Good night my friend" used to
    substring-match hue_lights ("good night") and toggle every light with
    empty entities. Only the router/agent may pick a workflow now; a null
    classification means the LLM provider answers."""

    class HueLike:
        name = "hue_lights"

        def __init__(self):
            self.calls = 0

        async def execute(self, intent, entities):
            self.calls += 1
            return WorkflowResult(
                status=WorkflowStatus.SUCCESS,
                message="I've toggled the all lights, sir.")

    hue = HueLike()
    llm = FakeLLM(reply="Good night, sir.")
    a = _assistant(llm, FakeRouter(workflow_name=None),
                   workflows=FakeWorkflows({"hue_lights": hue}))
    assert await a.process_input("Good night my friend") == "Good night, sir."
    assert hue.calls == 0
    assert a.last_route == "chat"
