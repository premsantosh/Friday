"""Intent-cache hits for data-bearing workflows (expose_data_to_agent).

Direct execution would speak the template and leave the LangGraph thread
without the turn, so follow-ups lose context. Instead the engine pre-runs the
tool, seeds Human + AI(tool_call) + ToolMessage onto the thread and one model
call composes the reply from the DATA block. Plain workflows keep the
zero-LLM direct path; without the engine the legacy path is unchanged.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.tools import DATA_BLOCK_HEADER
from tests.agent_fakes import (EchoTimeWorkflow, FakeLockWorkflow, RecordingContext,
                               RecordingIntentCache, ScriptedChatModel, make_assistant,
                               make_engine, make_workflow_manager, tool_call)
from tests.test_agent_hitl import LOCK_SPECS, make_gate

QUESTION = "how was last night's run?"


class DataTimeWorkflow(EchoTimeWorkflow):
    expose_data_to_agent = True

    @property
    def name(self):
        return "time_data"


class ExplodingDataWorkflow(DataTimeWorkflow):
    async def execute(self, intent, entities):
        raise RuntimeError("records unavailable")


def build(*, wf=None, script=(), hit=("time_data", {"topic": "stale"}), engine=True):
    wf = wf or DataTimeWorkflow()
    wm = make_workflow_manager(wf)
    model = ScriptedChatModel().push(*script)
    cache = RecordingIntentCache(hit=hit)
    context = RecordingContext()
    eng = make_engine(workflows=wm, model=model, intent_cache=cache, context=context) if engine else None
    a = make_assistant(engine=eng, workflows=wm, intent_cache=cache, context=context)
    return a, wf, model, cache, context


@pytest.mark.asyncio
async def test_hit_for_data_bearing_workflow_composes_in_one_call():
    a, wf, model, _, _ = build(script=[AIMessage(content="Composed, sir.")])
    assert await a.process_input(QUESTION) == "Composed, sir."
    assert a.last_route == "cache+agent:time_data"
    assert a.agent_engine.last_turn_kind == "cached_tool"
    assert wf.calls == [(QUESTION, {})]                  # stale cached topic dropped
    assert len(model.calls) == 1                         # no tool-selection call
    ai, tool = model.calls[0][-2], model.calls[0][-1]
    assert isinstance(ai, AIMessage) and ai.tool_calls[0]["name"] == "time_data"
    assert ai.tool_calls[0]["id"].startswith("cached_")
    assert isinstance(tool, ToolMessage) and tool.tool_call_id == ai.tool_calls[0]["id"]
    assert tool.content.startswith("It is noon, sir.") and DATA_BLOCK_HEADER in tool.content


@pytest.mark.asyncio
async def test_follow_up_sees_full_exchange():
    a, _, model, cache, _ = build(script=[AIMessage(content="Composed, sir."),
                                          AIMessage(content="More, sir.")])
    await a.process_input(QUESTION)
    cache.hit = None
    assert await a.process_input("more details please") == "More, sir."
    kinds = [type(m).__name__ for m in model.calls[1][1:]]     # skip the system message
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]
    humans = [m.content for m in model.calls[1] if isinstance(m, HumanMessage)]
    assert humans == [QUESTION, "more details please"]


@pytest.mark.asyncio
async def test_cached_tool_turn_updates_context_and_rewrites_cache():
    a, _, _, cache, context = build(script=[AIMessage(content="Composed, sir.")])
    await a.process_input(QUESTION)
    assert context.updates == [("time_data", {}, QUESTION)]
    assert cache.stored == [(QUESTION, "time_data", {})]


@pytest.mark.asyncio
async def test_plain_workflow_hit_executes_directly():
    a, wf, model, _, _ = build(wf=EchoTimeWorkflow(), hit=("time_check", {}))
    assert await a.process_input("the time please") == "It is noon, sir."
    assert a.last_route == "cache:time_check"
    assert model.calls == [] and wf.calls == [("the time please", {})]


@pytest.mark.asyncio
async def test_hit_without_engine_uses_legacy_direct_execution():
    a, _, _, _, _ = build(engine=False)
    assert await a.process_input(QUESTION) == "It is noon, sir."
    assert a.last_route == "cache:time_data"


@pytest.mark.asyncio
async def test_tool_error_reaches_model_not_fallback():
    a, _, model, cache, context = build(wf=ExplodingDataWorkflow(),
                                        script=[AIMessage(content="Records are down, sir.")])
    assert await a.process_input(QUESTION) == "Records are down, sir."
    assert a.last_route == "cache+agent:time_data"
    assert len(model.calls) == 1
    assert model.calls[0][-1].content.startswith("ERROR:")
    assert cache.stored == [] and context.updates == []


@pytest.mark.asyncio
async def test_engine_failure_before_execution_falls_back_to_handle(monkeypatch):
    a, wf, model, _, _ = build(script=[AIMessage(content="Plain chat, sir.")])

    async def boom(*args, **kwargs):
        raise LookupError("no such tool")
    monkeypatch.setattr(a.agent_engine, "handle_cached_tool", boom)
    assert await a.process_input(QUESTION) == "Plain chat, sir."
    assert a.last_route == "chat" and wf.calls == []
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_model_may_call_tool_again_after_cached_result():
    a, wf, model, cache, _ = build(script=[
        tool_call("time_data", {"intent": "detail", "entities": {"topic": "evals"}}, "c2"),
        AIMessage(content="Detail, sir.")])
    assert await a.process_input(QUESTION) == "Detail, sir."
    assert a.last_route == "cache+agent:time_data"
    assert len(model.calls) == 2 and len(wf.calls) == 2
    assert cache.stored == []                            # two tool calls: not a cacheable route


@pytest.mark.asyncio
async def test_pending_interrupt_delegates_to_handle():
    lock, data = FakeLockWorkflow(), DataTimeWorkflow()
    model = ScriptedChatModel().push(
        tool_call("locks", {"intent": "unlock the back door",
                            "entities": {"action": "unlock", "door": "back"}}),
        AIMessage(content="Unlocked, sir."))
    engine = make_engine(workflows=[lock, data], model=model, gate=make_gate(),
                         gate_specs=LOCK_SPECS)
    assert "unlock the back door" in (await engine.handle("unlock the back door")).lower()
    reply = await engine.handle_cached_tool("yes", workflow_name="time_data", entities={})
    assert engine.last_turn_kind == "confirm"
    assert data.calls == []                              # the cached tool never ran
    assert reply == "Unlocked, sir." or lock.calls      # confirmation went through
