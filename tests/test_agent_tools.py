"""Phase 2 — native tool calling over the workflow registry, start_task handoff,
intent-cache writeback guard, iteration cap."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.tools import START_TASK, build_tools, tool_description
from config import AgentConfig
from core.conversation import InMemorySessionStore, SessionManager
from tests.agent_fakes import (
    EchoTimeWorkflow,
    FailingWorkflow,
    FakeReminderWorkflow,
    RecordingContext,
    RecordingIntentCache,
    ScriptedChatModel,
    make_assistant,
    make_engine,
    make_workflow_manager,
    tool_call,
)
from workflows import TimeWorkflow, create_default_workflow_manager


def test_schema_generated_for_every_registered_workflow():
    wm = create_default_workflow_manager()
    wm.register(TimeWorkflow())
    ts = build_tools(wm)
    assert {t.name for t in ts.tools} == set(wm.workflows)
    for tool in ts.tools:
        props = tool.tool_call_schema.model_json_schema()["properties"]
        assert set(props) == {"intent", "entities"}, tool.name
        wf = wm.workflows[tool.name]
        assert tool.description == tool_description(wf)
        assert wf.description.rstrip(".") in tool.description
        if wf.trigger.examples:
            assert wf.trigger.examples[0] in tool.description
    assert ts.simple_names == set(wm.workflows)
    assert not ts.gated_names and not ts.conversational_names


@pytest.mark.asyncio
async def test_tool_call_executes_workflow_and_speaks_result():
    wf = EchoTimeWorkflow()
    model = ScriptedChatModel().push(
        tool_call("time_check", {"intent": "what's the time", "entities": {"zone": "local"}}),
        AIMessage(content="It is noon, sir."))
    engine = make_engine(workflows=[wf], model=model)
    assert await engine.handle("what's the time") == "It is noon, sir."
    assert wf.calls == [("what's the time", {"zone": "local"})]
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and tool_msgs[0].content == "It is noon, sir."
    assert model.bind_kwargs.get("parallel_tool_calls") is False
    assert {t.name for t in model.bound_tools} >= {"time_check", "schedule_wakeup"}


@pytest.mark.asyncio
async def test_failure_becomes_error_result_and_is_not_cached():
    cache = RecordingIntentCache()
    model = ScriptedChatModel().push(
        tool_call("broken", {"intent": "do it", "entities": {}}),
        AIMessage(content="I'm afraid the widget is being uncooperative, sir."))
    engine = make_engine(workflows=[FailingWorkflow()], model=model, intent_cache=cache)
    reply = await engine.handle("use the broken thing")
    assert reply.startswith("I'm afraid the widget")
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content.startswith("ERROR: boom")
    assert cache.stored == []


@pytest.mark.asyncio
async def test_start_task_opens_session_and_returns_opening_line_verbatim():
    sessions = SessionManager(InMemorySessionStore(), make_workflow_manager(FakeReminderWorkflow()))
    context = RecordingContext()
    model = ScriptedChatModel().push(
        tool_call(START_TASK, {"workflow_name": "reminder", "intent": "remind me", "entities": {}}))
    engine = make_engine(workflows=[FakeReminderWorkflow()], model=model, sessions=sessions,
                         context=context)
    assert engine.tool_set.conversational_names == {"reminder"}
    assert engine.tool_set.by_name("reminder") is None      # not a direct tool
    assert START_TASK in [t.name for t in engine.tool_set.tools]

    reply = await engine.handle("remind me about something")
    assert reply == "What should I remind you about, sir?"
    assert len(model.calls) == 1                              # handoff ends the turn, no 2nd LLM call
    assert sessions.has_active("default")
    assert context.updates == [("reminder", {}, "remind me")]


@pytest.mark.asyncio
async def test_start_task_rejects_unknown_or_duplicate_tasks():
    sessions = SessionManager(InMemorySessionStore(), make_workflow_manager(FakeReminderWorkflow()))
    model = ScriptedChatModel().push(
        tool_call(START_TASK, {"workflow_name": "nope", "intent": "x", "entities": {}}),
        AIMessage(content="I don't have such a task, sir."))
    engine = make_engine(workflows=[FakeReminderWorkflow()], model=model, sessions=sessions)
    assert await engine.handle("do the nope") == "I don't have such a task, sir."
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content.startswith("ERROR: unknown task")


@pytest.mark.asyncio
async def test_handoff_then_next_turn_routes_into_the_legacy_session():
    wm = make_workflow_manager(FakeReminderWorkflow())
    sessions = SessionManager(InMemorySessionStore(), wm)
    model = ScriptedChatModel().push(
        tool_call(START_TASK, {"workflow_name": "reminder", "intent": "remind me", "entities": {}}))
    engine = make_engine(workflows=wm, model=model, sessions=sessions)
    a = make_assistant(engine=engine, workflows=wm, sessions=sessions)

    # Phrased so Step 1 keyword matching ("remind") doesn't catch it: the graph
    # must pick start_task itself.
    assert await a.process_input("I need a nudge about something") == "What should I remind you about, sir?"
    assert await a.process_input("buy milk") == "And when?"               # Step 0: session owns the turn
    assert await a.process_input("six") == "Remind you to buy milk at six?"
    assert await a.process_input("yes") == "Consider it done, sir."
    assert len(model.calls) == 1                                           # graph untouched meanwhile
    assert not sessions.has_active("default")


@pytest.mark.asyncio
async def test_intent_cache_writeback_only_for_single_simple_tool_on_raw_input():
    async def run(*, cacheable, script, workflows):
        cache = RecordingIntentCache()
        context = RecordingContext()
        model = ScriptedChatModel().push(*script)
        engine = make_engine(workflows=workflows, model=model, intent_cache=cache, context=context)
        await engine.handle("the time please", cacheable=cacheable)
        return cache, context

    single = [tool_call("time_check", {"intent": "time", "entities": {"fmt": "24h"}}),
              AIMessage(content="Noon.")]
    cache, context = await run(cacheable=True, script=single, workflows=[EchoTimeWorkflow()])
    assert cache.stored == [("the time please", "time_check", {"fmt": "24h"})]
    assert context.updates == [("time_check", {"fmt": "24h"}, "the time please")]

    cache, _ = await run(cacheable=False, script=single, workflows=[EchoTimeWorkflow()])
    assert cache.stored == []                       # context-enriched turn: never cached

    two = [tool_call("time_check", {"intent": "time", "entities": {}}, "c1"),
           tool_call("time_check", {"intent": "time", "entities": {}}, "c2"),
           AIMessage(content="Noon, twice.")]
    cache, _ = await run(cacheable=True, script=two, workflows=[EchoTimeWorkflow()])
    assert cache.stored == []                       # multi-step run: not a cacheable route

    cache, _ = await run(cacheable=True,
                         script=[AIMessage(content="Just chatting.")], workflows=[EchoTimeWorkflow()])
    assert cache.stored == []                       # no tool used


@pytest.mark.asyncio
async def test_tool_iteration_cap_closes_calls_and_apologises():
    wf = EchoTimeWorkflow()
    cfg = AgentConfig(engine="langgraph", max_tool_iterations=2)
    model = ScriptedChatModel()
    for i in range(10):
        model.push(tool_call("time_check", {"intent": "again", "entities": {}}, f"c{i}"))
    engine = make_engine(workflows=[wf], model=model, agent_config=cfg)
    reply = await engine.handle("keep checking the time")
    assert "more steps than I am permitted" in reply
    assert len(wf.calls) == 2                       # cap honoured
    assert len(model.calls) == 3                    # 2 allowed iterations + the one that overflowed
    # History stays valid: every tool_call has a ToolMessage.
    async with engine._checkpoints.open() as saver:
        app = engine._graph.compile(checkpointer=saver)
        state = await app.aget_state({"configurable": {"thread_id": engine.thread_id("default")}})
    msgs = state.values["messages"]
    call_ids = {c["id"] for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])}
    result_ids = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
    assert call_ids == result_ids
