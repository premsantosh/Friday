"""Phase 0/1 — config, scaffolding, and the checkpointed conversational core."""

import asyncio
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import agent as agent_pkg
from agent.nodes import trim_plan
from agent.store import SqliteAgentStore
from config import AgentConfig, AssistantConfig
from tests.agent_fakes import (
    EchoTimeWorkflow,
    FakeLLM,
    ScriptedChatModel,
    make_assistant,
    make_engine,
    tool_call,
)


# --------------------------------------------------------------- Phase 0

def test_agent_config_defaults_and_env(monkeypatch):
    cfg = AssistantConfig()
    assert cfg.agent.engine == "legacy"
    assert cfg.agent.tracing is False
    assert cfg.agent.tracing_sampling_rate == 1.0
    assert cfg.agent.max_tokens == 1024

    monkeypatch.setenv("FRIDAY_AGENT_ENGINE", "LangGraph")
    monkeypatch.setenv("FRIDAY_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "friday-dev")
    monkeypatch.setenv("LANGSMITH_TRACING_SAMPLING_RATE", "0.25")
    env = AgentConfig.from_env()
    assert env.engine == "langgraph"
    assert env.tracing is True
    assert env.tracing_project == "friday-dev"
    assert env.tracing_sampling_rate == 0.25


def test_conftest_strips_engine_env():
    import os
    assert "FRIDAY_AGENT_ENGINE" not in os.environ
    assert "LANGSMITH_API_KEY" not in os.environ
    assert AgentConfig.from_env().engine == "legacy"


def test_package_is_available_here():
    assert agent_pkg.is_available(), agent_pkg.unavailable_reason()


def test_assistant_builds_no_engine_by_default():
    a = make_assistant()
    assert a._build_agent_engine() is None
    assert a.engine_label == "legacy router"


def test_assistant_falls_back_when_engine_unavailable_or_failing(monkeypatch):
    a = make_assistant()
    a.config.agent.engine = "langgraph"
    monkeypatch.setattr(agent_pkg, "is_available", lambda: False)
    assert a._build_agent_engine() is None

    monkeypatch.setattr(agent_pkg, "is_available", lambda: True)
    monkeypatch.setattr(agent_pkg.AgentEngine, "from_assistant",
                        classmethod(lambda cls, asst: (_ for _ in ()).throw(RuntimeError("no key"))))
    assert a._build_agent_engine() is None


# --------------------------------------------------------------- Phase 1

@pytest.mark.asyncio
async def test_history_persists_across_turns_in_memory():
    model = ScriptedChatModel().push(AIMessage(content="Good afternoon, sir."),
                                     AIMessage(content="Still here, sir."))
    engine = make_engine(model=model)

    assert await engine.handle("hello") == "Good afternoon, sir."
    assert await engine.handle("still there?") == "Still here, sir."

    # Second model call saw the whole conversation so far (system + 3 messages).
    second = model.calls[1]
    assert isinstance(second[0], SystemMessage)
    assert [type(m).__name__ for m in second[1:]] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert second[1].content == "hello" and second[2].content == "Good afternoon, sir."


@pytest.mark.asyncio
async def test_system_prompt_rebuilt_each_turn_and_not_stored(tmp_path):
    llm = FakeLLM(ephemeral=False, preferences="tea: Earl Grey")
    model = ScriptedChatModel()
    engine = make_engine(model=model, llm=llm)
    await engine.handle("what do I like?")
    system = model.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert "Jarvis" in system.content                     # personality prompt
    assert "Earl Grey" in system.content                  # memory context block injected
    assert "<friday_context>" in system.content           # hidden from traces by marker
    # Sarcasm commands still go through the shared provider.
    assert llm.sarcasm_checks == ["what do I like?"]
    # Nothing but human/ai messages in the checkpoint.
    async with engine._checkpoints.open() as saver:
        app = engine._graph.compile(checkpointer=saver)
        state = await app.aget_state({"configurable": {"thread_id": engine.thread_id("default")}})
    assert all(not isinstance(m, SystemMessage) for m in state.values["messages"])
    assert state.values["context_block"].startswith("<user_preferences>")


@pytest.mark.asyncio
async def test_history_persists_across_engine_instances_sqlite(tmp_path):
    path = str(tmp_path / "ckpt.db")
    store = SqliteAgentStore(path)
    m1 = ScriptedChatModel().push(AIMessage(content="Noted: the cat is called Biscuit."))
    e1 = make_engine(model=m1, checkpoint_path=path, store=store)
    await e1.handle("my cat is called Biscuit")

    m2 = ScriptedChatModel().push(AIMessage(content="Biscuit, sir."))
    e2 = make_engine(model=m2, checkpoint_path=path, store=SqliteAgentStore(path))
    assert await e2.handle("what is my cat called?") == "Biscuit, sir."
    texts = [m.content for m in m2.calls[0] if isinstance(m, HumanMessage)]
    assert texts == ["my cat is called Biscuit", "what is my cat called?"]


@pytest.mark.asyncio
async def test_handle_works_from_a_secondary_thread_owned_event_loop(tmp_path):
    """Friday's channels each own a private loop; aiosqlite connections are
    loop-bound, so the per-invocation saver must work from any of them."""
    path = str(tmp_path / "ckpt.db")
    store = SqliteAgentStore(path)
    model = ScriptedChatModel().push(AIMessage(content="From the main loop."),
                                     AIMessage(content="From another loop."))
    engine = make_engine(model=model, checkpoint_path=path, store=store)
    assert await engine.handle("one") == "From the main loop."

    result = {}

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["reply"] = loop.run_until_complete(engine.handle("two"))
        except Exception as e:  # pragma: no cover - surfaced by the assert below
            result["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=30)
    assert result.get("reply") == "From another loop.", result.get("error")
    # and the second loop saw the first loop's history
    assert [m.content for m in model.calls[1] if isinstance(m, HumanMessage)] == ["one", "two"]


def test_trim_plan_keeps_whole_turns():
    msgs = [
        HumanMessage(content="h1", id="1"),
        AIMessage(content="", id="2", tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="r", tool_call_id="c1", id="3"),
        AIMessage(content="a1", id="4"),
        HumanMessage(content="h2", id="5"),
        AIMessage(content="a2", id="6"),
    ]
    # A naive cut at -4 would land on the ToolMessage; we advance to "h2".
    removals = trim_plan(msgs, 4)
    assert [r.id for r in removals] == ["1", "2", "3", "4"]
    # Window entirely inside one turn: keep from the last human.
    assert [r.id for r in trim_plan(msgs, 1)] == ["1", "2", "3", "4"]
    assert trim_plan(msgs, 10) == []


@pytest.mark.asyncio
async def test_history_is_trimmed_pair_aware_in_the_graph():
    wf = EchoTimeWorkflow()
    model = ScriptedChatModel()
    cfg = AgentConfig(engine="langgraph", history_max_messages=4)
    model.push(tool_call("time_check", {"intent": "time", "entities": {}}),
               AIMessage(content="Noon, sir."))
    engine = make_engine(workflows=[wf], model=model, agent_config=cfg, with_wakeups=False)
    await engine.handle("the time?")          # human, ai(tool), tool, ai  = 4
    await engine.handle("thanks")             # + human, ai               = 6 → trim to a turn boundary
    async with engine._checkpoints.open() as saver:
        app = engine._graph.compile(checkpointer=saver)
        state = await app.aget_state({"configurable": {"thread_id": engine.thread_id("default")}})
    msgs = state.values["messages"]
    assert isinstance(msgs[0], HumanMessage)
    assert not any(isinstance(m, ToolMessage) for m in msgs)  # the tool pair went together
    assert [m.content for m in msgs] == ["thanks", "Indeed, sir."]


@pytest.mark.asyncio
async def test_finalize_persists_memory_and_response_cache():
    llm = FakeLLM(ephemeral=False, keys=["fact.tea"])
    model = ScriptedChatModel().push(AIMessage(content="Earl Grey, sir."))
    engine = make_engine(model=model, llm=llm)

    assert await engine.handle("what tea do I like?") == "Earl Grey, sir."
    assert llm.store.turns == [("user", "what tea do I like?"), ("assistant", "Earl Grey, sir.")]
    assert llm.extracted == [("what tea do I like?", "Earl Grey, sir.")]

    # Same question again: response cache hit, no LLM call, history stays coherent.
    assert await engine.handle("what tea do I like?") == "Earl Grey, sir."
    assert len(model.calls) == 1
    assert llm.stats["cache_hits"] == 1

    # Confidence feedback: the next turn confirms last turn's retrieved facts ...
    await engine.handle("lovely")
    assert ("bump", "fact.tea") in llm.store.confidence
    # ... and a correction drops them.
    await engine.handle("no, that's wrong")
    assert ("drop", "fact.tea") in llm.store.confidence


@pytest.mark.asyncio
async def test_ephemeral_mode_writes_nothing():
    llm = FakeLLM(ephemeral=True)
    engine = make_engine(model=ScriptedChatModel(), llm=llm)
    assert engine.ephemeral
    await engine.handle("hello")
    assert llm.extracted == [] and llm.store is None


@pytest.mark.asyncio
async def test_reset_starts_a_fresh_thread():
    model = ScriptedChatModel()
    engine = make_engine(model=model)
    await engine.handle("remember the number 42")
    engine.reset()
    await engine.handle("what number?")
    assert [m.content for m in model.calls[1] if isinstance(m, HumanMessage)] == ["what number?"]


@pytest.mark.asyncio
async def test_empty_model_reply_gets_a_fallback_line():
    model = ScriptedChatModel().push(AIMessage(content=""))
    engine = make_engine(model=model)
    reply = await engine.handle("...")
    assert reply.startswith("I'm afraid I have nothing useful to say")


@pytest.mark.asyncio
async def test_orphaned_tool_call_is_repaired_on_the_next_turn():
    """A cancelled/killed tool step leaves an AI tool_call with no result; the
    next turn must close it or the provider rejects the whole thread."""
    wf = EchoTimeWorkflow()
    model = ScriptedChatModel().push(AIMessage(content="Good evening, sir."))
    engine = make_engine(workflows=[wf], model=model)
    tid = engine.thread_id("default")
    async with engine._checkpoints.open() as saver:
        app = engine._graph.compile(checkpointer=saver)
        await app.aupdate_state(
            {"configurable": {"thread_id": tid}},
            {"messages": [HumanMessage(content="do the slow thing"),
                          tool_call("time_check", {"intent": "slow", "entities": {}}, "orphan1")],
             "user_id": "default", "handoff_message": "", "tool_iterations": 1},
            as_node="agent")
    assert await engine.handle("hello again") == "Good evening, sir."
    seen = model.calls[0][1:]
    assert [type(m).__name__ for m in seen] == ["HumanMessage", "AIMessage", "ToolMessage", "HumanMessage"]
    assert seen[2].tool_call_id == "orphan1" and seen[2].content.startswith("ERROR:")
    assert wf.calls == []                                 # never re-executed


@pytest.mark.asyncio
async def test_error_after_a_tool_ran_does_not_fall_back_to_legacy():
    """Provider dies after the graph already executed a workflow: the legacy
    router must not run the request a second time."""
    wf = EchoTimeWorkflow()
    model = ScriptedChatModel().push(tool_call("time_check", {"intent": "brew", "entities": {}}),
                                     RuntimeError("provider overloaded"))
    engine = make_engine(workflows=[wf], model=model)
    a = make_assistant(engine=engine)
    reply = await a.process_input("brew the thing")
    assert "something went wrong partway through" in reply and "time_check" in reply
    assert wf.calls == [("brew", {})]                    # executed exactly once
    assert a.intent_router.calls == []                    # legacy router NOT consulted
    # Thread is healthy afterwards.
    model.push(AIMessage(content="Still here, sir."))
    assert await a.process_input("you there?") == "Still here, sir."


@pytest.mark.asyncio
async def test_error_before_any_tool_still_falls_back_to_legacy():
    model = ScriptedChatModel().push(RuntimeError("provider down"))
    engine = make_engine(workflows=[EchoTimeWorkflow()], model=model)
    a = make_assistant(engine=engine)
    assert await a.process_input("hello") == "router reply: hello"


def test_ephemeral_engine_does_not_offer_schedule_wakeup():
    """--chat/--test have no BackgroundTaskRunner, so the tool must not exist
    there (it would promise a wake-up that can never fire)."""
    from agent import AgentEngine, EngineDeps
    from agent.checkpoint import CheckpointerProvider
    from agent.store import AgentStore
    from agent.tools import SCHEDULE_WAKEUP
    from config import AgentConfig
    from workflows.base import WorkflowManager

    def build(path):
        deps = EngineDeps(llm=FakeLLM(), workflows=WorkflowManager(), model=ScriptedChatModel(),
                          agent_config=AgentConfig(engine="langgraph"), store=AgentStore())
        return AgentEngine(deps, checkpoints=CheckpointerProvider(path))

    assert SCHEDULE_WAKEUP not in [t.name for t in build(None).tool_set.tools]


def test_durable_engine_offers_schedule_wakeup(tmp_path):
    from agent import AgentEngine, EngineDeps
    from agent.checkpoint import CheckpointerProvider
    from agent.tools import SCHEDULE_WAKEUP
    from config import AgentConfig
    from workflows.base import WorkflowManager

    path = str(tmp_path / "ckpt.db")
    deps = EngineDeps(llm=FakeLLM(), workflows=WorkflowManager(), model=ScriptedChatModel(),
                      agent_config=AgentConfig(engine="langgraph"), store=SqliteAgentStore(path))
    engine = AgentEngine(deps, checkpoints=CheckpointerProvider(path))
    assert SCHEDULE_WAKEUP in [t.name for t in engine.tool_set.tools]


@pytest.mark.asyncio
async def test_response_cache_hit_still_runs_sarcasm_check_and_counts():
    llm = FakeLLM(ephemeral=False)
    model = ScriptedChatModel().push(AIMessage(content="Quite, sir."))
    engine = make_engine(model=model, llm=llm)
    await engine.handle("be more sarcastic please")
    await engine.handle("be more sarcastic please")    # cache hit
    assert llm.sarcasm_checks == ["be more sarcastic please"] * 2
    assert llm.stats["total_requests"] == 2 and llm.stats["cache_hits"] == 1


# ------------------------------------------------------ assistant seam

@pytest.mark.asyncio
async def test_process_input_uses_engine_when_present():
    model = ScriptedChatModel().push(AIMessage(content="graph says hello"))
    engine = make_engine(model=model)
    a = make_assistant(engine=engine)
    assert await a.process_input("hello there") == "graph says hello"
    assert a.intent_router.calls == []          # legacy router not consulted


@pytest.mark.asyncio
async def test_process_input_falls_back_to_legacy_when_engine_raises():
    model = ScriptedChatModel().push(RuntimeError("provider down"))
    engine = make_engine(model=model)
    a = make_assistant(engine=engine)
    assert await a.process_input("hello there") == "router reply: hello there"
    assert a.intent_router.calls == ["hello there"]


@pytest.mark.asyncio
async def test_process_input_legacy_when_no_engine():
    a = make_assistant(engine=None)
    assert await a.process_input("hello there") == "router reply: hello there"


def test_clear_history_resets_engine_thread():
    engine = make_engine(model=ScriptedChatModel())
    a = make_assistant(engine=engine)
    before = engine.thread_id("default")
    a.clear_history()
    assert engine.thread_id("default") != before
