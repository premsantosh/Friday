"""Phase 4 — self-scheduled wake-ups serviced by BackgroundTaskRunner."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.store import AgentStore, SqliteAgentStore
from agent.tools import SCHEDULE_WAKEUP, GateSpec
from core.conversation import BackgroundTaskRunner, InMemorySessionStore, SessionManager
from core.harness import ActionGate, ActionKind, AuditLog
from tests.agent_fakes import (
    EchoTimeWorkflow,
    FakeLockWorkflow,
    ScriptedChatModel,
    make_engine,
    make_workflow_manager,
    tool_call,
)


class Clock:
    def __init__(self, now=2_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def wake_call(minutes=5, note="check whether the oven is off", call_id="w1"):
    return tool_call(SCHEDULE_WAKEUP, {"delay_minutes": minutes, "note": note}, call_id)


def make_runner(engine, notified):
    sessions = SessionManager(InMemorySessionStore(), make_workflow_manager())
    return BackgroundTaskRunner(sessions, notified.append, tick_seconds=1, agent_engine=engine)


@pytest.mark.asyncio
async def test_schedule_wakeup_tool_writes_a_row():
    clock = Clock()
    model = ScriptedChatModel().push(wake_call(5), AIMessage(content="I shall check back in five minutes, sir."))
    engine = make_engine(workflows=[EchoTimeWorkflow()], model=model, clock=clock)
    assert await engine.handle("check the oven in five minutes") == "I shall check back in five minutes, sir."
    wakes = engine.store.list_wakes()
    assert len(wakes) == 1
    assert wakes[0].user_id == "default"
    assert wakes[0].wake_at == clock.now + 300
    assert wakes[0].payload == {"note": "check whether the oven is off"}


@pytest.mark.asyncio
async def test_wakeup_delay_is_clamped():
    clock = Clock()
    model = ScriptedChatModel().push(wake_call(0), AIMessage(content="Right away, sir."))
    engine = make_engine(model=model, clock=clock)
    await engine.handle("remind me now-ish")
    assert engine.store.list_wakes()[0].wake_at == clock.now + 60    # at least one minute


@pytest.mark.asyncio
async def test_due_wake_resumes_the_thread_and_notifies():
    clock = Clock()
    model = ScriptedChatModel().push(
        wake_call(5), AIMessage(content="Noted, sir."),
        AIMessage(content="Five minutes have passed and the oven is off, sir."))
    engine = make_engine(workflows=[EchoTimeWorkflow()], model=model, clock=clock)
    await engine.handle("check the oven in five minutes")

    notified = []
    runner = make_runner(engine, notified)
    await runner._tick()
    assert notified == []                       # not due yet
    assert len(engine.store.list_wakes()) == 1

    clock.now += 301
    await runner._tick()
    assert notified == ["Five minutes have passed and the oven is off, sir."]
    assert engine.store.list_wakes() == []       # consumed
    # The wake arrived on the same thread, after the earlier exchange.
    wake_turn = model.calls[2]
    humans = [m.content for m in wake_turn if isinstance(m, HumanMessage)]
    assert humans[0] == "check the oven in five minutes"
    assert "Scheduled wake-up" in humans[-1] and "oven is off" in humans[-1]

    await runner._tick()                         # idempotent once consumed
    assert len(notified) == 1


@pytest.mark.asyncio
async def test_wake_is_postponed_while_a_confirmation_is_pending():
    clock = Clock()
    specs = {"locks": GateSpec(kind=ActionKind.DEVICE_CONTROL,
                               requires_confirmation=lambda i, e: True,
                               question=lambda i, e: "Shall I unlock, sir?")}
    gate = ActionGate.with_defaults(kill_switch_env="TEST_KILL_SWITCH", audit=AuditLog(":memory:"))
    model = ScriptedChatModel().push(
        wake_call(1), AIMessage(content="Noted."),
        tool_call("locks", {"intent": "unlock", "entities": {"action": "unlock"}}, "c1"),
        AIMessage(content="Unlocked, sir."),
        AIMessage(content="Wake handled, sir."))
    engine = make_engine(workflows=[FakeLockWorkflow()], model=model, gate=gate, gate_specs=specs,
                         clock=clock)
    await engine.handle("wake me in a minute")
    assert await engine.handle("unlock the door") == "Shall I unlock, sir?"

    notified = []
    runner = make_runner(engine, notified)
    clock.now += 120
    await runner._tick()
    assert notified == [] and len(engine.store.list_wakes()) == 1     # postponed, row kept

    assert await engine.handle("yes") == "Unlocked, sir."
    await runner._tick()
    assert notified == ["Wake handled, sir."] and engine.store.list_wakes() == []


@pytest.mark.asyncio
async def test_failed_wake_is_dropped_and_logged():
    clock = Clock()
    model = ScriptedChatModel().push(wake_call(1), AIMessage(content="Noted."), RuntimeError("provider down"))
    engine = make_engine(model=model, clock=clock)
    await engine.handle("wake me")
    notified = []
    runner = make_runner(engine, notified)
    clock.now += 120
    await runner._tick()
    assert notified == [] and engine.store.list_wakes() == []


@pytest.mark.asyncio
async def test_wakes_persist_in_sqlite_store(tmp_path):
    path = str(tmp_path / "ckpt.db")
    clock = Clock()
    store = SqliteAgentStore(path)
    model = ScriptedChatModel().push(wake_call(2), AIMessage(content="Noted."))
    engine = make_engine(model=model, clock=clock, checkpoint_path=path, store=store)
    await engine.handle("wake me in two")
    again = SqliteAgentStore(path)
    assert [w.payload for w in again.list_wakes("default")] == [{"note": "check whether the oven is off"}]
    assert again.due_wakes(clock.now) == []
    assert len(again.due_wakes(clock.now + 121)) == 1
    again.delete_wake(again.list_wakes()[0].wake_id)
    assert again.list_wakes() == []


@pytest.mark.asyncio
async def test_runner_without_engine_is_unchanged():
    notified = []
    sessions = SessionManager(InMemorySessionStore(), make_workflow_manager())
    runner = BackgroundTaskRunner(sessions, notified.append, tick_seconds=1)
    assert runner.agent_engine is None
    await runner._tick()
    assert notified == []


def test_agent_store_epochs():
    store = AgentStore()
    assert store.get_epoch("u") == 0
    assert store.bump_epoch("u") == 1 and store.bump_epoch("u") == 2
    assert store.get_epoch("other") == 0
