"""Phase 3 — human-in-the-loop gates: durable interrupts, resume-by-next-turn,
global-escape cancel, soft expiry, kill switch."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.engine import SET_ASIDE_LINE
from agent.store import SqliteAgentStore
from agent.tools import GATE_SPECS, GateSpec
from core.harness import EXEC_OK, GATE_DENY, ActionGate, ActionKind, AuditLog
from tests.agent_fakes import (
    EchoTimeWorkflow,
    FakeLockWorkflow,
    ScriptedChatModel,
    make_assistant,
    make_engine,
    make_workflow_manager,
    tool_call,
)
from workflows.base import WorkflowTrigger, Workflow, WorkflowResult, WorkflowStatus

LOCK_SPECS = {"locks": GateSpec(
    kind=ActionKind.DEVICE_CONTROL,
    requires_confirmation=lambda intent, e: e.get("action") == "unlock",
    question=lambda intent, e: f"Shall I unlock the {e.get('door', 'front')} door, sir?",
)}


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make_gate(audit=None):
    return ActionGate.with_defaults(kill_switch_env="TEST_KILL_SWITCH",
                                    audit=audit or AuditLog(":memory:"))


def unlock_call(door="back", call_id="c1"):
    return tool_call("locks", {"intent": f"unlock the {door} door",
                               "entities": {"action": "unlock", "door": door}}, call_id)


def lock_engine(model, *, audit=None, clock=None, checkpoint_path=None, store=None, wf=None,
                timeout=600.0):
    wf = wf or FakeLockWorkflow()
    engine = make_engine(workflows=[wf], model=model, gate=make_gate(audit), gate_specs=LOCK_SPECS,
                         clock=clock or Clock(), checkpoint_path=checkpoint_path, store=store,
                         confirmation_timeout_s=timeout)
    return engine, wf


def test_default_gate_specs_cover_hass_unlock():
    spec = GATE_SPECS["hass_locks"]
    assert spec.kind == ActionKind.DEVICE_CONTROL
    assert spec.requires_confirmation("unlock the back door", {"action": "unlock"})
    assert spec.requires_confirmation("please unlock the door", {})
    assert not spec.requires_confirmation("lock the back door", {"action": "lock"})
    assert not spec.requires_confirmation("is the garage locked", {"action": "status"})
    assert spec.question("", {"door": "garage"}) == "Shall I unlock the garage door, sir?"
    # Models vary in what they put in the entity; never say "back door door".
    assert spec.question("", {"door": "back door"}) == "Shall I unlock the back door, sir?"
    assert spec.question("", {"door": "The Side Door"}) == "Shall I unlock the side door, sir?"
    assert spec.question("", {}) == "Shall I unlock the front door, sir?"


@pytest.mark.asyncio
async def test_interrupt_surfaces_the_plan_and_holds_the_action():
    model = ScriptedChatModel().push(unlock_call("back"))
    engine, wf = lock_engine(model)
    assert engine.tool_set.gated_names == {"locks"}
    reply = await engine.handle("unlock the back door")
    assert reply == "Shall I unlock the back door, sir?"
    assert wf.calls == []
    assert await engine.has_pending_interrupt("default")


@pytest.mark.asyncio
async def test_yes_executes_exactly_once_through_the_gate():
    audit = AuditLog(":memory:")
    model = ScriptedChatModel().push(unlock_call("back"), AIMessage(content="The back door is open, sir."))
    engine, wf = lock_engine(model, audit=audit)
    await engine.handle("unlock the back door")
    assert await engine.handle("yes") == "The back door is open, sir."
    assert wf.calls == [("unlock the back door", {"action": "unlock", "door": "back"})]
    assert not await engine.has_pending_interrupt("default")
    # The tool result the model saw is the workflow's message; audit shows one clean execution.
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert tool_msgs[-1].content == "The back door is unlocked, sir."
    events = [r["event"] for r in audit._connect().execute(
        "SELECT event FROM audit_events ORDER BY id")]
    assert events.count(EXEC_OK) == 1 and GATE_DENY not in events


@pytest.mark.asyncio
async def test_no_declines_without_an_llm_round_trip_and_thread_stays_usable():
    model = ScriptedChatModel().push(unlock_call("back"), AIMessage(content="Anything else, sir?"))
    engine, wf = lock_engine(model)
    await engine.handle("unlock the back door")
    assert await engine.handle("no") == SET_ASIDE_LINE.format(title="sir")
    assert wf.calls == [] and len(model.calls) == 1
    assert not await engine.has_pending_interrupt("default")
    # Next turn is a fresh, valid conversation (no orphaned tool_use).
    assert await engine.handle("thanks") == "Anything else, sir?"
    msgs = model.calls[1][1:]
    assert [type(m).__name__ for m in msgs] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]
    assert msgs[2].content.startswith("DECLINED:")


@pytest.mark.asyncio
async def test_unclear_answer_reasks_and_never_approves():
    model = ScriptedChatModel().push(unlock_call("side"))
    engine, wf = lock_engine(model)
    await engine.handle("unlock the side door")
    reply = await engine.handle("hmm what was the question")
    assert reply == "I need a yes or a no, sir. Shall I unlock the side door, sir?"
    assert wf.calls == [] and await engine.has_pending_interrupt("default")


@pytest.mark.asyncio
async def test_global_escape_cancels_pending_confirmation_via_process_input():
    model = ScriptedChatModel().push(unlock_call("back"))
    engine, wf = lock_engine(model)
    a = make_assistant(engine=engine)
    assert await a.process_input("unlock the back door") == "Shall I unlock the back door, sir?"
    assert await a.process_input("never mind") == "Very well, sir. I've set that aside."
    assert wf.calls == [] and not await engine.has_pending_interrupt("default")


@pytest.mark.asyncio
async def test_pending_answer_takes_priority_over_keyword_match():
    """'yes please' must answer the question, not fall into Step 1/2/3."""
    class KeywordYes(Workflow):
        def __init__(self):
            self.calls = 0

        @property
        def name(self):
            return "yes_machine"

        @property
        def description(self):
            return "Matches yes"

        @property
        def trigger(self):
            return WorkflowTrigger(keywords=["yes"])

        async def execute(self, intent, entities):
            self.calls += 1
            return WorkflowResult(status=WorkflowStatus.SUCCESS, message="keyword path ran")

    kw = KeywordYes()
    model = ScriptedChatModel().push(unlock_call("back"), AIMessage(content="Done, sir."))
    engine, wf = lock_engine(model)
    a = make_assistant(engine=engine, workflows=make_workflow_manager(kw))
    assert await a.process_input("unlock the back door") == "Shall I unlock the back door, sir?"
    assert await a.process_input("yes please") == "Done, sir."
    assert kw.calls == 0 and len(wf.calls) == 1


@pytest.mark.asyncio
async def test_interrupt_survives_an_engine_restart(tmp_path):
    path = str(tmp_path / "ckpt.db")
    m1 = ScriptedChatModel().push(unlock_call("garage"))
    e1, wf1 = lock_engine(m1, checkpoint_path=path, store=SqliteAgentStore(path))
    assert await e1.handle("unlock the garage") == "Shall I unlock the garage door, sir?"

    # "Process restart": new engine, new model, new workflow instance, same DB.
    m2 = ScriptedChatModel().push(AIMessage(content="Garage open, sir."))
    e2, wf2 = lock_engine(m2, checkpoint_path=path, store=SqliteAgentStore(path))
    assert await e2.has_pending_interrupt("default")
    assert await e2.handle("go ahead") == "Garage open, sir."
    assert wf1.calls == [] and wf2.calls == [("unlock the garage door", {"action": "unlock", "door": "garage"})]


@pytest.mark.asyncio
async def test_expired_confirmation_lapses_gracefully():
    clock = Clock()
    model = ScriptedChatModel().push(unlock_call("back"), AIMessage(content="Good evening, sir."))
    engine, wf = lock_engine(model, clock=clock, timeout=60)
    await engine.handle("unlock the back door")
    clock.now += 61
    reply = await engine.handle("hello again")
    assert reply == "That confirmation lapsed, sir, so I set it aside. Good evening, sir."
    assert wf.calls == [] and not await engine.has_pending_interrupt("default")
    # The fresh turn saw the declined tool result, then the new human message.
    msgs = model.calls[1][1:]
    assert isinstance(msgs[-1], HumanMessage) and msgs[-1].content == "hello again"
    assert any(isinstance(m, ToolMessage) and m.content.startswith("DECLINED:") for m in msgs)


@pytest.mark.asyncio
async def test_kill_switch_blocks_even_after_yes(monkeypatch):
    audit = AuditLog(":memory:")
    model = ScriptedChatModel().push(unlock_call("back"),
                                     AIMessage(content="I'm afraid that is blocked, sir."))
    engine, wf = lock_engine(model, audit=audit)
    await engine.handle("unlock the back door")
    monkeypatch.setenv("TEST_KILL_SWITCH", "1")          # flipped while the question was pending
    assert await engine.handle("yes") == "I'm afraid that is blocked, sir."
    assert wf.calls == []
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert tool_msgs[-1].content.startswith("REFUSED:")
    events = [r["event"] for r in audit._connect().execute("SELECT event FROM audit_events")]
    assert GATE_DENY in events and EXEC_OK not in events


@pytest.mark.asyncio
async def test_ungated_actions_on_a_gated_workflow_pass_straight_through():
    model = ScriptedChatModel().push(
        tool_call("locks", {"intent": "lock the front door", "entities": {"action": "lock", "door": "front"}}),
        AIMessage(content="Secured, sir."))
    engine, wf = lock_engine(model)
    assert await engine.handle("lock the front door") == "Secured, sir."
    assert wf.calls == [("lock the front door", {"action": "lock", "door": "front"})]
    assert not await engine.has_pending_interrupt("default")


@pytest.mark.asyncio
async def test_gated_success_is_never_written_to_the_intent_cache():
    from tests.agent_fakes import RecordingIntentCache
    cache = RecordingIntentCache()
    model = ScriptedChatModel().push(unlock_call("back"), AIMessage(content="Open, sir."))
    wf = FakeLockWorkflow()
    engine = make_engine(workflows=[wf], model=model, gate=make_gate(), gate_specs=LOCK_SPECS,
                         intent_cache=cache, clock=Clock())
    await engine.handle("unlock the back door")
    await engine.handle("yes")
    assert wf.calls and cache.stored == []


@pytest.mark.asyncio
async def test_execution_failure_after_yes_is_reported_not_hidden():
    model = ScriptedChatModel().push(unlock_call("back"),
                                     AIMessage(content="The lock did not respond, sir."))
    engine, wf = lock_engine(model, wf=FakeLockWorkflow(fail=True))
    await engine.handle("unlock the back door")
    assert await engine.handle("yes") == "The lock did not respond, sir."
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert tool_msgs[-1].content.startswith("ERROR: timeout")
