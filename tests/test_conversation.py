"""
Tests for the multi-turn agent framework (MT1 + MT2).

MT1 — ConversationContext (turnstile-ctx adapter): enrich/update continuity.
MT2 — Session core: slot-filling, confirmation gate, escapes, expiry, persistence.
"""

import time

import pytest

from core.conversation import (
    BackgroundTaskRunner,
    ConversationContext,
    InMemorySessionStore,
    Session,
    SessionManager,
    SessionStatus,
    SqliteSessionStore,
    TurnControl,
    TurnResult,
)
from workflows.base import ConversationalWorkflow, WorkflowManager, WorkflowTrigger


# --------------------------------------------------------------------------- fixtures

class FakeReminderWorkflow(ConversationalWorkflow):
    """A 2-slot conversational workflow: collect task → collect time → confirm."""

    session_timeout_s = 600

    @property
    def name(self) -> str:
        return "reminder"

    @property
    def description(self) -> str:
        return "Set a reminder."

    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(keywords=["remind", "reminder"], examples=["remind me to..."])

    async def start(self, intent, entities, session):
        return TurnResult.ask("What should I remind you about, sir?", next_state="collect_task")

    async def resume(self, text, session):
        if session.fsm_state == "collect_task":
            return TurnResult.ask("And when?", slots_update={"task": text}, next_state="collect_time")
        if session.fsm_state == "collect_time":
            return TurnResult.confirm(
                f"Remind you to {session.slots['task']} at {text}?",
                slots_update={"when": text},
                next_state="confirm",
            )
        if session.fsm_state == "confirm":
            if text.strip().lower() in ("yes", "y", "correct", "yep"):
                return TurnResult.complete("Consider it done, sir.")
            return TurnResult.cancel("As you wish, sir.")
        return TurnResult.complete("")


def make_manager():
    wf_manager = WorkflowManager()
    wf_manager.register(FakeReminderWorkflow())
    return SessionManager(InMemorySessionStore(), wf_manager, default_timeout_s=600)


class FakePollingWorkflow(ConversationalWorkflow):
    """Goes straight to WAITING; completes after N ticks. Models wait-and-book."""

    session_timeout_s = 600

    def __init__(self, ticks_until_done: int = 2):
        self.ticks_until_done = ticks_until_done

    @property
    def name(self) -> str:
        return "poller"

    @property
    def description(self) -> str:
        return "Polls for something."

    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(keywords=["poll"], examples=["poll for x"])

    async def start(self, intent, entities, session):
        return TurnResult.background("I'll keep an eye on that, sir.",
                                     wake_at=0, slots_update={"ticks": 0})

    async def resume(self, text, session):
        return TurnResult.complete("")

    async def on_tick(self, session):
        ticks = session.slots.get("ticks", 0) + 1
        if ticks >= self.ticks_until_done:
            return TurnResult.complete("It's ready, sir.", slots_update={"ticks": ticks})
        return TurnResult.background("still waiting", wake_at=0, slots_update={"ticks": ticks})


def make_poller_manager(ticks_until_done: int = 2):
    wf_manager = WorkflowManager()
    wf_manager.register(FakePollingWorkflow(ticks_until_done))
    return SessionManager(InMemorySessionStore(), wf_manager, default_timeout_s=600)


# ----------------------------------------------------------------- MT1: context

def test_context_starts_empty_and_enriches_after_update():
    ctx = ConversationContext(persist_path=None)
    assert ctx.enabled  # turnstile-ctx installed

    # No context yet → utterance unchanged.
    assert ctx.enrich("turn it down") == "turn it down"

    # After a successful route, the next turn inherits the domain.
    ctx.update("shelly", {"device": "desk lamp"})
    enriched = ctx.enrich("turn it down")
    assert enriched != "turn it down"
    assert "context:" in enriched and "domain=shelly" in enriched


def test_context_clear_resets():
    ctx = ConversationContext(persist_path=None)
    ctx.update("weather", {})
    ctx.clear()
    assert ctx.enrich("and tomorrow?") == "and tomorrow?"


# ----------------------------------------------------------------- MT2: sessions

@pytest.mark.asyncio
async def test_slot_filling_through_to_completion():
    mgr = make_manager()
    wf = mgr.workflows.workflows["reminder"]

    turn = await mgr.open(wf, "remind me", {}, "default")
    assert turn.control == TurnControl.CONTINUE
    assert mgr.has_active("default")

    turn = await mgr.handle("default", "buy milk")
    assert "when" in turn.message.lower()
    assert mgr.get_active("default").slots["task"] == "buy milk"

    turn = await mgr.handle("default", "5pm")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert mgr.get_active("default").status == SessionStatus.AWAITING_CONFIRMATION

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert not mgr.has_active("default")  # session closed


@pytest.mark.asyncio
async def test_confirmation_no_cancels():
    mgr = make_manager()
    wf = mgr.workflows.workflows["reminder"]
    await mgr.open(wf, "remind me", {}, "default")
    await mgr.handle("default", "call mum")
    await mgr.handle("default", "tomorrow")
    turn = await mgr.handle("default", "no")
    assert turn.control == TurnControl.CANCEL
    assert not mgr.has_active("default")


@pytest.mark.asyncio
async def test_global_escape_detection_and_cancel():
    mgr = make_manager()
    wf = mgr.workflows.workflows["reminder"]
    await mgr.open(wf, "remind me", {}, "default")
    assert mgr.has_active("default")

    assert mgr.is_global_escape("cancel")
    assert mgr.is_global_escape("Never mind!")
    assert not mgr.is_global_escape("buy milk")

    mgr.cancel("default", "user aborted")
    assert not mgr.has_active("default")


@pytest.mark.asyncio
async def test_session_expiry():
    mgr = make_manager()
    wf = mgr.workflows.workflows["reminder"]
    await mgr.open(wf, "remind me", {}, "default")

    session = mgr.store.get_active_for_user("default")
    session.expires_at = time.time() - 1  # force expiry
    mgr.store.save(session)

    assert mgr.get_active("default") is None
    assert mgr.store.get(session.session_id).status == SessionStatus.EXPIRED


def test_sqlite_store_roundtrip(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "sessions.db"))
    s = Session.new("default", "reminder", timeout_s=600)
    s.slots = {"task": "buy milk", "when": "5pm"}
    s.fsm_state = "confirm"
    store.save(s)

    loaded = store.get(s.session_id)
    assert loaded is not None
    assert loaded.slots == {"task": "buy milk", "when": "5pm"}
    assert loaded.fsm_state == "confirm"
    assert loaded.status == SessionStatus.ACTIVE
    assert store.get_active_for_user("default").session_id == s.session_id


# ----------------------------------------------------------------- MT3: background

@pytest.mark.asyncio
async def test_background_promotes_to_waiting_then_completes():
    mgr = make_poller_manager(ticks_until_done=2)
    wf = mgr.workflows.workflows["poller"]

    turn = await mgr.open(wf, "poll for x", {}, "default")
    assert turn.control == TurnControl.BACKGROUND
    # WAITING is not the "active dialogue" — it won't hijack the next command.
    assert not mgr.has_active("default")
    assert len(mgr.store.list_waiting()) == 1

    # First tick: advanced but still waiting (re-scheduled BACKGROUND).
    advanced = await mgr.tick_waiting()
    assert len(advanced) == 1
    assert advanced[0][1].control == TurnControl.BACKGROUND
    assert len(mgr.store.list_waiting()) == 1

    # Second tick: completes.
    advanced = await mgr.tick_waiting()
    assert len(advanced) == 1
    _session, result = advanced[0]
    assert result.control == TurnControl.COMPLETE
    assert result.message == "It's ready, sir."
    assert len(mgr.store.list_waiting()) == 0


@pytest.mark.asyncio
async def test_tick_respects_wake_at():
    mgr = make_poller_manager(ticks_until_done=1)
    wf = mgr.workflows.workflows["poller"]
    await mgr.open(wf, "poll", {}, "default")

    # Push wake_at into the future → tick should skip it.
    session = mgr.store.list_waiting()[0]
    session.wake_at = time.time() + 3600
    mgr.store.save(session)

    assert await mgr.tick_waiting() == []
    assert len(mgr.store.list_waiting()) == 1


@pytest.mark.asyncio
async def test_sweep_expired_closes_abandoned_dialogue():
    mgr = make_manager()
    wf = mgr.workflows.workflows["reminder"]
    await mgr.open(wf, "remind me", {}, "default")  # ACTIVE

    session = mgr.store.get_active_for_user("default")
    session.expires_at = time.time() - 1
    mgr.store.save(session)

    swept = mgr.sweep_expired()
    assert len(swept) == 1
    assert mgr.store.get(session.session_id).status == SessionStatus.EXPIRED


def test_background_runner_thread_notifies(tmp_path):
    """End-to-end: the runner thread ticks a WAITING session and fires the notifier."""
    mgr = make_poller_manager(ticks_until_done=1)
    wf = mgr.workflows.workflows["poller"]

    import asyncio
    asyncio.run(mgr.open(wf, "poll", {}, "default"))

    messages = []
    runner = BackgroundTaskRunner(mgr, notify=messages.append, tick_seconds=1)
    runner.start()
    try:
        deadline = time.time() + 5
        while not messages and time.time() < deadline:
            time.sleep(0.05)
    finally:
        runner.stop()

    assert messages == ["It's ready, sir."]
    assert len(mgr.store.list_waiting()) == 0


def test_sqlite_store_lists_waiting(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "sessions.db"))
    waiting = Session.new("default", "reservations", timeout_s=600)
    waiting.status = SessionStatus.WAITING
    store.save(waiting)
    active = Session.new("default", "reminder", timeout_s=600)
    store.save(active)

    waiting_ids = {s.session_id for s in store.list_waiting()}
    assert waiting.session_id in waiting_ids
    assert active.session_id not in waiting_ids
