"""self_repair: confirmation gating, harness enforcement, audit trail, revert."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from core.conversation import InMemorySessionStore, SessionManager
from core.conversation.session import SessionStatus, TurnControl
from research.db import ResearchStore
from workflows import SelfRepairWorkflow, WorkflowManager


class KickstartRecorder:
    def __init__(self, returncode=0):
        self.calls = 0
        self.returncode = returncode

    def __call__(self):
        self.calls += 1
        return SimpleNamespace(returncode=self.returncode, stdout="", stderr="")


@pytest.fixture
def audit_db(tmp_path, monkeypatch):
    path = tmp_path / "audit.db"
    monkeypatch.setenv("FRIDAY_AUDIT_DB", str(path))
    return path


def audit_events(path):
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute(
            "SELECT event FROM audit_events ORDER BY id")]
    finally:
        conn.close()


def make_manager(workflow):
    wf_manager = WorkflowManager()
    wf_manager.register(workflow)
    return SessionManager(InMemorySessionStore(), wf_manager,
                         default_timeout_s=600)


@pytest.mark.asyncio
async def test_rerun_asks_for_confirmation_first(audit_db):
    kick = KickstartRecorder()
    wf = SelfRepairWorkflow(launchctl_kickstart=kick)
    sessions = make_manager(wf)
    turn = await sessions.open(wf, "re-run the nightly training", {}, "u")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "learning cycle" in turn.message
    assert kick.calls == 0


@pytest.mark.asyncio
async def test_no_never_executes(audit_db):
    kick = KickstartRecorder()
    wf = SelfRepairWorkflow(launchctl_kickstart=kick)
    sessions = make_manager(wf)
    await sessions.open(wf, "re-run the nightly training", {}, "u")
    turn = await sessions.handle("u", "no")
    assert turn.control == TurnControl.CANCEL
    assert kick.calls == 0
    assert audit_events(audit_db) == []


@pytest.mark.asyncio
async def test_unclear_reasks_instead_of_executing(audit_db):
    kick = KickstartRecorder()
    wf = SelfRepairWorkflow(launchctl_kickstart=kick)
    sessions = make_manager(wf)
    await sessions.open(wf, "re-run the nightly training", {}, "u")
    turn = await sessions.handle("u", "what do you mean")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert kick.calls == 0


@pytest.mark.asyncio
async def test_yes_executes_and_audits(audit_db):
    kick = KickstartRecorder()
    wf = SelfRepairWorkflow(launchctl_kickstart=kick)
    sessions = make_manager(wf)
    await sessions.open(wf, "re-run the nightly training", {}, "u")
    turn = await sessions.handle("u", "yes please")
    assert turn.control == TurnControl.COMPLETE
    assert "starting now" in turn.message
    assert kick.calls == 1
    assert audit_events(audit_db) == ["EXEC_STARTED", "EXEC_OK"]


@pytest.mark.asyncio
async def test_kill_switch_refuses(audit_db, monkeypatch):
    monkeypatch.setenv("FRIDAY_KILL_SWITCH", "1")
    kick = KickstartRecorder()
    wf = SelfRepairWorkflow(launchctl_kickstart=kick)
    sessions = make_manager(wf)
    await sessions.open(wf, "re-run the nightly training", {}, "u")
    turn = await sessions.handle("u", "yes")
    assert "decline" in turn.message.lower()
    assert kick.calls == 0
    assert audit_events(audit_db) == ["GATE_DENY"]


@pytest.mark.asyncio
async def test_failed_kickstart_reports_and_audits_fail(audit_db):
    kick = KickstartRecorder(returncode=3)
    wf = SelfRepairWorkflow(launchctl_kickstart=kick)
    sessions = make_manager(wf)
    await sessions.open(wf, "re-run the nightly training", {}, "u")
    turn = await sessions.handle("u", "yes")
    assert "declined to start" in turn.message
    assert audit_events(audit_db) == ["EXEC_STARTED", "EXEC_FAIL"]


@pytest.mark.asyncio
async def test_ambiguous_request_asks_which_repair(audit_db):
    wf = SelfRepairWorkflow(launchctl_kickstart=KickstartRecorder())
    sessions = make_manager(wf)
    turn = await sessions.open(wf, "repair yourself", {}, "u")
    assert turn.control == TurnControl.CONTINUE
    assert "Which repair" in turn.message
    turn = await sessions.handle("u", "the nightly run please")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION


@pytest.fixture
def artifacts_tree(tmp_path):
    art_dir = tmp_path / "artifacts"
    lora = art_dir / "lora"
    (lora / "v20260801").mkdir(parents=True)
    (lora / "v20260826").mkdir()
    (lora / "current").write_text("v20260826\n")
    return art_dir


@pytest.mark.asyncio
async def test_revert_moves_pointer_and_emits_one_event(audit_db, artifacts_tree,
                                                        tmp_path):
    db_path = str(tmp_path / "research.db")
    ResearchStore(db_path).close()  # pre-create so events land somewhere known
    wf = SelfRepairWorkflow(artifacts_dir=artifacts_tree, db_path=db_path)
    sessions = make_manager(wf)
    turn = await sessions.open(
        wf, "revert the lora model to v20260801", {}, "u")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "v20260801" in turn.message
    turn = await sessions.handle("u", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert (artifacts_tree / "lora" / "current").read_text().strip() == "v20260801"

    store = ResearchStore(db_path)
    events = store.query("SELECT event, detail FROM events WHERE event = 'artifact.advanced'")
    store.close()
    assert len(events) == 1
    assert "self_repair" in events[0]["detail"]

    from research.events import KNOWN_EVENTS
    assert "artifact.advanced" in KNOWN_EVENTS


@pytest.mark.asyncio
async def test_revert_unknown_version_fails_politely(audit_db, artifacts_tree,
                                                     tmp_path):
    db_path = str(tmp_path / "research.db")
    wf = SelfRepairWorkflow(artifacts_dir=artifacts_tree, db_path=db_path)
    sessions = make_manager(wf)
    await sessions.open(wf, "revert the lora model to v19990101", {}, "u")
    turn = await sessions.handle("u", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert "unknown version" in turn.message
    assert (artifacts_tree / "lora" / "current").read_text().strip() == "v20260826"


@pytest.mark.asyncio
async def test_revert_without_version_asks_and_lists(audit_db, artifacts_tree,
                                                     tmp_path):
    wf = SelfRepairWorkflow(artifacts_dir=artifacts_tree,
                            db_path=str(tmp_path / "research.db"))
    sessions = make_manager(wf)
    turn = await sessions.open(wf, "revert the lora model", {}, "u")
    assert turn.control == TurnControl.CONTINUE
    assert "v20260801" in turn.message  # available versions listed
    turn = await sessions.handle("u", "v20260801")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    turn = await sessions.handle("u", "yes")
    assert (artifacts_tree / "lora" / "current").read_text().strip() == "v20260801"


def test_agent_path_exposes_self_repair_only_via_start_task():
    from agent.tools import build_tools

    manager = WorkflowManager()
    manager.register(SelfRepairWorkflow(launchctl_kickstart=KickstartRecorder()))
    sessions = make_manager(manager.get_workflow("self_repair"))
    ts = build_tools(manager, sessions=sessions, gate_specs={})
    names = [t.name for t in ts.tools]
    assert "self_repair" not in names
    assert "start_task" in names
    assert ts.conversational_names == {"self_repair"}
