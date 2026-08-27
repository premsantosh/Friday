"""Guard tests: self-awareness must cover future abilities automatically.

These are the contract in introspection/README.md, enforced the same way
test_research_events.py enforces the event taxonomy — if a change breaks
automatic coverage, a test fails rather than Friday quietly going blind.
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

from introspection import (
    CheckResult,
    CheckStatus,
    Paths,
    Probes,
    gather_status,
    run_doctor,
)
from workflows.base import Workflow, WorkflowManager, WorkflowResult, WorkflowStatus, WorkflowTrigger


def make_probes():
    return Probes(
        launchctl=lambda args: SimpleNamespace(returncode=0, stdout=""),
        http_get=lambda url, timeout: 200,
        now=time.time,
    )


def make_paths(tmp_path):
    state_dir = tmp_path / "friday-state"
    (state_dir / "research" / "logs").mkdir(parents=True)
    return Paths(state_dir=state_dir, research_db=state_dir / "research.db",
                 artifacts_dir=state_dir / "research",
                 audit_db=state_dir / "audit.db")


class ToyWorkflow(Workflow):
    """A future ability that opted into the introspection hooks."""

    @property
    def name(self):
        return "toy_ability"

    @property
    def description(self):
        return "A brand-new ability"

    @property
    def trigger(self):
        return WorkflowTrigger(examples=["Do the toy thing"])

    async def execute(self, intent, entities):
        return WorkflowResult(status=WorkflowStatus.SUCCESS, message="done")

    def status_snapshot(self):
        return {"available": True, "queue_depth": 3}

    def health_checks(self):
        return [CheckResult("toy_ability.queue", CheckStatus.WARN, "queue backed up")]


def test_new_database_appears_without_code_changes(tmp_path):
    paths = make_paths(tmp_path)
    sqlite3.connect(paths.state_dir / "brand_new_subsystem.db").close()
    status = gather_status(paths, make_probes())
    assert "brand_new_subsystem.db" in status["storage"]["databases"]
    checks = {r.name for r in run_doctor(paths, make_probes())}
    assert "storage.brand_new_subsystem.db" in checks


def test_new_log_file_appears_without_code_changes(tmp_path):
    paths = make_paths(tmp_path)
    (paths.artifacts_dir / "logs" / "new_channel.log").write_text("x\n")
    status = gather_status(paths, make_probes())
    assert "new_channel.log" in status["storage"]["logs"]


def test_new_artifact_arm_appears_without_code_changes(tmp_path):
    paths = make_paths(tmp_path)
    arm = paths.artifacts_dir / "fourth_arm"
    (arm / "v20260901").mkdir(parents=True)
    (arm / "current").write_text("v20260901\n")
    status = gather_status(paths, make_probes())
    assert "fourth_arm" in status["arms"]["arms"]
    checks = {r.name: r for r in run_doctor(paths, make_probes())}
    assert checks["arms.fourth_arm.pointer"].status is CheckStatus.PASS


def test_registered_workflow_hooks_merge_into_status_and_doctor(tmp_path):
    manager = WorkflowManager()
    manager.register(ToyWorkflow())
    paths = make_paths(tmp_path)

    status = gather_status(paths, make_probes(), workflow_manager=manager)
    assert status["workflows"]["toy_ability"] == {"available": True, "queue_depth": 3}

    checks = {r.name: r for r in run_doctor(paths, make_probes(),
                                            workflow_manager=manager)}
    assert checks["toy_ability.queue"].status is CheckStatus.WARN


def test_workflow_without_hooks_is_silent_but_listed(tmp_path):
    class Plain(ToyWorkflow):
        @property
        def name(self):
            return "plain_ability"

        def status_snapshot(self):
            return None

        def health_checks(self):
            return []

    manager = WorkflowManager()
    manager.register(Plain())
    status = gather_status(make_paths(tmp_path), make_probes(),
                           workflow_manager=manager)
    # No snapshot — but the capability listing (used by self_status) has it.
    assert "plain_ability" not in status["workflows"]
    assert "plain_ability" in manager.list_workflows()
    assert "plain_ability" in manager.get_all_context_for_llm()


def test_crashing_workflow_hook_is_contained(tmp_path):
    class Crashy(ToyWorkflow):
        @property
        def name(self):
            return "crashy"

        def status_snapshot(self):
            raise RuntimeError("boom")

        def health_checks(self):
            raise RuntimeError("boom")

    manager = WorkflowManager()
    manager.register(Crashy())
    paths = make_paths(tmp_path)
    status = gather_status(paths, make_probes(), workflow_manager=manager)
    assert status["workflows"]["crashy"]["available"] is False
    checks = {r.name: r for r in run_doctor(paths, make_probes(),
                                            workflow_manager=manager)}
    assert checks["workflow.crashy"].status is CheckStatus.FAIL


def test_every_registered_production_workflow_is_in_capabilities(monkeypatch):
    """Registering a workflow (the only way an ability enters Friday) must make
    it self-reportable — main.create_workflow_manager with every integration
    enabled, checked against the capability listing self_status reads."""
    monkeypatch.setenv("HUE_BRIDGE_IP", "192.0.2.1")
    monkeypatch.setenv("HASS_TOKEN", "x")
    monkeypatch.setenv("HASS_URL", "http://192.0.2.2:8123")
    monkeypatch.setenv("SHELLY_DEVICES", "plug=192.0.2.3")
    monkeypatch.setenv("COFFEE_MACHINE_IP", "192.0.2.4")
    monkeypatch.setenv("RESERVATIONS_ENABLED", "1")

    from main import create_workflow_manager

    manager = create_workflow_manager()
    listing = manager.get_all_context_for_llm()
    for name in manager.list_workflows():
        assert name in listing
    # The self-awareness workflows themselves are always registered.
    assert "self_status" in manager.list_workflows()
    assert "self_repair" in manager.list_workflows()
