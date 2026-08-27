"""self_status: topic dispatch, graceful degradation, agent tool generation."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from introspection import Paths, Probes
from research.db import ResearchStore
from workflows import SelfStatusWorkflow, WorkflowManager, WorkflowStatus
from workflows.base import create_default_workflow_manager

NOW = time.time()


def make_probes():
    return Probes(
        launchctl=lambda args: SimpleNamespace(
            returncode=0, stdout="1\t0\tcom.friday.nightly\n"),
        http_get=lambda url, timeout: 200,
        now=lambda: NOW,
    )


@pytest.fixture
def paths(tmp_path):
    state_dir = tmp_path / "friday-state"
    art_dir = state_dir / "research"
    art_dir.mkdir(parents=True)
    store = ResearchStore(str(state_dir / "research.db"))
    store.execute(
        "INSERT INTO runs (started_ts, finished_ts, stage_status) VALUES (?, ?, ?)",
        (NOW - 5 * 3600, NOW - 5 * 3600 + 240,
         json.dumps({"harvest": "ok (1s)", "train": "ok (240s)",
                     "eval": "ok (30s)"})))
    store.close()
    lora = art_dir / "lora"
    (lora / "v20260826").mkdir(parents=True)
    (lora / "current").write_text("v20260826\n")
    return Paths(state_dir=state_dir, research_db=state_dir / "research.db",
                 artifacts_dir=art_dir, audit_db=state_dir / "audit.db")


def make_workflow(paths, manager=None):
    return SelfStatusWorkflow(workflow_manager=manager, paths=paths,
                              probes=make_probes())


@pytest.mark.asyncio
async def test_lora_question_answers_from_runs(paths):
    wf = make_workflow(paths)
    result = await wf.execute("Did the LoRA run last night?", {})
    assert result.status == WorkflowStatus.SUCCESS
    assert result.data["topic"] == "nightly"
    assert "completed" in result.message
    assert "v20260826" in result.message


@pytest.mark.asyncio
async def test_failed_run_is_reported(paths):
    store = ResearchStore(str(paths.research_db))
    store.execute(
        "INSERT INTO runs (started_ts, finished_ts, stage_status) VALUES (?, ?, ?)",
        (NOW - 3600, NOW - 3500,
         json.dumps({"harvest": "ok (1s)",
                     "train": "FAILED: RuntimeError: boom"})))
    store.close()
    result = await make_workflow(paths).execute("did your training run last night?", {})
    assert "train" in result.message and "failed" in result.message.lower()


@pytest.mark.asyncio
async def test_health_topic_runs_doctor(paths):
    result = await make_workflow(paths).execute("run a self-diagnosis", {})
    assert result.data["topic"] == "health"
    assert result.data["checks"]
    names = {c["name"] for c in result.data["checks"]}
    assert "nightly.last_run" in names
    assert "checks" in result.message


@pytest.mark.asyncio
async def test_capabilities_topic_lists_registered_workflows(paths):
    manager = create_default_workflow_manager()
    wf = make_workflow(paths, manager=manager)
    manager.register(wf)
    stats_calls = []
    wf.bind_runtime(llm_stats_fn=lambda: stats_calls.append(1) or "Requests: 7",
                    engine_label_fn=lambda: "legacy router")
    result = await wf.execute("what can you do?", {})
    assert result.data["topic"] == "capabilities"
    for name in manager.list_workflows():
        assert name in result.message
    assert stats_calls == [1]
    assert result.data["engine"] == "legacy router"


@pytest.mark.asyncio
async def test_explicit_topic_entity_wins(paths):
    result = await make_workflow(paths).execute("status please", {"topic": "jobs"})
    assert result.data["topic"] == "jobs"
    assert "com.friday.nightly" in result.message


@pytest.mark.asyncio
async def test_missing_records_degrade_to_success(tmp_path):
    root = tmp_path / "empty"
    paths = Paths(state_dir=root, research_db=root / "research.db",
                  artifacts_dir=root / "research", audit_db=root / "audit.db")
    wf = SelfStatusWorkflow(paths=paths, probes=make_probes())
    wf.bind_runtime(ephemeral=True)
    result = await wf.execute("did the lora run last night?", {})
    # SUCCESS, not FAILURE: FAILURE would trigger the synthetic apology LLM call.
    assert result.status == WorkflowStatus.SUCCESS
    assert "not attached in this mode" in result.message
    assert not root.exists()


def test_agent_tool_is_generated_with_standard_schema():
    from agent.tools import build_tools

    manager = WorkflowManager()
    manager.register(SelfStatusWorkflow(workflow_manager=manager))
    ts = build_tools(manager, gate_specs={})
    names = [t.name for t in ts.tools]
    assert "self_status" in names
    tool = next(t for t in ts.tools if t.name == "self_status")
    assert "Did the LoRA run last night?" in tool.description
    assert set(tool.args) == {"intent", "entities"}
