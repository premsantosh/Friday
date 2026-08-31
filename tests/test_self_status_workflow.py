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
         json.dumps({"harvest": "ok (1s)",
                     "train": "ok (240s): advanced to v20260826 (43 train ex)",
                     "eval": "ok (30s): facts: 62.5% (n=4), lora: 37.5% (n=4)"})))
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
    # The spoken line carries the train/eval numbers, not just pass/fail,
    # and the raw stage notes stay intact in DATA for the agent.
    assert "advanced to v20260826" in result.message
    assert "62.5%" in result.message
    stages = result.data["status"]["nightly"]["runs"][0]["stages"]
    assert stages["train"] == "ok (240s): advanced to v20260826 (43 train ex)"


def test_tool_description_mentions_data_detail():
    desc = SelfStatusWorkflow().description
    assert "per-stage" in desc and "topic" in desc


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
    assert "RuntimeError: boom" in result.message      # the failing stage's note


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


# --------------------------------------------------------- record topics (v2)

CSV_HEADER = ("date,run_id,arm,opponent,artifact_version,judge,split,"
              "n_prompts,n_decisive,wins,losses,win_rate,p_value,"
              "n_control,control_win_rate,arm_style,opponent_style\n")


@pytest.fixture
def results_dir(tmp_path):
    rdir = tmp_path / "results"
    (rdir / "nightly").mkdir(parents=True)
    (rdir / "nightly" / "20260824.md").write_text("# Nightly 20260824\n"
                                                  "per-category detail here\n")
    (rdir / "eval.csv").write_text(
        CSV_HEADER
        + "20260817,2,memory,base,v20260817,sonnet:x,curated,32,22,9,13,0.409,0.700,6,0.500,0.95,0.95\n"
        + "20260824,3,memory,base,v20260824,sonnet:x,curated,32,24,14,10,0.583,0.041,6,0.500,0.96,0.95\n")
    return rdir


@pytest.mark.asyncio
async def test_evals_topic_reports_win_rate_and_bar(paths, results_dir):
    wf = SelfStatusWorkflow(paths=paths, probes=make_probes(),
                            results_dir=results_dir)
    result = await wf.execute("What were the results of your last eval?", {})
    assert result.data["topic"] == "evals"
    assert "2026-08-24" in result.message
    assert "58%" in result.message
    assert "memory" in result.message
    # The structured records ride along for the agent model to mine.
    series = result.data["evals"]["series"]["memory/curated"]
    assert [e["date"] for e in series] == ["20260817", "20260824"]
    assert result.data["reports"]["reports"][0]["name"] == "20260824.md"


@pytest.mark.asyncio
async def test_history_topic_reports_runs_and_versions(paths, results_dir):
    wf = SelfStatusWorkflow(paths=paths, probes=make_probes(),
                            results_dir=results_dir)
    result = await wf.execute("How many runs have you done over time?", {})
    assert result.data["topic"] == "history"
    assert "1 learning run" in result.message
    assert "lora: 1 version" in result.message
    assert result.data["runs"]["total"] == 1
    assert "lora" in result.data["artifacts"]


@pytest.mark.asyncio
async def test_insights_topic_computes_trend(paths, results_dir):
    wf = SelfStatusWorkflow(paths=paths, probes=make_probes(),
                            results_dir=results_dir)
    result = await wf.execute("How is your training trending?", {})
    assert result.data["topic"] == "insights"
    assert "41% → 58%" in result.message
    assert "improving" in result.message
    assert result.data["evals"]["available"] is True


@pytest.mark.asyncio
async def test_insights_single_eval_is_honest_about_trends(paths, tmp_path):
    rdir = tmp_path / "results-one"
    rdir.mkdir()
    (rdir / "eval.csv").write_text(
        CSV_HEADER
        + "20260824,3,memory,base,v20260824,sonnet:x,curated,32,24,14,10,0.583,0.041,6,0.500,0.96,0.95\n")
    wf = SelfStatusWorkflow(paths=paths, probes=make_probes(), results_dir=rdir)
    result = await wf.execute("any insights on your learning?", {})
    assert "too early" in result.message


@pytest.mark.asyncio
async def test_evals_topic_without_csv_degrades(paths, tmp_path):
    wf = SelfStatusWorkflow(paths=paths, probes=make_probes(),
                            results_dir=tmp_path / "empty-results")
    result = await wf.execute("what were your eval results?", {})
    assert result.status == WorkflowStatus.SUCCESS
    assert "no evaluation results" in result.message
