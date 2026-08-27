"""recall_conversation: keyword + date recall over the owner's transcript."""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from introspection import records
from research.db import ResearchStore
from workflows import RecallConversationWorkflow, WorkflowManager, WorkflowStatus
from workflows.introspection import _extract_keyword, _find_date_phrase

NOW = time.time()
NOW_DT = datetime.fromtimestamp(NOW)


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "research.db"
    store = ResearchStore(str(path))
    yday_start, _ = records.past_date("yesterday", NOW_DT)
    for ts, text, reply in (
            (NOW - 600, "any news on the dentist front", "None yet, sir."),
            (yday_start + 9 * 3600, "book the dentist for thursday",
             "Consider it done, sir."),
            (yday_start + 20 * 3600, "dim the lights", "Done, sir.")):
        store.execute(
            "INSERT INTO exchanges (ts, user_text, reply_text, route, channel)"
            " VALUES (?, ?, ?, 'chat', 'text')", (ts, text, reply))
    store.close()
    return path


def make_workflow(tmp_path, transcript_path=None):
    return RecallConversationWorkflow(
        research_db=transcript_path or tmp_path / "none.db",
        memory_db=tmp_path / "no-memory.db")


@pytest.mark.asyncio
async def test_keyword_recall(tmp_path, transcript):
    wf = make_workflow(tmp_path, transcript)
    result = await wf.execute("When did I last mention the dentist?", {})
    assert result.status == WorkflowStatus.SUCCESS
    assert result.data["total_matches"] == 2
    assert result.data["query"] == "dentist"
    assert "2 exchanges" in result.message
    assert "dentist" in result.message


@pytest.mark.asyncio
async def test_date_recall_filters_to_the_day(tmp_path, transcript):
    wf = make_workflow(tmp_path, transcript)
    result = await wf.execute("What did we talk about yesterday?", {})
    assert result.data["total_matches"] == 2       # the two yesterday rows
    users = [m["user"] for m in result.data["matches"]]
    assert users == ["book the dentist for thursday", "dim the lights"]


@pytest.mark.asyncio
async def test_explicit_entities_win(tmp_path, transcript):
    wf = make_workflow(tmp_path, transcript)
    result = await wf.execute("did we discuss it?",
                              {"query": "lights", "date": "yesterday"})
    assert result.data["total_matches"] == 1
    assert result.data["matches"][0]["user"] == "dim the lights"


@pytest.mark.asyncio
async def test_no_match_is_honest(tmp_path, transcript):
    wf = make_workflow(tmp_path, transcript)
    result = await wf.execute("when did I mention the submarine?", {})
    assert "nothing" in result.message
    assert result.data["total_matches"] == 0


@pytest.mark.asyncio
async def test_missing_records_degrade_to_success(tmp_path):
    wf = make_workflow(tmp_path)
    wf.bind_runtime(ephemeral=True)
    result = await wf.execute("what did we talk about yesterday?", {})
    assert result.status == WorkflowStatus.SUCCESS
    assert "not attached in this mode" in result.message
    assert not (tmp_path / "none.db").exists()


def test_agent_tool_generated_with_data_channel():
    from agent.tools import build_tools

    manager = WorkflowManager()
    manager.register(RecallConversationWorkflow())
    ts = build_tools(manager, gate_specs={})
    tool = next(t for t in ts.tools if t.name == "recall_conversation")
    assert "What did we talk about yesterday?" in tool.description
    assert RecallConversationWorkflow.expose_data_to_agent is True


def test_date_phrase_extraction():
    assert _find_date_phrase("what did we say yesterday?") == "yesterday"
    assert _find_date_phrase("what happened last Tuesday evening") == "last tuesday"
    assert _find_date_phrase("on 2026-08-01 please") == "2026-08-01"
    assert _find_date_phrase("around August 3 I think") == "august 3"
    assert _find_date_phrase("what did we talk about?") is None
    # No fuzzy false positives on ordinary sentences.
    assert _find_date_phrase("tell me about 2 things") is None


def test_keyword_extraction():
    assert _extract_keyword("When did I last mention the dentist?") == "dentist"
    assert _extract_keyword("what did we talk about yesterday") is None
    assert _extract_keyword("did I say anything about the submarine trip") == "submarine"
