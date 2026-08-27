"""The DATA block: opt-in structured data in agent tool messages."""

from __future__ import annotations

import json

import pytest

from agent.tools import DATA_BLOCK_HEADER, build_tools, format_result
from workflows.base import (
    Workflow,
    WorkflowManager,
    WorkflowResult,
    WorkflowStatus,
    WorkflowTrigger,
)


class RecordsWorkflow(Workflow):
    expose_data_to_agent = True
    result = WorkflowResult(status=WorkflowStatus.SUCCESS,
                            message="Here is the trend, sir.",
                            data={"series": [1, 2, 3]})

    @property
    def name(self):
        return "records_wf"

    @property
    def description(self):
        return "test"

    @property
    def trigger(self):
        return WorkflowTrigger(examples=["show records"])

    async def execute(self, intent, entities):
        return self.result


class PlainWorkflow(RecordsWorkflow):
    expose_data_to_agent = False

    @property
    def name(self):
        return "plain_wf"


async def run_tool(workflow):
    manager = WorkflowManager()
    manager.register(workflow)
    ts = build_tools(manager, gate_specs={})
    tool = next(t for t in ts.tools if t.name == workflow.name)
    return await tool.coroutine("show records", {})


@pytest.mark.asyncio
async def test_opted_in_workflow_gets_data_block():
    out = await run_tool(RecordsWorkflow())
    assert out.startswith("Here is the trend, sir.")
    assert DATA_BLOCK_HEADER in out
    payload = out.split(DATA_BLOCK_HEADER + "\n", 1)[1]
    assert json.loads(payload) == {"series": [1, 2, 3]}


@pytest.mark.asyncio
async def test_default_workflow_message_is_unchanged():
    wf = PlainWorkflow()
    out = await run_tool(wf)
    assert out == format_result(wf.result)
    assert DATA_BLOCK_HEADER not in out


@pytest.mark.asyncio
async def test_failure_results_never_carry_data():
    wf = RecordsWorkflow()
    wf.result = WorkflowResult(status=WorkflowStatus.FAILURE,
                               message="broken", error="boom",
                               data={"diagnostics": True})
    out = await run_tool(wf)
    assert out.startswith("ERROR:")
    assert DATA_BLOCK_HEADER not in out


@pytest.mark.asyncio
async def test_empty_data_appends_nothing():
    wf = RecordsWorkflow()
    wf.result = WorkflowResult(status=WorkflowStatus.SUCCESS,
                               message="fine", data=None)
    out = await run_tool(wf)
    assert out == "fine"


@pytest.mark.asyncio
async def test_unserializable_data_degrades_to_plain_message():
    circular: dict = {}
    circular["self"] = circular
    wf = RecordsWorkflow()
    wf.result = WorkflowResult(status=WorkflowStatus.SUCCESS, message="fine",
                               data=circular)
    out = await run_tool(wf)
    assert out == "fine"
