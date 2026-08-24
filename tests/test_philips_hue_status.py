"""Tests for the Hue status action and the state-query guard.

Regression for a live incident: "Can you check if the lights are on" keyword-
matched the lights workflow with empty entities, fell through to the default
toggle, and turned on every light in the house. State questions must be
read-only.
"""

from __future__ import annotations

import pytest

from workflows.base import WorkflowStatus
from workflows.philips_hue import PhilipsHueLightsWorkflow


class FakeHueClient:
    def __init__(self):
        self._discovered = True
        self._all_lights_group_id = "gl-all"
        self._light_names = {"desk lamp": "light-3"}
        self._group_names = {"bedroom": "gl-bedroom"}
        self.mutations: list = []
        self.lights = [
            {"id": "light-1", "metadata": {"name": "Kitchen"}, "on": {"on": True}},
            {"id": "light-2", "metadata": {"name": "Hall"}, "on": {"on": False}},
            {"id": "light-3", "metadata": {"name": "Desk Lamp"}, "on": {"on": False}},
        ]
        self.grouped = [
            {"id": "gl-all", "on": {"on": True}},
            {"id": "gl-bedroom", "on": {"on": False}},
        ]

    async def get_lights(self):
        return self.lights

    async def get_grouped_lights(self):
        return self.grouped

    async def set_group_action(self, gid, state):
        self.mutations.append(("group", gid, state))
        return []

    async def set_light_state(self, lid, state):
        self.mutations.append(("light", lid, state))
        return []

    def find_group_id(self, name):
        return self._group_names.get(name.lower())

    def find_light_id(self, name):
        return self._light_names.get(name.lower())


@pytest.fixture
def workflow():
    return PhilipsHueLightsWorkflow(client=FakeHueClient())


# ------------------------------------------------------------ query detection

@pytest.mark.parametrize("text", [
    "Can you check if the lights are on",
    "are the lights on",
    "Is the bedroom lamp off?",
    "what lights are on right now",
    "lights status",
    "tell me if the lights are on",
])
def test_state_queries_detected(text):
    assert PhilipsHueLightsWorkflow._is_state_query(text)


@pytest.mark.parametrize("text", [
    "turn on the lights",
    "can you turn on the lights",
    "lights off please",
    "dim the bedroom lights to 50",
])
def test_commands_not_detected_as_queries(text):
    assert not PhilipsHueLightsWorkflow._is_state_query(text)


# -------------------------------------------------------------- status action

@pytest.mark.asyncio
async def test_state_question_reports_and_never_mutates(workflow):
    result = await workflow.execute("Can you check if the lights are on", {})
    assert result.status == WorkflowStatus.SUCCESS
    assert workflow.client.mutations == []          # THE regression assertion
    assert "1 of 3 lights are on" in result.message
    assert "Kitchen" in result.message


@pytest.mark.asyncio
async def test_status_all_off_and_all_on(workflow):
    for light in workflow.client.lights:
        light["on"]["on"] = False
    result = await workflow.execute("are the lights on", {})
    assert "All 3 lights are off" in result.message

    for light in workflow.client.lights:
        light["on"]["on"] = True
    result = await workflow.execute("are the lights on", {})
    assert "All 3 lights are on" in result.message
    assert workflow.client.mutations == []


@pytest.mark.asyncio
async def test_status_for_specific_room_and_light(workflow):
    result = await workflow.execute("is it on?", {"action": "status", "room": "bedroom"})
    assert result.message == "The bedroom lights are off, sir."

    result = await workflow.execute("is it on?", {"action": "status", "room": "desk lamp"})
    assert result.message == "The desk lamp light is off, sir."
    assert workflow.client.mutations == []


@pytest.mark.asyncio
async def test_explicit_action_entity_overrides_query_guard(workflow):
    # The router explicitly said turn on — a trailing "?" must not veto it.
    result = await workflow.execute("could you turn the lights on?", {"action": "on"})
    assert result.status == WorkflowStatus.SUCCESS
    assert workflow.client.mutations != []


@pytest.mark.asyncio
async def test_plain_command_still_mutates(workflow):
    result = await workflow.execute("turn on the lights", {})
    assert result.status == WorkflowStatus.SUCCESS
    assert workflow.client.mutations == [("group", "gl-all", {"on": {"on": True}})]
