"""Step-1 intent-cache hardening in core/assistant.py: volatile-entity
dropping, gated-workflow purge, and the legacy writeback guard."""

from __future__ import annotations

import pytest

from llm.router import RouteResult
from tests.agent_fakes import (
    EchoTimeWorkflow,
    FakeLockWorkflow,
    RecordingIntentCache,
    make_assistant,
    make_workflow_manager,
)


@pytest.mark.asyncio
async def test_cache_hit_drops_volatile_entities_but_keeps_the_rest():
    wf = EchoTimeWorkflow()
    cache = RecordingIntentCache(
        hit=("time_check", {"zone": "local", "date": "2026-08-29", "time": "19:00"}))
    a = make_assistant(workflows=make_workflow_manager(wf), intent_cache=cache)

    await a.process_input("the time please")
    assert a.last_route == "cache:time_check"
    assert wf.calls == [("the time please", {"zone": "local"})]  # date/time gone


@pytest.mark.asyncio
async def test_cached_gated_workflow_is_purged_and_never_executed():
    wf = FakeLockWorkflow(name="hass_locks")
    cache = RecordingIntentCache(hit=("hass_locks", {"action": "unlock"}))
    a = make_assistant(workflows=make_workflow_manager(wf), intent_cache=cache)

    reply = await a.process_input("unlock the back door")
    assert wf.calls == []                          # the gate was not bypassed
    assert cache.deleted == ["hass_locks"]         # poisoned entry self-healed
    assert reply == "legacy: unlock the back door"  # fell through to normal routing


class _GatedRouter:
    def route(self, text, workflow_manager):
        return RouteResult(workflow_name="hass_locks", entities={"action": "unlock"})


@pytest.mark.asyncio
async def test_legacy_writeback_never_caches_gated_workflows():
    wf = FakeLockWorkflow(name="hass_locks")
    cache = RecordingIntentCache(hit=None)
    a = make_assistant(workflows=make_workflow_manager(wf), intent_cache=cache,
                       router=_GatedRouter())

    await a.process_input("unlock the back door")
    assert wf.calls, "legacy route should still execute the workflow"
    assert cache.stored == []                      # but never cache it
