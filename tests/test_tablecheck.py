"""
Tests for the TableCheck channel and the standing-watch workflow integration
(M8): fixture-driven parsing, concrete availability, the watch dialogue, the
polling tick (notify / dedupe / backoff / drift-pause), release-policy
research with self-scheduled burst, and store-mediated control verbs.
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from core.conversation import InMemorySessionStore, SessionManager, TurnControl
from core.conversation.session import SessionStatus
from workflows.base import WorkflowManager
from workflows.reservations import (
    ChannelDecision,
    ChannelRouter,
    ReservationMethod,
    ReservationWorkflow,
)
from workflows.reservations.channels import (
    AvailabilityStatus,
    SchemaDrift,
    TableCheckChannel,
    TableCheckError,
    slug_from_url,
)
from workflows.reservations.models import ReleasePolicy

from tests.test_reservations import make_test_gate

FIXTURES = Path(__file__).parent / "fixtures" / "tablecheck"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


class FakeFetch:
    """Programmable fetcher: one payload/status at a time, call log kept."""

    def __init__(self, payload=None, status=200):
        self.payload, self.status = payload, status
        self.calls = []

    def set(self, payload, status=200):
        self.payload, self.status = payload, status

    async def __call__(self, url, headers):
        self.calls.append(url)
        return self.status, self.payload


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return True


class FakeDiscovery:
    """Skips search plumbing: a fixed ChannelDecision + optional release policy."""

    def __init__(self, decision, policy=None):
        self._decision, self._policy = decision, policy
        self.policy_lookups = 0

    def discover(self, business_name, location=None, target_url=None, kind="dining"):
        return self._decision

    def resolve_release_policy(self, business_name, location=None):
        self.policy_lookups += 1
        return self._policy


BENFIDDICH_URL = "https://www.tablecheck.com/en/benfiddich-tokyo/reserve"


def tablecheck_decision():
    return ChannelDecision(method=ReservationMethod.TABLECHECK,
                           business_name="Bar BenFiddich", url=BENFIDDICH_URL,
                           address="Shinjuku, Tokyo", confidence=0.9)


def make_watch_env(fetch=None, policy=None, decision=None):
    """(manager, workflow, store, fetch, notifier) wired for watch tests."""
    fetch = fetch or FakeFetch(fixture("month_open"))
    store = InMemorySessionStore()
    notifier = FakeNotifier()
    channel = TableCheckChannel(fetcher=fetch)
    wf = ReservationWorkflow(
        discovery=FakeDiscovery(decision or tablecheck_decision(), policy=policy),
        router=ChannelRouter({ReservationMethod.TABLECHECK: channel}),
        notifier=notifier, llm=None, gate=make_test_gate(), session_store=store)
    wf._tc_unit_spacing_s = 0   # no politeness sleeps in tests
    manager = WorkflowManager()
    manager.register(wf)
    mgr = SessionManager(store, manager, default_timeout_s=1800)
    return mgr, wf, store, fetch, notifier


def future(days):
    return (date.today() + timedelta(days=days)).isoformat()


async def start_watch(mgr, wf, days_out=30):
    d = date.today() + timedelta(days=days_out)
    text = (f"Watch BenFiddich for {d.month}/{d.day} at 7pm or 9pm for 2")
    turn = await mgr.open(wf, text, {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION, turn.message
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND
    session = mgr.store.list_waiting()[0]
    assert session.slots["watchlist_state"]["slug"] == "benfiddich-tokyo"
    return session, d.isoformat()


async def tick(mgr, session):
    """Force the session due and run one background cycle."""
    session.wake_at = 0
    mgr.store.save(session)
    await mgr.tick_waiting()
    return mgr.store.get(session.session_id)


# ------------------------------------------------------------------- channel

def test_slug_from_url_variants():
    assert slug_from_url(BENFIDDICH_URL) == "benfiddich-tokyo"
    assert slug_from_url("https://www.tablecheck.com/ja/benfiddich-tokyo") == "benfiddich-tokyo"
    assert slug_from_url("https://www.tablecheck.com/shops/benfiddich-tokyo/reserve") \
        == "benfiddich-tokyo"
    assert slug_from_url("https://www.opentable.com/lazy-bear") is None
    assert slug_from_url(None) is None


def test_parse_open_month_fixture():
    state = TableCheckChannel._parse_raw(fixture("month_open"), "2026-09")
    assert state["venue_status"] == "ok"
    assert state["dates"]["2026-09-05"] == {
        "open": True, "slots": {"19:00": "available", "21:00": "available"}}
    assert state["dates"]["2026-09-04"] == {"open": False, "slots": {}}


def test_parse_closed_and_halted_fixtures():
    closed = TableCheckChannel._parse_raw(fixture("month_closed"), "2026-09")
    assert closed["dates"] == {} and closed["venue_status"] == "ok"
    halted = TableCheckChannel._parse_raw(fixture("halted"), "2026-09")
    assert halted["venue_status"] == "halted"


def test_parse_alternate_shapes():
    days_list = {"available_days": [
        {"date": "2026-09-05", "times": ["7:00 PM"]},
        {"date": "2026-09-06", "available": False}]}
    state = TableCheckChannel._parse_raw(days_list, "2026-09")
    assert state["dates"]["2026-09-05"]["slots"] == {"19:00": "available"}
    assert state["dates"]["2026-09-06"] == {"open": False, "slots": {}}

    slot_map = {"availability": {"2026-09-05": {"19:00": "available", "21:00": "full"}}}
    state = TableCheckChannel._parse_raw(slot_map, "2026-09")
    assert state["dates"]["2026-09-05"]["slots"] == {"19:00": "available", "21:00": "full"}
    assert state["dates"]["2026-09-05"]["open"] is True


def test_parse_drift_raises():
    with pytest.raises(SchemaDrift):
        TableCheckChannel._parse_raw({"totally": "different"}, "2026-09")
    with pytest.raises(SchemaDrift):
        TableCheckChannel._parse_raw(["not", "an", "object"], "2026-09")


@pytest.mark.asyncio
async def test_fetch_month_builds_unit_url():
    fetch = FakeFetch(fixture("month_open"))
    state = await TableCheckChannel(fetcher=fetch).fetch_month("benfiddich-tokyo", 2, "2026-09")
    assert state["month"] == "2026-09"
    url = fetch.calls[0]
    assert "benfiddich-tokyo" in url and "num_people=2" in url
    assert "start_date=2026-09-01" in url and "end_date=2026-09-30" in url


@pytest.mark.asyncio
async def test_fetch_month_http_errors():
    fetch = FakeFetch(None, status=429)
    channel = TableCheckChannel(fetcher=fetch)
    with pytest.raises(TableCheckError) as exc:
        await channel.fetch_month("x", 2, "2026-09")
    assert exc.value.status == 429 and not isinstance(exc.value, SchemaDrift)
    fetch.set(None, status=503)
    with pytest.raises(TableCheckError):
        await channel.fetch_month("x", 2, "2026-09")
    fetch.set(None, status=404)   # endpoint moved → drift, not retry
    with pytest.raises(SchemaDrift):
        await channel.fetch_month("x", 2, "2026-09")


@pytest.mark.asyncio
async def test_check_availability_concrete():
    channel = TableCheckChannel(fetcher=FakeFetch(fixture("month_open")))
    slots = {"date": "2026-09-05", "time": "19:00", "party_size": 2,
             "channel_decision": {"url": BENFIDDICH_URL}}
    availability = await channel.check_availability(slots)
    assert availability.status == AvailabilityStatus.AVAILABLE
    assert availability.options == ["19:00", "21:00"]

    availability = await channel.check_availability({**slots, "date": "2026-09-04"})
    assert availability.status == AvailabilityStatus.UNAVAILABLE

    channel = TableCheckChannel(fetcher=FakeFetch(fixture("halted")))
    availability = await channel.check_availability(slots)
    assert availability.status == AvailabilityStatus.UNKNOWN


def test_booking_url_params():
    url = TableCheckChannel.booking_url("benfiddich-tokyo", "2026-09-05", "19:00", 2)
    assert url.startswith("https://www.tablecheck.com/en/benfiddich-tokyo/reserve?")
    assert "start_date=2026-09-05" in url and "num_people=2" in url


@pytest.mark.asyncio
async def test_commit_is_manual_handoff():
    channel = TableCheckChannel(fetcher=FakeFetch(fixture("month_open")))
    plan = await channel.prepare(
        {"date": "2026-09-05", "time": "19:00", "party_size": 2,
         "channel_decision": {"url": BENFIDDICH_URL}}, tablecheck_decision())
    result = await channel.commit(plan)
    assert not result.success and result.needs_manual
    assert "benfiddich-tokyo" in (plan.details.get("url") or "")


# ------------------------------------------------------------- watch dialogue

@pytest.mark.asyncio
async def test_watch_gather_confirm_and_start():
    mgr, wf, store, fetch, notifier = make_watch_env()
    session, iso = await start_watch(mgr, wf)
    ws = session.slots["watchlist_state"]
    assert ws["criteria"] == [{
        "date_start": iso, "date_end": iso, "party_size": 2,
        "times": ["19:00", "21:00"], "label": "", "priority": 100,
        "status": "active"}]
    assert session.fsm_state == "watching_list"


@pytest.mark.asyncio
async def test_watch_asks_for_missing_dates_then_party():
    mgr, wf, store, fetch, notifier = make_watch_env()
    turn = await mgr.open(wf, "Keep an eye on BenFiddich for a table", {}, "default")
    assert turn.control == TurnControl.CONTINUE and "dates" in turn.message.lower()
    d = date.today() + timedelta(days=20)
    turn = await mgr.handle("default", f"{d.month}/{d.day}")
    assert "how many" in turn.message.lower()
    turn = await mgr.handle("default", "two of us")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    turn = await mgr.handle("default", "no")
    assert turn.control == TurnControl.CANCEL


@pytest.mark.asyncio
async def test_watch_declined_for_non_tablecheck_venue():
    decision = ChannelDecision(method=ReservationMethod.OPENTABLE,
                               business_name="Lazy Bear",
                               url="https://www.opentable.com/lazy-bear")
    mgr, wf, store, fetch, notifier = make_watch_env(decision=decision)
    d = date.today() + timedelta(days=20)
    turn = await mgr.open(wf, f"Watch Lazy Bear for {d.month}/{d.day} for 2", {}, "default")
    assert turn.control == TurnControl.COMPLETE
    assert "TableCheck" in turn.message


# ------------------------------------------------------------------ the tick

@pytest.mark.asyncio
async def test_tick_baseline_then_notify_on_new_slot_with_dedupe():
    # Published-but-full month: the cancellation-watching scenario.
    full = {"availability_calendar": {}, "shop": {"status": "open"}}
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(full))
    session, iso = await start_watch(mgr, wf)
    full["availability_calendar"] = {iso: []}
    notifier.messages.clear()

    session = await tick(mgr, session)          # baseline snapshot, no events
    assert [m for m in notifier.messages if "OPEN" in m] == []

    fetch.set({"availability_calendar": {iso: ["19:00"]}, "shop": {"status": "open"}})
    session = await tick(mgr, session)
    opens = [m for m in notifier.messages if "OPEN x2" in m]
    assert len(opens) == 1
    assert "benfiddich-tokyo" in opens[0] and "book now" in opens[0].lower()

    session = await tick(mgr, session)          # unchanged → no repeat
    assert len([m for m in notifier.messages if "OPEN x2" in m]) == 1

    # close → reopen notifies again
    fetch.set({"availability_calendar": {iso: []}, "shop": {"status": "open"}})
    session = await tick(mgr, session)
    fetch.set({"availability_calendar": {iso: ["19:00"]}, "shop": {"status": "open"}})
    session = await tick(mgr, session)
    assert len([m for m in notifier.messages if "OPEN x2" in m]) == 2


@pytest.mark.asyncio
async def test_tick_unmatched_slot_is_silent():
    payload = {"availability_calendar": {}, "shop": {"status": "open"}}
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(payload))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()
    other = (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
    fetch.set({"availability_calendar": {iso: ["17:00"], other: ["19:00"]},
               "shop": {"status": "open"}})   # wrong time / wrong date
    await tick(mgr, session)
    assert [m for m in notifier.messages if "OPEN" in m] == []


@pytest.mark.asyncio
async def test_tick_multi_party_criteria_fetch_units():
    mgr, wf, store, fetch, notifier = make_watch_env(
        FakeFetch({"availability_calendar": {}, "shop": {"status": "open"}}))
    session, iso = await start_watch(mgr, wf)
    ws = session.slots["watchlist_state"]
    ws["criteria"].append({"date_start": iso, "date_end": iso, "party_size": 4,
                           "times": None, "label": "", "priority": 50,
                           "status": "active"})
    store.save(session)
    fetch.calls.clear()
    await tick(mgr, session)
    assert len(fetch.calls) == 2
    assert any("num_people=2" in u for u in fetch.calls)
    assert any("num_people=4" in u for u in fetch.calls)


@pytest.mark.asyncio
async def test_tick_backoff_alert_and_recovery():
    mgr, wf, store, fetch, notifier = make_watch_env(
        FakeFetch({"availability_calendar": {}, "shop": {"status": "open"}}))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()

    fetch.set(None, status=503)
    session = await tick(mgr, session)
    session = await tick(mgr, session)
    assert [m for m in notifier.messages if "⚠️" in m] == []
    session = await tick(mgr, session)          # third consecutive failure
    assert session.slots["watchlist_state"]["failures"] == 3
    assert len([m for m in notifier.messages if "⚠️" in m]) == 1
    assert session.wake_at > time.time() + wf._tc_watch_interval_s  # backed off

    fetch.set({"availability_calendar": {}, "shop": {"status": "open"}})
    session = await tick(mgr, session)
    assert session.slots["watchlist_state"]["failures"] == 0


@pytest.mark.asyncio
async def test_tick_schema_drift_pauses_and_alerts():
    mgr, wf, store, fetch, notifier = make_watch_env(
        FakeFetch({"availability_calendar": {}, "shop": {"status": "open"}}))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()
    fetch.set({"unexpected": "shape"})
    session = await tick(mgr, session)
    ws = session.slots["watchlist_state"]
    assert ws["paused"] is True
    assert any("paused" in m.lower() for m in notifier.messages)
    calls_before = len(fetch.calls)
    session = await tick(mgr, session)          # paused → no further fetches
    assert len(fetch.calls) == calls_before


@pytest.mark.asyncio
async def test_tick_halted_and_resumed_notify_once():
    mgr, wf, store, fetch, notifier = make_watch_env(
        FakeFetch({"availability_calendar": {}, "shop": {"status": "open"}}))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()
    fetch.set(fixture("halted"))
    session = await tick(mgr, session)
    session = await tick(mgr, session)
    assert len([m for m in notifier.messages if "halted" in m]) == 1
    fetch.set({"availability_calendar": {}, "shop": {"status": "open"}})
    await tick(mgr, session)
    assert any("again" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_calendar_publish_sends_one_summary():
    payload = {"availability_calendar": {}, "shop": {"status": "open"}}
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(payload))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()
    other = (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
    fetch.set({"availability_calendar": {iso: ["19:00", "21:00"], other: ["19:00"]},
               "shop": {"status": "open"}})
    session = await tick(mgr, session)
    cal = [m for m in notifier.messages if "calendar is live" in m]
    assert len(cal) == 1
    assert [m for m in notifier.messages if "OPEN x2" in m] == []  # summary covers it
    session = await tick(mgr, session)          # no repeat, no late per-slot spam
    assert len([m for m in notifier.messages if "calendar is live" in m]) == 1
    assert [m for m in notifier.messages if "OPEN x2" in m] == []


# ------------------------------------- release-policy research + burst window

@pytest.mark.asyncio
async def test_closed_calendar_triggers_research_and_burst_schedule():
    policy = ReleasePolicy(days_in_advance=25, release_time="9am", timezone="JST",
                           confidence=0.9, source_quote="reservations open 25 days out at 9am")
    mgr, wf, store, fetch, notifier = make_watch_env(
        FakeFetch({"availability_calendar": {}, "shop": {"status": "open"}}),
        policy=policy)
    session, iso = await start_watch(mgr, wf, days_out=40)
    notifier.messages.clear()
    session = await tick(mgr, session)
    ws = session.slots["watchlist_state"]
    rel = ws["release"]
    assert rel["checked"] and rel["source"] == "web"
    assert rel["fire_ts"] > time.time()
    assert rel["burst_until"] == rel["fire_ts"] + wf._tc_burst_tail_s
    assert any("reservations open" in m for m in notifier.messages)
    assert wf.discovery.policy_lookups == 1
    # Idle cadence until the drop, waking no later than the burst start.
    assert session.wake_at <= rel["fire_ts"] - wf._tc_burst_s + 1
    session = await tick(mgr, session)
    assert wf.discovery.policy_lookups == 1     # research runs once per watch


@pytest.mark.asyncio
async def test_research_low_confidence_stays_idle():
    policy = ReleasePolicy(days_in_advance=25, release_time="9am", timezone="JST",
                           confidence=0.2)
    mgr, wf, store, fetch, notifier = make_watch_env(
        FakeFetch({"availability_calendar": {}, "shop": {"status": "open"}}),
        policy=policy)
    session, iso = await start_watch(mgr, wf, days_out=40)
    notifier.messages.clear()
    session = await tick(mgr, session)
    rel = session.slots["watchlist_state"]["release"]
    assert rel == {"checked": True}
    assert any("couldn't pin down" in m for m in notifier.messages)
    assert session.wake_at >= time.time() + wf._tc_watch_interval_s  # idle, not normal


@pytest.mark.asyncio
async def test_manual_release_policy_sets_burst_at_confirmation():
    mgr, wf, store, fetch, notifier = make_watch_env(
        FakeFetch({"availability_calendar": {}, "shop": {"status": "open"}}))
    d = date.today() + timedelta(days=40)
    turn = await mgr.open(
        wf, f"Watch BenFiddich for {d.month}/{d.day} at 9pm for 2 — "
            f"they open 25 days in advance at 9am JST", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND
    assert "Reservations open" in turn.message
    ws = mgr.store.list_waiting()[0].slots["watchlist_state"]
    assert ws["release"]["source"] == "you" and ws["release"]["fire_ts"] > time.time()
    # Already known → the closed-calendar tick never re-researches.
    session = await tick(mgr, mgr.store.list_waiting()[0])
    assert wf.discovery.policy_lookups == 0


@pytest.mark.asyncio
async def test_calendar_publish_before_predicted_drop_beats_the_burst():
    policy = ReleasePolicy(days_in_advance=25, release_time="9am", timezone="JST",
                           confidence=0.9)
    payload = {"availability_calendar": {}, "shop": {"status": "open"}}
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(payload), policy=policy)
    session, iso = await start_watch(mgr, wf, days_out=40)
    session = await tick(mgr, session)          # research → idle until drop
    notifier.messages.clear()
    fetch.set({"availability_calendar": {iso: ["21:00"]}, "shop": {"status": "open"}})
    session = await tick(mgr, session)
    assert any("calendar is live" in m for m in notifier.messages)
    # Availability exists now → normal cadence, not parked until the drop.
    assert session.wake_at <= time.time() + wf._tc_watch_interval_s + 60


# ----------------------------------------------------------- control verbs

@pytest.mark.asyncio
async def test_stop_watching_closes_the_watch():
    mgr, wf, store, fetch, notifier = make_watch_env()
    session, iso = await start_watch(mgr, wf)
    turn = await mgr.open(wf, "Stop watching BenFiddich", {}, "default")
    assert turn.control == TurnControl.COMPLETE and "called off" in turn.message
    session = await tick(mgr, session)
    assert session.status == SessionStatus.DONE


@pytest.mark.asyncio
async def test_fulfilled_marks_and_quietly_closes():
    mgr, wf, store, fetch, notifier = make_watch_env()
    session, iso = await start_watch(mgr, wf)
    turn = await mgr.open(wf, "We got the BenFiddich table!", {}, "default")
    assert "Splendid" in turn.message
    session = await tick(mgr, session)
    assert session.status == SessionStatus.DONE
    assert all("run its course" not in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_status_reports_standing_watches():
    mgr, wf, store, fetch, notifier = make_watch_env()
    await start_watch(mgr, wf)
    turn = await mgr.open(wf, "Any luck with the BenFiddich watch?", {}, "default")
    assert turn.control == TurnControl.COMPLETE
    assert "Bar BenFiddich" in turn.message


@pytest.mark.asyncio
async def test_control_verbs_without_watches_degrade_gracefully():
    mgr, wf, store, fetch, notifier = make_watch_env()
    turn = await mgr.open(wf, "Stop watching BenFiddich", {}, "default")
    assert "not watching anything" in turn.message
    turn = await mgr.open(wf, "Any luck with the watch?", {}, "default")
    assert "No standing watches" in turn.message


@pytest.mark.asyncio
async def test_second_ask_folds_into_existing_watch():
    mgr, wf, store, fetch, notifier = make_watch_env()
    session, iso = await start_watch(mgr, wf)
    d2 = date.today() + timedelta(days=35)
    turn = await mgr.open(
        wf, f"Also watch BenFiddich for {d2.month}/{d2.day} for 4", {}, "default")
    assert turn.control == TurnControl.COMPLETE and "added" in turn.message.lower()
    ws = store.get(session.session_id).slots["watchlist_state"]
    assert len(ws["criteria"]) == 2
    assert ws["criteria"][1]["party_size"] == 4
    assert len(store.list_waiting()) == 1       # still one watch session


@pytest.mark.asyncio
async def test_expired_criteria_end_the_watch_with_notice():
    mgr, wf, store, fetch, notifier = make_watch_env()
    session, iso = await start_watch(mgr, wf)
    ws = session.slots["watchlist_state"]
    past = (date.today() - timedelta(days=2)).isoformat()
    for c in ws["criteria"]:
        c["date_start"] = c["date_end"] = past
    store.save(session)
    session = await tick(mgr, session)
    assert session.status == SessionStatus.DONE
    assert any("Watch ended" in m for m in notifier.messages)
