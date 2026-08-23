"""
Tests for the TableCheck channel and the standing-watch workflow integration
(M8/M9): fixture-driven parsing of the real widget API, concrete availability,
the booking flow (cart → GMO token → checkout) with every failure path releasing
the hold, the watch dialogue, the polling tick (notify / dedupe / backoff /
drift-pause), release research + burst, auto-booking from a watch, and
store-mediated control verbs.

Fixtures under tests/fixtures/tablecheck/ are trimmed copies of live responses
captured from the reserve widget's own XHR traffic (2026-08-19).
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
    CommitPlan,
    SchemaDrift,
    TableCheckChannel,
    TableCheckError,
    slug_from_url,
)
from workflows.reservations.models import ReleasePolicy
from workflows.reservations.payment import VirtualCard

from tests.test_reservations import make_test_gate

FIXTURES = Path(__file__).parent / "fixtures" / "tablecheck"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


class FakeFetch:
    """Route-aware fake of the channel's HTTP layer. `routes` maps a path
    fragment → (status, payload) or a callable(method, url, body) → (status,
    payload). Every call is logged as (method, path, body)."""

    def __init__(self, routes=None, calendar=None, status=200):
        self.routes = dict(routes or {})
        self.calls = []
        self.status = status
        if "booking_pages" not in self.routes:
            self.routes["booking_pages"] = (200, fixture("booking_pages"))
        if calendar is not None:
            self.set_calendar(calendar)

    def set_calendar(self, payload, status=200):
        self.routes["/booking/calendar"] = (status, payload)

    def set(self, path, payload, status=200):
        """payload: a JSON body, a (status, body) tuple, or a callable."""
        if callable(payload) or (isinstance(payload, tuple) and len(payload) == 2):
            self.routes[path] = payload
        else:
            self.routes[path] = (status, payload)

    async def __call__(self, method, url, headers, body):
        path = url.split("/v2", 1)[-1]
        self.calls.append((method, path, body))
        # Longest fragment wins ("/booking/checkout_status/" over "/booking/checkout").
        for frag, handler in sorted(self.routes.items(), key=lambda kv: -len(kv[0])):
            if frag in path:
                if callable(handler):
                    return handler(method, url, body)
                return handler
        return 404, {"errors": [{"code": "not_found"}]}


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


class FakePayment:
    def __init__(self):
        self.mints = 0

    def mint_single_use(self, amount_usd=None, memo=""):
        self.mints += 1
        return VirtualCard(token="t", pan="4111111111111111", cvv="123", exp_month="06",
                           exp_year="2031", last_four="1111", spend_limit_usd=10.0)


async def fake_tokenizer(public_key, card, holder):
    assert public_key == "9200008019306" and holder
    return "gmo-token-" + card.last_four


BENFIDDICH_URL = "https://www.tablecheck.com/en/benfiddich-tokyo/reserve"

CLOSED = fixture("calendar_closed")
OPEN = fixture("calendar_open")   # 5th, 9th, 10th bookable; other days full/closed


def calendar_for(open_days, month="2026-09"):
    """A calendar payload with the given ISO dates bookable, the rest full."""
    cal = {}
    for d in range(1, 31):
        iso = f"{month}-{d:02d}"
        cal[iso] = [] if iso in open_days else ["cache_unavailable"]
    return {"calendar": cal}


def tablecheck_decision():
    return ChannelDecision(method=ReservationMethod.TABLECHECK,
                           business_name="Bar BenFiddich", url=BENFIDDICH_URL,
                           address="Shinjuku, Tokyo", confidence=0.9)


def make_watch_env(fetch=None, policy=None, decision=None, payment=None):
    """(manager, workflow, store, fetch, notifier) wired for watch tests."""
    fetch = fetch or FakeFetch(calendar=CLOSED)
    store = InMemorySessionStore()
    notifier = FakeNotifier()
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    wf = ReservationWorkflow(
        discovery=FakeDiscovery(decision or tablecheck_decision(), policy=policy),
        router=ChannelRouter({ReservationMethod.TABLECHECK: channel}),
        notifier=notifier, llm=None, gate=make_test_gate(), session_store=store,
        payment=payment)
    wf._tc_unit_spacing_s = 0   # no politeness sleeps in tests
    wf._tc_burst_inner_s = 0    # one cycle per tick in tests
    manager = WorkflowManager()
    manager.register(wf)
    mgr = SessionManager(store, manager, default_timeout_s=1800)
    return mgr, wf, store, fetch, notifier


def future(days):
    return (date.today() + timedelta(days=days)).isoformat()


async def start_watch(mgr, wf, days_out=30, text=None):
    d = date.today() + timedelta(days=days_out)
    text = text or f"Watch BenFiddich for {d.month}/{d.day} at 7pm or 9pm for 2"
    turn = await mgr.open(wf, text, {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION, turn.message
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND, turn.message
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


def test_parse_calendar_flags():
    state = TableCheckChannel._parse_raw(OPEN, "2026-09")
    assert state["venue_status"] == "ok" and state["month"] == "2026-09"
    assert state["dates"]["2026-09-09"] == {"open": True, "slots": {}}
    assert state["dates"]["2026-09-03"] == {"open": False, "slots": {}}   # cache_unavailable
    assert "2026-09-01" not in state["dates"]                             # closed → absent
    closed = TableCheckChannel._parse_raw(CLOSED, "2026-09")
    assert closed["dates"] == {}


def test_parse_calendar_drift_raises():
    for bad in ({"totally": "different"}, ["not", "an", "object"],
                {"calendar": {"2026-09-09": ["brand_new_flag"]}},
                {"calendar": {"not-a-date": []}},
                {"errors": [{"code": "not_found"}]}):
        with pytest.raises(SchemaDrift):
            TableCheckChannel._parse_raw(bad, "2026-09")


def test_parse_meals_to_venue_local_times():
    times = TableCheckChannel._parse_meals(fixture("meals"), "Asia/Tokyo")
    assert times == {"19:00": "available", "21:00": "full", "23:00": "available"}
    with pytest.raises(SchemaDrift):
        TableCheckChannel._parse_meals({"meals": {"all_day": [{"x": 1}]}}, "Asia/Tokyo")


@pytest.mark.asyncio
async def test_fetch_month_posts_calendar_body():
    fetch = FakeFetch(calendar=OPEN)
    state = await TableCheckChannel(fetcher=fetch).fetch_month("benfiddich-tokyo", 2, "2026-09")
    assert state["month"] == "2026-09"
    method, path, body = next(c for c in fetch.calls if "/booking/calendar" in c[1])
    assert method == "POST" and body["shop_id"] == "benfiddich-tokyo"
    assert body["pax_adult"] == 2
    assert body["start_date"].startswith("2026-09-01T00:00:00.000+09:00")
    assert body["end_date"].startswith("2026-09-30T23:59:59.999+09:00")


@pytest.mark.asyncio
async def test_fetch_month_http_errors():
    fetch = FakeFetch()
    channel = TableCheckChannel(fetcher=fetch)
    fetch.set_calendar(None, status=429)
    with pytest.raises(TableCheckError) as exc:
        await channel.fetch_month("x", 2, "2026-09")
    assert exc.value.status == 429 and not isinstance(exc.value, SchemaDrift)
    fetch.set_calendar(None, status=503)
    with pytest.raises(TableCheckError):
        await channel.fetch_month("x", 2, "2026-09")
    fetch.set_calendar(None, status=404)   # endpoint moved → drift, not retry
    with pytest.raises(SchemaDrift):
        await channel.fetch_month("x", 2, "2026-09")


@pytest.mark.asyncio
async def test_check_availability_concrete():
    fetch = FakeFetch(calendar=OPEN)
    fetch.set("/booking/meals_v2", fixture("meals"))
    channel = TableCheckChannel(fetcher=fetch)
    slots = {"date": "2026-09-09", "time": "19:00", "party_size": 4,
             "channel_decision": {"url": BENFIDDICH_URL}}
    availability = await channel.check_availability(slots)
    assert availability.status == AvailabilityStatus.AVAILABLE
    assert availability.options == ["19:00", "23:00"]

    availability = await channel.check_availability({**slots, "time": "21:00"})
    assert availability.status == AvailabilityStatus.UNAVAILABLE

    availability = await channel.check_availability({**slots, "date": "2026-09-03"})
    assert availability.status == AvailabilityStatus.UNAVAILABLE

    fetch.set_calendar(None, status=503)
    availability = await channel.check_availability(slots)
    assert availability.status == AvailabilityStatus.UNKNOWN


def test_booking_url_params():
    url = TableCheckChannel.booking_url("benfiddich-tokyo", "2026-09-05", "19:00", 2)
    assert url.startswith("https://www.tablecheck.com/en/benfiddich-tokyo/reserve?")
    assert "start_date=2026-09-05" in url and "num_people=2" in url


def test_default_answers_and_service_category_from_venue_settings():
    channel = TableCheckChannel(fetcher=FakeFetch())
    import asyncio
    info = asyncio.run(channel.venue_info("benfiddich-tokyo"))
    assert info["time_zone"] == "Asia/Tokyo" and info["max_num_people"] == 4
    # Only the required question is answered: the "I agree" radio → its option.
    answers = TableCheckChannel._default_answers(info)
    assert answers == [{"question_id": "68e9e28bf96afb2961e68c7b", "is_selected": False,
                        "question_option_id": "68e9e28bf96afb2961e68c7e"}]
    # "counter: 1-2 people" / "table: 2 to 4 people" → party 4 gets the table.
    assert TableCheckChannel._pick_service_category(info, 4, None) == "661e2cf31ec3ef01ef82a0b9"
    assert TableCheckChannel._pick_service_category(info, 1, None) == "661e2ceaeefe0c01f9b2c054"
    assert TableCheckChannel._pick_service_category(info, 1, "661e2cf31ec3ef01ef82a0b9") \
        == "661e2cf31ec3ef01ef82a0b9"


def booking_plan(**overrides):
    details = {"slug": "benfiddich-tokyo", "business_name": "Bar BenFiddich",
               "date": "2026-09-09", "time": "19:00", "party_size": 4,
               "guest_name": "Ada Lovelace", "email": "ada@example.com",
               "phone": "+14155550100", **overrides}
    return CommitPlan(channel="tablecheck", summary="book", details=details,
                      requires_card=True)


def booking_fetch(cart=None, checkout=None):
    fetch = FakeFetch(calendar=OPEN)
    fetch.set("/booking/menu_items", fixture("menu_items"))
    fetch.set("/booking/cart/init", {})
    fetch.set("/booking/cart", cart or (200, fixture("cart_ok")))
    fetch.set("/booking/checkout", checkout or (200, {"reservation": {"slug": "ABCD1234"}}))
    return fetch


@pytest.mark.asyncio
async def test_commit_books_end_to_end_with_card_registration():
    fetch = booking_fetch()
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    card = FakePayment().mint_single_use()
    result = await channel.commit(booking_plan(), payment=card)
    assert result.success and result.confirmation == "ABCD1234", result.message

    cart_call = next(c for c in fetch.calls if c[1] == "/booking/cart" and c[0] == "POST")
    body = cart_call[2]
    assert body["start_at"] == "2026-09-09T19:00:00.000+09:00"
    assert body["pax_adult"] == 4 and body["shop_slug"] == "benfiddich-tokyo"
    assert body["customer"] == {"first_name": "Ada", "last_name": "Lovelace",
                                "is_single_name": False, "email": "ada@example.com",
                                "phone": "+14155550100"}
    # English seat-only item, group order, one of; table service category; agreement answered.
    assert body["orders"] == [{"is_group_order": True, "qty": 1,
                               "menu_item_id": "661e3ba9c1cc2201eb5bbaee", "voucher_ids": []}]
    assert body["service_category_id"] == "661e2cf31ec3ef01ef82a0b9"
    assert body["answers"][0]["question_option_id"] == "68e9e28bf96afb2961e68c7e"

    checkout = next(c for c in fetch.calls if "/booking/checkout/" in c[1])
    assert checkout[0] == "PUT" and checkout[1].endswith("6a853ec2fa551d7161765a81")
    cb = checkout[2]
    assert cb["payment_method"] == "card" and cb["gateway_token"] == "gmo-token-1111"
    assert cb["card_digits"] == "1111" and cb["expiry_year"] == "2031" \
        and cb["expiry_month"] == "06" and cb["card_brand"] == "visa"
    # The card number itself never travels in the API body.
    assert "4111111111111111" not in json.dumps(cb)
    assert not any(c[0] == "DELETE" for c in fetch.calls)   # success keeps the cart


@pytest.mark.asyncio
async def test_commit_reports_gone_slot_and_not_open():
    fetch = booking_fetch(cart=(200, fixture("cart_no_availability")))
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(), payment=FakePayment().mint_single_use())
    assert not result.success and result.error == "slot_not_found"

    not_open = {"availability": {"code": "max_time_cutoff", "type": "max_time_cutoff",
                                 "message": "This venue cannot be reserved after 2026-09-01"},
                "is_request_enabled": False}
    fetch.set("/booking/cart", (200, not_open))
    result = await channel.commit(booking_plan(), payment=FakePayment().mint_single_use())
    assert not result.success and result.error == "not_open"
    assert not any("/booking/checkout" in c[1] for c in fetch.calls)


@pytest.mark.asyncio
async def test_commit_releases_hold_on_failures():
    # Validation errors → cart deleted, manual hand-off with the reason.
    bad = json.loads(json.dumps(fixture("cart_ok")))
    bad["cart"]["validation_errors"] = {"customer": {"phone": ["Phone is invalid"]}}
    fetch = booking_fetch(cart=(200, bad))
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(), payment=FakePayment().mint_single_use())
    assert not result.success and result.error == "validation" and "Phone" in result.message
    assert any(c[0] == "DELETE" and c[1].endswith("6a853ec2fa551d7161765a81") for c in fetch.calls)

    # Card required but none supplied → release, hand off.
    fetch = booking_fetch()
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(), payment=None)
    assert result.error == "card_required" and result.needs_manual
    assert any(c[0] == "DELETE" for c in fetch.calls)

    # A real up-front charge is beyond the cap → refuse, release.
    charge = json.loads(json.dumps(fixture("cart_ok")))
    charge["cart"]["payment_transaction_type"] = "charge"
    charge["cart"]["payment_total_amt"] = "20000.0"
    fetch = booking_fetch(cart=(200, charge))
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(), payment=FakePayment().mint_single_use())
    assert result.error == "over_cap" and any(c[0] == "DELETE" for c in fetch.calls)

    # Checkout error → release; 3-DS redirect → keep the hold, hand the link over.
    fetch = booking_fetch(checkout=(200, {"error": {"code": "card_declined",
                                                    "message": "Card declined"}}))
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(), payment=FakePayment().mint_single_use())
    assert result.error.startswith("checkout_error") and "declined" in result.message.lower()
    assert any(c[0] == "DELETE" for c in fetch.calls)

    fetch = booking_fetch(checkout=(200, {"checkout_status": {
        "status": "redirect_3ds", "redirect_url": "https://acs.example/3ds"}}))
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(), payment=FakePayment().mint_single_use())
    assert result.error == "3ds_pending" and "https://acs.example/3ds" in result.message
    assert not any(c[0] == "DELETE" for c in fetch.calls)


@pytest.mark.asyncio
async def test_commit_polls_pending_checkout():
    polls = {"n": 0}

    def status(method, url, body):
        polls["n"] += 1
        if polls["n"] < 2:
            return 200, {"checkout_status": {"status": "pending"}}
        return 200, {"reservation": {"slug": "ZZ99"}}
    fetch = booking_fetch(checkout=(200, {"checkout_status": {"status": "pending"}}))
    fetch.set("/booking/checkout_status/", status)
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(), payment=FakePayment().mint_single_use())
    assert result.success and result.confirmation == "ZZ99"


@pytest.mark.asyncio
async def test_commit_refuses_incomplete_contact_and_oversize_party():
    fetch = booking_fetch()
    channel = TableCheckChannel(fetcher=fetch, tokenizer=fake_tokenizer)
    result = await channel.commit(booking_plan(email=""), payment=FakePayment().mint_single_use())
    assert result.error == "missing_contact"
    result = await channel.commit(booking_plan(party_size=6), payment=FakePayment().mint_single_use())
    assert result.error == "party_too_large"
    assert not any(c[1] == "/booking/cart" for c in fetch.calls)   # never held anything


# ------------------------------------------------------------- watch dialogue

def test_extract_watch_request_ordinal_list_and_book_verb():
    mgr, wf, store, fetch, notifier = make_watch_env()
    slots = wf._extract_watch_request(
        "Watch https://www.tablecheck.com/en/benfiddich-tokyo/reserve Bar BenFiddich "
        "for September 5th, 9th, 10th or 11th for 4 and book it")
    assert [r[0][-2:] for r in slots["watch_ranges"]] == ["05", "09", "10", "11"]
    assert slots["party_size"] == 4 and slots["watch_autobook"] is True
    assert slots["target_url"].endswith("/benfiddich-tokyo/reserve")
    plain = wf._extract_watch_request("Watch BenFiddich for September 5th for 2")
    assert "watch_autobook" not in plain


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
    assert "autobook" not in ws


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
async def test_tick_baseline_then_notify_on_new_date_with_dedupe():
    # Published-but-full month: the cancellation-watching scenario.
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=calendar_for([])))
    session, iso = await start_watch(mgr, wf, days_out=30)
    notifier.messages.clear()

    session = await tick(mgr, session)          # baseline snapshot, no events
    assert [m for m in notifier.messages if "OPEN" in m] == []

    fetch.set_calendar(calendar_for([iso], month=iso[:7]))
    session = await tick(mgr, session)
    opens = [m for m in notifier.messages if "OPEN x2" in m]
    assert len(opens) == 1
    assert "benfiddich-tokyo" in opens[0] and "book now" in opens[0].lower()

    session = await tick(mgr, session)          # unchanged → no repeat
    assert len([m for m in notifier.messages if "OPEN x2" in m]) == 1

    # close → reopen notifies again
    fetch.set_calendar(calendar_for([], month=iso[:7]))
    session = await tick(mgr, session)
    fetch.set_calendar(calendar_for([iso], month=iso[:7]))
    session = await tick(mgr, session)
    assert len([m for m in notifier.messages if "OPEN x2" in m]) == 2


@pytest.mark.asyncio
async def test_tick_unmatched_date_is_silent():
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=calendar_for([])))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()
    other = (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
    fetch.set_calendar(calendar_for([other], month=iso[:7]))   # wrong date
    await tick(mgr, session)
    assert [m for m in notifier.messages if "OPEN" in m] == []


@pytest.mark.asyncio
async def test_tick_multi_party_criteria_fetch_units():
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=calendar_for([])))
    session, iso = await start_watch(mgr, wf)
    ws = session.slots["watchlist_state"]
    ws["criteria"].append({"date_start": iso, "date_end": iso, "party_size": 4,
                           "times": None, "label": "", "priority": 50,
                           "status": "active"})
    store.save(session)
    fetch.calls.clear()
    await tick(mgr, session)
    cal_calls = [c for c in fetch.calls if "/booking/calendar" in c[1]]
    assert len(cal_calls) == 2
    assert {c[2]["pax_adult"] for c in cal_calls} == {2, 4}


@pytest.mark.asyncio
async def test_tick_backoff_alert_and_recovery():
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=calendar_for([])))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()

    fetch.set_calendar(None, status=503)
    session = await tick(mgr, session)
    session = await tick(mgr, session)
    assert [m for m in notifier.messages if "⚠️" in m] == []
    session = await tick(mgr, session)          # third consecutive failure
    assert session.slots["watchlist_state"]["failures"] == 3
    assert len([m for m in notifier.messages if "⚠️" in m]) == 1
    assert session.wake_at > time.time() + wf._tc_watch_interval_s  # backed off

    fetch.set_calendar(calendar_for([]))
    session = await tick(mgr, session)
    assert session.slots["watchlist_state"]["failures"] == 0


@pytest.mark.asyncio
async def test_tick_schema_drift_pauses_and_alerts():
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=calendar_for([])))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()
    fetch.set_calendar({"unexpected": "shape"})
    session = await tick(mgr, session)
    ws = session.slots["watchlist_state"]
    assert ws["paused"] is True
    assert any("paused" in m.lower() for m in notifier.messages)
    calls_before = len(fetch.calls)
    session = await tick(mgr, session)          # paused → no further fetches
    assert len(fetch.calls) == calls_before


@pytest.mark.asyncio
async def test_calendar_publish_sends_one_summary():
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=CLOSED))
    session, iso = await start_watch(mgr, wf)
    await tick(mgr, session)
    notifier.messages.clear()
    other = (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
    fetch.set_calendar(calendar_for([iso, other], month=iso[:7]))
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
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=CLOSED), policy=policy)
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
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=CLOSED), policy=policy)
    session, iso = await start_watch(mgr, wf, days_out=40)
    notifier.messages.clear()
    session = await tick(mgr, session)
    rel = session.slots["watchlist_state"]["release"]
    assert rel == {"checked": True}
    assert any("couldn't pin down" in m for m in notifier.messages)
    assert session.wake_at >= time.time() + wf._tc_watch_interval_s  # idle, not normal


@pytest.mark.asyncio
async def test_manual_release_policy_sets_burst_at_confirmation():
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=CLOSED))
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
    mgr, wf, store, fetch, notifier = make_watch_env(FakeFetch(calendar=CLOSED), policy=policy)
    session, iso = await start_watch(mgr, wf, days_out=40)
    session = await tick(mgr, session)          # research → idle until drop
    notifier.messages.clear()
    fetch.set_calendar(calendar_for([iso], month=iso[:7]))
    session = await tick(mgr, session)
    assert any("calendar is live" in m for m in notifier.messages)
    # Availability exists now → normal cadence, not parked until the drop.
    assert session.wake_at <= time.time() + wf._tc_watch_interval_s + 60


def test_burst_cadence_floor_inside_window():
    mgr, wf, store, fetch, notifier = make_watch_env()
    now = time.time()
    ws = {"release": {"fire_ts": now - 10, "burst_until": now + 3600}}
    assert wf._watch_in_burst(ws, now)
    wake = wf._watch_next_wake(ws, now, any_open=False)
    assert wf._tc_burst_s <= wake - now <= wf._tc_burst_s + 10
    ws = {"release": {"fire_ts": now + 7200, "burst_until": now + 10800}}
    assert not wf._watch_in_burst(ws, now)


# --------------------------------------------------------------- auto-book

def open_calendar_fetch(iso):
    fetch = FakeFetch(calendar=CLOSED)
    fetch.set("/booking/meals_v2", {"meals": {"all_day": [
        {"t": f"{iso}T10:00:00Z", "a": True, "rr": False},
        {"t": f"{iso}T12:00:00Z", "a": True, "rr": False}]}})
    fetch.set("/booking/menu_items", fixture("menu_items"))
    fetch.set("/booking/cart/init", {})
    fetch.set("/booking/cart", (200, fixture("cart_ok")))
    fetch.set("/booking/checkout", (200, {"reservation": {"slug": "BF2026"}}))
    return fetch


async def start_autobook_watch(mgr, wf, days_out=30):
    d = date.today() + timedelta(days=days_out)
    text = f"Watch BenFiddich for {d.month}/{d.day} for 4 and book it"
    turn = await mgr.open(wf, text, {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION, turn.message
    assert "book it" in turn.message and "Ada Lovelace" in turn.message
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND, turn.message
    session = mgr.store.list_waiting()[0]
    return session, d.isoformat()


@pytest.fixture
def contact_env(monkeypatch):
    monkeypatch.setenv("RESERVATION_GUEST_NAME", "Ada Lovelace")
    monkeypatch.setenv("RESERVATION_USER_EMAIL", "ada@example.com")
    monkeypatch.setenv("RESERVATION_USER_PHONE", "+14155550100")


@pytest.mark.asyncio
async def test_autobook_watch_books_when_a_date_opens(contact_env, monkeypatch):
    d = date.today() + timedelta(days=30)
    iso = d.isoformat()
    payment = FakePayment()
    mgr, wf, store, fetch, notifier = make_watch_env(open_calendar_fetch(iso), payment=payment)
    monkeypatch.setattr(wf, "_watch_contact_facts", lambda merged: {
        "guest_name": "Ada Lovelace", "email": "ada@example.com", "phone": "+14155550100"})
    session, iso = await start_autobook_watch(mgr, wf)
    ws = session.slots["watchlist_state"]
    assert ws["autobook"]["plan"]["details"]["party_size"] == 4
    approvals = session.slots["_harness"]["approvals"]
    assert {a["kind"] for a in approvals} == {"book", "mint_card"}
    assert all(a["max_uses"] > 1 for a in approvals)

    session = await tick(mgr, session)          # closed → baseline
    notifier.messages.clear()
    fetch.set_calendar(calendar_for([iso], month=iso[:7]))
    session = await tick(mgr, session)

    assert session.status == SessionStatus.DONE
    assert session.slots["booking_result"] == {"success": True, "confirmation": "BF2026"}
    texts = "\n".join(notifier.messages)
    assert "booking now" in texts and "Done" in texts and "BF2026" in texts
    assert payment.mints == 1
    cart_body = next(c[2] for c in fetch.calls if c[1] == "/booking/cart" and c[0] == "POST")
    assert cart_body["start_at"].startswith(f"{iso}T19:00")   # earliest seating first
    assert cart_body["customer"]["email"] == "ada@example.com"


@pytest.mark.asyncio
async def test_autobook_tries_next_seating_when_first_is_taken(contact_env, monkeypatch):
    d = date.today() + timedelta(days=30)
    iso = d.isoformat()
    fetch = open_calendar_fetch(iso)
    carts = {"n": 0}

    def cart(method, url, body):
        carts["n"] += 1
        if carts["n"] == 1:
            return 200, fixture("cart_no_availability")     # 19:00 gone
        return 200, fixture("cart_ok")                       # 21:00 fine
    fetch.set("/booking/cart", cart)
    mgr, wf, store, fetch, notifier = make_watch_env(fetch, payment=FakePayment())
    monkeypatch.setattr(wf, "_watch_contact_facts", lambda merged: {
        "guest_name": "Ada Lovelace", "email": "ada@example.com", "phone": "+14155550100"})
    session, iso = await start_autobook_watch(mgr, wf)
    await tick(mgr, session)
    fetch.set_calendar(calendar_for([iso], month=iso[:7]))
    session = await tick(mgr, session)
    assert session.status == SessionStatus.DONE
    bodies = [c[2] for c in fetch.calls if c[1] == "/booking/cart" and c[0] == "POST"]
    assert [b["start_at"][11:16] for b in bodies] == ["19:00", "21:00"]
    assert any("slipped away" in m for m in notifier.messages)


@pytest.mark.asyncio
async def test_autobook_pauses_on_hard_failure_and_resumes_on_request(contact_env, monkeypatch):
    d = date.today() + timedelta(days=30)
    iso = d.isoformat()
    fetch = open_calendar_fetch(iso)
    fetch.set("/booking/checkout", (200, {"error": {"code": "card_declined",
                                                    "message": "Card declined"}}))
    mgr, wf, store, fetch, notifier = make_watch_env(fetch, payment=FakePayment())
    monkeypatch.setattr(wf, "_watch_contact_facts", lambda merged: {
        "guest_name": "Ada Lovelace", "email": "ada@example.com", "phone": "+14155550100"})
    session, iso = await start_autobook_watch(mgr, wf)
    await tick(mgr, session)
    fetch.set_calendar(calendar_for([iso], month=iso[:7]))
    session = await tick(mgr, session)
    ws = session.slots["watchlist_state"]
    assert session.status != SessionStatus.DONE
    assert ws["autobook"]["paused"].startswith("checkout_error")
    assert any("paused auto-booking" in m for m in notifier.messages)
    assert any(c[0] == "DELETE" for c in fetch.calls)          # hold released
    checkouts = len([c for c in fetch.calls if "/booking/checkout/" in c[1]])
    session = await tick(mgr, session)                          # still open, but paused
    assert len([c for c in fetch.calls if "/booking/checkout/" in c[1]]) == checkouts

    turn = await mgr.open(wf, "Any luck with the BenFiddich watch?", {}, "default")
    assert "auto-booking paused" in turn.message
    fetch.set("/booking/checkout", (200, {"reservation": {"slug": "OK1"}}))
    turn = await mgr.open(wf, "resume booking", {}, "default")
    assert "back on" in turn.message
    session = await tick(mgr, store.get(session.session_id))
    assert session.status == SessionStatus.DONE


class NoCard:
    """A card service that refuses (e.g. spent single-use card, API down)."""
    def mint_single_use(self, amount_usd=None, memo=""):
        return None


@pytest.mark.asyncio
async def test_autobook_without_card_hands_off(contact_env, monkeypatch):
    d = date.today() + timedelta(days=30)
    iso = d.isoformat()
    mgr, wf, store, fetch, notifier = make_watch_env(open_calendar_fetch(iso), payment=NoCard())
    monkeypatch.setattr(wf, "_watch_contact_facts", lambda merged: {
        "guest_name": "Ada Lovelace", "email": "ada@example.com", "phone": "+14155550100"})
    session, iso = await start_autobook_watch(mgr, wf)
    await tick(mgr, session)
    fetch.set_calendar(calendar_for([iso], month=iso[:7]))
    session = await tick(mgr, session)
    ws = session.slots["watchlist_state"]
    assert ws["autobook"]["paused"] == "card"
    assert any("couldn't set up the card" in m for m in notifier.messages)
    assert not any(c[1] == "/booking/cart" and c[0] == "POST" for c in fetch.calls)


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
    turn = await mgr.open(wf, "resume booking", {}, "default")
    assert "no auto-booking watch" in turn.message


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
