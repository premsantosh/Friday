"""
Tests for the Reservations agent — M1 (discovery + multi-turn collection, no commits).
"""

import pytest

from core.conversation import InMemorySessionStore, SessionManager, TurnControl
from core.harness import ActionGate, AuditLog
from search.provider import SearchResult
from workflows.base import WorkflowManager
from workflows.reservations import (
    BusinessDiscovery,
    ChannelDecision,
    ChannelRouter,
    ReservationMethod,
    ReservationWorkflow,
)
from workflows.reservations.channels import (
    Availability,
    AvailabilityStatus,
    BookingResult,
    CommitPlan,
    OpenTableChannel,
    ReservationChannel,
)
from workflows.reservations.calendar import CalendarService
from workflows.reservations.notify import TelegramNotifier
from workflows.reservations.payment import (
    ManualCardService,
    PrivacyCardService,
    VirtualCard,
    card_service_from_env,
)

from datetime import datetime


# --------------------------------------------------------------------------- fakes

class FakeSearch:
    def __init__(self, results):
        self._results = results

    def search(self, query, max_results=5, include_domains=None):
        return self._results


class FakeYelp:
    def __init__(self, business):
        self._business = business

    def match(self, name, location=""):
        return self._business


def opentable_result():
    return [SearchResult(title="Lazy Bear | OpenTable",
                         snippet="Book a table at Lazy Bear on OpenTable.",
                         url="https://www.opentable.com/lazy-bear")]


def yelp_business():
    return {
        "name": "Lazy Bear",
        "phone": "+14155551234",
        "url": "https://www.yelp.com/biz/lazy-bear",
        "location": {"display_address": ["3416 19th St", "San Francisco, CA 94110"]},
    }


class FakeChannel(ReservationChannel):
    """A bookable channel with controllable availability/commit, for gate tests."""
    method = ReservationMethod.OPENTABLE
    can_commit = True

    def __init__(self, status=AvailabilityStatus.UNKNOWN, result=None, requires_card=False):
        self._status = status
        self._result = result
        self._requires_card = requires_card
        self.committed = False
        self.payment_received = "unset"

    async def check_availability(self, slots):
        return Availability(status=self._status)

    async def prepare(self, slots, decision):
        return CommitPlan(
            channel="opentable",
            summary="book it",
            details={"business_name": decision.business_name or slots.get("business_name"),
                     "party_size": slots.get("party_size"), "date": slots.get("date"),
                     "time": slots.get("time")},
            requires_card=self._requires_card or decision.requires_card_hint,
        )

    async def commit(self, plan, payment=None):
        self.committed = True
        self.payment_received = payment
        return self._result or BookingResult(success=True, message="Booked, sir.",
                                             confirmation="ABC123")


class FakePayment:
    """Stand-in for PrivacyCardService (no network)."""
    def __init__(self, card="default"):
        self._card = card
        self.minted = False

    def mint_single_use(self, amount_usd=None, memo=""):
        self.minted = True
        if self._card == "default":
            return VirtualCard(token="t_123", pan="4111111111111111", cvv="123",
                               exp_month="12", exp_year="2030", last_four="1111",
                               spend_limit_usd=10.0)
        return self._card  # None to simulate a mint failure


def card_hint_result():
    return [SearchResult(title="Lazy Bear | OpenTable",
                         snippet="A credit card is required to hold your reservation.",
                         url="https://www.opentable.com/lazy-bear")]


def make_test_gate():
    """Real gate + policies, but an in-memory audit DB so tests leave no files."""
    from workflows.reservations.workflow import RESERVATION_MACHINE
    return ActionGate.with_defaults(kill_switch_env="RESERVATION_KILL_SWITCH",
                                    audit=AuditLog(":memory:"),
                                    gate_states=RESERVATION_MACHINE.gate_states)


def make_reservation_manager(discovery, router=None):
    wf_manager = WorkflowManager()
    wf_manager.register(ReservationWorkflow(discovery=discovery, router=router, llm=None,
                                            gate=make_test_gate()))
    return SessionManager(InMemorySessionStore(), wf_manager, default_timeout_s=1800)


def opentable_router(channel):
    return ChannelRouter({ReservationMethod.OPENTABLE: channel})


# ----------------------------------------------------------------- discovery

def test_discovery_classifies_opentable_from_link():
    disc = BusinessDiscovery(search_provider=FakeSearch(opentable_result()),
                             yelp_client=FakeYelp(yelp_business()))
    decision = disc.discover("Lazy Bear")
    assert decision.method == ReservationMethod.OPENTABLE
    assert decision.business_name == "Lazy Bear"
    assert "opentable.com" in decision.url
    assert decision.phone == "+14155551234"
    assert "San Francisco" in decision.address


class RecordingSearch:
    """A search provider that records each call (and returns fixed results)."""
    def __init__(self, results):
        self._results = results
        self.queries = []
        self.calls = []

    def search(self, query, max_results=5, include_domains=None):
        self.queries.append(query)
        self.calls.append({"query": query, "include_domains": include_domains})
        return self._results


def test_discovery_search_includes_location():
    """Regression: without the city in the query, common names resolve to the
    wrong place (e.g. "Flores" -> a New York venue instead of San Francisco)."""
    search = RecordingSearch(opentable_result())
    disc = BusinessDiscovery(search_provider=search, yelp_client=FakeYelp(yelp_business()))
    disc.discover("Flores", "San Francisco")
    assert search.queries, "discovery should have run a web search"
    assert "San Francisco" in search.queries[0]
    assert "Flores" in search.queries[0]


def test_discovery_falls_back_to_phone():
    # Yelp gives a phone, but no booking link is found → reserve by phone.
    disc = BusinessDiscovery(search_provider=FakeSearch([]), yelp_client=FakeYelp(yelp_business()))
    decision = disc.discover("Lazy Bear")
    assert decision.method == ReservationMethod.PHONE
    assert decision.phone == "+14155551234"


def test_discovery_unknown_when_nothing_found():
    disc = BusinessDiscovery(search_provider=FakeSearch([]), yelp_client=FakeYelp(None))
    decision = disc.discover("Nowhere Cafe")
    assert decision.method == ReservationMethod.UNKNOWN


def test_discovery_detects_card_hint_and_appointment_platform():
    results = [SearchResult(title="Fellow Barber booking",
                            snippet="Book online. A credit card is required to hold your slot.",
                            url="https://www.vagaro.com/fellowbarber")]
    disc = BusinessDiscovery(search_provider=FakeSearch(results), yelp_client=FakeYelp(None))
    decision = disc.discover("Fellow Barber")
    assert decision.method == ReservationMethod.GENERIC_WEB
    assert decision.requires_card_hint is True


# ----------------------------------------------------- discovery via Google Places

from workflows.reservations import GooglePlacesClient, YelpClient  # noqa: E402


def google_place():
    return {
        "displayName": {"text": "Lazy Bear"},
        "formattedAddress": "3416 19th St, San Francisco, CA 94110",
        "internationalPhoneNumber": "+1 415-555-1234",
        "websiteUri": "https://www.lazybearsf.com",
        "regularOpeningHours": {
            # Google: day 0 = Sunday. Tue 17:00–22:00 and Sun 17:00–02:00 (overnight).
            "periods": [
                {"open": {"day": 2, "hour": 17, "minute": 0},
                 "close": {"day": 2, "hour": 22, "minute": 0}},
                {"open": {"day": 0, "hour": 17, "minute": 0},
                 "close": {"day": 1, "hour": 2, "minute": 0}},
            ]
        },
    }


def test_google_places_client_normalizes_to_resolver_shape():
    captured = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"places": [google_place()]}

    def poster(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return Resp()

    biz = GooglePlacesClient("g-key", poster=poster).match("Lazy Bear", "San Francisco")

    assert captured["url"].endswith("places:searchText")
    assert captured["headers"]["X-Goog-Api-Key"] == "g-key"
    assert "places.regularOpeningHours" in captured["headers"]["X-Goog-FieldMask"]
    assert captured["body"]["textQuery"] == "Lazy Bear San Francisco"

    assert biz["name"] == "Lazy Bear"
    assert biz["phone"] == "+14155551234"                  # normalized for dialing
    assert biz["url"] == "https://www.lazybearsf.com"
    assert "San Francisco" in biz["location"]["display_address"][0]
    # Day indices converted to Yelp convention (0 = Monday): Tue → 1, Sun → 6.
    opens = biz["hours"][0]["open"]
    assert {"day": 1, "start": "1700", "end": "2200"} in opens
    assert {"day": 6, "start": "1700", "end": "0200"} in opens


def test_google_places_client_returns_none_on_failure():
    def poster(*a, **k):
        raise RuntimeError("network down")

    assert GooglePlacesClient("g-key", poster=poster).match("Lazy Bear") is None


def test_discovery_prefers_google_over_yelp(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "g")
    monkeypatch.setenv("YELP_API_KEY", "y")
    assert isinstance(BusinessDiscovery.from_env().business, GooglePlacesClient)

    monkeypatch.delenv("GOOGLE_PLACES_API_KEY")
    assert isinstance(BusinessDiscovery.from_env().business, YelpClient)

    monkeypatch.delenv("YELP_API_KEY")
    assert BusinessDiscovery.from_env().business is None


def test_discovery_carries_google_hours_into_decision_and_phone_plan():
    # No booking link found + a phone number → PHONE method, with the resolver's
    # hours flowing through to the channel decision (off-hours deferral input).
    normalized = GooglePlacesClient._normalize(google_place())

    class FakeGoogle:
        source_name = "google_places"

        def match(self, name, location=""):
            return normalized

    disc = BusinessDiscovery(FakeSearch([]), FakeGoogle())
    decision = disc.discover("Lazy Bear")
    assert decision.method == ReservationMethod.PHONE
    assert decision.phone == "+14155551234"
    assert decision.hours == normalized["hours"]
    assert decision.source == "google_places"
    # And the hours survive the slots round-trip (restart safety).
    assert decision.to_dict()["hours"] == normalized["hours"]

    # The phone channel reads them: Tuesday 6pm → open; Tuesday 3pm → deferred
    # to the 17:00 opening (Yelp-day 1 = Tuesday).
    ch = PhoneChannel(client=None)
    open_now, _ = ch.is_open_now({"hours": decision.hours}, datetime(2026, 6, 16, 18, 0))
    assert open_now is True
    closed, next_open = ch.is_open_now({"hours": decision.hours}, datetime(2026, 6, 16, 15, 0))
    assert closed is False
    assert datetime.fromtimestamp(next_open).hour == 17


# ----------------------------------------------------------------- extraction

def test_extract_slots_from_full_request():
    wf = ReservationWorkflow(discovery=BusinessDiscovery(FakeSearch([]), FakeYelp(None)), llm=None)
    slots = wf._extract_slots("book a table for 2 at Lazy Bear next friday at 7pm")
    assert slots["party_size"] == 2
    assert slots["business_name"].lower() == "lazy bear"
    # Values are canonicalized at extraction (harness normalize): 24h time,
    # ISO date resolved against now, with the verbatim phrasing kept in *_raw.
    assert slots["time"] == "19:00"
    parsed = datetime.fromisoformat(slots["date"])
    assert parsed.weekday() == 4                      # a Friday
    assert parsed.date() > datetime.now().date() - __import__("datetime").timedelta(days=1)
    assert slots["date_raw"].lower() == "next friday"


# ----------------------------------------------------------------- workflow flow

@pytest.mark.asyncio
async def test_full_request_awaits_confirmation_then_books():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear next friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "Lazy Bear" in turn.message
    assert "book it" in turn.message.lower()
    assert fake.committed is False  # nothing booked before approval

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert fake.committed is True
    assert "Booked" in turn.message
    assert not mgr.has_active("default")


def test_merge_entities_normalizes_and_passes_through():
    wf = ReservationWorkflow(discovery=BusinessDiscovery(FakeSearch([]), FakeYelp(None)),
                             llm=None, gate=make_test_gate())
    merged = wf._merge_entities({"party_size": "2", "time": "7pm",
                                 "location": "San Francisco", "junk": "ignored"})
    assert merged["party_size"] == 2          # essentials canonicalized
    assert merged["time"] == "19:00"
    assert merged["location"] == "San Francisco"  # free-text passed through
    assert "junk" not in merged
    assert wf._merge_entities(None) == {}


@pytest.mark.asyncio
async def test_followup_reservation_inherits_recent_details():
    """A quick follow-up that names only a new place reuses the date/time/party
    from the request just before it, instead of re-asking for everything."""
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    # First request supplies the WHEN and HOW MANY, then we walk away (cancel).
    await mgr.open(wf, "book a table for 2 at Lazy Bear next friday at 7pm", {}, "default")
    await mgr.handle("default", "no")
    assert not mgr.has_active("default")

    # Follow-up names only a new place — no date/time/party in the message.
    turn = await mgr.open(wf, "actually make a reservation at Flores instead", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION  # didn't re-ask for the details
    assert "7:00 pm" in turn.message.lower() or "7:00pm" in turn.message.lower()


@pytest.mark.asyncio
async def test_recent_details_not_inherited_after_expiry(monkeypatch):
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    mgr = make_reservation_manager(disc, router=opentable_router(FakeChannel()))
    wf = mgr.workflows.workflows["reservations"]
    wf._recent_ttl_s = 0  # everything is "too old" → never inherit

    await mgr.open(wf, "book a table for 2 at Lazy Bear next friday at 7pm", {}, "default")
    await mgr.handle("default", "no")

    turn = await mgr.open(wf, "make a reservation at Flores", {}, "default")
    # With nothing inherited, it must collect the missing essentials again.
    assert turn.control == TurnControl.CONTINUE


# --------------------------------------------------- release-policy detection (Phase 2)

def test_resolve_release_policy_searches_reddit_and_grounds():
    quote = "Bungalow drops reservations 30 days in advance at 10am ET, set an alarm."
    search = RecordingSearch([SearchResult(
        title="r/FoodNYC — booking Bungalow", snippet=quote,
        url="https://www.reddit.com/r/FoodNYC/comments/abc")])
    llm = FakeLLM({"opens_days_in_advance": 30, "release_time": "10am",
                   "release_timezone": "ET", "rolling": True, "confidence": 0.8,
                   "source_quote": quote, "notes": "per a Reddit thread"})
    disc = BusinessDiscovery(search_provider=search, yelp_client=FakeYelp(None), llm=llm)

    policy = disc.resolve_release_policy("Bungalow", "New York")
    assert policy is not None
    assert policy.days_in_advance == 30
    assert policy.release_time == "10am"
    assert policy.timezone == "ET"
    assert policy.confidence == 0.8
    # A Reddit-scoped search was actually issued.
    assert any(c["include_domains"] == ["reddit.com"] for c in search.calls)
    # Queries carry only business identity, never PII.
    assert all("Bungalow" in c["query"] for c in search.calls)


def test_resolve_release_policy_drops_ungrounded_fields():
    """The LLM 'remembers' a time that isn't in the evidence → that field is
    dropped, so no usable policy (we never snipe on a hallucinated time)."""
    search = FakeSearch([SearchResult(title="Bungalow", snippet="A great spot.",
                                      url="https://www.reddit.com/r/x")])
    llm = FakeLLM({"opens_days_in_advance": 30, "release_time": "10am",
                   "release_timezone": "ET", "rolling": True, "confidence": 0.9,
                   "source_quote": "made up", "notes": ""})
    disc = BusinessDiscovery(search_provider=search, yelp_client=FakeYelp(None), llm=llm)
    assert disc.resolve_release_policy("Bungalow", "New York") is None


class StubDiscovery:
    """Decouples the workflow test from live lookups."""
    def __init__(self, decision, policy=None):
        self._decision, self._policy = decision, policy

    def discover(self, name, location=None, target_url=None, kind="dining"):
        return self._decision

    def resolve_release_policy(self, name, location=None):
        return self._policy


def far_future_date(days: int = 75) -> str:
    """An "M/D" date comfortably past every snipe horizon.

    Hardcoding a calendar date here quietly rots: once the wall clock passes
    (date - release_days_in_advance) the booking window is already open and the
    snipe tests stop exercising a snipe at all.
    """
    from datetime import datetime, timedelta
    target = datetime.now() + timedelta(days=days)
    return f"{target.month}/{target.day}"


def _bookable_decision():
    return ChannelDecision(method=ReservationMethod.OPENTABLE, business_name="Bungalow",
                           url="https://www.opentable.com/r/bungalow-ny")


@pytest.mark.asyncio
async def test_autodetected_policy_offers_snipe():
    from workflows.reservations.models import ReleasePolicy
    policy = ReleasePolicy(days_in_advance=30, release_time="10am", timezone="ET",
                           confidence=0.8, source_quote="opens 30 days out at 10am ET")
    mgr = make_reservation_manager(StubDiscovery(_bookable_decision(), policy),
                                   router=opentable_router(FakeChannel()))
    wf = mgr.workflows.workflows["reservations"]

    # No release policy stated by the user; dining date is far enough out.
    turn = await mgr.open(
        wf, f"book a table for 2 at Bungalow on {far_future_date()} at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "open" in turn.message.lower()
    assert "opens 30 days out" in turn.message  # cites the discovered source

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND
    assert mgr.store.list_waiting()[0].slots.get("snipe_state") is not None


@pytest.mark.asyncio
async def test_near_term_booking_skips_autodetect():
    from workflows.reservations.models import ReleasePolicy
    policy = ReleasePolicy(days_in_advance=30, release_time="10am", timezone="ET",
                           confidence=0.9, source_quote="opens 30 days out at 10am ET")
    mgr = make_reservation_manager(StubDiscovery(_bookable_decision(), policy),
                                   router=opentable_router(FakeChannel()))
    wf = mgr.workflows.workflows["reservations"]

    # "next friday" is within the auto-detect horizon → no lookup, book normally.
    turn = await mgr.open(wf, "book a table for 2 at Bungalow next friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "book it" in turn.message.lower()   # the normal booking gate, not a snipe


# ----------------------------------------------------------------- snipe (Phase 1)

def test_snipe_datetime_math():
    from workflows.reservations.snipe import (
        compute_release_fire_ts, describe_fire, resolve_timezone)
    from datetime import datetime as _dt

    tz = resolve_timezone("ET")
    assert tz.key == "America/New_York"
    # dining 2026-08-15, opens 30 days ahead at 10:00 ET -> 2026-07-16 10:00 ET
    ts = compute_release_fire_ts("2026-08-15", 30, "10:00", tz)
    local = _dt.fromtimestamp(ts, tz)
    assert (local.year, local.month, local.day, local.hour, local.minute) == (2026, 7, 16, 10, 0)
    assert "Jul 16" in describe_fire(ts, tz)
    # aliases, explicit IANA, and a garbage fallback
    assert resolve_timezone("PT").key == "America/Los_Angeles"
    assert resolve_timezone("Europe/London").key == "Europe/London"
    assert resolve_timezone("not-a-zone", "America/Chicago").key == "America/Chicago"
    assert compute_release_fire_ts("not-a-date", 30, "10:00", tz) is None


@pytest.mark.asyncio
async def test_snipe_schedules_then_books_at_fire():
    import time as _time
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    # Dining date far enough out that the window (30 days prior) is still future.
    turn = await mgr.open(
        wf, f"book a table for 2 at Lazy Bear on {far_future_date()} at 7pm — reservations "
            f"open 30 days in advance at 10am ET", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "reservations open" in turn.message.lower()   # snipe gate, not a normal book
    assert fake.committed is False

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND
    assert fake.committed is False                        # nothing booked yet

    sess = mgr.store.list_waiting()[0]
    assert sess.slots.get("snipe_state") is not None
    assert sess.fsm_state == "scheduled"

    # Pretend the window just opened, then let the scheduled tick race it.
    sess.slots["snipe_state"]["fire_ts"] = _time.time() - 1
    sess.slots["snipe_state"]["deadline_ts"] = _time.time() + 60
    turn = await wf.on_tick(sess)
    assert turn.control == TurnControl.COMPLETE
    assert fake.committed is True
    assert "booked" in turn.message.lower()


@pytest.mark.asyncio
async def test_window_already_open_books_normally_not_sniped():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    mgr = make_reservation_manager(disc, router=opentable_router(FakeChannel()))
    wf = mgr.workflows.workflows["reservations"]

    # "next friday" minus 30 days is in the past → window already open → book now.
    turn = await mgr.open(
        wf, "book a table for 2 at Lazy Bear next friday at 7pm — reservations open "
            "30 days in advance at 10am", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "reservations open" not in turn.message.lower()   # the normal booking gate
    assert "book it" in turn.message.lower()


@pytest.mark.asyncio
async def test_snipe_miss_reports_honestly():
    import time as _time
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    miss = FakeChannel(result=BookingResult(success=False, needs_manual=True,
                                            error="slot_not_found", message="No slot."))
    mgr = make_reservation_manager(disc, router=opentable_router(miss))
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, f"book a table for 2 at Lazy Bear on {far_future_date()} at 7pm — "
                       f"reservations open 30 days in advance at 10am ET", {}, "default")
    await mgr.handle("default", "yes")
    sess = mgr.store.list_waiting()[0]
    sess.slots["snipe_state"]["fire_ts"] = _time.time() - 1
    sess.slots["snipe_state"]["deadline_ts"] = _time.time() + 0.5  # almost no budget
    turn = await wf.on_tick(sess)
    assert turn.control == TurnControl.COMPLETE
    assert "sold out" in turn.message.lower()
    assert turn.slots_update.get("snipe_state") is None  # session clears the parked state


@pytest.mark.asyncio
async def test_decline_does_not_book():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    turn = await mgr.handle("default", "no")
    assert turn.control == TurnControl.CANCEL
    assert fake.committed is False
    assert not mgr.has_active("default")


@pytest.mark.asyncio
async def test_unavailable_offers_to_wait():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel(status=AvailabilityStatus.UNAVAILABLE)
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION  # M6: offer to watch, not give up
    assert "keep checking" in turn.message.lower()
    assert fake.committed is False

    turn = await mgr.handle("default", "no")  # decline the watch
    assert turn.control == TurnControl.CANCEL


@pytest.mark.asyncio
async def test_kill_switch_is_research_only(monkeypatch):
    monkeypatch.setenv("RESERVATION_KILL_SWITCH", "1")
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.COMPLETE
    assert "research-only" in turn.message.lower()
    assert fake.committed is False


@pytest.mark.asyncio
async def test_phone_method_describes_only():
    # No booking link + a phone → method PHONE, which has no bookable channel yet.
    disc = BusinessDiscovery(FakeSearch([]), FakeYelp(yelp_business()))
    fake = FakeChannel()  # only registered for OPENTABLE
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.COMPLETE
    assert fake.committed is False


@pytest.mark.asyncio
async def test_missing_slots_are_collected_then_gated():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "I'd like to make a reservation", {}, "default")
    assert turn.control == TurnControl.CONTINUE
    assert "establishment" in turn.message.lower()
    assert "date" in (await mgr.handle("default", "Lazy Bear")).message.lower()
    assert "time" in (await mgr.handle("default", "Friday")).message.lower()
    assert "how many" in (await mgr.handle("default", "7pm")).message.lower()

    turn = await mgr.handle("default", "2")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION  # gated, not auto-booked


@pytest.mark.asyncio
async def test_unclear_reply_reasks_then_cancels_never_books():
    # "wait, what?" at a booking gate must neither book nor silently cancel:
    # re-ask once, then read a second unclear as a no.
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")

    turn = await mgr.handle("default", "wait, what time was that?")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION   # re-asked, still gated
    assert "yes or no" in turn.message.lower()
    assert fake.committed is False

    turn = await mgr.handle("default", "hmm, the weather is nice")
    assert turn.control == TurnControl.CANCEL               # second unclear = no
    assert fake.committed is False


@pytest.mark.asyncio
async def test_pii_purged_when_session_ends():
    # Terminal sessions lose guest PII (spec §8 L2) but keep the booking facts.
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear next friday at 7pm", {}, "default")
    session_id = mgr.store.list_active_dialogue()[0].session_id
    await mgr.handle("default", "yes")

    done = mgr.store.get(session_id)
    assert done.slots.get("raw_request") is None          # purged
    assert done.slots.get("business_name") == "Lazy Bear"  # kept


# ----------------------------------------------------------------- channels / router

def test_router_selects_browser_channel_else_none():
    router = ChannelRouter.from_env()
    assert router.select(ChannelDecision(method=ReservationMethod.OPENTABLE)) is not None
    assert router.select(ChannelDecision(method=ReservationMethod.PHONE)) is None
    assert router.select(ChannelDecision(method=ReservationMethod.UNKNOWN)) is None


@pytest.mark.asyncio
async def test_kill_switch_flipped_mid_flight_blocks_commit(monkeypatch):
    # The switch is OFF at gating time but flipped before the "yes" — the gate
    # re-checks at execution time, so nothing is booked (the old per-channel
    # check couldn't catch this).
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION

    monkeypatch.setenv("RESERVATION_KILL_SWITCH", "1")
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert "research-only" in turn.message.lower()
    assert fake.committed is False


@pytest.mark.asyncio
async def test_browser_channel_requires_card_without_payment(tmp_path):
    ch = OpenTableChannel(str(tmp_path))
    plan = CommitPlan(channel="opentable", summary="x",
                      details={"business_name": "X", "url": "http://x"}, requires_card=True)
    result = await ch.commit(plan)
    assert result.success is False
    assert result.error == "card_required"


# ----------------------------------------------------------------- M3: payment

def _manual_card_env(monkeypatch, number="4111 1111 1111 1111"):
    monkeypatch.delenv("PRIVACY_API_KEY", raising=False)
    monkeypatch.delenv("RESERVATION_CARD_PROVIDER", raising=False)
    monkeypatch.setenv("RESERVATION_CARD_NUMBER", number)
    monkeypatch.setenv("RESERVATION_CARD_CVV", "123")
    monkeypatch.setenv("RESERVATION_CARD_EXP_MONTH", "9")
    monkeypatch.setenv("RESERVATION_CARD_EXP_YEAR", "27")


def test_manual_card_from_env(monkeypatch):
    _manual_card_env(monkeypatch)
    svc = ManualCardService.from_env()
    assert svc is not None
    card = svc.mint_single_use()
    assert isinstance(card, VirtualCard)
    assert card.pan == "4111111111111111"      # separators stripped
    assert card.last_four == "1111"
    assert card.exp_month == "09" and card.exp_year == "2027"
    assert card.spend_limit_usd == 10.0
    assert "4111111111111111" not in repr(svc)  # PAN never in repr/logs


def test_manual_card_carries_billing_zip(monkeypatch):
    _manual_card_env(monkeypatch)
    monkeypatch.setenv("RESERVATION_CARD_ZIP", "94115")
    card = ManualCardService.from_env().mint_single_use()
    assert card.zip_code == "94115"   # threaded into card-on-file checkouts


def test_opentable_card_decline_regex():
    """A rejected card at "Complete" must be recognised so the booking hands off
    (the user needs a valid card), while benign policy text must not trip it."""
    from workflows.reservations.channels.opentable import _CARD_DECLINE_RE

    assert _CARD_DECLINE_RE.search("Your card was declined. Please try a different card.")
    assert _CARD_DECLINE_RE.search("There was a problem with your payment.")
    assert _CARD_DECLINE_RE.search("We were unable to process your card.")
    # The no-show policy and confirmation copy must NOT read as a decline.
    assert not _CARD_DECLINE_RE.search("No-shows will be subject to a charge of $45 per person.")
    assert not _CARD_DECLINE_RE.search("You're confirmed! We'll see you on June 30.")


def test_manual_card_rejects_invalid_number_and_over_cap(monkeypatch):
    _manual_card_env(monkeypatch, number="4111111111111112")   # fails Luhn
    assert ManualCardService.from_env() is None

    _manual_card_env(monkeypatch)
    svc = ManualCardService.from_env()
    assert svc.mint_single_use(amount_usd=25) is None          # hard cap holds
    monkeypatch.setenv("RESERVATION_CARD_LIMIT_USD", "50")     # can't raise it
    assert ManualCardService.from_env().limit_usd == 10.0


def test_card_provider_selection(monkeypatch):
    _manual_card_env(monkeypatch)
    assert isinstance(card_service_from_env(), ManualCardService)

    monkeypatch.setenv("PRIVACY_API_KEY", "pk")                # privacy wins by default
    assert isinstance(card_service_from_env(), PrivacyCardService)

    monkeypatch.setenv("RESERVATION_CARD_PROVIDER", "manual")  # explicit override
    assert isinstance(card_service_from_env(), ManualCardService)

    monkeypatch.setenv("RESERVATION_CARD_PROVIDER", "off")
    assert card_service_from_env() is None


@pytest.mark.asyncio
async def test_manual_card_passed_to_commit(monkeypatch):
    _manual_card_env(monkeypatch)
    disc = BusinessDiscovery(FakeSearch(card_hint_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]
    wf.payment = ManualCardService.from_env()

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert isinstance(fake.payment_received, VirtualCard)
    assert fake.payment_received.last_four == "1111"
    assert fake.payment_received.token == "manual"


def test_card_service_enforces_hard_cap():
    # A higher configured limit is clamped to $10, and over-cap mints are refused
    # without any network call.
    svc = PrivacyCardService("api-key", limit_usd=50)
    assert svc.limit_usd == 10.0
    assert svc.mint_single_use(amount_usd=25) is None


@pytest.mark.asyncio
async def test_no_card_needed_books_without_minting():
    # Discovery found no card/deposit hint → the gate message mentions no card,
    # nothing is minted (even with a payment service configured), and commit
    # proceeds with payment=None.
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    pay = FakePayment()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]
    wf.payment = pay

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "card" not in turn.message.lower()

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert fake.committed is True
    assert pay.minted is False              # payment service never touched
    assert fake.payment_received is None    # commit ran card-free


@pytest.mark.asyncio
async def test_card_minted_and_passed_to_commit():
    disc = BusinessDiscovery(FakeSearch(card_hint_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    pay = FakePayment()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    mgr.workflows.workflows["reservations"].payment = pay
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "card" in turn.message.lower()  # gate discloses the card

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert pay.minted is True
    assert isinstance(fake.payment_received, VirtualCard)
    assert fake.payment_received.spend_limit_usd == 10.0


@pytest.mark.asyncio
async def test_mint_failure_aborts_booking():
    disc = BusinessDiscovery(FakeSearch(card_hint_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    mgr.workflows.workflows["reservations"].payment = FakePayment(card=None)  # mint fails
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert "one-time card" in turn.message.lower()
    assert fake.committed is False  # never booked without the card


@pytest.mark.asyncio
async def test_no_payment_service_hands_off_when_card_needed():
    disc = BusinessDiscovery(FakeSearch(card_hint_result()), FakeYelp(yelp_business()))
    # Real browser channel; no payment service configured.
    import tempfile
    router = ChannelRouter({ReservationMethod.OPENTABLE: OpenTableChannel(tempfile.mkdtemp())})
    mgr = make_reservation_manager(disc, router=router)
    mgr.workflows.workflows["reservations"].payment = None
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert "card" in turn.message.lower()  # honest hand-off, not a booking


# ----------------------------------------------------------------- M4: calendar + signal

def test_calendar_builds_event_with_details():
    # The harness hands the calendar canonical values only (ISO date, 24h time).
    captured = {}

    def inserter(calendar_id, event):
        captured["calendar_id"] = calendar_id
        captured["event"] = event
        return "https://calendar.google.com/event?eid=abc"

    svc = CalendarService(calendar_id="primary", inserter=inserter)
    link = svc.create_event({
        "business_name": "Lazy Bear", "party_size": 2, "date": "2026-06-02", "time": "19:00",
        "address": "3416 19th St", "method": "opentable", "confirmation": "XYZ9",
    })
    assert link.startswith("https://calendar.google.com")
    ev = captured["event"]
    assert "Lazy Bear" in ev["summary"]
    assert ev["location"] == "3416 19th St"
    assert ev["start"]["dateTime"].startswith("2026-06-02T19:00")
    assert "XYZ9" in ev["description"]


def test_calendar_noop_without_inserter():
    svc = CalendarService(inserter=None)
    assert svc.create_event({"business_name": "X", "date": "2026-06-02", "time": "19:00"}) is None


def test_calendar_skips_when_time_unparseable():
    # Non-canonical input means skip, never guess.
    svc = CalendarService(inserter=lambda c, e: "link")
    assert svc.create_event({"business_name": "X", "date": "someday", "time": "later"}) is None


def test_telegram_from_env_none_when_unconfigured(monkeypatch):
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_NOTIFY_CHAT_ID", "TELEGRAM_ALLOWED_CHAT_IDS"):
        monkeypatch.delenv(var, raising=False)
    assert TelegramNotifier.from_env() is None


def test_telegram_from_env_falls_back_to_first_allowed_chat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.delenv("TELEGRAM_NOTIFY_CHAT_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "555, 777")
    n = TelegramNotifier.from_env()
    assert n is not None and n.chat_id == "555"


def test_telegram_send_builds_payload():
    captured = {}

    class Resp:
        status_code = 200

    def poster(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    n = TelegramNotifier("123:ABC", "555", poster=poster)
    assert n.send("hello") is True
    assert captured["url"] == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert captured["json"] == {"chat_id": "555", "text": "hello"}


def test_telegram_send_never_raises():
    def poster(*a, **k):
        raise RuntimeError("network down")

    n = TelegramNotifier("123:ABC", "555", poster=poster)
    assert n.send("hi") is False


class _RecordingCalendar:
    def __init__(self):
        self.facts = None

    def create_event(self, facts):
        self.facts = facts
        return "https://cal/evt"


class _RecordingNotifier:
    def __init__(self):
        self.message = None

    def send(self, message):
        self.message = message
        return True


@pytest.mark.asyncio
async def test_confirmed_booking_creates_calendar_and_telegram():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]
    cal, sig = _RecordingCalendar(), _RecordingNotifier()
    wf.calendar, wf.notifier = cal, sig

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    turn = await mgr.handle("default", "yes")

    assert turn.control == TurnControl.COMPLETE
    assert "calendar" in turn.message.lower()
    assert cal.facts["business_name"] == "Lazy Bear"
    assert cal.facts["confirmation"] == "ABC123"
    assert sig.message and "Lazy Bear" in sig.message


# ----------------------------------------------------------------- M5: phone (Bland.ai)

from workflows.reservations.channels import (  # noqa: E402
    BlandClient, DialAndBridgeChannel, PhoneChannel, PhoneOutcome,
)


class FakePhoneClient:
    def __init__(self, results):
        self._results = list(results)
        self.placed = None

    def place_call(self, phone, task, request_data=None):
        self.placed = (phone, task)
        return "call_1"

    def get_result(self, call_id):
        return self._results.pop(0) if self._results else {"completed": True, "summary": "confirmed"}


def _force_waiting_due(mgr):
    for s in mgr.store.list_waiting():
        s.wake_at = 0
        mgr.store.save(s)


def test_phone_classify_outcomes():
    ch = PhoneChannel(client=None)
    assert ch._classify({"completed": False}).outcome == PhoneOutcome.PENDING
    confirmed = ch._classify({"completed": True, "summary": "all set",
                              "analysis": {"outcome": "confirmed", "confirmation_number": "OT9"}})
    assert confirmed.outcome == PhoneOutcome.CONFIRMED
    assert confirmed.confirmation == "OT9"
    nofree = ch._classify({"completed": True, "summary": "Sorry, we are fully booked tonight."})
    assert nofree.outcome == PhoneOutcome.NO_AVAILABILITY


def test_is_open_now_and_next_open():
    ch = PhoneChannel(client=None)
    hours = [{"open": [{"day": 0, "start": "1700", "end": "2200"}]}]  # Monday 5–10pm
    monday_6pm = datetime(2026, 6, 1, 18, 0)
    monday_3pm = datetime(2026, 6, 1, 15, 0)

    open_now, _ = ch.is_open_now({"hours": hours}, monday_6pm)
    assert open_now is True

    closed, next_open = ch.is_open_now({"hours": hours}, monday_3pm)
    assert closed is False
    assert datetime.fromtimestamp(next_open).hour == 17


def test_router_phone_wiring(monkeypatch):
    monkeypatch.setenv("RESERVATION_PHONE_PROVIDER", "dial_and_bridge")
    assert isinstance(
        ChannelRouter.from_env().select(ChannelDecision(method=ReservationMethod.PHONE)),
        DialAndBridgeChannel,
    )
    monkeypatch.setenv("RESERVATION_PHONE_PROVIDER", "off")
    assert ChannelRouter.from_env().select(ChannelDecision(method=ReservationMethod.PHONE)) is None


@pytest.mark.asyncio
async def test_phone_call_flow_confirms_via_polling():
    disc = BusinessDiscovery(FakeSearch([]), FakeYelp(yelp_business()))  # -> PHONE method
    client = FakePhoneClient([
        {"completed": False},  # first poll: still ringing
        {"completed": True, "summary": "booked",
         "analysis": {"outcome": "confirmed", "confirmation_number": "PH42",
                      "negotiated_time": "7:30pm"}},
    ])
    phone = PhoneChannel(client=client, callback_number="+1999", guest_name="Prem")
    router = ChannelRouter({ReservationMethod.PHONE: phone})
    mgr = make_reservation_manager(disc, router=router)
    wf = mgr.workflows.workflows["reservations"]
    cal, sig = _RecordingCalendar(), _RecordingNotifier()
    wf.calendar, wf.notifier = cal, sig

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND          # call placed, now WAITING
    assert "calling" in turn.message.lower()
    assert client.placed is not None                       # Bland call was placed
    assert not mgr.has_active("default")                   # WAITING isn't active dialogue

    # First background tick: still ringing.
    _force_waiting_due(mgr)
    advanced = await mgr.tick_waiting()
    assert advanced[0][1].control == TurnControl.BACKGROUND

    # Second tick: confirmed → completes + calendar/signal fire.
    _force_waiting_due(mgr)
    advanced = await mgr.tick_waiting()
    _s, result = advanced[0]
    assert result.control == TurnControl.COMPLETE
    assert "booked" in result.message.lower() and "7:30pm" in result.message
    assert cal.facts["confirmation"] == "PH42"
    assert sig.message is not None
    assert not mgr.store.list_waiting()


@pytest.mark.asyncio
async def test_kill_switch_blocks_waiting_phone_redial(monkeypatch):
    # The approved call fails (no answer) and the session goes WAITING for a
    # retry; the switch is flipped meanwhile. The redial fires through the
    # gate at execution time → blocked. Pre-harness, place_call never checked
    # the switch, so the retry would have dialed anyway.
    class CountingPhoneClient(FakePhoneClient):
        def __init__(self, results):
            super().__init__(results)
            self.calls_placed = 0

        def place_call(self, phone, task, request_data=None):
            self.calls_placed += 1
            return super().place_call(phone, task, request_data)

    disc = BusinessDiscovery(FakeSearch([]), FakeYelp(yelp_business()))  # -> PHONE
    client = CountingPhoneClient([
        {"completed": True, "summary": "no answer, went to voicemail"},  # attempt 1 fails
    ])
    phone = PhoneChannel(client=client)
    mgr = make_reservation_manager(disc, router=ChannelRouter({ReservationMethod.PHONE: phone}))
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    await mgr.handle("default", "yes")                 # call placed (attempt 1)
    assert client.calls_placed == 1
    _force_waiting_due(mgr)
    _s, result = (await mgr.tick_waiting())[0]
    assert result.control == TurnControl.BACKGROUND    # no answer → retry scheduled

    monkeypatch.setenv("RESERVATION_KILL_SWITCH", "1")  # flipped while WAITING
    _force_waiting_due(mgr)
    _s, result = (await mgr.tick_waiting())[0]
    assert result.control == TurnControl.COMPLETE
    assert "research-only" in result.message.lower()
    assert client.calls_placed == 1                    # the redial never fired


@pytest.mark.asyncio
async def test_phone_no_availability_reports_back():
    disc = BusinessDiscovery(FakeSearch([]), FakeYelp(yelp_business()))
    client = FakePhoneClient([{"completed": True, "summary": "we are fully booked"}])
    phone = PhoneChannel(client=client)
    mgr = make_reservation_manager(disc, router=ChannelRouter({ReservationMethod.PHONE: phone}))
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    await mgr.handle("default", "yes")
    _force_waiting_due(mgr)
    _s, result = (await mgr.tick_waiting())[0]
    assert result.control == TurnControl.COMPLETE
    assert "nothing" in result.message.lower() or "another" in result.message.lower()


# ----------------------------------------------------------------- M5b: email

from workflows.reservations.channels import (  # noqa: E402
    EmailChannel, EmailOutcome,
)
from workflows.reservations.channels.email import default_drafter  # noqa: E402


class FakeEmailSender:
    def __init__(self):
        self.sent = None

    def send(self, to, subject, body):
        self.sent = (to, subject, body)
        return True


class FakeEmailReader:
    def __init__(self, replies):
        self._replies = list(replies)

    def fetch_reply(self, to, subject):
        return self._replies.pop(0) if self._replies else None


def test_email_classify_reply():
    ch = EmailChannel(sender=FakeEmailSender())
    assert ch._classify_reply("We've confirmed your table, see you then!") == EmailOutcome.CONFIRMED
    assert ch._classify_reply("Unfortunately we're fully booked.") == EmailOutcome.DECLINED
    assert ch._classify_reply("Please call us to finalise.") == EmailOutcome.ASKS_TO_CALL
    assert ch._classify_reply("What time exactly?") == EmailOutcome.NEEDS_INFO


def test_default_drafter_includes_details():
    dec = ChannelDecision(method=ReservationMethod.EMAIL, business_name="Lazy Bear",
                          email="book@lazybear.com")
    slots = {"party_size": 2, "date": "Friday", "time": "7pm", "guest_name": "Prem",
             "phone": "+1999"}
    subject, body = default_drafter(slots, dec)
    assert "Prem" in subject
    assert "2 on Friday at 7pm" in body
    assert "+1999" in body


def test_email_send_and_poll():
    sender, reader = FakeEmailSender(), FakeEmailReader(["We have you confirmed! Confirmation: LB777"])
    ch = EmailChannel(sender=sender, reader=reader)
    plan = CommitPlan(channel="email", summary="x",
                      details={"to": "book@lazybear.com", "subject": "Reservation request",
                               "body": "Hello"})
    assert ch.send(plan) is True
    assert sender.sent[0] == "book@lazybear.com"

    reply = ch.poll_reply("book@lazybear.com", "Reservation request")
    assert reply.outcome == EmailOutcome.CONFIRMED
    assert reply.confirmation == "LB777"


def test_router_email_wiring(monkeypatch):
    monkeypatch.setenv("RESERVATION_EMAIL_PROVIDER", "smtp")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("RESERVATION_EMAIL_FROM", "me@example.com")
    sel = ChannelRouter.from_env().select(ChannelDecision(method=ReservationMethod.EMAIL))
    assert isinstance(sel, EmailChannel)

    monkeypatch.setenv("RESERVATION_EMAIL_PROVIDER", "off")
    assert ChannelRouter.from_env().select(ChannelDecision(method=ReservationMethod.EMAIL)) is None


@pytest.mark.asyncio
async def test_phone_email_requested_hands_off_to_email():
    disc = BusinessDiscovery(FakeSearch([]), FakeYelp(yelp_business()))  # -> PHONE
    phone_client = FakePhoneClient([
        {"completed": True, "summary": "please email us",
         "analysis": {"outcome": "email_requested", "email_address": "book@lazybear.com"}},
    ])
    phone = PhoneChannel(client=phone_client)
    sender = FakeEmailSender()
    reader = FakeEmailReader([None, "Confirmed — see you then! Confirmation: LB777"])
    email = EmailChannel(sender=sender, reader=reader)
    router = ChannelRouter({ReservationMethod.PHONE: phone, ReservationMethod.EMAIL: email})
    mgr = make_reservation_manager(disc, router=router)
    wf = mgr.workflows.workflows["reservations"]
    cal, sig = _RecordingCalendar(), _RecordingNotifier()
    wf.calendar, wf.notifier = cal, sig

    # Approve the call → it's placed → poll returns "email us" → hand off to an email draft.
    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    await mgr.handle("default", "yes")
    _force_waiting_due(mgr)
    _s, handoff = (await mgr.tick_waiting())[0]
    assert handoff.control == TurnControl.AWAIT_CONFIRMATION
    assert "email" in handoff.message.lower()
    assert mgr.has_active("default")          # promoted back to an active dialogue

    # Approve the draft → email is sent → WAITING for a reply.
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND
    assert sender.sent[0] == "book@lazybear.com"

    # First reply poll: nothing yet; second: confirmed.
    _force_waiting_due(mgr)
    assert (await mgr.tick_waiting())[0][1].control == TurnControl.BACKGROUND
    _force_waiting_due(mgr)
    _s, result = (await mgr.tick_waiting())[0]
    assert result.control == TurnControl.COMPLETE
    assert "confirmed" in result.message.lower()
    assert cal.facts["confirmation"] == "LB777"
    assert sig.message is not None


@pytest.mark.asyncio
async def test_email_draft_edit_then_send():
    # Drive the email confirm state directly via the phone handoff, then edit before sending.
    disc = BusinessDiscovery(FakeSearch([]), FakeYelp(yelp_business()))
    phone_client = FakePhoneClient([
        {"completed": True, "summary": "email please",
         "analysis": {"outcome": "email_requested", "email_address": "book@lazybear.com"}},
    ])
    sender = FakeEmailSender()
    email = EmailChannel(sender=sender, reader=FakeEmailReader([]))
    router = ChannelRouter({ReservationMethod.PHONE: PhoneChannel(client=phone_client),
                            ReservationMethod.EMAIL: email})
    mgr = make_reservation_manager(disc, router=router)
    wf = mgr.workflows.workflows["reservations"]

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    await mgr.handle("default", "yes")
    _force_waiting_due(mgr)
    await mgr.tick_waiting()  # -> email draft, AWAIT_CONFIRMATION

    # An edit instruction re-drafts and re-confirms (does NOT send).
    turn = await mgr.handle("default", "please add that we'd like a quiet table")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert sender.sent is None

    # Now approve.
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND
    assert sender.sent is not None
    assert "quiet table" in sender.sent[2]


# ----------------------------------------------------------------- M6: wait-and-book

class FakeWatchChannel(ReservationChannel):
    """Returns a sequence of availability statuses across check_availability calls."""
    method = ReservationMethod.OPENTABLE
    can_commit = True

    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.committed = False
        self.payment_received = "unset"

    async def check_availability(self, slots):
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return Availability(status=status)

    async def prepare(self, slots, decision):
        return CommitPlan(channel="opentable", summary="book it",
                          details={"business_name": decision.business_name or slots.get("business_name"),
                                   "party_size": slots.get("party_size"), "date": slots.get("date"),
                                   "time": slots.get("time")})

    async def commit(self, plan, payment=None):
        self.committed = True
        return BookingResult(success=True, message="Booked, sir.", confirmation="W123")


@pytest.mark.asyncio
async def test_wait_and_book_opens_then_books():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    # unavailable at gate, unavailable on first watch tick, then available.
    chan = FakeWatchChannel([
        AvailabilityStatus.UNAVAILABLE, AvailabilityStatus.UNAVAILABLE, AvailabilityStatus.AVAILABLE,
    ])
    mgr = make_reservation_manager(disc, router=opentable_router(chan))
    wf = mgr.workflows.workflows["reservations"]
    wf.calendar, wf.notifier = _RecordingCalendar(), _RecordingNotifier()

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION       # offered to wait

    turn = await mgr.handle("default", "yes please")           # start watching
    assert turn.control == TurnControl.BACKGROUND
    assert not mgr.has_active("default")

    # First watch tick: still nothing.
    _force_waiting_due(mgr)
    assert (await mgr.tick_waiting())[0][1].control == TurnControl.BACKGROUND

    # Second watch tick: a table opened → re-gate for approval (active again).
    _force_waiting_due(mgr)
    _s, opened = (await mgr.tick_waiting())[0]
    assert opened.control == TurnControl.AWAIT_CONFIRMATION
    assert "opened" in opened.message.lower()
    assert mgr.has_active("default")
    assert chan.committed is False                             # still not booked without a yes

    # Approve the now-available slot → books.
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert chan.committed is True
    assert "booked" in turn.message.lower()


@pytest.mark.asyncio
async def test_wait_gives_up_after_deadline():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    chan = FakeWatchChannel([AvailabilityStatus.UNAVAILABLE])
    mgr = make_reservation_manager(disc, router=opentable_router(chan))
    wf = mgr.workflows.workflows["reservations"]
    wf.watch_deadline_days = 0  # expire immediately

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    await mgr.handle("default", "yes")
    _force_waiting_due(mgr)
    _s, result = (await mgr.tick_waiting())[0]
    assert result.control == TurnControl.COMPLETE
    assert "nothing opened" in result.message.lower()


# ----------------------------------------------------------------- M7: sandbox bot fallback

from workflows.reservations.channels import (  # noqa: E402
    BotCandidate, DockerSandbox, GitHubBotFinder, SandboxBotChannel, SandboxResult,
)


def _repo(full_name, stars, language, pushed="2026-05-01T00:00:00Z", archived=False):
    return {"fullName": full_name, "url": f"https://github.com/{full_name}",
            "stargazersCount": stars, "language": language, "pushedAt": pushed,
            "isArchived": archived}


def test_finder_vets_candidates():
    results = [
        _repo("a/archived", 100, "Python", archived=True),     # reject: archived
        _repo("b/lowstars", 1, "Python"),                       # reject: too few stars
        _repo("c/badlang", 50, "Brainfuck"),                    # reject: language
        _repo("d/stale", 50, "Python", pushed="2018-01-01T00:00:00Z"),  # reject: stale
        _repo("e/good", 80, "Python"),                          # accept
    ]
    finder = GitHubBotFinder(searcher=lambda q: results)
    cand = finder.find("opentable reservation bot")
    assert cand is not None and cand.full_name == "e/good"


def test_sandbox_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RESERVATION_ALLOW_SANDBOX_BOTS", raising=False)
    assert SandboxBotChannel.from_env() is None


def _sandbox_with(candidate_repos, run_result):
    finder = GitHubBotFinder(searcher=lambda q: candidate_repos)
    sandbox = DockerSandbox(runner=lambda url, payload: run_result)
    return SandboxBotChannel(finder, sandbox)


@pytest.mark.asyncio
async def test_sandbox_fallback_offered_and_runs_on_consent():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    # Browser channel "books" but can't automate → needs_manual.
    fake = FakeChannel(result=BookingResult(success=False, needs_manual=True,
                                            message="I couldn't complete it automatically."))
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]
    wf.calendar, wf.notifier = _RecordingCalendar(), _RecordingNotifier()
    wf.sandbox = _sandbox_with([_repo("acme/ot-bot", 120, "Python")],
                               SandboxResult(True, output="Reservation confirmed",
                                             confirmation="BOT9"))

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    turn = await mgr.handle("default", "yes")  # approve booking → browser fails → offer sandbox
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "acme/ot-bot" in turn.message
    assert "third-party" in turn.message.lower()

    turn = await mgr.handle("default", "yes")  # consent to the sandbox
    assert turn.control == TurnControl.COMPLETE
    assert "booked" in turn.message.lower()
    assert wf.calendar.facts["confirmation"] == "BOT9"


@pytest.mark.asyncio
async def test_sandbox_fallback_declined():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel(result=BookingResult(success=False, needs_manual=True, message="can't"))
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]
    ran = {"called": False}

    def _runner(url, payload):
        ran["called"] = True
        return SandboxResult(True)

    wf.sandbox = SandboxBotChannel(GitHubBotFinder(searcher=lambda q: [_repo("a/b", 99, "Python")]),
                                   DockerSandbox(runner=_runner))

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    await mgr.handle("default", "yes")          # browser fails → offer sandbox
    turn = await mgr.handle("default", "no")    # decline
    assert turn.control == TurnControl.COMPLETE
    assert "third-party" in turn.message.lower()
    assert ran["called"] is False               # never ran untrusted code


@pytest.mark.asyncio
async def test_no_sandbox_just_hands_off():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel(result=BookingResult(success=False, needs_manual=True,
                                            message="Finish it here: http://x"))
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]
    wf.sandbox = None  # disabled

    await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.COMPLETE
    assert "finish it here" in turn.message.lower()


# ----------------------------------------------------------- LLM refinement (live-env work)

from workflows.reservations import llm as resllm  # noqa: E402


class FakeLLM:
    """Stands in for ReservationLLM: returns a canned dict (or None = LLM failure)."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def complete_json(self, system, user, max_tokens=700):
        self.calls.append((system, user))
        return self._response


def test_discovery_email_only_heuristic():
    # No LLM at all: the deterministic fallback still detects book-by-email.
    results = [SearchResult(title="Lazy Bear — Reservations",
                            snippet="Email us at book@lazybear.com to reserve a table.",
                            url="https://lazybear.com/reservations")]
    disc = BusinessDiscovery(FakeSearch(results), FakeYelp(None))
    decision = disc.discover("Lazy Bear")
    assert decision.method == ReservationMethod.EMAIL
    assert decision.email == "book@lazybear.com"


def test_discovery_llm_detects_email_only():
    # Heuristic would say PHONE (Yelp number, no platform link); the LLM reads the
    # snippet and reroutes to EMAIL with the validated address.
    results = [SearchResult(title="Lazy Bear",
                            snippet="To arrange a table, write to book@lazybear.com.",
                            url="https://lazybear.com")]
    llm = FakeLLM({"method": "email", "email": "book@lazybear.com", "url": None,
                   "requires_card": False, "confidence": 0.9, "notes": "email-only"})
    disc = BusinessDiscovery(FakeSearch(results), FakeYelp(yelp_business()), llm=llm)
    decision = disc.discover("Lazy Bear")
    assert decision.method == ReservationMethod.EMAIL
    assert decision.email == "book@lazybear.com"
    assert "llm" in decision.source
    # Evidence sent to the LLM holds only business data — never user PII (§8 L5).
    _system, user = llm.calls[0]
    assert "Lazy Bear" in user


def test_discovery_llm_cannot_override_platform_match():
    # The OpenTable domain match is high-precision; an LLM disagreement only
    # contributes the card hint, never the method.
    llm = FakeLLM({"method": "generic_web", "url": None, "email": None,
                   "requires_card": True, "confidence": 0.9, "notes": "n"})
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()), llm=llm)
    decision = disc.discover("Lazy Bear")
    assert decision.method == ReservationMethod.OPENTABLE
    assert decision.requires_card_hint is True


def test_classify_method_rejects_hallucinated_contacts():
    evidence = {"business": None,
                "search_results": [{"title": "Lazy Bear", "url": "https://lazybear.com",
                                    "snippet": "call us"}]}
    # An email verdict whose address isn't in the evidence is useless → None.
    llm = FakeLLM({"method": "email", "email": "fake@nope.com", "url": None,
                   "requires_card": False, "confidence": 1.0, "notes": ""})
    assert resllm.classify_method(llm, evidence) is None
    # A hallucinated URL is dropped while the method survives.
    llm = FakeLLM({"method": "generic_web", "url": "https://hallucinated.example",
                   "email": None, "requires_card": False, "confidence": 0.7, "notes": ""})
    out = resllm.classify_method(llm, evidence)
    assert out["method"] == "generic_web"
    assert out["url"] is None


def test_extract_slots_llm_refines_regex():
    # "four of us" and "half past seven" defeat the regexes; the LLM gets them,
    # then the harness canonicalizes ("half past seven" → 19:30, evening rule).
    llm = FakeLLM({"business_name": "Lazy Bear", "date": "next Friday",
                   "time": "half past seven", "party_size": "4", "service_type": "dinner"})
    wf = ReservationWorkflow(discovery=BusinessDiscovery(FakeSearch([]), FakeYelp(None)), llm=llm)
    slots = wf._extract_slots("book dinner for four of us at Lazy Bear next Friday at half past seven")
    assert slots["party_size"] == 4
    assert slots["time"] == "19:30"
    assert slots["time_raw"] == "half past seven"
    assert slots["business_name"] == "Lazy Bear"
    assert slots["service_type"] == "dinner"


def test_extract_slots_falls_back_to_regex_on_llm_failure():
    llm = FakeLLM(None)  # the call fails → regex baseline must still work
    wf = ReservationWorkflow(discovery=BusinessDiscovery(FakeSearch([]), FakeYelp(None)), llm=llm)
    slots = wf._extract_slots("book a table for 2 at Lazy Bear next friday at 7pm")
    assert slots["party_size"] == 2
    assert slots["business_name"].lower() == "lazy bear"


def test_make_llm_drafter_uses_llm_and_falls_back():
    def fallback(slots, decision, instruction=None):
        return "fallback subject", "fallback body"

    dec = ChannelDecision(method=ReservationMethod.EMAIL, business_name="Lazy Bear")
    good = resllm.make_llm_drafter(FakeLLM({"subject": "S", "body": "B"}), fallback=fallback)
    assert good({}, dec) == ("S", "B")
    broken = resllm.make_llm_drafter(FakeLLM(None), fallback=fallback)
    assert broken({}, dec) == ("fallback subject", "fallback body")


@pytest.mark.asyncio
async def test_email_only_discovery_routes_to_email_gate():
    # Discovery says EMAIL → the workflow gates on the draft, then sends on "yes".
    results = [SearchResult(title="Lazy Bear — Reservations",
                            snippet="Email us at book@lazybear.com to reserve a table.",
                            url="https://lazybear.com/reservations")]
    disc = BusinessDiscovery(FakeSearch(results), FakeYelp(None))
    sender = FakeEmailSender()
    email = EmailChannel(sender=sender, reader=FakeEmailReader([]))
    router = ChannelRouter({ReservationMethod.EMAIL: email})
    mgr = make_reservation_manager(disc, router=router)
    wf = mgr.workflows.workflows["reservations"]

    turn = await mgr.open(wf, "book a table for 2 at Lazy Bear friday at 7pm", {}, "default")
    assert turn.control == TurnControl.AWAIT_CONFIRMATION
    assert "book@lazybear.com" in turn.message

    turn = await mgr.handle("default", "yes")
    assert turn.control == TurnControl.BACKGROUND
    assert sender.sent[0] == "book@lazybear.com"


# --------------------------------------------------------------- resy success detection

def test_resy_confirmed_regex_recognizes_booked_panel():
    """Regression: Resy's success panel reads "Reservation Booked." with
    "check your inbox for a confirmation email" / "Continue to Reservation
    Details" — none contain the word "confirmed". The detector must still match
    these (a live booking once mis-reported as failed because it didn't)."""
    from workflows.reservations.channels.resy import _CONFIRMED_RE

    booked = ("Reservation Booked. Please check your inbox for a confirmation "
              "email. Continue to Reservation Details")
    assert _CONFIRMED_RE.search(booked)
    assert _CONFIRMED_RE.search("Your reservation is confirmed")
    assert _CONFIRMED_RE.search("You're all set")

    # Pre-booking widget states must NOT read as success.
    assert not _CONFIRMED_RE.search("Select a time  6:00 PM  6:15 PM  Reserve Now")
    assert not _CONFIRMED_RE.search("Log in or sign up to continue")
    assert not _CONFIRMED_RE.search("Reservation details  Party of 2  Reserve Now")


# ----------------------------------------------------------- opentable booking flow

def test_opentable_booking_url_preselects_datetime_and_covers():
    from workflows.reservations.channels.opentable import OpenTableChannel

    url = OpenTableChannel._booking_url({
        "url": "https://www.opentable.com/r/lazy-bear?ref=123",
        "date": "2026-06-14", "time": "19:00", "party_size": 2,
    })
    assert url.startswith("https://www.opentable.com/r/lazy-bear?")
    assert "dateTime=2026-06-14T19%3A00" in url
    assert "covers=2" in url
    assert "ref=123" not in url  # the venue's own query is dropped, not appended

    # A non-OpenTable URL isn't ours to drive.
    assert OpenTableChannel._booking_url(
        {"url": "https://resy.com/cities/ny/lazy-bear"}) is None
    assert OpenTableChannel._booking_url({"url": None}) is None


def test_opentable_time_normalization_matches_slot_labels():
    from workflows.reservations.channels.opentable import OpenTableChannel

    assert OpenTableChannel._to_12h("19:00") == (7, "00", "PM")
    assert OpenTableChannel._to_12h("00:30") == (12, "30", "AM")
    assert OpenTableChannel._to_12h("12:15") == (12, "15", "PM")
    assert OpenTableChannel._to_12h("09:45") == (9, "45", "AM")
    assert OpenTableChannel._to_12h("bogus") is None
    assert OpenTableChannel._display_time("19:00") == "7:00 PM"


def test_opentable_confirmed_regex_recognizes_confirmation_page():
    """OpenTable's confirmation page reads "You're confirmed" / "Reservation
    confirmed" / "Your table is booked" — none of which is the bare word the
    naive detector looks for. Pre-booking states must not read as success."""
    from workflows.reservations.channels.opentable import _CONFIRMED_RE, _LOGIN_RE

    assert _CONFIRMED_RE.search("You're confirmed! We'll see you on Sunday.")
    assert _CONFIRMED_RE.search("Reservation confirmed")
    assert _CONFIRMED_RE.search("Your table is booked")

    # The slot grid / details page must NOT read as a confirmation.
    assert not _CONFIRMED_RE.search("Select a time  7:00 PM  7:15 PM  Complete reservation")
    assert not _CONFIRMED_RE.search("Reservation details  Party of 2")

    # The header's "Sign in" link must NOT trip the session-expired wall; only a
    # genuine "sign in to complete" prompt should.
    assert not _LOGIN_RE.search("Home  Restaurants  Sign in  My reservations")
    assert _LOGIN_RE.search("Sign in to complete your reservation")
