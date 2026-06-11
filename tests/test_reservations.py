"""
Tests for the Reservations agent — M1 (discovery + multi-turn collection, no commits).
"""

import pytest

from core.conversation import InMemorySessionStore, SessionManager, TurnControl
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
from workflows.reservations.notify import SignalNotifier
from workflows.reservations.payment import PrivacyCardService, VirtualCard

from datetime import datetime


# --------------------------------------------------------------------------- fakes

class FakeSearch:
    def __init__(self, results):
        self._results = results

    def search(self, query, max_results=5):
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


def make_reservation_manager(discovery, router=None):
    wf_manager = WorkflowManager()
    wf_manager.register(ReservationWorkflow(discovery=discovery, router=router, llm=None))
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


# ----------------------------------------------------------------- extraction

def test_extract_slots_from_full_request():
    wf = ReservationWorkflow(discovery=BusinessDiscovery(FakeSearch([]), FakeYelp(None)), llm=None)
    slots = wf._extract_slots("book a table for 2 at Lazy Bear next friday at 7pm")
    assert slots["party_size"] == 2
    assert slots["business_name"].lower() == "lazy bear"
    assert "7pm" in slots["time"].lower().replace(" ", "")
    assert "friday" in slots["date"].lower()


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


# ----------------------------------------------------------------- channels / router

def test_router_selects_browser_channel_else_none():
    router = ChannelRouter.from_env()
    assert router.select(ChannelDecision(method=ReservationMethod.OPENTABLE)) is not None
    assert router.select(ChannelDecision(method=ReservationMethod.PHONE)) is None
    assert router.select(ChannelDecision(method=ReservationMethod.UNKNOWN)) is None


@pytest.mark.asyncio
async def test_browser_channel_kill_switch_blocks_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("RESERVATION_KILL_SWITCH", "1")
    ch = OpenTableChannel(str(tmp_path))
    plan = CommitPlan(channel="opentable", summary="x", details={"url": "http://x"})
    result = await ch.commit(plan)
    assert result.success is False
    assert result.needs_manual is True
    assert result.error == "kill_switch"


@pytest.mark.asyncio
async def test_browser_channel_requires_card_without_payment(tmp_path):
    ch = OpenTableChannel(str(tmp_path))
    plan = CommitPlan(channel="opentable", summary="x",
                      details={"business_name": "X", "url": "http://x"}, requires_card=True)
    result = await ch.commit(plan)
    assert result.success is False
    assert result.error == "card_required"


# ----------------------------------------------------------------- M3: payment

def test_card_service_enforces_hard_cap():
    # A higher configured limit is clamped to $10, and over-cap mints are refused
    # without any network call.
    svc = PrivacyCardService("api-key", limit_usd=50)
    assert svc.limit_usd == 10.0
    assert svc.mint_single_use(amount_usd=25) is None


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

def _june_2026():
    return datetime(2026, 6, 1, 9, 0, 0)  # a Monday, fixed "now" for deterministic dates


def test_calendar_builds_event_with_details():
    captured = {}

    def inserter(calendar_id, event):
        captured["calendar_id"] = calendar_id
        captured["event"] = event
        return "https://calendar.google.com/event?eid=abc"

    svc = CalendarService(calendar_id="primary", inserter=inserter, now=_june_2026)
    link = svc.create_event({
        "business_name": "Lazy Bear", "party_size": 2, "date": "tomorrow", "time": "7pm",
        "address": "3416 19th St", "method": "opentable", "confirmation": "XYZ9",
    })
    assert link.startswith("https://calendar.google.com")
    ev = captured["event"]
    assert "Lazy Bear" in ev["summary"]
    assert ev["location"] == "3416 19th St"
    assert ev["start"]["dateTime"].startswith("2026-06-02T19:00")
    assert "XYZ9" in ev["description"]


def test_calendar_noop_without_inserter():
    svc = CalendarService(inserter=None, now=_june_2026)
    assert svc.create_event({"business_name": "X", "date": "tomorrow", "time": "7pm"}) is None


def test_calendar_skips_when_time_unparseable():
    svc = CalendarService(inserter=lambda c, e: "link", now=_june_2026)
    assert svc.create_event({"business_name": "X", "date": "someday", "time": "later"}) is None


def test_signal_from_env_none_when_unconfigured(monkeypatch):
    for var in ("SIGNAL_CLI_URL", "SIGNAL_FROM_NUMBER", "SIGNAL_TO_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    assert SignalNotifier.from_env() is None


def test_signal_send_builds_payload():
    captured = {}

    class Resp:
        status_code = 201

    def poster(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    n = SignalNotifier("http://localhost:8080", "+1111", "+2222", poster=poster)
    assert n.send("hello") is True
    assert captured["url"].endswith("/v2/send")
    assert captured["json"] == {"message": "hello", "number": "+1111", "recipients": ["+2222"]}


def test_signal_send_never_raises():
    def poster(*a, **k):
        raise RuntimeError("network down")

    n = SignalNotifier("http://x", "+1", "+2", poster=poster)
    assert n.send("hi") is False


class _RecordingCalendar:
    def __init__(self):
        self.facts = None

    def create_event(self, facts):
        self.facts = facts
        return "https://cal/evt"


class _RecordingSignal:
    def __init__(self):
        self.message = None

    def send(self, message):
        self.message = message
        return True


@pytest.mark.asyncio
async def test_confirmed_booking_creates_calendar_and_signal():
    disc = BusinessDiscovery(FakeSearch(opentable_result()), FakeYelp(yelp_business()))
    fake = FakeChannel()
    mgr = make_reservation_manager(disc, router=opentable_router(fake))
    wf = mgr.workflows.workflows["reservations"]
    cal, sig = _RecordingCalendar(), _RecordingSignal()
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
    cal, sig = _RecordingCalendar(), _RecordingSignal()
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
    cal, sig = _RecordingCalendar(), _RecordingSignal()
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
    wf.calendar, wf.notifier = _RecordingCalendar(), _RecordingSignal()

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
    wf.calendar, wf.notifier = _RecordingCalendar(), _RecordingSignal()
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
    # "four of us" and "half past seven" defeat the regexes; the LLM gets them.
    llm = FakeLLM({"business_name": "Lazy Bear", "date": "next Friday",
                   "time": "half past seven", "party_size": "4", "service_type": "dinner"})
    wf = ReservationWorkflow(discovery=BusinessDiscovery(FakeSearch([]), FakeYelp(None)), llm=llm)
    slots = wf._extract_slots("book dinner for four of us at Lazy Bear next Friday at half past seven")
    assert slots["party_size"] == 4
    assert slots["time"] == "half past seven"
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
