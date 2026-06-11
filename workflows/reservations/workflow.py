"""
ReservationWorkflow (M1 + M2).

A ConversationalWorkflow that:
  1. extracts what it can from the opening request,
  2. asks for any missing essentials (business, date, time, party size),
  3. looks the business up and classifies how it takes reservations,
  4. (M2) selects a channel, checks availability, and — for browser-bookable
     methods — presents the plan and waits for the user's explicit approval
     before committing the booking.

Nothing is booked without an explicit "yes", and the kill switch
(RESERVATION_KILL_SWITCH) forces research-only mode. Phone (M5), email (M5b),
payment (M3), calendar/notify (M4), and the sandbox fallback (M7) come later;
for those methods the workflow still just describes the plan.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from ..base import ConversationalWorkflow, WorkflowTrigger
from core.conversation.session import Session, TurnResult
from .calendar import CalendarService
from .channels import AvailabilityStatus, CommitPlan, ReservationChannel, kill_switch_on
from .channels.sandbox_bot import BotCandidate, SandboxBotChannel
from .channels.email import EMAIL_POLL_SECONDS
from .channels.phone import POLL_SECONDS, RETRY_DELAY_SECONDS, PhoneOutcome
from .discovery import BusinessDiscovery
from .llm import ReservationLLM
from .llm import extract_slots as llm_extract_slots
from .models import ESSENTIAL_SLOTS, SLOT_PROMPTS, ChannelDecision, ReservationMethod
from .notify import SignalNotifier
from .payment import PrivacyCardService
from .router import ChannelRouter

_AFFIRMATIVE = {
    "yes", "y", "yeah", "yep", "yup", "correct", "confirm", "confirmed", "go ahead",
    "do it", "book it", "please do", "sounds good", "ok", "okay", "sure",
}

_UNSET = object()

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?|o'?clock)\b", re.I)
_PARTY_RE = re.compile(
    r"\b(?:for|party of|table for|reservation for)\s+(\d{1,3})\b"
    r"|\b(\d{1,3})\s+(?:people|persons|guests|of us|pax)\b",
    re.I,
)
# Capture "at <Name>" / "with <Name>"; the name must start with a letter (so "at 8"
# isn't read as a venue) and stops at the next date/time/qualifier word.
_BUSINESS_RE = re.compile(
    r"\b(?:at|with)\s+([A-Za-z][\w'&.\- ]*?)"
    r"(?=\s+(?:for|on|at|next|this|tomorrow|today|tonight)\b|[?.!,]|$)",
    re.I,
)
_BUSINESS_STOPWORDS = {
    "a table", "the table", "me a table", "a reservation", "the reservation",
    "an appointment", "the appointment", "a booking", "the", "a", "an", "me", "us", "it",
}
_DATE_RE = re.compile(
    r"\b(today|tonight|tomorrow|this (?:weekend|evening|afternoon)|"
    r"(?:next |this )?(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
    re.I,
)


class ReservationWorkflow(ConversationalWorkflow):
    """Make a reservation at an appointment-taking business (restaurant, salon, spa…)."""

    session_timeout_s = 1800  # reservations span far longer than a one-shot command

    def __init__(self, discovery: Optional[BusinessDiscovery] = None,
                 router: Optional[ChannelRouter] = None,
                 payment: Optional[PrivacyCardService] = None,
                 calendar: Optional[CalendarService] = None,
                 notifier: Optional[SignalNotifier] = None,
                 sandbox: Optional[SandboxBotChannel] = None,
                 llm: Any = _UNSET):
        # Sentinel: llm=None explicitly disables LLM refinement (tests); omitting
        # it wires the real one from the environment.
        self.llm = ReservationLLM.from_env() if llm is _UNSET else llm
        self.discovery = discovery or BusinessDiscovery.from_env()
        self.router = router or ChannelRouter.from_env()
        self.sandbox = sandbox if sandbox is not None else SandboxBotChannel.from_env()
        self._pending_sandbox: Dict[str, Tuple[BotCandidate, CommitPlan]] = {}
        self.payment = payment if payment is not None else PrivacyCardService.from_env()
        self.calendar = calendar if calendar is not None else CalendarService.from_env()
        self.notifier = notifier if notifier is not None else SignalNotifier.from_env()
        self.watch_interval_seconds = int(os.getenv("RESERVATION_WATCH_INTERVAL_SECONDS", "1800"))
        self.watch_deadline_days = float(os.getenv("RESERVATION_WATCH_DEADLINE_DAYS", "3"))
        # Transient cache of the prepared (channel, plan) per session, set when we
        # ask for confirmation and consumed on "yes". Rebuilt from slots if lost.
        self._pending: Dict[str, Tuple[ReservationChannel, CommitPlan]] = {}

    @property
    def name(self) -> str:
        return "reservations"

    @property
    def description(self) -> str:
        return (
            "Make a reservation or appointment at a business — looks it up online, "
            "figures out how they take bookings, and arranges it."
        )

    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=[
                "reservation", "reserve", "book a table", "book a reservation",
                "make a booking", "make a reservation", "appointment", "book an appointment",
            ],
            patterns=[
                r"\b(book|reserve|get).{0,20}(table|reservation|appointment|booking)\b",
                r"\bmake (?:me )?(?:a |an )?(reservation|booking|appointment)\b",
                r"\b(get|set up).{0,15}(appointment|booking)\b",
            ],
            examples=[
                "Book me a table for 2 at Lazy Bear next Friday at 7pm",
                "Make a dinner reservation for four tomorrow at 8",
                "Book a haircut appointment at Fellow Barber on Saturday",
                "Reserve a massage for two this weekend",
            ],
        )

    # --------------------------------------------------------------- dialogue
    async def start(self, intent: str, entities: Dict[str, Any], session: Session) -> TurnResult:
        # Off the event loop: extraction may make an LLM call.
        extracted = await asyncio.to_thread(self._extract_slots, intent)
        extracted["raw_request"] = intent
        return await self._advance(session, extracted)

    async def resume(self, text: str, session: Session) -> TurnResult:
        state = session.fsm_state
        if state.startswith("collect_"):
            slot = state[len("collect_"):]
            return await self._advance(session, {slot: self._coerce(slot, text)})
        if state == "confirm":
            return await self._handle_confirmation(text, session)
        if state == "confirm_email":
            return await self._handle_email_confirmation(text, session)
        if state == "confirm_wait":
            return self._handle_wait_confirmation(text, session)
        if state == "confirm_sandbox":
            return await self._handle_sandbox_consent(text, session)
        return TurnResult.complete("Very good, sir.")

    def _handle_wait_confirmation(self, text: str, session: Session) -> TurnResult:
        if not self._is_affirmative(text):
            return TurnResult.cancel("Very well, sir — I'll leave it.")
        now = time.time()
        watch_state = {"started_at": now, "attempts": 0,
                       "deadline": now + self.watch_deadline_days * 86400}
        biz = (session.slots.get("channel_decision") or {}).get("business_name", "them")
        return TurnResult.background(
            f"Right, sir — I'll keep an eye on {biz} and let you know the moment a table opens.",
            wake_at=now + self.watch_interval_seconds,
            slots_update={"watch_state": watch_state},
        )

    async def _advance(self, session: Session, slots_update: Dict[str, Any]) -> TurnResult:
        merged = {**session.slots, **(slots_update or {})}
        missing = [s for s in ESSENTIAL_SLOTS if not merged.get(s)]
        if missing:
            nxt = missing[0]
            return TurnResult.ask(
                SLOT_PROMPTS[nxt], slots_update=slots_update, next_state=f"collect_{nxt}"
            )

        # All essentials present → look the business up (off the event loop).
        decision = await asyncio.to_thread(
            self.discovery.discover, merged["business_name"], merged.get("location")
        )
        base_update = {**(slots_update or {}), "channel_decision": decision.to_dict()}
        return await self._route_and_gate(session, merged, decision, base_update)

    async def _route_and_gate(self, session, slots, decision, base_update) -> TurnResult:
        """Pick a channel; for bookable methods, check availability and gate on confirmation."""
        channel = self.router.select(decision)

        # Research-only, or a method we can't book yet (unknown) → describe & stop.
        if kill_switch_on() or channel is None:
            return TurnResult.complete(self._plan_message(slots, decision, channel is not None),
                                       slots_update=base_update)

        base_update = {**base_update, "active_channel": decision.method.value}

        # Email: gate on an editable draft rather than an availability check.
        if getattr(channel, "is_email", False):
            return await self._gate_email(session, slots, decision, channel, base_update)

        availability = await channel.check_availability(slots)
        if availability.status == AvailabilityStatus.UNAVAILABLE:
            opts = (" Closest I see: " + ", ".join(availability.options)) if availability.options else ""
            return TurnResult.confirm(
                f"I'm afraid {decision.business_name} has nothing for {slots.get('party_size')} "
                f"on {slots.get('date')} at {slots.get('time')}, sir.{opts} "
                f"I can keep checking and book the moment something opens — shall I?",
                slots_update=base_update, next_state="confirm_wait",
            )

        plan = await channel.prepare(slots, decision)
        self._pending[session.session_id] = (channel, plan)
        update = {**base_update, "commit_plan": plan.to_dict()}
        return TurnResult.confirm(self._confirm_message(plan, availability),
                                  slots_update=update, next_state="confirm")

    async def _gate_email(self, session, slots, decision, channel, base_update) -> TurnResult:
        plan = await channel.prepare(slots, decision)
        if not plan.details.get("to"):
            return TurnResult.complete(
                f"I'd email {decision.business_name}, sir, but I don't have their address. "
                f"Could you provide it?", slots_update=base_update)
        self._pending[session.session_id] = (channel, plan)
        update = {**base_update, "commit_plan": plan.to_dict()}
        return TurnResult.confirm(self._email_confirm_message(plan),
                                  slots_update=update, next_state="confirm_email")

    async def _handle_confirmation(self, text: str, session: Session) -> TurnResult:
        if not self._is_affirmative(text):
            self._pending.pop(session.session_id, None)
            return TurnResult.cancel("Understood, sir — I'll hold off and not book anything.")

        channel, plan = self._resolve_pending(session)
        if channel is None or plan is None:
            return TurnResult.complete(
                "I've lost the details of that booking, sir — let's start it again."
            )

        # Phone is asynchronous: place (or defer) the call and go to WAITING.
        if getattr(channel, "is_async", False):
            return await self._start_phone_call(session, channel, plan)

        # Mint a single-use ($10-capped) card only if the booking needs one and we can.
        payment = None
        if plan.requires_card and self.payment is not None:
            payment = await asyncio.to_thread(self.payment.mint_single_use)
            if payment is None:
                self._pending.pop(session.session_id, None)
                return TurnResult.complete(
                    "I couldn't set up a secure one-time card for the deposit, sir, "
                    "so I've not booked it."
                )

        result = await channel.commit(plan, payment=payment)
        self._pending.pop(session.session_id, None)

        if result.success:
            message = result.message + await self._on_confirmed(session, plan, result.confirmation)
            update = {"booking_result": {"success": True, "confirmation": result.confirmation}}
            return TurnResult.complete(message, slots_update=update)

        # Our own automation couldn't do it — optionally offer the sandboxed bot fallback (M7).
        if result.needs_manual:
            offered = await self._maybe_offer_sandbox(session, plan)
            if offered is not None:
                return offered

        return TurnResult.complete(result.message,
                                   slots_update={"booking_result": {"success": False,
                                                                     "needs_manual": True}})

    async def _maybe_offer_sandbox(self, session: Session, plan: CommitPlan) -> Optional[TurnResult]:
        """Offer a vetted third-party bot (sandboxed) as a fallback — with explicit, repo-named consent."""
        if self.sandbox is None or session.slots.get("sandbox_tried"):
            return None
        biz = plan.details.get("business_name")
        candidate = await asyncio.to_thread(self.sandbox.find_bot, biz, plan.channel)
        if candidate is None:
            return None
        self._pending_sandbox[session.session_id] = (candidate, plan)
        return TurnResult.confirm(
            f"My own automation couldn't complete it, sir. I found a community bot, "
            f"{candidate.full_name} ({candidate.stars}★, {candidate.language}). It's third-party "
            f"code, so I'd run it in an isolated sandbox with no access to your system or accounts. "
            f"Shall I try it?",
            slots_update={"sandbox_tried": True, "active_channel": "sandbox",
                          "commit_plan": plan.to_dict(),
                          "sandbox_candidate": {"full_name": candidate.full_name,
                                                "url": candidate.url, "stars": candidate.stars,
                                                "language": candidate.language,
                                                "pushed_at": candidate.pushed_at}},
            next_state="confirm_sandbox",
        )

    async def _handle_sandbox_consent(self, text: str, session: Session) -> TurnResult:
        if not self._is_affirmative(text):
            self._pending_sandbox.pop(session.session_id, None)
            return TurnResult.complete(
                "Understood, sir — I won't run third-party code. You'll need to book this one.")

        pending = self._pending_sandbox.pop(session.session_id, None)
        if pending is not None:
            candidate, plan = pending
        else:  # rebuilt after a restart
            c = session.slots.get("sandbox_candidate") or {}
            plan_d = session.slots.get("commit_plan")
            if not c or not plan_d:
                return TurnResult.complete("I've lost that bot's details, sir — let's start over.")
            candidate = BotCandidate(**c)
            plan = CommitPlan.from_dict(plan_d)

        if self.sandbox is None:
            return TurnResult.complete("The sandbox isn't available right now, sir.")

        result = await asyncio.to_thread(self.sandbox.run_bot, candidate, plan.details)
        biz = plan.details.get("business_name")
        if result.success:
            note = await self._on_confirmed(session, plan, result.confirmation)
            return TurnResult.complete(f"It worked, sir — {biz} is booked via the community bot.{note}")
        return TurnResult.complete(
            f"The community bot couldn't complete it either, sir — best to book "
            f"{biz} yourself: {plan.details.get('url') or 'their site'}.")

    async def _on_confirmed(self, session: Session, plan: CommitPlan,
                            confirmation: Optional[str]) -> str:
        """Calendar event + Signal record on a confirmed booking. Never fails the booking."""
        decision = session.slots.get("channel_decision") or {}
        facts = {
            **plan.details,
            "address": decision.get("address"),
            "method": plan.channel,
            "confirmation": confirmation,
        }

        note = ""
        if self.calendar is not None:
            try:
                if await asyncio.to_thread(self.calendar.create_event, facts):
                    note = " I've added it to your calendar, sir."
            except Exception:
                logger.warning("Calendar event failed", exc_info=True)

        if self.notifier is not None:
            try:
                await asyncio.to_thread(self.notifier.send, self._signal_text(facts))
            except Exception:
                logger.warning("Signal notification failed", exc_info=True)

        return note

    @staticmethod
    def _signal_text(facts: Dict[str, Any]) -> str:
        conf = f" Confirmation: {facts['confirmation']}." if facts.get("confirmation") else ""
        return (
            f"✅ Reservation confirmed: {facts.get('business_name')} for "
            f"{facts.get('party_size')} on {facts.get('date')} at {facts.get('time')}.{conf}"
        )

    # --------------------------------------------------------------- phone (async)
    max_call_retries = 3

    async def _start_phone_call(self, session: Session, channel, plan: CommitPlan) -> TurnResult:
        """On approval: defer to opening if closed, else place the Bland call → WAITING."""
        biz = plan.details.get("business_name")
        open_now, next_open = channel.is_open_now(plan.details, datetime.now())
        phone_state = {"placed": False, "call_ref": None, "retries": 0}

        if not open_now:
            when = (f" around {datetime.fromtimestamp(next_open):%-I %p on %A}"
                    if next_open else " when they open")
            return TurnResult.background(
                f"{biz} is closed just now, sir — I'll ring them{when}.",
                wake_at=next_open or (time.time() + RETRY_DELAY_SECONDS),
                slots_update={"phone_state": phone_state},
            )

        call_ref = await asyncio.to_thread(channel.place_call, plan)
        if call_ref is None:
            return TurnResult.complete(
                f"I wasn't able to place the call to {biz}, sir — you may need to ring them."
            )
        phone_state.update(placed=True, call_ref=call_ref)
        return TurnResult.background(
            f"I'm calling {biz} now, sir — I'll let you know how it goes.",
            wake_at=time.time() + POLL_SECONDS,
            slots_update={"phone_state": phone_state},
        )

    async def on_tick(self, session: Session) -> Optional[TurnResult]:
        if session.slots.get("phone_state"):
            return await self._tick_phone(session)
        if session.slots.get("email_state"):
            return await self._tick_email(session)
        if session.slots.get("watch_state"):
            return await self._tick_watch(session)
        return None

    async def _tick_watch(self, session: Session) -> Optional[TurnResult]:
        """Wait-and-book: re-check availability; when a slot opens, re-gate for approval."""
        ws = session.slots["watch_state"]
        decision = self._decision_from_slots(session)
        channel = self.router.select(decision)
        if channel is None:
            return TurnResult.complete("I can no longer reach that booking channel, sir.")

        now = time.time()
        availability = await channel.check_availability(session.slots)
        if availability.status == AvailabilityStatus.AVAILABLE:
            plan = await channel.prepare(session.slots, decision)
            self._pending[session.session_id] = (channel, plan)
            biz = decision.business_name
            return TurnResult.confirm(
                f"Good news, sir — a table opened at {biz} for {session.slots.get('time')}! "
                + self._confirm_message(plan, availability),
                slots_update={"commit_plan": plan.to_dict(), "watch_state": None},
                next_state="confirm",
            )

        if now > ws["deadline"]:
            biz = decision.business_name
            return TurnResult.complete(
                f"I kept checking {biz} for {self.watch_deadline_days:g} days, sir, but nothing "
                f"opened up. Shall I try another day or place?")
        return TurnResult.background(
            "", wake_at=now + self.watch_interval_seconds,
            slots_update={"watch_state": {**ws, "attempts": ws["attempts"] + 1}})

    async def _tick_phone(self, session: Session) -> Optional[TurnResult]:
        ps = session.slots.get("phone_state")
        channel = self.router.select(ChannelDecision(method=ReservationMethod.PHONE))
        if channel is None:
            return TurnResult.complete("I've lost the phone connection for that booking, sir.")
        plan = CommitPlan.from_dict(session.slots["commit_plan"])
        biz = plan.details.get("business_name")
        now = time.time()

        # Deferred (closed): place the call once it's open.
        if not ps.get("placed"):
            open_now, next_open = channel.is_open_now(plan.details, datetime.now())
            if not open_now:
                return TurnResult.background("", wake_at=next_open or (now + RETRY_DELAY_SECONDS))
            call_ref = await asyncio.to_thread(channel.place_call, plan)
            if call_ref is None:
                return self._phone_retry(ps, biz, now)
            return TurnResult.background(
                f"I'm calling {biz} now, sir.", wake_at=now + POLL_SECONDS,
                slots_update={"phone_state": {**ps, "placed": True, "call_ref": call_ref}})

        # Placed: poll for the outcome.
        result = await asyncio.to_thread(channel.poll, ps["call_ref"])
        if result.outcome == PhoneOutcome.PENDING:
            return TurnResult.background("", wake_at=now + POLL_SECONDS)
        return await self._map_phone_outcome(session, plan, ps, result, biz, now)

    async def _map_phone_outcome(self, session, plan, ps, result, biz, now) -> TurnResult:
        o = result.outcome
        if o == PhoneOutcome.CONFIRMED:
            note = await self._on_confirmed(session, plan, result.confirmation)
            when = result.negotiated_time or plan.details.get("time")
            return TurnResult.complete(f"Booked, sir — {biz} is confirmed for {when}.{note}")
        if o == PhoneOutcome.NO_AVAILABILITY:
            return TurnResult.complete(
                f"{biz} had nothing for your time, sir. Shall I try another day or place?")
        if o == PhoneOutcome.EMAIL_REQUESTED:
            return await self._handoff_to_email(session, plan, result.email, biz)
        if o == PhoneOutcome.CALLBACK_REQUIRED:
            return TurnResult.complete(
                f"{biz} said they'd call back, sir — I'll keep an ear out.")
        if o == PhoneOutcome.NEEDS_INFO:
            q = f' They asked: "{result.question}".' if result.question else ""
            return TurnResult.complete(
                f"{biz} needs more from us before booking, sir.{q} How would you like to proceed?")
        # FAILED → retry within limits.
        return self._phone_retry(ps, biz, now)

    def _phone_retry(self, ps: Dict[str, Any], biz: str, now: float) -> TurnResult:
        retries = ps.get("retries", 0) + 1
        if retries > self.max_call_retries:
            return TurnResult.complete(
                f"I couldn't reach {biz} after {self.max_call_retries} tries, sir — "
                f"you may want to call them directly.")
        return TurnResult.background(
            "", wake_at=now + RETRY_DELAY_SECONDS,
            slots_update={"phone_state": {"placed": False, "call_ref": None, "retries": retries}})

    # --------------------------------------------------------------- email (async)
    email_deadline_days = 5

    async def _handoff_to_email(self, session, phone_plan, email_addr, biz) -> TurnResult:
        """A phone call asked us to email instead → draft a request and gate it for approval."""
        channel = self.router.select(ChannelDecision(method=ReservationMethod.EMAIL))
        if channel is None or not email_addr:
            return TurnResult.complete(
                f"{biz} asked to be emailed, sir" +
                (f" at {email_addr}" if email_addr else "") +
                ", but I can't send email right now — you may want to write to them.")
        decision = self._decision_from_slots(session)
        decision.email = email_addr
        plan = await channel.prepare(session.slots, decision)
        self._pending[session.session_id] = (channel, plan)
        # Promote this backgrounded session to an active confirmation dialogue.
        return TurnResult.confirm(
            f"{biz} asked to be emailed, sir. " + self._email_confirm_message(plan),
            slots_update={"active_channel": "email", "business_email": email_addr,
                          "commit_plan": plan.to_dict(), "phone_state": None},
            next_state="confirm_email",
        )

    async def _handle_email_confirmation(self, text: str, session: Session) -> TurnResult:
        channel, plan = self._resolve_pending(session)
        if channel is None or plan is None:
            return TurnResult.complete("I've lost that draft, sir — let's start it again.")

        low = text.strip().lower().rstrip("!.")
        if self._is_affirmative(text):
            sent = await asyncio.to_thread(channel.send, plan)
            if not sent:
                self._pending.pop(session.session_id, None)
                return TurnResult.complete("I couldn't send that email, sir.")
            biz = plan.details.get("business_name")
            email_state = {"to": plan.details["to"], "subject": plan.details["subject"],
                           "sent_at": time.time(), "nudged": False}
            return TurnResult.background(
                f"I've emailed {biz}, sir — I'll let you know when they reply.",
                wake_at=time.time() + EMAIL_POLL_SECONDS,
                slots_update={"email_state": email_state})

        if low in ("no", "don't", "do not", "cancel", "stop", "nope", "no thanks", "scrap it"):
            self._pending.pop(session.session_id, None)
            return TurnResult.cancel("Understood, sir — I won't send it.")

        # Anything else is treated as an edit instruction → re-draft and re-confirm.
        decision = self._decision_from_slots(session)
        subject, body = await asyncio.to_thread(
            channel.drafter, session.slots, decision, instruction=text)
        plan.details["subject"], plan.details["body"] = subject, body
        self._pending[session.session_id] = (channel, plan)
        return TurnResult.confirm(
            "I've revised it, sir. " + self._email_confirm_message(plan),
            slots_update={"commit_plan": plan.to_dict()}, next_state="confirm_email")

    async def _tick_email(self, session: Session) -> Optional[TurnResult]:
        es = session.slots.get("email_state")
        channel = self.router.select(ChannelDecision(method=ReservationMethod.EMAIL))
        if channel is None:
            return TurnResult.complete("I've lost the email connection for that request, sir.")
        plan = CommitPlan.from_dict(session.slots["commit_plan"])
        biz = plan.details.get("business_name")
        now = time.time()

        reply = await asyncio.to_thread(channel.poll_reply, es["to"], es["subject"])
        outcome = reply.outcome

        if outcome.value == "pending":
            # Deadline / one-time nudge handling.
            age_days = (now - es["sent_at"]) / 86400
            if age_days >= self.email_deadline_days:
                return TurnResult.complete(
                    f"I've had no reply from {biz} after {self.email_deadline_days} days, sir — "
                    f"shall I try calling instead?")
            return TurnResult.background("", wake_at=now + EMAIL_POLL_SECONDS)

        if outcome.value == "confirmed":
            note = await self._on_confirmed(session, plan, reply.confirmation)
            return TurnResult.complete(f"Good news, sir — {biz} confirmed by email.{note}")
        if outcome.value == "declined":
            return TurnResult.complete(
                f"{biz} replied that they can't accommodate it, sir. Try another day or place?")
        if outcome.value == "asks_to_call":
            return TurnResult.complete(
                f"{biz} asked us to call to finalise, sir. Shall I ring them?")
        # needs_info
        return TurnResult.complete(
            f"{biz} replied with a question, sir: \"{reply.text[:200]}\" How shall I respond?")

    def _decision_from_slots(self, session: Session) -> ChannelDecision:
        d = session.slots.get("channel_decision") or {}
        try:
            method = ReservationMethod(d.get("method", "unknown"))
        except ValueError:
            method = ReservationMethod.UNKNOWN
        return ChannelDecision(
            method=method,
            business_name=d.get("business_name") or session.slots.get("business_name"),
            url=d.get("url"), phone=d.get("phone"),
            email=d.get("email") or session.slots.get("business_email"),
            address=d.get("address"),
        )

    @staticmethod
    def _email_confirm_message(plan: CommitPlan) -> str:
        d = plan.details
        return (f"I've drafted this to {d.get('to')} — subject \"{d.get('subject')}\". "
                f"Shall I send it, sir? (Or tell me what to change.)")

    def _resolve_pending(self, session: Session):
        pending = self._pending.get(session.session_id)
        if pending is not None:
            return pending
        # Rebuilt after a restart from the persisted plan + active channel.
        plan_d = session.slots.get("commit_plan")
        method_s = session.slots.get("active_channel") or (
            session.slots.get("channel_decision") or {}).get("method")
        if not plan_d or not method_s:
            return None, None
        try:
            method = ReservationMethod(method_s)
        except ValueError:
            return None, None
        channel = self.router.select(ChannelDecision(method=method))
        return (channel, CommitPlan.from_dict(plan_d)) if channel else (None, None)

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        t = text.strip().lower().rstrip("!.")
        if t in _AFFIRMATIVE:
            return True
        first = t.split()[0] if t else ""
        if first in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm"}:
            return True
        return any(p in t for p in ("go ahead", "book it", "do it", "please do", "sounds good"))

    # --------------------------------------------------------------- messaging
    @staticmethod
    def _plan_message(slots: Dict[str, Any], decision: ChannelDecision,
                      bookable: bool = False) -> str:
        """Used when we can't gate a booking (research-only, or phone/email/unknown method)."""
        name = decision.business_name or slots.get("business_name", "the business")
        addr = f" ({decision.address})" if decision.address else ""
        party = slots.get("party_size", "your party")
        date = slots.get("date", "the requested date")
        time = slots.get("time", "the requested time")

        if bookable and kill_switch_on():
            tail = "I'm in research-only mode (kill switch on), so I won't book it."
        else:
            tail = "I can't place this type of booking automatically yet, so I've not booked anything."

        return (
            f"I found {name}{addr}, sir. {decision.method_phrase()} "
            f"You've asked for {party} on {date} at {time}. {tail}"
        )

    @staticmethod
    def _confirm_message(plan: CommitPlan, availability) -> str:
        name = plan.details.get("business_name", "the business")
        party = plan.details.get("party_size", "your party")
        date = plan.details.get("date")
        time = plan.details.get("time")
        verify = f" ({availability.note})" if availability.note else ""
        card = ""
        if plan.requires_card:
            card = " They may need a card to hold it; I'd use a single-use card capped at $10."
        return (
            f"I'm ready to book {name} for {party} on {date} at {time} via {plan.channel}.{verify}"
            f"{card} Shall I go ahead and book it, sir?"
        )

    # --------------------------------------------------------------- extraction
    def _extract_slots(self, text: str) -> Dict[str, Any]:
        """Parse the opening request: regex baseline, refined by the LLM when
        available (handles e.g. "half past seven", implicit party sizes).
        Anything missed is asked for."""
        slots = self._extract_slots_regex(text)
        refined = llm_extract_slots(self.llm, text)
        if refined:
            slots.update(refined)
        return slots

    def _extract_slots_regex(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}

        m = _PARTY_RE.search(text)
        if m:
            slots["party_size"] = int(m.group(1) or m.group(2))

        m = _TIME_RE.search(text)
        if m:
            slots["time"] = m.group(0).strip()

        m = _DATE_RE.search(text)
        if m:
            slots["date"] = m.group(0).strip()

        m = _BUSINESS_RE.search(text)
        if m:
            name = m.group(1).strip(" .,'\"").strip()
            lowered = name.lower()
            # Reject filler captures ("a table", "make a reservation"...).
            is_filler = (
                lowered in _BUSINESS_STOPWORDS
                or any(w in lowered for w in ("reservation", "booking", "appointment"))
                or len(lowered) < 2
            )
            if name and not is_filler:
                slots["business_name"] = name

        return slots

    @staticmethod
    def _coerce(slot: str, text: str) -> Any:
        text = text.strip()
        if slot == "party_size":
            m = re.search(r"\d{1,3}", text)
            if m:
                return int(m.group(0))
        return text
