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
from typing import Any, Dict, Optional

from ..base import ConversationalWorkflow, WorkflowTrigger
from core.conversation.session import Session, TurnResult
from core.harness import (
    Action,
    ActionGate,
    ActionKind,
    AuditLog,
    ConfirmDecision,
    EgressViolation,
    IllegalTransition,
    Machine,
    NormalizeCtx,
    Refusal,
    Sink,
    SinkMode,
    display_date,
    display_time,
    guard,
    normalize_time,
    parse_confirmation,
)
from .calendar import CalendarService
from .channels import AvailabilityStatus, CommitPlan, kill_switch_on
from .channels.sandbox_bot import BotCandidate, SandboxBotChannel
from .channels.email import EMAIL_POLL_SECONDS
from .channels.phone import POLL_SECONDS, RETRY_DELAY_SECONDS, PhoneOutcome
from .discovery import BusinessDiscovery
from .llm import ReservationLLM
from .llm import extract_slots as llm_extract_slots
from .models import (
    ESSENTIAL_SLOTS,
    SLOT_PROMPTS,
    SLOT_SPECS_BY_NAME,
    ChannelDecision,
    ReservationMethod,
)
from .notify import TelegramNotifier
from .payment import HARD_CAP_USD, PrivacyCardService, card_service_from_env
from .router import ChannelRouter
from .snipe import (
    compute_release_fire_ts,
    describe_fire,
    resolve_timezone,
    timezone_for_location,
)

_UNSET = object()

logger = logging.getLogger(__name__)

# Outbound boundaries owned by the workflow itself. The email body is free
# text the user approved; the scan stops card/secret-shaped content only.
SMTP_SINK = Sink("smtp", SinkMode.SCAN)
# Calendar + Telegram go to the user's own destinations; scan-only.
USER_DEST_SINK = Sink("user_destination", SinkMode.SCAN)

# The reservation dialogue as a closed state machine (spec §6 / harness §6).
# Terminal outcomes (DONE/CANCELLED) are session *statuses*, not states; the
# gate states are the only places a "yes" can become an approval.
RESERVATION_MACHINE = Machine(
    name="reservations",
    states=frozenset({
        "start",            # opening turn
        "collecting",       # asking for the next essential slot
        "confirm",          # booking gate (browser/phone commit)
        "confirm_email",    # editable email-draft gate
        "confirm_wait",     # offer to watch for an opening
        "confirm_snipe",    # approve a scheduled snipe at the release time
        "confirm_sandbox",  # repo-named consent gate for untrusted code
        "calling",          # WAITING: call placed/deferred, polling Bland
        "awaiting_reply",   # WAITING: email sent, polling the thread
        "watching",         # WAITING: re-checking availability
        "scheduled",        # WAITING: parked until the reservation window opens
    }),
    initial="start",
    gate_states=frozenset({"confirm", "confirm_email", "confirm_wait",
                           "confirm_snipe", "confirm_sandbox"}),
    transitions={
        ("start", "need_slot"): "collecting",
        ("start", "gate_booking"): "confirm",
        ("start", "gate_email"): "confirm_email",
        ("start", "offer_wait"): "confirm_wait",
        ("start", "gate_snipe"): "confirm_snipe",
        ("collecting", "need_slot"): "collecting",
        ("collecting", "gate_booking"): "confirm",
        ("collecting", "gate_email"): "confirm_email",
        ("collecting", "offer_wait"): "confirm_wait",
        ("collecting", "gate_snipe"): "confirm_snipe",
        ("confirm", "call_started"): "calling",
        ("confirm", "offer_sandbox"): "confirm_sandbox",
        ("confirm_wait", "watch"): "watching",
        ("confirm_snipe", "schedule"): "scheduled",
        ("watching", "slot_opened"): "confirm",
        ("calling", "gate_email"): "confirm_email",
        ("confirm_email", "redraft"): "confirm_email",
        ("confirm_email", "sent"): "awaiting_reply",
    },
)

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
# Release policy: "tables drop 30 days in advance at 10am ET".
_RELEASE_DAYS_RE = re.compile(
    r"\b(\d{1,3})\s*days?\s*(?:in advance|ahead|before|out|prior|early)\b", re.I)
_RELEASE_CTX_RE = re.compile(r"(?:open|drop|release|go live|available|bookable)\w*", re.I)
_RELEASE_TZ_RE = re.compile(r"\b(ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|GMT|UTC)\b")


class ReservationWorkflow(ConversationalWorkflow):
    """Make a reservation at an appointment-taking business (restaurant, salon, spa…)."""

    session_timeout_s = 1800  # reservations span far longer than a one-shot command

    # Purged by the framework when the session ends (spec §8 L2). Booking facts
    # (business, date, time, confirmation) are kept for "what did you book?".
    pii_slots = ("guest_name", "phone", "email", "special_requests", "raw_request")

    def __init__(self, discovery: Optional[BusinessDiscovery] = None,
                 router: Optional[ChannelRouter] = None,
                 payment: Optional[PrivacyCardService] = None,
                 calendar: Optional[CalendarService] = None,
                 notifier: Optional[TelegramNotifier] = None,
                 sandbox: Optional[SandboxBotChannel] = None,
                 gate: Optional[ActionGate] = None,
                 llm: Any = _UNSET):
        # Sentinel: llm=None explicitly disables LLM refinement (tests); omitting
        # it wires the real one from the environment.
        self.llm = ReservationLLM.from_env() if llm is _UNSET else llm
        self.discovery = discovery or BusinessDiscovery.from_env()
        self.router = router or ChannelRouter.from_env()
        self.sandbox = sandbox if sandbox is not None else SandboxBotChannel.from_env()
        self.payment = payment if payment is not None else card_service_from_env()
        self.calendar = calendar if calendar is not None else CalendarService.from_env()
        self.notifier = notifier if notifier is not None else TelegramNotifier.from_env()
        # Every irreversible action (book / call / email / mint / sandbox) goes
        # through the gate; the (channel, plan) under approval lives in
        # session.slots, never in process memory. Approvals are only valid if
        # recorded in one of the machine's gate states.
        self.machine = RESERVATION_MACHINE
        self.gate = gate or ActionGate.with_defaults(
            kill_switch_env="RESERVATION_KILL_SWITCH", audit=AuditLog.from_env(),
            gate_states=self.machine.gate_states)
        self.watch_interval_seconds = int(os.getenv("RESERVATION_WATCH_INTERVAL_SECONDS", "1800"))
        self.watch_deadline_days = float(os.getenv("RESERVATION_WATCH_DEADLINE_DAYS", "3"))
        # Short-term cross-request memory: the WHEN/HOW-MANY of the last
        # reservation a user described, so a quick follow-up ("...book Flores
        # instead") reuses the date/time/party they just gave. Keyed by user;
        # expires fast so it never leaks into an unrelated booking later.
        self._recent: Dict[str, Dict[str, Any]] = {}
        self._recent_ttl_s = float(os.getenv("RESERVATION_CONTEXT_TTL_S", "900"))
        # Sniping (Phase 1): when the user supplies a release policy, park the
        # booking until the window opens and race it. Tunables are env-driven.
        self._snipe_enabled = os.getenv("RESERVATION_SNIPE_ENABLED", "true").lower() \
            in ("1", "true", "yes")
        self._snipe_lead_s = float(os.getenv("RESERVATION_SNIPE_LEAD_SECONDS", "5"))
        self._snipe_window_s = float(os.getenv("RESERVATION_SNIPE_RETRY_BUDGET_S", "90"))
        self._snipe_retry_gap_s = float(os.getenv("RESERVATION_SNIPE_RETRY_GAP_SECONDS", "8"))
        self._snipe_max_attempts = int(os.getenv("RESERVATION_SNIPE_MAX_ATTEMPTS", "6"))
        # Auto-detect the release policy (Phase 2) when the user didn't state one
        # and the dining date is far enough out to be worth checking.
        self._snipe_autodetect = os.getenv("RESERVATION_SNIPE_AUTODETECT", "true").lower() \
            in ("1", "true", "yes")
        self._snipe_autodetect_min_days = float(
            os.getenv("RESERVATION_SNIPE_AUTODETECT_MIN_DAYS", "14"))
        self._snipe_min_confidence = float(os.getenv("RESERVATION_SNIPE_MIN_CONFIDENCE", "0.4"))

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
        try:
            # Off the event loop: extraction may make an LLM call.
            extracted = await asyncio.to_thread(self._extract_slots, intent)
        except EgressViolation:
            return TurnResult.complete(
                "I stopped that request before sending it out, sir — it contained "
                "something that mustn't leave this machine. Best rephrase it without "
                "card or credential details.")
        # Honour structured entities the router/context layer resolved, but never
        # let them override what this message actually said.
        for key, value in self._merge_entities(entities).items():
            extracted.setdefault(key, value)
        # Inherit any unspecified date/time/party from a very recent request.
        extracted = self._inherit_recent(session, extracted)
        extracted["raw_request"] = intent
        return await self._advance(session, extracted)

    @staticmethod
    def _merge_entities(entities: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize handed-in entities into slot values (essentials canonicalized,
        free-text fields passed through). Unusable values are dropped, not guessed."""
        if not entities:
            return {}
        out: Dict[str, Any] = {}
        ctx = NormalizeCtx.fresh()
        for name in ESSENTIAL_SLOTS:
            raw = entities.get(name)
            if raw is None:
                continue
            value = SLOT_SPECS_BY_NAME[name].normalize(str(raw), ctx)
            if value is not None:
                out[name] = value
        for name in ("location", "email", "phone", "special_requests"):
            if entities.get(name):
                out[name] = entities[name]
        return out

    def _inherit_recent(self, session: Session, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing date/time/party from the user's last reservation request,
        if it was recent. WHERE is never inherited — only WHEN and HOW MANY."""
        rec = self._recent.get(session.user_id)
        if not rec or (time.time() - rec["at"]) > self._recent_ttl_s:
            return extracted
        for key, value in rec["slots"].items():
            extracted.setdefault(key, value)
        return extracted

    def _remember_recent(self, session: Session, slots: Dict[str, Any]) -> None:
        keep = {k: slots[k] for k in ("date", "time", "party_size") if slots.get(k)}
        if keep:
            self._recent[session.user_id] = {"slots": keep, "at": time.time()}

    async def resume(self, text: str, session: Session) -> TurnResult:
        handler = {
            "collecting": self._handle_collect,
            "confirm": self._handle_confirmation,
            "confirm_email": self._handle_email_confirmation,
            "confirm_wait": self._handle_wait_confirmation,
            "confirm_snipe": self._handle_snipe_confirmation,
            "confirm_sandbox": self._handle_sandbox_consent,
        }.get(session.fsm_state)
        if handler is None:
            return TurnResult.complete("Very good, sir.")
        try:
            return await handler(text, session)
        except IllegalTransition:
            logger.exception("Illegal FSM transition in session %s", session.session_id)
            return TurnResult.complete(
                "I've hit an internal snag with that task, sir — best we start it afresh.")

    async def _handle_collect(self, text: str, session: Session) -> TurnResult:
        slot = session.slots.get("_pending_slot") or ""
        spec = SLOT_SPECS_BY_NAME.get(slot)
        if spec is None:
            return await self._advance(session, {})
        value = spec.normalize(text, NormalizeCtx.fresh())
        if value is None:
            # The answer didn't canonicalize (bad date, time, number…) — re-ask
            # rather than store a raw string downstream code would guess at.
            return TurnResult.ask(spec.reask,
                                  next_state=self._fire(session, "need_slot"))
        return await self._advance(session, {slot: value})

    def _fire(self, session: Session, event: str) -> str:
        """Validated state transition: the next fsm_state, or IllegalTransition."""
        return self.machine.next(session.fsm_state, event)

    async def _handle_wait_confirmation(self, text: str, session: Session) -> TurnResult:
        decision = parse_confirmation(text)
        if decision == ConfirmDecision.UNCLEAR:
            return self._reask(session, "Shall I keep watching for an opening, sir — yes or no?")
        if decision != ConfirmDecision.YES:
            return TurnResult.cancel("Very well, sir — I'll leave it.")
        now = time.time()
        watch_state = {"started_at": now, "attempts": 0,
                       "deadline": now + self.watch_deadline_days * 86400}
        biz = (session.slots.get("channel_decision") or {}).get("business_name", "them")
        return TurnResult.background(
            f"Right, sir — I'll keep an eye on {biz} and let you know the moment a table opens.",
            wake_at=now + self.watch_interval_seconds,
            slots_update={"watch_state": watch_state},
            next_state=self._fire(session, "watch"),
        )

    async def _advance(self, session: Session, slots_update: Dict[str, Any]) -> TurnResult:
        merged = {**session.slots, **(slots_update or {})}
        missing = [s for s in ESSENTIAL_SLOTS if not merged.get(s)]
        if missing:
            nxt = missing[0]
            return TurnResult.ask(
                SLOT_PROMPTS[nxt],
                slots_update={**(slots_update or {}), "_pending_slot": nxt},
                next_state=self._fire(session, "need_slot"),
            )

        # All essentials present → remember the WHEN/HOW-MANY for a quick
        # follow-up, then look the business up (off the event loop).
        self._remember_recent(session, merged)
        try:
            decision = await asyncio.to_thread(
                self.discovery.discover, merged["business_name"], merged.get("location")
            )
        except EgressViolation:
            return TurnResult.complete(
                "I stopped that lookup, sir — it was about to include something "
                "it shouldn't. Could you give me just the business name and city?")
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

        # Snipe: a release policy (stated by the user, or auto-detected) places
        # the window in the future → schedule the attempt rather than try (and
        # fail) now. Phone is excluded for now (this covers browser booking).
        if not getattr(channel, "is_async", False):
            snipe = await self._resolve_snipe(slots, decision)
            if snipe is not None:
                plan = await channel.prepare(slots, decision)
                update = {**base_update, "commit_plan": plan.to_dict(), "snipe_plan": snipe}
                return TurnResult.confirm(self._snipe_confirm_message(plan, snipe),
                                          slots_update=update,
                                          next_state=self._fire(session, "gate_snipe"))

        availability = await channel.check_availability(slots)
        if availability.status == AvailabilityStatus.UNAVAILABLE:
            opts = (" Closest I see: " + ", ".join(availability.options)) if availability.options else ""
            return TurnResult.confirm(
                f"I'm afraid {decision.business_name} has nothing for {slots.get('party_size')} "
                f"on {display_date(slots.get('date'))} at {display_time(slots.get('time'))}, "
                f"sir.{opts} I can keep checking and book the moment something opens — shall I?",
                slots_update=base_update, next_state=self._fire(session, "offer_wait"),
            )

        plan = await channel.prepare(slots, decision)
        update = {**base_update, "commit_plan": plan.to_dict()}
        return TurnResult.confirm(self._confirm_message(plan, availability),
                                  slots_update=update,
                                  next_state=self._fire(session, "gate_booking"))

    async def _gate_email(self, session, slots, decision, channel, base_update) -> TurnResult:
        plan = await channel.prepare(slots, decision)
        if not plan.details.get("to"):
            return TurnResult.complete(
                f"I'd email {decision.business_name}, sir, but I don't have their address. "
                f"Could you provide it?", slots_update=base_update)
        update = {**base_update, "commit_plan": plan.to_dict()}
        return TurnResult.confirm(self._email_confirm_message(plan),
                                  slots_update=update,
                                  next_state=self._fire(session, "gate_email"))

    async def _handle_confirmation(self, text: str, session: Session) -> TurnResult:
        decision = parse_confirmation(text)
        if decision == ConfirmDecision.UNCLEAR:
            # A question or an aside is not a "no" — re-ask once, never approve.
            return self._reask(session, "Just to be clear, sir — shall I book it, yes or no?")
        if decision != ConfirmDecision.YES:
            return TurnResult.cancel("Understood, sir — I'll hold off and not book anything.")

        channel, plan = self._resolve_pending(session)
        if channel is None or plan is None:
            return TurnResult.complete(
                "I've lost the details of that booking, sir — let's start it again."
            )

        # Phone is asynchronous: place (or defer) the call and go to WAITING.
        if getattr(channel, "is_async", False):
            call_action = self._action(ActionKind.PLACE_CALL, session, plan)
            # One approval covers the bounded redials of this exact call plan,
            # and survives an off-hours deferral to the next opening.
            self.gate.record_approval(session, call_action,
                                      ttl_s=self._call_ttl_s(), max_uses=self.max_call_retries + 1)
            return await self._start_phone_call(session, channel, plan)

        book_action = self._action(ActionKind.BOOK, session, plan)
        self.gate.record_approval(session, book_action)
        # The confirm message disclosed the card; the same yes covers the mint.
        if plan.requires_card and self.payment is not None:
            self.gate.record_approval(session, self._mint_action(session, plan))

        payment, mint_error = await self._execute_mint(session, plan)
        if mint_error is not None:
            return mint_error

        outcome = await self.gate.execute(
            book_action, session, lambda: channel.commit(plan, payment=payment))
        if not outcome.ok:
            return TurnResult.complete(self._refusal_message(outcome.refusal))
        result = outcome.result

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

    async def _execute_mint(self, session: Session, plan: CommitPlan):
        """Run the (already-approved) single-use card mint when one is needed.
        Returns (payment, error_turn): payment is None when no card is required;
        error_turn is a terminal TurnResult when the mint is refused or fails."""
        if not (plan.requires_card and self.payment is not None):
            return None, None
        mint_action = self._mint_action(session, plan)
        minted = await self.gate.execute(
            mint_action, session,
            lambda: asyncio.to_thread(self.payment.mint_single_use))
        if not minted.ok:
            return None, TurnResult.complete(self._refusal_message(minted.refusal))
        if minted.result is None:
            return None, TurnResult.complete(
                "I couldn't set up a secure one-time card for the deposit, sir, "
                "so I've not booked it.")
        return minted.result, None

    # --------------------------------------------------------------- snipe (Phase 1+2)
    async def _resolve_snipe(self, slots: Dict[str, Any], decision) -> Optional[Dict[str, Any]]:
        """The snipe plan for this booking, or None to book the normal way.

        Prefers a policy the user stated; otherwise (Phase 2) auto-detects one
        from the web/Reddit when the dining date is far enough out to be worth
        the lookup. Either way, only returns a plan when the window is still in
        the future."""
        manual = self._snipe_from_manual(slots, decision)
        if manual is not None:
            return manual
        if not (self._snipe_enabled and self._snipe_autodetect):
            return None
        date_iso = slots.get("date")
        if not date_iso or self._days_until(date_iso) < self._snipe_autodetect_min_days:
            return None
        business = (getattr(decision, "business_name", None) or slots.get("business_name"))
        try:
            policy = await asyncio.to_thread(
                self.discovery.resolve_release_policy, business, slots.get("location"))
        except Exception:
            logger.warning("Release-policy auto-detection failed", exc_info=True)
            return None
        if policy is None or policy.confidence < self._snipe_min_confidence:
            return None
        return self._build_snipe(date_iso, policy.days_in_advance, policy.release_time,
                                 policy.timezone, decision, slots, source="web",
                                 confidence=policy.confidence, quote=policy.source_quote)

    def _snipe_from_manual(self, slots: Dict[str, Any], decision) -> Optional[Dict[str, Any]]:
        if not self._snipe_enabled:
            return None
        return self._build_snipe(slots.get("date"), slots.get("release_days_ahead"),
                                 slots.get("release_time"), slots.get("release_timezone"),
                                 decision, slots, source="you")

    def _build_snipe(self, date_iso, days, release_time_raw, tz_name, decision, slots, *,
                     source: str, confidence: float = 1.0, quote: str = ""
                     ) -> Optional[Dict[str, Any]]:
        """Normalize a (date, lead-days, release-time, tz) into a snipe plan, or
        None if incomplete/unparseable or the window has already opened."""
        if not (days and release_time_raw and date_iso):
            return None
        release_hhmm = normalize_time(str(release_time_raw))
        if not release_hhmm:
            return None
        # When the release tz isn't stated, prefer the *venue's* zone (from its
        # address/city) over the user's own — a NYC drop opens on NYC time.
        venue_tz = getattr(decision, "timezone", None) or timezone_for_location(
            getattr(decision, "address", None) or (slots or {}).get("location"))
        tz = resolve_timezone(tz_name, venue_tz)
        fire_ts = compute_release_fire_ts(date_iso, int(days), release_hhmm, tz)
        if fire_ts is None or fire_ts <= time.time() + self._snipe_lead_s:
            return None   # bad input, or the window is already open → just book now
        return {"fire_ts": fire_ts, "display": describe_fire(fire_ts, tz),
                "days_ahead": int(days), "release_hhmm": release_hhmm, "tz": str(tz),
                "source": source, "confidence": round(float(confidence), 2),
                "quote": (quote or "")[:300]}

    @staticmethod
    def _days_until(date_iso: str) -> float:
        from datetime import date
        try:
            return (date.fromisoformat(date_iso) - date.today()).days
        except (ValueError, TypeError):
            return 0.0

    async def _handle_snipe_confirmation(self, text: str, session: Session) -> TurnResult:
        decision = parse_confirmation(text)
        if decision == ConfirmDecision.UNCLEAR:
            return self._reask(session, "Shall I set up the timed booking, sir — yes or no?")
        if decision != ConfirmDecision.YES:
            return TurnResult.cancel("Very well, sir — I'll not schedule it.")

        channel, plan = self._resolve_pending(session)
        snipe = session.slots.get("snipe_plan")
        if channel is None or plan is None or not snipe:
            return TurnResult.complete(
                "I've lost the details of that booking, sir — let's start it again.")

        now = time.time()
        fire_ts = float(snipe["fire_ts"])
        # Approvals must outlive the wait — keep them valid until the retry
        # window closes (with a margin), and allow one use per planned attempt.
        ttl = (fire_ts - now) + self._snipe_window_s + 300
        book_action = self._action(ActionKind.BOOK, session, plan)
        self.gate.record_approval(session, book_action, ttl_s=ttl,
                                  max_uses=self._snipe_max_attempts + 1)
        if plan.requires_card and self.payment is not None:
            self.gate.record_approval(session, self._mint_action(session, plan), ttl_s=ttl)

        biz = plan.details.get("business_name")
        snipe_state = {"fire_ts": fire_ts, "deadline_ts": fire_ts + self._snipe_window_s,
                       "attempts": 0}
        return TurnResult.background(
            f"Right, sir — I'll stand ready and book {biz} the moment reservations open "
            f"({snipe['display']}). I'll message you on Telegram either way.",
            wake_at=max(now, fire_ts - self._snipe_lead_s),
            slots_update={"snipe_state": snipe_state},
            next_state=self._fire(session, "schedule"))

    # Errors that won't fix themselves on a retry within the same window.
    _SNIPE_RETRYABLE = frozenset({"slot_not_found", "no_slot"})

    async def _tick_snipe(self, session: Session) -> Optional[TurnResult]:
        """At (or after) the release instant: provision the card once, then race
        the booking with bounded retries until it lands or the window closes."""
        ss = session.slots["snipe_state"]
        now = time.time()
        if now < ss["fire_ts"] - 2:
            return TurnResult.background("", wake_at=ss["fire_ts"])  # final precise wait

        channel, plan = self._resolve_pending(session)
        biz = (plan.details.get("business_name") if plan else None) or "the venue"
        if channel is None or plan is None:
            return TurnResult.complete(
                f"I lost the booking details for {biz}, sir — let's redo it.",
                slots_update={"snipe_state": None})

        remaining = ss["fire_ts"] - time.time()
        if remaining > 0:
            await asyncio.sleep(remaining)

        payment, mint_error = await self._execute_mint(session, plan)
        if mint_error is not None:
            await self._snipe_notify(session, f"⏰ Couldn't book {biz} — card setup failed, sir.")
            return mint_error

        attempt = int(ss.get("attempts", 0))
        while time.time() <= ss["deadline_ts"]:
            book_action = self._action(ActionKind.BOOK, session, plan, attempt=attempt)
            outcome = await self.gate.execute(
                book_action, session, lambda: channel.commit(plan, payment=payment))
            attempt += 1
            if not outcome.ok:
                msg = self._refusal_message(outcome.refusal)
                await self._snipe_notify(session, f"⏰ Couldn't book {biz}, sir — {msg}")
                return self._snipe_failed(msg)
            result = outcome.result
            if result.success:
                note = await self._on_confirmed(session, plan, result.confirmation)
                return TurnResult.complete(
                    f"Got it, sir — {biz} is booked.{note}",
                    slots_update={"snipe_state": None,
                                  "booking_result": {"success": True,
                                                     "confirmation": result.confirmation}})
            if result.error not in self._SNIPE_RETRYABLE:
                # login/card/unconfirmed → don't hammer it (avoids a double-book).
                await self._snipe_notify(session, f"⏰ Couldn't book {biz}, sir — {result.message}")
                return self._snipe_failed(result.message)
            if time.time() + self._snipe_retry_gap_s > ss["deadline_ts"]:
                break
            await asyncio.sleep(self._snipe_retry_gap_s)

        miss = (f"I tried the moment {biz} opened, sir, but couldn't grab a "
                f"{display_time(plan.details.get('time'))} table — it likely sold out. "
                f"Best to try another date.")
        await self._snipe_notify(session, f"⏰ {miss}")
        return self._snipe_failed(miss)

    @staticmethod
    def _snipe_failed(message: str) -> TurnResult:
        return TurnResult.complete(
            message, slots_update={"snipe_state": None,
                                   "booking_result": {"success": False, "needs_manual": True}})

    async def _snipe_notify(self, session: Session, message: str) -> None:
        """Best-effort Telegram note on a snipe outcome (success already notifies
        via _on_confirmed). The unattended user learns the result either way."""
        if self.notifier is None:
            return
        try:
            await asyncio.to_thread(self.notifier.send, guard(USER_DEST_SINK, message))
        except Exception:
            logger.warning("Snipe Telegram notification failed", exc_info=True)

    # ------------------------------------------------------------ gate helpers
    def _action(self, kind: ActionKind, session: Session, plan: CommitPlan,
                scope: Optional[str] = None, attempt: int = 0) -> Action:
        return Action(kind=kind, session_id=session.session_id, workflow=self.name,
                      plan=plan.to_dict(), scope=scope, attempt=attempt)

    def _mint_action(self, session: Session, plan: CommitPlan) -> Action:
        amount = getattr(self.payment, "limit_usd", HARD_CAP_USD) if self.payment else None
        return Action(kind=ActionKind.MINT_CARD, session_id=session.session_id,
                      workflow=self.name, amount_usd=amount,
                      plan={"purpose": "single_use_card",
                            "business_name": plan.details.get("business_name"),
                            "amount_usd": amount})

    @staticmethod
    def _call_ttl_s() -> float:
        """An approved call plan survives off-hours deferral up to the call deadline."""
        try:
            days = float(os.getenv("RESERVATION_CALL_DEADLINE_DAYS", "7"))
        except ValueError:
            days = 7.0
        return days * 86400

    @staticmethod
    def _refusal_message(refusal: Refusal) -> str:
        if refusal.code == "kill_switch":
            return ("I'm in research-only mode (kill switch on), sir, "
                    "so I've stopped short of acting on it.")
        if refusal.code == "duplicate":
            return "I've already carried that out, sir — I won't run it twice."
        if refusal.code == "over_cap":
            return ("That would need a card above my $10 limit, sir — "
                    "you'll need to handle this one.")
        if refusal.code == "egress_violation":
            return ("I stopped that, sir — it was about to send something "
                    "that mustn't leave this machine, so I haven't acted on it.")
        # no_approval / scope_mismatch: the plan changed under us.
        return ("The details no longer match what you approved, sir — "
                "let's run through it again before I act.")

    async def _maybe_offer_sandbox(self, session: Session, plan: CommitPlan) -> Optional[TurnResult]:
        """Offer a vetted third-party bot (sandboxed) as a fallback — with explicit, repo-named consent."""
        if self.sandbox is None or session.slots.get("sandbox_tried"):
            return None
        biz = plan.details.get("business_name")
        candidate = await asyncio.to_thread(self.sandbox.find_bot, biz, plan.channel)
        if candidate is None:
            return None
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
            next_state=self._fire(session, "offer_sandbox"),
        )

    async def _handle_sandbox_consent(self, text: str, session: Session) -> TurnResult:
        decision = parse_confirmation(text)
        if decision == ConfirmDecision.UNCLEAR:
            return self._reask(session, "Shall I run that community bot in the sandbox, "
                                        "sir — yes or no?")
        if decision != ConfirmDecision.YES:
            return TurnResult.complete(
                "Understood, sir — I won't run third-party code. You'll need to book this one.")

        c = session.slots.get("sandbox_candidate") or {}
        plan_d = session.slots.get("commit_plan")
        if not c or not plan_d:
            return TurnResult.complete("I've lost that bot's details, sir — let's start over.")
        candidate = BotCandidate(**c)
        plan = CommitPlan.from_dict(plan_d)

        if self.sandbox is None:
            return TurnResult.complete("The sandbox isn't available right now, sir.")

        # Consent is scoped to this exact repo; running anything else is refused.
        action = self._action(ActionKind.RUN_UNTRUSTED_CODE, session, plan,
                              scope=candidate.full_name)
        self.gate.record_approval(session, action)
        outcome = await self.gate.execute(
            action, session,
            lambda: asyncio.to_thread(self.sandbox.run_bot, candidate, plan.details),
            success_of=lambda r: bool(r and r.success))
        if not outcome.ok:
            return TurnResult.complete(self._refusal_message(outcome.refusal))
        result = outcome.result
        biz = plan.details.get("business_name")
        if result.success:
            note = await self._on_confirmed(session, plan, result.confirmation)
            return TurnResult.complete(f"It worked, sir — {biz} is booked via the community bot.{note}")
        return TurnResult.complete(
            f"The community bot couldn't complete it either, sir — best to book "
            f"{biz} yourself: {plan.details.get('url') or 'their site'}.")

    async def _on_confirmed(self, session: Session, plan: CommitPlan,
                            confirmation: Optional[str]) -> str:
        """Calendar event + Telegram record on a confirmed booking. Never fails the booking."""
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
                guard(USER_DEST_SINK, facts)  # user-owned destination; card scan only
                if await asyncio.to_thread(self.calendar.create_event, facts):
                    note = " I've added it to your calendar, sir."
            except Exception:
                logger.warning("Calendar event failed", exc_info=True)

        if self.notifier is not None:
            try:
                message = guard(USER_DEST_SINK, self._notify_text(facts))
                await asyncio.to_thread(self.notifier.send, message)
            except Exception:
                logger.warning("Telegram notification failed", exc_info=True)

        return note

    @staticmethod
    def _notify_text(facts: Dict[str, Any]) -> str:
        conf = f" Confirmation: {facts['confirmation']}." if facts.get("confirmation") else ""
        return (
            f"✅ Reservation confirmed: {facts.get('business_name')} for "
            f"{facts.get('party_size')} on {display_date(facts.get('date'))} "
            f"at {display_time(facts.get('time'))}.{conf}"
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
                next_state=self._fire(session, "call_started"),
            )

        outcome = await self._gated_place_call(session, channel, plan, attempt=0)
        if not outcome.ok:
            return TurnResult.complete(self._refusal_message(outcome.refusal))
        call_ref = outcome.result
        if call_ref is None:
            return TurnResult.complete(
                f"I wasn't able to place the call to {biz}, sir — you may need to ring them."
            )
        phone_state.update(placed=True, call_ref=call_ref)
        return TurnResult.background(
            f"I'm calling {biz} now, sir — I'll let you know how it goes.",
            wake_at=time.time() + POLL_SECONDS,
            slots_update={"phone_state": phone_state},
            next_state=self._fire(session, "call_started"),
        )

    async def _gated_place_call(self, session: Session, channel, plan: CommitPlan, attempt: int):
        """All dials go through the gate: the approved call plan must still match,
        the kill switch is re-checked at dial time, and each attempt fires once."""
        action = self._action(ActionKind.PLACE_CALL, session, plan, attempt=attempt)
        return await self.gate.execute(
            action, session, lambda: asyncio.to_thread(channel.place_call, plan),
            success_of=lambda r: r is not None)

    async def on_tick(self, session: Session) -> Optional[TurnResult]:
        if session.slots.get("snipe_state"):
            return await self._tick_snipe(session)
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
            biz = decision.business_name
            return TurnResult.confirm(
                f"Good news, sir — a table opened at {biz} for {session.slots.get('time')}! "
                + self._confirm_message(plan, availability),
                slots_update={"commit_plan": plan.to_dict(), "watch_state": None},
                next_state=self._fire(session, "slot_opened"),
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
            outcome = await self._gated_place_call(session, channel, plan,
                                                   attempt=ps.get("retries", 0))
            if not outcome.ok:
                return TurnResult.complete(self._refusal_message(outcome.refusal))
            call_ref = outcome.result
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
        # Promote this backgrounded session to an active confirmation dialogue.
        return TurnResult.confirm(
            f"{biz} asked to be emailed, sir. " + self._email_confirm_message(plan),
            slots_update={"active_channel": "email", "business_email": email_addr,
                          "commit_plan": plan.to_dict(), "phone_state": None},
            next_state=self._fire(session, "gate_email"),
        )

    async def _handle_email_confirmation(self, text: str, session: Session) -> TurnResult:
        channel, plan = self._resolve_pending(session)
        if channel is None or plan is None:
            return TurnResult.complete("I've lost that draft, sir — let's start it again.")

        decision = parse_confirmation(text, editable=True)
        if decision == ConfirmDecision.YES:
            action = self._action(ActionKind.SEND_EMAIL, session, plan)
            self.gate.record_approval(session, action)

            def _guarded_send():
                guard(SMTP_SINK, {"to": plan.details.get("to"),
                                  "subject": plan.details.get("subject"),
                                  "body": plan.details.get("body")})
                return channel.send(plan)

            outcome = await self.gate.execute(
                action, session, lambda: asyncio.to_thread(_guarded_send),
                success_of=bool)
            if not outcome.ok:
                return TurnResult.complete(self._refusal_message(outcome.refusal))
            if not outcome.result:
                return TurnResult.complete("I couldn't send that email, sir.")
            biz = plan.details.get("business_name")
            email_state = {"to": plan.details["to"], "subject": plan.details["subject"],
                           "sent_at": time.time(), "nudged": False}
            return TurnResult.background(
                f"I've emailed {biz}, sir — I'll let you know when they reply.",
                wake_at=time.time() + EMAIL_POLL_SECONDS,
                slots_update={"email_state": email_state},
                next_state=self._fire(session, "sent"))

        if decision == ConfirmDecision.NO:
            return TurnResult.cancel("Understood, sir — I won't send it.")

        # Anything else is treated as an edit instruction → re-draft and re-confirm.
        # The edited plan replaces the stored one, so a later "yes" approves (and
        # hashes) exactly this revision.
        decision = self._decision_from_slots(session)
        subject, body = await asyncio.to_thread(
            channel.drafter, session.slots, decision, instruction=text)
        plan.details["subject"], plan.details["body"] = subject, body
        return TurnResult.confirm(
            "I've revised it, sir. " + self._email_confirm_message(plan),
            slots_update={"commit_plan": plan.to_dict()},
            next_state=self._fire(session, "redraft"))

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
            address=d.get("address"), hours=d.get("hours"),
        )

    @staticmethod
    def _email_confirm_message(plan: CommitPlan) -> str:
        d = plan.details
        return (f"I've drafted this to {d.get('to')} — subject \"{d.get('subject')}\". "
                f"Shall I send it, sir? (Or tell me what to change.)")

    def _resolve_pending(self, session: Session):
        """The (channel, plan) under approval, rebuilt purely from persisted slots —
        the plan the user approved survives restarts and is the one that commits."""
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
    def _reask(session: Session, question: str) -> TurnResult:
        """An unclear reply at a strict gate re-asks once; a second one in the
        session is read as a no. UNCLEAR can never approve."""
        count = int(session.slots.get("_unclear_count") or 0) + 1
        if count > 1:
            return TurnResult.cancel(
                "I'll take that as a no, sir — nothing has been done.")
        return TurnResult.confirm(question, slots_update={"_unclear_count": count})

    # --------------------------------------------------------------- messaging
    @staticmethod
    def _plan_message(slots: Dict[str, Any], decision: ChannelDecision,
                      bookable: bool = False) -> str:
        """Used when we can't gate a booking (research-only, or phone/email/unknown method)."""
        name = decision.business_name or slots.get("business_name", "the business")
        addr = f" ({decision.address})" if decision.address else ""
        party = slots.get("party_size", "your party")
        date = display_date(slots.get("date", "the requested date"))
        time = display_time(slots.get("time", "the requested time"))

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
        # The user approves the *resolved* facts ("Friday, June 19 at 7:00 pm"),
        # not the ambiguous phrase they originally used.
        date = display_date(plan.details.get("date"))
        time = display_time(plan.details.get("time"))
        verify = f" ({availability.note})" if availability.note else ""
        card = ""
        if plan.requires_card:
            card = " They may need a card to hold it; I'd use a single-use card capped at $10."
        return (
            f"I'm ready to book {name} for {party} on {date} at {time} via {plan.channel}.{verify}"
            f"{card} Shall I go ahead and book it, sir?"
        )

    @staticmethod
    def _snipe_confirm_message(plan: CommitPlan, snipe: Dict[str, Any]) -> str:
        d = plan.details
        date = display_date(d.get("date"))
        time = display_time(d.get("time"))
        card = ""
        if plan.requires_card:
            card = " They may want a card to hold it; I'd use a single-use card capped at $10."
        if snipe.get("source") == "you":
            lead = (f"{d.get('business_name')} isn't open for booking yet for "
                    f"{d.get('party_size')} on {date} at {time}, sir — reservations open "
                    f"{snipe['display']}.")
        else:
            # Auto-detected: hedge honestly and cite the source so the user can judge.
            cite = f' (a source notes: "{snipe["quote"][:160]}")' if snipe.get("quote") else ""
            lead = (f"I looked into {d.get('business_name')}, sir — it appears reservations for "
                    f"{date} at {time} open {snipe['display']}{cite}.")
        return (f"{lead} I can stand ready and try to grab it the instant they do "
                f"(best effort — these go fast).{card} Shall I set that up, sir?")

    # --------------------------------------------------------------- extraction
    def _extract_slots(self, text: str) -> Dict[str, Any]:
        """Parse the opening request: regex baseline, refined by the LLM when
        available (handles e.g. "half past seven", implicit party sizes), then
        **canonicalized** — dates resolve to ISO against now, times to 24h.
        Whatever fails to normalize is dropped and asked for."""
        slots = self._extract_slots_regex(text)
        refined = llm_extract_slots(self.llm, text)
        if refined:
            slots.update(refined)
        return self._normalize_slots(slots)

    @staticmethod
    def _normalize_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
        ctx = NormalizeCtx.fresh()
        for name, spec in SLOT_SPECS_BY_NAME.items():
            raw = slots.get(name)
            if raw is None:
                continue
            value = spec.normalize(str(raw), ctx)
            if value is None:
                slots.pop(name)          # unusable → the slot loop will ask
            else:
                if str(value) != str(raw):
                    slots[f"{name}_raw"] = raw   # what was actually said
                slots[name] = value
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

        self._extract_release_policy_regex(text, slots)
        return slots

    @staticmethod
    def _extract_release_policy_regex(text: str, slots: Dict[str, Any]) -> None:
        """Pull a stated release policy ("tables drop 30 days in advance at 10am
        ET") so sniping works even without the LLM. The release time is taken
        from after the open/drop keyword so it isn't confused with the dining
        time stated earlier in the sentence."""
        m = _RELEASE_DAYS_RE.search(text)
        if not m:
            return
        slots["release_days_ahead"] = int(m.group(1))
        kw = _RELEASE_CTX_RE.search(text)
        region = text[kw.start():] if kw else text
        tm = _TIME_RE.search(region)
        if tm:
            slots["release_time"] = tm.group(0).strip()
        tz = _RELEASE_TZ_RE.search(text)
        if tz:
            slots["release_timezone"] = tz.group(1)

